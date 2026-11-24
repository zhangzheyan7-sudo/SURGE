from __future__ import annotations

import csv
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from .metrics import MetricAccumulator, format_metrics
from .utils import (
    atomic_json_dump, autocast_context, compact_history_row,
    compact_training_history, ensure_dir, to_device,
)


CHECKPOINT_FORMAT_VERSION = 6


def output_signature(config: dict[str, Any]) -> tuple[str, int]:
    sgu = config.get("model", {}).get("fine", {}).get("sgu_flow", {})
    mode = str(sgu.get("output_mode", "dense"))
    revision = int(sgu.get("output_revision", 1 if mode == "affine_global_residual" else 0))
    return mode, revision


def atomic_copy_checkpoint(source: str | Path, destination: str | Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        # Windows FlushFileBuffers requires a writable handle.
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def build_optimizer(model: torch.nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    coarse_parameters = []
    fine_parameters = []
    recovery_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(("coarse.", "s2gf.")):
            coarse_parameters.append(parameter)
        elif name.startswith("coarse_recovery."):
            recovery_parameters.append(parameter)
        else:
            fine_parameters.append(parameter)
    training = config["training"]
    if str(training.get("scheduler", "warmup_cosine_floor")) != "warmup_cosine_floor":
        raise ValueError("Only scheduler=warmup_cosine_floor is supported")
    groups = [
        {"params": coarse_parameters, "lr": float(training["learning_rate_coarse"]), "name": "coarse"},
        {"params": fine_parameters, "lr": float(training["learning_rate_fine"]), "name": "fine"},
    ]
    if recovery_parameters:
        groups.append({
            "params": recovery_parameters,
            "lr": float(training.get("learning_rate_recovery", 1.0e-4)),
            "name": "recovery",
        })
    return torch.optim.AdamW(groups, weight_decay=float(training.get("weight_decay", 1.0e-4)))


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    *,
    epochs_override: int | None = None,
    warmup_override: int | None = None,
    min_lr_ratio_override: float | None = None,
    min_lr_ratios_override: list[float] | tuple[float, ...] | None = None,
    recovery_start_epoch: int | None = None,
    cosine_cycle_epochs_override: int | None = None,
    constant_lr_override: bool | None = None,
):
    epochs = int(epochs_override if epochs_override is not None else config["training"]["epochs"])
    warmup = int(warmup_override if warmup_override is not None else config["training"].get("warmup_epochs", 0))
    training = config["training"]
    constant_lr = bool(constant_lr_override if constant_lr_override is not None else training.get("constant_lr", False))
    cosine_cycle_epochs = int(
        cosine_cycle_epochs_override if cosine_cycle_epochs_override is not None
        else training.get("cosine_cycle_epochs", 0)
    )
    if cosine_cycle_epochs != 0 and cosine_cycle_epochs < 2:
        raise ValueError("cosine_cycle_epochs must be 0 or at least 2")
    normal_groups = [group for group in optimizer.param_groups if group.get("name") != "recovery"]
    if min_lr_ratios_override is not None:
        min_lr_ratios = [float(value) for value in min_lr_ratios_override[: len(normal_groups)]]
    elif min_lr_ratio_override is not None:
        min_lr_ratios = [float(min_lr_ratio_override)] * len(normal_groups)
    else:
        fallback_ratio = float(training.get("min_lr_ratio", 0.1))
        configured_minimums = {
            "coarse": float(training.get("min_learning_rate_coarse", training["learning_rate_coarse"] * fallback_ratio)),
            "fine": float(training.get("min_learning_rate_fine", training["learning_rate_fine"] * fallback_ratio)),
        }
        min_lr_ratios = [
            configured_minimums[str(group.get("name"))]
            / max(float(group.get("initial_lr", group["lr"])), 1.0e-16)
            for group in normal_groups
        ]
    if len(min_lr_ratios) != len(normal_groups):
        raise ValueError("Scheduler minimum-LR ratios must match coarse/fine parameter groups")
    if constant_lr:
        min_lr_ratios = [1.0] * len(normal_groups)
    elif any(not 0.0 < ratio < 1.0 for ratio in min_lr_ratios):
        raise ValueError("Each minimum learning rate must satisfy 0 < min_lr < initial_lr")
    warmup_start_ratio = float(training.get("warmup_start_ratio", 0.1))
    if not 0.0 < warmup_start_ratio <= 1.0:
        raise ValueError("warmup_start_ratio must be in (0,1]")

    def make_schedule(min_lr_ratio: float):
        def schedule(epoch: int) -> float:
            if constant_lr:
                return 1.0
            if warmup > 0 and epoch < warmup:
                progress = epoch / max(warmup - 1, 1)
                return warmup_start_ratio + (1.0 - warmup_start_ratio) * progress
            if cosine_cycle_epochs:
                phase = max(epoch - warmup, 0) % cosine_cycle_epochs
                cosine = 0.5 * (1.0 + math.cos(2.0 * math.pi * phase / cosine_cycle_epochs))
            else:
                progress = (epoch - warmup) / max(epochs - warmup - 1, 1)
                cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        return schedule

    ratio_by_name = {
        str(group.get("name")): ratio for group, ratio in zip(normal_groups, min_lr_ratios)
    }
    recovery_start = int(recovery_start_epoch or 0)
    recovery_warmup = int(training.get("recovery_warmup_epochs", 2))
    recovery_start_ratio = float(training.get("recovery_warmup_start_ratio", 0.1))

    def recovery_schedule(epoch: int) -> float:
        local_epoch = epoch - recovery_start
        if local_epoch <= 0:
            return recovery_start_ratio
        if recovery_warmup > 1 and local_epoch < recovery_warmup:
            progress = local_epoch / float(recovery_warmup - 1)
            return recovery_start_ratio + (1.0 - recovery_start_ratio) * progress
        return 1.0

    schedules = [
        recovery_schedule if group.get("name") == "recovery" else make_schedule(ratio_by_name[str(group.get("name"))])
        for group in optimizer.param_groups
    ]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedules)
    scheduler.dinotoporeg_spec = {
        "epochs": epochs,
        "warmup": warmup,
        "min_lr_ratios": min_lr_ratios,
        "warmup_start_ratio": warmup_start_ratio,
        "recovery_start_epoch": recovery_start,
        "recovery_warmup_epochs": recovery_warmup,
        "cosine_cycle_epochs": cosine_cycle_epochs,
        "constant_lr": constant_lr,
    }
    return scheduler


def create_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _forward(model, batch: dict[str, Any]) -> dict[str, Any]:
    return model(
        batch["optical"],
        batch["sar"],
        batch["dino_optical"],
        batch["dino_sar"],
        batch.get("optical_content_mask"),
        batch.get("sar_content_mask"),
    )


def run_epoch(
    model: torch.nn.Module,
    loader,
    criterion,
    device: torch.device,
    amp: bool,
    optimizer: torch.optim.Optimizer | None = None,
    scaler=None,
    accumulation_steps: int = 1,
    gradient_clip: float = 0.0,
    print_every: int = 20,
    description: str = "train",
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    accumulator = MetricAccumulator()
    geometry_accumulator = MetricAccumulator()
    raw_accumulator = MetricAccumulator()
    geometry_seen = False
    raw_seen = False
    residual_sum = 0.0
    residual_pixels = 0.0
    residual_max = 0.0
    fit_sums: dict[str, float] = {}
    fit_counts: dict[str, int] = {}

    def add_fit_diagnostics(name: str, fit: dict[str, Any]) -> None:
        fit_valid = fit.get("geometry_fit_valid", fit.get("global_fit_valid"))
        if torch.is_tensor(fit_valid):
            key = f"{name}_fit_fallback_ratio"
            fit_sums[key] = fit_sums.get(key, 0.0) + (~fit_valid.detach().bool()).float().sum().item()
            fit_counts[key] = fit_counts.get(key, 0) + fit_valid.numel()
        for suffix in ("effective_points", "spatial_coverage"):
            value = fit.get(f"geometry_fit_{suffix}", fit.get(f"global_fit_{suffix}"))
            if torch.is_tensor(value):
                key = f"{name}_fit_{suffix}"
                fit_sums[key] = fit_sums.get(key, 0.0) + value.detach().float().sum().item()
                fit_counts[key] = fit_counts.get(key, 0) + value.numel()
    if training:
        optimizer.zero_grad(set_to_none=True)
    progress = tqdm(loader, desc=description, dynamic_ncols=True)
    for step, batch in enumerate(progress):
        batch = to_device(batch, device)
        with torch.set_grad_enabled(training):
            with autocast_context(device, amp):
                outputs = _forward(model, batch)
                loss, components = criterion(
                    outputs, batch["flow"], batch["valid"], batch["optical"], batch["affine"]
                )
                scaled_loss = loss / max(accumulation_steps, 1)
            if training:
                scaler.scale(scaled_loss).backward()
                should_step = (step + 1) % accumulation_steps == 0 or (step + 1) == len(loader)
                if should_step:
                    scaler.unscale_(optimizer)
                    if gradient_clip > 0:
                        clip_grad_norm_(model.parameters(), gradient_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
        batch_metrics = accumulator.update(outputs["flow"], batch["flow"], batch["valid"], components)
        if torch.is_tensor(outputs.get("geometry_projected_flow")):
            geometry_seen = True
            geometry_accumulator.update(
                outputs["geometry_projected_flow"].float(), batch["flow"].float(), batch["valid"],
            )
        if torch.is_tensor(outputs.get("raw_dense_flow")):
            raw_seen = True
            raw_accumulator.update(outputs["raw_dense_flow"].float(), batch["flow"].float(), batch["valid"])
        if torch.is_tensor(outputs.get("residual_flow")):
            valid_residual = batch["valid"].squeeze(1).bool()
            magnitude = torch.linalg.vector_norm(outputs["residual_flow"].detach().float(), dim=1)[valid_residual]
            if magnitude.numel():
                residual_sum += magnitude.sum().item()
                residual_pixels += magnitude.numel()
                residual_max = max(residual_max, magnitude.amax().item())
        add_fit_diagnostics("final", outputs)
        coarse_fit = {
            "geometry_fit_effective_points": outputs.get("affine_effective_points", outputs.get("effective_point_count")),
            "geometry_fit_spatial_coverage": outputs.get("affine_spatial_coverage"),
        }
        add_fit_diagnostics("coarse", coarse_fit)
        coarse_fallback = outputs.get("affine_fallback_ratio")
        if coarse_fallback is not None:
            values = torch.as_tensor(coarse_fallback).detach().float()
            count = outputs["flow"].shape[0] if values.numel() == 1 else values.numel()
            fit_sums["coarse_fit_fallback_ratio"] = fit_sums.get("coarse_fit_fallback_ratio", 0.0) + values.mean().item() * count
            fit_counts["coarse_fit_fallback_ratio"] = fit_counts.get("coarse_fit_fallback_ratio", 0) + count
        for stage_name, stage in zip(("s8", "s4", "s2"), outputs.get("stage_results", [])):
            add_fit_diagnostics(stage_name, stage)
        if step % max(print_every, 1) == 0 or step + 1 == len(loader):
            progress.set_postfix(
                loss=f"{loss.detach().item():.4f}",
                EPE=f"{batch_metrics['epe']:.3f}",
                CMR1=f"{100.0 * batch_metrics['cmr_1px']:.1f}%",
                CMR2=f"{100.0 * batch_metrics['cmr_2px']:.1f}%",
                CMR3=f"{100.0 * batch_metrics['cmr_3px']:.1f}%",
                CMR5=f"{100.0 * batch_metrics['cmr_5px']:.1f}%",
            )
    metrics = accumulator.compute()
    if geometry_seen:
        metrics["geometry_epe"] = geometry_accumulator.compute()["epe"]
    if raw_seen:
        metrics["raw_dense_epe"] = raw_accumulator.compute()["epe"]
    if residual_pixels > 0:
        metrics["final_residual_mean_px"] = residual_sum / residual_pixels
        metrics["final_residual_max_px"] = residual_max
    metrics.update({name: fit_sums[name] / count for name, count in fit_counts.items() if count})
    return metrics


def save_checkpoint(
    path: str | Path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_epe: float,
    config: dict[str, Any],
    history: list[dict[str, float]],
    *,
    epochs_without_improvement: int | None = None,
    meaningful_best_epe: float | None = None,
    validation_epe: float | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "epoch": epoch,
        "best_epe": best_epe,
        "model": model.trainable_state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scheduler_spec": getattr(scheduler, "dinotoporeg_spec", None),
        "scaler": scaler.state_dict(),
        "config": config,
        "history": compact_training_history(history) if config.get("training", {}).get("history_mode") == "compact" else history,
        "history_schema_version": 2 if config.get("training", {}).get("history_mode") == "compact" else 1,
    }
    if epochs_without_improvement is not None and meaningful_best_epe is not None:
        payload["epochs_without_improvement"] = int(epochs_without_improvement)
        payload["meaningful_best_epe"] = float(meaningful_best_epe)
        payload["saved_output_mode"], payload["saved_output_revision"] = output_signature(config)
    if validation_epe is not None:
        payload["validation_epe"] = float(validation_epe)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, temporary)
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(
    path: str | Path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    map_location: str | torch.device = "cpu",
    resume_mode: str = "exact",
    checkpoint_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    if checkpoint_data is not None:
        checkpoint = checkpoint_data
    else:
        try:
            checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=map_location)
    version = int(checkpoint.get("format_version", 0)) if isinstance(checkpoint, dict) else 0
    if resume_mode not in ("exact", "migrate"):
        raise ValueError("resume_mode must be 'exact' or 'migrate'")
    report: dict[str, Any] = {
        "checkpoint_version": version,
        "resume_mode": resume_mode,
        "optimizer_status": "not requested",
        "scheduler_status": "pending",
        "scaler_status": "not requested",
        "ema_status": "not present (EMA is not used by this project)",
    }

    if resume_mode == "exact":
        if version != CHECKPOINT_FORMAT_VERSION:
            raise RuntimeError(
                f"Exact resume requires checkpoint format v{CHECKPOINT_FORMAT_VERSION}; "
                f"got v{version}. Use --resume-mode migrate for the pre-coarsefix checkpoint."
            )
        model.load_trainable_state_dict(checkpoint["model"])
        loaded_numel = sum(value.numel() for value in checkpoint["model"].values())
        total_numel = loaded_numel
        if optimizer is not None and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            report["optimizer_status"] = "exact"
    else:
        if version not in (4, 5, CHECKPOINT_FORMAT_VERSION):
            raise RuntimeError(f"Migration supports checkpoint formats 4, 5 and {CHECKPOINT_FORMAT_VERSION}; got {version}")
        current = model.trainable_state_dict()
        old_state = checkpoint["model"]
        loaded = {
            name: value
            for name, value in old_state.items()
            if name in current and current[name].shape == value.shape
        }
        merged = dict(current)
        merged.update(loaded)
        model.load_trainable_state_dict(merged)
        loaded_numel = sum(value.numel() for value in loaded.values())
        total_numel = sum(value.numel() for value in old_state.values())
        missing_old = sorted(set(old_state) - set(loaded))
        if missing_old:
            raise RuntimeError(
                "Migration does not rename or silently discard old tensors; unmatched old keys: "
                + ", ".join(missing_old[:10])
            )
        if optimizer is not None and "optimizer" in checkpoint:
            old_optimizer = checkpoint["optimizer"]
            old_groups = {str(group.get("name")): group for group in old_optimizer["param_groups"]}
            restored_states = 0
            expected_states = 0
            for group in optimizer.param_groups:
                name = str(group.get("name"))
                if name == "recovery" or name not in old_groups:
                    continue
                old_group = old_groups[name]
                old_ids = list(old_group["params"])
                parameters = list(group["params"])
                expected_states += len(old_ids)
                if len(parameters) != len(old_ids):
                    raise RuntimeError(
                        f"Cannot migrate optimizer group {name}: old={len(old_ids)}, new={len(parameters)}"
                    )
                for parameter, old_id in zip(parameters, old_ids):
                    state = old_optimizer.get("state", {}).get(old_id)
                    if state is None:
                        continue
                    compatible = True
                    migrated_state: dict[str, Any] = {}
                    for key, value in state.items():
                        if torch.is_tensor(value):
                            if value.ndim > 0 and value.shape != parameter.shape:
                                compatible = False
                                break
                            migrated_state[key] = value.detach().to(parameter.device).clone()
                        else:
                            migrated_state[key] = value
                    if compatible:
                        optimizer.state[parameter] = migrated_state
                        restored_states += 1
                for key, value in old_group.items():
                    if key not in ("params", "name"):
                        group[key] = value
            report["optimizer_status"] = (
                f"migrated {restored_states}/{expected_states} old parameter states; "
                "new recovery parameters use fresh AdamW state"
            )

    loaded_ratio = loaded_numel / max(total_numel, 1)
    if loaded_ratio <= 0.90:
        raise RuntimeError(f"Old-parameter loading ratio must exceed 90%; got {100.0 * loaded_ratio:.2f}%")
    report["loaded_parameter_ratio"] = loaded_ratio
    if scheduler is not None and "scheduler" in checkpoint and resume_mode == "exact":
        scheduler.load_state_dict(checkpoint["scheduler"])
        report["scheduler_status"] = "exact"
    if scaler is not None and "scaler" in checkpoint:
        try:
            scaler.load_state_dict(checkpoint["scaler"])
            report["scaler_status"] = "restored"
        except (KeyError, RuntimeError, ValueError) as error:
            report["scaler_status"] = f"fresh ({error})"
    checkpoint["_resume_report"] = report
    return checkpoint


def save_history(history: list[dict[str, Any]], output_dir: str | Path, history_mode: str = "full") -> None:
    if history_mode == "compact":
        history = compact_training_history(history)
    output_dir = ensure_dir(output_dir)
    columns: list[str] = []
    for row in history:
        for key in row:
            if key not in columns:
                columns.append(key)
    normalized = [
        {key: row.get(key, float("nan")) for key in columns} for row in history
    ]
    # Keep the historical prefix intact when a continuation adds diagnostics.
    # Only the rectangular CSV export needs missing-column placeholders.
    atomic_json_dump(history, output_dir / "history.json")
    if history:
        with (output_dir / "history.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(normalized)
    plot_history(history, output_dir / "training_curves.png")


def plot_history(history: list[dict[str, float]], path: str | Path) -> None:
    if not history:
        return
    path = Path(path)
    script = Path(__file__).resolve().parents[2] / "plot_training_history.py"
    command = [sys.executable, "-B", str(script), "--history", str(path.parent / "history.json"), "--output", str(path)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f"WARNING: training curve could not be generated: {error}", file=sys.stderr)
        return
    if result.returncode != 0:
        print(f"WARNING: training curve could not be generated: {result.stderr.strip()}", file=sys.stderr)


def _load_continuous_history(
    checkpoint: dict[str, Any], output_dir: Path
) -> list[dict[str, float]]:
    checkpoint_history = list(checkpoint.get("history", []))
    history_path = output_dir / "history.json"
    if history_path.exists():
        import json

        with history_path.open("r", encoding="utf-8") as handle:
            disk_history = list(json.load(handle))
        checkpoint_epoch = int(checkpoint.get("epoch", -1)) + 1
        if disk_history and int(disk_history[-1].get("epoch", -1)) == checkpoint_epoch:
            return disk_history
    return checkpoint_history


def _build_migrated_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    checkpoint: dict[str, Any],
    start_epoch: int,
):
    old_scheduler = checkpoint.get("scheduler") or {}
    old_spec = checkpoint.get("scheduler_spec") or {}
    old_optimizer = checkpoint.get("optimizer") or {"param_groups": []}
    old_groups = {str(group.get("name")): group for group in old_optimizer.get("param_groups", [])}
    old_base_lrs = list(old_scheduler.get("base_lrs", []))
    old_group_order = [str(group.get("name")) for group in old_optimizer.get("param_groups", [])]
    old_base_by_name = {
        name: old_base_lrs[index]
        for index, name in enumerate(old_group_order)
        if index < len(old_base_lrs)
    }
    current_lrs: dict[str, float] = {}
    for group in optimizer.param_groups:
        name = str(group.get("name"))
        if name == "recovery":
            base = float(config["training"].get("learning_rate_recovery", 1.0e-4))
            current = base * float(config["training"].get("recovery_warmup_start_ratio", 0.1))
        else:
            base = float(old_base_by_name.get(name, group["lr"]))
            current = float(old_groups.get(name, group).get("lr", group["lr"]))
        group["initial_lr"] = base
        group["lr"] = current
        current_lrs[name] = current

    scheduler = build_scheduler(
        optimizer,
        config,
        epochs_override=old_spec.get("epochs"),
        warmup_override=old_spec.get("warmup"),
        min_lr_ratio_override=old_spec.get("min_lr_ratio"),
        min_lr_ratios_override=old_spec.get("min_lr_ratios"),
        recovery_start_epoch=start_epoch,
    )
    last_epoch = int(old_scheduler.get("last_epoch", start_epoch))
    scheduler.last_epoch = last_epoch
    scheduler._step_count = int(old_scheduler.get("_step_count", last_epoch + 1))
    scheduler.base_lrs = [float(group["initial_lr"]) for group in optimizer.param_groups]
    scheduler._last_lr = [current_lrs[str(group.get("name"))] for group in optimizer.param_groups]
    for group, learning_rate in zip(optimizer.param_groups, scheduler._last_lr):
        group["lr"] = float(learning_rate)
    return scheduler


def _dry_run_batch(
    model,
    criterion,
    train_loader,
    optimizer,
    scaler,
    device: torch.device,
    amp: bool,
) -> None:
    model.train(True)
    optimizer.zero_grad(set_to_none=True)
    batch = to_device(next(iter(train_loader)), device)
    with autocast_context(device, amp):
        outputs = _forward(model, batch)
        loss, _ = criterion(
            outputs, batch["flow"], batch["valid"], batch["optical"], batch["affine"]
        )
    scaler.scale(loss).backward()
    finite_gradients = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    optimizer.zero_grad(set_to_none=True)
    print(
        f"one-batch forward/loss/backward: OK; loss={loss.detach().item():.6f}; "
        f"finite_gradients={finite_gradients}"
    )


def _history_output_signature(row: dict[str, float]) -> tuple[str, int]:
    mode = "affine_global_residual" if row.get("output_affine_global") == 1.0 else "dense"
    return mode, int(row.get("output_revision", 1 if mode == "affine_global_residual" else 0))


def _advance_early_stopping(
    epe: float, meaningful_best_epe: float, count: int, min_delta: float,
) -> tuple[float, int]:
    if math.isfinite(epe) and epe < meaningful_best_epe - min_delta:
        return epe, 0
    return meaningful_best_epe, count + 1


def _restore_early_stopping(
    checkpoint: dict[str, Any], history: list[dict[str, float]], output_dir: Path,
    signature: tuple[str, int], min_delta: float,
) -> tuple[float, int]:
    saved_signature = (
        checkpoint.get("saved_output_mode", output_signature(checkpoint.get("config", {}))[0]),
        checkpoint.get("saved_output_revision"),
    )
    if (saved_signature == signature and "epochs_without_improvement" in checkpoint
            and "meaningful_best_epe" in checkpoint):
        return float(checkpoint["meaningful_best_epe"]), int(checkpoint["epochs_without_improvement"])

    # Older format-6 checkpoints did not persist this state. Replay the exact
    # online rule, starting at the matching output's recorded baseline if any.
    meaningful_best_epe = float("inf")
    baseline_epoch = 0
    metadata_path = output_dir / "output_transition.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            target_mode = metadata.get("target_output_mode", "dense")
            target_revision = int(metadata.get("target_output_revision", 1 if target_mode == "affine_global_residual" else 0))
            if (target_mode, target_revision) == signature:
                baseline_epoch = int(metadata["completed_epoch"])
                meaningful_best_epe = float(metadata["baseline_epe"])
        except (OSError, ValueError, KeyError, TypeError) as error:
            print(f"WARNING: output baseline metadata could not be read: {error}", file=sys.stderr)
    count = 0
    matched = False
    for row in history:
        if _history_output_signature(row) != signature or int(row["epoch"]) <= baseline_epoch:
            continue
        matched = True
        meaningful_best_epe, count = _advance_early_stopping(
            float(row.get("val_epe", float("inf"))), meaningful_best_epe, count, min_delta,
        )
    if not matched and not math.isfinite(meaningful_best_epe):
        meaningful_best_epe = float(checkpoint.get("best_epe", float("inf")))
    print(f"Replayed early-stopping history for {signature[0]} revision {signature[1]}: count={count}")
    return meaningful_best_epe, count


def _ensure_best_checkpoint(
    checkpoint: dict[str, Any], source: Path, output_dir: Path, signature: tuple[str, int],
) -> None:
    best_path = output_dir / "best.pt"
    expected_epe = float(checkpoint.get("best_epe", float("inf")))
    problem = "missing"
    if best_path.exists():
        try:
            best = torch.load(best_path, map_location="cpu", weights_only=False)
            if not isinstance(best, dict) or int(best.get("format_version", 0)) != CHECKPOINT_FORMAT_VERSION:
                raise ValueError("unsupported checkpoint structure")
            if output_signature(best.get("config", {})) != signature:
                raise ValueError("output mode/revision differs from the resumed checkpoint")
            best_state = best.get("model", {})
            if set(best_state) != set(checkpoint.get("model", {})):
                raise ValueError("model parameter keys are incomplete")
            if any(not torch.is_tensor(value) or tuple(value.shape) != tuple(checkpoint["model"][key].shape)
                   for key, value in best_state.items()):
                raise ValueError("model parameter shapes are incompatible")
            if not math.isclose(float(best.get("best_epe", float("inf"))), expected_epe, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("best EPE differs from the resumed checkpoint")
            return
        except Exception as error:
            problem = f"invalid ({error})"
    selection_epe = checkpoint.get("validation_epe")
    if selection_epe is None:
        history = checkpoint.get("history", [])
        if history and _history_output_signature(history[-1]) == signature:
            selection_epe = history[-1].get("val_epe")
    if selection_epe is not None and math.isfinite(expected_epe) and math.isclose(
        float(selection_epe), expected_epe, rel_tol=1e-12, abs_tol=1e-12,
    ):
        atomic_copy_checkpoint(source, best_path)
        print(f"WARNING: best.pt was {problem}; rebuilt atomically from the resumed best weights.", file=sys.stderr)
    else:
        print(
            f"WARNING: best.pt is {problem}; resumed weights are not the recorded historical best. "
            "Historical best weights cannot be reconstructed from last.pt; restore a matching backup or epoch checkpoint.",
            file=sys.stderr,
        )


def _record_output_transition(metadata: dict[str, Any], output_dir: Path) -> None:
    latest_path = output_dir / "output_transition.json"
    all_path = output_dir / "output_transitions.json"
    transitions = json.loads(all_path.read_text(encoding="utf-8")) if all_path.exists() else []
    if not isinstance(transitions, list):
        raise RuntimeError("output_transitions.json must contain a list; existing transition records were preserved")

    def identity(item):
        source_mode = item.get("source_output_mode", "dense")
        target_mode = item.get("target_output_mode", "dense")
        return (
            source_mode, item.get("source_output_revision", 1 if source_mode == "affine_global_residual" else 0),
            target_mode, item.get("target_output_revision", 1 if target_mode == "affine_global_residual" else 0),
            item.get("completed_epoch"),
        )

    if latest_path.exists():
        previous = json.loads(latest_path.read_text(encoding="utf-8"))
        if not any(identity(item) == identity(previous) for item in transitions):
            transitions.append(previous)
    matches = [index for index, item in enumerate(transitions) if identity(item) == identity(metadata)]
    if matches:
        transitions[matches[-1]] = metadata
    else:
        transitions.append(metadata)
    atomic_json_dump(transitions, all_path)
    atomic_json_dump(metadata, latest_path)


def training_epoch_limit(training: dict[str, Any]) -> int:
    """Limit this execution without changing the schedule's total epoch budget."""
    total = int(training["epochs"])
    stop = int(training.get("stop_after_epoch", 0))
    if total <= 0 or stop < 0:
        raise ValueError("epochs must be positive and stop_after_epoch nonnegative")
    return min(total, stop) if stop else total


def train_model(model, criterion, train_loader, val_loader, config: dict[str, Any], device: torch.device) -> list[dict[str, float]]:
    training = config["training"]
    output_dir = ensure_dir(training["output_dir"])
    optimizer = build_optimizer(model, config)
    amp = bool(training.get("amp", True)) and device.type == "cuda"
    scaler = create_grad_scaler(amp)
    start_epoch = 0
    best_epe = float("inf")
    history: list[dict[str, float]] = []
    resume = str(training.get("resume", "")).strip()
    resume_mode = str(training.get("resume_mode", "exact"))
    dry_run_resume = bool(training.get("dry_run_resume", False))
    restart_scheduler = bool(training.get("restart_scheduler_on_resume", False))
    output_mode, output_revision = output_signature(config)
    output_transition = False
    scheduler = None
    if resume:
        if config.get("methodology") == "surge_paper_v1":
            resumed = torch.load(resume, map_location="cpu", weights_only=False)
            if resumed.get("config", {}).get("methodology") != "surge_paper_v1":
                raise ValueError("Old-method checkpoints are initialization only: use --warm-start, not --resume. A new optimizer, schedule and validation best are required.")
        checkpoint = load_checkpoint(
            resume,
            model,
            optimizer,
            scaler=scaler,
            map_location=device,
            resume_mode=resume_mode,
        )
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        report = checkpoint["_resume_report"]
        if not restart_scheduler and resume_mode == "exact":
            scheduler_spec = checkpoint.get("scheduler_spec") or {}
            scheduler = build_scheduler(
                optimizer,
                config,
                epochs_override=scheduler_spec.get("epochs"),
                warmup_override=scheduler_spec.get("warmup"),
                min_lr_ratio_override=scheduler_spec.get("min_lr_ratio"),
                min_lr_ratios_override=scheduler_spec.get("min_lr_ratios"),
                recovery_start_epoch=scheduler_spec.get("recovery_start_epoch"),
                cosine_cycle_epochs_override=scheduler_spec.get("cosine_cycle_epochs", 0),
                constant_lr_override=scheduler_spec.get("constant_lr", False),
            )
            if "scheduler" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler"])
                for group, learning_rate in zip(optimizer.param_groups, scheduler.get_last_lr()):
                    group["lr"] = float(learning_rate)
                report["scheduler_status"] = "exact"
        elif not restart_scheduler:
            scheduler = _build_migrated_scheduler(
                optimizer, config, checkpoint, start_epoch
            )
            report["scheduler_status"] = (
                f"migrated at last_epoch={scheduler.last_epoch}; coarse/fine continue checkpoint LR; "
                "recovery uses independent 2-epoch warmup"
            )
        best_epe = float(checkpoint.get("best_epe", best_epe))
        history = _load_continuous_history(checkpoint, output_dir)
        if training.get("history_mode") == "compact":
            history = compact_training_history(history)
        print(f"Resumed from {resume} at epoch {start_epoch + 1}")
        if restart_scheduler:
            total_epochs = int(training["epochs"])
            remaining_epochs = total_epochs - start_epoch
            if remaining_epochs <= 0:
                raise ValueError(
                    f"Checkpoint already completed {start_epoch} epochs, but configured total is {total_epochs}. "
                    "Set --epochs to a value greater than the checkpoint epoch."
                )
            restart_lrs = {
                "coarse": float(training["learning_rate_coarse"]),
                "fine": float(training["learning_rate_fine"]),
                "recovery": float(training.get("learning_rate_recovery", 1.0e-4)),
            }
            for group in optimizer.param_groups:
                learning_rate = restart_lrs[str(group.get("name"))]
                group["lr"] = learning_rate
                group["initial_lr"] = learning_rate
            scheduler = build_scheduler(
                optimizer,
                config,
                epochs_override=remaining_epochs,
                warmup_override=int(training.get("resume_warmup_epochs", 0)),
                recovery_start_epoch=0,
            )
            report["scheduler_status"] = "restarted by explicit --restart-scheduler"
            print(
                "Restarted LR schedule for "
                f"{remaining_epochs} remaining epochs: coarse={restart_lrs['coarse']:.3e}, fine={restart_lrs['fine']:.3e}, "
                f"floors=({float(training.get('min_learning_rate_coarse', 0.0)):.3e}, "
                f"{float(training.get('min_learning_rate_fine', 0.0)):.3e})"
            )
        if dry_run_resume:
            lr_by_name = {str(group.get("name")): float(group["lr"]) for group in optimizer.param_groups}
            history_last_epoch = int(history[-1]["epoch"]) if history else 0
            print(f"checkpoint epoch: {int(checkpoint.get('epoch', -1)) + 1}")
            print(f"history last epoch: {history_last_epoch}")
            print(f"next epoch: {start_epoch + 1}")
            print(f"old parameter loading ratio: {100.0 * report['loaded_parameter_ratio']:.4f}%")
            print(f"optimizer restore status: {report['optimizer_status']}")
            print(f"scheduler restore status: {report['scheduler_status']}")
            print(f"current coarse LR: {lr_by_name.get('coarse', float('nan')):.6e}")
            print(f"current fine LR: {lr_by_name.get('fine', float('nan')):.6e}")
            print(
                f"new module LR: {lr_by_name.get('recovery', float('nan')):.6e} "
                f"(target={float(training.get('learning_rate_recovery', 1.0e-4)):.6e})"
            )
            print(f"history length: {len(history)}")
            _dry_run_batch(model, criterion, train_loader, optimizer, scaler, device, amp)
            return history
        source_output_mode, source_output_revision = output_signature(checkpoint.get("config", {}))
        source_data = checkpoint.get("config", {}).get("data", {})
        target_data = config.get("data", {})
        source_crop = source_data.get("evaluation_crop_size", source_data.get("center_crop_size"))
        target_crop = target_data.get("evaluation_crop_size", target_data.get("center_crop_size"))
        validation_protocol_changed = (
            tuple(source_crop) if source_crop is not None else None
        ) != (tuple(target_crop) if target_crop is not None else None)
        if validation_protocol_changed:
            print(f"Validation crop transition: {source_crop} -> {target_crop}; re-evaluating restored weights to initialize best EPE.")
        source_spatial_only = bool(checkpoint.get('config',{}).get('model',{}).get('coarse',{}).get('spatial_only',False))
        target_spatial_only = bool(config.get('model',{}).get('coarse',{}).get('spatial_only',False))
        coarse_path_changed = source_spatial_only != target_spatial_only
        if coarse_path_changed:
            print('Coarse path changed to spatial-only: validating the restored weights before fine-tuning.')
        if coarse_path_changed or validation_protocol_changed or (output_mode == "affine_global_residual" and (
            source_output_mode, source_output_revision
        ) != (output_mode, output_revision)):
            # Changing the output changes the quantity selected by best.pt.
            # Evaluate the restored weights without advancing any training state.
            print(
                f"Output transition: {source_output_mode} revision {source_output_revision} -> "
                f"{output_mode} revision {output_revision}; "
                f"validating restored epoch {start_epoch} before continuing epoch {start_epoch + 1}."
            )
            with torch.inference_mode():
                baseline_metrics = run_epoch(
                    model, val_loader, criterion, device, amp,
                    print_every=int(training.get("print_every", 20)),
                    description=f"val output baseline {start_epoch}",
                )
            baseline_epe = float(baseline_metrics["epe"])
            if not math.isfinite(baseline_epe):
                raise RuntimeError("Output-transition validation produced non-finite EPE; continuation was not saved")
            transition_metadata = {
                "source_checkpoint": str(Path(resume).resolve()),
                "source_spatial_only": source_spatial_only,
                "target_spatial_only": target_spatial_only,
                "source_output_mode": source_output_mode,
                "target_output_mode": output_mode,
                "source_output_revision": source_output_revision,
                "target_output_revision": output_revision,
                "completed_epoch": start_epoch,
                "next_epoch": start_epoch + 1,
                "history_length": len(history),
                "previous_best_epe": best_epe,
                "source_validation_crop": source_crop,
                "target_validation_crop": target_crop,
                "baseline_epe": baseline_epe,
                "baseline_metrics": {key: float(value) for key, value in baseline_metrics.items()},
                "learning_rates": {str(group.get("name")): float(group["lr"]) for group in optimizer.param_groups},
                "optimizer_restore_status": report["optimizer_status"],
                "scheduler_restore_status": report["scheduler_status"],
                "scaler_restore_status": report["scaler_status"],
            }
            best_epe = baseline_epe
            last_path = output_dir / "last.pt"
            # Commit the transition journal first. If checkpoint replacement
            # fails, the old output remains resumable and the journal identity
            # makes the retried baseline replace its pending record.
            _record_output_transition(transition_metadata, output_dir)
            save_checkpoint(
                last_path, model, optimizer, scheduler, scaler, start_epoch - 1, best_epe, config, history,
                epochs_without_improvement=0, meaningful_best_epe=best_epe, validation_epe=baseline_epe,
            )
            atomic_copy_checkpoint(last_path, output_dir / "best.pt")
            output_transition = True
            print(format_metrics(baseline_metrics, "OUTPUT BASELINE"))
            print(
                f"New output baseline EPE: {best_epe:.4f}; best.pt and last.pt initialized; "
                "epoch, learning rate, optimizer, scaler and historical curves preserved; early stopping reset."
            )
        else:
            _ensure_best_checkpoint(checkpoint, Path(resume), output_dir, (output_mode, output_revision))
    else:
        if dry_run_resume:
            raise ValueError("--dry-run-resume requires --resume")
        scheduler = build_scheduler(optimizer, config)

    if scheduler is None:
        raise RuntimeError("Learning-rate scheduler was not initialized")

    early_stopping_patience = int(training.get("early_stopping_patience", 0))
    early_stopping_min_delta = float(training.get("early_stopping_min_delta", 0.0))
    epochs_without_improvement = 0
    meaningful_best_epe = best_epe
    if resume and not output_transition:
        meaningful_best_epe, epochs_without_improvement = _restore_early_stopping(
            checkpoint, history, output_dir, (output_mode, output_revision), early_stopping_min_delta,
        )
    if resume and training.get("reset_early_stopping_on_resume", False):
        previous_count = epochs_without_improvement
        epochs_without_improvement = 0
        print(f"Reset early-stopping counter: {previous_count} -> 0; historical best EPE and optimizer preserved.")
    if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
        print(f"Early stopping was already reached at epoch {start_epoch}; no additional epoch was run.")
        return history

    execution_end = training_epoch_limit(training)
    if execution_end < int(training["epochs"]):
        print(f"Screening pause: stop after epoch {execution_end}; LR schedule remains {training['epochs']} epochs.")
    for epoch in range(start_epoch, execution_end):
        epoch_start = time.perf_counter()
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            amp,
            optimizer=optimizer,
            scaler=scaler,
            accumulation_steps=int(training.get("accumulation_steps", 1)),
            gradient_clip=float(training.get("gradient_clip", 0.0)),
            print_every=int(training.get("print_every", 20)),
            description=f"train {epoch + 1}/{training['epochs']}",
        )
        with torch.inference_mode():
            val_metrics = run_epoch(
                model,
                val_loader,
                criterion,
                device,
                amp,
                print_every=int(training.get("print_every", 20)),
                description=f"val   {epoch + 1}/{training['epochs']}",
            )
        epoch_lrs = {str(group.get("name")): float(group["lr"]) for group in optimizer.param_groups}
        scheduler.step()
        row: dict[str, float] = {"epoch": epoch + 1, "seconds": time.perf_counter() - epoch_start}
        row["output_affine_global"] = float(output_mode == "affine_global_residual")
        row["output_revision"] = float(output_revision)
        row["stage"] = str(config.get("training_stage", "full"))
        row["epe_statistics"] = "valid_pixel_weighted"
        row.update({f"train_{key}": float(value) for key, value in train_metrics.items()})
        row.update({f"val_{key}": float(value) for key, value in val_metrics.items()})
        for group_name, lr in epoch_lrs.items():
            row[f"lr_{group_name}"] = lr
            row[f"lr_floor_{group_name}"] = float(training.get(f"min_learning_rate_{group_name}", 0.0))
        print(format_metrics(train_metrics, "TRAIN"))
        print(format_metrics(val_metrics, "VALID"))
        print(
            f"LR coarse={row['lr_coarse']:.3e} "
            f"(floor={float(training.get('min_learning_rate_coarse', 0.0)):.3e}) | "
            f"fine={row['lr_fine']:.3e} "
            f"(floor={float(training.get('min_learning_rate_fine', 0.0)):.3e})"
        )

        previous_best_epe = best_epe
        is_best = val_metrics["epe"] < previous_best_epe
        if is_best:
            best_epe = float(val_metrics["epe"])
            print(f"New best validation EPE: {best_epe:.4f}")
        meaningful_best_epe, epochs_without_improvement = _advance_early_stopping(
            float(val_metrics["epe"]), meaningful_best_epe, epochs_without_improvement, early_stopping_min_delta,
        )
        row.update(
            best_val_epe=best_epe, meaningful_best_epe=meaningful_best_epe,
            epochs_without_improvement=epochs_without_improvement,
        )
        history.append(compact_history_row(row) if training.get("history_mode") == "compact" else row)
        save_history(history, output_dir, str(training.get("history_mode", "full")))
        print(f"Epoch {epoch + 1} took {row['seconds'] / 60.0:.1f} min; history saved: {output_dir / 'history.csv'}")
        last_path = output_dir / "last.pt"
        save_checkpoint(
            last_path, model, optimizer, scheduler, scaler, epoch, best_epe, config, history,
            epochs_without_improvement=epochs_without_improvement,
            meaningful_best_epe=meaningful_best_epe, validation_epe=float(val_metrics["epe"]),
        )
        if is_best:
            atomic_copy_checkpoint(last_path, output_dir / "best.pt")
        if bool(training.get("save_epoch_checkpoints", True)) and (epoch + 1) % int(training.get("save_every", 1)) == 0:
            atomic_copy_checkpoint(last_path, output_dir / f"epoch_{epoch + 1:03d}.pt")
        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch + 1}: validation EPE did not improve by "
                f"at least {early_stopping_min_delta:.4f} for {epochs_without_improvement} epochs."
            )
            break
    return history
