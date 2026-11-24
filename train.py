from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dinotoporeg.config import load_config
from dinotoporeg.data_interface import add_data_arguments, apply_data_arguments
from dinotoporeg.data import build_train_val_loaders
from dinotoporeg.engine import train_model
from dinotoporeg.losses import RegistrationLoss
from dinotoporeg.model import DINOTopoReg
from dinotoporeg.utils import atomic_json_dump, ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DINOv3 topology-assisted optical-SAR registration")
    parser.add_argument(
        "--config",
        default="configs/whu_paper_aligned.yaml",
        help="Training configuration. Select configs/os_paper_aligned.yaml for OS.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", default="", help="Exact vNext training continuation checkpoint")
    parser.add_argument("--warm-start", default="", help="Import shared compatible weights from an old TSFR/AGMR checkpoint; start a FRESH optimizer/schedule")
    parser.add_argument(
        "--resume-mode",
        choices=("exact", "migrate"),
        default="exact",
        help="Use exact for format-compatible checkpoints or migrate for the pre-coarsefix checkpoint",
    )
    parser.add_argument(
        "--dry-run-resume",
        action="store_true",
        help="Audit resume state and run one forward/loss/backward batch without training",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override the total epoch count")
    parser.add_argument("--reset-early-stopping", action="store_true", help="Reset only the restored no-improvement counter; retain best EPE and optimizer")
    parser.add_argument("--stop-after-epoch", type=int, default=0,
                        help="Pause after this completed epoch without shortening the LR schedule; 0 disables")
    parser.add_argument(
        "--restart-scheduler",
        action="store_true",
        help="Keep resumed optimizer moments but restart the LR schedule for the remaining epochs",
    )
    parser.add_argument("--lr-coarse", type=float, default=None, help="Override coarse-stage learning rate")
    parser.add_argument("--lr-fine", type=float, default=None, help="Override fine-stage learning rate")
    parser.add_argument(
        "--cosine-cycle-epochs", type=int, default=None,
        help="Full smooth cosine cycle in epochs (peak -> floor -> peak); 0 disables cycling",
    )
    parser.add_argument(
        "--min-lr-ratio",
        type=float,
        default=None,
        help="Cosine-schedule floor as a fraction of each group's initial LR (0 <= ratio < 1)",
    )
    parser.add_argument("--min-lr-coarse", type=float, default=None, help="Absolute coarse-stage LR floor")
    parser.add_argument("--min-lr-fine", type=float, default=None, help="Absolute fine-stage LR floor")
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="Stop after this many epochs without meaningful validation-EPE improvement; 0 disables",
    )
    parser.add_argument(
        "--no-epoch-checkpoints",
        action="store_true",
        help="Save only best.pt and last.pt; do not create epoch_NNN.pt checkpoints",
    )
    parser.add_argument(
        "--stage",
        choices=("affine_only", "affine_residual", "affine_graph", "full", "baseline", "tfe", "graph"),
        default="full",
    )
    add_data_arguments(parser)
    return parser.parse_args()


def require_fresh_output_directory(config: dict) -> None:
    """A fresh run must not overwrite an existing run's weights or curves."""
    training = config["training"]
    if training.get("resume"):
        return
    output_dir = Path(training["output_dir"])
    existing = [name for name in ("last.pt", "best.pt", "history.json", "history.csv")
                if (output_dir / name).exists()]
    if existing:
        raise FileExistsError(
            f"Fresh training requires a new output directory: {output_dir} already contains {existing}. "
            "For a fresh run change training.output_dir; to continue this run explicitly pass --resume."
        )


def main() -> None:
    args = parse_args()
    config = apply_data_arguments(load_config(args.config), args)
    aliases = {"baseline": "affine_only", "tfe": "affine_residual", "graph": "affine_graph"}
    stage = aliases.get(args.stage, args.stage)
    if config["model"].get("fine", {}).get("mode") in ("tsfr", "sgu_flow") and args.stage != "full":
        raise ValueError("TSFR / SGU-Flow architectures require --stage full. Use the corresponding legacy configuration for ablations.")
    geometry = config["model"].setdefault("geometry", {})
    geometry["affine_enabled"] = True
    geometry["residual_enabled"] = (
        stage in ("affine_residual", "full")
        and config["model"].get("fine", {}).get("mode") not in ("tsfr", "sgu_flow")
    )
    config["model"]["coarse"]["graph_enabled"] = (
        bool(config["model"]["coarse"].get("graph_enabled", True))
        if config["model"].get("fine", {}).get("mode") == "sgu_flow"
        else stage in ("affine_graph", "full")
    )
    config["model"]["coarse"]["uot_enabled"] = bool(
        config["model"]["coarse"].get("uot_enabled", True)
    )
    config["model"]["fine"]["tfe_enabled"] = geometry["residual_enabled"]
    if not config["model"]["coarse"]["graph_enabled"]:
        config["loss"]["w_relation"] = 0.0
    affine_global_output = (
        config["model"].get("fine", {}).get("mode") == "sgu_flow"
        and config["model"]["fine"].get("sgu_flow", {}).get("output_mode") == "affine_global_residual"
    )
    if not geometry["residual_enabled"]:
        for key in ("w_residual", "w_residual_smooth", "w_residual_magnitude", "w_residual_consistency"):
            if affine_global_output and key == "w_residual_smooth":
                continue
            config["loss"][key] = 0.0
    if args.stage != "full":
        config["training"]["output_dir"] = str(Path(config["training"]["output_dir"]) / args.stage)
    config["training_stage"] = stage
    if args.resume:
        config["training"]["resume"] = str(Path(args.resume).resolve())
    if args.warm_start and (args.resume or config["training"].get("resume")):
        raise ValueError("--warm-start cannot be combined with --resume; the optimizer must start fresh")
    config["training"]["resume_mode"] = args.resume_mode
    config["training"]["dry_run_resume"] = bool(args.dry_run_resume)
    if args.epochs is not None:
        if args.epochs <= 0:
            raise ValueError("--epochs must be positive")
        config["training"]["epochs"] = args.epochs
    if args.stop_after_epoch < 0:
        raise ValueError("--stop-after-epoch must be nonnegative")
    config["training"]["stop_after_epoch"] = args.stop_after_epoch
    config["training"]["reset_early_stopping_on_resume"] = bool(args.reset_early_stopping)
    if args.reset_early_stopping and not config["training"].get("resume"):
        raise ValueError("--reset-early-stopping requires a resume checkpoint")
    if args.restart_scheduler:
        config["training"]["restart_scheduler_on_resume"] = True
    if args.cosine_cycle_epochs is not None:
        if args.cosine_cycle_epochs != 0 and args.cosine_cycle_epochs < 2:
            raise ValueError("--cosine-cycle-epochs must be 0 or at least 2")
        if config["training"].get("resume") and not args.restart_scheduler:
            raise ValueError("Changing the resumed cosine cycle requires --restart-scheduler")
        config["training"]["cosine_cycle_epochs"] = args.cosine_cycle_epochs
    if args.lr_coarse is not None:
        if args.lr_coarse <= 0:
            raise ValueError("--lr-coarse must be positive")
        config["training"]["learning_rate_coarse"] = args.lr_coarse
    if args.lr_fine is not None:
        if args.lr_fine <= 0:
            raise ValueError("--lr-fine must be positive")
        config["training"]["learning_rate_fine"] = args.lr_fine
    if args.min_lr_ratio is not None:
        if not 0.0 < args.min_lr_ratio < 1.0:
            raise ValueError("--min-lr-ratio must satisfy 0 < ratio < 1")
        print("DEPRECATION: --min-lr-ratio is retained for compatibility; prefer absolute --min-lr-coarse/--min-lr-fine.")
        config["training"]["min_lr_ratio"] = args.min_lr_ratio
        config["training"]["min_learning_rate_coarse"] = (
            float(config["training"]["learning_rate_coarse"]) * args.min_lr_ratio
        )
        config["training"]["min_learning_rate_fine"] = (
            float(config["training"]["learning_rate_fine"]) * args.min_lr_ratio
        )
    if args.min_lr_coarse is not None:
        if not 0.0 < args.min_lr_coarse < float(config["training"]["learning_rate_coarse"]):
            raise ValueError("--min-lr-coarse must satisfy 0 < floor < coarse initial LR")
        config["training"]["min_learning_rate_coarse"] = args.min_lr_coarse
    if args.min_lr_fine is not None:
        if not 0.0 < args.min_lr_fine < float(config["training"]["learning_rate_fine"]):
            raise ValueError("--min-lr-fine must satisfy 0 < floor < fine initial LR")
        config["training"]["min_learning_rate_fine"] = args.min_lr_fine
    if args.early_stopping_patience is not None:
        if args.early_stopping_patience < 0:
            raise ValueError("--early-stopping-patience cannot be negative")
        config["training"]["early_stopping_patience"] = args.early_stopping_patience
    if args.no_epoch_checkpoints:
        config["training"]["save_epoch_checkpoints"] = False
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "cpu":
        print("WARNING: DINOv3 ViT-L/16 training on CPU is impractically slow; use a CUDA GPU.")

    require_fresh_output_directory(config)
    output_dir = ensure_dir(config["training"]["output_dir"])
    if not args.dry_run_resume:
        atomic_json_dump(config, output_dir / "resolved_config.json")
    train_loader, val_loader, split_summary = build_train_val_loaders(config)
    if not args.dry_run_resume:
        atomic_json_dump(split_summary, output_dir / "split_summary.json")
    print("Dataset split:")
    print(json.dumps(split_summary, ensure_ascii=False, indent=2))

    model = DINOTopoReg(config).to(device)
    if args.warm_start:
        # Only copy exact name + shape matches. No stale optimizer state or
        # incompatible old fine-refiner tensors are reused.
        try:
            old = torch.load(args.warm_start, map_location="cpu", weights_only=False)
        except TypeError:
            old = torch.load(args.warm_start, map_location="cpu")
        source = old.get("model", old) if isinstance(old, dict) else old
        current = model.trainable_state_dict()
        compatible = {k: v for k,v in source.items() if k in current and tuple(v.shape)==tuple(current[k].shape)}
        if config.get("methodology") == "surge_paper_v1" and (set(compatible) != set(current) or set(compatible) != set(source)):
            raise RuntimeError("Paper initialization requires a complete name-and-shape match; partial warm-start is disabled")
        current.update(compatible)
        model.load_trainable_state_dict(current)
        print(f"Warm-start: copied {len(compatible)} tensors out of {len(source)} from {args.warm_start}. "
              "New SGU-Flow layers remain randomly initialized; optimizer and LR schedule reset.")
        if not compatible:
            raise RuntimeError("Warm-start found no compatible parameters; check checkpoint architecture")
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    frozen = sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad)
    print(f"Device: {device}; trainable parameters: {trainable:,}; frozen parameters: {frozen:,}")
    criterion = RegistrationLoss(config["loss"]).to(device)
    train_model(model, criterion, train_loader, val_loader, config, device)


if __name__ == "__main__":
    main()
