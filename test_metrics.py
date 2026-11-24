from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import tempfile
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dinotoporeg.config import load_config
from dinotoporeg.data import build_test_loader
from dinotoporeg.engine import load_checkpoint, output_signature
from dinotoporeg.geometry import transform_points
from dinotoporeg.metrics import MetricAccumulator, compute_registration_metrics, format_metrics
from dinotoporeg.model import DINOTopoReg
from dinotoporeg.utils import atomic_json_dump, autocast_context, ensure_dir, resize_flow, to_device


def affine_corner_error(predicted: torch.Tensor, target: torch.Tensor, size: tuple[int, int]) -> float:
    # Model outputs may be BF16/FP16 under AMP while dataset affine targets are
    # FP32.  Evaluate this metric in FP32 so both einsum operands share a dtype
    # and the reported pixel error is not unnecessarily quantized.
    predicted = predicted.float()
    target = target.to(device=predicted.device, dtype=torch.float32)
    height, width = size
    points = predicted.new_tensor((
        (0.0, 0.0), (width - 1.0, 0.0), (0.0, height - 1.0),
        (width - 1.0, height - 1.0), ((width - 1.0) / 2.0, (height - 1.0) / 2.0),
    ))
    difference = transform_points(points, predicted) - transform_points(points, target)
    return float(torch.linalg.vector_norm(difference, dim=-1).mean().item())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute test-set optical-SAR registration metrics")
    parser.add_argument("--config", default="configs/whu_paper_aligned.yaml")
    parser.add_argument("--checkpoint", default="", help="Defaults to config.evaluation.checkpoint")
    parser.add_argument("--datasets", default="all", help="all or comma-separated configured dataset names")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="", help="Parent directory for a uniquely named evaluation run")
    parser.add_argument("--max-samples", type=int, default=0, help="0 evaluates the complete test set")
    parser.add_argument(
        "--allow-output-override", action="store_true",
        help="Explicitly evaluate a checkpoint using a different output mode/revision or affine output settings",
    )
    return parser.parse_args()


def evaluation_output_metadata(config: dict, checkpoint: dict, allow_override: bool = False) -> dict:
    def definition(cfg):
        mode, revision = output_signature(cfg)
        sgu = cfg.get("model", {}).get("fine", {}).get("sgu_flow", {})
        output = sgu.get("affine_output", {})
        progressive = sgu.get("progressive_geometry", {})
        is_sgu = cfg.get("model", {}).get("fine", {}).get("mode") == "sgu_flow"
        progressive_definition = {"enabled": bool(progressive.get("enabled", False))}
        if progressive_definition["enabled"]:
            defaults = {
                "residual_max_px": 1.0, "residual_smoothing_kernel": 3,
                "fit_grid_step": 1, "fit_irls_iterations": 2,
                "fit_huber_delta_px": 3.0, "balance_grid": 4, "balance_max_gain": 4.0,
            }
            progressive_definition.update({key: type(value)(progressive.get(key, value))
                                           for key, value in defaults.items()})
        return {
            "global_dustbin_enabled": bool(cfg.get("model", {}).get("coarse", {}).get("dustbin_enabled", True)),
            "local_dustbin_enabled": bool(sgu.get("uot_flow", {}).get("dustbin_enabled", True)),
            "relation_support_mode": cfg.get("model", {}).get("coarse", {}).get("relation_support_mode", "legacy_same_offset"),
            "spatial_only": bool(cfg.get("model", {}).get("coarse", {}).get("spatial_only", False)),
            "output_mode": mode,
            "output_revision": revision,
            "residual_max_px": float(output.get("residual_max_px", 1.0)) if mode == "affine_global_residual" else None,
            "subpixel_sampling_mode": str(sgu.get("subpixel", {}).get("sampling_mode", "integer")) if is_sgu else None,
            "progressive_geometry": progressive_definition if is_sgu else None,
            "affine_output": {
                "residual_smoothing_kernel": int(output.get("residual_smoothing_kernel", 5)),
                "fit_irls_iterations": int(output.get("fit_irls_iterations", 2)),
                "fit_huber_delta_px": float(output.get("fit_huber_delta_px", 3.0)),
                "geometry_fit_stride": int(sgu.get("geometry_fit_stride", 2)),
                "balance_grid": int(output.get("balance_grid", 1)),
                "balance_max_gain": float(output.get("balance_max_gain", 4.0)),
            } if mode == "affine_global_residual" else None,
        }

    actual = definition(config)
    stored = definition(checkpoint.get("config", {}))
    changed = actual != stored
    if changed and not allow_override:
        raise ValueError(
            f"Evaluation output differs from checkpoint: checkpoint={stored}, requested={actual}. "
            "Use the matching configuration, or pass --allow-output-override for an explicit controlled comparison."
        )
    if changed:
        print("WARNING: explicitly evaluating a different output definition via --allow-output-override", file=sys.stderr)
    return {
        "actual_config_path": config.get("_config_path"),
        "checkpoint_config_path": checkpoint.get("config", {}).get("_config_path"),
        "output_mode": actual["output_mode"],
        "output_revision": actual["output_revision"],
        "subpixel_sampling_mode": actual["subpixel_sampling_mode"],
        "progressive_geometry": actual["progressive_geometry"],
        "residual_max_px": actual["residual_max_px"],
        "checkpoint_output_mode": stored["output_mode"],
        "checkpoint_output_revision": stored["output_revision"],
        "checkpoint_residual_max_px": stored["residual_max_px"],
        "actual_output_definition": actual,
        "checkpoint_output_definition": stored,
        "output_override": changed,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint or config["evaluation"]["checkpoint"]).resolve()
    checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    output_metadata = evaluation_output_metadata(config, checkpoint_data, args.allow_output_override)
    configured_names = [entry["name"] for entry in config["data"]["datasets"]]
    selected = configured_names if args.datasets.lower() == "all" else [item.strip() for item in
                                                                        args.datasets.split(",") if item.strip()]
    loader = build_test_loader(config, selected, batch_size=int(config["evaluation"].get("batch_size", 1)))
    output_root = ensure_dir(args.output_dir or config["evaluation"]["output_dir"])
    output_dir = Path(tempfile.mkdtemp(prefix="test_", dir=output_root))

    model = DINOTopoReg(config).to(device)
    checkpoint = load_checkpoint(checkpoint_path, model, map_location=device, checkpoint_data=checkpoint_data)
    model.eval()
    amp = bool(config["evaluation"].get("amp", True)) and device.type == "cuda"
    overall = MetricAccumulator()
    by_dataset: defaultdict[str, MetricAccumulator] = defaultdict(MetricAccumulator)
    stage_overall: defaultdict[str, MetricAccumulator] = defaultdict(MetricAccumulator)
    rows: list[dict[str, object]] = []
    total_seconds = 0.0
    total_samples = 0

    with torch.inference_mode():
        for batch in tqdm(loader, desc="test metrics", dynamic_ncols=True):
            if args.max_samples and total_samples >= args.max_samples:
                break
            batch = to_device(batch, device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            with autocast_context(device, amp):
                outputs = model(
                    batch["optical"], batch["sar"], batch["dino_optical"], batch["dino_sar"],
                    batch.get("optical_content_mask"), batch.get("sar_content_mask"),
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            batch_size = outputs["flow"].shape[0]
            full_size = batch["flow"].shape[-2:]
            stage_predictions = {
                "coarse_dense": resize_flow(outputs["coarse_flow"], full_size),
                "initial_global": resize_flow(outputs["initial_affine_flow"], full_size),
                "refined_global": resize_flow(outputs["affine_flow"], full_size),
            }
            stage_predictions.update(
                {f"refiner_{stage + 1}": resize_flow(flow, full_size) for stage, flow in enumerate(outputs["flows"])}
            )
            total_seconds += elapsed
            for index in range(batch_size):
                if args.max_samples and total_samples >= args.max_samples:
                    break
                prediction = outputs["flow"][index: index + 1]
                target = batch["flow"][index: index + 1]
                valid = batch["valid"][index: index + 1]
                dataset_name = batch["dataset"][index]
                sample_metrics = compute_registration_metrics(prediction, target, valid)
                overall.update(prediction, target, valid)
                by_dataset[dataset_name].update(prediction, target, valid)
                row = {
                    "dataset": dataset_name,
                    "sample_id": batch["sample_id"][index],
                    "source_id": batch["source_id"][index],
                    "epe": sample_metrics["epe"],
                    "mae": sample_metrics["mae"],
                    "rmse": sample_metrics["rmse"],
                    "cmr_1px": sample_metrics["cmr_1px"],
                    "cmr_2px": sample_metrics["cmr_2px"],
                    "cmr_3px": sample_metrics["cmr_3px"],
                    "cmr_5px": sample_metrics["cmr_5px"],
                    "valid_pixels": int(sample_metrics["valid_pixels"]),
                    "inference_seconds": elapsed / batch_size,
                    "residual_mean_magnitude": float(
                        torch.linalg.vector_norm(outputs["residual_flow"][index], dim=0).mean().item()
                    ),
                    "affine_corner_error": affine_corner_error(
                        outputs["affine_matrix_pixel"][index], batch["affine"][index], full_size
                    ),
                }
                for stage_name, stage_prediction in stage_predictions.items():
                    stage_sample = stage_prediction[index: index + 1]
                    stage_metrics = compute_registration_metrics(stage_sample, target, valid)
                    stage_overall[stage_name].update(stage_sample, target, valid)
                    row[f"{stage_name}_epe"] = stage_metrics["epe"]
                    row[f"{stage_name}_cmr_1px"] = stage_metrics["cmr_1px"]
                    row[f"{stage_name}_cmr_5px"] = stage_metrics["cmr_5px"]
                rows.append(row)
                total_samples += 1

    results = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)) + 1,
        "samples": total_samples,
        "mean_inference_seconds_per_pair": total_seconds / max(total_samples, 1),
        "overall": overall.compute(),
        "stages": {name: accumulator.compute() for name, accumulator in stage_overall.items()},
        "datasets": {name: accumulator.compute() for name, accumulator in by_dataset.items()},
        **output_metadata,
    }
    stem = "_".join(selected)
    json_path = output_dir / f"test_metrics_{stem}.json"
    csv_path = output_dir / f"test_per_sample_{stem}.csv"
    atomic_json_dump(results, json_path)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["dataset", "sample_id"])
        writer.writeheader()
        writer.writerows(rows)
    print(format_metrics(results["overall"], "TEST OVERALL"))
    for name, metrics in results["stages"].items():
        print(format_metrics(metrics, f"STAGE {name}"))
    for name, metrics in results["datasets"].items():
        print(format_metrics(metrics, f"TEST {name}"))
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
