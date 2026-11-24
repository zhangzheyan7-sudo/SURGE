from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from pathlib import Path

import torch
from PIL import ImageColor
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dinotoporeg.config import load_config
from dinotoporeg.data import build_test_loader
from dinotoporeg.engine import load_checkpoint
from dinotoporeg.metrics import compute_registration_metrics
from dinotoporeg.model import DINOTopoReg
from dinotoporeg.utils import autocast_context, ensure_dir, to_device
from dinotoporeg.visualization import save_registration_visuals


def dataset_output_dir(output_root: str | Path, dataset: str) -> Path:
    """Return a dataset-specific visualization directory."""
    folder_names = {
        "whu_opt_sar512": "WHU",
        "os_dataset": "OSdataset",
        "ubcv2": "UBCv2",
    }
    folder_name = folder_names.get(dataset, dataset)
    root = Path(output_root)
    if root.name.casefold() == folder_name.casefold():
        return ensure_dir(root)
    return ensure_dir(root / folder_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize matching lines, warp overlay and checkerboard")
    parser.add_argument("--config", default="configs/whu_paper_aligned.yaml")
    parser.add_argument("--checkpoint", default="", help="Defaults to config.visualization.checkpoint")
    parser.add_argument("--dataset", help="Defaults to the dataset in the selected config")
    parser.add_argument("--sample-id", default="", help="Exact test sample_id; empty uses --index")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--count", type=int, default=700)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="", help="Parent directory for a uniquely named visualization run")
    parser.add_argument("--line-color", type=lambda color: ImageColor.getcolor(color, "RGB"),
                        help="Color for matches with GT endpoint error < 1px; errors >= 1px stay red")
    parser.add_argument("--grid-step", type=int, help="Pixel spacing of candidate matching points")
    parser.add_argument("--max-points", type=int, help="Maximum number of matching lines")
    parser.add_argument("--spread-lines", action="store_true", help="Spread matching points across the valid image area")
    parser.add_argument("--image-gap", type=int, default=0, help="Gap between the two images in pixels")
    parser.add_argument("--gap-color", default="black", help="Background color between the two images")
    parser.add_argument("--line-region", nargs=4, type=float, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
                        help="Visible line region as fractions of the combined image width and height")
    args = parser.parse_args()
    if args.grid_step is not None and args.grid_step < 1:
        parser.error("--grid-step must be positive")
    if args.max_points is not None and args.max_points < 1:
        parser.error("--max-points must be positive")
    if args.image_gap < 0:
        parser.error("--image-gap cannot be negative")
    if args.line_region is not None:
        left, top, right, bottom = args.line_region
        if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
            parser.error("--line-region requires increasing bounds between 0 and 1")
    return args


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    args.dataset = args.dataset or config["data"]["datasets"][0]["name"]
    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint or config["visualization"]["checkpoint"]).resolve()
    output_root = args.output_dir or config["visualization"]["output_dir"]
    run_dir = tempfile.mkdtemp(prefix="visualize_", dir=ensure_dir(output_root))
    output_dir = dataset_output_dir(run_dir, args.dataset)
    loader = build_test_loader(config, [args.dataset], batch_size=1)
    model = DINOTopoReg(config).to(device)
    load_checkpoint(checkpoint_path, model, map_location=device)
    model.eval()
    amp = bool(config.get("evaluation", {}).get("amp", True)) and device.type == "cuda"
    visual_cfg = config["visualization"]
    csv_path = output_dir / "visualization_records.csv"
    fieldnames = [
        "dataset", "sample_id", "source_id", "epe", "mae", "rmse",
        "cmr_1px", "cmr_2px", "cmr_3px", "cmr_5px", "valid_pixels",
        "affine_fit_residual_mean_px", "affine_inlier_ratio",
        "matching_lines", "warp", "checkerboard",
    ]
    saved = 0
    with csv_path.open("w", encoding="utf-8-sig", newline="") as csv_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=fieldnames)
        writer.writeheader()
        with torch.inference_mode():
            for index, batch in enumerate(tqdm(loader, desc="visualize", dynamic_ncols=True)):
                matches_id = not args.sample_id or batch["sample_id"][0] == args.sample_id
                matches_index = bool(args.sample_id) or index >= args.index
                if not (matches_id and matches_index):
                    continue
                batch = to_device(batch, device)
                with autocast_context(device, amp):
                    outputs = model(
                        batch["optical"], batch["sar"], batch["dino_optical"], batch["dino_sar"],
                        batch.get("optical_content_mask"), batch.get("sar_content_mask"),
                    )
                paths = save_registration_visuals(
                    batch,
                    outputs,
                    output_dir,
                    grid_step=args.grid_step or int(visual_cfg.get("grid_step", 64)),
                    max_points=args.max_points or int(visual_cfg.get("max_points", 64)),
                    checker_size=int(visual_cfg.get("checker_size", 32)),
                    alpha=float(visual_cfg.get("alpha", 0.5)),
                    line_color=args.line_color,
                    image_gap=args.image_gap,
                    gap_color=args.gap_color,
                    line_region=args.line_region,
                    spread_lines=args.spread_lines,
                )
                metrics = compute_registration_metrics(
                    outputs["flow"][:1].float(),
                    batch["flow"][:1].float(),
                    batch["valid"][:1],
                )
                writer.writerow({
                    "dataset": batch["dataset"][0],
                    "sample_id": batch["sample_id"][0],
                    "source_id": batch["source_id"][0],
                    **metrics,
                    "affine_fit_residual_mean_px": float(
                        outputs["affine_fit_residual_mean"][0].detach().float().cpu()
                    ),
                    "affine_inlier_ratio": float(
                        outputs["affine_inlier_ratio"][0].detach().float().cpu()
                    ),
                    **paths,
                })
                csv_handle.flush()
                saved += 1
                print("Saved visualization set:")
                for name, path in paths.items():
                    print(f"  {name}: {path}")
                if saved >= args.count:
                    break
    if saved == 0:
        raise ValueError("No test sample matched the requested --sample-id/--index")
    print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()
