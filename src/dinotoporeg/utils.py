from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_json_dump(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def resize_flow(flow: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Resize a Bx2xHxW flow while preserving displacement units."""
    source_h, source_w = flow.shape[-2:]
    target_h, target_w = size
    if (source_h, source_w) == (target_h, target_w):
        return flow
    resized = F.interpolate(flow, size=size, mode="bilinear", align_corners=True)
    resized[:, 0] *= (target_w - 1) / max(source_w - 1, 1)
    resized[:, 1] *= (target_h - 1) / max(source_h - 1, 1)
    return resized


def resize_mask(mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(mask.float(), size=size, mode="nearest") > 0.5


def flow_warp(moving: torch.Tensor, flow: torch.Tensor, padding_mode: str = "zeros") -> torch.Tensor:
    """Sample a moving image/feature at reference coordinates p + flow(p)."""
    batch, _, height, width = moving.shape
    if flow.shape[-2:] != (height, width):
        flow = resize_flow(flow, (height, width))
    yy, xx = torch.meshgrid(
        torch.arange(height, device=moving.device, dtype=flow.dtype),
        torch.arange(width, device=moving.device, dtype=flow.dtype),
        indexing="ij",
    )
    grid_x = xx.unsqueeze(0) + flow[:, 0]
    grid_y = yy.unsqueeze(0) + flow[:, 1]
    grid_x = 2.0 * grid_x / max(width - 1, 1) - 1.0
    grid_y = 2.0 * grid_y / max(height - 1, 1) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1)
    return F.grid_sample(
        moving,
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=True,
    )


def autocast_context(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return torch.autocast(device_type=device.type, enabled=False)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    result = {}
    for key, value in batch.items():
        result[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return result


def compact_history_row(row: dict[str, Any]) -> dict[str, Any]:
    """One history schema shared by JSON, CSV and checkpoint exports.

    Pixel-weighted endpoint metrics are kept separate from older batch-averaged
    loss diagnostics. Constant placeholder losses and duplicate aliases are not
    carried into compact histories. Optimizer/checkpoint state is unaffected.
    """
    metadata = {
        "epoch", "seconds", "stage", "output_affine_global", "output_revision",
        "epe_statistics", "best_val_epe", "meaningful_best_epe", "epochs_without_improvement",
    }
    metrics = {
        "loss", "epe", "geometry_epe", "raw_dense_epe", "valid_pixels",
        "cmr_1px", "cmr_2px", "cmr_3px", "cmr_5px",
        "final_residual_mean_px", "final_residual_max_px",
        "total_flow_loss", "affine_corner_loss", "local_uot_loss",
        "geometry_consistency_loss", "residual_smooth_loss", "match_loss",
        "candidate_coverage_loss", "relation_loss",
        "initial_affine_loss", "initial_affine_weight",
        "final_epe_batch", "geometry_epe_batch", "raw_dense_epe_batch",
    }
    result: dict[str, Any] = {}
    for key, value in row.items():
        if key in metadata or key.startswith("lr_"):
            result[key] = value
            continue
        split, separator, name = key.partition("_")
        if separator and split in ("train", "val") and (
            name in metrics or re.fullmatch(
                r"(?:s8|s4|s2|final|coarse)_fit_(?:fallback_ratio|spatial_coverage)", name,
            )
        ):
            result[key] = value
    for split in ("train", "val"):
        if f"{split}_geometry_epe" not in result:
            for old_name, new_name in (
                ("final_full_loss", "final_epe_batch"),
                ("recovered_affine_epe", "geometry_epe_batch"),
            ):
                if f"{split}_{old_name}" in row:
                    result[f"{split}_{new_name}"] = row[f"{split}_{old_name}"]
        if row.get("epe_statistics") != "valid_pixel_weighted" and f"{split}_raw_dense_epe" in result:
            result[f"{split}_raw_dense_epe_batch"] = result.pop(f"{split}_raw_dense_epe")
    return result


def compact_training_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [compact_history_row(row) for row in history]
