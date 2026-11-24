from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from PIL.PngImagePlugin import PngInfo

from .utils import ensure_dir, flow_warp


def _rgb_uint8(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().float().cpu().clamp(0.0, 1.0)
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.shape[0] == 1:
        tensor = tensor.expand(3, -1, -1)
    return (tensor.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)


def checkerboard(reference: np.ndarray, warped: np.ndarray, block: int) -> np.ndarray:
    height, width = reference.shape[:2]
    yy, xx = np.mgrid[:height, :width]
    choose_reference = ((xx // block + yy // block) % 2 == 0)[..., None]
    return np.where(choose_reference, reference, warped).astype(np.uint8)


def matching_lines(
    optical: np.ndarray,
    sar: np.ndarray,
    flow: np.ndarray,
    valid: np.ndarray,
    confidence: np.ndarray,
    target_flow: np.ndarray | None,
    grid_step: int,
    max_points: int,
    line_color: tuple[int, int, int] | None = None,
    image_gap: int = 0,
    gap_color: str = "black",
    line_region: Sequence[float] | None = None,
    spread_lines: bool = False,
) -> Image.Image:
    """Draw antialiased, one-pixel correspondences without endpoint markers.

    Green (or the supplied line color) means endpoint error below 1 input-image pixel;
    errors >= 1px (or nonfinite GT) are red. Without GT, accuracy is unknown
    and lines are neutral gray.
    """
    height, width = optical.shape[:2]
    sar_offset = width + image_gap
    canvas = Image.new("RGB", (sar_offset + width, height), gap_color)
    canvas.paste(Image.fromarray(optical), (0, 0))
    canvas.paste(Image.fromarray(sar), (sar_offset, 0))
    # Supersample only the lines so the underlying image pixels stay sharp.
    line_scale = 4
    line_layer = Image.new("RGBA", (canvas.width * line_scale, height * line_scale))
    draw = ImageDraw.Draw(line_layer)
    candidates = []
    margin = max(grid_step // 2, 1)
    rng = np.random.default_rng(42) if spread_lines else None
    for center_y in range(margin, height, grid_step):
        for center_x in range(margin, width, grid_step):
            x, y = center_x, center_y
            if rng is not None:
                x = int(rng.integers(x - margin, min(x - margin + grid_step, width)))
                y = int(rng.integers(y - margin, min(y - margin + grid_step, height)))
            target_x = x + float(flow[y, x, 0])
            target_y = y + float(flow[y, x, 1])
            if valid[y, x] and 0 <= target_x < width and 0 <= target_y < height:
                candidates.append((float(confidence[y, x]), x, y, target_x, target_y))
    if rng is not None and len(candidates) > max_points:
        indices = np.linspace(0, len(candidates), max_points, endpoint=False, dtype=int)
        selected = [candidates[index] for index in indices]
    else:
        if not spread_lines:
            candidates.sort(reverse=True)
        selected = candidates[:max_points]
    correct_count = 0
    incorrect_count = 0
    for _, x, y, target_x, target_y in selected:
        if target_flow is None:
            continue
        if target_flow is not None:
            difference = flow[y, x].astype(np.float64) - target_flow[y, x].astype(np.float64)
            error_px = float(np.linalg.norm(difference))
            correct = np.isfinite(error_px) and error_px < 1.0
            correct_count += int(correct)
            if not correct:
                incorrect_count += 1
                continue
            color = (255, 255, 0)
        coordinates = tuple(
            round(value * line_scale) for value in (x, y, sar_offset + target_x, target_y)
        )
        draw.line(coordinates, fill=(*color, 255), width=line_scale)
    line_layer = line_layer.resize(canvas.size, resample=Image.Resampling.LANCZOS)
    if line_region is not None:
        left, top, right, bottom = line_region
        box = (round(left * canvas.width), round(top * height),
               round(right * canvas.width), round(bottom * height))
        clipped_layer = Image.new("RGBA", canvas.size)
        clipped_layer.paste(line_layer.crop(box), box[:2])
        line_layer = clipped_layer
    result = Image.alpha_composite(canvas.convert("RGBA"), line_layer).convert("RGB")
    result.info["matching_line_count"] = str(correct_count)
    if target_flow is not None:
        result.info["correct_line_count_epe_lt_1px"] = str(correct_count)
        result.info["incorrect_line_count"] = "0"
        result.info["hidden_candidate_count"] = str(incorrect_count)
        result.info["visualization_filter"] = "GT EPE < 1px only; metrics unfiltered"
    return result


def save_registration_visuals(
    batch: dict[str, Any],
    outputs: dict[str, Any],
    output_dir: str | Path,
    grid_step: int = 64,
    max_points: int = 64,
    checker_size: int = 32,
    alpha: float = 0.5,
    line_color: tuple[int, int, int] | None = None,
    image_gap: int = 0,
    gap_color: str = "black",
    line_region: Sequence[float] | None = None,
    spread_lines: bool = False,
) -> dict[str, str]:
    output_dir = ensure_dir(output_dir)
    dataset_name = batch["dataset"][0]
    sample_id = batch["sample_id"][0]
    prefix = f"{dataset_name}_sample_{sample_id}"
    optical = _rgb_uint8(batch["optical"])
    from .rgb_display import resolve_optical
    optical, display_metadata = resolve_optical(optical, dataset_name, sample_id)
    sar = _rgb_uint8(batch["sar"])
    prediction_tensor = outputs["flow"][:1]
    sar_tensor = batch["sar"][:1]
    warped_tensor = flow_warp(sar_tensor, prediction_tensor.to(sar_tensor))
    warped = _rgb_uint8(warped_tensor)
    prediction = prediction_tensor[0].detach().float().cpu().permute(1, 2, 0).numpy()
    target = batch.get("flow")
    target_np = target[0].detach().float().cpu().permute(1, 2, 0).numpy() if target is not None else None
    valid = batch["valid"][0, 0].detach().cpu().numpy().astype(bool)
    confidence = F.interpolate(outputs["coarse_confidence"].float(), size=prediction.shape[:2], mode="bilinear", align_corners=True)
    confidence_np = confidence[0, 0].detach().cpu().numpy()

    chess = checkerboard(optical, warped, checker_size)
    lines = matching_lines(
        optical, sar, prediction, valid, confidence_np, target_np, grid_step, max_points,
        line_color=line_color, image_gap=image_gap, gap_color=gap_color, line_region=line_region,
        spread_lines=spread_lines,
    )
    outputs_paths = {
        "matching_lines": str(output_dir / f"{prefix}_matching_lines.png"),
        "warp": str(output_dir / f"{prefix}_warp.png"),
        "checkerboard": str(output_dir / f"{prefix}_checkerboard.png"),
    }
    metadata = PngInfo()
    lines.info.update(display_metadata)
    for key, value in lines.info.items():
        metadata.add_text(key, str(value))
    lines.save(outputs_paths["matching_lines"], pnginfo=metadata)
    Image.fromarray(optical).save(output_dir / f"{prefix}_optical_display.png")
    Image.fromarray(warped).save(outputs_paths["warp"])
    Image.fromarray(chess).save(outputs_paths["checkerboard"])
    return outputs_paths
