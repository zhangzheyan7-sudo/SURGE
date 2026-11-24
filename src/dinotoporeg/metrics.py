from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable

import torch


THRESHOLDS = (1, 2, 3, 5)


def compute_registration_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    thresholds: Iterable[int] = THRESHOLDS,
) -> dict[str, float]:
    valid_pixels = valid.squeeze(1).bool()
    difference = prediction - target
    endpoint = torch.linalg.vector_norm(difference, dim=1)
    if not valid_pixels.any():
        result = {"epe": float("nan"), "mae": float("nan"), "rmse": float("nan"), "valid_pixels": 0.0}
        result.update({f"cmr_{threshold}px": float("nan") for threshold in thresholds})
        return result
    endpoint_valid = endpoint[valid_pixels]
    difference_valid = difference.permute(0, 2, 3, 1)[valid_pixels]
    result = {
        "epe": endpoint_valid.mean().item(),
        "mae": difference_valid.abs().mean().item(),
        "rmse": torch.sqrt((difference_valid * difference_valid).sum(dim=-1).mean()).item(),
        "valid_pixels": float(endpoint_valid.numel()),
    }
    result.update({f"cmr_{threshold}px": (endpoint_valid <= threshold).float().mean().item() for threshold in thresholds})
    return result


class MetricAccumulator:
    def __init__(self, thresholds: Iterable[int] = THRESHOLDS):
        self.thresholds = tuple(thresholds)
        self.sums: defaultdict[str, float] = defaultdict(float)
        self.valid_pixels = 0.0
        self.loss_weight = 0.0
        self.component_names: set[str] = set()

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
        losses: dict[str, torch.Tensor | float] | None = None,
    ) -> dict[str, float]:
        valid_pixels = valid.squeeze(1).bool()
        difference = prediction.detach() - target.detach()
        endpoint = torch.linalg.vector_norm(difference, dim=1)
        pixel_count = float(valid_pixels.sum().item())
        if pixel_count > 0:
            difference_valid = difference.permute(0, 2, 3, 1)[valid_pixels]
            endpoint_valid = endpoint[valid_pixels]
            self.sums["epe"] += endpoint_valid.sum().item()
            self.sums["mae"] += difference_valid.abs().sum().item() / 2.0
            self.sums["squared_endpoint"] += (difference_valid * difference_valid).sum().item()
            for threshold in self.thresholds:
                self.sums[f"cmr_{threshold}px"] += (endpoint_valid <= threshold).sum().item()
            self.valid_pixels += pixel_count
        batch_size = float(prediction.shape[0])
        if losses:
            for name, value in losses.items():
                scalar = value.detach().item() if torch.is_tensor(value) else float(value)
                self.sums[name] += scalar * batch_size
                self.component_names.add(name)
            self.loss_weight += batch_size
        return compute_registration_metrics(prediction.detach(), target.detach(), valid.detach(), self.thresholds)

    def compute(self) -> dict[str, float]:
        if self.valid_pixels <= 0:
            result = {"epe": float("nan"), "mae": float("nan"), "rmse": float("nan"), "valid_pixels": 0.0}
            result.update({f"cmr_{threshold}px": float("nan") for threshold in self.thresholds})
        else:
            result = {
                "epe": self.sums["epe"] / self.valid_pixels,
                "mae": self.sums["mae"] / self.valid_pixels,
                "rmse": math.sqrt(self.sums["squared_endpoint"] / self.valid_pixels),
                "valid_pixels": self.valid_pixels,
            }
            result.update({f"cmr_{threshold}px": self.sums[f"cmr_{threshold}px"] / self.valid_pixels for threshold in self.thresholds})
        if self.loss_weight > 0:
            for key in sorted(self.component_names):
                result[key] = self.sums[key] / self.loss_weight
        return result


def format_metrics(metrics: dict[str, Any], prefix: str = "") -> str:
    values = [f"EPE={metrics.get('epe', float('nan')):.4f}"]
    if "loss" in metrics:
        values.append(f"loss={metrics['loss']:.4f}")
    values.extend([f"CMR@{threshold}={100.0 * metrics.get(f'cmr_{threshold}px', float('nan')):.2f}%" for threshold in THRESHOLDS])
    return (prefix + " " if prefix else "") + " | ".join(values)
