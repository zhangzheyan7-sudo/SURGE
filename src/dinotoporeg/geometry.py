from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


def _batched_affine(affine: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if affine.shape[-2:] != (2, 3):
        raise ValueError(f"Expected affine [...,2,3], got {tuple(affine.shape)}")
    squeezed = affine.ndim == 2
    return (affine.unsqueeze(0) if squeezed else affine), squeezed


def _homogeneous(affine: torch.Tensor) -> torch.Tensor:
    bottom = affine.new_zeros((*affine.shape[:-2], 1, 3))
    bottom[..., 0, 2] = 1.0
    return torch.cat((affine, bottom), dim=-2)


def _pixel_to_normalized_matrix(
    size: Sequence[int], device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    height, width = int(size[0]), int(size[1])
    sx = 2.0 / max(width - 1, 1)
    sy = 2.0 / max(height - 1, 1)
    return torch.tensor(
        ((sx, 0.0, -1.0), (0.0, sy, -1.0), (0.0, 0.0, 1.0)),
        device=device,
        dtype=dtype,
    )


def pixel_affine_to_normalized(
    affine_pixel: torch.Tensor,
    source_size: Sequence[int],
    target_size: Sequence[int] | None = None,
) -> torch.Tensor:
    """Convert an optical-pixel -> SAR-pixel affine to align_corners=True coordinates."""
    affine, squeezed = _batched_affine(affine_pixel)
    target_size = source_size if target_size is None else target_size
    # Matrix inversion is unsupported for BF16/FP16 and is numerically sensitive.
    with torch.autocast(device_type=affine.device.type, enabled=False):
        affine = affine.float()
        source_to_normalized = _pixel_to_normalized_matrix(source_size, affine.device, affine.dtype)
        target_to_normalized = _pixel_to_normalized_matrix(target_size, affine.device, affine.dtype)
        normalized = target_to_normalized @ _homogeneous(affine) @ torch.linalg.inv(source_to_normalized)
    result = normalized[..., :2, :]
    return result[0] if squeezed else result


def normalized_affine_to_pixel(
    affine_normalized: torch.Tensor,
    source_size: Sequence[int],
    target_size: Sequence[int] | None = None,
) -> torch.Tensor:
    """Convert an align_corners=True normalized optical -> SAR affine to pixels."""
    affine, squeezed = _batched_affine(affine_normalized)
    target_size = source_size if target_size is None else target_size
    # Keep the inverse and surrounding matrix products in FP32 under AMP.
    with torch.autocast(device_type=affine.device.type, enabled=False):
        affine = affine.float()
        source_to_normalized = _pixel_to_normalized_matrix(source_size, affine.device, affine.dtype)
        target_to_normalized = _pixel_to_normalized_matrix(target_size, affine.device, affine.dtype)
        pixel = torch.linalg.inv(target_to_normalized) @ _homogeneous(affine) @ source_to_normalized
    result = pixel[..., :2, :]
    return result[0] if squeezed else result


def transform_points(points: torch.Tensor, affine: torch.Tensor) -> torch.Tensor:
    """Apply affine matrices to points stored as [...,N,2]."""
    matrices, squeezed = _batched_affine(affine)
    if points.ndim == 2:
        points = points.unsqueeze(0)
        points_squeezed = True
    else:
        points_squeezed = False
    if points.shape[0] == 1 and matrices.shape[0] > 1:
        points = points.expand(matrices.shape[0], -1, -1)
    if matrices.shape[0] == 1 and points.shape[0] > 1:
        matrices = matrices.expand(points.shape[0], -1, -1)
    ones = torch.ones_like(points[..., :1])
    transformed = torch.einsum("bij,bnj->bni", matrices, torch.cat((points, ones), dim=-1))
    if points_squeezed and squeezed:
        return transformed[0]
    return transformed


def affine_to_flow(
    affine: torch.Tensor,
    size: Sequence[int],
    *,
    normalized: bool = True,
    target_size: Sequence[int] | None = None,
) -> torch.Tensor:
    """Generate optical-coordinate displacement flow for an affine transform."""
    matrices, squeezed = _batched_affine(affine)
    height, width = int(size[0]), int(size[1])
    target_size = size if target_size is None else target_size
    work = normalized_affine_to_pixel(matrices, size, target_size) if normalized else matrices
    yy, xx = torch.meshgrid(
        torch.arange(height, device=work.device, dtype=work.dtype),
        torch.arange(width, device=work.device, dtype=work.dtype),
        indexing="ij",
    )
    source = torch.stack((xx, yy), dim=-1).reshape(1, -1, 2).expand(work.shape[0], -1, -1)
    destination = transform_points(source, work)
    flow = (destination - source).transpose(1, 2).reshape(work.shape[0], 2, height, width)
    return flow[0] if squeezed else flow


class DifferentiableRobustAffineEstimator(nn.Module):
    """FP32 weighted affine least squares with detached Huber IRLS weights."""

    def __init__(self, config: dict | None = None):
        super().__init__()
        config = config or {}
        self.regularization = float(config.get("regularization", 1.0e-4))
        self.irls_iterations = int(config.get("irls_iterations", 2))
        self.huber_delta = float(config.get("huber_delta", 0.05))
        self.condition_threshold = float(config.get("condition_threshold", 1.0e7))
        self.min_weight = float(config.get("min_weight", 1.0e-6))
        self.balance_grid = int(config.get("balance_grid", 1))
        self.min_spatial_cells = int(config.get("min_spatial_cells", 1))
        self.min_affine_points = int(config.get("min_affine_points", 3))
        self.normalize_fit_weights = bool(config.get("normalize_fit_weights", False))
        if self.balance_grid <= 0:
            raise ValueError("affine.balance_grid must be positive")

    @staticmethod
    def identity(batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        value = torch.tensor(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), device=device, dtype=dtype)
        return value.unsqueeze(0).expand(batch, -1, -1).clone()

    def _solve(self, design: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = targets.shape[0]
        design_b = design.unsqueeze(0).expand(batch, -1, -1)
        normal = torch.einsum("bni,bn,bnj->bij", design_b, weights, design_b)
        rhs = torch.einsum("bni,bn,bnk->bik", design_b, weights, targets)
        eye = torch.eye(3, device=normal.device, dtype=normal.dtype).unsqueeze(0)
        regularized = normal + self.regularization * eye
        condition = torch.linalg.cond(regularized)
        solved = torch.linalg.solve(regularized, rhs)
        fallback = torch.linalg.pinv(regularized) @ rhs
        use_fallback = (~torch.isfinite(solved).all(dim=(-2, -1))) | (condition > self.condition_threshold)
        theta = torch.where(use_fallback[:, None, None], fallback, solved)
        return theta, condition

    def _fit(
        self, design: torch.Tensor, targets: torch.Tensor, mass: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        theta, condition = self._solve(design, targets, mass)
        for _ in range(self.irls_iterations):
            predicted = torch.einsum("ni,bik->bnk", design, theta)
            residual = torch.linalg.vector_norm(predicted - targets, dim=-1)
            robust = torch.where(
                residual <= self.huber_delta,
                torch.ones_like(residual),
                self.huber_delta / residual.clamp_min(1.0e-8),
            ).detach()
            theta, condition = self._solve(design, targets, mass * robust)
        return theta, condition

    def _spatially_balanced_weights(
        self, source: torch.Tensor, mass: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        grid = self.balance_grid
        if grid == 1:
            occupied = (mass.sum(-1) > self.min_weight).long()
            return mass, occupied
        x = (((source[:, 0] + 1.0) * 0.5 * grid).floor().long()).clamp(0, grid - 1)
        y = (((source[:, 1] + 1.0) * 0.5 * grid).floor().long()).clamp(0, grid - 1)
        cell_index = y * grid + x
        cells = grid * grid
        cell_mass = mass.new_zeros((mass.shape[0], cells))
        cell_mass.scatter_add_(1, cell_index.unsqueeze(0).expand(mass.shape[0], -1), mass)
        gathered = cell_mass.gather(1, cell_index.unsqueeze(0).expand(mass.shape[0], -1))
        balanced = torch.where(gathered > self.min_weight, mass / gathered.clamp_min(self.min_weight), mass)
        if self.normalize_fit_weights:
            valid_count = (mass > self.min_weight).sum(-1).to(mass.dtype)
            balanced = balanced * (
                valid_count / balanced.sum(-1).clamp_min(self.min_weight)
            ).unsqueeze(-1)
        occupied = (cell_mass > self.min_weight).sum(-1)
        return balanced, occupied

    def forward(
        self,
        source_coordinates: torch.Tensor,
        target_coordinates: torch.Tensor,
        weights: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        device_type = target_coordinates.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            source = source_coordinates.float()
            targets = target_coordinates.float()
            mass = weights.float().clamp_min(0.0)
            if source.ndim == 3:
                if source.shape[0] != 1:
                    raise ValueError("source_coordinates must be [N,2] or [1,N,2]")
                source = source[0]
            if valid is not None:
                mass = mass * valid.float()
            finite = torch.isfinite(targets).all(dim=-1) & torch.isfinite(mass)
            mass = torch.where(finite, mass, torch.zeros_like(mass))
            effective = (mass > self.min_weight).sum(dim=-1)
            design = torch.cat((source, torch.ones_like(source[:, :1])), dim=-1)
            identity = self.identity(targets.shape[0], targets.device, targets.dtype)
            safe_targets = torch.nan_to_num(targets)
            original_theta, original_condition = self._fit(design, safe_targets, mass)
            balanced_mass, occupied_cells = self._spatially_balanced_weights(source, mass)
            balanced_theta, balanced_condition = self._fit(design, safe_targets, balanced_mass)
            use_balanced = (occupied_cells >= self.min_spatial_cells) & (
                effective >= self.min_affine_points
            )
            theta = torch.where(use_balanced[:, None, None], balanced_theta, original_theta)
            condition = torch.where(use_balanced, balanced_condition, original_condition)
            matrix = theta.transpose(1, 2)
            usable = (effective >= 3) & torch.isfinite(matrix).all(dim=(-2, -1))
            matrix = torch.where(usable[:, None, None], matrix, identity)
            fitted = transform_points(source, matrix)
            residual = torch.linalg.vector_norm(fitted - safe_targets, dim=-1)
            residual_mean = (residual * mass).sum(-1) / mass.sum(-1).clamp_min(self.min_weight)
            return {
                "matrix_normalized": matrix,
                "fit_residual": residual,
                "affine_fit_residual": residual_mean,
                "effective_point_count": effective.float(),
                "condition_number": condition,
                "affine_spatial_coverage": occupied_cells.float() / float(self.balance_grid**2),
                "affine_fallback_ratio": (~use_balanced).float(),
            }
