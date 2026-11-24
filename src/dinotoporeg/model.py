from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import (
    DifferentiableRobustAffineEstimator,
    affine_to_flow,
    normalized_affine_to_pixel,
    transform_points,
)
from .graph import (
    EdgeRelationEncoder,
    GridGraphBuilder,
    HybridDynamicGraphBuilder,
    OpticalNodeAdapter,
    ResidualGATEncoder,
    SARNodeAdapter,
    SharedNodeProjection,
    SparseLocalUnbalancedSinkhornMatcher,
    SparseRelationReasoner,
    UnbalancedSinkhornMatcher,
)
from .utils import flow_warp, resize_flow
from .tsfr import TwoScaleGeometricRefiner
from .sgu_flow import SGUFlowRefiner
from .structural_fusion import CrossModalStructuralLocalEncoder, SemanticStructureFusion


def _groups(channels: int) -> int:
    for value in (8, 4, 2, 1):
        if channels % value == 0:
            return value
    return 1


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.GELU(),
        )


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(channels, channels),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x + self.block(x))


class FrozenDINOv3(nn.Module):
    """Official local DINOv3 ViT-L/16 SAT-493M wrapper."""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        repo_dir = Path(config["repo_dir"])
        weights = Path(config["weights"])
        if not (repo_dir / "hubconf.py").exists():
            raise FileNotFoundError(f"Official DINOv3 repository not found: {repo_dir}")
        if not weights.exists():
            raise FileNotFoundError(f"DINOv3 checkpoint not found: {weights}")
        repo_text = str(repo_dir)
        if repo_text not in sys.path:
            sys.path.insert(0, repo_text)
        from dinov3.hub.backbones import dinov3_vitl16

        self.backbone = dinov3_vitl16(pretrained=False, weights=str(weights))
        try:
            state = torch.load(weights, map_location="cpu", weights_only=True, mmap=True)
        except TypeError:
            state = torch.load(weights, map_location="cpu")
        for key in ("model", "state_dict", "teacher", "backbone"):
            if isinstance(state, dict) and isinstance(state.get(key), dict):
                state = state[key]
                break
        self.backbone.load_state_dict(state, strict=True)
        self.frozen = bool(config.get("freeze", True))
        if self.frozen:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)
            self.backbone.eval()
        self.output_dim = int(config.get("feature_dim", 1024))

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def extract() -> torch.Tensor:
            features = self.backbone.get_intermediate_layers(x, n=1, reshape=True, norm=True)
            if isinstance(features, (tuple, list)):
                features = features[-1]
            return features

        if self.frozen:
            with torch.no_grad():
                return extract()
        return extract()


class RelationGraphEncoder(nn.Module):
    """Sparse 8-neighbour intra-image relation encoding on the patch grid."""

    def __init__(self, channels: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.GroupNorm(_groups(channels), channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x + self.encoder(x), dim=1)


class SparseUOTMatcher(nn.Module):
    """Top-k masked Sinkhorn normalization with a relaxed uniform marginal."""

    def __init__(self, iterations: int):
        super().__init__()
        self.iterations = int(iterations)

    def forward(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        value = logits.float().masked_fill(~mask, -1.0e4)
        for _ in range(self.iterations):
            value = value - torch.logsumexp(value, dim=-1, keepdim=True)
            value = value.masked_fill(~mask, -1.0e4)
            value = value - torch.logsumexp(value, dim=-2, keepdim=True)
            value = value.masked_fill(~mask, -1.0e4)
        return torch.softmax(value, dim=-1).to(dtype=logits.dtype)


class MatchDecoder(nn.Module):
    def __init__(self, channels: int, residual_radius: float):
        super().__init__()
        hidden = max(channels, 128)
        self.residual_radius = float(residual_radius)
        self.body = nn.Sequential(
            ConvNormAct(2 * channels + 3, hidden),
            ResidualBlock(hidden),
            ResidualBlock(hidden),
        )
        self.flow_head = nn.Conv2d(hidden, 2, 3, padding=1)
        self.confidence_head = nn.Conv2d(hidden, 1, 3, padding=1)

    def forward(
        self,
        reference: torch.Tensor,
        matched_moving: torch.Tensor,
        flow: torch.Tensor,
        confidence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale = float(max(flow.shape[-2:]))
        hidden = self.body(torch.cat((reference, matched_moving, flow / scale, confidence), dim=1))
        residual = torch.tanh(self.flow_head(hidden)) * self.residual_radius
        confidence_logit = torch.logit(confidence.clamp(1.0e-4, 1.0 - 1.0e-4))
        return residual, torch.sigmoid(confidence_logit + self.confidence_head(hidden))


class GraphAssistedCoarseMatcher(nn.Module):
    def __init__(self, input_dim: int, config: dict[str, Any]):
        super().__init__()
        channels = int(config["projection_dim"])
        self.projection = nn.Conv2d(input_dim, channels, 1)
        self.relation = RelationGraphEncoder(channels)
        self.decoder = (
            MatchDecoder(channels, float(config.get("match_residual_radius", 2.0)))
            if bool(geometry.get("residual_enabled", True)) else None
        )
        self.uot = SparseUOTMatcher(int(config.get("sinkhorn_iterations", 3)))
        self.temperature = float(config.get("temperature", 0.07))
        self.topk = int(config.get("topk", 8))
        self.graph_enabled = bool(config.get("graph_enabled", True))
        self.graph_weight = float(config.get("graph_weight", 1.0))
        self.mutual_weight = float(config.get("mutual_weight", 0.5))
        self.graph_sigma = float(config.get("graph_sigma", 2.0))

    @staticmethod
    def _coordinates(height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        yy, xx = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing="ij",
        )
        return torch.stack((xx, yy), dim=-1).reshape(-1, 2)

    def forward(self, reference: torch.Tensor, moving: torch.Tensor) -> dict[str, torch.Tensor]:
        # AMP may run the projection convolutions in BF16/FP16.  Keep the
        # correlation, graph prior and Sinkhorn path in FP32: graph geometry
        # operations can otherwise promote only part of the expression to
        # FP32 and make scatter_/einsum receive mixed dtypes.
        reference = F.normalize(self.projection(reference.float()).float(), dim=1)
        moving = F.normalize(self.projection(moving.float()).float(), dim=1)
        batch, channels, height, width = reference.shape
        count = height * width
        ref_flat = reference.flatten(2)
        mov_flat = moving.flatten(2)
        raw_logits = torch.einsum("bcn,bcm->bnm", ref_flat, mov_flat) / self.temperature

        relation_ref = self.relation(reference).float().flatten(2)
        relation_mov = self.relation(moving).float().flatten(2)
        relation_logits = torch.einsum("bcn,bcm->bnm", relation_ref, relation_mov) / math.sqrt(channels)
        fused_logits = raw_logits + (self.graph_weight * relation_logits if self.graph_enabled else 0.0)

        k = min(self.topk, count)
        top_values, top_indices = fused_logits.topk(k, dim=-1)
        coordinates = self._coordinates(height, width, fused_logits.device, fused_logits.dtype)
        candidate_coordinates = coordinates[top_indices]
        source_coordinates = coordinates.view(1, count, 1, 2)
        candidate_displacements = candidate_coordinates - source_coordinates
        initial_probabilities = torch.softmax(top_values, dim=-1)
        expected_displacement = (initial_probabilities.unsqueeze(-1) * candidate_displacements).sum(dim=2)
        displacement_map = expected_displacement.transpose(1, 2).reshape(batch, 2, height, width)

        ones = torch.ones((1, 1, height, width), device=fused_logits.device, dtype=fused_logits.dtype)
        neighbour_count = F.avg_pool2d(ones, 3, stride=1, padding=1) * 9.0
        neighbour_sum = F.avg_pool2d(displacement_map, 3, stride=1, padding=1) * 9.0
        neighbour_mean = (neighbour_sum / neighbour_count.clamp_min(1.0)).flatten(2).transpose(1, 2).unsqueeze(2)
        graph_score = -torch.linalg.vector_norm(candidate_displacements - neighbour_mean, dim=-1) / self.graph_sigma

        reverse_best = fused_logits.argmax(dim=1)
        reverse_for_candidates = reverse_best.gather(1, top_indices.reshape(batch, -1)).reshape(batch, count, k)
        source_index = torch.arange(count, device=fused_logits.device).view(1, count, 1)
        mutual_score = (reverse_for_candidates == source_index).to(fused_logits.dtype)
        candidate_logits = top_values
        if self.graph_enabled:
            candidate_logits = candidate_logits + self.graph_weight * graph_score + self.mutual_weight * mutual_score

        graph_logits = torch.full_like(fused_logits, -1.0e4)
        graph_logits.scatter_(-1, top_indices, candidate_logits.to(graph_logits.dtype))
        candidate_mask = torch.zeros_like(fused_logits, dtype=torch.bool)
        candidate_mask.scatter_(-1, top_indices, True)
        probabilities = self.uot(graph_logits, candidate_mask)

        expected_target = torch.einsum("bnm,md->bnd", probabilities, coordinates)
        coarse_flow = (expected_target - coordinates.unsqueeze(0)).transpose(1, 2).reshape(batch, 2, height, width)
        matched_moving = torch.einsum("bnm,bcm->bcn", probabilities, mov_flat).reshape(batch, channels, height, width)
        entropy = -(probabilities.clamp_min(1.0e-8) * probabilities.clamp_min(1.0e-8).log()).sum(dim=-1)
        confidence = probabilities.amax(dim=-1) * (1.0 - entropy / math.log(max(k, 2)))
        confidence = confidence.clamp(0.0, 1.0).reshape(batch, 1, height, width)
        coarse_flow, confidence = self.decoder(reference, matched_moving, coarse_flow, confidence)

        backward_probabilities = torch.softmax(graph_logits.transpose(1, 2).float(), dim=-1).to(graph_logits.dtype)
        backward_source = torch.einsum("bmn,nd->bmd", backward_probabilities, coordinates)
        cycle_coordinates = torch.einsum("bnm,bmd->bnd", probabilities, backward_source)
        return {
            "flow": coarse_flow,
            "confidence": confidence,
            "raw_logits": search_logits,
            "graph_logits": graph_logits,
            "candidate_indices": top_indices,
            "candidate_probabilities": probabilities.gather(-1, top_indices),
            "cycle_coordinates": cycle_coordinates,
            "source_coordinates": coordinates,
        }


class WeightedAffineEstimator(nn.Module):
    """Differentiable robust affine fit from the graph-assisted coarse flow.

    The estimator uses normalized coordinates for numerical stability and
    confidence-weighted IRLS to suppress locally inconsistent coarse matches.
    It predicts no free affine parameters of its own: the six parameters are
    solved from the current coarse correspondences, so gradients still reach
    the coarse matcher through the weighted least-squares system.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.confidence_power = float(config.get("confidence_power", 2.0))
        self.min_weight = float(config.get("min_weight", 0.01))
        self.regularization = float(config.get("regularization", 1.0e-4))
        self.irls_iterations = int(config.get("irls_iterations", 3))
        self.huber_delta = float(config.get("huber_delta", 2.5))
        self.min_huber_pixels = float(config.get("min_huber_pixels", 0.25))

    @staticmethod
    def _coordinates(
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        yy, xx = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing="ij",
        )
        x_normalized = 2.0 * xx / max(width - 1, 1) - 1.0
        y_normalized = 2.0 * yy / max(height - 1, 1) - 1.0
        design = torch.stack((x_normalized, y_normalized, torch.ones_like(x_normalized)), dim=-1)
        return xx, yy, design.reshape(-1, 3)

    def _solve(
        self,
        design: torch.Tensor,
        destination: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        batch = destination.shape[0]
        design_batch = design.unsqueeze(0).expand(batch, -1, -1)
        normal = torch.einsum("bni,bn,bnj->bij", design_batch, weights, design_batch)
        rhs = torch.einsum("bni,bn,bnk->bik", design_batch, weights, destination)
        identity = torch.eye(3, device=normal.device, dtype=normal.dtype).unsqueeze(0)
        return torch.linalg.solve(normal + self.regularization * identity, rhs)

    def forward(self, coarse_flow: torch.Tensor, confidence: torch.Tensor) -> dict[str, torch.Tensor]:
        # cuSOLVER does not implement BF16/FP16 LU factorization. The system is
        # only 3x3, so keeping this small block in FP32 is both stable and cheap.
        with torch.autocast(device_type=coarse_flow.device.type, enabled=False):
            return self._forward_fp32(coarse_flow, confidence)

    def _forward_fp32(self, coarse_flow: torch.Tensor, confidence: torch.Tensor) -> dict[str, torch.Tensor]:
        work_flow = coarse_flow.float()
        work_confidence = confidence.float().clamp(0.0, 1.0)
        batch, _, height, width = work_flow.shape
        xx, yy, design = self._coordinates(height, width, work_flow.device, work_flow.dtype)

        target_x = xx.unsqueeze(0) + work_flow[:, 0]
        target_y = yy.unsqueeze(0) + work_flow[:, 1]
        target_x_normalized = 2.0 * target_x / max(width - 1, 1) - 1.0
        target_y_normalized = 2.0 * target_y / max(height - 1, 1) - 1.0
        destination = torch.stack((target_x_normalized, target_y_normalized), dim=-1).reshape(batch, -1, 2)

        base_weights = work_confidence.flatten(1).pow(self.confidence_power)
        base_weights = self.min_weight + (1.0 - self.min_weight) * base_weights
        robust_weights = torch.ones_like(base_weights)
        theta = self._solve(design, destination, base_weights)
        scale = work_flow.new_tensor((max(width - 1, 1) / 2.0, max(height - 1, 1) / 2.0))
        cutoff = work_flow.new_full((batch, 1), self.min_huber_pixels)

        for _ in range(self.irls_iterations):
            predicted = torch.einsum("ni,bik->bnk", design, theta)
            residual = torch.linalg.vector_norm((predicted - destination) * scale.view(1, 1, 2), dim=-1)
            median = residual.median(dim=1, keepdim=True).values
            mad = (residual - median).abs().median(dim=1, keepdim=True).values
            robust_scale = (1.4826 * mad).clamp_min(self.min_huber_pixels / self.huber_delta)
            cutoff = (self.huber_delta * robust_scale).clamp_min(self.min_huber_pixels)
            robust_weights = torch.where(
                residual <= cutoff,
                torch.ones_like(residual),
                cutoff / residual.clamp_min(1.0e-6),
            )
            # The robust classification is deliberately detached. This avoids
            # unstable second-order behavior at the Huber switching boundary,
            # while the affine solution remains differentiable w.r.t. flow.
            theta = self._solve(design, destination, base_weights * robust_weights.detach())

        predicted = torch.einsum("ni,bik->bnk", design, theta)
        predicted_x = (predicted[..., 0] + 1.0) * (max(width - 1, 1) / 2.0)
        predicted_y = (predicted[..., 1] + 1.0) * (max(height - 1, 1) / 2.0)
        affine_flow = torch.stack(
            (
                predicted_x.reshape(batch, height, width) - xx.unsqueeze(0),
                predicted_y.reshape(batch, height, width) - yy.unsqueeze(0),
            ),
            dim=1,
        )
        fit_residual = torch.linalg.vector_norm(work_flow - affine_flow, dim=1, keepdim=True)
        inlier_ratio = (fit_residual.flatten(1) <= cutoff).float().mean(dim=1)
        return {
            "flow": affine_flow.to(dtype=coarse_flow.dtype),
            "matrix_normalized": theta.transpose(1, 2),
            "fit_residual": fit_residual,
            "fit_residual_mean": fit_residual.flatten(1).mean(dim=1),
            "inlier_ratio": inlier_ratio,
            "local_hint": (work_flow - affine_flow).to(dtype=coarse_flow.dtype),
        }


class GraphAssistedCoarseMatcherV2(nn.Module):
    """Cross-modal node matching, sparse relation reasoning, UOT and affine fit."""

    def __init__(
        self,
        input_dim: int,
        config: dict[str, Any],
        geometry: dict[str, Any],
        pyramid_channels: list[int] | None = None,
    ):
        super().__init__()
        channels = int(config.get("projection_dim", 256))
        adapter_hidden = int(config.get("adapter_dim", max(channels // 4, 32)))
        adapter_dropout = float(config.get("node_adapter_dropout", 0.1))
        self.shared_projection = SharedNodeProjection(input_dim, channels)
        self.optical_adapter = OpticalNodeAdapter(channels, adapter_hidden, adapter_dropout)
        self.sar_adapter = SARNodeAdapter(channels, adapter_hidden, adapter_dropout)
        self.optical_matchability = nn.Conv2d(channels, 1, 1)
        self.sar_matchability = nn.Conv2d(channels, 1, 1)
        self.graph_builder = GridGraphBuilder(
            int(config.get("graph_neighbors", 8)), 0
        )
        self.spatial_only = bool(config.get("spatial_only", False))
        self.relation_support_mode = str(config.get("relation_support_mode", "legacy_same_offset"))
        if self.relation_support_mode not in {"legacy_same_offset", "neighborhood_probability"}:
            raise ValueError("Unknown coarse.relation_support_mode")
        if self.relation_support_mode == "neighborhood_probability" and not self.spatial_only:
            raise ValueError("Paper neighbourhood RGCM requires coarse.spatial_only=true")
        self.dynamic_graph_enabled = bool(config.get("dynamic_graph_enabled", True))
        self.gat_enabled = bool(config.get("gat_enabled", True))
        # Graph structure is already a *learned* SALA descriptor. A separate
        # trainable projection before discrete KNN would receive zero gradient;
        # instead use the end-to-end trainable CSLE/SALA descriptor directly.
        self.hybrid_graph_builder = (HybridDynamicGraphBuilder(
            int(config.get("graph_neighbors", 8)),
            int(config.get("semantic_k", 4)),
            int(config.get("structure_k", 4)),
        ) if self.dynamic_graph_enabled else None)
        self.gat = (ResidualGATEncoder(
            channels,
            int(config.get("gat_layers", 2)),
            int(config.get("gat_heads", 4)),
            int(config.get("gat_edge_dim", 16)),
            float(config.get("gat_dropout", 0.1)),
        ) if self.dynamic_graph_enabled and self.gat_enabled else None)
        # Build the relation reasoner when two-round coarse MA-UOT is enabled.
        if bool(config.get("graph_enabled", True)) and int(config.get("graph_rounds", 2)) > 1:
            relation_encoder = EdgeRelationEncoder(
                channels, int(config.get("relation_dim", 32)),
                float(config.get("relation_dropout", 0.1)),
            )
            self.reasoner = SparseRelationReasoner(relation_encoder, config)
        else:
            self.reasoner = None
        self.uot = UnbalancedSinkhornMatcher(
            epsilon=float(config.get("uot_epsilon", 0.08)),
            tau_source=float(config.get("uot_tau_source", 0.8)),
            tau_target=float(config.get("uot_tau_target", 0.8)),
            iterations=int(config.get("uot_iterations", 12)),
            dustbin_score=float(config.get("dustbin_score", -1.0)),
            dustbin_enabled=bool(config.get("dustbin_enabled", True)),
        )
        self.estimator = DifferentiableRobustAffineEstimator(geometry)
        self.decoder = (
            MatchDecoder(channels, float(config.get("match_residual_radius", 2.0)))
            if bool(geometry.get("residual_enabled", True)) else None
        )
        self.temperature = float(config.get("temperature", 0.07))
        legacy_topk = int(config.get("topk", 8))
        self.pre_topk = int(config.get("pre_topk", legacy_topk))
        self.graph_topk = int(config.get("graph_topk", legacy_topk))
        self.topk = self.graph_topk
        if self.pre_topk < self.graph_topk:
            raise ValueError("coarse.pre_topk must be >= coarse.graph_topk")
        self.graph_enabled = bool(config.get("graph_enabled", True))
        self.uot_enabled = bool(config.get("uot_enabled", True))
        self.graph_rounds = max(1, int(config.get("graph_rounds", 2)))
        self.graph_support_weight = float(config.get("graph_support_weight", 0.5))
        self.mutual_weight = float(config.get("mutual_weight", 0.5))
        self.cycle_mass_threshold = float(config.get("cycle_mass_threshold", 1.0e-5))
        self.affine_enabled = bool(geometry.get("affine_enabled", True))
        self.residual_enabled = bool(geometry.get("residual_enabled", True))
        self.multiscale_uot_enabled = bool(config.get("multiscale_uot_enabled", True))
        self.multiscale_uot_dim = int(config.get("multiscale_uot_dim", 96))
        self.s8_window_radius = int(config.get("s8_window_radius", 4))
        self.s4_window_radius = int(config.get("s4_window_radius", 3))
        self.local_uot_confidence_threshold = float(
            config.get("local_uot_confidence_threshold", 0.01)
        )
        self.prior_blend_max = float(config.get("prior_blend_max", 0.75))
        self.local_uot = SparseLocalUnbalancedSinkhornMatcher(
            epsilon=float(config.get("local_uot_epsilon", 0.08)),
            tau_source=float(config.get("local_uot_tau_source", 0.8)),
            tau_target=float(config.get("local_uot_tau_target", 0.8)),
            iterations=int(config.get("local_uot_iterations", 8)),
            dustbin_score=float(config.get("local_uot_dustbin_score", -1.0)),
        ) if self.multiscale_uot_enabled else None
        if pyramid_channels is None:
            pyramid_channels = [128, 96, 64]
        if len(pyramid_channels) < 3:
            raise ValueError("pyramid_channels must provide s16, s8 and s4 channels")
        self.s8_projection = (
            nn.Conv2d(int(pyramid_channels[1]), self.multiscale_uot_dim, 1, bias=False)
            if self.multiscale_uot_enabled else None
        )
        self.s4_projection = (
            nn.Conv2d(int(pyramid_channels[2]), self.multiscale_uot_dim, 1, bias=False)
            if self.multiscale_uot_enabled else None
        )

    @staticmethod
    def _scatter_candidates(values: torch.Tensor, indices: torch.Tensor, count: int) -> torch.Tensor:
        result = values.new_zeros((*values.shape[:-1], count))
        result.scatter_add_(-1, indices, values)
        return result

    def _match(
        self,
        logits: torch.Tensor,
        indices: torch.Tensor,
        source_mass: torch.Tensor,
        target_mass: torch.Tensor,
        count: int,
        source_valid: torch.Tensor | None = None,
        target_valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.uot_enabled:
            return self.uot(
                logits, indices, count, source_mass, target_mass, source_valid, target_valid
            )
        candidate_valid = torch.ones_like(logits, dtype=torch.bool)
        if target_valid is not None:
            candidate_valid &= target_valid.gather(1, indices.reshape(indices.shape[0], -1)).reshape_as(indices)
        if source_valid is not None:
            candidate_valid &= source_valid.unsqueeze(-1)
        conditional = torch.softmax(logits.float().masked_fill(~candidate_valid, -1.0e4), dim=-1)
        conditional = conditional * candidate_valid.float()
        conditional = conditional / conditional.sum(-1, keepdim=True).clamp_min(1.0e-8)
        conditional = conditional.to(logits.dtype)
        candidate_mass = conditional * source_mass.unsqueeze(-1)
        transport = self._scatter_candidates(candidate_mass, indices, count)
        zeros_source = source_mass.new_zeros(source_mass.shape)
        zeros_target = target_mass.new_zeros(target_mass.shape)
        return {
            "transport": transport,
            "transport_with_dustbin": transport,
            "unmatched_source_mass": zeros_source,
            "unmatched_target_mass": zeros_target,
            "row_transport_mass": transport.sum(-1),
            "column_transport_mass": transport.sum(-2),
        }

    def _estimate_affine(
        self,
        graph: dict[str, torch.Tensor],
        match: dict[str, torch.Tensor],
        optical_matchability: torch.Tensor,
        source_valid: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        transport = match["transport"].float()
        row_mass = match["row_transport_mass"].float()
        coordinates = graph["coordinates"].to(transport)
        expected_target = torch.einsum("bnm,md->bnd", transport, coordinates)
        expected_target = expected_target / row_mass.unsqueeze(-1).clamp_min(1.0e-8)
        conditional = transport / row_mass.unsqueeze(-1).clamp_min(1.0e-8)
        entropy = -(conditional.clamp_min(1.0e-8) * conditional.clamp_min(1.0e-8).log()).sum(-1)
        entropy_confidence = torch.exp(-entropy)
        max_probability = conditional.amax(-1)
        fit_weights = row_mass * max_probability * entropy_confidence * optical_matchability
        fit_valid = row_mass > 1.0e-8
        if source_valid is not None:
            fit_valid &= source_valid
        affine = self.estimator(coordinates, expected_target, fit_weights, fit_valid)
        if not self.affine_enabled:
            affine["matrix_normalized"] = self.estimator.identity(
                transport.shape[0], transport.device, transport.dtype
            )
        confidence = (max_probability * entropy_confidence).clamp(0.0, 1.0)
        return affine, expected_target, confidence, entropy

    @staticmethod
    def _local_graph_support(
        raw_logits: torch.Tensor,
        candidate_indices: torch.Tensor,
        graph: dict[str, torch.Tensor],
        height: int,
        width: int,
        source_valid: torch.Tensor,
        target_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Score Top-K candidates by 8-neighbour translation consistency.

        This is deliberately O(B*N*Q*K), rather than invoking the full relation
        reasoner before pruning (whose pair construction is quadratic in K).
        The unchanged full relation module still operates on the final Top-K.
        """
        batch, count, topk = candidate_indices.shape
        neighbours = graph["neighbor_indices"]
        graph_valid = graph["neighbor_valid"]
        source_index = torch.arange(count, device=raw_logits.device)
        source_x = source_index.remainder(width).view(count, 1)
        source_y = torch.div(source_index, width, rounding_mode="floor").view(count, 1)
        neighbour_x = neighbours.remainder(width)
        neighbour_y = torch.div(neighbours, width, rounding_mode="floor")
        offset_x = neighbour_x - source_x
        offset_y = neighbour_y - source_y

        candidate_x = candidate_indices.remainder(width).unsqueeze(2)
        candidate_y = torch.div(candidate_indices, width, rounding_mode="floor").unsqueeze(2)
        shifted_x = candidate_x + offset_x.view(1, count, -1, 1)
        shifted_y = candidate_y + offset_y.view(1, count, -1, 1)
        shifted_valid = (
            (shifted_x >= 0)
            & (shifted_x < width)
            & (shifted_y >= 0)
            & (shifted_y < height)
        )
        shifted_indices = (
            shifted_y.clamp(0, height - 1) * width + shifted_x.clamp(0, width - 1)
        ).long()

        neighbour_sources = neighbours.view(1, count, -1, 1).expand(batch, -1, -1, topk)
        batch_index = torch.arange(batch, device=raw_logits.device).view(batch, 1, 1, 1)
        relation_logits = raw_logits[batch_index, neighbour_sources, shifted_indices]
        relation_valid = shifted_valid & graph_valid.view(1, count, -1, 1)
        relation_valid &= source_valid[:, neighbours].unsqueeze(-1)
        relation_valid &= target_valid.gather(1, shifted_indices.reshape(batch, -1)).reshape_as(
            shifted_indices
        )
        relation_weight = relation_valid.to(relation_logits.dtype)
        support = (torch.sigmoid(relation_logits) * relation_weight).sum(dim=2)
        support = support / relation_weight.sum(dim=2).clamp_min(1.0)
        return support.clamp_min(1.0e-8)

    @staticmethod
    def _hybrid_graph_stats(graph: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        valid = graph["neighbor_valid"]
        edge_types = graph["edge_types"]
        return {
            "spatial_neighbors": (valid & (edge_types == 0)).sum(-1).float().mean(),
            "semantic_neighbors": (valid & (edge_types == 1)).sum(-1).float().mean(),
            "structure_neighbors": (valid & (edge_types == 2)).sum(-1).float().mean(),
        }

    def _local_uot_scale(
        self,
        optical_features: torch.Tensor,
        sar_features: torch.Tensor,
        projection: nn.Module,
        prior_flow: torch.Tensor,
        radius: int,
        source_valid_map: torch.Tensor | None,
        target_valid_map: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        optical = F.normalize(projection(optical_features).float(), dim=1)
        sar = F.normalize(projection(sar_features).float(), dim=1)
        batch, channels, height, width = optical.shape
        count = height * width
        prior = resize_flow(prior_flow, (height, width)).float()
        yy, xx = torch.meshgrid(
            torch.arange(height, device=optical.device),
            torch.arange(width, device=optical.device),
            indexing="ij",
        )
        offsets_y, offsets_x = torch.meshgrid(
            torch.arange(-radius, radius + 1, device=optical.device),
            torch.arange(-radius, radius + 1, device=optical.device),
            indexing="ij",
        )
        offsets_x = offsets_x.reshape(1, 1, -1)
        offsets_y = offsets_y.reshape(1, 1, -1)
        predicted_x = xx.reshape(1, count).float() + prior[:, 0].flatten(1)
        predicted_y = yy.reshape(1, count).float() + prior[:, 1].flatten(1)
        candidate_x = predicted_x.round().long().unsqueeze(-1) + offsets_x
        candidate_y = predicted_y.round().long().unsqueeze(-1) + offsets_y
        candidate_valid = (
            (candidate_x >= 0)
            & (candidate_x < width)
            & (candidate_y >= 0)
            & (candidate_y < height)
        )
        safe_x = candidate_x.clamp(0, width - 1)
        safe_y = candidate_y.clamp(0, height - 1)
        candidate_indices = (safe_y * width + safe_x).long()
        if source_valid_map is None:
            source_valid = torch.ones((batch, count), dtype=torch.bool, device=optical.device)
        else:
            source_valid = source_valid_map.flatten(1).bool()
        if target_valid_map is None:
            target_valid = torch.ones((batch, count), dtype=torch.bool, device=optical.device)
        else:
            target_valid = target_valid_map.flatten(1).bool()
        gathered_target_valid = target_valid.gather(
            1, candidate_indices.reshape(batch, -1)
        ).reshape_as(candidate_indices)
        candidate_valid &= source_valid.unsqueeze(-1) & gathered_target_valid

        optical_nodes = optical.flatten(2).transpose(1, 2)
        sar_nodes = sar.flatten(2).transpose(1, 2)
        batch_index = torch.arange(batch, device=optical.device).view(batch, 1, 1)
        candidate_nodes = sar_nodes[batch_index, candidate_indices]
        candidate_logits = (
            optical_nodes.unsqueeze(2) * candidate_nodes
        ).sum(-1) / self.temperature
        candidate_logits = candidate_logits.masked_fill(~candidate_valid, -1.0e4)
        local_match = self.local_uot(
            candidate_logits,
            candidate_indices,
            candidate_valid,
            source_valid.float(),
            target_valid.float(),
            count,
        )
        row_mass = local_match["row_transport_mass"].float()
        edge_transport = local_match["edge_transport"].float()
        conditional = edge_transport / row_mass.unsqueeze(-1).clamp_min(1.0e-8)
        candidate_coordinates = torch.stack(
            (candidate_x.float(), candidate_y.float()), dim=-1
        )
        expected = (conditional.unsqueeze(-1) * candidate_coordinates).sum(dim=2)
        source_coordinates = torch.stack((xx, yy), dim=-1).reshape(1, count, 2).float()
        flow = (expected - source_coordinates).transpose(1, 2).reshape(batch, 2, height, width)
        centered = candidate_coordinates - expected.unsqueeze(2)
        covariance = (conditional.unsqueeze(-1) * centered.square()).sum(dim=2)
        covariance = covariance.transpose(1, 2).reshape(batch, 2, height, width)
        entropy = local_match["transport_entropy"].float()
        valid_count = candidate_valid.sum(-1).clamp_min(1)
        normalized_entropy = entropy / valid_count.clamp_min(2).float().log()
        maximum_probability = conditional.amax(-1)
        row_score = (row_mass / source_valid.float().clamp_min(1.0e-8)).clamp(0.0, 1.0)
        confidence = (
            row_score.clamp_min(1.0e-8)
            * maximum_probability.clamp_min(1.0e-8)
            * torch.exp(-normalized_entropy).clamp_min(1.0e-8)
        ).pow(1.0 / 3.0)
        finite = torch.isfinite(flow).all(dim=1).flatten(1)
        good = (
            candidate_valid.any(-1)
            & (row_mass > 1.0e-8)
            & finite
            & torch.isfinite(confidence)
            & (confidence >= self.local_uot_confidence_threshold)
        )
        good_map = good.reshape(batch, 1, height, width)
        flow = torch.where(good_map, flow, prior)
        confidence_map = torch.where(
            good_map, confidence.reshape(batch, 1, height, width), torch.zeros_like(good_map, dtype=flow.dtype)
        )
        default_covariance = covariance.new_full(covariance.shape, float(max(radius, 1) ** 2))
        covariance = torch.where(good_map, torch.nan_to_num(covariance), default_covariance)
        entropy_map = torch.nan_to_num(entropy).reshape(batch, 1, height, width)
        return {
            "flow": flow.to(optical_features.dtype),
            "confidence": confidence_map.to(optical_features.dtype),
            "entropy": entropy_map.to(optical_features.dtype),
            "covariance": covariance.to(optical_features.dtype),
            "candidate_count": candidate_logits.new_tensor(candidate_logits.shape[-1]),
        }

    def forward(
        self,
        reference: torch.Tensor,
        moving: torch.Tensor,
        optical_content_mask: torch.Tensor | None = None,
        sar_content_mask: torch.Tensor | None = None,
        optical_structure: torch.Tensor | None = None,
        sar_structure: torch.Tensor | None = None,
        optical_s8: torch.Tensor | None = None,
        sar_s8: torch.Tensor | None = None,
        optical_s4: torch.Tensor | None = None,
        sar_s4: torch.Tensor | None = None,
        optical_s8_mask: torch.Tensor | None = None,
        sar_s8_mask: torch.Tensor | None = None,
        optical_s4_mask: torch.Tensor | None = None,
        sar_s4_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        optical_base = self.shared_projection(reference.float()).float()
        sar_base = self.shared_projection(moving.float()).float()
        optical_adapted = optical_base + self.optical_adapter(optical_base).float()
        sar_adapted = sar_base + self.sar_adapter(sar_base).float()
        batch, channels, height, width = optical_adapted.shape
        count = height * width
        if optical_content_mask is None:
            optical_content_mask = torch.ones(
                (batch, 1, height, width), dtype=torch.bool, device=optical_adapted.device
            )
        if sar_content_mask is None:
            sar_content_mask = torch.ones(
                (batch, 1, height, width), dtype=torch.bool, device=sar_adapted.device
            )
        source_valid = optical_content_mask.flatten(1).bool()
        target_valid = sar_content_mask.flatten(1).bool()
        optical_nodes = optical_adapted.flatten(2).transpose(1, 2)
        sar_nodes = sar_adapted.flatten(2).transpose(1, 2)
        empty_stat = optical_adapted.new_zeros(())
        optical_graph_stats = {
            "spatial_neighbors": empty_stat,
            "semantic_neighbors": empty_stat,
            "structure_neighbors": empty_stat,
        }
        sar_graph_stats = {key: value.clone() for key, value in optical_graph_stats.items()}
        if self.dynamic_graph_enabled and not self.spatial_only:
            # The main model supplies CSLE/SALA structural features. Standalone
            # matcher calls fall back to learned projected node descriptors;
            # at no point are raw-image gradients computed.
            if optical_structure is None:
                optical_structure = optical_adapted
            if sar_structure is None:
                sar_structure = sar_adapted
            if optical_structure.shape != optical_adapted.shape or sar_structure.shape != sar_adapted.shape:
                raise ValueError('Learned graph structure must have [B,projection_dim,H/16,W/16]')
            optical_structure = F.normalize(optical_structure.float(), dim=1)
            sar_structure = F.normalize(sar_structure.float(), dim=1)
            optical_structure = optical_structure.flatten(2).transpose(1, 2)
            sar_structure = sar_structure.flatten(2).transpose(1, 2)
            optical_hybrid_graph = self.hybrid_graph_builder(
                optical_nodes, optical_structure, height, width, source_valid
            )
            sar_hybrid_graph = self.hybrid_graph_builder(
                sar_nodes, sar_structure, height, width, target_valid
            )
            optical_graph_stats = self._hybrid_graph_stats(optical_hybrid_graph)
            sar_graph_stats = self._hybrid_graph_stats(sar_hybrid_graph)
            if self.gat_enabled:
                optical_nodes = self.gat(optical_nodes, optical_hybrid_graph)
                sar_nodes = self.gat(sar_nodes, sar_hybrid_graph)
        optical_nodes = F.normalize(optical_nodes, dim=-1)
        sar_nodes = F.normalize(sar_nodes, dim=-1)
        optical = optical_nodes.transpose(1, 2).reshape(batch, channels, height, width)
        sar = sar_nodes.transpose(1, 2).reshape(batch, channels, height, width)
        graph = self.graph_builder(height, width, optical.device)
        optical_nodes = optical_nodes * source_valid.unsqueeze(-1).to(optical_nodes.dtype)
        sar_nodes = sar_nodes * target_valid.unsqueeze(-1).to(sar_nodes.dtype)
        raw_logits = torch.einsum("bnc,bmc->bnm", optical_nodes, sar_nodes) / self.temperature
        search_logits = raw_logits.masked_fill(~target_valid.unsqueeze(1), -1.0e4)
        search_logits = search_logits.masked_fill(~source_valid.unsqueeze(-1), -1.0e4)
        pre_k = min(self.pre_topk, count)
        pre_candidate_logits, pre_candidate_indices = search_logits.topk(pre_k, dim=-1)
        pre_target_valid = target_valid.gather(
            1, pre_candidate_indices.reshape(batch, -1)
        ).reshape_as(pre_candidate_indices)
        pre_candidate_valid = source_valid.unsqueeze(-1) & pre_target_valid
        pre_candidate_logits = pre_candidate_logits.masked_fill(~pre_candidate_valid, -1.0e4)
        optical_mass = (
            torch.sigmoid(self.optical_matchability(optical)).flatten(1).float().clamp_min(1.0e-4)
            * source_valid.float()
        )
        sar_mass = (
            torch.sigmoid(self.sar_matchability(sar)).flatten(1).float().clamp_min(1.0e-4)
            * target_valid.float()
        )

        reverse_best = search_logits.argmax(dim=1)
        reverse_for_pre = reverse_best.gather(
            1, pre_candidate_indices.reshape(batch, -1)
        ).reshape(batch, count, pre_k)
        source_index = torch.arange(count, device=raw_logits.device).view(1, count, 1)
        pre_mutual_score = (reverse_for_pre == source_index).to(raw_logits.dtype)
        if self.relation_support_mode == "neighborhood_probability":
            from .paper_rgcm import neighborhood_support
            pre_graph_support, pre_relation_valid = neighborhood_support(
                raw_logits, pre_candidate_indices, graph, source_valid, target_valid,
            )
        else:
            pre_graph_support = self._local_graph_support(
                raw_logits, pre_candidate_indices, graph, height, width, source_valid, target_valid,
            )
            pre_relation_valid = pre_candidate_valid
        rerank_logits = (
            pre_candidate_logits
            + self.graph_support_weight * pre_graph_support.log()
            + self.mutual_weight * pre_mutual_score
        ).masked_fill(~pre_candidate_valid, -1.0e4)
        k = min(self.graph_topk, pre_k)
        current_logits, keep_positions = rerank_logits.topk(k, dim=-1)
        candidate_indices = pre_candidate_indices.gather(-1, keep_positions)
        candidate_logits = pre_candidate_logits.gather(-1, keep_positions)
        mutual_score = pre_mutual_score.gather(-1, keep_positions)
        candidate_valid = pre_candidate_valid.gather(-1, keep_positions)
        current_logits = current_logits.masked_fill(~candidate_valid, -1.0e4)
        match = self._match(
            current_logits, candidate_indices, optical_mass, sar_mass, count, source_valid, target_valid
        )
        affine, expected_target, confidence, entropy = self._estimate_affine(
            graph, match, optical_mass, source_valid
        )
        relation = {
            "graph_support": torch.ones_like(current_logits),
            "relation_cost": torch.zeros_like(current_logits),
            "geometry_cost": torch.zeros_like(current_logits),
            "compatibility": current_logits.new_zeros((batch, count, k, 1, 1)),
            "association_mass": current_logits.new_zeros((batch, count, 1, 1, 1)),
        }
        if self.relation_support_mode == "neighborhood_probability":
            support = pre_graph_support.gather(-1, keep_positions)
            relation_valid = pre_relation_valid.gather(-1, keep_positions) & candidate_valid
            # Section III-E specifies the role, but not the formula, of L_rel.
            # Explicit implementation choice: assignment-weighted -log(R_ij).
            relation.update(
                graph_support=support,
                relation_cost=torch.where(relation_valid, -support.log(), torch.zeros_like(support)),
                association_mass=relation_valid.any(-1).to(support.dtype).view(batch, count, 1, 1, 1),
            )
        rounds = self.graph_rounds if self.graph_enabled and not self.spatial_only else 1
        for _ in range(1, rounds):
            candidate_transport = match["transport"].gather(-1, candidate_indices)
            relation = self.reasoner(
                optical_nodes,
                sar_nodes,
                graph,
                candidate_indices,
                candidate_transport,
                affine["matrix_normalized"],
            )
            current_logits = (
                candidate_logits
                + self.graph_support_weight * relation["graph_support"].clamp_min(1.0e-8).log()
                + self.mutual_weight * mutual_score
            )
            current_logits = current_logits.masked_fill(~candidate_valid, -1.0e4)
            match = self._match(
                current_logits, candidate_indices, optical_mass, sar_mass, count, source_valid, target_valid
            )
            affine, expected_target, confidence, entropy = self._estimate_affine(
                graph, match, optical_mass, source_valid
            )

        matrix_normalized = affine["matrix_normalized"]
        affine_flow = affine_to_flow(matrix_normalized, (height, width), normalized=True)
        row_mass = match["row_transport_mass"].float()
        conditional = match["transport"].float() / row_mass.unsqueeze(-1).clamp_min(1.0e-8)
        matched_moving = torch.einsum("bnm,bmc->bnc", conditional, sar_nodes)
        matched_moving = matched_moving.transpose(1, 2).reshape(batch, channels, height, width)
        confidence_map = confidence.reshape(batch, 1, height, width)
        if self.decoder is not None:
            coarse_residual, confidence_map = self.decoder(
                optical, matched_moving, affine_flow, confidence_map
            )
        else:
            coarse_residual = torch.zeros_like(affine_flow)
        coarse_total = affine_flow + coarse_residual

        column_mass = match["column_transport_mass"].float()
        valid_columns = column_mass > self.cycle_mass_threshold
        backward_source = torch.einsum("bnm,nd->bmd", match["transport"].float(), graph["coordinates"].to(raw_logits))
        backward_source = backward_source / column_mass.unsqueeze(-1).clamp_min(1.0e-8)
        cycle_transport = match["transport"].float() * valid_columns.unsqueeze(1)
        cycle_row_mass = cycle_transport.sum(-1)
        cycle_coordinates = torch.einsum("bnm,bmd->bnd", cycle_transport, backward_source)
        cycle_coordinates = cycle_coordinates / cycle_row_mass.unsqueeze(-1).clamp_min(1.0e-8)
        cycle_weight = cycle_row_mass * optical_mass
        transport_entropy = entropy
        fitted_target = transform_points(graph["coordinates"].to(raw_logits), matrix_normalized)
        pixel_scale = raw_logits.new_tensor((max(width - 1, 1) / 2.0, max(height - 1, 1) / 2.0))
        affine_fit_residual_px = torch.linalg.vector_norm(
            (fitted_target - expected_target) * pixel_scale, dim=-1
        )
        candidate_transport = match["transport"].gather(-1, candidate_indices)
        graph_logits = raw_logits.new_full(raw_logits.shape, -1.0e4)
        graph_logits.scatter_(-1, candidate_indices, current_logits.to(graph_logits.dtype))
        border_denominator = (source_valid.sum().float() * float(pre_k)).clamp_min(1.0)
        border_candidate_ratio = (
            ((~pre_target_valid) & source_valid.unsqueeze(-1)).sum().float() / border_denominator
        )
        coordinate_delta = graph["coordinates"].to(conditional).view(1, 1, count, 2) - expected_target.unsqueeze(2)
        covariance_s16 = torch.einsum(
            "bnm,bnmd->bnd", conditional, coordinate_delta.square()
        )
        pixel_variance_scale = conditional.new_tensor(
            ((max(width - 1, 1) / 2.0) ** 2, (max(height - 1, 1) / 2.0) ** 2)
        )
        covariance_s16 = (covariance_s16 * pixel_variance_scale).transpose(1, 2).reshape(
            batch, 2, height, width
        )
        multiscale_flows = [coarse_total]
        multiscale_confidences = [confidence_map]
        multiscale_entropies = [transport_entropy.reshape(batch, 1, height, width)]
        multiscale_covariances = [covariance_s16]
        multiscale_candidate_counts = [current_logits.new_tensor(k)]
        pyramid_available = all(
            value is not None for value in (optical_s8, sar_s8, optical_s4, sar_s4)
        )
        if self.multiscale_uot_enabled and pyramid_available:
            s8_result = self._local_uot_scale(
                optical_s8,
                sar_s8,
                self.s8_projection,
                coarse_total,
                self.s8_window_radius,
                optical_s8_mask,
                sar_s8_mask,
            )
            s4_result = self._local_uot_scale(
                optical_s4,
                sar_s4,
                self.s4_projection,
                s8_result["flow"],
                self.s4_window_radius,
                optical_s4_mask,
                sar_s4_mask,
            )
            for local_result in (s8_result, s4_result):
                multiscale_flows.append(local_result["flow"])
                multiscale_confidences.append(local_result["confidence"])
                multiscale_entropies.append(local_result["entropy"])
                multiscale_covariances.append(local_result["covariance"])
                multiscale_candidate_counts.append(local_result["candidate_count"])
        elif pyramid_available:
            s8_flow = resize_flow(coarse_total, optical_s8.shape[-2:])
            s4_flow = resize_flow(s8_flow, optical_s4.shape[-2:])
            for fallback, features in ((s8_flow, optical_s8), (s4_flow, optical_s4)):
                multiscale_flows.append(fallback)
                multiscale_confidences.append(features.new_zeros((batch, 1, *features.shape[-2:])))
                multiscale_entropies.append(features.new_zeros((batch, 1, *features.shape[-2:])))
                multiscale_covariances.append(features.new_zeros((batch, 2, *features.shape[-2:])))
                multiscale_candidate_counts.append(features.new_zeros(()))
        return {
            "flow": coarse_total,
            "coarse_total_flow": coarse_total,
            "coarse_residual_flow": coarse_residual,
            "affine_flow_coarse": affine_flow,
            "confidence": confidence_map,
            "raw_logits": raw_logits,
            "graph_logits": graph_logits,
            "candidate_logits": current_logits,
            "candidate_indices": candidate_indices,
            "pre_candidate_indices": pre_candidate_indices,
            "pre_candidate_logits": pre_candidate_logits,
            "pre_rerank_logits": rerank_logits,
            "candidate_probabilities": candidate_transport,
            "transport": match["transport"],
            "transport_with_dustbin": match["transport_with_dustbin"],
            "unmatched_source_mass": match["unmatched_source_mass"],
            "unmatched_target_mass": match["unmatched_target_mass"],
            "row_transport_mass": match["row_transport_mass"],
            "column_transport_mass": match["column_transport_mass"],
            "transport_entropy": transport_entropy,
            "expected_target_coordinates": expected_target,
            "source_coordinates": graph["coordinates"],
            "cycle_coordinates": cycle_coordinates,
            "cycle_weight": cycle_weight,
            "cycle_valid_columns": valid_columns,
            "optical_matchability": optical_mass,
            "sar_matchability": sar_mass,
            "affine_matrix_normalized": matrix_normalized,
            "affine_matrix_pixel_coarse": normalized_affine_to_pixel(matrix_normalized, (height, width)),
            "affine_fit_residual": affine_fit_residual_px,
            "affine_fit_residual_mean": (
                affine_fit_residual_px * match["row_transport_mass"].float()
            ).sum(-1) / match["row_transport_mass"].float().sum(-1).clamp_min(1.0e-8),
            "affine_inlier_ratio": (affine_fit_residual_px <= 1.0).float().mean(-1),
            "effective_point_count": affine["effective_point_count"],
            "affine_spatial_coverage": affine["affine_spatial_coverage"],
            "affine_fallback_ratio": affine["affine_fallback_ratio"],
            "border_candidate_ratio": border_candidate_ratio,
            "coarse_source_valid": source_valid,
            "coarse_target_valid": target_valid,
            "graph_support": relation["graph_support"],
            "relation_cost": relation["relation_cost"],
            "relation_valid": ((pre_relation_valid.gather(-1, keep_positions) & candidate_valid)
                               if self.relation_support_mode == "neighborhood_probability" else None),
            "geometry_cost": relation["geometry_cost"],
            "compatibility": relation["compatibility"],
            "association_mass": relation["association_mass"],
            "multiscale_uot_flows": multiscale_flows,
            "multiscale_uot_confidences": multiscale_confidences,
            "multiscale_uot_entropies": multiscale_entropies,
            "multiscale_uot_covariances": multiscale_covariances,
            "multiscale_uot_candidate_counts": multiscale_candidate_counts,
            "optical_hybrid_graph_stats": optical_graph_stats,
            "sar_hybrid_graph_stats": sar_graph_stats,
        }


class S8AffineRecovery(nn.Module):
    """Recover low-confidence coarse correspondences in a local s8 window."""

    def __init__(self, channels: int, config: dict[str, Any]):
        super().__init__()
        self.enabled = bool(config.get("enabled", False))
        self.search_radius = int(config.get("search_radius", 3))
        self.window_size = int(config.get("window_size", 2 * self.search_radius + 1))
        if self.window_size != 2 * self.search_radius + 1 or self.window_size % 2 != 1:
            raise ValueError("coarse_recovery.window_size must equal 2 * search_radius + 1")
        self.temperature = float(config.get("temperature", 0.07))
        projection_dim = int(config.get("projection_dim", min(channels, 32)))
        self.query = nn.Conv2d(channels, projection_dim, 1, bias=False)
        self.key = nn.Conv2d(channels, projection_dim, 1, bias=False)
        # Near-zero initial strength preserves a migrated A0 while allowing the
        # recovery branch to acquire influence during its short warmup.
        self.recovery_strength_logit = nn.Parameter(torch.tensor(-4.0))
        yy, xx = torch.meshgrid(
            torch.arange(-self.search_radius, self.search_radius + 1, dtype=torch.float32),
            torch.arange(-self.search_radius, self.search_radius + 1, dtype=torch.float32),
            indexing="ij",
        )
        self.register_buffer(
            "offsets", torch.stack((xx, yy), dim=-1).reshape(-1, 2), persistent=False
        )

    @staticmethod
    def _coordinates(height: int, width: int, reference: torch.Tensor) -> torch.Tensor:
        yy, xx = torch.meshgrid(
            torch.arange(height, device=reference.device, dtype=reference.dtype),
            torch.arange(width, device=reference.device, dtype=reference.dtype),
            indexing="ij",
        )
        return torch.stack(
            (
                2.0 * xx / max(width - 1, 1) - 1.0,
                2.0 * yy / max(height - 1, 1) - 1.0,
            ),
            dim=-1,
        ).reshape(-1, 2)

    def forward(
        self,
        reference: torch.Tensor,
        moving: torch.Tensor,
        initial_affine: torch.Tensor,
        source_valid: torch.Tensor,
        target_valid: torch.Tensor,
        coarse_entropy: torch.Tensor,
        estimator: DifferentiableRobustAffineEstimator,
        coarse_candidate_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, _, height, width = reference.shape
        if not self.enabled:
            zeros = reference.new_zeros((batch, 1, height, width))
            return initial_affine, zeros, zeros

        query = F.normalize(self.query(reference).float(), dim=1)
        key = F.normalize(self.key(moving).float(), dim=1)
        coordinates = self._coordinates(height, width, query)
        predicted = transform_points(coordinates, initial_affine.float())
        offset_scale = query.new_tensor(
            (2.0 / max(width - 1, 1), 2.0 / max(height - 1, 1))
        )
        normalized_offsets = self.offsets.to(query) * offset_scale
        candidate_count = normalized_offsets.shape[0]
        candidate_grid = predicted.reshape(batch, 1, height, width, 2) + normalized_offsets.view(
            1, candidate_count, 1, 1, 2
        )
        in_bounds = (candidate_grid.abs() <= 1.0).all(dim=-1)
        flat_grid = candidate_grid.reshape(batch * candidate_count, height, width, 2)
        repeated_key = key.unsqueeze(1).expand(-1, candidate_count, -1, -1, -1).reshape(
            batch * candidate_count, key.shape[1], height, width
        )
        sampled_key = F.grid_sample(
            repeated_key, flat_grid, mode="bilinear", padding_mode="zeros", align_corners=True
        ).reshape(batch, candidate_count, key.shape[1], height * width)
        query_nodes = query.flatten(2)
        logits = torch.einsum("bdn,bkdn->bnk", query_nodes, sampled_key) / self.temperature

        repeated_mask = target_valid.float().unsqueeze(1).expand(
            -1, candidate_count, -1, -1, -1
        ).reshape(batch * candidate_count, 1, height, width)
        sampled_valid = F.grid_sample(
            repeated_mask, flat_grid, mode="nearest", padding_mode="zeros", align_corners=True
        ).reshape(batch, candidate_count, height * width).transpose(1, 2) > 0.5
        candidate_valid = (
            sampled_valid
            & in_bounds.reshape(batch, candidate_count, height * width).transpose(1, 2)
            & source_valid.flatten(1).unsqueeze(-1)
        )
        weights = torch.softmax(logits.masked_fill(~candidate_valid, -1.0e4), dim=-1)
        weights = weights * candidate_valid.float()
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1.0e-8)
        expected_delta = torch.einsum("bnk,kd->bnd", weights, normalized_offsets)
        correlation_peak = weights.amax(-1)

        entropy_s8 = F.interpolate(
            coarse_entropy.float(), size=(height, width), mode="bilinear", align_corners=True
        ).flatten(1)
        uncertainty = (
            entropy_s8 / math.log(max(int(coarse_candidate_count), 2))
        ).clamp(0.0, 1.0)
        uniform_peak = 1.0 / float(candidate_count)
        peak_confidence = ((correlation_peak - uniform_peak) / (1.0 - uniform_peak)).clamp(0.0, 1.0)
        strength = torch.sigmoid(self.recovery_strength_logit)
        gate = strength * uncertainty * peak_confidence * source_valid.flatten(1).float()
        corrected_target = predicted + gate.unsqueeze(-1) * expected_delta
        fit_weights = source_valid.flatten(1).float() * (0.25 + 0.75 * correlation_peak)
        recovered = estimator(
            coordinates, corrected_target, fit_weights, source_valid.flatten(1)
        )["matrix_normalized"]
        return (
            recovered,
            gate.reshape(batch, 1, height, width),
            correlation_peak.reshape(batch, 1, height, width),
        )


class LocalCrossAttention(nn.Module):
    """Memory-bounded local Transformer feature enhancement."""

    def __init__(self, channels: int, attention_dim: int, window: int, enabled: bool = True):
        super().__init__()
        if window % 2 != 1:
            raise ValueError("Local attention window must be odd")
        self.enabled = enabled
        self.window = int(window)
        self.radius = window // 2
        self.query = nn.Conv2d(channels, attention_dim, 1, bias=False)
        self.key = nn.Conv2d(channels, attention_dim, 1, bias=False)
        self.value = nn.Conv2d(channels, attention_dim, 1, bias=False)
        yy, xx = torch.meshgrid(
            torch.arange(-self.radius, self.radius + 1, dtype=torch.float32),
            torch.arange(-self.radius, self.radius + 1, dtype=torch.float32),
            indexing="ij",
        )
        self.register_buffer("offsets", torch.stack((xx, yy), dim=0).reshape(1, 2, window * window, 1), persistent=False)
        self.attention_dim = int(attention_dim)

    def forward(self, reference: torch.Tensor, moving: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, _, height, width = reference.shape
        query = self.query(reference)
        value = self.value(moving)
        if not self.enabled:
            zeros = reference.new_zeros((batch, 2, height, width))
            confidence = reference.new_ones((batch, 1, height, width))
            return value, zeros, confidence
        key = self.key(moving)
        locations = height * width
        candidates = self.window * self.window
        query = query.flatten(2).unsqueeze(2)
        key = F.unfold(key, self.window, padding=self.radius).reshape(batch, self.attention_dim, candidates, locations)
        logits = (query * key).sum(dim=1) / math.sqrt(self.attention_dim)
        weights = torch.softmax(logits.float(), dim=1).to(logits.dtype)
        value = F.unfold(value, self.window, padding=self.radius).reshape(batch, self.attention_dim, candidates, locations)
        context = (value * weights.unsqueeze(1)).sum(dim=2).reshape(batch, self.attention_dim, height, width)
        expected_offset = (weights.unsqueeze(1) * self.offsets.to(weights.dtype)).sum(dim=2).reshape(batch, 2, height, width)
        entropy = -(weights.clamp_min(1.0e-8) * weights.clamp_min(1.0e-8).log()).sum(dim=1)
        confidence = (1.0 - entropy / math.log(candidates)).clamp(0.0, 1.0).reshape(batch, 1, height, width)
        return context, expected_offset, confidence


class WarpRefiner(nn.Module):
    def __init__(
        self,
        channels: int,
        attention_dim: int,
        window: int,
        residual_radius: float,
        tfe_enabled: bool,
        dropout: float,
    ):
        super().__init__()
        self.tfe = LocalCrossAttention(channels, attention_dim, window, enabled=tfe_enabled)
        hidden = max(channels, 64)
        self.body = nn.Sequential(
            ConvNormAct(2 * channels + attention_dim + 8, hidden),
            ResidualBlock(hidden),
            nn.Dropout2d(float(dropout)),
            ResidualBlock(hidden),
        )
        self.residual_head = nn.Conv2d(hidden, 2, 3, padding=1)
        self.gate_head = nn.Conv2d(hidden, 1, 3, padding=1)
        self.residual_radius = float(residual_radius)

    def forward(
        self,
        reference: torch.Tensor,
        moving_warped: torch.Tensor,
        flow: torch.Tensor,
        local_residual: torch.Tensor,
        prior_confidence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context, offset_hint, attention_confidence = self.tfe(reference, moving_warped)
        scale = float(max(flow.shape[-2:]))
        inputs = torch.cat(
            (
                reference,
                moving_warped,
                context,
                flow / scale,
                local_residual / scale,
                prior_confidence,
                offset_hint,
                attention_confidence,
            ),
            dim=1,
        )
        hidden = self.body(inputs)
        gate = torch.sigmoid(self.gate_head(hidden)) * prior_confidence.clamp(0.05, 1.0)
        learned = torch.tanh(self.residual_head(hidden)) * self.residual_radius
        delta = gate * (offset_hint + learned)
        return delta, gate, attention_confidence


class DINOTopoReg(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        model_cfg = config["model"]
        self.tsfr_enabled = str(model_cfg.get("fine", {}).get("mode", "legacy")) == "tsfr"
        self.sgu_enabled = str(model_cfg.get("fine", {}).get("mode", "legacy")) == "sgu_flow"
        self.motion_mode = "sgu_flow" if self.sgu_enabled else ("similarity_refit" if self.tsfr_enabled else "affine_residual")
        geometry_cfg = model_cfg.get("geometry", {})
        self.affine_enabled = bool(geometry_cfg.get("affine_enabled", True))
        self.residual_enabled = bool(geometry_cfg.get("residual_enabled", True))
        self.dino = FrozenDINOv3(model_cfg["dino"])
        estimator_cfg = dict(model_cfg.get("affine", {}))
        estimator_cfg.update(geometry_cfg)
        fine_cfg = model_cfg["fine"]
        channels = [int(value) for value in fine_cfg["channels"]]
        # CSLE is evaluated ONCE per modality and reused by all matching levels.
        local_cfg = model_cfg.get('csle', {})
        self.csle = CrossModalStructuralLocalEncoder(
            c8=channels[1], c4=channels[2], c2=channels[3],
            stem_dim=int(local_cfg.get('stem_dim', 48)),
        )
        fused_dim = int(model_cfg.get('fusion', {}).get('dim', model_cfg['coarse'].get('projection_dim', 256)))
        if fused_dim != int(model_cfg['coarse'].get('projection_dim', 256)):
            raise ValueError('fusion.dim must equal coarse.projection_dim for learned graph structures')
        self.s2gf = SemanticStructureFusion(
            self.dino.output_dim, channels[1], channels[2], fused_dim
        )
        self.coarse = GraphAssistedCoarseMatcherV2(
            fused_dim, model_cfg["coarse"], estimator_cfg, channels
        )
        if self.sgu_enabled:
            self.sgu = SGUFlowRefiner(channels[1], channels[2], channels[3], fine_cfg['sgu_flow'])
            self.coarse_recovery = None
            content_cfg = config.get('data', {}).get('content_mask', {})
            self.content_mask_enabled = bool(content_cfg.get('enabled', False))
            self.patch_valid_ratio = float(content_cfg.get('patch_valid_ratio', .60))
            return
        if self.tsfr_enabled:
            self.tsfr = TwoScaleGeometricRefiner(channels[2], channels[3], fine_cfg.get("tsfr", {}))
            self.coarse_recovery = None
            content_cfg = config.get("data", {}).get("content_mask", {})
            self.content_mask_enabled = bool(content_cfg.get("enabled", False))
            self.patch_valid_ratio = float(content_cfg.get("patch_valid_ratio", 0.60))
            return
        attention_dims = [int(value) for value in fine_cfg["attention_dims"]]
        windows = [int(value) for value in fine_cfg["windows"]]
        radii = [float(value) for value in fine_cfg["residual_radii"]]
        if not (len(channels) == len(attention_dims) == len(windows) == len(radii) == 5):
            raise ValueError("Fine configuration must define exactly five refinement stages")
        # Legacy five-level decoder, if used, is also fed by CSLE instead of
        # constructing additional independent modality CNN pyramids.
        self.legacy_coarse_adapter = nn.Conv2d(fused_dim, channels[0], 1)
        self.legacy_full_adapter = nn.Conv2d(channels[3], channels[4], 3, padding=1)
        self.refiners = nn.ModuleList(
            [
                WarpRefiner(
                    channel,
                    dim,
                    window,
                    radius,
                    bool(fine_cfg.get("tfe_enabled", True)),
                    float(fine_cfg.get("dropout", 0.0)),
                )
                for channel, dim, window, radius in zip(channels, attention_dims, windows, radii)
            ]
        )
        self.coarse_recovery = S8AffineRecovery(
            channels[1], model_cfg.get("coarse_recovery", {"enabled": False})
        )
        content_cfg = config.get("data", {}).get("content_mask", {})
        self.content_mask_enabled = bool(content_cfg.get("enabled", False))
        self.patch_valid_ratio = float(content_cfg.get("patch_valid_ratio", 0.60))

    def _downsample_content_mask(
        self, mask: torch.Tensor | None, size: tuple[int, int], reference: torch.Tensor
    ) -> torch.Tensor:
        if mask is None or not self.content_mask_enabled:
            return torch.ones(
                (reference.shape[0], 1, *size), dtype=torch.bool, device=reference.device
            )
        return F.adaptive_avg_pool2d(mask.float(), size) >= self.patch_valid_ratio

    def forward(
        self,
        optical: torch.Tensor,
        sar: torch.Tensor,
        dino_optical: torch.Tensor,
        dino_sar: torch.Tensor,
        optical_content_mask: torch.Tensor | None = None,
        sar_content_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        combined = torch.cat((dino_optical, dino_sar), dim=0)
        dino_features = self.dino(combined)
        dino_optical_features, dino_sar_features = dino_features.chunk(2, dim=0)
        coarse_size = dino_optical_features.shape[-2:]
        optical_s16_mask = self._downsample_content_mask(
            optical_content_mask, coarse_size, dino_optical_features
        )
        sar_s16_mask = self._downsample_content_mask(
            sar_content_mask, coarse_size, dino_sar_features
        )
        # Both SALA scales use exactly the same CSLE tensors that later drive
        # the 1/8 and 1/4 UOT refinement stages.
        optical_local, sar_local = self.csle(optical, sar)
        optical_mask8 = self._downsample_content_mask(optical_content_mask, optical_local['s8'].shape[-2:], optical_local['s8'])
        sar_mask8 = self._downsample_content_mask(sar_content_mask, sar_local['s8'].shape[-2:], sar_local['s8'])
        optical_mask4 = self._downsample_content_mask(optical_content_mask, optical_local['s4'].shape[-2:], optical_local['s4'])
        sar_mask4 = self._downsample_content_mask(sar_content_mask, sar_local['s4'].shape[-2:], sar_local['s4'])
        optical_fusion = self.s2gf(dino_optical_features, optical_local, optical_mask4, optical_mask8)
        sar_fusion = self.s2gf(dino_sar_features, sar_local, sar_mask4, sar_mask8)
        fused_optical, fused_sar = optical_fusion['fused'], sar_fusion['fused']
        common_coarse_args = dict(
            optical_structure=optical_fusion['structure'],
            sar_structure=sar_fusion['structure'],
        )
        if self.sgu_enabled:
            opt8, opt4, opt2 = (optical_local[k] for k in ('s8','s4','s2'))
            sar8, sar4, sar2 = (sar_local[k] for k in ('s8','s4','s2'))
            coarse = self.coarse(fused_optical, fused_sar,
                                 optical_s16_mask, sar_s16_mask,
                                 **common_coarse_args)
            masks = [self._downsample_content_mask(a, fe.shape[-2:], fe)
                     for a, fe in ((optical_content_mask, opt8), (sar_content_mask, sar8),
                                    (optical_content_mask, opt4), (sar_content_mask, sar4),
                                    (optical_content_mask, opt2), (sar_content_mask, sar2))]
            fine = self.sgu(opt8, sar8, opt4, sar4, opt2, sar2,
                            coarse['affine_matrix_normalized'], coarse['confidence'],
                            *masks, optical.shape[-2:],
                            full_source_mask=optical_content_mask if self.content_mask_enabled else None,
                            full_target_mask=sar_content_mask if self.content_mask_enabled else None)
            from .geometry import pixel_affine_to_normalized
            fine['affine_matrix_normalized'] = pixel_affine_to_normalized(fine['affine_matrix_pixel'], optical.shape[-2:])
            fine.update(motion_mode=self.motion_mode,
                        coarse_flow=coarse['coarse_total_flow'],
                        coarse_total_flow=coarse['coarse_total_flow'],
                        coarse_residual_flow=coarse['coarse_residual_flow'],
                        coarse_confidence=coarse['confidence'],
                        affine_matrix=fine['affine_matrix_pixel'],
                        affine_matrix_initial_normalized=coarse['affine_matrix_normalized'],
                        affine_flow_coarse=coarse['affine_flow_coarse'],
                        affine_flows=[resize_flow(fine['initial_affine_flow'], f.shape[-2:]) for f in fine['flows']],
                        residual_flows=fine['local_residuals'],
                        gates=[r['local_gate'] for r in fine['stage_results']],
                        attention_confidences=[r['confidence'] for r in fine['stage_results']],
                        s8_recovery_gate=torch.zeros_like(coarse['confidence']),
                        s8_recovery_peak=torch.zeros_like(coarse['confidence']),
                        affine_fit_residual=coarse['affine_fit_residual'],
                        affine_fit_residual_mean=coarse['affine_fit_residual_mean'],
                        effective_point_count=coarse['effective_point_count'],
                        affine_inlier_ratio=coarse['affine_inlier_ratio'],
                        affine_spatial_coverage=coarse['affine_spatial_coverage'],
                        affine_effective_points=coarse['effective_point_count'],
                        affine_fallback_ratio=coarse['affine_fallback_ratio'])
            for key in ('raw_logits', 'graph_logits', 'candidate_indices', 'pre_candidate_indices',
                        'pre_candidate_logits', 'pre_rerank_logits',
                        'cycle_coordinates', 'source_coordinates', 'transport', 'transport_with_dustbin',
                        'candidate_logits', 'candidate_probabilities', 'expected_target_coordinates',
                        'row_transport_mass', 'column_transport_mass', 'unmatched_source_mass',
                        'unmatched_target_mass', 'transport_entropy', 'cycle_weight', 'cycle_valid_columns',
                        'optical_matchability', 'sar_matchability', 'graph_support', 'relation_cost', 'relation_valid',
                        'geometry_cost', 'compatibility', 'association_mass', 'coarse_source_valid',
                        'coarse_target_valid', 'multiscale_uot_flows', 'multiscale_uot_confidences',
                        'multiscale_uot_entropies', 'multiscale_uot_covariances',
                        'optical_hybrid_graph_stats', 'sar_hybrid_graph_stats', 'border_candidate_ratio'):
                fine[key] = coarse[key]
            fine['structural_reliability_optical'] = optical_fusion['reliability']
            fine['structural_reliability_sar'] = sar_fusion['reliability']
            return fine
        if self.tsfr_enabled:
            opt4, opt2 = optical_local['s4'], optical_local['s2']
            sar4, sar2 = sar_local['s4'], sar_local['s2']
            coarse = self.coarse(
                fused_optical, fused_sar,
                optical_s16_mask, sar_s16_mask,
                **common_coarse_args,
            )
            # The only UOT in this architecture occurs in the coarse matcher.
            opt4_mask = self._downsample_content_mask(optical_content_mask, opt4.shape[-2:], opt4)
            sar4_mask = self._downsample_content_mask(sar_content_mask, sar4.shape[-2:], sar4)
            opt2_mask = self._downsample_content_mask(optical_content_mask, opt2.shape[-2:], opt2)
            sar2_mask = self._downsample_content_mask(sar_content_mask, sar2.shape[-2:], sar2)
            fine = self.tsfr(
                opt4, sar4, opt2, sar2, coarse["affine_matrix_normalized"],
                optical.shape[-2:], opt4_mask, sar4_mask, opt2_mask, sar2_mask,
                coarse_confidence=coarse["confidence"],
            )
            full_size = optical.shape[-2:]
            fine.update({
                "motion_mode": self.motion_mode,
                "coarse_flow": coarse["coarse_total_flow"],
                "coarse_total_flow": coarse["coarse_total_flow"],
                "coarse_residual_flow": coarse["coarse_residual_flow"],
                "coarse_confidence": coarse["confidence"],
                "affine_matrix": fine["affine_matrix_pixel"],
                "affine_matrix_initial": normalized_affine_to_pixel(
                    coarse["affine_matrix_normalized"], full_size),
                "affine_matrix_initial_normalized": coarse["affine_matrix_normalized"],
                "affine_flow_coarse": coarse["affine_flow_coarse"],
                "affine_flows": [affine_to_flow(coarse["affine_matrix_normalized"], opt4.shape[-2:], normalized=True),
                                 affine_to_flow(fine["affine_matrix_normalized"], opt2.shape[-2:], normalized=True)],
                "local_residuals": fine["residual_flows"],
                "gates": [],
                "attention_confidences": [],
                "s8_recovery_gate": torch.zeros_like(coarse["confidence"]),
                "s8_recovery_peak": torch.zeros_like(coarse["confidence"]),
                "border_candidate_ratio": coarse["border_candidate_ratio"],
                "affine_fit_residual": coarse["affine_fit_residual"],
                "affine_fit_residual_mean": coarse["affine_fit_residual_mean"],
                "effective_point_count": coarse["effective_point_count"],
                "affine_inlier_ratio": coarse["affine_inlier_ratio"],
                "affine_spatial_coverage": coarse["affine_spatial_coverage"],
                "affine_effective_points": coarse["effective_point_count"],
                "affine_fallback_ratio": coarse["affine_fallback_ratio"],
            })
            for key in (
                "raw_logits", "graph_logits", "candidate_indices", "pre_candidate_indices",
                "pre_candidate_logits", "pre_rerank_logits",
                "cycle_coordinates", "source_coordinates", "transport", "transport_with_dustbin",
                "candidate_logits", "candidate_probabilities", "expected_target_coordinates",
                "row_transport_mass", "column_transport_mass", "unmatched_source_mass",
                "unmatched_target_mass", "transport_entropy", "cycle_weight", "cycle_valid_columns",
                "optical_matchability", "sar_matchability", "graph_support", "relation_cost", "relation_valid",
                "geometry_cost", "compatibility", "association_mass", "coarse_source_valid",
                "coarse_target_valid", "multiscale_uot_flows", "multiscale_uot_confidences",
                "multiscale_uot_entropies", "multiscale_uot_covariances",
                "optical_hybrid_graph_stats", "sar_hybrid_graph_stats",
            ):
                fine[key] = coarse[key]
            fine['structural_reliability_optical'] = optical_fusion['reliability']
            fine['structural_reliability_sar'] = sar_fusion['reliability']
            return fine
        optical_features = [
            self.legacy_coarse_adapter(fused_optical),
            optical_local['s8'], optical_local['s4'], optical_local['s2'],
            self.legacy_full_adapter(F.interpolate(optical_local['s2'], size=optical.shape[-2:], mode='bilinear', align_corners=False))
        ]
        sar_features = [
            self.legacy_coarse_adapter(fused_sar),
            sar_local['s8'], sar_local['s4'], sar_local['s2'],
            self.legacy_full_adapter(F.interpolate(sar_local['s2'], size=sar.shape[-2:], mode='bilinear', align_corners=False))
        ]
        s8_size = optical_features[1].shape[-2:]
        optical_s8_mask = self._downsample_content_mask(
            optical_content_mask, s8_size, optical_features[1]
        )
        sar_s8_mask = self._downsample_content_mask(
            sar_content_mask, s8_size, sar_features[1]
        )
        s4_size = optical_features[2].shape[-2:]
        optical_s4_mask = self._downsample_content_mask(
            optical_content_mask, s4_size, optical_features[2]
        )
        sar_s4_mask = self._downsample_content_mask(
            sar_content_mask, s4_size, sar_features[2]
        )
        coarse = self.coarse(
            fused_optical,
            fused_sar,
            optical_s16_mask,
            sar_s16_mask,
            **common_coarse_args,
            optical_s8=optical_features[1],
            sar_s8=sar_features[1],
            optical_s4=optical_features[2],
            sar_s4=sar_features[2],
            optical_s8_mask=optical_s8_mask,
            sar_s8_mask=sar_s8_mask,
            optical_s4_mask=optical_s4_mask,
            sar_s4_mask=sar_s4_mask,
        )
        affine_matrix_initial = coarse["affine_matrix_normalized"]
        coarse_entropy = coarse["transport_entropy"].reshape(
            optical.shape[0], 1, *coarse_size
        )
        affine_matrix_normalized, recovery_gate, recovery_peak = self.coarse_recovery(
            optical_features[1],
            sar_features[1],
            affine_matrix_initial,
            optical_s8_mask,
            sar_s8_mask,
            coarse_entropy,
            self.coarse.estimator,
            self.coarse.graph_topk,
        )
        local_residual = coarse["coarse_residual_flow"]
        flow = coarse["coarse_total_flow"]
        confidence = coarse["confidence"]
        flows: list[torch.Tensor] = []
        affine_flows: list[torch.Tensor] = []
        residual_flows: list[torch.Tensor] = []
        gates: list[torch.Tensor] = []
        attention_confidences: list[torch.Tensor] = []
        for stage_index, (reference, moving, refiner) in enumerate(
            zip(optical_features, sar_features, self.refiners)
        ):
            target_size = reference.shape[-2:]
            affine_at_scale = affine_to_flow(affine_matrix_normalized, target_size, normalized=True)
            local_residual = resize_flow(local_residual, target_size)
            flow = affine_at_scale + local_residual
            confidence = F.interpolate(confidence, size=target_size, mode="bilinear", align_corners=True)
            if (
                self.coarse.multiscale_uot_enabled
                and stage_index in (1, 2)
                and len(coarse["multiscale_uot_flows"]) > stage_index
            ):
                prior_flow = resize_flow(
                    coarse["multiscale_uot_flows"][stage_index], target_size
                ).to(flow)
                prior_confidence = F.interpolate(
                    coarse["multiscale_uot_confidences"][stage_index].float(),
                    size=target_size,
                    mode="bilinear",
                    align_corners=True,
                ).to(flow)
                blend_weight = prior_confidence.clamp(0.0, 1.0) * self.coarse.prior_blend_max
                flow = (1.0 - blend_weight) * flow + blend_weight * prior_flow
                local_residual = flow - affine_at_scale
                confidence = torch.maximum(confidence, prior_confidence)
            moving_warped = flow_warp(moving, flow)
            delta, gate, attention_confidence = refiner(
                reference, moving_warped, flow, local_residual, confidence
            )
            if not self.residual_enabled:
                delta = torch.zeros_like(delta)
            local_residual = local_residual + delta
            flow = affine_at_scale + local_residual
            confidence = (0.5 * confidence + 0.5 * gate * attention_confidence).clamp(0.0, 1.0)
            flows.append(flow)
            affine_flows.append(affine_at_scale)
            residual_flows.append(local_residual)
            gates.append(gate)
            attention_confidences.append(attention_confidence)
        full_size = optical.shape[-2:]
        affine_matrix_initial_pixel = normalized_affine_to_pixel(
            affine_matrix_initial, full_size
        )
        affine_matrix_pixel = normalized_affine_to_pixel(
            affine_matrix_normalized, full_size
        )
        initial_affine_flow = affine_to_flow(
            affine_matrix_initial, full_size, normalized=True
        )
        result = {
            "flow": flows[-1],
            "motion_mode": self.motion_mode,
            "flows": flows,
            "coarse_flow": coarse["coarse_total_flow"],
            "coarse_total_flow": coarse["coarse_total_flow"],
            "coarse_residual_flow": coarse["coarse_residual_flow"],
            "coarse_confidence": coarse["confidence"],
            "affine_flow": affine_flows[-1],
            "initial_affine_flow": initial_affine_flow,
            "affine_flow_coarse": coarse["affine_flow_coarse"],
            "affine_flows": affine_flows,
            "affine_matrix": affine_matrix_pixel,
            "affine_matrix_initial": affine_matrix_initial_pixel,
            "affine_matrix_initial_normalized": affine_matrix_initial,
            "affine_matrix_pixel": affine_matrix_pixel,
            "affine_matrix_normalized": affine_matrix_normalized,
            "affine_fit_residual": coarse["affine_fit_residual"],
            "affine_fit_residual_mean": coarse["affine_fit_residual_mean"],
            "effective_point_count": coarse["effective_point_count"],
            "affine_inlier_ratio": coarse["affine_inlier_ratio"],
            "affine_spatial_coverage": coarse["affine_spatial_coverage"],
            "affine_effective_points": coarse["effective_point_count"],
            "affine_fallback_ratio": coarse["affine_fallback_ratio"],
            "s8_recovery_gate": recovery_gate,
            "s8_recovery_peak": recovery_peak,
            "residual_flow": residual_flows[-1],
            "residual_flows": residual_flows,
            "local_residuals": residual_flows,
            "raw_logits": coarse["raw_logits"],
            "graph_logits": coarse["graph_logits"],
            "candidate_indices": coarse["candidate_indices"],
            "pre_candidate_indices": coarse["pre_candidate_indices"],
            "border_candidate_ratio": coarse["border_candidate_ratio"],
            "cycle_coordinates": coarse["cycle_coordinates"],
            "source_coordinates": coarse["source_coordinates"],
            "gates": gates,
            "attention_confidences": attention_confidences,
            "multiscale_uot_flows": coarse["multiscale_uot_flows"],
            "multiscale_uot_confidences": coarse["multiscale_uot_confidences"],
            "multiscale_uot_entropies": coarse["multiscale_uot_entropies"],
            "multiscale_uot_covariances": coarse["multiscale_uot_covariances"],
            "optical_hybrid_graph_stats": coarse["optical_hybrid_graph_stats"],
            "sar_hybrid_graph_stats": coarse["sar_hybrid_graph_stats"],
        }
        for key in (
            "transport",
            "pre_candidate_logits", "pre_rerank_logits",
            "transport_with_dustbin",
            "candidate_logits",
            "candidate_probabilities",
            "expected_target_coordinates",
            "row_transport_mass",
            "column_transport_mass",
            "unmatched_source_mass",
            "unmatched_target_mass",
            "transport_entropy",
            "cycle_weight",
            "cycle_valid_columns",
            "optical_matchability",
            "sar_matchability",
            "graph_support",
            "relation_cost", "relation_valid",
            "geometry_cost",
            "compatibility",
            "association_mass",
        ):
            result[key] = coarse[key]
        return result

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        if not self.dino.frozen:
            return self.state_dict()
        return {key: value for key, value in self.state_dict().items() if not key.startswith("dino.backbone.")}

    def load_trainable_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        incompatible = self.load_state_dict(state, strict=False)
        ignored_prefix = "dino.backbone." if self.dino.frozen else ""
        unexpected = [key for key in incompatible.unexpected_keys if not ignored_prefix or not key.startswith(ignored_prefix)]
        missing = [key for key in incompatible.missing_keys if not ignored_prefix or not key.startswith(ignored_prefix)]
        if unexpected or missing:
            raise RuntimeError(
                "当前 checkpoint 与本模型结构不兼容；请使用 train.py --warm-start 导入兼容参数并重新训练。 "
                f"missing={missing}, unexpected={unexpected}"
            )
