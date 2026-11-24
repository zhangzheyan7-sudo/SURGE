from __future__ import annotations

from typing import Any
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import affine_to_flow, transform_points
from .utils import resize_flow, resize_mask


def masked_mean(values: torch.Tensor, mask: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    mask = mask.to(values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(1)
    total_weight = mask if weights is None else mask * weights.to(values.dtype)
    while total_weight.ndim < values.ndim:
        total_weight = total_weight.unsqueeze(1)
    return (values * total_weight).sum() / total_weight.expand_as(values).sum().clamp_min(1.0)


class RegistrationLoss(nn.Module):
    """Geometry-aware supervision for total, affine, residual and sparse matching paths."""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.paper_objective = config.get("objective_mode") == "paper_four_terms"
        self.relation_eta = float(config.get("relation_eta", 0.1))
        if not math.isfinite(self.relation_eta) or self.relation_eta < 0:
            raise ValueError("relation_eta must be finite and nonnegative")
        self.weights = {
            "initial_affine": float(config.get("w_initial_affine", 0.0)),
            "total_flow": float(config.get("w_total_flow", config.get("flow_weight", 1.0))),
            "affine_corner": float(config.get("w_affine_corner", 0.10)),
            "affine_flow": float(config.get("w_affine_flow", 0.0)),
            "residual": float(config.get("w_residual", 0.0)),
            "match": float(config.get("w_match", config.get("match_weight", 0.05))),
            "candidate": float(config.get("w_candidate", 0.01)),
            "relation": float(config.get("w_relation", 0.005)),
            "cycle": float(config.get("w_cycle", 0.0)),
            "residual_smooth": float(config.get("w_residual_smooth", config.get("smooth_weight", 0.005))),
            "residual_magnitude": float(config.get("w_residual_magnitude", 0.0)),
            "residual_consistency": float(config.get("w_residual_consistency", 0.0)),
            "fine_points": float(config.get("w_fine_points", 0.0)),
            "local_uot": float(config.get("w_local_uot", 0.1)),
            "geometry_consistency": float(config.get("w_geometry_consistency", 0.02)),
            "final_full": float(config.get("w_final_full", 0.0)),
        }
        if not math.isfinite(self.weights["initial_affine"]) or self.weights["initial_affine"] < 0:
            raise ValueError("w_initial_affine must be finite and non-negative")
        self.scale_weights = [float(value) for value in config.get("scale_weights", [0.05, 0.1, 0.15, 0.2, 0.5])]
        self.epsilon = float(config.get("charbonnier_epsilon", 1.0e-3))
        self.include_edge_midpoints = bool(config.get("affine_edge_midpoints", True))
        self.candidate_rank_margin = float(config.get("candidate_rank_margin", 0.1))
        self.candidate_rank_temperature = float(config.get("candidate_rank_temperature", 1.0))
        if self.candidate_rank_margin < 0 or self.candidate_rank_temperature <= 0:
            raise ValueError("Candidate ranking margin must be nonnegative and temperature positive")

    def _endpoint_loss(self, prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        error = prediction - target
        endpoint = torch.sqrt((error * error).sum(dim=1, keepdim=True) + self.epsilon**2)
        return masked_mean(endpoint, valid)

    def _flow_loss(self, predictions: list[torch.Tensor], target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        if len(predictions) != len(self.scale_weights):
            raise ValueError(f"Expected {len(self.scale_weights)} flow scales, got {len(predictions)}")
        total = target.new_zeros(())
        for prediction, weight in zip(predictions, self.scale_weights):
            total = total + weight * self._endpoint_loss(
                prediction,
                resize_flow(target, prediction.shape[-2:]),
                resize_mask(valid, prediction.shape[-2:]),
            )
        return total if self.paper_objective else total / max(sum(self.scale_weights), 1.0e-8)

    @staticmethod
    def _coarse_soft_targets(
        flow: torch.Tensor, valid: torch.Tensor, height: int, width: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flow_scaled = resize_flow(flow, (height, width))
        valid_scaled = resize_mask(valid, (height, width)).squeeze(1)
        yy, xx = torch.meshgrid(
            torch.arange(height, device=flow.device, dtype=flow.dtype),
            torch.arange(width, device=flow.device, dtype=flow.dtype),
            indexing="ij",
        )
        target_x = xx.unsqueeze(0) + flow_scaled[:, 0]
        target_y = yy.unsqueeze(0) + flow_scaled[:, 1]
        inside = (target_x >= 0) & (target_x <= width - 1) & (target_y >= 0) & (target_y <= height - 1)
        safe_x = target_x.clamp(0, width - 1)
        safe_y = target_y.clamp(0, height - 1)
        x0 = safe_x.floor()
        y0 = safe_y.floor()
        x1 = (x0 + 1).clamp(max=width - 1)
        y1 = (y0 + 1).clamp(max=height - 1)
        wx, wy = safe_x - x0, safe_y - y0
        indices = torch.stack((y0 * width + x0, y0 * width + x1, y1 * width + x0, y1 * width + x1), dim=-1).long()
        weights = torch.stack(((1 - wx) * (1 - wy), wx * (1 - wy), (1 - wx) * wy, wx * wy), dim=-1)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1.0e-8)
        batch = flow.shape[0]
        return indices.reshape(batch, -1, 4), weights.reshape(batch, -1, 4), (valid_scaled & inside).reshape(batch, -1)

    def _matching_losses(
        self, outputs: dict[str, Any], flow: torch.Tensor, valid: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        raw_logits = outputs["raw_logits"].float()
        height, width = outputs["coarse_total_flow"].shape[-2:]
        indices, soft_weights, target_valid = self._coarse_soft_targets(flow, valid, height, width)
        source_content = outputs.get("coarse_source_valid")
        target_content = outputs.get("coarse_target_valid")
        if source_content is not None:
            target_valid &= source_content.bool()
        if target_content is not None:
            neighbour_content = target_content.gather(
                1, indices.reshape(indices.shape[0], -1)
            ).reshape_as(indices)
            soft_weights = soft_weights * neighbour_content.to(soft_weights.dtype)
            content_mass = soft_weights.sum(-1, keepdim=True)
            target_valid &= content_mass.squeeze(-1) > 0
            soft_weights = soft_weights / content_mass.clamp_min(1.0e-8)
        # Padding nodes are excluded from both candidate search and the CE
        # denominator. Use a finite sentinel and replace all-invalid rows before
        # softmax, so zero-weight GT neighbours can never multiply -inf/NaN.
        if target_content is not None:
            target_content = target_content.bool()
            masked_logits = raw_logits.masked_fill(
                ~target_content.unsqueeze(1), torch.finfo(raw_logits.dtype).min,
            )
            masked_logits = torch.where(
                target_content.any(-1).view(-1, 1, 1), masked_logits, torch.zeros_like(masked_logits),
            )
        else:
            masked_logits = raw_logits
        log_probabilities = F.log_softmax(masked_logits, dim=-1)
        selected = log_probabilities.gather(-1, indices)
        soft_ce = -(selected * soft_weights).sum(-1)
        match_loss = masked_mean(soft_ce, target_valid)

        def candidate_coverage(candidates: torch.Tensor) -> torch.Tensor:
            hits = candidates.unsqueeze(-1) == indices.unsqueeze(-2)
            neighbour_present = hits.any(dim=-2).to(soft_weights.dtype)
            return (neighbour_present * soft_weights).sum(-1).clamp(0.0, 1.0)

        coverage = candidate_coverage(outputs["candidate_indices"])
        pre_coverage = candidate_coverage(outputs.get("pre_candidate_indices", outputs["candidate_indices"]))
        # The discrete coverage above remains an honest diagnostic. Its old
        # -log(coverage) penalty had no parameter gradient. Instead supervise
        # scores against detached TopK boundaries before and after graph rerank.
        candidate_loss = self._candidate_ranking_loss(
            outputs, raw_logits, indices, soft_weights, target_valid, target_content,
        )
        recall = masked_mean((coverage > 0).float(), target_valid)
        pre_recall = masked_mean((pre_coverage > 0).float(), target_valid)
        pruning_drop = (pre_recall - recall).clamp_min(0.0)
        coverage_mean = masked_mean(coverage, target_valid)
        outputs["candidate_coverage"] = coverage
        outputs["candidate_recall"] = (coverage > 0).to(coverage.dtype)
        border_ratio = torch.as_tensor(
            outputs.get("border_candidate_ratio", 0.0), device=raw_logits.device
        ).float().mean()
        return (
            match_loss,
            candidate_loss,
            recall,
            coverage_mean,
            pre_recall,
            pruning_drop,
            border_ratio,
        )

    def _candidate_ranking_loss(
        self, outputs: dict[str, Any], raw_logits: torch.Tensor,
        target_indices: torch.Tensor, target_weights: torch.Tensor,
        row_valid: torch.Tensor, target_content: torch.Tensor | None,
    ) -> torch.Tensor:
        """Train GT neighbours to survive raw preselection and graph pruning.

        Integer memberships are used only to gather scores. Each GT score is
        differentiable, including GT absent from preselection. Detached cutoffs
        prevent the shortcut of lowering retained scores instead of improving GT.
        This adds no parameters and keeps old checkpoints format-compatible.
        """
        pre_indices = outputs.get("pre_candidate_indices", outputs["candidate_indices"])
        final_indices = outputs["candidate_indices"]
        gt_raw = raw_logits.gather(-1, target_indices)

        def content_valid(candidates: torch.Tensor) -> torch.Tensor:
            if target_content is None:
                return torch.ones_like(candidates, dtype=torch.bool)
            return target_content.gather(1, candidates.flatten(1)).reshape_as(candidates)

        def cutoff(scores: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            has_candidate = valid.any(-1)
            boundary = scores.masked_fill(~valid, float("inf")).amin(-1)
            boundary = torch.where(has_candidate, boundary, torch.zeros_like(boundary)).detach()
            return boundary, has_candidate

        def rank_cost(scores: torch.Tensor, boundary: torch.Tensor) -> torch.Tensor:
            difference = boundary.unsqueeze(-1) + self.candidate_rank_margin - scores
            return self.candidate_rank_temperature * F.softplus(
                difference / self.candidate_rank_temperature,
            )

        pre_scores = outputs.get("pre_candidate_logits")
        if pre_scores is None:
            pre_scores = raw_logits.gather(-1, pre_indices)
        pre_boundary, pre_has_candidate = cutoff(pre_scores.float(), content_valid(pre_indices))
        pre_row_loss = (rank_cost(gt_raw, pre_boundary) * target_weights).sum(-1)
        pre_loss = masked_mean(pre_row_loss, row_valid & pre_has_candidate)

        rerank_scores = outputs.get("pre_rerank_logits")
        if rerank_scores is None:
            # Older/legacy model outputs still receive a differentiable raw-score
            # loss against the final retained set, rather than a constant penalty.
            final_boundary, final_has_candidate = cutoff(
                raw_logits.gather(-1, final_indices), content_valid(final_indices),
            )
            final_row_loss = (rank_cost(gt_raw, final_boundary) * target_weights).sum(-1)
            final_loss = masked_mean(final_row_loss, row_valid & final_has_candidate)
        else:
            rerank_scores = rerank_scores.float()
            gt_hits = pre_indices.unsqueeze(-1) == target_indices.unsqueeze(-2)
            gt_present = gt_hits.any(-2)
            gt_positions = gt_hits.to(torch.int64).argmax(-2)
            gt_rerank = rerank_scores.gather(-1, gt_positions)
            kept_hits = pre_indices.unsqueeze(-1) == final_indices.unsqueeze(-2)
            kept_positions = kept_hits.to(torch.int64).argmax(-2)
            kept_scores = rerank_scores.gather(-1, kept_positions)
            final_boundary, final_has_candidate = cutoff(
                kept_scores, content_valid(final_indices) & kept_hits.any(-2),
            )
            present_weight = target_weights * gt_present.to(target_weights.dtype)
            final_row_loss = (rank_cost(gt_rerank, final_boundary) * present_weight).sum(-1)
            final_loss = masked_mean(
                final_row_loss, row_valid & final_has_candidate & (present_weight.sum(-1) > 0),
            )
        return 0.5 * (pre_loss + final_loss)

    def _cycle_loss(self, outputs: dict[str, Any], valid: torch.Tensor) -> torch.Tensor:
        height, width = outputs["coarse_total_flow"].shape[-2:]
        valid_scaled = resize_mask(valid, (height, width)).flatten(1)
        source = outputs["source_coordinates"].to(outputs["cycle_coordinates"]).unsqueeze(0)
        error = torch.linalg.vector_norm(outputs["cycle_coordinates"] - source, dim=-1)
        weights = torch.nan_to_num(outputs["cycle_weight"].float())
        return masked_mean(error, valid_scaled, weights)

    def _affine_corner_loss(
        self, predicted: torch.Tensor, target: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        points = [
            (0.0, 0.0),
            (width - 1.0, 0.0),
            (0.0, height - 1.0),
            (width - 1.0, height - 1.0),
            ((width - 1.0) / 2.0, (height - 1.0) / 2.0),
        ]
        if self.include_edge_midpoints:
            points.extend((
                ((width - 1.0) / 2.0, 0.0),
                ((width - 1.0) / 2.0, height - 1.0),
                (0.0, (height - 1.0) / 2.0),
                (width - 1.0, (height - 1.0) / 2.0),
            ))
        coordinates = predicted.new_tensor(points)
        error = transform_points(coordinates, predicted.float()) - transform_points(coordinates, target.float())
        return torch.linalg.vector_norm(error, dim=-1).mean()

    @staticmethod
    def _smoothness_loss(residual: torch.Tensor, optical: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        gray = optical.mean(dim=1, keepdim=True)
        dx = residual[..., 1:] - residual[..., :-1]
        dy = residual[..., 1:, :] - residual[..., :-1, :]
        image_dx = (gray[..., 1:] - gray[..., :-1]).abs()
        image_dy = (gray[..., 1:, :] - gray[..., :-1, :]).abs()
        valid_x = valid[..., 1:] & valid[..., :-1]
        valid_y = valid[..., 1:, :] & valid[..., :-1, :]
        return 0.5 * (
            masked_mean(dx.abs() * torch.exp(-10.0 * image_dx), valid_x)
            + masked_mean(dy.abs() * torch.exp(-10.0 * image_dy), valid_y)
        )

    def _residual_magnitude(self, residual: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return masked_mean(torch.linalg.vector_norm(residual, dim=1, keepdim=True), valid)

    def _residual_consistency(self, residuals: list[torch.Tensor]) -> torch.Tensor:
        if len(residuals) < 2:
            return residuals[-1].sum() * 0.0
        values = []
        for previous, current in zip(residuals[:-1], residuals[1:]):
            upsampled = resize_flow(previous, current.shape[-2:])
            values.append((current - upsampled).abs().mean())
        return torch.stack(values).mean()

    @staticmethod
    def _relation_loss(outputs: dict[str, Any]) -> torch.Tensor:
        probability = torch.nan_to_num(outputs["candidate_probabilities"].float())
        relation = torch.nan_to_num(outputs["relation_cost"].float())
        if outputs.get("relation_valid") is not None:
            probability = probability * outputs["relation_valid"].to(probability.dtype)
            return (probability * relation).sum() / probability.sum().clamp_min(1.e-8)
        association = outputs.get("association_mass")
        if association is not None:
            association_valid = torch.nan_to_num(association.float()).sum(dim=(-2, -1)) > 0
            probability = probability * association_valid.to(probability.dtype)
        return (probability * relation).sum() / probability.sum().clamp_min(1.0e-8)

    def forward(
        self,
        outputs: dict[str, Any],
        target_flow: torch.Tensor,
        valid: torch.Tensor,
        optical: torch.Tensor,
        target_affine: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        height, width = target_flow.shape[-2:]
        total_flow_loss = self._flow_loss(outputs["flows"], target_flow, valid)
        affine_corner_loss = self._affine_corner_loss(outputs["affine_matrix_pixel"], target_affine, height, width)
        zero = total_flow_loss.new_zeros(())
        affine_flow_loss = zero
        residual_loss = zero
        (
            match_loss,
            candidate_loss,
            candidate_recall,
            candidate_coverage,
            pre_candidate_recall,
            candidate_pruning_recall_drop,
            border_candidate_ratio,
        ) = self._matching_losses(outputs, target_flow, valid)
        relation_loss = self._relation_loss(outputs)
        cycle_loss = zero
        final_residual = outputs["residual_flow"]
        residual_smooth_loss = self._smoothness_loss(final_residual, optical, valid)
        residual_magnitude_loss = zero
        residual_consistency_loss = zero
        fine_point_loss = zero
        local_uot_loss = zero
        local_uot_coverage = zero
        geometry_consistency_loss = zero
        final_full_loss = zero
        sgu_diagnostics = {}
        if outputs.get('motion_mode') == 'sgu_flow':
            geo = outputs['geometry_projected_flow'].float()
            weight = F.interpolate(outputs['confidence'].detach().float(), size=geo.shape[-2:], mode='bilinear', align_corners=True)
            # Prevent confidence collapse: absolute EPE and coarse match losses remain unweighted.
            # Constrain the unbounded predictor as well: after output clipping the
            # emitted residual is already small and would hide dense-flow noise.
            consistency_flow = outputs.get('raw_dense_flow', outputs['flow']).float()
            geom_err = torch.linalg.vector_norm(consistency_flow-geo, dim=1, keepdim=True)
            geo_fit_valid = outputs.get('geometry_fit_valid')
            geo_mask = valid.bool()
            if geo_fit_valid is not None:
                geo_mask = geo_mask & geo_fit_valid.view(-1,1,1,1)
            geometry_consistency_loss = masked_mean(geom_err, geo_mask, weights=(.1+.9*weight))
            final_full_loss = self._endpoint_loss(outputs['flow'], target_flow, valid)
            if 'raw_dense_flow' in outputs:
                sgu_diagnostics['raw_dense_epe'] = self._endpoint_loss(
                    outputs['raw_dense_flow'], target_flow, valid
                )
                residual_norm = torch.linalg.vector_norm(outputs['residual_flow'].float(), dim=1, keepdim=True)
                sgu_diagnostics['final_residual_mean_px'] = masked_mean(residual_norm, valid)
                sgu_diagnostics['final_residual_max_px'] = residual_norm.amax()
            uot_losses=[]; coverages=[]
            for stage in outputs['stage_results'][:2]:
                sh,sw=stage['flow'].shape[-2:]
                target_scaled=resize_flow(target_flow,(sh,sw))
                valid_scaled=resize_mask(valid,(sh,sw)).flatten(1)
                src=stage['local_uot_source']
                gt_destination=src+target_scaled.flatten(2).transpose(1,2)
                candidates=stage['local_uot_targets']
                candidate_valid=stage['local_uot_valid']
                distance=torch.linalg.vector_norm(candidates-gt_destination.unsqueeze(2),dim=-1)
                distance=distance.masked_fill(~candidate_valid,1.e5)
                nearest_distance,nearest_index=distance.min(-1)
                if self.paper_objective:
                    # Supervise only GT points within the search window, and
                    # require their nearest sampled location to be valid content.
                    lower = candidates.amin(-2)
                    upper = candidates.amax(-2)
                    inside = ((gt_destination >= lower) & (gt_destination <= upper)).all(-1)
                    nearest_unmasked = torch.linalg.vector_norm(
                        candidates - gt_destination.unsqueeze(2), dim=-1,
                    ).argmin(-1)
                    nearest_is_valid = candidate_valid.gather(-1, nearest_unmasked.unsqueeze(-1)).squeeze(-1)
                    has_target = inside & nearest_is_valid & valid_scaled
                else:
                    has_target=(nearest_distance<=1.5)&valid_scaled
                probabilities=stage['local_uot_edges'].gather(-1,nearest_index.unsqueeze(-1)).squeeze(-1)
                if has_target.any() or self.paper_objective:
                    uot_losses.append(masked_mean(-torch.log(probabilities.clamp_min(1e-6)),has_target))
                coverages.append((has_target.float()*valid_scaled.float()).sum()/valid_scaled.float().sum().clamp_min(1))
            if uot_losses:
                local_uot_loss=torch.stack(uot_losses).mean()
            if coverages:
                local_uot_coverage=torch.stack(coverages).mean()
            # Record where the error remains large: initial T0, 1/8 UOT-Flow,
            # 1/4 UOT-Flow and 1/2 subpixel. All values use each scale's own
            # pixel units and GT resampled to the same grid.
            stages = outputs.get('stage_results', [])
            prior_stage_flow = outputs['initial_affine_flow']
            for stage_name, stage in zip(('s8', 's4', 's2'), stages):
                current_flow = stage['flow']
                stage_size = current_flow.shape[-2:]
                valid_stage = resize_mask(valid, stage_size)
                sgu_diagnostics[f'{stage_name}_flow_epe'] = self._endpoint_loss(
                    current_flow, resize_flow(target_flow, stage_size), valid_stage
                )
                prior_resized = resize_flow(prior_stage_flow, stage_size)
                delta = torch.linalg.vector_norm(current_flow.float()-prior_resized.float(), dim=1, keepdim=True)
                sgu_diagnostics[f'{stage_name}_flow_update'] = masked_mean(delta, valid_stage)
                if stage_name in {'s8', 's4'}:
                    sgu_diagnostics[f'{stage_name}_uot_confidence'] = stage['confidence'].float().mean()
                    sgu_diagnostics[f'{stage_name}_ot_blend'] = stage['local_ot_blend'].float().mean()
                    sgu_diagnostics[f'{stage_name}_refiner_gate'] = stage['local_gate'].float().mean()
                prior_stage_flow = current_flow
        if "fine_targets_pixels" in outputs and self.weights["fine_points"] > 0:
            points = outputs["fine_source_pixels"].float()
            grid = torch.stack((2.0 * points[..., 0] / max(width - 1, 1) - 1.0,
                                2.0 * points[..., 1] / max(height - 1, 1) - 1.0), dim=-1)
            sampled_gt = F.grid_sample(target_flow.float(), grid.unsqueeze(2),
                                       mode="bilinear", padding_mode="zeros", align_corners=True)
            gt_points = points + sampled_gt.squeeze(-1).transpose(1, 2)
            selected_valid = F.grid_sample(valid.float(), grid.unsqueeze(2), mode="nearest",
                                           padding_mode="zeros", align_corners=True).squeeze(1).squeeze(-1) > 0.5
            # No predicted-confidence weighting here: avoid the trivial confidence-collapse solution.
            for stage_weight, predicted in zip((0.4, 0.6), outputs["fine_targets_pixels"]):
                error = torch.linalg.vector_norm(predicted.float() - gt_points, dim=-1)
                fine_point_loss = fine_point_loss + stage_weight * masked_mean(error, selected_valid)
        # Directly supervise the coarse affine in full-resolution pixel units.
        # Use GT validity only: learned confidence cannot suppress this penalty.
        affine_reference = outputs.get("initial_affine_flow")
        if self.weights["initial_affine"] > 0 and affine_reference is None:
            raise ValueError("w_initial_affine > 0 requires outputs.initial_affine_flow")
        initial_affine_epe = (
            self._endpoint_loss(affine_reference, target_flow, valid)
            if affine_reference is not None else zero
        )
        total = (
            self.weights["initial_affine"] * initial_affine_epe
            + self.weights["total_flow"] * total_flow_loss
            + self.weights["affine_corner"] * affine_corner_loss
            + self.weights["match"] * match_loss
            + self.weights["candidate"] * candidate_loss
            + self.weights["relation"] * relation_loss
            + self.weights["residual_smooth"] * residual_smooth_loss
            + self.weights["fine_points"] * fine_point_loss
            + self.weights["local_uot"] * local_uot_loss
            + self.weights["geometry_consistency"] * geometry_consistency_loss
            + self.weights["final_full"] * final_full_loss
        )
        rg_loss = match_loss + self.relation_eta * relation_loss
        if self.paper_objective:
            # Eq. (27): exactly four terms. Auxiliary diagnostics do not train.
            total = (self.weights["match"] * rg_loss
                     + self.weights["local_uot"] * local_uot_loss
                     + self.weights["total_flow"] * total_flow_loss
                     + self.weights["affine_corner"] * affine_corner_loss)
        entropy = torch.nan_to_num(outputs["transport_entropy"].float()).mean()
        row_mass = torch.nan_to_num(outputs["row_transport_mass"].float()).mean()
        recovered_affine_epe = self._endpoint_loss(outputs['geometry_projected_flow'], target_flow, valid) if outputs.get('motion_mode') == 'sgu_flow' else zero
        recovery_improvement = zero
        return total, {
            "rg_loss": rg_loss,
            "loss": total,
            "total": total,
            "fine_point_loss": fine_point_loss,
            "local_uot_loss": local_uot_loss,
            "local_uot_coverage": local_uot_coverage,
            "geometry_consistency_loss": geometry_consistency_loss,
            "final_full_loss": final_full_loss,
            "total_flow_loss": total_flow_loss,
            "flow_loss": total_flow_loss,
            "affine_corner_loss": affine_corner_loss,
            "affine_flow_loss": affine_flow_loss,
            "affine_epe": initial_affine_epe,
            "affine_corner_error": affine_corner_loss,
            "residual_loss": residual_loss,
            "residual_magnitude_loss": residual_magnitude_loss,
            "residual_smooth_loss": residual_smooth_loss,
            "residual_consistency_loss": residual_consistency_loss,
            "match_loss": match_loss,
            "candidate_coverage_loss": candidate_loss,
            "relation_loss": relation_loss,
            "cycle_loss": cycle_loss,
            "candidate_recall": candidate_recall,
            "pre_candidate_recall": pre_candidate_recall,
            "graph_candidate_recall": candidate_recall,
            "candidate_pruning_recall_drop": candidate_pruning_recall_drop,
            "border_candidate_ratio": border_candidate_ratio,
            "candidate_coverage": candidate_coverage,
            "transport_entropy": entropy,
            "row_transport_mass": row_mass,
            "graph_w_feature": outputs.get('graph_cost_weights',zero.new_zeros((2,3)))[:,0].mean(),
            "graph_w_spatial": outputs.get('graph_cost_weights',zero.new_zeros((2,3)))[:,1].mean(),
            "graph_w_semantic": outputs.get('graph_cost_weights',zero.new_zeros((2,3)))[:,2].mean(),
            "affine_spatial_coverage": torch.as_tensor(
                outputs.get("affine_spatial_coverage", float("nan")), device=target_flow.device
            ).float().mean(),
            "affine_effective_points": torch.as_tensor(
                outputs.get("affine_effective_points", outputs.get("effective_point_count", float("nan"))),
                device=target_flow.device,
            ).float().mean(),
            "affine_fallback_ratio": torch.as_tensor(
                outputs.get("affine_fallback_ratio", float("nan")), device=target_flow.device
            ).float().mean(),
            "initial_affine_epe": initial_affine_epe,
            "initial_affine_loss": initial_affine_epe,
            "initial_affine_weight": zero.new_tensor(self.weights["initial_affine"]),
            "recovered_affine_epe": recovered_affine_epe,
            "s8_recovery_improvement": recovery_improvement,
            "s8_recovery_gate": torch.as_tensor(
                outputs.get("s8_recovery_gate", float("nan")), device=target_flow.device
            ).float().mean(),
            **sgu_diagnostics,
        }
