"""Two-scale (1/4 then 1/2) fine correspondences + robust similarity refit.

No local UOT and no free dense residual field. Matches are expressed in full-image
optical->SAR pixel coordinates and the final dense mapping comes from Sim(2).
"""
from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F

from .geometry import affine_to_flow, normalized_affine_to_pixel, pixel_affine_to_normalized, transform_points


def _coordinate_grid(height: int, width: int, device, dtype):
    yy, xx = torch.meshgrid(torch.arange(height, device=device, dtype=dtype),
                            torch.arange(width, device=device, dtype=dtype), indexing='ij')
    return torch.stack((xx, yy), -1)


def _normalized(points: torch.Tensor, h: int, w: int):
    scale = points.new_tensor([max(w - 1, 1), max(h - 1, 1)])
    return 2.0 * points / scale - 1.0


def _sample(feat: torch.Tensor, points: torch.Tensor):
    b, _, h, w = feat.shape
    grid = _normalized(points, h, w).view(b, -1, 1, 2)
    return F.grid_sample(feat, grid, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze(-1).transpose(1, 2)


class FineLocalMatcher(nn.Module):
    def __init__(self, channels: int, descriptor_dim: int, window: int, temperature: float):
        super().__init__()
        if window < 3 or window % 2 != 1:
            raise ValueError('Fine matching window must be odd and >=3')
        self.opt_proj = nn.Conv2d(channels, descriptor_dim, 1)
        self.sar_proj = nn.Conv2d(channels, descriptor_dim, 1)
        self.window = window
        self.temperature = temperature
        offsets = _coordinate_grid(window, window, 'cpu', torch.float32) - (window // 2)
        self.register_buffer('offsets', offsets.reshape(-1, 2), persistent=False)

    def forward(self, opt: torch.Tensor, sar: torch.Tensor,
                source_pixels: torch.Tensor, target_pixels: torch.Tensor,
                full_size: tuple[int, int], source_mask: torch.Tensor,
                target_mask: torch.Tensor):
        # Pixel-space points are mapped to pyramid coordinates by align_corners=True.
        b, _, h, w = opt.shape
        full_h, full_w = full_size
        scalers = source_pixels.new_tensor([(full_w - 1) / max(w - 1, 1),
                                            (full_h - 1) / max(h - 1, 1)])
        offsets = self.offsets.to(dtype=source_pixels.dtype) * scalers
        queries = _sample(F.normalize(self.opt_proj(opt.float()), dim=1),
                          source_pixels / scalers)
        keys = F.normalize(self.sar_proj(sar.float()), dim=1)
        points = target_pixels.unsqueeze(2) + offsets.view(1, 1, -1, 2)
        candidates = points.reshape(b, -1, 2)
        sampled_keys = _sample(keys, candidates / scalers).view(b, source_pixels.shape[1], -1, keys.shape[1])
        logits = (queries.unsqueeze(2) * sampled_keys).sum(-1) / max(self.temperature, 1e-4)
        candidate_valid = ((points[..., 0] >= 0) & (points[..., 0] <= full_w - 1)
                           & (points[..., 1] >= 0) & (points[..., 1] <= full_h - 1))
        target_valid = _sample(target_mask.float(), candidates / scalers).view(b, -1, offsets.shape[0]) > 0.9
        source_valid = _sample(source_mask.float(), source_pixels / scalers).squeeze(-1) > 0.9
        valid = candidate_valid & target_valid & source_valid.unsqueeze(-1)
        any_valid = valid.any(-1)
        prob = torch.softmax(logits.masked_fill(~valid, -1.0e4), dim=-1) * valid.float()
        prob = prob / prob.sum(-1, keepdim=True).clamp_min(1.0e-8)
        delta = torch.einsum('bnk,kd->bnd', prob, offsets)
        entropy = -(prob.clamp_min(1e-8) * prob.clamp_min(1e-8).log()).sum(-1)
        concentration = (1.0 - entropy / math.log(self.window * self.window)).clamp(0, 1)
        peak = prob.amax(-1)
        confidence = (0.1 + 0.9 * concentration) * peak.sqrt() * any_valid.float()
        return target_pixels + delta, confidence, any_valid, prob


class TwoScaleGeometricRefiner(nn.Module):
    def __init__(self, c4: int, c2: int, cfg: dict):
        super().__init__()
        dim = int(cfg.get('descriptor_dim', 32))
        self.match4 = FineLocalMatcher(c4, dim, int(cfg.get('window_s4', 7)), float(cfg.get('temperature', 0.1)))
        self.match2 = FineLocalMatcher(c2, dim, int(cfg.get('window_s2', 5)), float(cfg.get('temperature', 0.1)))
        self.grid_stride_s4 = int(cfg.get('grid_stride_s4', 2))
        self.irls_iterations = int(cfg.get('irls_iterations', 2))
        self.robust_delta = float(cfg.get('huber_delta_px', 3.0))
        self.min_points = int(cfg.get('min_points', 12))
        self.max_correction = float(cfg.get('max_correction_px', 30.0))
        self.initial_confidence_floor = float(cfg.get('initial_confidence_floor', 0.02))

    def _fit_similarity(self, src_px, dst_px, weight, initial_px, full_size):
        """Weighted Sim(2) fit in isotropically normalized *pixel* coordinates.

        Centering and scaling by the SAME scalar in x/y preserves true pixel-space
        rotation + uniform scale on rectangular as well as square image sizes.
        """
        b, n, _ = src_px.shape
        h, w = full_size
        center = src_px.new_tensor([(w - 1) / 2.0, (h - 1) / 2.0])
        length = float(max(w - 1, h - 1, 1) / 2.0)
        src = ((src_px.float() - center) / length)
        dst = ((dst_px.float() - center) / length)
        weight = torch.nan_to_num(weight.float()).clamp_min(0.0)
        selected = weight > 1e-5
        can_fit = selected.sum(-1) >= self.min_points
        iter_weight = weight
        candidate_pixel = initial_px.float()
        for _ in range(self.irls_iterations + 1):
            sum_w = iter_weight.sum(-1, keepdim=True).clamp_min(1e-8)
            mean_src = (src * iter_weight.unsqueeze(-1)).sum(1) / sum_w
            mean_dst = (dst * iter_weight.unsqueeze(-1)).sum(1) / sum_w
            xs = src - mean_src.unsqueeze(1)
            ys = dst - mean_dst.unsqueeze(1)
            den = (iter_weight * xs.square().sum(-1)).sum(-1).clamp_min(1e-8)
            a = (iter_weight * (xs[..., 0] * ys[..., 0] + xs[..., 1] * ys[..., 1])).sum(-1) / den
            c = (iter_weight * (xs[..., 0] * ys[..., 1] - xs[..., 1] * ys[..., 0])).sum(-1) / den
            row0 = torch.stack((a, -c), -1)
            row1 = torch.stack((c, a), -1)
            rot = torch.stack((row0, row1), -2)
            shift = mean_dst - torch.einsum('bij,bj->bi', rot, mean_src)
            translation = center - torch.einsum('bij,j->bi', rot, center) + length * shift
            candidate_pixel = torch.cat((rot, translation.unsqueeze(-1)), dim=-1)
            resid = torch.linalg.vector_norm(transform_points(src_px, candidate_pixel) - dst_px, dim=-1)
            robust = torch.where(resid <= self.robust_delta, torch.ones_like(resid),
                                 self.robust_delta / resid.clamp_min(1e-6)).detach()
            iter_weight = weight * robust
        difference = torch.linalg.vector_norm(transform_points(src_px, candidate_pixel)
                                               - transform_points(src_px, initial_px), dim=-1)
        mean_correction = (difference * weight).sum(-1) / weight.sum(-1).clamp_min(1e-8)
        accepted = can_fit & torch.isfinite(candidate_pixel).all(dim=(-2, -1)) & (mean_correction < self.max_correction)
        result_pixel = torch.where(accepted[:, None, None], candidate_pixel, initial_px.float())
        result_norm = pixel_affine_to_normalized(result_pixel, full_size)
        return result_norm, accepted, mean_correction

    def forward(self, opt4, sar4, opt2, sar2, initial_normalized,
                full_size, opt_mask4, sar_mask4, opt_mask2, sar_mask2,
                coarse_confidence=None):
        b, _, h4, w4 = opt4.shape
        full_h, full_w = full_size
        grid = _coordinate_grid(h4, w4, opt4.device, torch.float32)
        grid = grid[::self.grid_stride_s4, ::self.grid_stride_s4].reshape(-1, 2)
        conversion = grid.new_tensor([(full_w - 1) / max(w4 - 1, 1),
                                      (full_h - 1) / max(h4 - 1, 1)])
        source_px = (grid * conversion).unsqueeze(0).expand(b, -1, -1)
        initial_px = normalized_affine_to_pixel(initial_normalized.float(), full_size)
        predicted = transform_points(source_px, initial_px)
        if coarse_confidence is not None:
            coarse_prob = _sample(F.interpolate(coarse_confidence.float(), size=(h4,w4), mode='bilinear', align_corners=True),
                                  grid.unsqueeze(0).expand(b,-1,-1)).squeeze(-1)
            prior = self.initial_confidence_floor + (1.0 - self.initial_confidence_floor) * coarse_prob
        else:
            prior = torch.ones((b,grid.shape[0]), device=opt4.device)
        match4, conf4, valid4, _ = self.match4(opt4, sar4, source_px, predicted, full_size, opt_mask4, sar_mask4)
        # If 1/4 stage cannot find a match, fall back to geometric prediction at that point.
        match4 = torch.where(valid4.unsqueeze(-1), match4, predicted)
        match2, conf2, valid2, _ = self.match2(opt2, sar2, source_px, match4, full_size, opt_mask2, sar_mask2)
        match2 = torch.where(valid2.unsqueeze(-1), match2, match4)
        weights = (conf4 * conf2 * prior * valid4.float() * valid2.float()).float()
        refined_normalized, accepted, correction = self._fit_similarity(source_px, match2, weights, initial_px, full_size)
        affine_final = normalized_affine_to_pixel(refined_normalized, full_size)
        # Flows contain global similarity estimates at s4/s2, for existing multi-scale training API.
        fit4, _, _ = self._fit_similarity(source_px, match4, conf4 * prior * valid4, initial_px, full_size)
        flow4 = affine_to_flow(fit4, opt4.shape[-2:], normalized=True)
        flow2 = affine_to_flow(refined_normalized, opt2.shape[-2:], normalized=True)
        full_flow = affine_to_flow(refined_normalized, full_size, normalized=True)
        initial_flow = affine_to_flow(initial_normalized, full_size, normalized=True)
        return dict(
            flow=full_flow, flows=[flow4, flow2], affine_matrix_normalized=refined_normalized,
            affine_matrix_pixel=affine_final, initial_affine_pixel=initial_px,
            affine_flow=full_flow, initial_affine_flow=initial_flow,
            residual_flow=full_flow - initial_flow,
            residual_flows=[flow4 - affine_to_flow(initial_normalized, opt4.shape[-2:], normalized=True),
                            flow2 - affine_to_flow(initial_normalized, opt2.shape[-2:], normalized=True)],
            fine_source_pixels=source_px, fine_targets_pixels=[match4, match2],
            fine_matches_weights=weights, fine_fit_accepted=accepted,
            fine_refit_mean_correction=correction,
            fine_valid=valid4 & valid2,
        )
