"""Learned cross-modal structure and DINOv3-anchored multi-scale fusion.

There are no handcrafted gradient, Sobel, ROEWA, phase-congruency, or image
finite-difference operators here. Structure quality is learned from matching
supervision; reliability is an estimate rather than a guaranteed overlap label.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


def _groups(dim: int) -> int:
    return next(g for g in (8, 4, 2, 1) if dim % g == 0)


class ConvGNAct(nn.Sequential):
    def __init__(self, in_dim: int, out_dim: int, stride: int = 1):
        super().__init__(
            nn.Conv2d(in_dim, out_dim, 3, stride, 1, bias=False),
            nn.GroupNorm(_groups(out_dim), out_dim),
            nn.GELU(),
        )


class LearnedResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            ConvGNAct(dim, dim),
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(dim), dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x + self.layers(x))


class CrossModalStructuralLocalEncoder(nn.Module):
    """CSLE: separate modality adapters followed by one SHARED learned encoder.

    Returns multi-scale structures for *each* image, without recalculation:
      s8: [B,c8,H/8,W/8], s4: [B,c4,H/4,W/4], s2: [B,c2,H/2,W/2].
    The s2 head is retained to drive the original half-resolution subpixel stage.
    """

    def __init__(self, c8: int = 96, c4: int = 64, c2: int = 48, stem_dim: int = 48):
        super().__init__()
        self.optical_adapter = nn.Sequential(ConvGNAct(3, stem_dim, 2), LearnedResidualBlock(stem_dim))
        self.sar_adapter = nn.Sequential(ConvGNAct(1, stem_dim, 2), LearnedResidualBlock(stem_dim))
        self.shared_half = nn.Sequential(ConvGNAct(stem_dim, c2), LearnedResidualBlock(c2))
        self.shared_quarter = nn.Sequential(ConvGNAct(stem_dim, c4, 2), LearnedResidualBlock(c4))
        self.shared_eighth = nn.Sequential(ConvGNAct(c4, c8, 2), LearnedResidualBlock(c8))
        # Learn a top-down local-context correction without discarding s4 detail.
        self.context4 = nn.Conv2d(c8, c4, 1, bias=False)
        self.context_gate = nn.Sequential(nn.Conv2d(2*c4, c4, 1), nn.Sigmoid())
        self.context_gain = nn.Parameter(torch.tensor(-2.0))

    def forward_one(self, image: torch.Tensor, modality: str) -> dict[str, torch.Tensor]:
        if modality not in ('optical', 'sar'):
            raise ValueError("modality must be 'optical' or 'sar'")
        stem = self.optical_adapter(image) if modality == 'optical' else self.sar_adapter(image)
        s2 = self.shared_half(stem)
        s4 = self.shared_quarter(stem)
        s8 = self.shared_eighth(s4)
        contextual = F.interpolate(self.context4(s8), size=s4.shape[-2:], mode='bilinear', align_corners=False)
        gate = self.context_gate(torch.cat((s4, contextual), dim=1))
        s4 = s4 + torch.sigmoid(self.context_gain) * gate * contextual
        return {'s8': s8, 's4': s4, 's2': s2}

    def forward(self, optical: torch.Tensor, sar: torch.Tensor) -> tuple[dict, dict]:
        return self.forward_one(optical, 'optical'), self.forward_one(sar, 'sar')


class SubpatchSemanticAttention(nn.Module):
    """Per-token semantic queries, spatially ordered 2x2 or 4x4 structural K/V."""

    def __init__(self, local_channels: int, dim: int, scale: int):
        super().__init__()
        if scale not in (2, 4):
            raise ValueError('subpatch scale must be 2 or 4')
        self.scale = scale
        self.key_value = nn.Conv2d(local_channels, 2*dim, 1)
        self.quality = nn.Conv2d(local_channels, 1, 3, padding=1)
        self.rel_pos = nn.Parameter(torch.zeros(1, 1, scale*scale, dim))

    def forward(self, query: torch.Tensor, local: torch.Tensor,
                valid_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        b, dim, h, w = query.shape
        if local.shape[-2:] != (h*self.scale, w*self.scale):
            raise ValueError('CSLE / DINO grid mismatch: choose dimensions divisible by 16')
        n, patches = h*w, self.scale*self.scale
        kv = F.unfold(self.key_value(local), kernel_size=self.scale, stride=self.scale)
        kv = kv.view(b, 2, dim, patches, n).permute(0, 4, 3, 1, 2)
        k, v = kv[..., 0, :], kv[..., 1, :]
        q = query.flatten(2).transpose(1, 2)
        quality = torch.sigmoid(self.quality(local))
        quality = F.unfold(quality, kernel_size=self.scale, stride=self.scale).transpose(1, 2)
        if valid_mask is not None:
            if valid_mask.shape[-2:] != local.shape[-2:]:
                valid_mask = F.interpolate(valid_mask.float(), size=local.shape[-2:], mode='nearest')
            validity = F.unfold(valid_mask.float(), kernel_size=self.scale, stride=self.scale).transpose(1, 2)
            quality = quality * validity
        k = k + self.rel_pos
        logits = (q.unsqueeze(2) * k).sum(-1) / math.sqrt(dim)
        logits = logits + torch.log(quality.clamp_min(1.e-6))
        weights = logits.softmax(-1)
        # For fully invalid patches, weights may be uniform, but their output is
        # zeroed and never converted into structural evidence.
        mass = quality.sum(-1) > 1.e-6
        aggregated = (weights.unsqueeze(-1) * v).sum(-2)
        aggregated = aggregated * mass.unsqueeze(-1).to(aggregated.dtype)
        confidence = (weights * quality).sum(-1).clamp(0, 1)
        return aggregated.transpose(1, 2).reshape(b, dim, h, w), confidence.view(b, 1, h, w)


class SemanticStructureFusion(nn.Module):
    """SALA (both scales) -> structural descriptor -> SRGF gated residual.

    Fused feature goes to the original graph / UOT matcher. Structural feature
    goes separately to its graph edge builder, replacing fixed Sobel statistics.
    """

    def __init__(self, dino_dim: int, c8: int, c4: int, dim: int = 256):
        super().__init__()
        self.semantic = nn.Sequential(nn.Conv2d(dino_dim, dim, 1, bias=False), nn.GroupNorm(_groups(dim), dim))
        self.sala8 = SubpatchSemanticAttention(c8, dim, scale=2)
        self.sala4 = SubpatchSemanticAttention(c4, dim, scale=4)
        self.local_fusion = nn.Sequential(ConvGNAct(2*dim, dim), nn.Conv2d(dim, dim, 1))
        self.structure_proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.gate = nn.Sequential(
            nn.Conv2d(2*dim+2, dim, 1), nn.GELU(), nn.Conv2d(dim, dim, 1), nn.Sigmoid()
        )
        self.local_residual = nn.Conv2d(dim, dim, 1, bias=False)
        self.mix_logit = nn.Parameter(torch.tensor(-2.0))

    def forward(self, dino16: torch.Tensor, local: dict,
                mask4: torch.Tensor | None = None,
                mask8: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        semantic = F.normalize(self.semantic(dino16.float()), dim=1)
        local8, rel8 = self.sala8(semantic, local['s8'].float(), mask8)
        local4, rel4 = self.sala4(semantic, local['s4'].float(), mask4)
        structural = self.local_fusion(torch.cat((local8, local4), 1))
        structure_feature = F.normalize(self.structure_proj(structural), dim=1)
        conf = torch.cat((rel8, rel4), dim=1)
        gate = self.gate(torch.cat((semantic, structure_feature, conf), dim=1))
        fused = F.normalize(semantic + torch.sigmoid(self.mix_logit)*gate*self.local_residual(structure_feature), dim=1)
        return {'fused': fused, 'structure': structure_feature, 'reliability': conf.mean(1,keepdim=True),
                'gate': gate, 'sala8': local8, 'sala4': local4}
