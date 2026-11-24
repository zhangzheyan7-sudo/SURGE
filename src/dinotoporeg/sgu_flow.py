"""SGU-Flow: graph-cost sparse UOT embedded in geometrically constrained dense flow.

Coordinate convention: optical -> SAR, each pyramid operates in its OWN pixel grid;
resize_flow performs the necessary pixel-unit conversion. No dense N-by-N matrix.
"""
from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F

from .graph import SparseLocalUnbalancedSinkhornMatcher
from .geometry import affine_to_flow, normalized_affine_to_pixel, pixel_affine_to_normalized, transform_points
from .utils import resize_flow, flow_warp


def _group(c):
    return next(g for g in (8, 4, 2, 1) if c % g == 0)


def _block(ci, co, stride=1):
    return nn.Sequential(nn.Conv2d(ci, co, 3, stride, 1, bias=False), nn.GroupNorm(_group(co), co), nn.GELU(),
                         nn.Conv2d(co, co, 3, 1, 1, bias=False), nn.GroupNorm(_group(co), co), nn.GELU())


class GraphCostDescriptor(nn.Module):
    """Lightweight spatial 8-neighbor and nonlocal semantic prototype-KNN descriptors.

    Semantic anchors are adaptive pooled nodes (<=8x8), NOT quadratic all-node KNN.
    This is an approximation to original full-grid semantic neighbor construction.
    """
    def __init__(self, channels, semantic_k=4, prototype_side=8):
        super().__init__()
        self.k = semantic_k
        self.prototype_side = prototype_side
        self.spatial_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.semantic_proj = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x, mask):
        mask = mask.float()
        valid = F.avg_pool2d(mask, 3, stride=1, padding=1) * 9 - mask
        summation = F.avg_pool2d(x * mask, 3, stride=1, padding=1) * 9 - x * mask
        spatial = summation / valid.clamp_min(1.0)
        spatial = F.normalize(self.spatial_proj(spatial).float(), dim=1)
        b, c, h, w = x.shape
        sh, sw = min(h, self.prototype_side), min(w, self.prototype_side)
        anchor_weight = F.adaptive_avg_pool2d(mask, (sh, sw))
        anchor = F.adaptive_avg_pool2d(x * mask, (sh, sw)) / anchor_weight.clamp_min(1.e-5)
        anchor = F.normalize(anchor.flatten(2).float(), dim=1)
        query = F.normalize(x.flatten(2).float(), dim=1)
        similarity = torch.bmm(query.transpose(1, 2), anchor)
        # Semantic edges must be NONLOCAL: exclude anchors that overlap each query's
        # immediate neighborhood in anchor-grid coordinates.
        yy, xx = torch.meshgrid(torch.arange(h, device=x.device), torch.arange(w, device=x.device), indexing='ij')
        ay, ax = torch.meshgrid(torch.arange(sh, device=x.device), torch.arange(sw, device=x.device), indexing='ij')
        dy = (yy.flatten()[:, None].float() + .5) / h * sh - (ay.flatten()[None].float() + .5)
        dx = (xx.flatten()[:, None].float() + .5) / w * sw - (ax.flatten()[None].float() + .5)
        remote = (dx.abs() > 1.0) | (dy.abs() > 1.0)
        scores = similarity.masked_fill(~remote.unsqueeze(0), -1.e4)
        scores = scores.masked_fill((anchor_weight.flatten(2) < .5), -1.e4)
        k = min(self.k, sh * sw)
        values, inds = scores.topk(k, dim=-1)
        weights = F.softmax(values, dim=-1)
        possible = values.amax(-1) > -1.e3
        weights = weights * possible.unsqueeze(-1)
        anch = anchor.transpose(1, 2)
        idx = torch.arange(b, device=x.device).view(b, 1, 1)
        semantic = (weights.unsqueeze(-1) * anch[idx, inds]).sum(-2)
        semantic = semantic.transpose(1, 2).reshape(b, c, h, w)
        semantic = F.normalize(self.semantic_proj(semantic).float(), dim=1)
        return spatial, semantic


class SparseUOTFlowStage(nn.Module):
    def __init__(self, channels, config, radius):
        super().__init__()
        dim = int(config.get('descriptor_dim', 32))
        self.opt_projection = nn.Conv2d(channels, dim, 1)
        self.sar_projection = nn.Conv2d(channels, dim, 1)
        self.value_projection = nn.Linear(dim, dim, bias=False)
        self.query_norm = nn.LayerNorm(dim)
        self.attention_gain = nn.Parameter(torch.tensor(-2.0))
        self.graph = GraphCostDescriptor(dim, int(config.get('semantic_k', 4)), int(config.get('prototype_side', 8)))
        self.mix_logits = nn.Parameter(torch.tensor([2.0, -0.5, -0.5], dtype=torch.float))
        self.geometry_weight = float(config.get('geometry_cost_weight', .1))
        self.radius = int(radius)
        self.attention_temperature = float(config.get('attention_temperature', 0.2))
        self.uot = SparseLocalUnbalancedSinkhornMatcher(
            epsilon=float(config.get('epsilon', .2)), tau_source=float(config.get('tau_source', .8)),
            tau_target=float(config.get('tau_target', .8)),
            iterations=int(config.get('iterations', 5)), dustbin_score=float(config.get('dustbin_score', 0.0)),
            dustbin_enabled=bool(config.get('dustbin_enabled', True)),
        )
        self.max_blend = float(config.get('max_blend', .7))
        # ``legacy_double`` is kept for exact ablation of the released vNext.
        # The preferred gate calibrates uncertain (but valid) UOT correspondences
        # once, and lets the local refiner learn its own residual gate.
        self.flow_gate_mode = str(config.get('flow_gate_mode', 'legacy_double'))
        if self.flow_gate_mode not in {'legacy_double', 'calibrated_single'}:
            raise ValueError(f'Unsupported uot_flow.flow_gate_mode: {self.flow_gate_mode}')
        self.ot_gate_logit_bias = nn.Parameter(torch.tensor(
            float(config.get('ot_gate_logit_bias', 1.5))
        )) if self.flow_gate_mode == 'calibrated_single' else None
        self.refiner = nn.Sequential(_block(channels*2+4, channels), nn.Conv2d(channels, 3, 3, padding=1))
        nn.init.zeros_(self.refiner[-1].weight)
        nn.init.zeros_(self.refiner[-1].bias)
        self.residual_radius = float(config.get('residual_radius', 2.0))

    def forward(self, opt, sar, prior_flow, source_mask, target_mask):
        # Flow, correlation and UOT all run in float32 under AMP for stability.
        b, _, h, w = opt.shape
        n = h*w
        opt_keys = F.normalize(self.opt_projection(opt).float(), dim=1)
        sar_keys = F.normalize(self.sar_projection(sar).float(), dim=1)
        opt_sp, opt_sem = self.graph(opt_keys, source_mask)
        sar_sp, sar_sem = self.graph(sar_keys, target_mask)
        optical = opt_keys.flatten(2).transpose(1, 2)
        sar_flat = sar_keys.flatten(2).transpose(1, 2)
        prior = resize_flow(prior_flow.float(), (h, w))
        yy, xx = torch.meshgrid(torch.arange(h, device=opt.device), torch.arange(w, device=opt.device), indexing='ij')
        base = torch.stack((xx, yy), dim=-1).reshape(1, n, 2).float()
        centers = base + prior.float().flatten(2).transpose(1, 2)
        r = self.radius
        sy, sx = torch.meshgrid(torch.arange(-r,r+1,device=opt.device), torch.arange(-r,r+1,device=opt.device), indexing='ij')
        offsets = torch.stack((sx.flatten(),sy.flatten()), dim=-1)
        cand = centers.round().long().unsqueeze(2) + offsets.view(1,1,-1,2)
        candidate_valid = (cand[...,0]>=0)&(cand[...,0]<w)&(cand[...,1]>=0)&(cand[...,1]<h)
        clamped = torch.stack((cand[...,0].clamp(0,w-1), cand[...,1].clamp(0,h-1)),dim=-1)
        inds = clamped[...,1]*w+clamped[...,0]
        src_valid = source_mask.flatten(1).bool()
        tgt_valid = target_mask.flatten(1).bool()
        candidate_valid = candidate_valid & src_valid.unsqueeze(-1) & tgt_valid.gather(1,inds.reshape(b,-1)).reshape_as(inds)
        ix = torch.arange(b,device=opt.device).view(b,1,1)
        keys = sar_flat[ix,inds]
        # Local cross-attention contextualizes Q with only geometry-constrained SAR candidates.
        attention_logits = (optical.unsqueeze(-2)*keys).sum(-1)/self.attention_temperature
        attention = F.softmax(attention_logits.masked_fill(~candidate_valid,-1e4),dim=-1)
        attention = attention * candidate_valid.float()
        attention = attention / attention.sum(-1,keepdim=True).clamp_min(1e-8)
        context = (attention.unsqueeze(-1) * self.value_projection(keys)).sum(-2)
        query = F.normalize(self.query_norm(optical + torch.sigmoid(self.attention_gain)*context),dim=-1)
        feat_score = (query.unsqueeze(2)*keys).sum(-1)
        ssp = sar_sp.flatten(2).transpose(1,2)[ix,inds]
        ssem = sar_sem.flatten(2).transpose(1,2)[ix,inds]
        osp = opt_sp.flatten(2).transpose(1,2)
        osem = opt_sem.flatten(2).transpose(1,2)
        spatial_score = (osp.unsqueeze(2)*ssp).sum(-1)
        semantic_score = (osem.unsqueeze(2)*ssem).sum(-1)
        geometry_cost = ((cand.float()-centers.unsqueeze(2)).square().sum(-1) / max(r*r,1)).clamp_max(16)
        coefficients = F.softmax(self.mix_logits,dim=0)
        # Log-kernel cost equals -score; NOT adjacency-matrix concatenation.
        scores = coefficients[0]*feat_score+coefficients[1]*spatial_score+coefficients[2]*semantic_score-self.geometry_weight*geometry_cost
        scores = scores.masked_fill(~candidate_valid,-1e4)
        result = self.uot(scores,inds,candidate_valid,src_valid.float(),tgt_valid.float(),n)
        mass = result['row_transport_mass'].float()
        probs = result['edge_transport'].float()/mass.unsqueeze(-1).clamp_min(1e-8)
        target = (probs.unsqueeze(-1)*cand.float()).sum(-2)
        ot_flow = (target-base).transpose(1,2).reshape(b,2,h,w)
        valid_rows = candidate_valid.any(-1) & (mass>1e-7)
        concentration = probs.amax(-1)
        valid_count = candidate_valid.sum(-1).clamp_min(2)
        entropy = result['transport_entropy'].float()/valid_count.float().log()
        confidence = (mass.clamp(0,1)*concentration*torch.exp(-entropy)).clamp(0,1)*valid_rows.float()
        confidence = confidence.reshape(b,1,h,w)
        entropy_map = entropy.reshape(b,1,h,w).clamp(0,1)
        if self.flow_gate_mode == 'calibrated_single':
            # At c=0.1, a +1.5 logit shift raises the calibrated confidence to
            # ~0.33 (rather than forcing two independent multiplications by 0.1).
            # Invalid rows ALWAYS have zero OT blend, even if the bias grows.
            confidence_logit = torch.logit(confidence.detach().clamp(1e-6, 1.0 - 1e-6))
            calibrated = torch.sigmoid(confidence_logit + self.ot_gate_logit_bias)
            blend = self.max_blend * calibrated * valid_rows.reshape(b, 1, h, w).float()
        else:
            blend = self.max_blend * confidence.detach()
        flow_init = prior * (1-blend) + ot_flow * blend
        flow_init = torch.where(valid_rows.reshape(b,1,h,w),flow_init,prior)
        warped = flow_warp(sar.float(),flow_init)
        residual_input = torch.cat((opt.float(),warped,flow_init/max(h,w),confidence,entropy_map),dim=1)
        prediction = self.refiner(residual_input)
        correction = prediction[:,:2].tanh()*self.residual_radius
        if self.flow_gate_mode == 'calibrated_single':
            # No second confidence multiplication: the refiner must still be
            # able to correct a weak/ambiguous OT assignment. It can only change
            # geometrically valid content, never SAR black borders/unmatched areas.
            target_at_init = flow_warp(target_mask.float(), flow_init)
            refiner_valid = source_mask.bool() & (target_at_init > 0.5)
            gate = torch.sigmoid(prediction[:,2:]) * refiner_valid.float()
        else:
            # The old double attenuation is preserved for controlled experiments.
            gate = torch.sigmoid(prediction[:,2:]) * confidence.detach()
        updated = flow_init + gate*correction
        return dict(flow=updated, confidence=confidence, local_uot_flow=ot_flow, local_uot_mass=mass.reshape(b,1,h,w),
                    local_uot_entropy=entropy_map, local_gate=gate, local_ot_blend=blend,
                    local_cost_weights=coefficients,
                    local_uot_edges=probs, local_uot_indices=inds, local_uot_valid=candidate_valid,
                    local_uot_source=base.expand(b,-1,-1), local_uot_targets=cand.float())


def sample_continuous_candidates(descriptors, coordinates, target_mask):
    """Sample B,N,K floating pixel coordinates without crossing invalid content."""
    h, w = descriptors.shape[-2:]
    grid = torch.stack((2 * coordinates[..., 0] / max(w - 1, 1) - 1,
                        2 * coordinates[..., 1] / max(h - 1, 1) - 1), -1)
    with torch.autocast(device_type=descriptors.device.type, enabled=False):
        sampled = F.grid_sample(descriptors.float(), grid.float(), align_corners=True,
                                mode='bilinear', padding_mode='zeros')
        support = F.grid_sample(target_mask.float(), grid.float(), align_corners=True,
                                mode='bilinear', padding_mode='zeros').squeeze(1)
    keys = F.normalize(sampled.permute(0, 2, 3, 1), dim=-1)
    inside = ((coordinates[..., 0] >= 0) & (coordinates[..., 0] <= w - 1)
              & (coordinates[..., 1] >= 0) & (coordinates[..., 1] <= h - 1))
    return keys, inside & (support >= 1.0 - 1.e-5)


class SubpixelHalfStage(nn.Module):
    """Local descriptor search, peak-neighborhood expectation and gated refinement.

    Continuous mode samples around floating prior centers; integer mode retains
    the old rounded grid for controlled comparisons. Neither adds parameters.
    """
    def __init__(self, channels, config):
        super().__init__()
        dim=int(config.get('descriptor_dim',32))
        self.q=nn.Conv2d(channels,dim,1)
        self.k=nn.Conv2d(channels,dim,1)
        self.temperature=float(config.get('temperature',.15))
        self.radius=int(config.get('radius',2))
        self.sampling_mode = str(config.get('sampling_mode', 'integer'))
        if self.sampling_mode not in {'integer', 'continuous'}:
            raise ValueError('subpixel.sampling_mode must be integer or continuous')
        if self.radius < 0 or self.temperature <= 0:
            raise ValueError('Invalid subpixel search radius or temperature')
        self.head=nn.Sequential(_block(channels*2+3,channels),nn.Conv2d(channels,3,3,padding=1))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
        self.residual_radius=float(config.get('residual_radius',1.0))

    def forward(self,opt,sar,prior,source_mask,target_mask):
        b,_,h,w=opt.shape
        prior=resize_flow(prior,(h,w))
        q=F.normalize(self.q(opt).float(),dim=1)
        k=F.normalize(self.k(sar).float(),dim=1)
        n=h*w;r=self.radius
        yy,xx=torch.meshgrid(torch.arange(h,device=opt.device),torch.arange(w,device=opt.device),indexing='ij')
        base=torch.stack((xx,yy),-1).view(1,n,2).float()
        centers=base+prior.flatten(2).transpose(1,2)
        sy,sx=torch.meshgrid(torch.arange(-r,r+1,device=opt.device),torch.arange(-r,r+1,device=opt.device),indexing='ij')
        offsets=torch.stack((sx.flatten(),sy.flatten()),-1)
        if self.sampling_mode == 'continuous':
            coords = centers.unsqueeze(2) + offsets.view(1, 1, -1, 2).float()
            keys, good = sample_continuous_candidates(k, coords, target_mask)
            # Nearest indices are diagnostics only; matching uses floating coords.
            nearest = coords.round().long()
            inds = nearest[...,1].clamp(0,h-1)*w + nearest[...,0].clamp(0,w-1)
            valid = good & source_mask.flatten(1).bool().unsqueeze(-1)
        else:
            coords=centers.round().long().unsqueeze(2)+offsets.view(1,1,-1,2)
            good=(coords[...,0]>=0)&(coords[...,0]<w)&(coords[...,1]>=0)&(coords[...,1]<h)
            inds=coords[...,1].clamp(0,h-1)*w+coords[...,0].clamp(0,w-1)
            valid=good & source_mask.flatten(1).bool().unsqueeze(-1)&target_mask.flatten(1).bool().gather(1,inds.reshape(b,-1)).reshape_as(inds)
            bb=torch.arange(b,device=opt.device).view(b,1,1)
            keys=k.flatten(2).transpose(1,2)[bb,inds]
        logits=(q.flatten(2).transpose(1,2).unsqueeze(2)*keys).sum(-1)/self.temperature
        logits=logits.masked_fill(~valid,-1e4)
        peak=logits.argmax(-1)
        # Restrict continuous expectation to the 3x3 neighborhood of the selected peak.
        kval=offsets.shape[0]
        neighborhood=[]
        neighborhood_valid=[]
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                ny = peak // (2*r+1) + dy
                nx = peak % (2*r+1) + dx
                neighborhood_valid.append((ny >= 0) & (ny <= 2*r) & (nx >= 0) & (nx <= 2*r))
                # Clamp only to keep gather indices safe. The mask below excludes
                # out-of-window positions, so edge peaks are never counted twice.
                neighborhood.append(ny.clamp(0,2*r)*(2*r+1)+nx.clamp(0,2*r))
        select=torch.stack(neighborhood,-1)
        selected_logits=logits.gather(-1,select)
        selected_valid=valid.gather(-1,select) & torch.stack(neighborhood_valid,-1)
        probs=F.softmax(selected_logits.masked_fill(~selected_valid,-1e4),-1)*selected_valid.float()
        probs=probs/probs.sum(-1,keepdim=True).clamp_min(1e-8)
        chosen=coords.gather(2,select.unsqueeze(-1).expand(-1,-1,-1,2)).float()
        expected=(probs.unsqueeze(-1)*chosen).sum(-2)
        conf=(probs.amax(-1)*valid.any(-1).float()).view(b,1,h,w)
        point_flow=(expected-base).transpose(1,2).reshape(b,2,h,w)
        # Differentiable residual predictor, so high-res features receive gradients
        # even where discrete peak selection is non-differentiable.
        war=flow_warp(sar.float(),point_flow)
        raw=self.head(torch.cat((opt.float(),war,point_flow/max(h,w),conf),dim=1))
        gate=torch.sigmoid(raw[:,2:])*conf.detach()
        corrected=prior+(point_flow-prior)*conf.detach()+gate*raw[:,:2].tanh()*self.residual_radius
        corrected=torch.where(valid.any(-1).view(b,1,h,w),corrected,prior)
        return dict(flow=corrected,confidence=conf,local_gate=gate,subpixel_peak=peak,
                    subpixel_logits=logits,subpixel_indices=inds,subpixel_valid=valid,
                    subpixel_coordinates=coords.float())


def fit_similarity_from_flow(flow, weight, grid_step=2):
    """Differentiable weighted 4-DoF Sim(2) fit in the current flow's pixel coordinates."""
    b,_,h,w=flow.shape
    yy,xx=torch.meshgrid(torch.arange(0,h,grid_step,device=flow.device,dtype=flow.dtype),torch.arange(0,w,grid_step,device=flow.device,dtype=flow.dtype),indexing='ij')
    pts=torch.stack((xx,yy),-1).reshape(1,-1,2).expand(b,-1,-1)
    destinations=pts+flow[:,:,::grid_step,::grid_step].flatten(2).transpose(1,2)
    weights=weight[:,:,::grid_step,::grid_step].flatten(1).float().detach().clamp_min(0)
    denom=weights.sum(-1,keepdim=True).clamp_min(1e-8)
    xs=(pts*weights[...,None]).sum(1)/denom
    ys=(destinations*weights[...,None]).sum(1)/denom
    p=pts-xs[:,None];q=destinations-ys[:,None]
    d=(weights*p.square().sum(-1)).sum(-1).clamp_min(1e-8)
    a=(weights*(p[...,0]*q[...,0]+p[...,1]*q[...,1])).sum(-1)/d
    c=(weights*(p[...,0]*q[...,1]-p[...,1]*q[...,0])).sum(-1)/d
    rotation=torch.stack((torch.stack((a,-c),-1),torch.stack((c,a),-1)),-2)
    shift=ys-torch.einsum('bij,bj->bi',rotation,xs)
    matrix=torch.cat((rotation,shift.unsqueeze(-1)),-1)
    valid=(weights>1e-5).sum(-1)>=6
    identity=torch.eye(2,3,device=flow.device,dtype=flow.dtype).unsqueeze(0).expand(b,-1,-1)
    matrix=torch.where(valid[:,None,None],matrix,identity)
    fitted=affine_to_flow(matrix,(h,w),normalized=False)
    return matrix,fitted,valid


def fit_affine_from_flow(flow, weight, fallback_matrix, grid_step=2,
                         irls_iterations=2, huber_delta_px=3.0, *, full_size=None,
                         balance_grid=1, balance_max_gain=4.0, return_diagnostics=False):
    """Fit six affine DoF in this flow's pixel units, retaining a coarse fallback.

    Coordinates are centered/scaled before the 3x3 solve. Confidence and robust
    weights are detached so the network cannot reduce supervision by suppressing
    them; gradients still pass from the fitted matrix to the predicted flow.
    Singular spatial support is rejected BEFORE solving, including collinear
    masks. All linear algebra remains FP32 inside an outer AMP context.
    Matrices and fitted flow use the input grid; full_size sets physical Huber
    units. Spatial balancing caps amplification of low-confidence cells.
    """
    if grid_step < 1 or irls_iterations < 0 or huber_delta_px <= 0:
        raise ValueError('Invalid affine fitting stride, iterations or Huber delta')
    if balance_grid < 1 or balance_max_gain < 1:
        raise ValueError('Invalid spatial balance grid or gain limit')
    with torch.autocast(device_type=flow.device.type, enabled=False):
        flow = flow.float()
        b, _, h, w = flow.shape
        yy, xx = torch.meshgrid(
            torch.arange(0, h, grid_step, device=flow.device, dtype=torch.float32),
            torch.arange(0, w, grid_step, device=flow.device, dtype=torch.float32),
            indexing='ij',
        )
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        sx, sy = max(cx, 1.0), max(cy, 1.0)
        design = torch.stack(((xx - cx) / sx, (yy - cy) / sy, torch.ones_like(xx)), -1).reshape(-1, 3)
        values = flow[:, :, ::grid_step, ::grid_step].flatten(2).transpose(1, 2)
        mass = weight[:, :, ::grid_step, ::grid_step].flatten(1).float().detach()
        finite = torch.isfinite(values).all(-1) & torch.isfinite(mass)
        mass = torch.where(finite, mass.clamp_min(0), torch.zeros_like(mass))
        values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        gx = (xx.flatten() / max(w - 1, 1) * balance_grid).long().clamp(0, balance_grid - 1)
        gy = (yy.flatten() / max(h - 1, 1) * balance_grid).long().clamp(0, balance_grid - 1)
        cells = (gy * balance_grid + gx).unsqueeze(0).expand(b, -1)
        cell_mass = mass.new_zeros(b, balance_grid ** 2).scatter_add(1, cells, mass)
        occupied = (cell_mass > 1.e-8).sum(-1)
        if balance_grid > 1:
            typical = cell_mass.sum(-1, keepdim=True) / occupied[:, None].clamp_min(1)
            local_mass = cell_mass.gather(1, cells)
            gain = (typical / local_mass.clamp_min(1.e-12)).clamp_max(balance_max_gain)
            mass = mass * gain
        full_h, full_w = (h, w) if full_size is None else full_size
        pixel_scale = flow.new_tensor((max(full_w - 1, 1) / max(w - 1, 1),
                                       max(full_h - 1, 1) / max(h - 1, 1)))
        eye = torch.eye(3, device=flow.device, dtype=torch.float32).unsqueeze(0)

        def solve(weights):
            normalized = weights / weights.sum(-1, keepdim=True).clamp_min(1.e-12)
            normal = torch.einsum('ni,bn,nj->bij', design, normalized, design)
            rhs = torch.einsum('ni,bn,bnk->bik', design, normalized, values)
            eigenvalues = torch.linalg.eigvalsh(normal.detach())
            valid = ((weights > 1.e-8).sum(-1) >= 6) & torch.isfinite(eigenvalues).all(-1)
            valid = valid & (eigenvalues[:, 0] > 1.e-6)
            valid = valid & (eigenvalues[:, -1] / eigenvalues[:, 0].clamp_min(1.e-12) < 1.e6)
            safe_normal = torch.where(valid[:, None, None], normal, eye)
            safe_rhs = torch.where(valid[:, None, None], rhs, torch.zeros_like(rhs))
            theta = torch.linalg.solve(safe_normal, safe_rhs)
            return theta, valid & torch.isfinite(theta).all(dim=(-2, -1))

        theta, valid = solve(mass)
        for _ in range(irls_iterations):
            predicted = torch.einsum('ni,bik->bnk', design, theta)
            distance = torch.linalg.vector_norm((predicted - values) * pixel_scale, dim=-1)
            robust = (huber_delta_px / distance.clamp_min(1.e-8)).clamp_max(1.0).detach()
            theta, current_valid = solve(mass * robust)
            valid = valid & current_valid
        # theta predicts DISPLACEMENT, so add the identity linear component.
        row_x = torch.stack((1.0 + theta[:, 0, 0] / sx,
                             theta[:, 1, 0] / sy,
                             theta[:, 2, 0] - theta[:, 0, 0] * cx / sx - theta[:, 1, 0] * cy / sy), -1)
        row_y = torch.stack((theta[:, 0, 1] / sx,
                             1.0 + theta[:, 1, 1] / sy,
                             theta[:, 2, 1] - theta[:, 0, 1] * cx / sx - theta[:, 1, 1] * cy / sy), -1)
        matrix = torch.stack((row_x, row_y), 1)
        matrix = torch.where(valid[:, None, None], matrix, fallback_matrix.float())
        fitted = affine_to_flow(matrix, (h, w), normalized=False)
    if return_diagnostics:
        return matrix, fitted, valid, {
            'geometry_fit_effective_points': (mass > 1.e-8).sum(-1).float(),
            'geometry_fit_spatial_coverage': occupied.float() / float(balance_grid ** 2),
        }
    return matrix, fitted, valid


def affine_global_residual_output(raw_flow, confidence, source_mask, target_mask,
                                 coarse_matrix_pixel, *, grid_step=4,
                                 residual_max_px=1.0, residual_smoothing_kernel=5,
                                  fit_irls_iterations=2, fit_huber_delta_px=3.0,
                                  balance_grid=1, balance_max_gain=4.0):
    """Global affine output plus a masked, smooth, bounded local correction.

    Everything is evaluated in FULL-resolution pixels, including the Euclidean
    residual limit and Huber threshold. Failed fits retain the coarse affine and
    receive no local correction rather than replacing valid motion by identity.
    This projection adds no trainable parameters or persistent buffers.
    """
    if residual_max_px < 0:
        raise ValueError('affine_output.residual_max_px must be nonnegative')
    if residual_smoothing_kernel < 1 or residual_smoothing_kernel % 2 != 1:
        raise ValueError('affine_output.residual_smoothing_kernel must be a positive odd integer')
    with torch.autocast(device_type=raw_flow.device.type, enabled=False):
        raw_flow = raw_flow.float()
        size = raw_flow.shape[-2:]
        source = F.interpolate(source_mask.float(), size=size, mode='nearest') > .5
        target = F.interpolate(target_mask.float(), size=size, mode='nearest')
        conf = F.interpolate(confidence.float(), size=size, mode='bilinear', align_corners=True).detach().clamp(0, 1)
        raw_target_valid = flow_warp(target, raw_flow) > .5
        finite = torch.isfinite(raw_flow).all(1, keepdim=True)
        fitting_weight = conf * source.float() * raw_target_valid.float() * finite.float()
        matrix, base, fit_valid, fit_diagnostics = fit_affine_from_flow(
            raw_flow, fitting_weight, coarse_matrix_pixel, grid_step,
            fit_irls_iterations, fit_huber_delta_px,
            balance_grid=balance_grid, balance_max_gain=balance_max_gain,
            return_diagnostics=True,
        )
        # Normalized masked smoothing prevents black borders from influencing
        # neighboring valid corrections. Both raw and affine destinations must
        # lie in SAR content, and the final destination is checked once more.
        valid = source & raw_target_valid & (flow_warp(target, base) > .5)
        valid = valid & finite & fit_valid.view(-1, 1, 1, 1)
        residual = torch.nan_to_num(raw_flow - base) * valid.float()
        kernel = residual_smoothing_kernel
        if kernel > 1:
            support = F.avg_pool2d(valid.float(), kernel, stride=1, padding=kernel // 2)
            residual = F.avg_pool2d(residual, kernel, stride=1, padding=kernel // 2) / support.clamp_min(1.e-8)
        if residual_max_px == 0:
            residual = torch.zeros_like(residual)
        else:
            # Radial tanh bounds the VECTOR norm, not each component separately.
            norm = torch.linalg.vector_norm(residual, dim=1, keepdim=True).clamp_min(1.e-6)
            residual = residual * (residual_max_px * torch.tanh(norm / residual_max_px) / norm)
        residual = residual * valid.float()
        valid = valid & (flow_warp(target, base + residual) > .5)
        residual = residual * valid.float()
        final = base + residual
    return dict(flow=final, geometry_projected_flow=base,
                affine_matrix_pixel=matrix, affine_flow=base,
                residual_flow=residual, final_output_residual=residual,
                raw_dense_flow=raw_flow, geometry_fit_valid=fit_valid,
                final_output_valid=valid, **fit_diagnostics)


def progressive_geometry_update(stage, source_mask, target_mask, full_size,
                                fallback_matrix_full, config, *, constrain_residual=True):
    """Fit a global transform from raw stage evidence; emit the next stage prior.

    Matrices returned in full pixels can be reused as the next fit's fallback.
    Huber and residual radii use full pixels even on rectangular feature grids.
    The last stage only updates geometry: its raw evidence is projected once at
    full resolution, with the original content masks, by the final output layer.
    """
    cap = float(config.get('residual_max_px', 1.0))
    kernel_full = int(config.get('residual_smoothing_kernel', 3))
    if cap < 0 or kernel_full < 1 or kernel_full % 2 != 1:
        raise ValueError('Invalid progressive_geometry residual limit or smoothing kernel')
    raw = stage['flow']
    size = raw.shape[-2:]
    with torch.autocast(device_type=raw.device.type, enabled=False):
        finite = torch.isfinite(raw).all(1, keepdim=True)
        raw = torch.nan_to_num(raw.float(), nan=0.0, posinf=0.0, neginf=0.0)
        source = source_mask.bool()
        target = target_mask.float()
        mapped = flow_warp(target, raw) > .5
        weight = stage['confidence'].float().detach() * source.float() * mapped.float() * finite.float()
        fallback_stage = normalized_affine_to_pixel(
            pixel_affine_to_normalized(fallback_matrix_full, full_size), size)
        matrix, base, fit_valid, diagnostics = fit_affine_from_flow(
            raw, weight, fallback_stage,
            grid_step=int(config.get('fit_grid_step', 1)),
            irls_iterations=int(config.get('fit_irls_iterations', 2)),
            huber_delta_px=float(config.get('fit_huber_delta_px', 3.0)),
            full_size=full_size, balance_grid=int(config.get('balance_grid', 4)),
            balance_max_gain=float(config.get('balance_max_gain', 4.0)),
            return_diagnostics=True,
        )
        matrix_full = normalized_affine_to_pixel(pixel_affine_to_normalized(matrix, size), full_size)
        # Preserve the exact most recent successful matrix on failed fits.
        matrix_full = torch.where(fit_valid[:, None, None], matrix_full, fallback_matrix_full.float())
        residual = torch.zeros_like(raw)
        if constrain_residual and cap > 0:
            valid = source & mapped & finite & (flow_warp(target, base) > .5)
            valid = valid & fit_valid.view(-1, 1, 1, 1)
            residual = (raw - base) * valid.float()
            h, w = size
            fh, fw = full_size
            scale_x = max(fw - 1, 1) / max(w - 1, 1)
            scale_y = max(fh - 1, 1) / max(h - 1, 1)
            scale = raw.new_tensor((scale_x, scale_y)).view(1, 2, 1, 1)
            # Convert the physical smoothing radius to each feature-grid axis.
            rx = round((kernel_full - 1) * .5 / scale_x)
            ry = round((kernel_full - 1) * .5 / scale_y)
            if rx > 0 or ry > 0:
                support = F.avg_pool2d(valid.float(), (2 * ry + 1, 2 * rx + 1),
                                       stride=1, padding=(ry, rx))
                residual = F.avg_pool2d(residual, (2 * ry + 1, 2 * rx + 1),
                                        stride=1, padding=(ry, rx)) / support.clamp_min(1.e-8)
            norm_full = torch.linalg.vector_norm(residual * scale, dim=1, keepdim=True).clamp_min(1.e-6)
            residual = residual * (cap * torch.tanh(norm_full / cap) / norm_full)
            valid = valid & (flow_warp(target, base + residual) > .5)
            residual = residual * valid.float()
    return dict(global_prior_flow=base + residual, geometry_flow=base,
                global_affine_matrix_pixel=matrix_full, geometry_fit_valid=fit_valid,
                **diagnostics)


class SGUFlowRefiner(nn.Module):
    def __init__(self,c8,c4,c2,config):
        super().__init__()
        self.s8=SparseUOTFlowStage(c8,config['uot_flow'],int(config['uot_flow'].get('radius_s8',3)))
        self.s4=SparseUOTFlowStage(c4,config['uot_flow'],int(config['uot_flow'].get('radius_s4',2)))
        self.s2=SubpixelHalfStage(c2,config.get('subpixel',{}))
        self.grid_step=int(config.get('geometry_fit_stride',2))
        self.output_mode=str(config.get('output_mode','dense'))
        if self.output_mode not in {'dense','affine_global_residual'}:
            raise ValueError(f'Unsupported sgu_flow.output_mode: {self.output_mode}')
        self.affine_output=dict(config.get('affine_output',{}))
        self.progressive_geometry = dict(config.get('progressive_geometry', {}))
        self.progressive_enabled = bool(self.progressive_geometry.get('enabled', False))
        if self.progressive_enabled and self.output_mode != 'affine_global_residual':
            raise ValueError('progressive_geometry requires affine_global_residual output')

    def forward(self,opt8,sar8,opt4,sar4,opt2,sar2,coarse_matrix,coarse_conf,
                mask8o,mask8s,mask4o,mask4s,mask2o,mask2s,full_size,
                full_source_mask=None,full_target_mask=None):
        # Coarse geometry is normalized by the coarse feature grid; convert first to
        # full-resolution pixel coordinates and then to each pyramid scale via flow resize.
        full_initial=affine_to_flow(coarse_matrix,full_size,normalized=True)
        initial=resize_flow(full_initial,opt8.shape[-2:])
        latest_matrix = normalized_affine_to_pixel(coarse_matrix, full_size)
        r8=self.s8(opt8,sar8,initial,mask8o,mask8s)
        if self.progressive_enabled:
            r8.update(progressive_geometry_update(r8, mask8o, mask8s, full_size,
                                                  latest_matrix, self.progressive_geometry))
            latest_matrix = r8['global_affine_matrix_pixel']
        r4=self.s4(opt4,sar4,r8.get('global_prior_flow', r8['flow']),mask4o,mask4s)
        if self.progressive_enabled:
            r4.update(progressive_geometry_update(r4, mask4o, mask4s, full_size,
                                                  latest_matrix, self.progressive_geometry))
            latest_matrix = r4['global_affine_matrix_pixel']
        r2=self.s2(opt2,sar2,r4.get('global_prior_flow', r4['flow']),mask2o,mask2s)
        if self.progressive_enabled:
            r2.update(progressive_geometry_update(r2, mask2o, mask2s, full_size,
                                                  latest_matrix, self.progressive_geometry,
                                                  constrain_residual=False))
            latest_matrix = r2['global_affine_matrix_pixel']
        flow=resize_flow(r2['flow'],full_size)
        # Retain the legacy Sim(2) diagnostic in dense mode. The affine-global
        # branch below replaces this diagnostic and the emitted final output.
        mapped_target_valid = flow_warp(mask2s.float(), r2['flow']).clamp(0, 1)
        weight = r2['confidence'] * mask2o.float() * mapped_target_valid
        matrix2,projected2,fit_valid=fit_similarity_from_flow(r2['flow'],weight,self.grid_step)
        projected=resize_flow(projected2,full_size)
        # The matrix is in 1/2-resolution pixel units, so convert via correspondence
        # mapping through full-sized flow, rather than mislabel as a full-pixel matrix.
        full_matrix,_,_=fit_similarity_from_flow(flow,F.interpolate(weight,size=full_size,mode='nearest'),self.grid_step*2)
        output = dict(flow=flow,flows=[r8['flow'],r4['flow'],r2['flow']],
                    stage_results=[r8,r4,r2],confidence=r2['confidence'],geometry_projected_flow=projected,
                    geometry_matrix_half=matrix2,affine_matrix_pixel=full_matrix,
                    affine_flow=projected,residual_flow=flow-projected,
                    geometry_fit_valid=fit_valid,affine_matrix_initial=normalized_affine_to_pixel(coarse_matrix,full_size),
                    initial_affine_flow=full_initial,
                    local_residuals=[r8['flow']-resize_flow(full_initial,r8['flow'].shape[-2:]),
                                     r4['flow']-resize_flow(full_initial,r4['flow'].shape[-2:]),
                                     r2['flow']-resize_flow(full_initial,r2['flow'].shape[-2:])],
                    local_uot_flows=[r8['local_uot_flow'],r4['local_uot_flow']],
                    local_uot_confidences=[r8['confidence'],r4['confidence']],
                    graph_cost_weights=torch.stack((r8['local_cost_weights'],r4['local_cost_weights'])))
        if self.output_mode == 'affine_global_residual':
            output.update(affine_global_residual_output(
                flow, r2['confidence'],
                mask2o if full_source_mask is None else full_source_mask,
                mask2s if full_target_mask is None else full_target_mask,
                latest_matrix, grid_step=self.grid_step * 2,
                **self.affine_output,
            ))
            normalized = pixel_affine_to_normalized(output['affine_matrix_pixel'], full_size)
            output['geometry_matrix_half'] = normalized_affine_to_pixel(normalized, r2['flow'].shape[-2:])
            output['output_mode'] = self.output_mode
        return output
