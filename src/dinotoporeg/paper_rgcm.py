"""Spatial RGCM from SURGE Section III-C, equations (11)-(13).

P0 is a masked, row-normalized soft assignment. Equation (12) sums over
ALL target neighbours for every source neighbour, then divides only by
the number of valid source neighbours. It is not same-offset consistency.
"""
from __future__ import annotations

import torch


def neighborhood_support(logits, candidates, graph, source_valid, target_valid):
    """Return B,N,K support and eligibility, with neutral isolated-node scores.

    The separable sums implement the two independent neighbourhood sums
    without materializing a B,N,K,8,8 tensor. Invalid rows/columns have zero
    probability; all-invalid and isolated nodes produce no relational loss.
    """
    logits = logits.float()
    batch, count, _ = logits.shape
    neighbors = graph["neighbor_indices"]
    edges = graph["neighbor_valid"]
    source_neighbors_valid = edges.unsqueeze(0) & source_valid[:, neighbors]
    target_neighbors_valid = edges.unsqueeze(0) & target_valid[:, neighbors]
    masked = logits.masked_fill(~target_valid[:, None, :], torch.finfo(logits.dtype).min)
    probabilities = masked.softmax(-1)
    probabilities = probabilities * source_valid[:, :, None] * target_valid[:, None, :]
    probabilities = probabilities / probabilities.sum(-1, keepdim=True).clamp_min(1.e-8)

    source_count = source_neighbors_valid.sum(-1)
    averaged = torch.zeros_like(probabilities)
    for edge in range(neighbors.shape[-1]):
        averaged = averaged + probabilities[:, neighbors[:, edge], :] * source_neighbors_valid[:, :, edge, None]
    averaged = averaged / source_count.clamp_min(1).unsqueeze(-1)

    support = logits.new_zeros(candidates.shape)
    target_count = torch.zeros_like(candidates)
    batch_index = torch.arange(batch, device=logits.device)[:, None, None]
    for edge in range(neighbors.shape[-1]):
        indices = neighbors[:, edge][candidates]
        valid_edge = target_neighbors_valid[batch_index, candidates, edge]
        support = support + averaged.gather(-1, indices) * valid_edge
        target_count = target_count + valid_edge.long()
    eligible = (source_valid[:, :, None] & target_valid.gather(1, candidates.flatten(1)).reshape_as(candidates)
                & (source_count[:, :, None] > 0) & (target_count > 0))
    support = torch.where(eligible, support, torch.ones_like(support))
    return support.clamp(min=1.e-8, max=1.0), eligible


def validate_paper_config(config):
    """Fail on settings that silently change the named paper method."""
    if config.get("methodology") != "surge_paper_v1":
        return
    coarse = config["model"]["coarse"]
    fine = config["model"]["fine"]
    sgu = fine["sgu_flow"]
    required = {
        "spatial_only": True, "relation_support_mode": "neighborhood_probability",
        "graph_neighbors": 8, "pre_topk": 24, "graph_topk": 8,
        "graph_enabled": True, "uot_enabled": True, "uot_iterations": 12,
        "dustbin_enabled": False,
    }
    for key, expected in required.items():
        if coarse.get(key) != expected:
            raise ValueError(f"Paper RGCM requires coarse.{key}={expected!r}")
    if fine["mode"] != "sgu_flow" or not sgu["progressive_geometry"]["enabled"]:
        raise ValueError("Paper LUFR requires SGU-flow with progressive affine feedback")
    if sgu["subpixel"]["sampling_mode"] != "continuous":
        raise ValueError("Paper subpixel stage requires continuous coordinate sampling")
    if sgu["output_mode"] != "affine_global_residual" or sgu["affine_output"]["residual_max_px"] != 0:
        raise ValueError("Paper output must be a pure six-DoF affine field")
    if config["loss"].get("objective_mode") != "paper_four_terms":
        raise ValueError("Paper training requires the four-term objective")
    if sgu["uot_flow"].get("dustbin_enabled", True):
        raise ValueError("Paper local UOT must use marginal relaxation without dustbin nodes")
    if (sgu['uot_flow']['radius_s8'], sgu['uot_flow']['radius_s4'], sgu['subpixel']['radius']) != (3, 2, 2):
        raise ValueError("Paper search windows must be 7x7, 5x5 and 5x5")
    for key, expected in dict(w_match=.05, w_local_uot=.1, w_total_flow=1., w_affine_corner=.1).items():
        if config['loss'].get(key) != expected:
            raise ValueError(f"Paper Eq. (27) requires loss.{key}={expected}")


def sparse_unbalanced_transport(logits, indices, valid, source, target, epsilon, tau_source, tau_target, iterations):
    """KL-relaxed UOT on the specified edges, without auxiliary dustbin nodes.

    Matches the N x M problem in Section III-C. Disconnected nodes have zero
    transported mass. Finite masked reductions avoid NaNs in their gradients.
    """
    with torch.autocast(device_type=logits.device.type, enabled=False):
        logits, source, target = logits.float(), source.float(), target.float()
        batch, count, _ = logits.shape
        target_count = target.shape[-1]
        valid = valid & (source[:, :, None] > 0) & (target.gather(1, indices.flatten(1)).reshape_as(indices) > 0)
        row_active = valid.any(-1)
        kernel = logits / epsilon
        log_a, log_b = source.clamp_min(1.e-8).log(), target.clamp_min(1.e-8).log()
        log_u, log_v = torch.zeros_like(source), torch.zeros_like(target)

        def scatter_sum(values):
            return values.new_zeros((batch, target_count)).scatter_add(1, indices.flatten(1), values.flatten(1))

        def column_logsumexp(values):
            masked = values.masked_fill(~valid, -1.e30)
            maxima = values.new_full((batch, target_count), -1.e30).scatter_reduce(
                1, indices.flatten(1), masked.flatten(1), reduce="amax", include_self=True,
            ).detach()
            shifts = torch.where(valid, values - maxima.gather(1, indices.flatten(1)).reshape_as(values), torch.zeros_like(values))
            sums = scatter_sum(shifts.exp() * valid)
            return torch.where(sums > 0, maxima + sums.clamp_min(1.e-30).log(), torch.zeros_like(sums)), sums > 0

        for _ in range(iterations):
            gathered_v = log_v.gather(1, indices.flatten(1)).reshape_as(kernel)
            row_lse = torch.logsumexp((kernel + gathered_v).masked_fill(~valid, -1.e30), -1)
            log_u = torch.where(row_active, tau_source / (tau_source + epsilon) * (log_a - row_lse), torch.zeros_like(log_a))
            column_lse, column_active = column_logsumexp(kernel + log_u[:, :, None])
            log_v = torch.where(column_active, tau_target / (tau_target + epsilon) * (log_b - column_lse), torch.zeros_like(log_b))
        gathered_v = log_v.gather(1, indices.flatten(1)).reshape_as(kernel)
        log_plan = kernel + log_u[:, :, None] + gathered_v
        edge = log_plan.masked_fill(~valid, -1.e30).exp() * valid
        row_mass, column_mass = edge.sum(-1), scatter_sum(edge)
        conditional = edge / row_mass[:, :, None].clamp_min(1.e-8)
        entropy = -(conditional * conditional.clamp_min(1.e-8).log()).sum(-1)
        return dict(edge_transport=edge, row_transport_mass=row_mass, column_transport_mass=column_mass,
                    unmatched_source_mass=(source - row_mass).clamp_min(0),
                    unmatched_target_mass=(target - column_mass).clamp_min(0), transport_entropy=entropy)
