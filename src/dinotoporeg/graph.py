from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels: int) -> int:
    for value in (8, 4, 2, 1):
        if channels % value == 0:
            return value
    return 1


class OpticalNodeAdapter(nn.Module):
    """Modality-specific residual branch initialized as an exact no-op."""

    def __init__(self, channels: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, 1, bias=False),
            nn.GroupNorm(_groups(hidden_dim), hidden_dim),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_dim, channels, 1, bias=False),
        )
        # Both modalities start in exactly the same shared relation space. The
        # residual branches learn only the modality correction that is needed.
        nn.init.zeros_(self.body[-1].weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.body(features)


class SARNodeAdapter(OpticalNodeAdapter):
    pass


class SharedNodeProjection(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, 1, bias=False),
            nn.GroupNorm(_groups(output_dim), output_dim),
            nn.GELU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.projection(features)


class GridGraphBuilder(nn.Module):
    """Create a strict 8-neighbour graph without padded fake neighbours."""

    def __init__(self, neighbors: int = 8, semantic_k: int = 0):
        super().__init__()
        if neighbors != 8:
            raise ValueError("The first graph implementation supports exactly 8 spatial neighbours")
        if semantic_k != 0:
            raise NotImplementedError("semantic_k is reserved but must remain 0 in this implementation")
        self.neighbors = neighbors
        self.semantic_k = semantic_k

    def forward(self, height: int, width: int, device: torch.device | None = None) -> dict[str, torch.Tensor]:
        device = device or torch.device("cpu")
        yy, xx = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        )
        offsets = torch.tensor(
            ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)),
            device=device,
        )
        nx = xx.reshape(-1, 1) + offsets[:, 0]
        ny = yy.reshape(-1, 1) + offsets[:, 1]
        valid = (nx >= 0) & (nx < width) & (ny >= 0) & (ny < height)
        indices = (ny.clamp(0, height - 1) * width + nx.clamp(0, width - 1)).long()
        relative = offsets.to(torch.float32).unsqueeze(0).expand(height * width, -1, -1).clone()
        relative[..., 0] /= max(width - 1, 1)
        relative[..., 1] /= max(height - 1, 1)
        relative = relative * valid.unsqueeze(-1)
        coords = torch.stack(
            (
                2.0 * xx.float() / max(width - 1, 1) - 1.0,
                2.0 * yy.float() / max(height - 1, 1) - 1.0,
            ),
            dim=-1,
        ).reshape(-1, 2)
        return {
            "neighbor_indices": indices,
            "neighbor_valid": valid,
            "relative_offsets": relative,
            "coordinates": coords,
        }


class HybridDynamicGraphBuilder(nn.Module):
    """Union of grid, semantic and CSLE-learned structural neighbours."""

    def __init__(
        self,
        spatial_neighbors: int = 8,
        semantic_k: int = 4,
        structure_k: int = 4,
    ):
        super().__init__()
        if spatial_neighbors != 8:
            raise ValueError("HybridDynamicGraphBuilder supports the strict 8-neighbour grid")
        if semantic_k < 0 or structure_k < 0:
            raise ValueError("semantic_k and structure_k must be non-negative")
        self.spatial_neighbors = int(spatial_neighbors)
        self.semantic_k = int(semantic_k)
        self.structure_k = int(structure_k)
        self.spatial_builder = GridGraphBuilder(spatial_neighbors, 0)

    @staticmethod
    def _masked_topk(
        similarity: torch.Tensor, excluded: torch.Tensor, count: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, nodes, available = similarity.shape
        if count == 0:
            return (
                torch.empty((batch, nodes, 0), dtype=torch.long, device=similarity.device),
                torch.empty((batch, nodes, 0), dtype=torch.bool, device=similarity.device),
            )
        actual = min(count, available)
        masked = similarity.masked_fill(excluded, -torch.inf)
        # stable=True gives deterministic lower-index tie breaking.
        order = torch.argsort(masked, dim=-1, descending=True, stable=True)[..., :actual]
        valid = torch.isfinite(masked.gather(-1, order))
        if actual < count:
            order = F.pad(order, (0, count - actual), value=0)
            valid = F.pad(valid, (0, count - actual), value=False)
        return order, valid

    @staticmethod
    def _mark_excluded(
        excluded: torch.Tensor, indices: torch.Tensor, valid: torch.Tensor
    ) -> None:
        batch, nodes, slots = indices.shape
        batch_index = torch.arange(batch, device=indices.device).view(batch, 1, 1).expand(-1, nodes, slots)
        node_index = torch.arange(nodes, device=indices.device).view(1, nodes, 1).expand(batch, -1, slots)
        excluded[batch_index[valid], node_index[valid], indices[valid]] = True

    def forward(
        self,
        nodes: torch.Tensor,
        structure: torch.Tensor,
        height: int,
        width: int,
        node_valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if nodes.ndim != 3 or structure.ndim != 3:
            raise ValueError("nodes and structure must have shape [B,N,C]")
        batch, count, _ = nodes.shape
        if count != height * width or structure.shape[:2] != (batch, count):
            raise ValueError("node count does not match height * width")
        if node_valid is None:
            node_valid = torch.ones((batch, count), dtype=torch.bool, device=nodes.device)
        elif node_valid.shape != (batch, count):
            raise ValueError("node_valid must have shape [B,N]")
        else:
            node_valid = node_valid.to(device=nodes.device, dtype=torch.bool)

        spatial = self.spatial_builder(height, width, nodes.device)
        spatial_indices = spatial["neighbor_indices"].unsqueeze(0).expand(batch, -1, -1)
        spatial_valid = spatial["neighbor_valid"].unsqueeze(0).expand(batch, -1, -1)
        spatial_target_valid = node_valid.gather(
            1, spatial_indices.reshape(batch, -1)
        ).reshape_as(spatial_indices)
        spatial_valid = (
            spatial_valid
            & node_valid.unsqueeze(-1)
            & spatial_target_valid
        )

        excluded = (
            torch.eye(count, dtype=torch.bool, device=nodes.device)
            .unsqueeze(0)
            .expand(batch, -1, -1)
            .clone()
        )
        # Invalid patches can be neither edge sources nor neighbour candidates.
        excluded |= ~node_valid.unsqueeze(-1)
        excluded |= ~node_valid.unsqueeze(1)
        self._mark_excluded(excluded, spatial_indices, spatial_valid)

        semantic_nodes = F.normalize(nodes.float(), dim=-1)
        semantic_similarity = torch.matmul(semantic_nodes, semantic_nodes.transpose(-1, -2))
        semantic_indices, semantic_valid = self._masked_topk(
            semantic_similarity, excluded, self.semantic_k
        )
        self._mark_excluded(excluded, semantic_indices, semantic_valid)

        structure_nodes = F.normalize(structure.float(), dim=-1)
        structure_similarity = torch.matmul(structure_nodes, structure_nodes.transpose(-1, -2))
        structure_indices, structure_valid = self._masked_topk(
            structure_similarity, excluded, self.structure_k
        )

        indices = torch.cat((spatial_indices, semantic_indices, structure_indices), dim=-1)
        valid = torch.cat((spatial_valid, semantic_valid, structure_valid), dim=-1)
        edge_types = torch.cat(
            (
                torch.zeros_like(spatial_indices),
                torch.ones_like(semantic_indices),
                torch.full_like(structure_indices, 2),
            ),
            dim=-1,
        )
        coordinates = spatial["coordinates"]
        safe_indices = torch.where(valid, indices, torch.zeros_like(indices))
        relative = coordinates[safe_indices] - coordinates.view(1, count, 1, 2)
        relative = relative * valid.unsqueeze(-1).to(relative.dtype)
        return {
            "neighbor_indices": safe_indices,
            "neighbor_valid": valid,
            "relative_offsets": relative,
            "edge_types": edge_types,
            "coordinates": coordinates,
        }


class EdgeAwareGraphAttentionLayer(nn.Module):
    """Sparse multi-head graph attention with a near-identity gated residual."""

    def __init__(
        self,
        channels: int,
        heads: int = 4,
        edge_dim: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        if channels % heads != 0:
            raise ValueError("GAT channels must be divisible by heads")
        self.channels = int(channels)
        self.heads = int(heads)
        self.head_dim = channels // heads
        self.dropout = float(dropout)
        self.pre_norm = nn.LayerNorm(channels)
        self.query = nn.Linear(channels, channels, bias=False)
        self.key = nn.Linear(channels, channels, bias=False)
        self.value = nn.Linear(channels, channels, bias=False)
        self.edge_type = nn.Embedding(3, edge_dim)
        self.offset_encoder = nn.Sequential(
            nn.Linear(2, edge_dim), nn.GELU(), nn.Linear(edge_dim, edge_dim)
        )
        self.edge_key = nn.Linear(edge_dim, channels, bias=False)
        self.edge_value = nn.Linear(edge_dim, channels, bias=False)
        self.edge_bias = nn.Linear(edge_dim, heads, bias=False)
        self.output = nn.Linear(channels, channels)
        self.gate = nn.Linear(2 * channels, channels)
        self.ffn_norm = nn.LayerNorm(channels)
        hidden = max(channels * 2, 64)
        self.ffn = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, channels),
        )
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -4.0)
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)

    def forward(
        self,
        nodes: torch.Tensor,
        neighbor_indices: torch.Tensor,
        neighbor_valid: torch.Tensor,
        edge_types: torch.Tensor,
        relative_offsets: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch, count, channels = nodes.shape
        if channels != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {channels}")
        valid = neighbor_valid.bool()
        if ((edge_types < 0) | (edge_types > 2))[valid].any():
            raise ValueError("valid edge type indices must be in [0, 2]")
        safe_indices = neighbor_indices.long().clamp(0, max(count - 1, 0))
        safe_types = torch.where(valid, edge_types.long(), torch.zeros_like(edge_types.long()))
        normalized = self.pre_norm(nodes)
        batch_indices = torch.arange(batch, device=nodes.device).view(batch, 1, 1)
        neighbours = normalized[batch_indices, safe_indices]
        query = self.query(normalized).view(batch, count, self.heads, self.head_dim)
        key = self.key(neighbours).view(batch, count, -1, self.heads, self.head_dim)
        value = self.value(neighbours).view(batch, count, -1, self.heads, self.head_dim)
        edge = self.edge_type(safe_types) + self.offset_encoder(relative_offsets.to(normalized.dtype))
        key = key + self.edge_key(edge).view(batch, count, -1, self.heads, self.head_dim)
        value = value + self.edge_value(edge).view(batch, count, -1, self.heads, self.head_dim)
        scores = (query.unsqueeze(2) * key).sum(-1) / math.sqrt(self.head_dim)
        scores = scores + self.edge_bias(edge)
        attention = torch.softmax(scores.float().masked_fill(~valid.unsqueeze(-1), -1.0e4), dim=2)
        attention = attention * valid.unsqueeze(-1).float()
        attention = attention / attention.sum(dim=2, keepdim=True).clamp_min(1.0e-8)
        attention = F.dropout(attention, self.dropout, self.training).to(value.dtype)
        message = (attention.unsqueeze(-1) * value).sum(dim=2).reshape(batch, count, channels)
        message = self.output(message)
        gate = torch.sigmoid(self.gate(torch.cat((nodes, message), dim=-1)))
        updated = nodes + gate * message
        updated = updated + self.ffn(self.ffn_norm(updated))
        if return_attention:
            return updated, attention
        return updated


class ResidualGATEncoder(nn.Module):
    def __init__(
        self,
        channels: int,
        layers: int = 2,
        heads: int = 4,
        edge_dim: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EdgeAwareGraphAttentionLayer(channels, heads, edge_dim, dropout)
                for _ in range(int(layers))
            ]
        )

    def forward(self, nodes: torch.Tensor, graph: dict[str, torch.Tensor]) -> torch.Tensor:
        for layer in self.layers:
            nodes = layer(
                nodes,
                graph["neighbor_indices"],
                graph["neighbor_valid"],
                graph["edge_types"],
                graph["relative_offsets"],
            )
        return nodes


class EdgeRelationEncoder(nn.Module):
    def __init__(self, feature_dim: int, relation_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.node_projection = nn.Linear(feature_dim, relation_dim, bias=False)
        input_dim = 2 * relation_dim + 4
        hidden = max(relation_dim * 2, 64)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, relation_dim),
        )

    def encode_pairs(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
        relative_offsets: torch.Tensor,
    ) -> torch.Tensor:
        first = F.normalize(self.node_projection(first), dim=-1)
        second = F.normalize(self.node_projection(second), dim=-1)
        cosine = F.cosine_similarity(first, second, dim=-1).unsqueeze(-1)
        distance = torch.linalg.vector_norm(relative_offsets, dim=-1, keepdim=True)
        values = torch.cat((
            (first - second).abs(),
            first * second,
            cosine,
            relative_offsets,
            distance,
        ), dim=-1)
        return self.mlp(values)

    def forward(self, nodes: torch.Tensor, graph: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode local edges from nodes [B,N,C] to [B,N,Q,R]."""
        batch, count, channels = nodes.shape
        indices = graph["neighbor_indices"]
        neighbours = nodes[:, indices.reshape(-1)].reshape(batch, count, indices.shape[1], channels)
        centers = nodes.unsqueeze(2).expand_as(neighbours)
        offsets = graph["relative_offsets"].to(nodes).unsqueeze(0).expand(batch, -1, -1, -1)
        relations = self.encode_pairs(centers, neighbours, offsets)
        return relations * graph["neighbor_valid"].to(nodes).unsqueeze(0).unsqueeze(-1)


class UnbalancedSinkhornMatcher(nn.Module):
    """KL-relaxed log-domain Sinkhorn with explicit source/target dustbins."""

    def __init__(
        self,
        epsilon: float = 0.08,
        tau_source: float = 0.8,
        tau_target: float = 0.8,
        iterations: int = 12,
        dustbin_score: float = -1.0,
        dustbin_enabled: bool = True,
    ):
        super().__init__()
        if min(epsilon, tau_source, tau_target) <= 0:
            raise ValueError("UOT epsilon and tau values must be positive")
        self.epsilon = float(epsilon)
        self.tau_source = float(tau_source)
        self.tau_target = float(tau_target)
        self.iterations = int(iterations)
        self.dustbin_score = nn.Parameter(torch.tensor(float(dustbin_score)))
        self.dustbin_enabled = bool(dustbin_enabled)

    def forward(
        self,
        candidate_logits: torch.Tensor,
        candidate_indices: torch.Tensor,
        target_count: int,
        source_mass: torch.Tensor,
        target_mass: torch.Tensor,
        source_valid: torch.Tensor | None = None,
        target_valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if not self.dustbin_enabled:
            from .paper_rgcm import sparse_unbalanced_transport
            source_ok = source_mass > 0 if source_valid is None else source_valid.bool()
            target_ok = target_mass > 0 if target_valid is None else target_valid.bool()
            valid = source_ok[:, :, None] & target_ok.gather(1, candidate_indices.flatten(1)).reshape_as(candidate_indices)
            result = sparse_unbalanced_transport(
                candidate_logits, candidate_indices, valid, source_mass * source_ok, target_mass * target_ok,
                self.epsilon, self.tau_source, self.tau_target, self.iterations,
            )
            plan = result["edge_transport"].new_zeros((*candidate_indices.shape[:2], target_count))
            plan = plan.scatter_add(-1, candidate_indices, result["edge_transport"])
            result.update(transport=plan, transport_with_dustbin=F.pad(plan, (0, 1, 0, 1)))
            return result
        device_type = candidate_logits.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            logits = candidate_logits.float()
            indices = candidate_indices.long()
            source = torch.nan_to_num(source_mass.float(), nan=0.0)
            target = torch.nan_to_num(target_mass.float(), nan=0.0)
            batch, source_count, _ = logits.shape
            if source_valid is None:
                source_valid = torch.ones_like(source, dtype=torch.bool)
            else:
                source_valid = source_valid.bool()
            if target_valid is None:
                target_valid = torch.ones_like(target, dtype=torch.bool)
            else:
                target_valid = target_valid.bool()
            source = torch.where(source_valid, source, torch.zeros_like(source)).clamp_min(1.0e-8)
            target = torch.where(target_valid, target, torch.zeros_like(target)).clamp_min(1.0e-8)
            full_logits = logits.new_full((batch, source_count + 1, target_count + 1), -torch.inf)
            normal = logits.new_full((batch, source_count, target_count), -torch.inf)
            normal.scatter_(-1, indices, logits)
            normal = normal.masked_fill(
                ~(source_valid.unsqueeze(-1) & target_valid.unsqueeze(1)), -torch.inf
            )
            full_logits[:, :source_count, :target_count] = normal
            full_logits[:, :source_count, target_count] = self.dustbin_score.float()
            full_logits[:, source_count, :target_count] = self.dustbin_score.float()
            full_logits[:, source_count, target_count] = self.dustbin_score.float()

            source_total = source.sum(-1, keepdim=True)
            target_total = target.sum(-1, keepdim=True)
            a = torch.cat((source, target_total), dim=-1).clamp_min(1.0e-8)
            b = torch.cat((target, source_total), dim=-1).clamp_min(1.0e-8)
            log_a, log_b = a.log(), b.log()
            log_kernel = full_logits / self.epsilon
            log_u = torch.zeros_like(log_a)
            log_v = torch.zeros_like(log_b)
            rho_source = self.tau_source / (self.tau_source + self.epsilon)
            rho_target = self.tau_target / (self.tau_target + self.epsilon)
            for _ in range(self.iterations):
                log_kv = torch.logsumexp(log_kernel + log_v.unsqueeze(1), dim=-1)
                log_u = rho_source * (log_a - log_kv)
                log_ktu = torch.logsumexp(log_kernel + log_u.unsqueeze(-1), dim=-2)
                log_v = rho_target * (log_b - log_ktu)
            log_plan = log_kernel + log_u.unsqueeze(-1) + log_v.unsqueeze(1)
            plan = torch.exp(log_plan.clamp(max=30.0))
            plan = torch.nan_to_num(plan)
            transport = plan[:, :source_count, :target_count]
            return {
                "transport": transport.to(candidate_logits.dtype),
                "transport_with_dustbin": plan.to(candidate_logits.dtype),
                "unmatched_source_mass": plan[:, :source_count, target_count].to(candidate_logits.dtype),
                "unmatched_target_mass": plan[:, source_count, :target_count].to(candidate_logits.dtype),
                "row_transport_mass": transport.sum(-1).to(candidate_logits.dtype),
                "column_transport_mass": transport.sum(-2).to(candidate_logits.dtype),
            }


class SparseLocalUnbalancedSinkhornMatcher(nn.Module):
    """Log-domain KL-relaxed UOT operating only on a sparse [B,N,K] edge list."""

    def __init__(
        self,
        epsilon: float = 0.08,
        tau_source: float = 0.8,
        tau_target: float = 0.8,
        iterations: int = 8,
        dustbin_score: float = -1.0,
        dustbin_enabled: bool = True,
    ):
        super().__init__()
        if min(epsilon, tau_source, tau_target) <= 0:
            raise ValueError("UOT epsilon and tau values must be positive")
        self.epsilon = float(epsilon)
        self.tau_source = float(tau_source)
        self.tau_target = float(tau_target)
        self.iterations = int(iterations)
        self.dustbin_score = nn.Parameter(torch.tensor(float(dustbin_score)))
        self.dustbin_enabled = bool(dustbin_enabled)

    @staticmethod
    def _scatter_logsumexp(
        values: torch.Tensor,
        indices: torch.Tensor,
        valid: torch.Tensor,
        target_count: int,
    ) -> torch.Tensor:
        batch = values.shape[0]
        batch_offset = torch.arange(batch, device=values.device).view(batch, 1, 1) * target_count
        flat_indices = (indices + batch_offset).reshape(-1)
        flat_values = values.masked_fill(~valid, -torch.inf).reshape(-1)
        maxima = values.new_full((batch * target_count,), -torch.inf)
        maxima.scatter_reduce_(0, flat_indices, flat_values, reduce="amax", include_self=True)
        gathered_max = maxima[flat_indices]
        finite = valid.reshape(-1) & torch.isfinite(flat_values) & torch.isfinite(gathered_max)
        stable_difference = torch.where(
            finite, flat_values - gathered_max, torch.zeros_like(flat_values)
        )
        exponentials = torch.exp(stable_difference) * finite.to(flat_values.dtype)
        sums = values.new_zeros((batch * target_count,))
        sums.scatter_add_(0, flat_indices, exponentials)
        result = maxima + sums.clamp_min(1.0e-30).log()
        result = torch.where(sums > 0, result, torch.full_like(result, -torch.inf))
        return result.view(batch, target_count)

    @staticmethod
    def _scatter_sum(
        values: torch.Tensor, indices: torch.Tensor, target_count: int
    ) -> torch.Tensor:
        batch = values.shape[0]
        result = values.new_zeros((batch, target_count))
        result.scatter_add_(1, indices.reshape(batch, -1), values.reshape(batch, -1))
        return result

    def forward(
        self,
        candidate_logits: torch.Tensor,
        candidate_indices: torch.Tensor,
        candidate_valid: torch.Tensor,
        source_mass: torch.Tensor,
        target_mass: torch.Tensor,
        target_count: int,
    ) -> dict[str, torch.Tensor]:
        if not self.dustbin_enabled:
            from .paper_rgcm import sparse_unbalanced_transport
            return sparse_unbalanced_transport(
                candidate_logits, candidate_indices, candidate_valid, source_mass, target_mass,
                self.epsilon, self.tau_source, self.tau_target, self.iterations,
            )
        input_dtype = candidate_logits.dtype
        with torch.autocast(device_type=candidate_logits.device.type, enabled=False):
            logits = torch.nan_to_num(candidate_logits.float(), nan=-1.0e4)
            indices = candidate_indices.long().clamp(0, max(target_count - 1, 0))
            valid = candidate_valid.bool()
            source = torch.nan_to_num(source_mass.float(), nan=0.0).clamp_min(0.0)
            target = torch.nan_to_num(target_mass.float(), nan=0.0).clamp_min(0.0)
            batch, source_count, _ = logits.shape
            if source.shape != (batch, source_count) or target.shape != (batch, target_count):
                raise ValueError("source_mass or target_mass has an incompatible shape")

            source_safe = source.clamp_min(1.0e-8)
            target_safe = target.clamp_min(1.0e-8)
            source_total = source_safe.sum(-1, keepdim=True)
            target_total = target_safe.sum(-1, keepdim=True)
            log_a = torch.cat((source_safe, target_total), dim=-1).log()
            log_b = torch.cat((target_safe, source_total), dim=-1).log()
            log_edge_kernel = logits / self.epsilon
            log_dustbin = self.dustbin_score.float() / self.epsilon
            log_u = torch.zeros_like(log_a)
            log_v = torch.zeros_like(log_b)
            rho_source = self.tau_source / (self.tau_source + self.epsilon)
            rho_target = self.tau_target / (self.tau_target + self.epsilon)

            for _ in range(self.iterations):
                edge_rows = log_edge_kernel + log_v[:, :target_count].gather(
                    1, indices.reshape(batch, -1)
                ).reshape_as(log_edge_kernel)
                edge_rows = edge_rows.masked_fill(~valid, -torch.inf)
                source_row_lse = torch.logsumexp(
                    torch.cat(
                        (
                            edge_rows,
                            (log_dustbin + log_v[:, target_count]).view(batch, 1, 1).expand(-1, source_count, -1),
                        ),
                        dim=-1,
                    ),
                    dim=-1,
                )
                dustbin_row_lse = torch.logsumexp(
                    torch.cat(
                        (
                            log_dustbin + log_v[:, :target_count],
                            (log_dustbin + log_v[:, target_count]).unsqueeze(-1),
                        ),
                        dim=-1,
                    ),
                    dim=-1,
                )
                row_lse = torch.cat((source_row_lse, dustbin_row_lse.unsqueeze(-1)), dim=-1)
                log_u = rho_source * (log_a - row_lse)

                edge_columns = log_edge_kernel + log_u[:, :source_count].unsqueeze(-1)
                target_column_lse = self._scatter_logsumexp(
                    edge_columns, indices, valid, target_count
                )
                target_column_lse = torch.logaddexp(
                    target_column_lse, log_dustbin + log_u[:, source_count].unsqueeze(-1)
                )
                dustbin_column_lse = torch.logsumexp(
                    torch.cat(
                        (
                            log_dustbin + log_u[:, :source_count],
                            (log_dustbin + log_u[:, source_count]).unsqueeze(-1),
                        ),
                        dim=-1,
                    ),
                    dim=-1,
                )
                column_lse = torch.cat(
                    (target_column_lse, dustbin_column_lse.unsqueeze(-1)), dim=-1
                )
                log_v = rho_target * (log_b - column_lse)

            gathered_v = log_v[:, :target_count].gather(
                1, indices.reshape(batch, -1)
            ).reshape_as(log_edge_kernel)
            log_edge_plan = log_edge_kernel + log_u[:, :source_count].unsqueeze(-1) + gathered_v
            edge_transport = torch.exp(log_edge_plan.clamp(max=30.0)) * valid.float()
            edge_transport = torch.nan_to_num(edge_transport)
            unmatched_source = torch.exp(
                (log_dustbin + log_u[:, :source_count] + log_v[:, target_count].unsqueeze(-1)).clamp(max=30.0)
            )
            unmatched_target = torch.exp(
                (log_dustbin + log_u[:, source_count].unsqueeze(-1) + log_v[:, :target_count]).clamp(max=30.0)
            )
            row_mass = edge_transport.sum(-1)
            column_mass = self._scatter_sum(edge_transport, indices, target_count)
            conditional = edge_transport / row_mass.unsqueeze(-1).clamp_min(1.0e-8)
            entropy = -(
                conditional.clamp_min(1.0e-8) * conditional.clamp_min(1.0e-8).log()
            ).sum(-1)
            outputs = {
                "edge_transport": edge_transport,
                "unmatched_source_mass": torch.nan_to_num(unmatched_source),
                "unmatched_target_mass": torch.nan_to_num(unmatched_target),
                "row_transport_mass": row_mass,
                "column_transport_mass": column_mass,
                "transport_entropy": torch.nan_to_num(entropy),
            }
            return {key: value.to(input_dtype) for key, value in outputs.items()}


class SparseRelationReasoner(nn.Module):
    """Vectorized O(B*N*Q*K^2) candidate-association relation update."""

    def __init__(self, relation_encoder: EdgeRelationEncoder, config: dict):
        super().__init__()
        self.encoder = relation_encoder
        self.relation_weight = float(config.get("relation_weight", 1.0))
        self.geometry_weight = float(config.get("geometry_weight", 1.0))
        self.temperature = float(config.get("relation_temperature", 0.2))

    def forward(
        self,
        optical_nodes: torch.Tensor,
        sar_nodes: torch.Tensor,
        graph: dict[str, torch.Tensor],
        candidate_indices: torch.Tensor,
        candidate_transport: torch.Tensor,
        affine: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, count, topk = candidate_indices.shape
        neighbours = graph["neighbor_indices"]
        valid = graph["neighbor_valid"].to(optical_nodes)
        q = neighbours.shape[1]
        channels = optical_nodes.shape[-1]
        optical_rel = self.encoder(optical_nodes, graph)

        j_nodes = sar_nodes.gather(1, candidate_indices.unsqueeze(-1).expand(-1, -1, -1, channels).reshape(batch, count * topk, channels))
        j_nodes = j_nodes.reshape(batch, count, topk, channels)
        neighbour_candidates = candidate_indices[:, neighbours.reshape(-1)].reshape(batch, count, q, topk)
        l_nodes = sar_nodes.gather(1, neighbour_candidates.reshape(batch, -1).unsqueeze(-1).expand(-1, -1, channels))
        l_nodes = l_nodes.reshape(batch, count, q, topk, channels)
        first = j_nodes[:, :, :, None, None, :].expand(-1, -1, -1, q, topk, -1)
        second = l_nodes[:, :, None, :, :, :].expand(-1, -1, topk, -1, -1, -1)

        coordinates = graph["coordinates"].to(optical_nodes)
        j_coordinates = coordinates[candidate_indices]
        l_coordinates = coordinates[neighbour_candidates]
        delta_sar = l_coordinates[:, :, None, :, :, :] - j_coordinates[:, :, :, None, None, :]
        sar_rel = self.encoder.encode_pairs(first, second, delta_sar)
        relation_cost_full = F.smooth_l1_loss(
            sar_rel,
            optical_rel[:, :, None, :, None, :].expand_as(sar_rel),
            reduction="none",
        ).mean(-1)

        delta_optical = graph["relative_offsets"].to(optical_nodes)
        transformed_delta = torch.einsum("bij,nqj->bnqi", affine[:, :2, :2], delta_optical)
        geometry_cost_full = torch.linalg.vector_norm(
            delta_sar - transformed_delta[:, :, None, :, None, :], dim=-1
        ) / math.sqrt(8.0)
        compatibility = torch.exp(-(
            self.relation_weight * relation_cost_full + self.geometry_weight * geometry_cost_full
        ) / max(self.temperature, 1.0e-6))

        neighbour_transport = candidate_transport[:, neighbours.reshape(-1)].reshape(batch, count, q, topk)
        support_weight = neighbour_transport[:, :, None, :, :] * valid[None, :, None, :, None]
        denominator = support_weight.sum(dim=(-2, -1)).clamp_min(1.0e-8)
        support = (support_weight * compatibility).sum(dim=(-2, -1)) / denominator
        relation_cost = (support_weight * relation_cost_full).sum(dim=(-2, -1)) / denominator
        geometry_cost = (support_weight * geometry_cost_full).sum(dim=(-2, -1)) / denominator
        return {
            "graph_support": support.clamp_min(1.0e-8),
            "relation_cost": relation_cost,
            "geometry_cost": geometry_cost,
            "compatibility": compatibility,
            "association_mass": support_weight,
        }
