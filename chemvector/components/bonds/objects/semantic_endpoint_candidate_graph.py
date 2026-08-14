"""Joint graph scoring over Bond candidates that share semantic endpoints."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from chemvector.components.bonds.objects.semantic_endpoint_pair_scorer import (
    PAIR_FEATURE_DIM,
)


NODE_FEATURE_DIM = 4


@dataclass(frozen=True, slots=True)
class SemanticEndpointCandidateGraphConfig:
    node_feature_dim: int = NODE_FEATURE_DIM
    edge_feature_dim: int = PAIR_FEATURE_DIM
    hidden_dim: int = 128
    message_passing_steps: int = 2
    max_degree: int = 8
    incidence_count_scale: float = 16.0

    def __post_init__(self) -> None:
        if min(
            self.node_feature_dim,
            self.edge_feature_dim,
            self.hidden_dim,
            self.message_passing_steps,
            self.max_degree,
        ) < 1:
            raise ValueError("Candidate graph dimensions must be positive")
        if self.incidence_count_scale <= 0.0:
            raise ValueError("Candidate graph incidence scale must be positive")


def candidate_graph_node_features(
    points: torch.Tensor,
    point_scores: torch.Tensor,
    pairs: torch.Tensor,
    *,
    incidence_count_scale: float = 16.0,
) -> torch.Tensor:
    """Build x/y, endpoint-confidence and candidate-incidence node features."""

    if points.ndim != 2 or points.size(-1) != 2:
        raise ValueError("Candidate graph points must have shape [nodes, 2]")
    if point_scores.shape != (len(points),):
        raise ValueError("Candidate graph point scores differ from points")
    if pairs.ndim != 2 or pairs.size(-1) != 2:
        raise ValueError("Candidate graph pairs must have shape [edges, 2]")
    if incidence_count_scale <= 0.0:
        raise ValueError("Candidate graph incidence scale must be positive")
    if len(pairs) and (
        int(pairs.min()) < 0 or int(pairs.max()) >= len(points)
    ):
        raise ValueError("Candidate graph pair index is outside the node set")
    incidence = points.new_zeros((len(points),))
    if len(pairs):
        ones = points.new_ones((len(pairs),))
        incidence.index_add_(0, pairs[:, 0], ones)
        incidence.index_add_(0, pairs[:, 1], ones)
    normalized_incidence = (incidence / incidence_count_scale).clamp(max=1.0)
    return torch.cat(
        (points.float(), point_scores.float()[:, None], normalized_incidence[:, None]),
        dim=-1,
    )


class _CandidateGraphBlock(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.edge_update = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.edge_norm = nn.LayerNorm(hidden_dim)
        self.node_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.node_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        node_hidden: torch.Tensor,
        edge_hidden: torch.Tensor,
        pairs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not len(edge_hidden):
            return node_hidden, edge_hidden
        begin = node_hidden[pairs[:, 0]]
        end = node_hidden[pairs[:, 1]]
        endpoint_sum = begin + end
        endpoint_difference = (begin - end).abs()
        edge_delta = self.edge_update(
            torch.cat((edge_hidden, endpoint_sum, endpoint_difference), dim=-1)
        )
        edge_hidden = self.edge_norm(edge_hidden + edge_delta)

        aggregate = edge_hidden.new_zeros(node_hidden.shape)
        degree = edge_hidden.new_zeros((len(node_hidden), 1))
        ones = edge_hidden.new_ones((len(edge_hidden), 1))
        for endpoint in (pairs[:, 0], pairs[:, 1]):
            aggregate.index_add_(0, endpoint, edge_hidden)
            degree.index_add_(0, endpoint, ones)
        aggregate = aggregate / degree.clamp_min(1.0)
        node_delta = self.node_update(torch.cat((node_hidden, aggregate), dim=-1))
        node_hidden = self.node_norm(node_hidden + node_delta)
        return node_hidden, edge_hidden


class SemanticEndpointCandidateGraph(nn.Module):
    """Score the complete sparse candidate graph instead of independent pairs."""

    def __init__(
        self,
        config: SemanticEndpointCandidateGraphConfig = (
            SemanticEndpointCandidateGraphConfig()
        ),
    ) -> None:
        super().__init__()
        self.config = config
        self.node_encoder = nn.Sequential(
            nn.LayerNorm(config.node_feature_dim),
            nn.Linear(config.node_feature_dim, config.hidden_dim),
            nn.GELU(),
        )
        self.edge_encoder = nn.Sequential(
            nn.LayerNorm(config.edge_feature_dim),
            nn.Linear(config.edge_feature_dim, config.hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            _CandidateGraphBlock(config.hidden_dim)
            for _ in range(config.message_passing_steps)
        )
        self.edge_head = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
        )
        self.node_keep_head = nn.Linear(config.hidden_dim, 1)
        self.node_degree_head = nn.Linear(config.hidden_dim, config.max_degree + 1)
        self.graph_count_head = nn.Sequential(
            nn.Linear(config.hidden_dim * 2 + 2, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        pairs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if (
            node_features.ndim != 2
            or node_features.size(-1) != self.config.node_feature_dim
        ):
            raise ValueError("Candidate graph node features have an invalid shape")
        if (
            edge_features.ndim != 2
            or edge_features.size(-1) != self.config.edge_feature_dim
        ):
            raise ValueError("Candidate graph edge features have an invalid shape")
        if pairs.shape != (len(edge_features), 2):
            raise ValueError("Candidate graph pairs differ from edge features")
        if len(pairs) and (
            int(pairs.min()) < 0 or int(pairs.max()) >= len(node_features)
        ):
            raise ValueError("Candidate graph pair index is outside the node set")

        node_hidden = self.node_encoder(node_features)
        edge_hidden = self.edge_encoder(edge_features)
        for block in self.blocks:
            node_hidden, edge_hidden = block(node_hidden, edge_hidden, pairs)

        if len(edge_hidden):
            begin = node_hidden[pairs[:, 0]]
            end = node_hidden[pairs[:, 1]]
            edge_context = torch.cat(
                (edge_hidden, begin + end, (begin - end).abs()), dim=-1
            )
            edge_logits = self.edge_head(edge_context).squeeze(-1)
            pooled_edges = edge_hidden.mean(dim=0)
        else:
            edge_logits = edge_hidden.new_empty((0,))
            pooled_edges = node_hidden.new_zeros((self.config.hidden_dim,))
        pooled_nodes = (
            node_hidden.mean(dim=0)
            if len(node_hidden)
            else pooled_edges.new_zeros((self.config.hidden_dim,))
        )
        graph_sizes = torch.stack(
            (
                torch.log1p(pooled_edges.new_tensor(float(len(node_hidden))))
                / torch.log(pooled_edges.new_tensor(257.0)),
                torch.log1p(pooled_edges.new_tensor(float(len(edge_hidden))))
                / torch.log(pooled_edges.new_tensor(4097.0)),
            )
        )
        graph_bond_count = F.softplus(
            self.graph_count_head(
                torch.cat((pooled_nodes, pooled_edges, graph_sizes), dim=-1)
            ).squeeze(-1)
        )
        return {
            "edge_logits": edge_logits,
            "node_keep_logits": self.node_keep_head(node_hidden).squeeze(-1),
            "node_degree_logits": self.node_degree_head(node_hidden),
            "graph_bond_count": graph_bond_count,
            "node_hidden": node_hidden,
            "edge_hidden": edge_hidden,
        }


def _latent_edge_targets(
    logits: torch.Tensor,
    target_bond_ids: torch.Tensor,
    bond_count: int,
) -> tuple[torch.Tensor, int]:
    """Choose one currently best candidate for every reachable unordered GT Bond."""

    targets = torch.zeros_like(logits, dtype=torch.bool)
    reachable = 0
    for bond_id in range(bond_count):
        indices = torch.where(target_bond_ids == bond_id)[0]
        if not len(indices):
            continue
        reachable += 1
        winner = indices[logits[indices].detach().argmax()]
        targets[winner] = True
    return targets, reachable


def semantic_endpoint_candidate_graph_loss(
    output: dict[str, torch.Tensor],
    target_bond_ids: torch.Tensor,
    node_target_degrees: torch.Tensor,
    *,
    bond_count: int,
    max_degree: int = 8,
    listwise_weight: float = 0.5,
    duplicate_weight: float = 0.25,
    node_keep_weight: float = 0.5,
    node_degree_weight: float = 0.5,
    cardinality_weight: float = 0.5,
    count_weight: float = 0.5,
    consistency_weight: float = 0.1,
    margin: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Joint latent-set, endpoint-role and whole-graph cardinality objective."""

    logits = output["edge_logits"]
    keep_logits = output["node_keep_logits"]
    degree_logits = output["node_degree_logits"]
    graph_count = output["graph_bond_count"]
    if logits.ndim != 1 or target_bond_ids.shape != logits.shape:
        raise ValueError("Candidate graph logits and target Bond IDs differ")
    if keep_logits.ndim != 1 or node_target_degrees.shape != keep_logits.shape:
        raise ValueError("Candidate graph node targets differ")
    if degree_logits.shape != (len(keep_logits), max_degree + 1):
        raise ValueError("Candidate graph degree logits have an invalid shape")
    if bond_count < 0 or margin <= 0.0:
        raise ValueError("Candidate graph loss configuration is invalid")
    if len(node_target_degrees) and (
        int(node_target_degrees.min()) < 0
        or int(node_target_degrees.max()) > max_degree
    ):
        raise ValueError("Candidate graph target degree is outside the supported range")

    latent_targets, reachable = _latent_edge_targets(
        logits, target_bond_ids, bond_count
    )
    positive_count = latent_targets.sum().clamp_min(1)
    negative_count = (~latent_targets).sum().clamp_min(1)
    edge_bce = F.binary_cross_entropy_with_logits(
        logits.float(),
        latent_targets.float(),
        pos_weight=(negative_count / positive_count).clamp(1.0, 32.0),
    )
    zero = edge_bce * 0.0
    negative_top = (
        logits[~latent_targets].float().max()
        if bool((~latent_targets).any())
        else zero
    )
    ranking_terms = []
    duplicate_terms = []
    for bond_id in range(bond_count):
        current = logits[target_bond_ids == bond_id].float()
        if not len(current):
            continue
        ranking_terms.append(F.softplus(margin - current.max() + negative_top))
        duplicate_terms.append((current.sigmoid().sum() - 1.0).square())
    listwise = torch.stack(ranking_terms).mean() if ranking_terms else zero
    duplicate = torch.stack(duplicate_terms).mean() if duplicate_terms else zero

    node_keep_targets = node_target_degrees > 0
    keep_positive = node_keep_targets.sum().clamp_min(1)
    keep_negative = (~node_keep_targets).sum().clamp_min(1)
    node_keep = F.binary_cross_entropy_with_logits(
        keep_logits.float(),
        node_keep_targets.float(),
        pos_weight=(keep_negative / keep_positive).clamp(1.0, 32.0),
    )
    node_degree = F.cross_entropy(
        degree_logits.float(), node_target_degrees.long()
    )

    probability_count = logits.float().sigmoid().sum()
    count_scale = float(max(bond_count, 1))
    reachable_target = probability_count.new_tensor(float(reachable))
    full_target = graph_count.new_tensor(float(bond_count))
    cardinality = F.smooth_l1_loss(
        probability_count / count_scale, reachable_target / count_scale
    )
    count = F.smooth_l1_loss(graph_count / count_scale, full_target / count_scale)
    consistency = F.smooth_l1_loss(
        probability_count / count_scale,
        graph_count.detach().clamp(max=float(max(bond_count, 1))) / count_scale,
    )
    loss = (
        edge_bce
        + listwise_weight * listwise
        + duplicate_weight * duplicate
        + node_keep_weight * node_keep
        + node_degree_weight * node_degree
        + cardinality_weight * cardinality
        + count_weight * count
        + consistency_weight * consistency
    )
    return {
        "loss": loss,
        "edge_bce_loss": edge_bce,
        "listwise_loss": listwise,
        "duplicate_loss": duplicate,
        "node_keep_loss": node_keep,
        "node_degree_loss": node_degree,
        "cardinality_loss": cardinality,
        "count_loss": count,
        "consistency_loss": consistency,
        "latent_positive_edges": latent_targets.sum(),
        "reachable_bonds": logits.new_tensor(reachable),
        "probability_bond_count": probability_count,
    }


__all__ = [
    "NODE_FEATURE_DIM",
    "SemanticEndpointCandidateGraph",
    "SemanticEndpointCandidateGraphConfig",
    "candidate_graph_node_features",
    "semantic_endpoint_candidate_graph_loss",
]
