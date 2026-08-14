"""DETR-style Bond query set over sparse semantic endpoint-pair memory."""

from __future__ import annotations

from dataclasses import dataclass
import math

from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
from torch.nn import functional as F

from chemvector.components.bonds.objects.semantic_endpoint_candidate_graph import (
    NODE_FEATURE_DIM,
    _CandidateGraphBlock,
)
from chemvector.components.bonds.objects.semantic_endpoint_pair_scorer import (
    PAIR_FEATURE_DIM,
)


@dataclass(frozen=True, slots=True)
class SemanticBondCandidateQuerySetConfig:
    node_feature_dim: int = NODE_FEATURE_DIM
    edge_feature_dim: int = PAIR_FEATURE_DIM
    d_model: int = 128
    graph_message_passing_steps: int = 2
    decoder_layers: int = 2
    attention_heads: int = 4
    feedforward_dim: int = 256
    query_count: int = 128

    def __post_init__(self) -> None:
        if min(
            self.node_feature_dim,
            self.edge_feature_dim,
            self.d_model,
            self.graph_message_passing_steps,
            self.decoder_layers,
            self.attention_heads,
            self.feedforward_dim,
            self.query_count,
        ) < 1:
            raise ValueError("Bond candidate query-set dimensions must be positive")
        if self.d_model % self.attention_heads:
            raise ValueError("Bond query-set model dimension must divide attention heads")


class SemanticBondCandidateQuerySet(nn.Module):
    """Map a variable candidate graph to a fixed, no-object-aware Bond set."""

    def __init__(
        self,
        config: SemanticBondCandidateQuerySetConfig = (
            SemanticBondCandidateQuerySetConfig()
        ),
    ) -> None:
        super().__init__()
        self.config = config
        self.node_encoder = nn.Sequential(
            nn.LayerNorm(config.node_feature_dim),
            nn.Linear(config.node_feature_dim, config.d_model),
            nn.GELU(),
        )
        self.edge_encoder = nn.Sequential(
            nn.LayerNorm(config.edge_feature_dim),
            nn.Linear(config.edge_feature_dim, config.d_model),
            nn.GELU(),
        )
        self.graph_blocks = nn.ModuleList(
            _CandidateGraphBlock(config.d_model)
            for _ in range(config.graph_message_passing_steps)
        )
        self.memory_fusion = nn.Sequential(
            nn.Linear(config.d_model * 3, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
        )
        layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.attention_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            layer, num_layers=config.decoder_layers, norm=nn.LayerNorm(config.d_model)
        )
        self.query_embeddings = nn.Parameter(
            torch.empty(config.query_count, config.d_model)
        )
        self.query_projection = nn.Linear(config.d_model, config.d_model)
        self.edge_projection = nn.Linear(config.d_model, config.d_model)
        self.existence_head = nn.Linear(config.d_model, 1)
        nn.init.normal_(self.query_embeddings, std=0.02)

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
            raise ValueError("Bond query-set node features have an invalid shape")
        if (
            edge_features.ndim != 2
            or edge_features.size(-1) != self.config.edge_feature_dim
        ):
            raise ValueError("Bond query-set edge features have an invalid shape")
        if pairs.shape != (len(edge_features), 2):
            raise ValueError("Bond query-set pairs differ from edge features")
        if not len(edge_features):
            raise ValueError("Bond query-set requires at least one candidate edge")
        if int(pairs.min()) < 0 or int(pairs.max()) >= len(node_features):
            raise ValueError("Bond query-set pair index is outside the node set")

        node_hidden = self.node_encoder(node_features)
        edge_hidden = self.edge_encoder(edge_features)
        for block in self.graph_blocks:
            node_hidden, edge_hidden = block(node_hidden, edge_hidden, pairs)
        begin = node_hidden[pairs[:, 0]]
        end = node_hidden[pairs[:, 1]]
        edge_hidden = edge_hidden + self.memory_fusion(
            torch.cat((edge_hidden, begin + end, (begin - end).abs()), dim=-1)
        )
        memory = edge_hidden.unsqueeze(0)
        graph_context = edge_hidden.mean(dim=0, keepdim=True)
        queries = (self.query_embeddings + graph_context).unsqueeze(0)
        decoded = self.decoder(queries, memory).squeeze(0)
        pointer_logits = (
            self.query_projection(decoded)
            @ self.edge_projection(edge_hidden).transpose(0, 1)
        ) / math.sqrt(self.config.d_model)
        return {
            "existence_logits": self.existence_head(decoded).squeeze(-1),
            "pointer_logits": pointer_logits,
            "query_hidden": decoded,
            "edge_hidden": edge_hidden,
            "node_hidden": node_hidden,
        }


def _group_pointer_log_probability(
    pointer_logits: torch.Tensor,
    target_bond_ids: torch.Tensor,
    bond_count: int,
) -> torch.Tensor:
    log_probability = F.log_softmax(pointer_logits.float(), dim=-1)
    columns = []
    for bond_id in range(bond_count):
        mask = target_bond_ids == bond_id
        if not bool(mask.any()):
            raise ValueError(f"Bond query-set target Bond {bond_id} is unreachable")
        columns.append(torch.logsumexp(log_probability[:, mask], dim=-1))
    return torch.stack(columns, dim=-1) if columns else log_probability.new_empty((len(pointer_logits), 0))


def semantic_bond_candidate_query_set_loss(
    output: dict[str, torch.Tensor],
    target_bond_ids: torch.Tensor,
    *,
    bond_count: int,
    pointer_weight: float = 2.0,
    cardinality_weight: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Hungarian set loss with group probability over equivalent edge candidates."""

    existence_logits = output["existence_logits"]
    pointer_logits = output["pointer_logits"]
    if existence_logits.ndim != 1:
        raise ValueError("Bond query-set existence logits must be one-dimensional")
    if pointer_logits.shape != (len(existence_logits), len(target_bond_ids)):
        raise ValueError("Bond query-set pointer logits and targets differ")
    if bond_count < 0 or bond_count > len(existence_logits):
        raise ValueError("Bond query-set target count exceeds query capacity")
    if min(pointer_weight, cardinality_weight) < 0.0:
        raise ValueError("Bond query-set loss weights must be non-negative")

    group_log_probability = _group_pointer_log_probability(
        pointer_logits, target_bond_ids, bond_count
    )
    if bond_count:
        cost = -group_log_probability.detach().cpu().numpy()
        rows, columns = linear_sum_assignment(cost)
        matched_queries = torch.as_tensor(
            rows, device=existence_logits.device, dtype=torch.long
        )
        matched_bonds = torch.as_tensor(
            columns, device=existence_logits.device, dtype=torch.long
        )
    else:
        matched_queries = torch.empty(0, device=existence_logits.device, dtype=torch.long)
        matched_bonds = matched_queries.clone()
    existence_targets = torch.zeros_like(existence_logits)
    existence_targets[matched_queries] = 1.0
    positive_count = max(bond_count, 1)
    negative_count = max(len(existence_logits) - bond_count, 1)
    existence = F.binary_cross_entropy_with_logits(
        existence_logits.float(),
        existence_targets,
        pos_weight=existence_logits.new_tensor(
            min(negative_count / positive_count, 16.0)
        ).float(),
    )
    pointer = (
        -group_log_probability[matched_queries, matched_bonds].mean()
        if bond_count
        else existence * 0.0
    )
    probability_count = existence_logits.float().sigmoid().sum()
    scale = float(max(bond_count, 1))
    cardinality = F.smooth_l1_loss(
        probability_count / scale,
        probability_count.new_tensor(float(bond_count) / scale),
    )
    return {
        "loss": existence + pointer_weight * pointer + cardinality_weight * cardinality,
        "existence_loss": existence,
        "pointer_loss": pointer,
        "cardinality_loss": cardinality,
        "probability_bond_count": probability_count,
        "matched_queries": matched_queries,
        "matched_bonds": matched_bonds,
    }


__all__ = [
    "SemanticBondCandidateQuerySet",
    "SemanticBondCandidateQuerySetConfig",
    "semantic_bond_candidate_query_set_loss",
]
