"""Independent DeepSets Count model over frozen G3 base/expert evidence."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class G3DeepSetsCountConfig:
    query_dim: int = 8
    candidate_dim: int = 41
    global_dim: int = 256
    hidden_dim: int = 64
    classes: int = 129


def g3_deepsets_count_features(
    base_output: dict[str, torch.Tensor],
    expert_output: dict[str, torch.Tensor],
    candidate_features: torch.Tensor,
    raw_scores: torch.Tensor,
    normalized_scores: torch.Tensor,
    scale_ranks: torch.Tensor,
    *,
    topk: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build query, candidate, and sorted global evidence without GT labels."""

    base_existence = base_output["existence_logits"].float()
    expert_existence = expert_output["existence_logits"].float()
    base_pointer = base_output["pointer_logits"].float().log_softmax(dim=-1)
    expert_pointer = expert_output["pointer_logits"].float().log_softmax(dim=-1)
    if base_pointer.shape != expert_pointer.shape:
        raise ValueError("G3 Count base/expert pointer shapes differ")
    batch, queries, candidates = base_pointer.shape
    if base_existence.shape != (batch, queries) or expert_existence.shape != (
        batch,
        queries,
    ):
        raise ValueError("G3 Count existence shapes differ")
    if candidate_features.shape != (batch, candidates, 32):
        raise ValueError("G3 Count candidate visual features have invalid shape")
    for value in (raw_scores, normalized_scores, scale_ranks):
        if value.shape != (batch, candidates):
            raise ValueError("G3 Count candidate scalar feature has invalid shape")
    if not 1 <= topk <= min(queries, candidates):
        raise ValueError("G3 Count top-k is outside query/candidate capacity")

    base_top2 = base_pointer.topk(2, dim=-1).values
    expert_top2 = expert_pointer.topk(2, dim=-1).values
    entropy_scale = math.log(candidates)
    query_features = torch.stack((
        base_existence,
        expert_existence,
        base_top2[..., 0],
        expert_top2[..., 0],
        -(base_pointer.exp() * base_pointer).sum(dim=-1) / entropy_scale,
        -(expert_pointer.exp() * expert_pointer).sum(dim=-1) / entropy_scale,
        base_top2[..., 0] - base_top2[..., 1],
        expert_top2[..., 0] - expert_top2[..., 1],
    ), dim=-1)

    base_edges = F.logsigmoid(base_existence).unsqueeze(-1) + base_pointer
    expert_edges = F.logsigmoid(expert_existence).unsqueeze(-1) + expert_pointer
    base_votes = F.one_hot(
        base_pointer.argmax(dim=-1), num_classes=candidates
    ).float().sum(dim=1) / queries
    expert_votes = F.one_hot(
        expert_pointer.argmax(dim=-1), num_classes=candidates
    ).float().sum(dim=1) / queries
    candidate_features_full = torch.cat((
        candidate_features.float(),
        raw_scores.float().unsqueeze(-1),
        normalized_scores.float().unsqueeze(-1),
        (scale_ranks.float() / candidates).unsqueeze(-1),
        base_edges.max(dim=1).values.unsqueeze(-1),
        expert_edges.max(dim=1).values.unsqueeze(-1),
        (base_edges.logsumexp(dim=1) - math.log(queries)).unsqueeze(-1),
        (expert_edges.logsumexp(dim=1) - math.log(queries)).unsqueeze(-1),
        base_votes.unsqueeze(-1),
        expert_votes.unsqueeze(-1),
    ), dim=-1)
    global_features = torch.cat((
        base_existence.topk(topk, dim=-1).values,
        expert_existence.topk(topk, dim=-1).values,
        base_edges.max(dim=1).values.topk(topk, dim=-1).values,
        expert_edges.max(dim=1).values.topk(topk, dim=-1).values,
    ), dim=-1)
    return query_features, candidate_features_full, global_features


class G3DeepSetsCount(nn.Module):
    def __init__(self, config: G3DeepSetsCountConfig = G3DeepSetsCountConfig()):
        super().__init__()
        self.config = config
        self.query_encoder = nn.Sequential(
            nn.LayerNorm(config.query_dim),
            nn.Linear(config.query_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
        )
        self.candidate_encoder = nn.Sequential(
            nn.LayerNorm(config.candidate_dim),
            nn.Linear(config.candidate_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
        )
        self.global_encoder = nn.Sequential(
            nn.LayerNorm(config.global_dim),
            nn.Linear(config.global_dim, 2 * config.hidden_dim),
            nn.GELU(),
        )
        aggregate_dim = 6 * config.hidden_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(aggregate_dim),
            nn.Linear(aggregate_dim, 2 * config.hidden_dim),
            nn.GELU(),
            nn.Linear(2 * config.hidden_dim, config.classes),
        )

    def forward(
        self,
        query_features: torch.Tensor,
        candidate_features: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        if query_features.ndim != 3 or query_features.size(-1) != self.config.query_dim:
            raise ValueError("G3 DeepSets Count query features have invalid shape")
        if candidate_features.ndim != 3 or candidate_features.size(-1) != self.config.candidate_dim:
            raise ValueError("G3 DeepSets Count candidate features have invalid shape")
        if global_features.shape != (len(query_features), self.config.global_dim):
            raise ValueError("G3 DeepSets Count global features have invalid shape")
        query = self.query_encoder(query_features)
        candidate = self.candidate_encoder(candidate_features)
        global_hidden = self.global_encoder(global_features)
        aggregate = torch.cat((
            query.mean(dim=1),
            query.max(dim=1).values,
            candidate.mean(dim=1),
            candidate.max(dim=1).values,
            global_hidden,
        ), dim=-1)
        return self.classifier(aggregate)


__all__ = [
    "G3DeepSetsCount",
    "G3DeepSetsCountConfig",
    "g3_deepsets_count_features",
]
