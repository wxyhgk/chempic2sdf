"""Learned existence scoring for frozen local-partner Bond candidates."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from chemvector.components.bonds.objects.semantic_endpoint_pair import (
    SemanticEndpointPairConfig,
    _blocker_support,
    _sample_pair_ribbons,
)


PAIR_FEATURE_DIM = 64


@dataclass(frozen=True, slots=True)
class SemanticEndpointPairScorerConfig:
    feature_dim: int = PAIR_FEATURE_DIM
    hidden_dim: int = 128

    def __post_init__(self) -> None:
        if min(self.feature_dim, self.hidden_dim) < 1:
            raise ValueError("Semantic endpoint pair scorer dimensions must be positive")


@torch.no_grad()
def extract_endpoint_pair_features(
    bond_probability: torch.Tensor,
    points: torch.Tensor,
    point_scores: torch.Tensor,
    pairs: torch.Tensor,
    pair_config: SemanticEndpointPairConfig = SemanticEndpointPairConfig(),
) -> torch.Tensor:
    if pairs.ndim != 2 or pairs.size(-1) != 2:
        raise ValueError("Pair scorer indices must have shape [pairs, 2]")
    if not len(pairs):
        return points.new_empty((0, PAIR_FEATURE_DIM))
    ribbons = _sample_pair_ribbons(
        bond_probability, points, pairs, pair_config
    ).float()
    longitudinal = ribbons.amax(dim=1)
    transverse = ribbons.mean(dim=-1)
    line_mean = longitudinal.mean(dim=-1, keepdim=True)
    lower_index = max(1, int(pair_config.longitudinal_samples * 0.2))
    lower = longitudinal.kthvalue(lower_index, dim=-1).values[:, None]
    coverage = (longitudinal >= 0.5).float().mean(dim=-1, keepdim=True)
    first = point_scores[pairs[:, 0]]
    second = point_scores[pairs[:, 1]]
    endpoint_scores = torch.stack(
        (torch.minimum(first, second), torch.maximum(first, second)), dim=-1
    )
    length = (
        (points[pairs[:, 1]] - points[pairs[:, 0]]).norm(dim=-1, keepdim=True)
    )
    blocker = _blocker_support(
        points, point_scores, pairs, pair_config
    )[:, None]
    features = torch.cat(
        (
            longitudinal,
            transverse,
            line_mean,
            lower,
            coverage,
            endpoint_scores,
            length,
            blocker,
        ),
        dim=-1,
    )
    if features.size(1) != PAIR_FEATURE_DIM:
        raise ValueError("Semantic endpoint pair feature dimension differs")
    return features


class SemanticEndpointPairScorer(nn.Module):
    def __init__(
        self,
        config: SemanticEndpointPairScorerConfig = SemanticEndpointPairScorerConfig(),
    ) -> None:
        super().__init__()
        self.config = config
        self.network = nn.Sequential(
            nn.LayerNorm(config.feature_dim),
            nn.Linear(config.feature_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.size(-1) != self.config.feature_dim:
            raise ValueError("Pair scorer features have an invalid shape")
        return self.network(features).squeeze(-1)


@torch.no_grad()
def deduplicate_scored_pair_segments(
    points: torch.Tensor,
    pairs: torch.Tensor,
    scores: torch.Tensor,
    *,
    radius_pixels: float = 4.0,
    image_size: float = 768.0,
) -> torch.Tensor:
    """Return score order after unordered endpoint-pair geometric NMS."""

    if pairs.ndim != 2 or pairs.size(-1) != 2 or scores.shape != (len(pairs),):
        raise ValueError("Pair segment NMS inputs differ")
    if min(radius_pixels, image_size) <= 0.0:
        raise ValueError("Pair segment NMS scales must be positive")
    order = scores.detach().float().cpu().argsort(descending=True)
    segments = points[pairs].detach().float().cpu()
    cells = torch.floor(segments * image_size / radius_pixels).long()
    selected: list[int] = []
    used: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for index in order.tolist():
        endpoints = [tuple(int(value) for value in point) for point in cells[index]]
        key = tuple(sorted(endpoints))
        if key in used:
            continue
        used.add(key)
        selected.append(index)
    return torch.tensor(selected, device=pairs.device, dtype=torch.long)


def semantic_endpoint_pair_scorer_loss(
    logits: torch.Tensor,
    target_bond_ids: torch.Tensor,
    *,
    bond_count: int,
    listwise_weight: float = 0.5,
    margin: float = 1.0,
) -> dict[str, torch.Tensor]:
    if logits.ndim != 1 or target_bond_ids.shape != logits.shape:
        raise ValueError("Pair scorer logits and target IDs differ")
    if bond_count < 0 or listwise_weight < 0.0 or margin <= 0.0:
        raise ValueError("Pair scorer loss configuration is invalid")
    positive = target_bond_ids >= 0
    positive_count = positive.sum().clamp_min(1)
    negative_count = (~positive).sum().clamp_min(1)
    bce = F.binary_cross_entropy_with_logits(
        logits.float(),
        positive.float(),
        pos_weight=(negative_count / positive_count).clamp(1.0, 16.0),
    )
    zero = bce * 0.0
    negative_top = logits[~positive].float().max() if bool((~positive).any()) else zero
    ranking = []
    reachable_bonds = 0
    for bond_id in range(bond_count):
        current = logits[target_bond_ids == bond_id].float()
        if not len(current):
            continue
        reachable_bonds += 1
        ranking.append(F.softplus(margin - current.max() + negative_top))
    listwise = torch.stack(ranking).mean() if ranking else zero
    return {
        "loss": bce + listwise_weight * listwise,
        "bce_loss": bce,
        "listwise_loss": listwise,
        "positive_pairs": positive.sum(),
        "reachable_bonds": logits.new_tensor(reachable_bonds),
    }


__all__ = [
    "PAIR_FEATURE_DIM",
    "SemanticEndpointPairScorer",
    "SemanticEndpointPairScorerConfig",
    "deduplicate_scored_pair_segments",
    "extract_endpoint_pair_features",
    "semantic_endpoint_pair_scorer_loss",
]
