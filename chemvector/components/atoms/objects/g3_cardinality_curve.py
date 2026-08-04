"""Local cardinality-curve refinement over exact base/expert assignments."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from chemvector.components.atoms.objects.g3_latent_matching import (
    g3_unrestricted_fixed_cardinality_assignment,
)


@dataclass(frozen=True)
class G3CardinalityCurveConfig:
    radius: int = 4
    point_dim: int = 13
    hidden_dim: int = 64
    max_count: int = 128

    def __post_init__(self) -> None:
        if self.radius < 1 or self.point_dim != len(G3_CARDINALITY_CURVE_FEATURES):
            raise ValueError("G3 cardinality curve configuration is invalid")
        if self.hidden_dim < 1 or self.max_count < 1:
            raise ValueError("G3 cardinality curve capacity is invalid")

    @property
    def points(self) -> int:
        return 2 * self.radius + 1


G3_CARDINALITY_CURVE_FEATURES = (
    "absolute_count_normalized",
    "coarse_offset_normalized",
    "valid_count",
    "base_total_score_normalized",
    "expert_total_score_normalized",
    "base_mean_score",
    "expert_mean_score",
    "expert_minus_base_total_normalized",
    "expert_minus_base_mean",
    "base_adjacent_increment",
    "expert_adjacent_increment",
    "coarse_local_probability",
    "coarse_local_log_probability",
)


def g3_cardinality_curve_features(
    base_edge_scores: torch.Tensor,
    expert_edge_scores: torch.Tensor,
    coarse_logits: torch.Tensor,
    *,
    config: G3CardinalityCurveConfig = G3CardinalityCurveConfig(),
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return nine local fixed-K curve points and their absolute counts."""

    if base_edge_scores.ndim != 2 or base_edge_scores.shape != expert_edge_scores.shape:
        raise ValueError("G3 cardinality base/expert score shapes differ")
    if not bool(torch.isfinite(base_edge_scores).all()) or not bool(
        torch.isfinite(expert_edge_scores).all()
    ):
        raise ValueError("G3 cardinality edge scores must be finite")
    capacity = min(base_edge_scores.shape)
    if config.max_count != capacity:
        raise ValueError("G3 cardinality configured capacity differs from score matrix")
    if coarse_logits.shape != (config.max_count + 1,) or not bool(
        torch.isfinite(coarse_logits).all()
    ):
        raise ValueError("G3 cardinality coarse logits have invalid shape")

    coarse_count = int(coarse_logits.argmax())
    offsets = torch.arange(-config.radius, config.radius + 1, dtype=torch.long)
    counts = coarse_count + offsets
    valid = (counts >= 0) & (counts <= config.max_count)
    local_probabilities = coarse_logits.float().softmax(dim=-1)
    local_log_probabilities = coarse_logits.float().log_softmax(dim=-1)
    totals = torch.zeros((config.points, 2), dtype=torch.float32)
    means = torch.zeros_like(totals)

    for index, count_tensor in enumerate(counts):
        if not bool(valid[index]):
            continue
        count = int(count_tensor)
        for model_index, scores in enumerate((base_edge_scores, expert_edge_scores)):
            assignment = g3_unrestricted_fixed_cardinality_assignment(scores, count)
            total = float(assignment["score"])
            totals[index, model_index] = total
            means[index, model_index] = total / max(count, 1)

    increments = torch.zeros_like(totals)
    for index in range(1, config.points):
        if bool(valid[index] and valid[index - 1]):
            increments[index] = totals[index] - totals[index - 1]

    curve = torch.zeros((config.points, config.point_dim), dtype=torch.float32)
    for index, count_tensor in enumerate(counts):
        if not bool(valid[index]):
            curve[index, 0] = float(count_tensor) / config.max_count
            curve[index, 1] = float(offsets[index]) / config.radius
            continue
        count = int(count_tensor)
        base_total, expert_total = totals[index]
        base_mean, expert_mean = means[index]
        curve[index] = torch.tensor((
            count / config.max_count,
            float(offsets[index]) / config.radius,
            1.0,
            float(base_total) / config.max_count,
            float(expert_total) / config.max_count,
            float(base_mean),
            float(expert_mean),
            float(expert_total - base_total) / config.max_count,
            float(expert_mean - base_mean),
            float(increments[index, 0]),
            float(increments[index, 1]),
            float(local_probabilities[count]),
            float(local_log_probabilities[count]),
        ))
    return curve, counts, coarse_count


def g3_cardinality_offset_target(
    true_count: int,
    coarse_count: int,
    *,
    radius: int = 4,
) -> tuple[int, bool]:
    """Return the clipped offset class and whether true Count is in-window."""

    if radius < 1 or true_count < 0 or coarse_count < 0:
        raise ValueError("G3 cardinality target arguments are invalid")
    offset = true_count - coarse_count
    return min(max(offset, -radius), radius) + radius, abs(offset) <= radius


class G3CardinalityCurveRefiner(nn.Module):
    def __init__(
        self,
        config: G3CardinalityCurveConfig = G3CardinalityCurveConfig(),
    ):
        super().__init__()
        self.config = config
        self.point_encoder = nn.Sequential(
            nn.LayerNorm(config.point_dim),
            nn.Linear(config.point_dim, config.hidden_dim),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.points * config.hidden_dim),
            nn.Linear(config.points * config.hidden_dim, 2 * config.hidden_dim),
            nn.GELU(),
            nn.Linear(2 * config.hidden_dim, config.points),
        )

    def forward(self, curve: torch.Tensor) -> torch.Tensor:
        expected = (self.config.points, self.config.point_dim)
        if curve.ndim != 3 or tuple(curve.shape[1:]) != expected:
            raise ValueError("G3 cardinality curve features have invalid shape")
        encoded = self.point_encoder(curve)
        logits = self.classifier(encoded.flatten(start_dim=1))
        return logits.masked_fill(curve[..., 2] <= 0.0, -torch.finfo(logits.dtype).max)


__all__ = [
    "G3_CARDINALITY_CURVE_FEATURES",
    "G3CardinalityCurveConfig",
    "G3CardinalityCurveRefiner",
    "g3_cardinality_curve_features",
    "g3_cardinality_offset_target",
]
