"""Deterministic pair evidence over wide semantic endpoint candidates."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class SemanticEndpointPairConfig:
    longitudinal_samples: int = 48
    transverse_samples: int = 9
    ribbon_radius_pixels: float = 8.0
    blocker_radius_pixels: float = 6.0
    minimum_pair_distance_pixels: float = 4.0
    image_size: float = 768.0
    pair_chunk_size: int = 2048

    def __post_init__(self) -> None:
        if min(
            self.longitudinal_samples,
            self.transverse_samples,
            self.pair_chunk_size,
        ) < 1:
            raise ValueError("Semantic endpoint pair dimensions must be positive")
        if min(
            self.ribbon_radius_pixels,
            self.blocker_radius_pixels,
            self.minimum_pair_distance_pixels,
            self.image_size,
        ) <= 0.0:
            raise ValueError("Semantic endpoint pair geometry scales must be positive")


def enumerate_endpoint_pairs(points: torch.Tensor) -> torch.Tensor:
    if points.ndim != 2 or points.size(-1) != 2:
        raise ValueError("Endpoint points must have shape [points, 2]")
    return torch.combinations(
        torch.arange(len(points), device=points.device, dtype=torch.long), r=2
    )


def _sample_pair_ribbons(
    bond_probability: torch.Tensor,
    points: torch.Tensor,
    pairs: torch.Tensor,
    config: SemanticEndpointPairConfig,
) -> torch.Tensor:
    if bond_probability.ndim != 4 or bond_probability.shape[:2] != (1, 1):
        raise ValueError("Pair ribbon input must have shape [1, 1, H, W]")
    if pairs.ndim != 2 or pairs.size(-1) != 2:
        raise ValueError("Endpoint pair indices must have shape [pairs, 2]")
    if not len(pairs):
        return bond_probability.new_empty(
            (0, config.transverse_samples, config.longitudinal_samples)
        )
    begin = points[pairs[:, 0]]
    end = points[pairs[:, 1]]
    delta = end - begin
    height, width = bond_probability.shape[-2:]
    delta_pixels = torch.stack(
        (delta[:, 0] * width, delta[:, 1] * height), dim=-1
    )
    lengths = delta_pixels.norm(dim=-1).clamp_min(1e-6)
    direction = delta_pixels / lengths[:, None]
    perpendicular = torch.stack((-direction[:, 1], direction[:, 0]), dim=-1)
    perpendicular = torch.stack(
        (perpendicular[:, 0] / width, perpendicular[:, 1] / height), dim=-1
    )
    longitudinal = torch.linspace(
        0.06,
        0.94,
        config.longitudinal_samples,
        device=points.device,
        dtype=points.dtype,
    )
    transverse = torch.linspace(
        -config.ribbon_radius_pixels,
        config.ribbon_radius_pixels,
        config.transverse_samples,
        device=points.device,
        dtype=points.dtype,
    )
    base = begin[:, None, :] + delta[:, None, :] * longitudinal[None, :, None]
    grid = base[:, None, :, :] + (
        perpendicular[:, None, None, :]
        * transverse[None, :, None, None]
    )
    grid = grid.mul(2.0).sub(1.0).reshape(
        1,
        len(pairs) * config.transverse_samples,
        config.longitudinal_samples,
        2,
    )
    return F.grid_sample(
        bond_probability.float(),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).reshape(
        len(pairs), config.transverse_samples, config.longitudinal_samples
    )


def _blocker_support(
    points: torch.Tensor,
    point_scores: torch.Tensor,
    pairs: torch.Tensor,
    config: SemanticEndpointPairConfig,
) -> torch.Tensor:
    begin = points[pairs[:, 0]]
    end = points[pairs[:, 1]]
    delta = end - begin
    squared_length = delta.square().sum(-1).clamp_min(1e-8)
    relative = points[None, :, :] - begin[:, None, :]
    projection = (relative * delta[:, None, :]).sum(-1) / squared_length[:, None]
    closest = begin[:, None, :] + projection[..., None] * delta[:, None, :]
    distance = (points[None, :, :] - closest).norm(dim=-1) * config.image_size
    interior = (projection >= 0.08) & (projection <= 0.92)
    support = point_scores[None] * torch.exp(
        -0.5 * (distance / config.blocker_radius_pixels).square()
    )
    support = support.masked_fill(~interior, 0.0)
    pair_rows = torch.arange(len(pairs), device=points.device)
    support[pair_rows, pairs[:, 0]] = 0.0
    support[pair_rows, pairs[:, 1]] = 0.0
    return support.amax(dim=-1)


@torch.no_grad()
def score_endpoint_pairs(
    bond_probability: torch.Tensor,
    points: torch.Tensor,
    point_scores: torch.Tensor,
    config: SemanticEndpointPairConfig = SemanticEndpointPairConfig(),
) -> dict[str, torch.Tensor]:
    if points.ndim != 2 or points.size(-1) != 2:
        raise ValueError("Endpoint points must have shape [points, 2]")
    if point_scores.shape != (len(points),):
        raise ValueError("Endpoint scores differ from endpoint points")
    pairs = enumerate_endpoint_pairs(points)
    if not len(pairs):
        empty = points.new_empty(0)
        return {
            "pairs": pairs,
            "length_pixels": empty,
            "line_mean": empty,
            "line_lower_quantile": empty,
            "line_coverage": empty,
            "blocker": empty,
            "endpoint_score": empty,
            "line_score": empty,
            "line_blocker_score": empty,
            "line_gated_blocker_score": empty,
        }

    outputs: dict[str, list[torch.Tensor]] = {
        "length_pixels": [],
        "line_mean": [],
        "line_lower_quantile": [],
        "line_coverage": [],
        "blocker": [],
        "endpoint_score": [],
        "line_score": [],
        "line_blocker_score": [],
        "line_gated_blocker_score": [],
    }
    lower_index = max(1, int(config.longitudinal_samples * 0.2))
    for start in range(0, len(pairs), config.pair_chunk_size):
        selected = pairs[start : start + config.pair_chunk_size]
        ribbons = _sample_pair_ribbons(
            bond_probability, points, selected, config
        )
        longitudinal_support = ribbons.amax(dim=1)
        line_mean = longitudinal_support.mean(dim=-1)
        lower = longitudinal_support.kthvalue(lower_index, dim=-1).values
        coverage = (longitudinal_support >= 0.5).float().mean(dim=-1)
        endpoint_score = torch.minimum(
            point_scores[selected[:, 0]], point_scores[selected[:, 1]]
        )
        blocker = _blocker_support(points, point_scores, selected, config)
        lengths = (
            (points[selected[:, 1]] - points[selected[:, 0]]).norm(dim=-1)
            * config.image_size
        )
        line_score = 0.5 * lower + 0.3 * line_mean + 0.2 * coverage
        line_score = line_score + 0.05 * endpoint_score
        valid = lengths >= config.minimum_pair_distance_pixels
        outputs["length_pixels"].append(lengths)
        outputs["line_mean"].append(line_mean)
        outputs["line_lower_quantile"].append(lower)
        outputs["line_coverage"].append(coverage)
        outputs["blocker"].append(blocker)
        outputs["endpoint_score"].append(
            endpoint_score.masked_fill(~valid, -1e6)
        )
        outputs["line_score"].append(line_score.masked_fill(~valid, -1e6))
        outputs["line_blocker_score"].append(
            (line_score - 0.5 * blocker).masked_fill(~valid, -1e6)
        )
        gated_blocker = ((blocker - 0.5) / 0.5).clamp_min(0.0)
        outputs["line_gated_blocker_score"].append(
            (line_score - 0.5 * gated_blocker).masked_fill(~valid, -1e6)
        )
    return {"pairs": pairs, **{key: torch.cat(value) for key, value in outputs.items()}}


__all__ = [
    "SemanticEndpointPairConfig",
    "enumerate_endpoint_pairs",
    "score_endpoint_pairs",
]
