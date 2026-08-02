"""Whole-set coverage veto for redundant selected carrier pairs.

The veto is target-free.  It uses only the frozen pair scores, the initial
disjoint matching, pair anchor IDs, and frozen-P0 decoded finite segments.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import math
from numbers import Integral, Real
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import torch


SCHEMA_VERSION = "evidence-carrier-pair-coverage-veto-0.1"


@dataclass(frozen=True, slots=True)
class EvidenceCarrierPairCoverageConfig:
    sample_points: int = 65
    maximum_angle_degrees: float = 6.0
    angle_dot_tolerance: float = 1e-7
    tube_radius_pixels: float = 3.0
    minimum_union_common_coverage: float = 0.98
    maximum_single_common_coverage: float = 0.90
    minimum_contributor_fraction: float = 0.10
    minimum_contributor_pairs: int = 2
    minimum_segment_length_pixels: float = 1e-6

    def __post_init__(self) -> None:
        if (
            isinstance(self.sample_points, bool)
            or not isinstance(self.sample_points, Integral)
            or int(self.sample_points) < 2
        ):
            raise ValueError("sample_points must be an integer >= 2")
        if (
            isinstance(self.minimum_contributor_pairs, bool)
            or not isinstance(self.minimum_contributor_pairs, Integral)
            or int(self.minimum_contributor_pairs) < 2
        ):
            raise ValueError("minimum_contributor_pairs must be an integer >= 2")
        positive = (
            "maximum_angle_degrees",
            "tube_radius_pixels",
            "minimum_segment_length_pixels",
        )
        for name in positive:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{name} must be finite and positive")
            value = float(value)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        if self.maximum_angle_degrees > 90.0:
            raise ValueError("maximum_angle_degrees cannot exceed 90")
        if (
            isinstance(self.angle_dot_tolerance, bool)
            or not isinstance(self.angle_dot_tolerance, Real)
            or not math.isfinite(float(self.angle_dot_tolerance))
            or float(self.angle_dot_tolerance) < 0.0
        ):
            raise ValueError("angle_dot_tolerance must be finite and nonnegative")
        object.__setattr__(
            self, "angle_dot_tolerance", float(self.angle_dot_tolerance)
        )
        fractions = (
            "minimum_union_common_coverage",
            "maximum_single_common_coverage",
            "minimum_contributor_fraction",
        )
        for name in fractions:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{name} must lie in [0, 1]")
            value = float(value)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
            object.__setattr__(self, name, value)


def _array(value: Any, *, dtype: np.dtype, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    try:
        result = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be array-like") from error
    return result


def _validate_inputs(
    selected_candidate_indices: Sequence[int],
    pair_scores: Any,
    pair_indices: Any,
    decoded_segments_pixels: Any,
    *,
    config: EvidenceCarrierPairCoverageConfig,
) -> tuple[tuple[int, ...], np.ndarray, np.ndarray, np.ndarray]:
    scores = _array(pair_scores, dtype=np.float64, name="pair_scores")
    pairs = _array(pair_indices, dtype=np.int64, name="pair_indices")
    decoded = _array(
        decoded_segments_pixels,
        dtype=np.float32,
        name="decoded_segments_pixels",
    )
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("pair_scores must be a finite one-dimensional array")
    if pairs.shape != (len(scores), 2):
        raise ValueError("pair_indices must have shape [pairs,2]")
    if decoded.ndim != 3 or decoded.shape[1:] != (2, 2):
        raise ValueError("decoded_segments_pixels must have shape [anchors,2,2]")
    if not np.isfinite(decoded).all():
        raise ValueError("decoded_segments_pixels must be finite")
    if len(pairs) and (
        (pairs < 0).any()
        or (pairs >= len(decoded)).any()
        or (pairs[:, 0] >= pairs[:, 1]).any()
    ):
        raise ValueError("pair_indices must contain ascending active anchor IDs")
    if len({tuple(map(int, pair)) for pair in pairs.tolist()}) != len(pairs):
        raise ValueError("pair_indices must not contain duplicates")
    selected = []
    for value in selected_candidate_indices:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError("selected candidate indices must be integers")
        index = int(value)
        if not 0 <= index < len(pairs):
            raise ValueError("selected candidate index is out of range")
        selected.append(index)
    if len(set(selected)) != len(selected):
        raise ValueError("selected candidate indices must be unique")
    used: set[int] = set()
    for index in selected:
        anchors = set(map(int, pairs[index]))
        if used.intersection(anchors):
            raise ValueError("selected candidate pairs must be anchor-disjoint")
        used.update(anchors)
    selected_anchors = sorted(used)
    if selected_anchors:
        segments = decoded[selected_anchors]
        lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=-1)
        if bool((lengths <= config.minimum_segment_length_pixels).any()):
            raise ValueError("selected decoded segments must be nondegenerate")
    return tuple(selected), scores, pairs, decoded


@lru_cache(maxsize=16)
def _sample_parameters(sample_points: int) -> np.ndarray:
    values = np.linspace(0.0, 1.0, sample_points, dtype=np.float32)
    values.setflags(write=False)
    return values


def _coverage_masks(
    query_segments: np.ndarray,
    contributor_segments: np.ndarray,
    *,
    config: EvidenceCarrierPairCoverageConfig,
) -> np.ndarray:
    """Return [contributor_pair,query_member,samples] exact masks."""

    sample_t = _sample_parameters(int(config.sample_points))
    query_delta = query_segments[:, 1] - query_segments[:, 0]
    query_points = (
        query_segments[:, 0, None, :]
        + sample_t[None, :, None] * query_delta[:, None, :]
    )
    flat = contributor_segments.reshape((-1, 2, 2))
    contributor_delta = flat[:, 1] - flat[:, 0]
    contributor_length = np.linalg.norm(contributor_delta, axis=-1)
    if bool((contributor_length <= config.minimum_segment_length_pixels).any()):
        raise ValueError("contributor decoded segments must be nondegenerate")

    query_axis = query_delta / np.linalg.norm(query_delta, axis=-1, keepdims=True)
    contributor_axis = contributor_delta / contributor_length[:, None]
    minimum_abs_dot = math.cos(math.radians(config.maximum_angle_degrees))
    dots = np.abs(query_axis @ contributor_axis.T)
    eligible = dots + config.angle_dot_tolerance >= minimum_abs_dot

    point_x = query_points[:, :, None, 0] - flat[None, None, :, 0, 0]
    point_y = query_points[:, :, None, 1] - flat[None, None, :, 0, 1]
    contributor_x = contributor_delta[None, None, :, 0]
    contributor_y = contributor_delta[None, None, :, 1]
    denominator = np.square(contributor_length)[None, None, :]
    projection_t = (
        point_x * contributor_x + point_y * contributor_y
    ) / denominator
    projection_t = np.clip(projection_t, 0.0, 1.0)
    residual_x = point_x - projection_t * contributor_x
    residual_y = point_y - projection_t * contributor_y
    distance_squared = (
        residual_x * residual_x + residual_y * residual_y
    )
    radius_squared = config.tube_radius_pixels * config.tube_radius_pixels
    covered = (distance_squared <= radius_squared) & eligible[:, None, :]
    contributor_count = len(contributor_segments)
    return covered.reshape(
        (2, int(config.sample_points), contributor_count, 2)
    ).any(axis=-1).transpose(2, 0, 1)


def _coverage_upper_bound(
    query_segments: np.ndarray,
    contributor_segments: np.ndarray,
    *,
    config: EvidenceCarrierPairCoverageConfig,
) -> np.ndarray:
    """Return a conservative upper bound for the discrete coverage samples."""

    flat = contributor_segments.reshape((-1, 2, 2))
    contributor_delta = flat[:, 1] - flat[:, 0]
    contributor_axis = contributor_delta / np.linalg.norm(
        contributor_delta, axis=-1, keepdims=True
    )
    sample_t = _sample_parameters(int(config.sample_points))
    result = []
    for query in query_segments:
        delta = query[1] - query[0]
        length = float(np.linalg.norm(delta))
        axis = delta / length
        minimum_abs_dot = math.cos(math.radians(config.maximum_angle_degrees))
        eligible = flat[
            np.abs(contributor_axis @ axis) + config.angle_dot_tolerance
            >= minimum_abs_dot
        ]
        if not len(eligible):
            result.append(0.0)
            continue
        projection = (eligible - query[0]) @ axis
        low = projection.min(axis=-1) - config.tube_radius_pixels
        high = projection.max(axis=-1) + config.tube_radius_pixels
        low = np.nextafter(low, np.full_like(low, -np.inf))
        high = np.nextafter(high, np.full_like(high, np.inf))
        query_points = query[0] + sample_t[:, None] * delta[None, :]
        sample_positions = (query_points - query[0]) @ axis
        covered = (
            (sample_positions[:, None] >= low[None, :])
            & (sample_positions[:, None] <= high[None, :])
        )
        result.append(float(covered.any(axis=1).mean()))
    return np.asarray(result, dtype=np.float64)


def apply_synergistic_pair_coverage_veto(
    selected_candidate_indices: Sequence[int],
    pair_scores: Any,
    pair_indices: Any,
    decoded_segments_pixels: Any,
    *,
    config: EvidenceCarrierPairCoverageConfig = EvidenceCarrierPairCoverageConfig(),
) -> dict[str, Any]:
    """Synchronously veto selected pairs explained only by stronger-pair synergy."""

    started = perf_counter()
    selected, scores, pairs, decoded = _validate_inputs(
        selected_candidate_indices,
        pair_scores,
        pair_indices,
        decoded_segments_pixels,
        config=config,
    )
    ranked = tuple(
        sorted(
            selected,
            key=lambda index: (
                -float(scores[index]),
                int(pairs[index, 0]),
                int(pairs[index, 1]),
            ),
        )
    )
    ranked_segments = decoded[pairs[np.asarray(ranked, dtype=np.int64)]]
    vetoed: list[int] = []
    diagnostics: list[dict[str, Any]] = []
    for rank, candidate_index in enumerate(ranked):
        if rank < config.minimum_contributor_pairs:
            continue
        contributor_indices = ranked[:rank]
        query_segments = ranked_segments[rank]
        contributor_segments = ranked_segments[:rank]
        upper_bound = _coverage_upper_bound(
            query_segments,
            contributor_segments,
            config=config,
        )
        if float(upper_bound.min()) + 1e-12 < config.minimum_union_common_coverage:
            continue
        masks = _coverage_masks(
            query_segments,
            contributor_segments,
            config=config,
        )
        coverage = masks.mean(axis=-1)
        union_coverage = masks.any(axis=0).mean(axis=-1)
        union_common = float(union_coverage.min())
        single_common = coverage.min(axis=-1)
        best_single_common = float(single_common.max())
        contributor_fraction = coverage.mean(axis=-1)
        active_contributors = np.where(
            contributor_fraction + 1e-12 >= config.minimum_contributor_fraction
        )[0]
        should_veto = (
            union_common + 1e-12 >= config.minimum_union_common_coverage
            and len(active_contributors) >= config.minimum_contributor_pairs
            and best_single_common < config.maximum_single_common_coverage - 1e-12
        )
        if not should_veto:
            continue
        vetoed.append(candidate_index)
        diagnostics.append(
            {
                "candidate_index": candidate_index,
                "anchor_ids": [int(value) for value in pairs[candidate_index]],
                "raw_score": float(scores[candidate_index]),
                "rank": rank,
                "union_coverage": [float(value) for value in union_coverage],
                "union_coverage_upper_bound": [
                    float(value) for value in upper_bound
                ],
                "union_common_coverage": union_common,
                "best_single_common_coverage": best_single_common,
                "contributor_candidate_indices": [
                    int(contributor_indices[index]) for index in active_contributors
                ],
                "contributor_count": int(len(active_contributors)),
            }
        )
    vetoed_set = set(vetoed)
    kept = tuple(index for index in selected if index not in vetoed_set)
    return {
        "schema_version": SCHEMA_VERSION,
        "config": asdict(config),
        "initial_selected_candidate_indices": list(selected),
        "kept_candidate_indices": list(kept),
        "vetoed_candidate_indices": list(vetoed),
        "initial_selected_pair_count": len(selected),
        "kept_pair_count": len(kept),
        "vetoed_pair_count": len(vetoed),
        "veto_diagnostics": diagnostics,
        "latency_ms": 1000.0 * (perf_counter() - started),
        "target_free": True,
        "synchronous_veto": True,
    }


__all__ = [
    "SCHEMA_VERSION",
    "EvidenceCarrierPairCoverageConfig",
    "apply_synergistic_pair_coverage_veto",
]
