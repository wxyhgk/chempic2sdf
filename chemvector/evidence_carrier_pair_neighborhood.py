"""Target-free near-collinear anchor neighborhood for one carrier pair."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np
import torch


PAIR_NEIGHBORHOOD_SCHEMA_VERSION = "evidence-carrier-pair-neighborhood-0.1"
PAIR_NEIGHBORHOOD_FEATURE_NAMES = (
    "signed_normal_offset_over_pair_separation",
    "angle_difference_over_limit",
    "whole_overlap_fraction",
    "left_half_overlap_fraction",
    "right_half_overlap_fraction",
    "relative_start",
    "relative_end",
    "length_ratio",
)
PAIR_NEIGHBORHOOD_FEATURE_DIM = len(PAIR_NEIGHBORHOOD_FEATURE_NAMES)


@dataclass(frozen=True, slots=True)
class PairNeighborhoodConfig:
    maximum_neighbors: int = 12
    maximum_angle_degrees: float = 8.0
    minimum_overlap_fraction: float = 0.20
    maximum_normal_offset_ratio: float = 4.0
    maximum_relative_extent: float = 2.0
    maximum_length_ratio: float = 2.0

    def __post_init__(self) -> None:
        if isinstance(self.maximum_neighbors, bool) or not isinstance(
            self.maximum_neighbors, int
        ) or self.maximum_neighbors < 1:
            raise ValueError("maximum_neighbors must be a positive integer")
        for name in (
            "maximum_angle_degrees",
            "minimum_overlap_fraction",
            "maximum_normal_offset_ratio",
            "maximum_relative_extent",
            "maximum_length_ratio",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be finite and positive")
            value = float(value)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        if self.maximum_angle_degrees > 90.0:
            raise ValueError("maximum_angle_degrees must not exceed 90")
        if self.minimum_overlap_fraction > 1.0:
            raise ValueError("minimum_overlap_fraction must not exceed 1")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PAIR_NEIGHBORHOOD_SCHEMA_VERSION,
            **asdict(self),
            "feature_names": list(PAIR_NEIGHBORHOOD_FEATURE_NAMES),
            "selection": "nearest_absolute_normal_offset_then_signed_sort",
        }


def _segments(value: Any) -> np.ndarray:
    segments = np.asarray(value, dtype=np.float64)
    if segments.ndim != 3 or segments.shape[1:] != (2, 2):
        raise ValueError("anchor segments must have shape [anchors,2,2]")
    if not np.isfinite(segments).all():
        raise ValueError("anchor segments contain non-finite values")
    if len(segments) and (
        np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1) <= 1e-8
    ).any():
        raise ValueError("anchor segments must be nondegenerate")
    return segments


def _pairs(value: Any, anchor_count: int) -> np.ndarray:
    pairs = np.asarray(value, dtype=np.int64)
    if pairs.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pair indices must have shape [pairs,2]")
    if (pairs < 0).any() or (pairs >= anchor_count).any():
        raise ValueError("pair indices exceed anchors")
    if (pairs[:, 0] == pairs[:, 1]).any():
        raise ValueError("pair indices must reference distinct anchors")
    return pairs


def _canonical_axis(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first_axis = first[1] - first[0]
    second_axis = second[1] - second[0]
    first_axis /= np.linalg.norm(first_axis)
    second_axis /= np.linalg.norm(second_axis)
    if float(first_axis @ second_axis) < 0.0:
        second_axis = -second_axis
    axis = first_axis + second_axis
    norm = float(np.linalg.norm(axis))
    axis = first_axis if norm <= 1e-8 else axis / norm
    if float(axis[0]) < 0.0 or (
        abs(float(axis[0])) <= 1e-12 and float(axis[1]) < 0.0
    ):
        axis = -axis
    return axis


def _overlap(first: tuple[float, float], second: tuple[float, float]) -> float:
    return max(0.0, min(first[1], second[1]) - max(first[0], second[0]))


def build_pair_neighborhood(
    anchor_segments_pixels: Any,
    pair_indices: Any,
    *,
    config: PairNeighborhoodConfig = PairNeighborhoodConfig(),
) -> dict[str, torch.Tensor]:
    """Return fixed ordered neighbor tokens without target-aware inputs."""

    segments = _segments(anchor_segments_pixels)
    pairs = _pairs(pair_indices, len(segments))
    features = np.zeros(
        (
            len(pairs),
            config.maximum_neighbors,
            PAIR_NEIGHBORHOOD_FEATURE_DIM,
        ),
        dtype=np.float32,
    )
    mask = np.zeros((len(pairs), config.maximum_neighbors), dtype=np.bool_)
    source_indices = np.full(
        (len(pairs), config.maximum_neighbors), -1, dtype=np.int64
    )
    for pair_offset, (first_index, second_index) in enumerate(pairs.tolist()):
        first, second = segments[first_index], segments[second_index]
        axis = _canonical_axis(first, second)
        normal = np.asarray((-axis[1], axis[0]), dtype=np.float64)
        first_interval = tuple(np.sort(first @ axis).tolist())
        second_interval = tuple(np.sort(second @ axis).tolist())
        lower = max(first_interval[0], second_interval[0])
        upper = min(first_interval[1], second_interval[1])
        if upper - lower <= 1e-8:
            lower = min(first_interval[0], second_interval[0])
            upper = max(first_interval[1], second_interval[1])
        length = max(upper - lower, 1e-6)
        midpoint = 0.5 * (lower + upper)
        first_normal = float(first.mean(axis=0) @ normal)
        second_normal = float(second.mean(axis=0) @ normal)
        center_normal = 0.5 * (first_normal + second_normal)
        separation = max(abs(first_normal - second_normal), 1.0)
        candidates = []
        for anchor_index, segment in enumerate(segments):
            if anchor_index in (first_index, second_index):
                continue
            delta = segment[1] - segment[0]
            anchor_length = float(np.linalg.norm(delta))
            unit = delta / anchor_length
            cosine = min(1.0, abs(float(unit @ axis)))
            angle = math.degrees(math.acos(cosine))
            if angle > config.maximum_angle_degrees:
                continue
            interval_values = np.sort(segment @ axis)
            interval = (float(interval_values[0]), float(interval_values[1]))
            whole_overlap = _overlap((lower, upper), interval) / length
            if whole_overlap < config.minimum_overlap_fraction:
                continue
            signed_offset = (
                float(segment.mean(axis=0) @ normal) - center_normal
            ) / separation
            if abs(signed_offset) > config.maximum_normal_offset_ratio:
                continue
            half_length = max(0.5 * length, 1e-6)
            left_overlap = _overlap((lower, midpoint), interval) / half_length
            right_overlap = _overlap((midpoint, upper), interval) / half_length
            token = np.asarray(
                (
                    np.clip(
                        signed_offset / config.maximum_normal_offset_ratio,
                        -1.0,
                        1.0,
                    ),
                    np.clip(angle / config.maximum_angle_degrees, 0.0, 1.0),
                    np.clip(whole_overlap, 0.0, 1.0),
                    np.clip(left_overlap, 0.0, 1.0),
                    np.clip(right_overlap, 0.0, 1.0),
                    np.clip(
                        (interval[0] - lower)
                        / length
                        / config.maximum_relative_extent,
                        -1.0,
                        1.0,
                    ),
                    np.clip(
                        (interval[1] - upper)
                        / length
                        / config.maximum_relative_extent,
                        -1.0,
                        1.0,
                    ),
                    np.clip(
                        anchor_length / length / config.maximum_length_ratio,
                        0.0,
                        1.0,
                    ),
                ),
                dtype=np.float32,
            )
            candidates.append((abs(signed_offset), signed_offset, anchor_index, token))
        selected = sorted(candidates, key=lambda row: (row[0], row[1], row[2]))[
            : config.maximum_neighbors
        ]
        selected.sort(key=lambda row: (row[1], row[2]))
        for neighbor_offset, (_, _, anchor_index, token) in enumerate(selected):
            features[pair_offset, neighbor_offset] = token
            mask[pair_offset, neighbor_offset] = True
            source_indices[pair_offset, neighbor_offset] = anchor_index
    return {
        "neighbor_features": torch.from_numpy(features),
        "neighbor_mask": torch.from_numpy(mask),
        "neighbor_source_indices": torch.from_numpy(source_indices),
    }


__all__ = [
    "PAIR_NEIGHBORHOOD_FEATURE_DIM",
    "PAIR_NEIGHBORHOOD_FEATURE_NAMES",
    "PairNeighborhoodConfig",
    "build_pair_neighborhood",
]
