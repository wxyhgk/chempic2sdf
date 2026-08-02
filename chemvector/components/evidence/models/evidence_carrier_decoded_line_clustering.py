"""Target-free clustering for decoded Evidence Carrier lines.

The clustering boundary deliberately accepts no target identifiers.  It only
uses a decoded finite line, its original LSD anchor, and a stable source ID.
Segments are canonicalized before deterministic complete-link agglomeration.
Each output representative is an actual cluster member (a medoid); coordinates
are never averaged.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class DecodedLineClusteringConfig:
    """Complete-link thresholds in the native 768px/q2047 protocol."""

    maximum_angle_degrees: float = 2.0
    maximum_normal_distance_pixels: float = 2.0
    minimum_interval_iou: float = 0.85
    maximum_endpoint_error_q: float = 8.0
    coordinate_max: float = 2047.0
    minimum_line_length_pixels: float = 1e-6

    def __post_init__(self) -> None:
        positive = (
            "maximum_angle_degrees",
            "maximum_normal_distance_pixels",
            "maximum_endpoint_error_q",
            "coordinate_max",
            "minimum_line_length_pixels",
        )
        for name in positive:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{name} must be a finite positive number")
            value = float(value)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a finite positive number")
            object.__setattr__(self, name, value)
        iou = self.minimum_interval_iou
        if isinstance(iou, bool) or not isinstance(iou, Real):
            raise ValueError("minimum_interval_iou must lie in [0, 1)")
        iou = float(iou)
        if not math.isfinite(iou) or not 0.0 <= iou < 1.0:
            raise ValueError("minimum_interval_iou must lie in [0, 1)")
        object.__setattr__(self, "minimum_interval_iou", iou)
        if self.maximum_angle_degrees > 90.0:
            raise ValueError("maximum_angle_degrees cannot exceed 90")


@dataclass(frozen=True, slots=True)
class DecodedLinePairGeometry:
    """Symmetric geometry used by complete-link and medoid selection."""

    angle_degrees: float
    normal_distance_pixels: float
    interval_iou: float
    endpoint_error_q: float
    normalized_distance: float
    compatible: bool


@dataclass(frozen=True, slots=True)
class DecodedLineCluster:
    """One deterministic cluster and its observed-member representative."""

    member_indices: tuple[int, ...]
    member_source_ids: tuple[str, ...]
    representative_index: int
    representative_source_id: str
    representative_segment_pixels: tuple[
        tuple[float, float], tuple[float, float]
    ]


@dataclass(frozen=True, slots=True)
class _LineRecord:
    input_index: int
    source_id: str
    segment: np.ndarray
    anchor: np.ndarray
    axis: np.ndarray
    normal: np.ndarray
    center: np.ndarray
    sort_key: tuple[Any, ...]


def canonicalize_decoded_segment(segment: Any) -> np.ndarray:
    """Return a finite segment whose axis points +x, or +y if vertical."""

    value = np.asarray(segment, dtype=np.float64)
    if value.shape != (2, 2):
        raise ValueError("Decoded lines must have shape [2,2]")
    if not np.isfinite(value).all():
        raise ValueError("Decoded lines must contain only finite coordinates")
    delta = value[1] - value[0]
    if float(np.linalg.norm(delta)) <= 0.0:
        raise ValueError("Decoded lines must have nonzero length")
    if float(delta[0]) < 0.0 or (
        float(delta[0]) == 0.0 and float(delta[1]) < 0.0
    ):
        value = value[::-1].copy()
    else:
        value = value.copy()
    return value


def _segment_key(segment: np.ndarray) -> tuple[float, ...]:
    return tuple(float(value) for value in segment.reshape(-1))


def _validate_canvas(width: int, height: int) -> tuple[int, int]:
    if isinstance(width, bool) or isinstance(height, bool):
        raise ValueError("Canvas dimensions must be integers >= 2")
    width = int(width)
    height = int(height)
    if width < 2 or height < 2:
        raise ValueError("Canvas dimensions must be integers >= 2")
    return width, height


def _pair_geometry(
    first: _LineRecord,
    second: _LineRecord,
    *,
    width: int,
    height: int,
    config: DecodedLineClusteringConfig,
) -> DecodedLinePairGeometry:
    dot = float(np.clip(np.dot(first.axis, second.axis), -1.0, 1.0))
    angle = math.degrees(math.acos(abs(dot)))
    center_delta = second.center - first.center
    normal_distance = max(
        abs(float(np.dot(center_delta, first.normal))),
        abs(float(np.dot(center_delta, second.normal))),
    )

    second_axis = second.axis if dot >= 0.0 else -second.axis
    common_axis = first.axis + second_axis
    common_length = float(np.linalg.norm(common_axis))
    if common_length <= config.minimum_line_length_pixels:
        common_axis = first.axis
    else:
        common_axis = common_axis / common_length
    first_interval = np.sort(first.segment @ common_axis)
    second_interval = np.sort(second.segment @ common_axis)
    intersection = max(
        0.0,
        min(float(first_interval[1]), float(second_interval[1]))
        - max(float(first_interval[0]), float(second_interval[0])),
    )
    union = max(float(first_interval[1]), float(second_interval[1])) - min(
        float(first_interval[0]), float(second_interval[0])
    )
    interval_iou = intersection / union if union > 0.0 else 0.0

    direct = np.linalg.norm(first.segment - second.segment, axis=-1).max()
    reverse = np.linalg.norm(first.segment - second.segment[::-1], axis=-1).max()
    endpoint_error_pixels = float(min(direct, reverse))
    q_per_pixel = max(
        config.coordinate_max / float(width - 1),
        config.coordinate_max / float(height - 1),
    )
    endpoint_error_q = endpoint_error_pixels * q_per_pixel

    normalized_distance = max(
        angle / config.maximum_angle_degrees,
        normal_distance / config.maximum_normal_distance_pixels,
        (1.0 - interval_iou) / (1.0 - config.minimum_interval_iou),
        endpoint_error_q / config.maximum_endpoint_error_q,
    )
    epsilon = 1e-12
    compatible = (
        angle <= config.maximum_angle_degrees + epsilon
        and normal_distance <= config.maximum_normal_distance_pixels + epsilon
        and interval_iou + epsilon >= config.minimum_interval_iou
        and endpoint_error_q <= config.maximum_endpoint_error_q + epsilon
    )
    return DecodedLinePairGeometry(
        angle_degrees=angle,
        normal_distance_pixels=normal_distance,
        interval_iou=interval_iou,
        endpoint_error_q=endpoint_error_q,
        normalized_distance=normalized_distance,
        compatible=compatible,
    )


def decoded_line_pair_geometry(
    first_segment_pixels: Any,
    second_segment_pixels: Any,
    *,
    width: int,
    height: int,
    config: DecodedLineClusteringConfig = DecodedLineClusteringConfig(),
) -> DecodedLinePairGeometry:
    """Measure the four target-free pair criteria for two decoded lines."""

    width, height = _validate_canvas(width, height)
    records = _build_records(
        np.stack((first_segment_pixels, second_segment_pixels)),
        np.stack((first_segment_pixels, second_segment_pixels)),
        ("first", "second"),
        config=config,
    )
    return _pair_geometry(
        records[0], records[1], width=width, height=height, config=config
    )


def _build_records(
    decoded_segments_pixels: Any,
    anchor_segments_pixels: Any,
    source_ids: Sequence[str],
    *,
    config: DecodedLineClusteringConfig,
) -> list[_LineRecord]:
    decoded = np.asarray(decoded_segments_pixels, dtype=np.float64)
    anchors = np.asarray(anchor_segments_pixels, dtype=np.float64)
    if decoded.ndim != 3 or decoded.shape[1:] != (2, 2):
        raise ValueError("decoded_segments_pixels must have shape [lines,2,2]")
    if anchors.shape != decoded.shape:
        raise ValueError("anchor_segments_pixels must match decoded line shape")
    if len(source_ids) != len(decoded):
        raise ValueError("source_ids must contain one ID per decoded line")
    normalized_ids = tuple(str(value) for value in source_ids)
    if any(not value for value in normalized_ids):
        raise ValueError("source_ids must be nonempty")
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ValueError("source_ids must be unique")
    records = []
    for index, source_id in enumerate(normalized_ids):
        segment = canonicalize_decoded_segment(decoded[index])
        anchor = canonicalize_decoded_segment(anchors[index])
        delta = segment[1] - segment[0]
        length = float(np.linalg.norm(delta))
        if length < config.minimum_line_length_pixels:
            raise ValueError("Decoded line is shorter than the configured minimum")
        axis = delta / length
        normal = np.asarray((-axis[1], axis[0]), dtype=np.float64)
        center = segment.mean(axis=0)
        key = (_segment_key(segment), _segment_key(anchor), source_id)
        records.append(
            _LineRecord(
                input_index=index,
                source_id=source_id,
                segment=segment,
                anchor=anchor,
                axis=axis,
                normal=normal,
                center=center,
                sort_key=key,
            )
        )
    records.sort(key=lambda record: record.sort_key)
    return records


def _cluster_key(cluster: tuple[int, ...], records: list[_LineRecord]) -> tuple:
    return tuple(records[index].sort_key for index in cluster)


def _medoid_index(
    cluster: tuple[int, ...],
    records: list[_LineRecord],
    distances: np.ndarray,
) -> int:
    ranked = []
    for candidate in cluster:
        total = math.fsum(float(distances[candidate, other]) for other in cluster)
        ranked.append((total, records[candidate].sort_key, candidate))
    return min(ranked)[2]


def cluster_decoded_lines(
    decoded_segments_pixels: Any,
    anchor_segments_pixels: Any,
    source_ids: Sequence[str],
    *,
    width: int,
    height: int,
    config: DecodedLineClusteringConfig = DecodedLineClusteringConfig(),
) -> tuple[DecodedLineCluster, ...]:
    """Cluster decoded lines with deterministic complete-link agglomeration.

    Every possible merge must satisfy all four thresholds for every cross-pair.
    The next merge is the compatible pair with the smallest complete-link
    normalized distance.  Geometry, anchor geometry, and stable source IDs
    make both merge and medoid tie-breaking independent of input order.
    """

    width, height = _validate_canvas(width, height)
    records = _build_records(
        decoded_segments_pixels,
        anchor_segments_pixels,
        source_ids,
        config=config,
    )
    count = len(records)
    if count == 0:
        return ()
    distances = np.zeros((count, count), dtype=np.float64)
    compatible = np.ones((count, count), dtype=bool)
    for first in range(count):
        for second in range(first + 1, count):
            geometry = _pair_geometry(
                records[first],
                records[second],
                width=width,
                height=height,
                config=config,
            )
            distances[first, second] = distances[second, first] = (
                geometry.normalized_distance
            )
            compatible[first, second] = compatible[second, first] = (
                geometry.compatible
            )

    clusters: list[tuple[int, ...]] = [(index,) for index in range(count)]
    while True:
        candidates = []
        for first_index, first in enumerate(clusters):
            for second_index in range(first_index + 1, len(clusters)):
                second = clusters[second_index]
                cross_pairs = [(left, right) for left in first for right in second]
                if not all(compatible[left, right] for left, right in cross_pairs):
                    continue
                complete_distance = max(
                    float(distances[left, right]) for left, right in cross_pairs
                )
                merged = tuple(sorted((*first, *second)))
                candidates.append(
                    (
                        complete_distance,
                        _cluster_key(merged, records),
                        first_index,
                        second_index,
                        merged,
                    )
                )
        if not candidates:
            break
        _, _, first_index, second_index, merged = min(candidates)
        clusters = [
            cluster
            for index, cluster in enumerate(clusters)
            if index not in (first_index, second_index)
        ]
        clusters.append(merged)
        clusters.sort(key=lambda cluster: _cluster_key(cluster, records))

    output = []
    for cluster in sorted(clusters, key=lambda value: _cluster_key(value, records)):
        medoid = _medoid_index(cluster, records, distances)
        member_records = [records[index] for index in cluster]
        representative = records[medoid]
        output.append(
            DecodedLineCluster(
                member_indices=tuple(record.input_index for record in member_records),
                member_source_ids=tuple(record.source_id for record in member_records),
                representative_index=representative.input_index,
                representative_source_id=representative.source_id,
                representative_segment_pixels=tuple(
                    tuple(float(value) for value in point)
                    for point in representative.segment
                ),
            )
        )
    return tuple(output)


__all__ = [
    "DecodedLineCluster",
    "DecodedLineClusteringConfig",
    "DecodedLinePairGeometry",
    "canonicalize_decoded_segment",
    "cluster_decoded_lines",
    "decoded_line_pair_geometry",
]
