"""Length-independent raster line proposals for semantic AtomPair bonds.

The dense BondObject head predicts a center, orientation, and finite extent from
one feature-map cell.  This module provides a complementary SVG-like path: LSD
observes a complete raster segment, then its two endpoints are attached to
predicted AtomObjects.  No maximum bond length is used anywhere in this path.

Multiple raster boundaries or physical lines that attach to the same unordered
AtomPair remain one semantic candidate.  Bond order is intentionally left to the
learned AtomPair group head.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError as error:  # pragma: no cover - checked by the CLI environment
    raise RuntimeError("line_segment_rescue requires opencv-python-headless") from error


@dataclass(frozen=True, slots=True)
class LineSegmentRescueConfig:
    minimum_length_pixels: float = 5.0
    maximum_endpoint_distance_pixels: float = 16.0
    maximum_angle_degrees: float = 12.0
    border_margin_pixels: float = 2.0
    maximum_candidates: int = 256

    def __post_init__(self) -> None:
        for name in (
            "minimum_length_pixels",
            "maximum_endpoint_distance_pixels",
            "maximum_angle_degrees",
            "border_margin_pixels",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{name} must be a finite non-negative number")
            value = float(value)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite non-negative number")
            object.__setattr__(self, name, value)
        if self.minimum_length_pixels <= 0.0:
            raise ValueError("minimum_length_pixels must be positive")
        if self.maximum_endpoint_distance_pixels <= 0.0:
            raise ValueError("maximum_endpoint_distance_pixels must be positive")
        if not 0.0 < self.maximum_angle_degrees <= 90.0:
            raise ValueError("maximum_angle_degrees must lie in (0, 90]")
        if isinstance(self.maximum_candidates, bool) or int(self.maximum_candidates) < 1:
            raise ValueError("maximum_candidates must be a positive integer")
        object.__setattr__(self, "maximum_candidates", int(self.maximum_candidates))


@dataclass(frozen=True, slots=True)
class AtomPairLineCandidate:
    atom_pair: tuple[int, int]
    representative_segment_pixels: tuple[
        tuple[float, float], tuple[float, float]
    ]
    support_segment_indices: tuple[int, ...]
    endpoint_error_pixels: float
    angle_error_degrees: float
    maximum_support_length_pixels: float


def _canonical_segment(segment: Any) -> np.ndarray:
    value = np.asarray(segment, dtype=np.float64)
    if value.shape != (2, 2):
        raise ValueError("Line segments must have shape [2,2]")
    if not np.isfinite(value).all():
        raise ValueError("Line segments must contain finite coordinates")
    delta = value[1] - value[0]
    if float(np.linalg.norm(delta)) <= 0.0:
        raise ValueError("Line segments must be nondegenerate")
    if (float(value[0, 1]), float(value[0, 0])) > (
        float(value[1, 1]),
        float(value[1, 0]),
    ):
        return value[::-1].copy()
    return value.copy()


def _is_canvas_border(
    segment: np.ndarray,
    *,
    width: int,
    height: int,
    margin: float,
) -> bool:
    if margin <= 0.0:
        return False
    x = segment[:, 0]
    y = segment[:, 1]
    return bool(
        np.all(x <= margin)
        or np.all(x >= float(width - 1) - margin)
        or np.all(y <= margin)
        or np.all(y >= float(height - 1) - margin)
    )


def detect_raster_line_segments(
    image: Image.Image,
    *,
    config: LineSegmentRescueConfig = LineSegmentRescueConfig(),
) -> np.ndarray:
    """Detect full-image raster segments without imposing a maximum length."""

    grayscale = np.asarray(image.convert("L"), dtype=np.uint8)
    ink = 255 - grayscale
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    detected = detector.detect(ink)[0]
    if detected is None:
        return np.empty((0, 2, 2), dtype=np.float64)
    height, width = grayscale.shape
    segments = []
    for raw in detected:
        segment = _canonical_segment(np.asarray(raw).reshape(-1)[:4].reshape(2, 2))
        if float(np.linalg.norm(segment[1] - segment[0])) < config.minimum_length_pixels:
            continue
        if _is_canvas_border(
            segment,
            width=width,
            height=height,
            margin=config.border_margin_pixels,
        ):
            continue
        segments.append(segment)
    if not segments:
        return np.empty((0, 2, 2), dtype=np.float64)
    return np.stack(segments)


def attach_segments_to_atom_pairs(
    segments_pixels: Any,
    atom_anchors_pixels: Any,
    *,
    atom_mask: Any | None = None,
    config: LineSegmentRescueConfig = LineSegmentRescueConfig(),
) -> tuple[AtomPairLineCandidate, ...]:
    """Attach observed segment endpoints to predicted AtomObjects.

    Endpoint distance and angular agreement are local tolerances.  The distance
    between the two atoms is never upper-bounded, so a clear 150px bond is no
    harder to represent than a clear 30px bond.
    """

    segments = np.asarray(segments_pixels, dtype=np.float64)
    if segments.size == 0:
        segments = np.empty((0, 2, 2), dtype=np.float64)
    if segments.ndim != 3 or segments.shape[1:] != (2, 2):
        raise ValueError("segments_pixels must have shape [segments,2,2]")
    anchors = np.asarray(atom_anchors_pixels, dtype=np.float64)
    if anchors.ndim != 2 or anchors.shape[1:] != (2,):
        raise ValueError("atom_anchors_pixels must have shape [atoms,2]")
    if not np.isfinite(anchors).all():
        raise ValueError("atom_anchors_pixels must contain finite coordinates")
    if atom_mask is None:
        active = np.arange(len(anchors), dtype=np.int64)
    else:
        mask = np.asarray(atom_mask, dtype=bool)
        if mask.shape != (len(anchors),):
            raise ValueError("atom_mask must contain one flag per atom")
        active = np.flatnonzero(mask)
    if not len(active) or not len(segments):
        return ()

    active_anchors = anchors[active]
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for segment_index, raw_segment in enumerate(segments):
        segment = _canonical_segment(raw_segment)
        endpoint_distances = np.linalg.norm(
            segment[:, None, :] - active_anchors[None, :, :], axis=-1
        )
        local_begin = int(endpoint_distances[0].argmin())
        local_end = int(endpoint_distances[1].argmin())
        if local_begin == local_end:
            continue
        begin_error = float(endpoint_distances[0, local_begin])
        end_error = float(endpoint_distances[1, local_end])
        endpoint_error = max(begin_error, end_error)
        if endpoint_error > config.maximum_endpoint_distance_pixels:
            continue
        begin = int(active[local_begin])
        end = int(active[local_end])
        segment_delta = segment[1] - segment[0]
        atom_delta = anchors[end] - anchors[begin]
        segment_length = float(np.linalg.norm(segment_delta))
        atom_distance = float(np.linalg.norm(atom_delta))
        if min(segment_length, atom_distance) <= 0.0:
            continue
        cosine = abs(
            float(np.dot(segment_delta, atom_delta) / (segment_length * atom_distance))
        )
        angle = math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))
        if angle > config.maximum_angle_degrees:
            continue
        pair = tuple(sorted((begin, end)))
        grouped.setdefault(pair, []).append(
            {
                "segment_index": segment_index,
                "segment": segment,
                "endpoint_error": endpoint_error,
                "angle": angle,
                "length": segment_length,
            }
        )

    candidates = []
    for pair, members in grouped.items():
        representative = min(
            members,
            key=lambda item: (
                item["endpoint_error"],
                item["angle"],
                -item["length"],
                item["segment_index"],
            ),
        )
        candidates.append(
            AtomPairLineCandidate(
                atom_pair=pair,
                representative_segment_pixels=tuple(
                    tuple(float(value) for value in point)
                    for point in representative["segment"]
                ),
                support_segment_indices=tuple(
                    sorted(int(item["segment_index"]) for item in members)
                ),
                endpoint_error_pixels=float(representative["endpoint_error"]),
                angle_error_degrees=float(representative["angle"]),
                maximum_support_length_pixels=max(
                    float(item["length"]) for item in members
                ),
            )
        )
    candidates.sort(
        key=lambda item: (
            item.endpoint_error_pixels,
            item.angle_error_degrees,
            -len(item.support_segment_indices),
            -item.maximum_support_length_pixels,
            item.atom_pair,
        )
    )
    return tuple(candidates[: config.maximum_candidates])


def detect_atom_pair_line_candidates(
    image: Image.Image,
    atom_anchors_pixels: Any,
    *,
    atom_mask: Any | None = None,
    config: LineSegmentRescueConfig = LineSegmentRescueConfig(),
) -> tuple[AtomPairLineCandidate, ...]:
    segments = detect_raster_line_segments(image, config=config)
    return attach_segments_to_atom_pairs(
        segments,
        atom_anchors_pixels,
        atom_mask=atom_mask,
        config=config,
    )

