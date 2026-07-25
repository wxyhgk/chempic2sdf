"""Target-free pairing of two LSD boundaries into one stroke carrier.

The pairing boundary accepts only raster pixels and raw LSD segments.  It does
not accept target lines, molecule metadata, or target identifiers.  Candidate
pairs must be near-collinear, overlap longitudinally, and enclose dark raster
ink.  Selected pairs are replaced by their geometric centerline; unmatched
LSD segments remain available to downstream keep/extent prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Any, Literal

import numpy as np
from PIL import Image


PairingStrategy = Literal["mutual_then_greedy", "mutual_only", "greedy"]


def _finite_real(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return result


def _positive_integer(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return result


@dataclass(frozen=True, slots=True)
class StrokeBoundaryPairingConfig:
    """Frozen validation calibration for 768px synthetic rasters."""

    maximum_angle_degrees: float = 3.0
    minimum_normal_separation_pixels: float = 0.5
    maximum_normal_separation_pixels: float = 4.0
    minimum_overlap_pixels: float = 8.0
    minimum_shorter_interval_overlap: float = 0.70
    minimum_interior_ink_occupancy: float = 0.80
    ink_darkness_threshold: float = 0.25
    longitudinal_samples: int = 17
    normal_samples: int = 5
    longitudinal_trim_fraction: float = 0.15
    reference_stroke_width_pixels: float = 3.0
    strategy: PairingStrategy = "mutual_then_greedy"

    def __post_init__(self) -> None:
        numeric = (
            "maximum_angle_degrees",
            "minimum_normal_separation_pixels",
            "maximum_normal_separation_pixels",
            "minimum_overlap_pixels",
            "minimum_shorter_interval_overlap",
            "minimum_interior_ink_occupancy",
            "ink_darkness_threshold",
            "longitudinal_trim_fraction",
            "reference_stroke_width_pixels",
        )
        for name in numeric:
            maximum = 1.0 if name in {
                "minimum_shorter_interval_overlap",
                "minimum_interior_ink_occupancy",
                "ink_darkness_threshold",
                "longitudinal_trim_fraction",
            } else None
            object.__setattr__(
                self,
                name,
                _finite_real(getattr(self, name), name, minimum=0.0, maximum=maximum),
            )
        object.__setattr__(
            self,
            "longitudinal_samples",
            _positive_integer(self.longitudinal_samples, "longitudinal_samples", minimum=2),
        )
        object.__setattr__(
            self,
            "normal_samples",
            _positive_integer(self.normal_samples, "normal_samples", minimum=2),
        )
        if self.maximum_angle_degrees <= 0.0 or self.maximum_angle_degrees > 90.0:
            raise ValueError("maximum_angle_degrees must lie in (0, 90]")
        if self.maximum_normal_separation_pixels <= 0.0:
            raise ValueError("maximum_normal_separation_pixels must be positive")
        if (
            self.minimum_normal_separation_pixels
            >= self.maximum_normal_separation_pixels
        ):
            raise ValueError("normal separation bounds must be strictly ordered")
        if self.minimum_overlap_pixels <= 0.0:
            raise ValueError("minimum_overlap_pixels must be positive")
        if not 0.0 < self.minimum_shorter_interval_overlap <= 1.0:
            raise ValueError("minimum_shorter_interval_overlap must lie in (0, 1]")
        if not 0.0 < self.minimum_interior_ink_occupancy <= 1.0:
            raise ValueError("minimum_interior_ink_occupancy must lie in (0, 1]")
        if not 0.0 < self.ink_darkness_threshold <= 1.0:
            raise ValueError("ink_darkness_threshold must lie in (0, 1]")
        if not 0.0 <= self.longitudinal_trim_fraction < 0.5:
            raise ValueError("longitudinal_trim_fraction must lie in [0, 0.5)")
        if self.reference_stroke_width_pixels <= 0.0:
            raise ValueError("reference_stroke_width_pixels must be positive")
        if self.strategy not in {"mutual_then_greedy", "mutual_only", "greedy"}:
            raise ValueError(f"Unsupported stroke pairing strategy: {self.strategy}")


@dataclass(frozen=True, slots=True)
class StrokeBoundaryPairGeometry:
    angle_degrees: float
    normal_separation_pixels: float
    overlap_pixels: float
    shorter_interval_overlap: float
    interval_iou: float
    interior_ink_occupancy: float
    interior_mean_darkness: float
    center_mean_darkness: float
    compatible: bool


@dataclass(frozen=True, slots=True)
class StrokeBoundaryPair:
    first_index: int
    second_index: int
    geometry: StrokeBoundaryPairGeometry


@dataclass(frozen=True, slots=True)
class StrokeCarrier:
    segment_pixels: tuple[tuple[float, float], tuple[float, float]]
    source_indices: tuple[int, ...]
    paired: bool


@dataclass(frozen=True, slots=True)
class StrokePairingResult:
    carriers: tuple[StrokeCarrier, ...]
    selected_pairs: tuple[StrokeBoundaryPair, ...]
    compatible_pair_count: int

    def segments_pixels(self) -> np.ndarray:
        if not self.carriers:
            return np.empty((0, 2, 2), dtype=np.float64)
        return np.asarray(
            [carrier.segment_pixels for carrier in self.carriers], dtype=np.float64
        )


@dataclass(frozen=True, slots=True)
class _LineRecord:
    index: int
    segment: np.ndarray
    axis: np.ndarray
    center: np.ndarray


def _canonical_segment(segment: Any) -> np.ndarray:
    value = np.asarray(segment, dtype=np.float64)
    if value.shape != (2, 2):
        raise ValueError("LSD segments must have shape [2,2]")
    if not np.isfinite(value).all():
        raise ValueError("LSD segments must contain finite coordinates")
    delta = value[1] - value[0]
    if float(np.linalg.norm(delta)) <= 1e-8:
        raise ValueError("LSD segments must be nondegenerate")
    if float(delta[0]) < 0.0 or (
        float(delta[0]) == 0.0 and float(delta[1]) < 0.0
    ):
        value = value[::-1].copy()
    else:
        value = value.copy()
    return value


def _records(segments_pixels: Any) -> list[_LineRecord]:
    segments = np.asarray(segments_pixels, dtype=np.float64)
    if segments.ndim != 3 or segments.shape[1:] != (2, 2):
        raise ValueError("segments_pixels must have shape [lines,2,2]")
    records = []
    for index, raw in enumerate(segments):
        segment = _canonical_segment(raw)
        delta = segment[1] - segment[0]
        axis = delta / np.linalg.norm(delta)
        records.append(
            _LineRecord(
                index=index,
                segment=segment,
                axis=axis,
                center=segment.mean(axis=0),
            )
        )
    return records


def _darkness_image(raster: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(raster, Image.Image):
        grayscale = np.asarray(raster.convert("L"), dtype=np.float64)
    else:
        value = np.asarray(raster)
        if value.ndim == 3 and value.shape[-1] in {3, 4}:
            if np.issubdtype(value.dtype, np.floating) and float(value.max()) <= 1.0:
                value = np.rint(np.clip(value, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                value = np.clip(value, 0, 255).astype(np.uint8)
            value = np.asarray(Image.fromarray(value).convert("L"))
        if value.ndim != 2:
            raise ValueError("Raster must be grayscale, RGB, or RGBA")
        grayscale = value.astype(np.float64)
    if grayscale.size == 0:
        raise ValueError("Raster cannot be empty")
    if not np.isfinite(grayscale).all():
        raise ValueError("Raster must contain finite pixels")
    if float(grayscale.max()) <= 1.0:
        normalized = np.clip(grayscale, 0.0, 1.0)
    else:
        normalized = np.clip(grayscale / 255.0, 0.0, 1.0)
    return 1.0 - normalized


def _bilinear_sample(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    height, width = image.shape
    x = np.clip(points[:, 0], 0.0, float(width - 1))
    y = np.clip(points[:, 1], 0.0, float(height - 1))
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = x - x0
    wy = y - y0
    return (
        (1.0 - wx) * (1.0 - wy) * image[y0, x0]
        + wx * (1.0 - wy) * image[y0, x1]
        + (1.0 - wx) * wy * image[y1, x0]
        + wx * wy * image[y1, x1]
    )


def _common_geometry(
    first: _LineRecord, second: _LineRecord
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float, float]:
    dot = float(np.clip(np.dot(first.axis, second.axis), -1.0, 1.0))
    aligned_second = second.axis if dot >= 0.0 else -second.axis
    common_axis = first.axis + aligned_second
    common_norm = float(np.linalg.norm(common_axis))
    if common_norm <= 1e-8:
        common_axis = first.axis.copy()
    else:
        common_axis /= common_norm
    normal = np.asarray((-common_axis[1], common_axis[0]), dtype=np.float64)
    first_interval = np.sort(first.segment @ common_axis)
    second_interval = np.sort(second.segment @ common_axis)
    overlap_lower = max(float(first_interval[0]), float(second_interval[0]))
    overlap_upper = min(float(first_interval[1]), float(second_interval[1]))
    overlap = max(0.0, overlap_upper - overlap_lower)
    first_normal = float(first.center @ normal)
    second_normal = float(second.center @ normal)
    return (
        common_axis,
        normal,
        first_interval,
        second_interval,
        overlap_lower,
        overlap_upper,
        first_normal,
        second_normal,
    )


def stroke_boundary_pair_geometry(
    first_segment_pixels: Any,
    second_segment_pixels: Any,
    raster: Image.Image | np.ndarray,
    *,
    config: StrokeBoundaryPairingConfig = StrokeBoundaryPairingConfig(),
) -> StrokeBoundaryPairGeometry:
    """Measure target-free geometry and raster ink between two LSD segments."""

    first, second = _records(np.stack((first_segment_pixels, second_segment_pixels)))
    darkness = _darkness_image(raster)
    return _measure_pair(first, second, darkness=darkness, config=config)


def _measure_pair(
    first: _LineRecord,
    second: _LineRecord,
    *,
    darkness: np.ndarray,
    config: StrokeBoundaryPairingConfig,
) -> StrokeBoundaryPairGeometry:
    dot = float(np.clip(np.dot(first.axis, second.axis), -1.0, 1.0))
    angle = math.degrees(math.acos(abs(dot)))
    (
        axis,
        normal,
        first_interval,
        second_interval,
        overlap_lower,
        overlap_upper,
        first_normal,
        second_normal,
    ) = _common_geometry(first, second)
    overlap = max(0.0, overlap_upper - overlap_lower)
    first_length = float(first_interval[1] - first_interval[0])
    second_length = float(second_interval[1] - second_interval[0])
    shorter_overlap = overlap / max(min(first_length, second_length), 1e-8)
    union = max(float(first_interval[1]), float(second_interval[1])) - min(
        float(first_interval[0]), float(second_interval[0])
    )
    interval_iou = overlap / union if union > 0.0 else 0.0
    separation = abs(second_normal - first_normal)

    epsilon = 1e-12
    geometry_compatible = (
        angle <= config.maximum_angle_degrees + epsilon
        and separation + epsilon >= config.minimum_normal_separation_pixels
        and separation <= config.maximum_normal_separation_pixels + epsilon
        and overlap + epsilon >= config.minimum_overlap_pixels
        and shorter_overlap + epsilon >= config.minimum_shorter_interval_overlap
    )
    if geometry_compatible:
        trim = config.longitudinal_trim_fraction * overlap
        positions = np.linspace(
            overlap_lower + trim,
            overlap_upper - trim,
            config.longitudinal_samples,
        )
        lower_normal = min(first_normal, second_normal)
        upper_normal = max(first_normal, second_normal)
        fractions = np.linspace(0.15, 0.85, config.normal_samples)
        strip_points = np.asarray(
            [
                axis * position
                + normal * (lower_normal + fraction * (upper_normal - lower_normal))
                for position in positions
                for fraction in fractions
            ],
            dtype=np.float64,
        )
        strip_darkness = _bilinear_sample(darkness, strip_points)
        occupancy = float(
            np.mean(strip_darkness >= config.ink_darkness_threshold)
        )
        mean_darkness = float(np.mean(strip_darkness))
        center_points = np.asarray(
            [
                axis * position
                + normal * (0.5 * (first_normal + second_normal))
                for position in positions
            ],
            dtype=np.float64,
        )
        center_darkness = float(np.mean(_bilinear_sample(darkness, center_points)))
    else:
        occupancy = mean_darkness = center_darkness = 0.0

    compatible = (
        geometry_compatible
        and occupancy + epsilon >= config.minimum_interior_ink_occupancy
    )
    return StrokeBoundaryPairGeometry(
        angle_degrees=angle,
        normal_separation_pixels=separation,
        overlap_pixels=overlap,
        shorter_interval_overlap=shorter_overlap,
        interval_iou=interval_iou,
        interior_ink_occupancy=occupancy,
        interior_mean_darkness=mean_darkness,
        center_mean_darkness=center_darkness,
        compatible=compatible,
    )


def _pair_rank(
    pair: StrokeBoundaryPair,
    config: StrokeBoundaryPairingConfig,
    records: list[_LineRecord],
) -> tuple[float | int, ...]:
    geometry = pair.geometry
    source_keys = sorted(
        tuple(float(value) for value in records[index].segment.reshape(-1))
        for index in (pair.first_index, pair.second_index)
    )
    return (
        -geometry.interval_iou,
        -geometry.interior_ink_occupancy,
        -geometry.interior_mean_darkness,
        geometry.angle_degrees,
        abs(
            geometry.normal_separation_pixels
            - config.reference_stroke_width_pixels
        ),
        -geometry.overlap_pixels,
        *source_keys[0],
        *source_keys[1],
    )


def _centerline(
    first: _LineRecord,
    second: _LineRecord,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    (
        axis,
        normal,
        _,
        _,
        overlap_lower,
        overlap_upper,
        first_normal,
        second_normal,
    ) = _common_geometry(first, second)
    normal_coordinate = 0.5 * (first_normal + second_normal)
    result = np.stack(
        (
            axis * overlap_lower + normal * normal_coordinate,
            axis * overlap_upper + normal * normal_coordinate,
        )
    )
    result = np.clip(
        result,
        (0.0, 0.0),
        (float(width - 1), float(height - 1)),
    )
    return _canonical_segment(result)


def pair_stroke_boundaries(
    segments_pixels: Any,
    raster: Image.Image | np.ndarray,
    *,
    config: StrokeBoundaryPairingConfig = StrokeBoundaryPairingConfig(),
) -> StrokePairingResult:
    """Pair compatible LSD boundaries without reading any target information."""

    records = _records(segments_pixels)
    darkness = _darkness_image(raster)
    height, width = darkness.shape
    compatible = []
    for first_index in range(len(records)):
        for second_index in range(first_index + 1, len(records)):
            geometry = _measure_pair(
                records[first_index],
                records[second_index],
                darkness=darkness,
                config=config,
            )
            if geometry.compatible:
                compatible.append(
                    StrokeBoundaryPair(first_index, second_index, geometry)
                )

    ranked = sorted(
        compatible,
        key=lambda pair: _pair_rank(pair, config, records),
    )
    selected: list[StrokeBoundaryPair] = []
    used: set[int] = set()

    if config.strategy in {"mutual_then_greedy", "mutual_only"}:
        best: dict[int, StrokeBoundaryPair] = {}
        for pair in ranked:
            for index in (pair.first_index, pair.second_index):
                if index not in best:
                    best[index] = pair
        mutual = [
            pair
            for pair in ranked
            if best.get(pair.first_index) is pair
            and best.get(pair.second_index) is pair
        ]
        for pair in mutual:
            if pair.first_index in used or pair.second_index in used:
                continue
            selected.append(pair)
            used.update((pair.first_index, pair.second_index))

    if config.strategy in {"mutual_then_greedy", "greedy"}:
        for pair in ranked:
            if pair.first_index in used or pair.second_index in used:
                continue
            selected.append(pair)
            used.update((pair.first_index, pair.second_index))

    by_first = {
        min(pair.first_index, pair.second_index): pair for pair in selected
    }
    carriers = []
    for index, record in enumerate(records):
        if index in used:
            pair = by_first.get(index)
            if pair is None:
                continue
            centerline = _centerline(
                records[pair.first_index],
                records[pair.second_index],
                width=width,
                height=height,
            )
            carriers.append(
                StrokeCarrier(
                    segment_pixels=tuple(map(tuple, centerline)),
                    source_indices=(pair.first_index, pair.second_index),
                    paired=True,
                )
            )
        else:
            carriers.append(
                StrokeCarrier(
                    segment_pixels=tuple(map(tuple, record.segment)),
                    source_indices=(index,),
                    paired=False,
                )
            )
    carriers.sort(
        key=lambda carrier: tuple(
            float(value) for point in carrier.segment_pixels for value in point
        )
    )
    return StrokePairingResult(
        carriers=tuple(carriers),
        selected_pairs=tuple(
            sorted(
                selected,
                key=lambda pair: _pair_rank(pair, config, records),
            )
        ),
        compatible_pair_count=len(compatible),
    )
