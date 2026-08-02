"""Deterministic local raster evidence for one Evidence Carrier pair.

The first channel contains inverted grayscale ink from the complete rendered
image.  The second channel marks the two candidate LSD segments so a scorer can
separate the candidate from nearby text, crossings, and adjacent bonds.  The
coordinate frame is invariant to candidate order and segment endpoint order.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real
from typing import Any

import numpy as np
from PIL import Image
import torch


PAIR_LOCAL_RASTER_SCHEMA_VERSION = "evidence-carrier-pair-local-raster-0.1"


def _finite_positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def _integer_at_least(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return result


@dataclass(frozen=True, slots=True)
class PairLocalRasterConfig:
    width: int = 48
    height: int = 24
    longitudinal_padding_pixels: float = 6.0
    normal_padding_pixels: float = 8.0
    minimum_longitudinal_half_extent_pixels: float = 12.0
    minimum_normal_half_extent_pixels: float = 8.0
    candidate_mask_radius_pixels: float = 2.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "width", _integer_at_least(self.width, "width", 2))
        object.__setattr__(
            self, "height", _integer_at_least(self.height, "height", 2)
        )
        for name in (
            "longitudinal_padding_pixels",
            "normal_padding_pixels",
            "minimum_longitudinal_half_extent_pixels",
            "minimum_normal_half_extent_pixels",
            "candidate_mask_radius_pixels",
        ):
            object.__setattr__(self, name, _finite_positive(getattr(self, name), name))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PAIR_LOCAL_RASTER_SCHEMA_VERSION,
            **asdict(self),
            "channels": ["inverted_grayscale_ink", "candidate_segment_mask"],
            "padding": "white_image_zero_ink",
            "sampling": "manual_bilinear_align_corners",
        }


def _darkness_image(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        grayscale = np.asarray(image.convert("L"), dtype=np.float64)
    else:
        value = np.asarray(image)
        if value.ndim == 3 and value.shape[-1] in {3, 4}:
            rgb = value[..., :3].astype(np.float64, copy=False)
            grayscale = (
                0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
            )
        elif value.ndim == 2:
            grayscale = value.astype(np.float64, copy=False)
        else:
            raise ValueError("pair local raster image must be HxW, HxWx3, or HxWx4")
    if grayscale.size == 0 or grayscale.ndim != 2:
        raise ValueError("pair local raster image must be nonempty")
    if not np.isfinite(grayscale).all():
        raise ValueError("pair local raster image contains non-finite values")
    maximum = float(grayscale.max(initial=0.0))
    if maximum <= 1.0:
        normalized = np.clip(grayscale, 0.0, 1.0)
    else:
        normalized = np.clip(grayscale / 255.0, 0.0, 1.0)
    return 1.0 - normalized


def _segments(value: Any) -> np.ndarray:
    segments = np.asarray(value, dtype=np.float64)
    if segments.size == 0:
        return np.empty((0, 2, 2), dtype=np.float64)
    if segments.ndim != 3 or segments.shape[1:] != (2, 2):
        raise ValueError("anchor_segments_pixels must have shape [anchors,2,2]")
    if not np.isfinite(segments).all():
        raise ValueError("anchor segments contain non-finite values")
    lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
    if (lengths <= 1e-8).any():
        raise ValueError("anchor segments must be nondegenerate")
    return segments


def _pairs(value: Any, anchor_count: int) -> np.ndarray:
    pairs = np.asarray(value, dtype=np.int64)
    if pairs.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pair_indices must have shape [pairs,2]")
    if (pairs < 0).any() or (pairs >= anchor_count).any():
        raise ValueError("pair_indices exceed anchor_segments_pixels")
    if (pairs[:, 0] == pairs[:, 1]).any():
        raise ValueError("pair candidates must reference two distinct anchors")
    return pairs


def _centerlines(value: Any | None, pair_count: int) -> np.ndarray | None:
    if value is None:
        return None
    centerlines = np.asarray(value, dtype=np.float64)
    if centerlines.size == 0 and pair_count == 0:
        return np.empty((0, 2, 2), dtype=np.float64)
    if centerlines.shape != (pair_count, 2, 2):
        raise ValueError("pair centerlines must have shape [pairs,2,2]")
    if not np.isfinite(centerlines).all():
        raise ValueError("pair centerlines contain non-finite values")
    lengths = np.linalg.norm(centerlines[:, 1] - centerlines[:, 0], axis=1)
    if pair_count and (lengths <= 1e-8).any():
        raise ValueError("pair centerlines must be nondegenerate")
    return centerlines


def _canonical_axis(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first_axis = first[1] - first[0]
    second_axis = second[1] - second[0]
    first_axis /= np.linalg.norm(first_axis)
    second_axis /= np.linalg.norm(second_axis)
    if float(np.dot(first_axis, second_axis)) < 0.0:
        second_axis = -second_axis
    axis = first_axis + second_axis
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-8:
        axis = first_axis
    else:
        axis /= norm
    if float(axis[0]) < 0.0 or (
        abs(float(axis[0])) <= 1e-12 and float(axis[1]) < 0.0
    ):
        axis = -axis
    return axis


def _overlap_interval(
    first: np.ndarray, second: np.ndarray, axis: np.ndarray
) -> tuple[float, float]:
    first_interval = np.sort(first @ axis)
    second_interval = np.sort(second @ axis)
    lower = max(float(first_interval[0]), float(second_interval[0]))
    upper = min(float(first_interval[1]), float(second_interval[1]))
    if upper - lower <= 1e-8:
        all_values = np.concatenate((first_interval, second_interval))
        lower, upper = float(all_values.min()), float(all_values.max())
    return lower, upper


def _pair_frame(
    first: np.ndarray,
    second: np.ndarray,
    centerline: np.ndarray | None,
    config: PairLocalRasterConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    axis = _canonical_axis(first, second)
    normal = np.asarray((-axis[1], axis[0]), dtype=np.float64)
    if centerline is None:
        lower, upper = _overlap_interval(first, second, axis)
    else:
        projected = np.sort(centerline @ axis)
        lower, upper = float(projected[0]), float(projected[1])
    normal_centers = np.asarray(
        (float(first.mean(axis=0) @ normal), float(second.mean(axis=0) @ normal))
    )
    origin = (
        axis * ((lower + upper) * 0.5)
        + normal * float(normal_centers.mean())
    )
    longitudinal_half_extent = max(
        (upper - lower) * 0.5 + config.longitudinal_padding_pixels,
        config.minimum_longitudinal_half_extent_pixels,
    )
    normal_half_extent = max(
        abs(float(normal_centers[1] - normal_centers[0])) * 0.5
        + config.normal_padding_pixels,
        config.minimum_normal_half_extent_pixels,
    )
    return origin, axis, normal, longitudinal_half_extent, normal_half_extent


def _sampling_points(
    origin: np.ndarray,
    axis: np.ndarray,
    normal: np.ndarray,
    longitudinal_half_extent: float,
    normal_half_extent: float,
    config: PairLocalRasterConfig,
) -> np.ndarray:
    longitudinal = np.linspace(
        -longitudinal_half_extent,
        longitudinal_half_extent,
        config.width,
        dtype=np.float64,
    )
    transverse = np.linspace(
        -normal_half_extent,
        normal_half_extent,
        config.height,
        dtype=np.float64,
    )
    return (
        origin[None, None, :]
        + longitudinal[None, :, None] * axis[None, None, :]
        + transverse[:, None, None] * normal[None, None, :]
    )


def _bilinear_sample_zero(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    height, width = image.shape
    x = points[..., 0]
    y = points[..., 1]
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    wx = x - x0
    wy = y - y0

    def gather(ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
        valid = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
        result = np.zeros(ix.shape, dtype=np.float64)
        if valid.any():
            result[valid] = image[iy[valid], ix[valid]]
        return result

    return (
        gather(x0, y0) * (1.0 - wx) * (1.0 - wy)
        + gather(x1, y0) * wx * (1.0 - wy)
        + gather(x0, y1) * (1.0 - wx) * wy
        + gather(x1, y1) * wx * wy
    )


def _point_segment_distance(points: np.ndarray, segment: np.ndarray) -> np.ndarray:
    start, end = segment
    delta = end - start
    denominator = float(np.dot(delta, delta))
    projection = np.clip(((points - start) @ delta) / denominator, 0.0, 1.0)
    closest = start + projection[..., None] * delta
    return np.linalg.norm(points - closest, axis=-1)


def extract_pair_local_rasters(
    image: Image.Image | np.ndarray,
    anchor_segments_pixels: Any,
    pair_indices: Any,
    *,
    pair_centerlines_pixels: Any | None = None,
    config: PairLocalRasterConfig = PairLocalRasterConfig(),
) -> torch.Tensor:
    """Return float32 pair crops with shape ``[pairs,2,height,width]``."""

    darkness = _darkness_image(image)
    segments = _segments(anchor_segments_pixels)
    pairs = _pairs(pair_indices, len(segments))
    centerlines = _centerlines(pair_centerlines_pixels, len(pairs))
    if not len(pairs):
        return torch.empty((0, 2, config.height, config.width), dtype=torch.float32)

    crops = np.empty((len(pairs), 2, config.height, config.width), dtype=np.float32)
    for index, (first_index, second_index) in enumerate(pairs.tolist()):
        first = segments[first_index]
        second = segments[second_index]
        centerline = None if centerlines is None else centerlines[index]
        frame = _pair_frame(first, second, centerline, config)
        points = _sampling_points(*frame, config)
        ink = _bilinear_sample_zero(darkness, points)
        distance = np.minimum(
            _point_segment_distance(points, first),
            _point_segment_distance(points, second),
        )
        mask = np.clip(
            1.0 - distance / config.candidate_mask_radius_pixels,
            0.0,
            1.0,
        )
        crops[index, 0] = ink.astype(np.float32, copy=False)
        crops[index, 1] = mask.astype(np.float32, copy=False)
    return torch.from_numpy(crops)


def quantize_pair_local_rasters(crops: torch.Tensor) -> torch.Tensor:
    """Encode normalized pair crops as deterministic uint8 cache payloads."""

    if crops.ndim != 4 or crops.size(1) != 2:
        raise ValueError("pair local crops must have shape [pairs,2,height,width]")
    if not crops.is_floating_point() or not bool(torch.isfinite(crops).all()):
        raise ValueError("pair local crops must be finite floating point")
    if crops.numel() and (
        float(crops.min()) < -1e-6 or float(crops.max()) > 1.0 + 1e-6
    ):
        raise ValueError("pair local crops must lie in [0,1]")
    return crops.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)


def dequantize_pair_local_rasters(crops: torch.Tensor) -> torch.Tensor:
    if crops.ndim != 4 or crops.size(1) != 2 or crops.dtype != torch.uint8:
        raise ValueError("quantized pair local crops must be uint8 [pairs,2,H,W]")
    return crops.to(torch.float32).div(255.0)


__all__ = [
    "PAIR_LOCAL_RASTER_SCHEMA_VERSION",
    "PairLocalRasterConfig",
    "dequantize_pair_local_rasters",
    "extract_pair_local_rasters",
    "quantize_pair_local_rasters",
]
