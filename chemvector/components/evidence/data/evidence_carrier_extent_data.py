"""Evidence-carrier targets for raster line extent refinement.

Raster proposals come only from OpenCV LSD.  Visual primitive sidecars are
used after proposal generation to label those carriers.  Association is
deliberately many-to-one: both raster edges of one thick physical stroke may
be positive evidence for the same target line.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
import torch
from torch.utils.data import IterableDataset
import webdataset as wds

from chemvector.audit_evidence_carrier_proposals import (
    COORDINATE_MAX,
    TargetLine,
    detect_lsd_lines,
    extract_targets,
)
from chemvector.bond_group_set import GROUP_KIND_TO_ID
from chemvector.visual_svg_data import VisualSVGCollator


IN_SCOPE_GROUP_KINDS = frozenset(("single_line", "parallel_2", "parallel_3"))
IGNORE_LABEL = -1
NEGATIVE_LABEL = 0
POSITIVE_LABEL = 1
PADDING_TARGET_ID = -1


def _finite_real(value: Any, name: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return result


def _bin_count(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 2:
        raise ValueError(f"{name} must be an integer >= 2")
    return int(value)


@dataclass(frozen=True, slots=True)
class EvidenceCarrierExtentTargetConfig:
    """Geometry-only target settings, independent of model feature widths."""

    minimum_length_pixels: float = 1.0
    positive_angle_degrees: float = 5.0
    positive_perpendicular_pixels: float = 3.0
    positive_overlap_pixels: float = 8.0
    compatibility_angle_degrees: float = 12.0
    compatibility_perpendicular_pixels: float = 8.0
    compatibility_overlap_pixels: float = 1.0
    exclusion_angle_degrees: float = 12.0
    exclusion_perpendicular_pixels: float = 8.0
    exclusion_overlap_pixels: float = 1.0
    exclusion_region_margin_pixels: float = 4.0
    text_region_margin_pixels: float = 3.0
    ambiguity_cost_margin: float = 0.25
    ambiguity_perpendicular_margin_pixels: float = 0.5
    ambiguity_angle_margin_degrees: float = 1.0
    ambiguity_overlap_fraction_margin: float = 0.05
    normal_offset_limit_pixels: float = 4.0
    normal_offset_bins: int = 17
    extent_bins: int = 385
    long_bond_pixels: float = 96.0

    def __post_init__(self) -> None:
        numeric = (
            "minimum_length_pixels",
            "positive_angle_degrees",
            "positive_perpendicular_pixels",
            "positive_overlap_pixels",
            "compatibility_angle_degrees",
            "compatibility_perpendicular_pixels",
            "compatibility_overlap_pixels",
            "exclusion_angle_degrees",
            "exclusion_perpendicular_pixels",
            "exclusion_overlap_pixels",
            "exclusion_region_margin_pixels",
            "text_region_margin_pixels",
            "ambiguity_cost_margin",
            "ambiguity_perpendicular_margin_pixels",
            "ambiguity_angle_margin_degrees",
            "ambiguity_overlap_fraction_margin",
            "normal_offset_limit_pixels",
            "long_bond_pixels",
        )
        for name in numeric:
            object.__setattr__(self, name, _finite_real(getattr(self, name), name))
        object.__setattr__(
            self,
            "normal_offset_bins",
            _bin_count(self.normal_offset_bins, "normal_offset_bins"),
        )
        object.__setattr__(
            self,
            "extent_bins",
            _bin_count(self.extent_bins, "extent_bins"),
        )
        if self.normal_offset_limit_pixels <= 0.0 or self.long_bond_pixels <= 0.0:
            raise ValueError("Offset limit and long-bond threshold must be positive")
        if self.positive_angle_degrees > self.compatibility_angle_degrees:
            raise ValueError("Compatibility angle must include the positive angle")
        if (
            self.positive_perpendicular_pixels
            > self.compatibility_perpendicular_pixels
        ):
            raise ValueError(
                "Compatibility perpendicular distance must include positives"
            )
        if self.positive_overlap_pixels < self.compatibility_overlap_pixels:
            raise ValueError("Compatibility overlap must not exceed positive overlap")
        if self.normal_offset_limit_pixels < self.positive_perpendicular_pixels:
            raise ValueError("normal_offset_limit_pixels must cover positive matches")
        if self.ambiguity_overlap_fraction_margin > 1.0:
            raise ValueError("ambiguity_overlap_fraction_margin must be <= 1")


def _contains_internal_test(value: Any) -> bool:
    return "internal_test" in str(value).lower().replace("-", "_")


def reject_internal_test_shards(shards: str | Iterable[str]) -> None:
    """Reject sealed shard paths before WebDataset opens them."""

    values = (shards,) if isinstance(shards, str) else tuple(shards)
    forbidden = []
    for value in values:
        raw = str(value)
        resolved = raw
        if "://" not in raw:
            resolved = str(Path(raw).expanduser().resolve(strict=False))
        if _contains_internal_test(raw) or _contains_internal_test(resolved):
            forbidden.append(raw)
    if forbidden:
        raise ValueError(
            "Evidence Carrier Extent must not read sealed internal_test shards: "
            + ", ".join(map(str, forbidden))
        )


def reject_internal_test_sample(sample: dict[str, Any]) -> None:
    """Reject sealed samples even when a caller bypasses path validation."""

    key = str(sample.get("key", sample.get("__key__", "")))
    serialized_meta = json.dumps(
        sample.get("meta", {}), ensure_ascii=True, sort_keys=True
    )
    if _contains_internal_test(key) or _contains_internal_test(serialized_meta):
        raise ValueError(
            "Evidence Carrier Extent must not read sealed internal_test samples"
        )


def _canvas_size(layers: dict[str, Any]) -> tuple[int, int]:
    try:
        width = int(layers["width"])
        height = int(layers["height"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("layers must contain integer width and height") from error
    if width < 2 or height < 2:
        raise ValueError("pixel-center canvases require width and height >= 2")
    return width, height


def _canonicalize_carriers(carriers: Any, width: int, height: int) -> np.ndarray:
    values = np.asarray(carriers, dtype=np.float64)
    if values.size == 0:
        return np.empty((0, 2, 2), dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (2, 2):
        raise ValueError("carrier segments must have shape [N, 2, 2]")
    if not np.isfinite(values).all():
        raise ValueError("carrier segments must be finite")
    tolerance = 1e-4
    if (
        (values[..., 0] < -tolerance).any()
        or (values[..., 0] > width - 1 + tolerance).any()
        or (values[..., 1] < -tolerance).any()
        or (values[..., 1] > height - 1 + tolerance).any()
    ):
        raise ValueError("carrier segments must lie inside the image canvas")
    result = np.clip(
        values.copy(),
        (0.0, 0.0),
        (float(width - 1), float(height - 1)),
    )
    delta = result[:, 1] - result[:, 0]
    vertical = np.abs(delta[:, 0]) <= np.finfo(np.float64).eps * 8.0
    reverse = (delta[:, 0] < 0.0) | (vertical & (delta[:, 1] < 0.0))
    result[reverse] = result[reverse, ::-1]
    lengths = np.linalg.norm(result[:, 1] - result[:, 0], axis=-1)
    if (lengths <= 1e-8).any():
        raise ValueError("carrier segments must be nondegenerate")
    return result


def detect_evidence_carriers(
    image: Image.Image,
    *,
    oracle_bond_layer: bool = False,
    target_config: EvidenceCarrierExtentTargetConfig = EvidenceCarrierExtentTargetConfig(),
) -> np.ndarray:
    """Generate target-independent, canonical LSD carriers in pixel space."""

    if not isinstance(oracle_bond_layer, bool):
        raise ValueError("oracle_bond_layer must be a boolean")

    return _canonicalize_carriers(
        detect_lsd_lines(
            image,
            oracle_bond_layer=oracle_bond_layer,
            minimum_length_pixels=target_config.minimum_length_pixels,
        ),
        *image.size,
    )


def _targets_pixels(
    targets: list[TargetLine], width: int, height: int
) -> np.ndarray:
    if not targets:
        return np.empty((0, 2, 2), dtype=np.float64)
    scale = np.asarray(
        ((width - 1) / COORDINATE_MAX, (height - 1) / COORDINATE_MAX)
    )
    return np.asarray([target.points_q for target in targets], dtype=np.float64) * scale


def _line_metrics(
    carriers: np.ndarray, targets_pixels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not len(carriers) or not len(targets_pixels):
        shape = (len(carriers), len(targets_pixels))
        empty = np.empty(shape, dtype=np.float64)
        return empty, empty.copy(), empty.copy()
    target_delta = targets_pixels[:, 1] - targets_pixels[:, 0]
    target_length = np.linalg.norm(target_delta, axis=-1)
    if (target_length <= 1e-8).any():
        raise ValueError("target physical lines must be nondegenerate")
    target_unit = target_delta / target_length[:, None]
    target_normal = np.stack((-target_unit[:, 1], target_unit[:, 0]), axis=-1)

    carrier_delta = carriers[:, 1] - carriers[:, 0]
    carrier_unit = carrier_delta / np.linalg.norm(carrier_delta, axis=-1)[:, None]
    cosine = np.abs(carrier_unit @ target_unit.T)
    angle = np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0)))
    midpoint_delta = carriers.mean(axis=1)[:, None, :] - targets_pixels[None, :, 0]
    perpendicular = np.abs(np.einsum("ntd,td->nt", midpoint_delta, target_normal))
    projections = np.einsum(
        "nted,td->nte",
        carriers[:, None, :, :] - targets_pixels[None, :, None, 0, :],
        target_unit,
    )
    lower = projections.min(axis=-1)
    upper = projections.max(axis=-1)
    overlap = np.maximum(
        0.0,
        np.minimum(upper, target_length[None, :]) - np.maximum(lower, 0.0),
    )
    return angle, perpendicular, overlap


def _support_mask(
    angles: np.ndarray,
    perpendicular: np.ndarray,
    overlaps: np.ndarray,
    target_pixels: np.ndarray,
    *,
    maximum_angle_degrees: float,
    maximum_perpendicular_pixels: float,
    minimum_overlap_pixels: float,
) -> np.ndarray:
    if not target_pixels.size:
        return np.zeros(angles.shape, dtype=bool)
    target_lengths = np.linalg.norm(
        target_pixels[:, 1] - target_pixels[:, 0], axis=-1
    )
    required_overlap = np.minimum(
        minimum_overlap_pixels, target_lengths * 0.25
    )[None, :]
    return (
        (angles <= maximum_angle_degrees)
        & (perpendicular <= maximum_perpendicular_pixels)
        & (overlaps >= required_overlap)
    )


def _association_costs(
    carriers: np.ndarray,
    angles: np.ndarray,
    perpendicular: np.ndarray,
    overlaps: np.ndarray,
    target_config: EvidenceCarrierExtentTargetConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if not angles.size:
        return angles.copy(), angles.copy()
    carrier_lengths = np.linalg.norm(carriers[:, 1] - carriers[:, 0], axis=-1)
    overlap_fraction = np.clip(
        overlaps / np.maximum(carrier_lengths[:, None], 1e-8), 0.0, 1.0
    )
    costs = (
        perpendicular / max(target_config.positive_perpendicular_pixels, 1e-8)
        + angles / max(target_config.positive_angle_degrees, 1e-8)
        + (1.0 - overlap_fraction)
    )
    return costs, overlap_fraction


def _ambiguous_candidates(
    carrier_index: int,
    first: int,
    second: int,
    *,
    costs: np.ndarray,
    angles: np.ndarray,
    perpendicular: np.ndarray,
    overlap_fraction: np.ndarray,
    target_config: EvidenceCarrierExtentTargetConfig,
) -> bool:
    return (
        abs(float(costs[carrier_index, first] - costs[carrier_index, second]))
        < target_config.ambiguity_cost_margin
        and abs(
            float(
                perpendicular[carrier_index, first]
                - perpendicular[carrier_index, second]
            )
        )
        < target_config.ambiguity_perpendicular_margin_pixels
        and abs(float(angles[carrier_index, first] - angles[carrier_index, second]))
        < target_config.ambiguity_angle_margin_degrees
        and abs(
            float(
                overlap_fraction[carrier_index, first]
                - overlap_fraction[carrier_index, second]
            )
        )
        < target_config.ambiguity_overlap_fraction_margin
    )


def _fraction_bin(value: float, bins: int) -> int:
    clipped = min(max(float(value), 0.0), 1.0)
    return min(bins - 1, max(0, int(round(clipped * (bins - 1)))))


def _offset_bin(value: float, *, limit: float, bins: int) -> int:
    normalized = (min(max(float(value), -limit), limit) + limit) / (2.0 * limit)
    return _fraction_bin(normalized, bins)


def _carrier_canvas_interval(
    carrier: np.ndarray, width: int, height: int
) -> tuple[np.ndarray, np.ndarray, float]:
    origin = carrier[0]
    delta = carrier[1] - carrier[0]
    unit = delta / np.linalg.norm(delta)
    lower = -float("inf")
    upper = float("inf")
    for axis, maximum in ((0, float(width - 1)), (1, float(height - 1))):
        component = float(unit[axis])
        coordinate = float(origin[axis])
        if abs(component) <= 1e-12:
            if not 0.0 <= coordinate <= maximum:
                raise ValueError("carrier line does not intersect the image canvas")
            continue
        values = sorted(((-coordinate) / component, (maximum - coordinate) / component))
        lower = max(lower, values[0])
        upper = min(upper, values[1])
    if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
        raise ValueError("carrier line has no finite image-boundary interval")
    return origin + unit * lower, unit, upper - lower


def _geometry_target(
    carrier: np.ndarray,
    target_pixels: np.ndarray,
    *,
    width: int,
    height: int,
    target_config: EvidenceCarrierExtentTargetConfig,
) -> tuple[float, int, float, float, int, int]:
    carrier_delta = carrier[1] - carrier[0]
    unit = carrier_delta / np.linalg.norm(carrier_delta)
    normal = np.asarray((-unit[1], unit[0]), dtype=np.float64)
    normal_offset = float(np.dot(target_pixels.mean(axis=0) - carrier.mean(axis=0), normal))
    shifted_carrier = carrier + normal[None, :] * normal_offset
    canvas_begin, shifted_unit, canvas_length = _carrier_canvas_interval(
        shifted_carrier, width, height
    )
    fractions = (
        (target_pixels - canvas_begin[None, :]) @ shifted_unit / canvas_length
    )
    left = float(np.clip(fractions.min(), 0.0, 1.0))
    right = float(np.clip(fractions.max(), 0.0, 1.0))
    if not left < right:
        raise ValueError("matched target projects to a degenerate carrier extent")
    return (
        normal_offset,
        _offset_bin(
            normal_offset,
            limit=target_config.normal_offset_limit_pixels,
            bins=target_config.normal_offset_bins,
        ),
        left,
        right,
        _fraction_bin(left, target_config.extent_bins),
        _fraction_bin(right, target_config.extent_bins),
    )


def _primitive_points_pixels(
    primitive: dict[str, Any], width: int, height: int
) -> np.ndarray:
    points = primitive.get("quantized_points")
    if not isinstance(points, list) or not points:
        raise ValueError("physical primitive lacks quantized_points")
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError("physical primitive points must have shape [N, 2]")
    return values * np.asarray(
        ((width - 1) / COORDINATE_MAX, (height - 1) / COORDINATE_MAX)
    )


def _expanded_box(points: np.ndarray, margin: float) -> tuple[float, float, float, float]:
    return (
        float(points[:, 0].min() - margin),
        float(points[:, 1].min() - margin),
        float(points[:, 0].max() + margin),
        float(points[:, 1].max() + margin),
    )


def _exclusion_boxes(
    primitives_payload: dict[str, Any],
    layers: dict[str, Any],
    group_kinds: dict[int, str],
    *,
    width: int,
    height: int,
    target_config: EvidenceCarrierExtentTargetConfig,
) -> list[tuple[float, float, float, float]]:
    primitives = {
        int(item["primitive_id"]): item
        for item in primitives_payload.get("primitives", [])
    }
    boxes: list[tuple[float, float, float, float]] = []
    for group_index, group in enumerate(primitives_payload.get("bond_groups", [])):
        if group_kinds.get(group_index) in IN_SCOPE_GROUP_KINDS:
            continue
        points = [
            _primitive_points_pixels(primitives[int(primitive_id)], width, height)
            for primitive_id in group.get("primitive_ids", [])
        ]
        if points:
            boxes.append(
                _expanded_box(
                    np.concatenate(points),
                    target_config.exclusion_region_margin_pixels,
                )
            )

    for atom in layers.get("atoms", []):
        if not bool(atom.get("visible_text")):
            continue
        bbox = atom.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            values = np.asarray(bbox, dtype=np.float64)
            if np.isfinite(values).all():
                margin = target_config.text_region_margin_pixels
                boxes.append(
                    (
                        float(values[0] - margin),
                        float(values[1] - margin),
                        float(values[2] + margin),
                        float(values[3] + margin),
                    )
                )
    for text in primitives_payload.get("text_instances", []):
        for primitive_id in text.get("primitive_ids", []):
            primitive = primitives.get(int(primitive_id))
            if primitive is None or primitive.get("kind") != "text_box":
                continue
            boxes.append(
                _expanded_box(
                    _primitive_points_pixels(primitive, width, height),
                    target_config.text_region_margin_pixels,
                )
            )
    return boxes


def _segment_intersects_box(
    segment: np.ndarray, box: tuple[float, float, float, float]
) -> bool:
    lower = np.asarray(box[:2], dtype=np.float64)
    upper = np.asarray(box[2:], dtype=np.float64)
    direction = segment[1] - segment[0]
    t_min, t_max = 0.0, 1.0
    for axis in range(2):
        if abs(float(direction[axis])) <= 1e-12:
            if not lower[axis] <= segment[0, axis] <= upper[axis]:
                return False
            continue
        values = sorted(
            (
                (lower[axis] - segment[0, axis]) / direction[axis],
                (upper[axis] - segment[0, axis]) / direction[axis],
            )
        )
        t_min = max(t_min, float(values[0]))
        t_max = min(t_max, float(values[1]))
        if t_max < t_min:
            return False
    return True


def _polygon_edge_targets(
    primitives_payload: dict[str, Any]
) -> list[TargetLine]:
    primitives = {
        int(item["primitive_id"]): item
        for item in primitives_payload.get("primitives", [])
    }
    result = []
    for group_index, group in enumerate(primitives_payload.get("bond_groups", [])):
        for primitive_id in group.get("primitive_ids", []):
            primitive = primitives[int(primitive_id)]
            if primitive.get("kind") not in {"triangle", "quad"}:
                continue
            points = primitive.get("quantized_points")
            if not isinstance(points, list) or len(points) < 3:
                raise ValueError("polygon primitive lacks its boundary points")
            for begin, end in zip(points, points[1:] + points[:1]):
                result.append(
                    TargetLine(
                        group_index=group_index,
                        group_kind="out_of_scope_polygon",
                        points_q=(
                            (float(begin[0]), float(begin[1])),
                            (float(end[0]), float(end[1])),
                        ),
                        long_group=False,
                        text_adjacent=False,
                    )
                )
    return result


def associate_evidence_carriers(
    carrier_segments_pixels: Any,
    primitives_payload: dict[str, Any],
    layers: dict[str, Any],
    *,
    target_config: EvidenceCarrierExtentTargetConfig = EvidenceCarrierExtentTargetConfig(),
) -> dict[str, torch.Tensor]:
    """Associate LSD carriers without imposing one-proposal-per-target matching."""

    width, height = _canvas_size(layers)
    carriers = _canonicalize_carriers(carrier_segments_pixels, width, height)
    target_lines, target_groups, _ = extract_targets(
        primitives_payload,
        layers,
        long_bond_pixels=target_config.long_bond_pixels,
    )
    group_kinds = {group.group_index: group.kind for group in target_groups}
    ordinary = [
        (line_id, target)
        for line_id, target in enumerate(target_lines)
        if target.group_kind in IN_SCOPE_GROUP_KINDS
    ]
    ordinary_targets = [target for _, target in ordinary]
    ordinary_pixels = _targets_pixels(ordinary_targets, width, height)
    angles, perpendicular, overlaps = _line_metrics(carriers, ordinary_pixels)
    positive_support = _support_mask(
        angles,
        perpendicular,
        overlaps,
        ordinary_pixels,
        maximum_angle_degrees=target_config.positive_angle_degrees,
        maximum_perpendicular_pixels=target_config.positive_perpendicular_pixels,
        minimum_overlap_pixels=target_config.positive_overlap_pixels,
    )
    compatible_support = _support_mask(
        angles,
        perpendicular,
        overlaps,
        ordinary_pixels,
        maximum_angle_degrees=target_config.compatibility_angle_degrees,
        maximum_perpendicular_pixels=target_config.compatibility_perpendicular_pixels,
        minimum_overlap_pixels=target_config.compatibility_overlap_pixels,
    )
    association_costs, overlap_fraction = _association_costs(
        carriers, angles, perpendicular, overlaps, target_config
    )

    exclusion_targets = [
        target
        for target in target_lines
        if target.group_kind not in IN_SCOPE_GROUP_KINDS
    ] + _polygon_edge_targets(primitives_payload)
    exclusion_pixels = _targets_pixels(exclusion_targets, width, height)
    exclusion_angles, exclusion_perpendicular, exclusion_overlaps = _line_metrics(
        carriers, exclusion_pixels
    )
    exclusion_support = _support_mask(
        exclusion_angles,
        exclusion_perpendicular,
        exclusion_overlaps,
        exclusion_pixels,
        maximum_angle_degrees=target_config.exclusion_angle_degrees,
        maximum_perpendicular_pixels=target_config.exclusion_perpendicular_pixels,
        minimum_overlap_pixels=target_config.exclusion_overlap_pixels,
    )
    boxes = _exclusion_boxes(
        primitives_payload,
        layers,
        group_kinds,
        width=width,
        height=height,
        target_config=target_config,
    )
    exclusion_region = np.asarray(
        [any(_segment_intersects_box(carrier, box) for box in boxes) for carrier in carriers],
        dtype=bool,
    )

    count = len(carriers)
    keep_labels = torch.full((count,), IGNORE_LABEL, dtype=torch.long)
    keep_loss_mask = torch.zeros(count, dtype=torch.bool)
    geometry_loss_mask = torch.zeros(count, dtype=torch.bool)
    normal_offset_pixels = torch.zeros(count, dtype=torch.float32)
    normal_offset_bin = torch.zeros(count, dtype=torch.long)
    extent_left_fraction = torch.zeros(count, dtype=torch.float32)
    extent_right_fraction = torch.zeros(count, dtype=torch.float32)
    extent_left_bin = torch.zeros(count, dtype=torch.long)
    extent_right_bin = torch.zeros(count, dtype=torch.long)
    target_line_id = torch.full((count,), PADDING_TARGET_ID, dtype=torch.long)
    group_id = torch.full((count,), PADDING_TARGET_ID, dtype=torch.long)
    group_type = torch.zeros(count, dtype=torch.long)

    for carrier_index, carrier in enumerate(carriers):
        candidates = np.flatnonzero(positive_support[carrier_index])
        if len(candidates):
            ranked = sorted(
                map(int, candidates),
                key=lambda target_index: (
                    float(association_costs[carrier_index, target_index]),
                    ordinary[target_index][0],
                ),
            )
            if len(ranked) > 1 and _ambiguous_candidates(
                carrier_index,
                ranked[0],
                ranked[1],
                costs=association_costs,
                angles=angles,
                perpendicular=perpendicular,
                overlap_fraction=overlap_fraction,
                target_config=target_config,
            ):
                continue
            best = ranked[0]
            line_id, target = ordinary[best]
            geometry = _geometry_target(
                carrier,
                ordinary_pixels[best],
                width=width,
                height=height,
                target_config=target_config,
            )
            keep_labels[carrier_index] = POSITIVE_LABEL
            keep_loss_mask[carrier_index] = True
            geometry_loss_mask[carrier_index] = True
            normal_offset_pixels[carrier_index] = geometry[0]
            normal_offset_bin[carrier_index] = geometry[1]
            extent_left_fraction[carrier_index] = geometry[2]
            extent_right_fraction[carrier_index] = geometry[3]
            extent_left_bin[carrier_index] = geometry[4]
            extent_right_bin[carrier_index] = geometry[5]
            target_line_id[carrier_index] = line_id
            group_id[carrier_index] = target.group_index
            group_type[carrier_index] = GROUP_KIND_TO_ID[target.group_kind]
            continue

        near_ordinary = bool(compatible_support[carrier_index].any())
        near_excluded = bool(exclusion_support[carrier_index].any()) or bool(
            exclusion_region[carrier_index]
        )
        if near_ordinary or near_excluded:
            continue
        keep_labels[carrier_index] = NEGATIVE_LABEL
        keep_loss_mask[carrier_index] = True

    equivalence_weight = torch.zeros(count, dtype=torch.float32)
    positive_line_ids = target_line_id[keep_labels == POSITIVE_LABEL]
    if len(positive_line_ids):
        line_ids, multiplicities = torch.unique(
            positive_line_ids, return_counts=True
        )
        for line_id, multiplicity in zip(line_ids.tolist(), multiplicities.tolist()):
            equivalence_weight[
                (keep_labels == POSITIVE_LABEL) & (target_line_id == line_id)
            ] = 1.0 / float(multiplicity)
    equivalence_weight[keep_labels == NEGATIVE_LABEL] = 1.0

    normalized = carriers / np.asarray((width - 1, height - 1), dtype=np.float64)
    return {
        "carrier_segments": torch.tensor(normalized, dtype=torch.float32),
        "carrier_mask": torch.ones(count, dtype=torch.bool),
        "keep_labels": keep_labels,
        "keep_loss_mask": keep_loss_mask,
        "equivalence_weight": equivalence_weight,
        "normal_offset_pixels": normal_offset_pixels,
        "normal_offset_bin": normal_offset_bin,
        "extent_left_fraction": extent_left_fraction,
        "extent_right_fraction": extent_right_fraction,
        "extent_left_bin": extent_left_bin,
        "extent_right_bin": extent_right_bin,
        "geometry_loss_mask": geometry_loss_mask,
        "target_line_id": target_line_id,
        "group_id": group_id,
        "group_type": group_type,
    }


def build_evidence_carrier_extent_target(
    image: Image.Image,
    primitives_payload: dict[str, Any],
    layers: dict[str, Any],
    *,
    oracle_bond_layer: bool = False,
    target_config: EvidenceCarrierExtentTargetConfig = EvidenceCarrierExtentTargetConfig(),
    carrier_segments_pixels: Any | None = None,
) -> dict[str, torch.Tensor]:
    """Detect evidence carriers and build one unpadded sample target."""

    if not isinstance(oracle_bond_layer, bool):
        raise ValueError("oracle_bond_layer must be a boolean")

    width, height = _canvas_size(layers)
    if image.size != (width, height):
        raise ValueError(
            f"Raster size {image.size} differs from layers canvas {(width, height)}"
        )
    carriers = (
        detect_evidence_carriers(
            image,
            oracle_bond_layer=oracle_bond_layer,
            target_config=target_config,
        )
        if carrier_segments_pixels is None
        else carrier_segments_pixels
    )
    return associate_evidence_carriers(
        carriers,
        primitives_payload,
        layers,
        target_config=target_config,
    )


class EvidenceCarrierExtentWebDataset(IterableDataset):
    """Visual SVG reader that retains layers.json for carrier supervision."""

    def __init__(
        self,
        shards: str | list[str],
        *,
        shardshuffle: bool = False,
        sample_shuffle_buffer: int = 0,
        validate_hashes: bool = True,
        limit: int | None = None,
    ) -> None:
        super().__init__()
        reject_internal_test_shards(shards)
        self.shards = shards
        self.shardshuffle = shardshuffle
        self.sample_shuffle_buffer = sample_shuffle_buffer
        self.validate_hashes = validate_hashes
        self.limit = limit

    def __iter__(self):
        dataset = wds.WebDataset(
            self.shards,
            shardshuffle=100 if self.shardshuffle else False,
            empty_check=False,
        )
        if self.sample_shuffle_buffer:
            dataset = dataset.shuffle(self.sample_shuffle_buffer)
        for sample_index, raw in enumerate(dataset, 1):
            if self.limit is not None and sample_index > self.limit:
                break
            meta = json.loads(raw["meta.json"])
            sample = {"key": raw["__key__"], "meta": meta}
            reject_internal_test_sample(sample)
            payloads = {
                "png": raw["png"],
                "bond.png": raw["bond.png"],
                "primitives.json": raw["primitives.json"],
                "layers.json": raw["layers.json"],
            }
            if self.validate_hashes:
                expected = meta.get("sha256", {})
                for name, payload in payloads.items():
                    if name in expected and hashlib.sha256(payload).hexdigest() != expected[name]:
                        raise ValueError(f"Hash mismatch for {raw['__key__']}.{name}")
            image = Image.open(io.BytesIO(payloads["png"])).convert("RGB")
            image.load()
            bond_layer = Image.open(io.BytesIO(payloads["bond.png"])).convert("L")
            bond_layer.load()
            yield {
                **sample,
                "image": image,
                "bond_layer": bond_layer,
                "primitives": json.loads(payloads["primitives.json"]),
                "layers": json.loads(payloads["layers.json"]),
            }


class EvidenceCarrierExtentCollator:
    """Run LSD per sample and dynamically pad evidence-carrier targets."""

    _FLOAT_FIELDS = (
        "carrier_segments",
        "equivalence_weight",
        "normal_offset_pixels",
        "extent_left_fraction",
        "extent_right_fraction",
    )
    _LONG_FIELDS = (
        "keep_labels",
        "normal_offset_bin",
        "extent_left_bin",
        "extent_right_bin",
        "target_line_id",
        "group_id",
        "group_type",
    )
    _BOOL_FIELDS = ("carrier_mask", "keep_loss_mask", "geometry_loss_mask")

    def __init__(
        self,
        *,
        image_size: int = 768,
        oracle_bond_layer: bool = False,
        target_config: EvidenceCarrierExtentTargetConfig = EvidenceCarrierExtentTargetConfig(),
    ) -> None:
        self.image_collator = VisualSVGCollator(tokenizer=None, image_size=image_size)
        self.image_size = int(image_size)
        if not isinstance(oracle_bond_layer, bool):
            raise ValueError("oracle_bond_layer must be a boolean")
        self.oracle_bond_layer = oracle_bond_layer
        self.target_config = target_config

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        if not samples:
            raise ValueError("Evidence Carrier Extent cannot collate an empty batch")
        targets = []
        for sample in samples:
            reject_internal_test_sample(sample)
            if not isinstance(sample.get("primitives"), dict):
                raise ValueError("Sample lacks a primitives sidecar")
            if not isinstance(sample.get("layers"), dict):
                raise ValueError("Sample lacks a layers sidecar")
            expected_size = (self.image_size, self.image_size)
            if sample["image"].size != expected_size:
                raise ValueError(
                    "Evidence Carrier Extent requires native-size rasters; "
                    f"got {sample['image'].size}, expected {expected_size}. "
                    "Detect carriers after an explicit geometry-aware resize."
                )
            if sample["bond_layer"].size != expected_size:
                raise ValueError(
                    "Evidence Carrier Extent bond layer and model canvas differ"
                )
            raster = sample["bond_layer"] if self.oracle_bond_layer else sample["image"]
            targets.append(
                build_evidence_carrier_extent_target(
                    raster,
                    sample["primitives"],
                    sample["layers"],
                    oracle_bond_layer=self.oracle_bond_layer,
                    target_config=self.target_config,
                )
            )

        counts = [int(target["carrier_mask"].sum()) for target in targets]
        capacity = max(max(counts), 1)
        batch_size = len(samples)
        padded: dict[str, torch.Tensor] = {
            "carrier_segments": torch.zeros(
                (batch_size, capacity, 2, 2), dtype=torch.float32
            ),
            "equivalence_weight": torch.zeros(
                (batch_size, capacity), dtype=torch.float32
            ),
            "normal_offset_pixels": torch.zeros(
                (batch_size, capacity), dtype=torch.float32
            ),
            "extent_left_fraction": torch.zeros(
                (batch_size, capacity), dtype=torch.float32
            ),
            "extent_right_fraction": torch.zeros(
                (batch_size, capacity), dtype=torch.float32
            ),
            "keep_labels": torch.full(
                (batch_size, capacity), IGNORE_LABEL, dtype=torch.long
            ),
            "normal_offset_bin": torch.zeros(
                (batch_size, capacity), dtype=torch.long
            ),
            "extent_left_bin": torch.zeros((batch_size, capacity), dtype=torch.long),
            "extent_right_bin": torch.zeros((batch_size, capacity), dtype=torch.long),
            "target_line_id": torch.full(
                (batch_size, capacity), PADDING_TARGET_ID, dtype=torch.long
            ),
            "group_id": torch.full(
                (batch_size, capacity), PADDING_TARGET_ID, dtype=torch.long
            ),
            "group_type": torch.zeros((batch_size, capacity), dtype=torch.long),
            "carrier_mask": torch.zeros((batch_size, capacity), dtype=torch.bool),
            "keep_loss_mask": torch.zeros((batch_size, capacity), dtype=torch.bool),
            "geometry_loss_mask": torch.zeros(
                (batch_size, capacity), dtype=torch.bool
            ),
        }
        for batch_index, (target, count) in enumerate(zip(targets, counts)):
            for name in (*self._FLOAT_FIELDS, *self._LONG_FIELDS, *self._BOOL_FIELDS):
                padded[name][batch_index, :count] = target[name]

        return {
            "pixel_values": torch.stack(
                [self.image_collator._image_tensor(sample["image"]) for sample in samples]
            ),
            **padded,
            "carrier_count": torch.tensor(counts, dtype=torch.long),
            "keys": [sample["key"] for sample in samples],
            "meta": [sample["meta"] for sample in samples],
            "primitives": [sample["primitives"] for sample in samples],
            "layers": [sample["layers"] for sample in samples],
            "bond_layers": [sample["bond_layer"] for sample in samples],
        }


__all__ = [
    "IGNORE_LABEL",
    "IN_SCOPE_GROUP_KINDS",
    "NEGATIVE_LABEL",
    "POSITIVE_LABEL",
    "EvidenceCarrierExtentCollator",
    "EvidenceCarrierExtentTargetConfig",
    "EvidenceCarrierExtentWebDataset",
    "associate_evidence_carriers",
    "build_evidence_carrier_extent_target",
    "detect_evidence_carriers",
    "reject_internal_test_sample",
    "reject_internal_test_shards",
]
