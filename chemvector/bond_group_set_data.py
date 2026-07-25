"""Padded Visual SVG v0.2 targets for bond-group set prediction."""

from __future__ import annotations

import json
from typing import Any

import torch

from Data.sdf.visual_svg_v0_2 import (
    ParsedVisualSVGSequenceV02,
    parse_visual_svg_v0_2,
    serialize_parsed_visual_svg_v0_2,
    visual_primitives_v0_1_to_visual_svg_v0_2,
    visual_svg_v0_2_to_dict,
)
from chemvector.bond_group_set import (
    GROUP_KIND_TO_ID,
    NO_OBJECT,
    TEXT_BOX,
)
from chemvector.visual_svg_data import VisualSVGCollator


BOND_GROUP_SET_QUERY_COUNT = 192
BOND_GROUP_SET_MAX_MEMBERS = 32
COORDINATE_QUANTIZATION = 2047.0
STROKE_WIDTH_QUANTIZATION = 63.0
_MEMBER_POINT_COUNTS = {"line": 2, "triangle": 3, "quad": 4}


def _canonical_payload(payload: dict[str, Any]) -> dict[str, Any]:
    schema = payload.get("schema_version")
    if schema == "visual-primitives-0.1":
        parsed = visual_primitives_v0_1_to_visual_svg_v0_2(payload)
        return visual_svg_v0_2_to_dict(parsed)
    if schema == "visual-svg-0.2":
        sequence = json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
        ) + "\n"
        parsed = parse_visual_svg_v0_2(sequence)
        # Round-trip through the protocol, rather than accepting a permissive
        # training-only dialect that could diverge from inference.
        return visual_svg_v0_2_to_dict(parsed)
    raise ValueError(f"Unsupported bond-group target schema: {schema!r}")


def _normalized_points(
    member: dict[str, Any],
    *,
    expected_count: int,
) -> list[tuple[float, float]]:
    points = member.get("points")
    if not isinstance(points, list) or len(points) != expected_count:
        raise ValueError(
            f"{member.get('kind', 'member')} requires {expected_count} points"
        )
    normalized = []
    for point in points:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError("Each Visual SVG v0.2 point must contain x and y")
        x, y = point
        if isinstance(x, bool) or not isinstance(x, int) or not 0 <= x <= 2047:
            raise ValueError(f"Invalid q2047 x coordinate: {x!r}")
        if isinstance(y, bool) or not isinstance(y, int) or not 0 <= y <= 2047:
            raise ValueError(f"Invalid q2047 y coordinate: {y!r}")
        normalized.append(
            (float(x) / COORDINATE_QUANTIZATION, float(y) / COORDINATE_QUANTIZATION)
        )
    return normalized


def _member_geometry(member: dict[str, Any]) -> tuple[list[float], int]:
    kind = member.get("kind")
    if kind not in _MEMBER_POINT_COUNTS:
        raise ValueError(f"Unsupported Visual SVG v0.2 member kind: {kind!r}")
    points = _normalized_points(member, expected_count=_MEMBER_POINT_COUNTS[kind])
    flattened = [coordinate for point in points for coordinate in point]
    return flattened, len(flattened)


def build_bond_group_set_target(
    payload: dict[str, Any],
    *,
    query_count: int = BOND_GROUP_SET_QUERY_COUNT,
    max_members: int = BOND_GROUP_SET_MAX_MEMBERS,
) -> dict[str, torch.Tensor]:
    """Encode one canonical v0.2 target without truncating objects or members."""

    if query_count < 1 or max_members < 1:
        raise ValueError("Bond Group Set capacities must be positive")
    canonical = _canonical_payload(payload)
    groups = canonical["bond_groups"]
    text_boxes = canonical["text_boxes"]
    object_count = len(groups) + len(text_boxes)
    if object_count > query_count:
        raise ValueError(
            f"Bond-group object count {object_count} exceeds query capacity {query_count}"
        )

    types = torch.full((query_count,), NO_OBJECT, dtype=torch.long)
    member_counts = torch.zeros(query_count, dtype=torch.long)
    member_geometry = torch.zeros(
        (query_count, max_members, 8),
        dtype=torch.float32,
    )
    member_geometry_mask = torch.zeros(
        (query_count, max_members, 8),
        dtype=torch.bool,
    )
    member_stroke = torch.zeros(
        (query_count, max_members),
        dtype=torch.float32,
    )
    member_mask = torch.zeros((query_count, max_members), dtype=torch.bool)
    text_geometry = torch.zeros((query_count, 4), dtype=torch.float32)
    target_mask = torch.zeros(query_count, dtype=torch.bool)

    for object_index, group in enumerate(groups):
        kind = group.get("kind")
        if kind not in GROUP_KIND_TO_ID or kind == "text_box":
            raise ValueError(f"Unsupported bond-group kind: {kind!r}")
        members = group.get("members")
        if not isinstance(members, list) or not members:
            raise ValueError(f"Bond group {kind!r} must contain members")
        if len(members) > max_members:
            raise ValueError(
                f"Bond group {kind!r} has {len(members)} members, exceeding "
                f"configured max_members={max_members}; targets are never truncated"
            )
        types[object_index] = GROUP_KIND_TO_ID[kind]
        member_counts[object_index] = len(members)
        target_mask[object_index] = True
        for member_index, member in enumerate(members):
            values, coordinate_count = _member_geometry(member)
            member_geometry[object_index, member_index, :coordinate_count] = (
                torch.tensor(values, dtype=torch.float32)
            )
            member_geometry_mask[
                object_index,
                member_index,
                :coordinate_count,
            ] = True
            stroke_width = member.get("stroke_width")
            if (
                isinstance(stroke_width, bool)
                or not isinstance(stroke_width, int)
                or not 1 <= stroke_width <= 63
            ):
                raise ValueError(f"Invalid member stroke width: {stroke_width!r}")
            member_stroke[object_index, member_index] = (
                float(stroke_width) / STROKE_WIDTH_QUANTIZATION
            )
            member_mask[object_index, member_index] = True

    text_offset = len(groups)
    for text_index, text_box in enumerate(text_boxes):
        object_index = text_offset + text_index
        points = _normalized_points(text_box, expected_count=2)
        (x0, y0), (x1, y1) = points
        if x1 <= x0 or y1 <= y0:
            raise ValueError("Visual SVG v0.2 text_box must have positive area")
        types[object_index] = TEXT_BOX
        text_geometry[object_index] = torch.tensor(
            ((x0 + x1) / 2.0, (y0 + y1) / 2.0, x1 - x0, y1 - y0),
            dtype=torch.float32,
        )
        target_mask[object_index] = True

    return {
        "types": types,
        "member_counts": member_counts,
        "member_geometry": member_geometry,
        "member_geometry_mask": member_geometry_mask,
        "member_stroke": member_stroke,
        "member_mask": member_mask,
        "text_geometry": text_geometry,
        "target_mask": target_mask,
        "object_count": torch.tensor(object_count, dtype=torch.long),
    }


def reject_internal_test_sample(sample: dict[str, Any]) -> None:
    """Reject sealed samples even if a caller bypassed shard-path validation."""

    key = str(sample.get("key", ""))
    meta = sample.get("meta", {})
    serialized_meta = json.dumps(meta, ensure_ascii=True, sort_keys=True)
    if "internal_test" in key.lower() or "internal_test" in serialized_meta.lower():
        raise ValueError("Bond Group Set must not read sealed internal_test samples")


class BondGroupSetCollator(VisualSVGCollator):
    """Collate raster images and configurable-capacity v0.2 set targets."""

    def __init__(
        self,
        *,
        image_size: int = 768,
        query_count: int = BOND_GROUP_SET_QUERY_COUNT,
        max_members: int = BOND_GROUP_SET_MAX_MEMBERS,
    ) -> None:
        super().__init__(tokenizer=None, image_size=image_size)
        if query_count < 1 or max_members < 1:
            raise ValueError("Bond Group Set capacities must be positive")
        self.query_count = query_count
        self.max_members = max_members

    @staticmethod
    def _target_payload(sample: dict[str, Any]) -> dict[str, Any]:
        if "visual_svg_v0_2" in sample:
            payload = sample["visual_svg_v0_2"]
        else:
            payload = sample.get("primitives")
        if not isinstance(payload, dict):
            raise ValueError("Sample has no Visual SVG v0.2 or v0.1 primitive target")
        return payload

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        for sample in samples:
            reject_internal_test_sample(sample)
        targets = [
            build_bond_group_set_target(
                self._target_payload(sample),
                query_count=self.query_count,
                max_members=self.max_members,
            )
            for sample in samples
        ]
        tensor_fields = (
            "types",
            "member_counts",
            "member_geometry",
            "member_geometry_mask",
            "member_stroke",
            "member_mask",
            "text_geometry",
            "target_mask",
            "object_count",
        )
        return {
            "pixel_values": torch.stack(
                [self._image_tensor(sample["image"]) for sample in samples]
            ),
            **{
                field: torch.stack([target[field] for target in targets])
                for field in tensor_fields
            },
            "keys": [sample["key"] for sample in samples],
            "meta": [sample["meta"] for sample in samples],
            "bond_layers": [sample["bond_layer"] for sample in samples],
        }


def target_to_visual_svg_v0_2_json(payload: dict[str, Any]) -> str:
    """Expose the exact protocol conversion used by the collator for audits."""

    canonical = _canonical_payload(payload)
    parsed: ParsedVisualSVGSequenceV02 = parse_visual_svg_v0_2(
        json.dumps(canonical, ensure_ascii=True, separators=(",", ":")) + "\n"
    )
    return serialize_parsed_visual_svg_v0_2(parsed)
