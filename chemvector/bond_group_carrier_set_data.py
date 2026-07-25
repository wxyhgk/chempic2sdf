"""Padded Visual SVG v0.2 targets for compact bond-group carriers."""

from __future__ import annotations

import json
from typing import Any

import torch

from Data.sdf.visual_svg_v0_2 import (
    ParsedVisualSVGSequenceV02,
    parse_visual_svg_v0_2,
    visual_primitives_v0_1_to_visual_svg_v0_2,
)
from chemvector.bond_group_carrier_geometry import (
    CARRIER_POINT_WIDTH,
    CARRIER_POSITIVE_WIDTH,
    CARRIER_SIGNED_WIDTH,
    DEFAULT_COORDINATE_SCALE,
    encode_bond_group_carrier,
)
from chemvector.bond_group_set import NO_OBJECT, TEXT_BOX
from chemvector.bond_group_set_data import (
    BondGroupSetCollator,
    reject_internal_test_sample,
)
from chemvector.visual_svg_data import VisualSVGWebDataset


BOND_GROUP_CARRIER_SET_QUERY_COUNT = 192
BOND_GROUP_CARRIER_SET_MAX_MEMBERS = 143


def _positive_capacity(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _canonical_parsed_payload(payload: dict[str, Any]) -> ParsedVisualSVGSequenceV02:
    if not isinstance(payload, dict):
        raise ValueError("Carrier Set target payload must be a JSON object")
    schema = payload.get("schema_version")
    if schema == "visual-primitives-0.1":
        return visual_primitives_v0_1_to_visual_svg_v0_2(payload)
    if schema == "visual-svg-0.2":
        sequence = json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
        ) + "\n"
        return parse_visual_svg_v0_2(sequence)
    raise ValueError(f"Unsupported carrier target schema: {schema!r}")


def build_bond_group_carrier_set_target(
    payload: dict[str, Any],
    *,
    query_count: int = BOND_GROUP_CARRIER_SET_QUERY_COUNT,
    max_members: int = BOND_GROUP_CARRIER_SET_MAX_MEMBERS,
) -> dict[str, torch.Tensor]:
    """Encode one v0.1/v0.2 sidecar without truncating objects or members."""

    query_count = _positive_capacity(query_count, "Carrier Set query_count")
    max_members = _positive_capacity(max_members, "Carrier Set max_members")
    parsed = _canonical_parsed_payload(payload)
    object_count = len(parsed.bond_groups) + len(parsed.text_boxes)
    if object_count > query_count:
        raise ValueError(
            f"Carrier object count {object_count} exceeds query capacity "
            f"{query_count}; targets are never truncated"
        )

    types = torch.full((query_count,), NO_OBJECT, dtype=torch.long)
    member_counts = torch.zeros(query_count, dtype=torch.long)
    carrier_points = torch.zeros(
        (query_count, CARRIER_POINT_WIDTH),
        dtype=torch.float32,
    )
    point_mask = torch.zeros_like(carrier_points, dtype=torch.bool)
    positive_params = torch.zeros(
        (query_count, CARRIER_POSITIVE_WIDTH),
        dtype=torch.float32,
    )
    positive_param_mask = torch.zeros_like(positive_params, dtype=torch.bool)
    signed_params = torch.zeros(
        (query_count, CARRIER_SIGNED_WIDTH),
        dtype=torch.float32,
    )
    signed_param_mask = torch.zeros_like(signed_params, dtype=torch.bool)
    group_stroke = torch.zeros(query_count, dtype=torch.float32)
    text_geometry = torch.zeros((query_count, 4), dtype=torch.float32)
    target_mask = torch.zeros(query_count, dtype=torch.bool)

    for object_index, group in enumerate(parsed.bond_groups):
        encoding = encode_bond_group_carrier(group)
        member_count = int(encoding.member_count)
        if member_count != len(group.members):
            raise ValueError(
                f"Carrier codec member count {member_count} differs from "
                f"protocol group size {len(group.members)}"
            )
        if member_count > max_members:
            raise ValueError(
                f"Bond group {group.kind!r} has {member_count} members, "
                f"exceeding configured max_members={max_members}; targets are "
                "never truncated"
            )
        types[object_index] = encoding.kind_id
        member_counts[object_index] = encoding.member_count
        carrier_points[object_index] = encoding.carrier_points
        point_mask[object_index] = encoding.carrier_point_mask
        positive_params[object_index] = encoding.positive_params
        positive_param_mask[object_index] = encoding.positive_param_mask
        signed_params[object_index] = encoding.signed_params
        signed_param_mask[object_index] = encoding.signed_param_mask
        group_stroke[object_index] = encoding.group_stroke
        target_mask[object_index] = True

    text_offset = len(parsed.bond_groups)
    for text_index, text_box in enumerate(parsed.text_boxes):
        object_index = text_offset + text_index
        x0 = float(text_box.x0) / DEFAULT_COORDINATE_SCALE
        y0 = float(text_box.y0) / DEFAULT_COORDINATE_SCALE
        x1 = float(text_box.x1) / DEFAULT_COORDINATE_SCALE
        y1 = float(text_box.y1) / DEFAULT_COORDINATE_SCALE
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
        "carrier_points": carrier_points,
        "point_mask": point_mask,
        "positive_params": positive_params,
        "positive_param_mask": positive_param_mask,
        "signed_params": signed_params,
        "signed_param_mask": signed_param_mask,
        "group_stroke": group_stroke,
        "text_geometry": text_geometry,
        "target_mask": target_mask,
        "object_count": torch.tensor(object_count, dtype=torch.long),
    }


class BondGroupCarrierSetCollator(BondGroupSetCollator):
    """Collate shared raster inputs with fixed-width carrier-set targets."""

    def __init__(
        self,
        *,
        image_size: int = 768,
        query_count: int = BOND_GROUP_CARRIER_SET_QUERY_COUNT,
        max_members: int = BOND_GROUP_CARRIER_SET_MAX_MEMBERS,
    ) -> None:
        super().__init__(
            image_size=image_size,
            query_count=_positive_capacity(query_count, "Carrier Set query_count"),
            max_members=_positive_capacity(max_members, "Carrier Set max_members"),
        )

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        for sample in samples:
            reject_internal_test_sample(sample)
        targets = [
            build_bond_group_carrier_set_target(
                self._target_payload(sample),
                query_count=self.query_count,
                max_members=self.max_members,
            )
            for sample in samples
        ]
        tensor_fields = (
            "types",
            "member_counts",
            "carrier_points",
            "point_mask",
            "positive_params",
            "positive_param_mask",
            "signed_params",
            "signed_param_mask",
            "group_stroke",
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


__all__ = [
    "BOND_GROUP_CARRIER_SET_MAX_MEMBERS",
    "BOND_GROUP_CARRIER_SET_QUERY_COUNT",
    "BondGroupCarrierSetCollator",
    "VisualSVGWebDataset",
    "build_bond_group_carrier_set_target",
]
