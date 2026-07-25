"""Oracle-keep proposal membership targets for frozen flat primitives.

The module deliberately separates image/GT collation from the frozen flat-model
forward pass.  Proposal inputs are copied only from flat-model outputs.  The
Visual SVG primitive sidecar is used exclusively to assign target group labels
and never supplies proposal coordinates or features.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from Data.sdf.visual_primitives_v0_1 import (
    validate_visual_primitive_references,
)
from Data.sdf.visual_svg_sequence_v0_1 import (
    LineRecordV01,
    QuadRecordV01,
    TriangleRecordV01,
)
from Data.sdf.visual_svg_v0_2 import (
    BondGroupRecordV02,
    LineMemberV02,
    ParsedVisualSVGSequenceV02,
    QuadMemberV02,
    TriangleMemberV02,
    canonicalize_visual_svg_v0_2,
)
from chemvector.bond_group_set import GROUP_KIND_TO_ID
from chemvector.bond_group_set_data import reject_internal_test_sample
from chemvector.evaluate_primitive_set import decoded_rows_to_parsed
from chemvector.evaluate_visual_svg import line_distance, polygon_distance
from chemvector.primitive_set import (
    LINE,
    QUAD,
    TEXT_BOX,
    TRIANGLE,
    decode_primitive_set,
)
from chemvector.primitive_set_data import (
    PRIMITIVE_SET_QUERY_COUNT,
    PrimitiveSetCollator,
)


DEFAULT_DECODE_MODE = "threshold"
DEFAULT_PROPOSAL_THRESHOLD = 0.5
DEFAULT_HIGH_CONFIDENCE_TOLERANCE_Q = 8.0
DEFAULT_ASSOCIATION_TOLERANCE_Q = 32.0
DEFAULT_SHELL_WEIGHT = 0.25
PADDING_GROUP_ID = -1

_PHYSICAL_KIND_TO_TYPE = {
    "line": LINE,
    "triangle": TRIANGLE,
    "quad": QUAD,
}
_FIXED_GROUP_KINDS = {
    "single_line",
    "parallel_2",
    "parallel_3",
    "solid_triangle",
    "solid_quad",
}


PhysicalMember = LineMemberV02 | TriangleMemberV02 | QuadMemberV02
PhysicalRecord = LineRecordV01 | TriangleRecordV01 | QuadRecordV01


@dataclass(frozen=True, slots=True)
class PrimitiveGroupTarget:
    """One physical primitive's group-only supervision."""

    primitive_id: int
    primitive_type: int
    group_id: int
    group_kind: str
    group_type: int
    group_count: int
    record: PhysicalRecord


@dataclass(frozen=True, slots=True)
class FlatProposal:
    """One renderable flat proposal with model-derived inputs only."""

    query_index: int
    primitive_type: int
    score: float
    geometry: torch.Tensor
    stroke: float
    record: PhysicalRecord


def _positive_number(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _validate_matching_configuration(
    *,
    high_confidence_tolerance_q: float,
    association_tolerance_q: float,
    shell_weight: float,
) -> tuple[float, float, float]:
    high = _positive_number(
        high_confidence_tolerance_q,
        "high_confidence_tolerance_q",
    )
    maximum = _positive_number(
        association_tolerance_q,
        "association_tolerance_q",
    )
    if high > maximum:
        raise ValueError(
            "high_confidence_tolerance_q must not exceed "
            "association_tolerance_q"
        )
    weight = float(shell_weight)
    if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("shell_weight must be finite and in [0, 1]")
    return high, maximum, weight


def _points(primitive: dict[str, Any], expected: int) -> tuple[tuple[int, int], ...]:
    values = primitive.get("quantized_points")
    if not isinstance(values, list) or len(values) != expected:
        raise ValueError(
            f"{primitive.get('kind', 'primitive')} requires {expected} "
            "quantized points"
        )
    result = []
    for point in values:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError("Each quantized point must contain x and y")
        x, y = point
        if isinstance(x, bool) or isinstance(y, bool):
            raise ValueError("Quantized coordinates must be integers")
        x, y = int(x), int(y)
        if not 0 <= x <= 2047 or not 0 <= y <= 2047:
            raise ValueError("Quantized coordinates must be in q2047")
        result.append((x, y))
    return tuple(result)


def _stroke(primitive: dict[str, Any]) -> int:
    value = primitive.get("stroke_width_q")
    if isinstance(value, bool):
        raise ValueError("stroke_width_q must be an integer")
    value = int(value)
    if not 1 <= value <= 63:
        raise ValueError("stroke_width_q must be in 1..63")
    return value


def _canonical_member(
    kind: str,
    points: tuple[tuple[int, int], ...],
    stroke_width: int,
) -> PhysicalMember:
    if kind == "line":
        member: PhysicalMember = LineMemberV02(
            *points[0],
            *points[1],
            stroke_width,
        )
        group_kind = "single_line"
    elif kind == "triangle":
        member = TriangleMemberV02(
            *(coordinate for point in points for coordinate in point),
            stroke_width,
        )
        group_kind = "solid_triangle"
    elif kind == "quad":
        member = QuadMemberV02(
            *(coordinate for point in points for coordinate in point),
            stroke_width,
        )
        group_kind = "solid_quad"
    else:
        raise ValueError(f"Unsupported physical primitive kind: {kind!r}")
    parsed = canonicalize_visual_svg_v0_2(
        ParsedVisualSVGSequenceV02(
            (BondGroupRecordV02(group_kind, (member,)),),
            (),
        )
    )
    return parsed.bond_groups[0].members[0]


def _as_v0_1(member: PhysicalMember) -> PhysicalRecord:
    values = [
        getattr(member, name)
        for name in member.__dataclass_fields__
        if name != "stroke_width"
    ]
    return {
        LineMemberV02: LineRecordV01,
        TriangleMemberV02: TriangleRecordV01,
        QuadMemberV02: QuadRecordV01,
    }[type(member)](*values, member.stroke_width)


def _primitive_record(primitive: dict[str, Any]) -> PhysicalRecord:
    kind = str(primitive.get("kind", ""))
    point_count = {"line": 2, "triangle": 3, "quad": 4}.get(kind)
    if point_count is None:
        raise ValueError(f"Unsupported physical primitive kind: {kind!r}")
    return _as_v0_1(
        _canonical_member(
            kind,
            _points(primitive, point_count),
            _stroke(primitive),
        )
    )


def _group_kind(
    group: dict[str, Any],
    primitives: list[dict[str, Any]],
) -> str:
    kinds = [str(primitive.get("kind", "")) for primitive in primitives]
    if kinds == ["triangle"]:
        return "solid_triangle"
    if kinds == ["quad"]:
        return "solid_quad"
    if not kinds or any(kind != "line" for kind in kinds):
        raise ValueError("Bond groups must contain only one physical member kind")
    if str(group.get("direction", "")) == "BEGINDASH":
        return "hashed_wedge"
    if len(kinds) == 1:
        return "single_line"
    if len(kinds) == 2:
        return "parallel_2"
    if len(kinds) == 3:
        return "parallel_3"
    raise ValueError(
        f"Cannot classify a {len(kinds)}-line group without BEGINDASH"
    )


def build_primitive_group_targets(
    payload: dict[str, Any],
) -> dict[int, PrimitiveGroupTarget]:
    """Map each physical primitive ID to one chemistry-free group target."""

    if not isinstance(payload, dict) or payload.get("schema_version") != (
        "visual-primitives-0.1"
    ):
        raise ValueError("Proposal membership requires visual-primitives-0.1")
    validate_visual_primitive_references(payload)
    primitives = payload.get("primitives")
    groups = payload.get("bond_groups")
    if not isinstance(primitives, list) or not isinstance(groups, list):
        raise ValueError("Primitive sidecar requires primitives and bond_groups arrays")
    by_id = {int(item["primitive_id"]): item for item in primitives}
    result: dict[int, PrimitiveGroupTarget] = {}
    for group_id, group in enumerate(groups):
        primitive_ids = [int(value) for value in group.get("primitive_ids", [])]
        members = [by_id[primitive_id] for primitive_id in primitive_ids]
        kind = _group_kind(group, members)
        group_type = GROUP_KIND_TO_ID[kind]
        group_count = len(members)
        for primitive_id, primitive in zip(primitive_ids, members):
            if primitive_id in result:
                raise ValueError(f"Primitive {primitive_id} belongs to multiple groups")
            primitive_kind = str(primitive.get("kind", ""))
            result[primitive_id] = PrimitiveGroupTarget(
                primitive_id=primitive_id,
                primitive_type=_PHYSICAL_KIND_TO_TYPE[primitive_kind],
                group_id=group_id,
                group_kind=kind,
                group_type=group_type,
                group_count=group_count,
                record=_primitive_record(primitive),
            )
    expected_physical = {
        int(primitive["primitive_id"])
        for primitive in primitives
        if primitive.get("kind") != "text_box"
    }
    if set(result) != expected_physical:
        raise ValueError("Every physical primitive must belong to one bond group")
    return result


def _prediction_record(payload: dict[str, Any]) -> PhysicalRecord:
    kind = str(payload.get("kind", ""))
    point_count = {"line": 2, "triangle": 3, "quad": 4}.get(kind)
    if point_count is None:
        raise ValueError(f"Unsupported proposal kind: {kind!r}")
    return _as_v0_1(
        _canonical_member(
            kind,
            _points(payload, point_count),
            _stroke(payload),
        )
    )


def _physical_proposals(
    decoded_rows: list[dict[str, Any]],
) -> list[FlatProposal]:
    converted = decoded_rows_to_parsed(decoded_rows)
    row_by_query = {int(row["query_index"]): row for row in decoded_rows}
    proposals = []
    for payload in converted.renderable_objects:
        primitive_type = int(payload.get("type_id", 0))
        if primitive_type == TEXT_BOX:
            continue
        if primitive_type not in (LINE, TRIANGLE, QUAD):
            continue
        query_index = int(payload["query_index"])
        row = row_by_query[query_index]
        geometry = row["geometry"]
        if not isinstance(geometry, torch.Tensor):
            geometry = torch.as_tensor(geometry)
        proposals.append(
            FlatProposal(
                query_index=query_index,
                primitive_type=primitive_type,
                score=float(row["score"]),
                geometry=geometry.detach(),
                stroke=float(row["stroke"]),
                record=_prediction_record(payload),
            )
        )
    return proposals


def _distance(left: PhysicalRecord, right: PhysicalRecord) -> float:
    if isinstance(left, LineRecordV01) and isinstance(right, LineRecordV01):
        return line_distance(left, right)
    if type(left) is type(right) and isinstance(
        left,
        (TriangleRecordV01, QuadRecordV01),
    ):
        return polygon_distance(left, right)
    return float("inf")


def associate_proposals_to_groups(
    proposals: list[FlatProposal],
    targets: dict[int, PrimitiveGroupTarget],
    *,
    high_confidence_tolerance_q: float = DEFAULT_HIGH_CONFIDENCE_TOLERANCE_Q,
    association_tolerance_q: float = DEFAULT_ASSOCIATION_TOLERANCE_Q,
    shell_weight: float = DEFAULT_SHELL_WEIGHT,
) -> list[dict[str, Any]]:
    """Hungarian-match same-kind proposals and mark usable group supervision."""

    high, maximum, shell_weight = _validate_matching_configuration(
        high_confidence_tolerance_q=high_confidence_tolerance_q,
        association_tolerance_q=association_tolerance_q,
        shell_weight=shell_weight,
    )
    assignments: dict[int, tuple[PrimitiveGroupTarget, float]] = {}
    target_values = list(targets.values())
    for primitive_type in (LINE, TRIANGLE, QUAD):
        proposal_indices = [
            index
            for index, proposal in enumerate(proposals)
            if proposal.primitive_type == primitive_type
        ]
        candidates = [proposals[index] for index in proposal_indices]
        expected = [
            target
            for target in target_values
            if target.primitive_type == primitive_type
        ]
        if not candidates or not expected:
            continue
        costs = np.asarray(
            [
                [_distance(candidate.record, target.record) for target in expected]
                for candidate in candidates
            ],
            dtype=np.float64,
        )
        rows, columns = linear_sum_assignment(costs)
        for row, column in zip(rows, columns):
            distance = float(costs[row, column])
            if distance > maximum:
                continue
            assignments[proposal_indices[int(row)]] = (
                expected[int(column)],
                distance,
            )

    matched_per_group = Counter(
        target.group_id for target, _ in assignments.values()
    )
    usable_groups = set()
    for target in target_values:
        matched = matched_per_group[target.group_id]
        if target.group_kind == "hashed_wedge":
            usable = matched >= 2
        elif target.group_kind in _FIXED_GROUP_KINDS:
            usable = matched == target.group_count
        else:
            raise ValueError(f"Unsupported group kind: {target.group_kind!r}")
        if usable:
            usable_groups.add(target.group_id)

    rows = []
    for proposal_index, proposal in enumerate(proposals):
        match = assignments.get(proposal_index)
        if match is None:
            continue
        target, distance = match
        usable = target.group_id in usable_groups
        weight = 1.0 if distance <= high else shell_weight
        if not usable:
            weight = 0.0
        rows.append(
            {
                "proposal": proposal,
                "group_id": target.group_id,
                "group_type": target.group_type,
                "group_count": target.group_count,
                "group_mask": usable,
                "match_distance_q": distance,
                "match_weight": weight,
            }
        )
    return rows


def _validate_flat_output(
    flat_output: dict[str, torch.Tensor],
    *,
    batch_size: int,
) -> tuple[int, int, torch.device]:
    required = ("type_logits", "geometry", "stroke", "object_features")
    missing = [name for name in required if name not in flat_output]
    if missing:
        raise ValueError(f"Flat output is missing tensors: {missing}")
    type_logits = flat_output["type_logits"]
    features = flat_output["object_features"]
    if type_logits.ndim != 3 or type_logits.size(0) != batch_size:
        raise ValueError("type_logits must have shape [batch, queries, classes]")
    if features.ndim != 3 or features.shape[:2] != type_logits.shape[:2]:
        raise ValueError("object_features must align with flat queries")
    for name in ("geometry", "stroke"):
        tensor = flat_output[name]
        if tensor.shape[:2] != type_logits.shape[:2]:
            raise ValueError(f"{name} must align with flat queries")
    return type_logits.size(1), features.size(2), features.device


def build_proposal_membership_batch(
    flat_output: dict[str, torch.Tensor],
    primitive_sidecars: list[dict[str, Any]],
    *,
    decode_mode: str = DEFAULT_DECODE_MODE,
    proposal_threshold: float = DEFAULT_PROPOSAL_THRESHOLD,
    high_confidence_tolerance_q: float = DEFAULT_HIGH_CONFIDENCE_TOLERANCE_Q,
    association_tolerance_q: float = DEFAULT_ASSOCIATION_TOLERANCE_Q,
    shell_weight: float = DEFAULT_SHELL_WEIGHT,
    max_proposals: int = PRIMITIVE_SET_QUERY_COUNT,
) -> dict[str, torch.Tensor]:
    """Build padded oracle-keep targets from frozen flat-model outputs."""

    if decode_mode not in {"threshold", "count_head"}:
        raise ValueError("decode_mode must be 'threshold' or 'count_head'")
    if not 0.0 <= proposal_threshold <= 1.0:
        raise ValueError("proposal_threshold must be in [0, 1]")
    if max_proposals < 1:
        raise ValueError("max_proposals must be positive")
    batch_size = len(primitive_sidecars)
    query_count, feature_width, device = _validate_flat_output(
        flat_output,
        batch_size=batch_size,
    )
    decoded = decode_primitive_set(
        flat_output,
        mode=decode_mode,
        threshold=proposal_threshold,
    )
    feature_dtype = flat_output["object_features"].dtype
    geometry_dtype = flat_output["geometry"].dtype
    proposal_features = torch.zeros(
        (batch_size, max_proposals, feature_width),
        device=device,
        dtype=feature_dtype,
    )
    proposal_geometry = torch.zeros(
        (batch_size, max_proposals, 8),
        device=device,
        dtype=geometry_dtype,
    )
    proposal_stroke = torch.zeros(
        (batch_size, max_proposals), device=device, dtype=torch.float32
    )
    proposal_score = torch.zeros_like(proposal_stroke)
    proposal_type = torch.zeros(
        (batch_size, max_proposals), device=device, dtype=torch.long
    )
    proposal_query_index = torch.full(
        (batch_size, max_proposals),
        -1,
        device=device,
        dtype=torch.long,
    )
    proposal_mask = torch.zeros(
        (batch_size, max_proposals), device=device, dtype=torch.bool
    )
    group_membership = torch.full(
        (batch_size, max_proposals),
        PADDING_GROUP_ID,
        device=device,
        dtype=torch.long,
    )
    group_type = torch.zeros(
        (batch_size, max_proposals), device=device, dtype=torch.long
    )
    group_count = torch.zeros_like(group_type)
    group_mask = torch.zeros_like(proposal_mask)
    match_distance_q = torch.full(
        (batch_size, max_proposals),
        float("inf"),
        device=device,
        dtype=torch.float32,
    )
    match_weight = torch.zeros_like(match_distance_q)
    proposal_count = torch.zeros(batch_size, device=device, dtype=torch.long)

    for batch_index, (sidecar, decoded_rows) in enumerate(
        zip(primitive_sidecars, decoded)
    ):
        targets = build_primitive_group_targets(sidecar)
        proposals = _physical_proposals(decoded_rows)
        matched = associate_proposals_to_groups(
            proposals,
            targets,
            high_confidence_tolerance_q=high_confidence_tolerance_q,
            association_tolerance_q=association_tolerance_q,
            shell_weight=shell_weight,
        )
        if len(matched) > max_proposals:
            raise ValueError(
                f"Matched proposal count {len(matched)} exceeds max_proposals="
                f"{max_proposals}; proposals are never truncated"
            )
        proposal_count[batch_index] = len(matched)
        for output_index, row in enumerate(matched):
            proposal: FlatProposal = row["proposal"]
            if not 0 <= proposal.query_index < query_count:
                raise ValueError("Decoded proposal query index is out of range")
            proposal_features[batch_index, output_index] = flat_output[
                "object_features"
            ][batch_index, proposal.query_index].detach()
            proposal_geometry[batch_index, output_index] = proposal.geometry.to(
                device=device,
                dtype=geometry_dtype,
            )
            proposal_stroke[batch_index, output_index] = proposal.stroke
            proposal_score[batch_index, output_index] = proposal.score
            proposal_type[batch_index, output_index] = proposal.primitive_type
            proposal_query_index[batch_index, output_index] = proposal.query_index
            proposal_mask[batch_index, output_index] = True
            group_membership[batch_index, output_index] = row["group_id"]
            group_type[batch_index, output_index] = row["group_type"]
            group_count[batch_index, output_index] = row["group_count"]
            group_mask[batch_index, output_index] = row["group_mask"]
            match_distance_q[batch_index, output_index] = row["match_distance_q"]
            match_weight[batch_index, output_index] = row["match_weight"]

    return {
        "proposal_features": proposal_features.detach(),
        "proposal_geometry": proposal_geometry.detach(),
        "proposal_stroke": proposal_stroke,
        "proposal_score": proposal_score,
        "proposal_type": proposal_type,
        "proposal_query_index": proposal_query_index,
        "proposal_mask": proposal_mask,
        "proposal_count": proposal_count,
        "group_membership": group_membership,
        "group_type": group_type,
        "group_count": group_count,
        "group_mask": group_mask,
        "match_distance_q": match_distance_q,
        "match_weight": match_weight,
    }


class BondGroupProposalMembershipCollator(PrimitiveSetCollator):
    """Two-stage collator for flat targets and frozen proposal membership."""

    def __init__(
        self,
        *,
        image_size: int = 768,
        decode_mode: str = DEFAULT_DECODE_MODE,
        proposal_threshold: float = DEFAULT_PROPOSAL_THRESHOLD,
        high_confidence_tolerance_q: float = (
            DEFAULT_HIGH_CONFIDENCE_TOLERANCE_Q
        ),
        association_tolerance_q: float = DEFAULT_ASSOCIATION_TOLERANCE_Q,
        shell_weight: float = DEFAULT_SHELL_WEIGHT,
        max_proposals: int = PRIMITIVE_SET_QUERY_COUNT,
    ) -> None:
        super().__init__(image_size=image_size)
        if decode_mode not in {"threshold", "count_head"}:
            raise ValueError("decode_mode must be 'threshold' or 'count_head'")
        if not 0.0 <= proposal_threshold <= 1.0:
            raise ValueError("proposal_threshold must be in [0, 1]")
        _validate_matching_configuration(
            high_confidence_tolerance_q=high_confidence_tolerance_q,
            association_tolerance_q=association_tolerance_q,
            shell_weight=shell_weight,
        )
        if max_proposals < 1:
            raise ValueError("max_proposals must be positive")
        self.decode_mode = decode_mode
        self.proposal_threshold = proposal_threshold
        self.high_confidence_tolerance_q = high_confidence_tolerance_q
        self.association_tolerance_q = association_tolerance_q
        self.shell_weight = shell_weight
        self.max_proposals = max_proposals

    def collate_flat_targets(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        for sample in samples:
            reject_internal_test_sample(sample)
            if not isinstance(sample.get("primitives"), dict):
                raise ValueError("Sample lacks a primitives sidecar")
        batch = super().__call__(samples)
        batch["primitives"] = [sample["primitives"] for sample in samples]
        return batch

    def attach_flat_output(
        self,
        batch: dict[str, Any],
        flat_output: dict[str, torch.Tensor],
    ) -> dict[str, Any]:
        primitives = batch.get("primitives")
        if not isinstance(primitives, list):
            raise ValueError("Flat target batch lacks primitive sidecars")
        membership = build_proposal_membership_batch(
            flat_output,
            primitives,
            decode_mode=self.decode_mode,
            proposal_threshold=self.proposal_threshold,
            high_confidence_tolerance_q=self.high_confidence_tolerance_q,
            association_tolerance_q=self.association_tolerance_q,
            shell_weight=self.shell_weight,
            max_proposals=self.max_proposals,
        )
        return {**batch, **membership}

    def __call__(
        self,
        samples: list[dict[str, Any]],
        flat_output: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        batch = self.collate_flat_targets(samples)
        if flat_output is None:
            return batch
        return self.attach_flat_output(batch, flat_output)


__all__ = [
    "DEFAULT_ASSOCIATION_TOLERANCE_Q",
    "DEFAULT_DECODE_MODE",
    "DEFAULT_HIGH_CONFIDENCE_TOLERANCE_Q",
    "DEFAULT_PROPOSAL_THRESHOLD",
    "DEFAULT_SHELL_WEIGHT",
    "BondGroupProposalMembershipCollator",
    "FlatProposal",
    "PrimitiveGroupTarget",
    "associate_proposals_to_groups",
    "build_primitive_group_targets",
    "build_proposal_membership_batch",
]
