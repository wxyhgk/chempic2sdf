"""Integration helpers for frozen flat proposals and membership grouping.

Every model input in this module is derived from frozen flat outputs.  GT
primitive sidecars contribute only labels, expected sequences and coverage
denominators; their coordinates never enter descriptors or carrier fitting.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import itertools
import math
from typing import Any, Iterable

import torch

from Data.sdf.visual_svg_v0_2 import (
    BondGroupRecordV02,
    LineMemberV02,
    ParsedVisualSVGSequenceV02,
    QuadMemberV02,
    TriangleMemberV02,
    canonicalize_visual_svg_v0_2,
    serialize_parsed_visual_svg_v0_2,
    visual_primitives_v0_1_to_visual_svg_v0_2,
)
from chemvector.bond_group_carrier_geometry import encode_bond_group_carrier
from chemvector.bond_group_proposal_membership import (
    FIXED_MEMBER_COUNTS,
    HASHED_WEDGE,
    LINE_GROUP_TYPES,
    PHYSICAL_GROUP_NAMES,
)
from chemvector.bond_group_set import (
    SOLID_QUAD,
    SOLID_TRIANGLE,
)
from chemvector.evaluate_bond_group_carrier_set import (
    carrier_rows_to_bond_group_rows,
)
from chemvector.primitive_set import LINE, NO_OBJECT, QUAD, TRIANGLE


PROPOSAL_DESCRIPTOR_DIM = 23
PROPOSAL_DESCRIPTOR_FIELDS = (
    *(f"geometry_{index}" for index in range(8)),
    *(f"active_mask_{index}" for index in range(8)),
    "stroke",
    "score",
    "query_reference_x",
    "query_reference_y",
    "center_x",
    "center_y",
    "span",
)
DEFAULT_GROUP_SLOTS = 192
DEFAULT_MAX_MEMBERS = 143
FLAT_TYPE_CLASSES = 5

_COORDINATE_COUNTS = {LINE: 4, TRIANGLE: 6, QUAD: 8}


@dataclass(frozen=True, slots=True)
class EvaluatorRowsResult:
    rows: tuple[dict[str, Any], ...]
    fit_failures: tuple[dict[str, Any], ...]


def _validate_proposal_batch(batch: dict[str, Any]) -> tuple[int, int, torch.device]:
    required = (
        "proposal_features",
        "proposal_geometry",
        "proposal_stroke",
        "proposal_score",
        "proposal_type",
        "proposal_query_index",
        "proposal_mask",
    )
    missing = [name for name in required if name not in batch]
    if missing:
        raise ValueError(f"Proposal batch is missing fields: {missing}")
    features = batch["proposal_features"]
    if features.ndim != 3:
        raise ValueError("proposal_features must have shape [batch, proposals, dim]")
    batch_size, proposals, _ = features.shape
    expected = {
        "proposal_geometry": (batch_size, proposals, 8),
        "proposal_stroke": (batch_size, proposals),
        "proposal_score": (batch_size, proposals),
        "proposal_type": (batch_size, proposals),
        "proposal_query_index": (batch_size, proposals),
        "proposal_mask": (batch_size, proposals),
    }
    for name, shape in expected.items():
        if batch[name].shape != shape:
            raise ValueError(
                f"{name} has shape {tuple(batch[name].shape)}, expected {shape}"
            )
    return batch_size, proposals, features.device


def _coordinate_active_mask(
    proposal_types: torch.Tensor,
    proposal_mask: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    counts = torch.zeros_like(proposal_types, dtype=torch.long)
    for primitive_type, count in _COORDINATE_COUNTS.items():
        counts = torch.where(
            proposal_types == primitive_type,
            torch.full_like(counts, count),
            counts,
        )
    invalid = proposal_mask.bool() & (counts == 0)
    if bool(invalid.any()):
        raise ValueError("Active proposals require line, triangle or quad type")
    indices = torch.arange(8, device=proposal_types.device)
    return (indices < counts.unsqueeze(-1)).to(dtype)


def _gather_query_references(
    query_reference_points: torch.Tensor,
    query_indices: torch.Tensor,
    proposal_mask: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    batch, proposals = query_indices.shape
    if query_reference_points.ndim == 2:
        if query_reference_points.size(-1) != 2:
            raise ValueError("Query references must end with x/y")
        references = query_reference_points.unsqueeze(0).expand(batch, -1, -1)
    elif query_reference_points.ndim == 3:
        if query_reference_points.shape[0] != batch or query_reference_points.size(-1) != 2:
            raise ValueError("Batched query references must be [batch, queries, 2]")
        references = query_reference_points
    else:
        raise ValueError("Query references must be [queries, 2] or [batch, queries, 2]")
    references = references.to(device=query_indices.device, dtype=dtype)
    query_count = references.size(1)
    active_indices = query_indices[proposal_mask.bool()]
    if len(active_indices) and bool(
        ((active_indices < 0) | (active_indices >= query_count)).any()
    ):
        raise ValueError("Active proposal query index is outside reference capacity")
    safe = query_indices.clamp(0, max(query_count - 1, 0))
    gathered = torch.gather(
        references,
        1,
        safe.unsqueeze(-1).expand(batch, proposals, 2),
    )
    return gathered.masked_fill(~proposal_mask.bool().unsqueeze(-1), 0.0)


def build_proposal_descriptors(
    batch: dict[str, Any],
    query_reference_points: torch.Tensor,
) -> torch.Tensor:
    """Build the strict 23-D descriptor from frozen proposal values only."""

    _, _, _ = _validate_proposal_batch(batch)
    geometry = batch["proposal_geometry"].detach()
    mask = batch["proposal_mask"].bool()
    if not bool(torch.isfinite(geometry[mask]).all()):
        raise ValueError("Active proposal geometry contains non-finite values")
    active_coordinates = _coordinate_active_mask(
        batch["proposal_type"].long(),
        mask,
        dtype=geometry.dtype,
    )
    point_mask = active_coordinates.reshape(*active_coordinates.shape[:-1], 4, 2)[
        ..., 0
    ].bool()
    points = geometry.reshape(*geometry.shape[:-1], 4, 2)
    point_count = point_mask.sum(dim=-1).clamp_min(1).to(geometry.dtype)
    center = (
        points * point_mask.unsqueeze(-1).to(geometry.dtype)
    ).sum(dim=-2) / point_count.unsqueeze(-1)
    positive_infinity = torch.full_like(points, float("inf"))
    negative_infinity = torch.full_like(points, float("-inf"))
    minimum = torch.where(point_mask.unsqueeze(-1), points, positive_infinity).amin(
        dim=-2
    )
    maximum = torch.where(point_mask.unsqueeze(-1), points, negative_infinity).amax(
        dim=-2
    )
    span = torch.linalg.vector_norm(maximum - minimum, dim=-1, keepdim=True)
    center = center.masked_fill(~mask.unsqueeze(-1), 0.0)
    span = span.masked_fill(~mask.unsqueeze(-1), 0.0)
    references = _gather_query_references(
        query_reference_points,
        batch["proposal_query_index"].long(),
        mask,
        dtype=geometry.dtype,
    )
    descriptor = torch.cat(
        (
            geometry,
            active_coordinates,
            batch["proposal_stroke"].detach().to(geometry.dtype).unsqueeze(-1),
            batch["proposal_score"].detach().to(geometry.dtype).unsqueeze(-1),
            references,
            center,
            span,
        ),
        dim=-1,
    )
    if descriptor.size(-1) != PROPOSAL_DESCRIPTOR_DIM:
        raise RuntimeError(
            f"Proposal descriptor width {descriptor.size(-1)} != "
            f"{PROPOSAL_DESCRIPTOR_DIM}"
        )
    descriptor = descriptor.masked_fill(~mask.unsqueeze(-1), 0.0)
    if not bool(torch.isfinite(descriptor).all()):
        raise ValueError("Proposal descriptors contain non-finite values")
    return descriptor.detach()


def gather_flat_type_logits(
    flat_type_logits: torch.Tensor,
    proposal_query_index: torch.Tensor,
    proposal_mask: torch.Tensor,
) -> torch.Tensor:
    """Gather frozen flat type logits by the retained proposal query indices."""

    if flat_type_logits.ndim != 3 or flat_type_logits.size(-1) != FLAT_TYPE_CLASSES:
        raise ValueError(
            "Flat type logits must have shape [batch, queries, 5]"
        )
    batch, proposals = proposal_query_index.shape
    if proposal_mask.shape != (batch, proposals):
        raise ValueError("Proposal mask and query indices differ")
    if flat_type_logits.size(0) != batch:
        raise ValueError("Flat logits and proposal batch sizes differ")
    query_count = flat_type_logits.size(1)
    active = proposal_query_index[proposal_mask.bool()]
    if len(active) and bool(((active < 0) | (active >= query_count)).any()):
        raise ValueError("Active proposal query index is outside flat logits")
    safe = proposal_query_index.long().clamp(0, max(query_count - 1, 0))
    gathered = torch.gather(
        flat_type_logits,
        1,
        safe.unsqueeze(-1).expand(batch, proposals, FLAT_TYPE_CLASSES),
    )
    return gathered.masked_fill(~proposal_mask.bool().unsqueeze(-1), 0.0).detach()


def _required_proposal_type(group_type: int) -> int:
    if group_type in LINE_GROUP_TYPES:
        return LINE
    if group_type == SOLID_TRIANGLE:
        return TRIANGLE
    if group_type == SOLID_QUAD:
        return QUAD
    raise ValueError(f"Unsupported physical group type: {group_type}")


def aggregate_group_slot_targets(
    batch: dict[str, Any],
    *,
    group_slots: int = DEFAULT_GROUP_SLOTS,
) -> dict[str, torch.Tensor]:
    """Aggregate per-proposal oracle labels into padded group-slot targets."""

    batch_size, proposals, device = _validate_proposal_batch(batch)
    required = (
        "group_membership",
        "group_type",
        "group_count",
        "group_mask",
        "match_weight",
    )
    missing = [name for name in required if name not in batch]
    if missing:
        raise ValueError(f"Proposal group labels are missing: {missing}")
    if group_slots < 1:
        raise ValueError("group_slots must be positive")
    proposal_shape = (batch_size, proposals)
    for name in required:
        if batch[name].shape != proposal_shape:
            raise ValueError(f"{name} must have shape {proposal_shape}")

    group_types = torch.zeros(
        (batch_size, group_slots), device=device, dtype=torch.long
    )
    member_counts = torch.zeros_like(group_types)
    group_mask = torch.zeros(
        (batch_size, group_slots), device=device, dtype=torch.bool
    )
    group_membership = torch.zeros(
        (batch_size, group_slots, proposals), device=device, dtype=torch.bool
    )
    group_weights = torch.zeros(
        (batch_size, group_slots), device=device, dtype=torch.float32
    )
    source_group_ids = torch.full(
        (batch_size, group_slots), -1, device=device, dtype=torch.long
    )
    proposal_keep = torch.zeros(
        (batch_size, proposals), device=device, dtype=torch.bool
    )
    proposal_types = torch.zeros(
        (batch_size, proposals), device=device, dtype=torch.long
    )
    known_counts = torch.zeros(batch_size, device=device, dtype=torch.long)

    valid_proposals = batch["proposal_mask"].bool()
    usable = valid_proposals & batch["group_mask"].bool()
    weights = batch["match_weight"].float()
    if not bool(torch.isfinite(weights[valid_proposals]).all()):
        raise ValueError("Active proposal match weights must be finite")
    if bool(((weights[valid_proposals] < 0.0) | (weights[valid_proposals] > 1.0)).any()):
        raise ValueError("Proposal match weights must be in [0, 1]")

    for batch_index in range(batch_size):
        ids = sorted(
            int(value)
            for value in torch.unique(
                batch["group_membership"][batch_index, usable[batch_index]]
            ).tolist()
            if int(value) >= 0
        )
        if len(ids) > group_slots:
            raise ValueError(
                f"Usable group count {len(ids)} exceeds group_slots={group_slots}; "
                "targets are never truncated"
            )
        known_counts[batch_index] = len(ids)
        for slot, source_id in enumerate(ids):
            members = usable[batch_index] & (
                batch["group_membership"][batch_index] == source_id
            )
            indices = torch.where(members)[0]
            if not len(indices):
                raise RuntimeError("A usable group has no supporting proposal")
            types = torch.unique(batch["group_type"][batch_index, indices])
            counts = torch.unique(batch["group_count"][batch_index, indices])
            if len(types) != 1 or len(counts) != 1:
                raise ValueError("One group has inconsistent type or member count")
            group_type = int(types[0])
            member_count = int(counts[0])
            support_count = len(indices)
            if group_type == HASHED_WEDGE:
                if member_count < 2 or support_count < 2 or support_count > member_count:
                    raise ValueError(
                        "Hashed target requires 2..member_count supporting proposals"
                    )
            else:
                fixed = FIXED_MEMBER_COUNTS.get(group_type)
                if fixed is None or member_count != fixed or support_count != fixed:
                    raise ValueError("Fixed group target has incomplete membership")
            group_types[batch_index, slot] = group_type
            member_counts[batch_index, slot] = member_count
            group_mask[batch_index, slot] = True
            group_membership[batch_index, slot, indices] = True
            group_weights[batch_index, slot] = weights[batch_index, indices].mean()
            source_group_ids[batch_index, slot] = source_id
            proposal_keep[batch_index, indices] = True
            proposal_types[batch_index, indices] = _required_proposal_type(group_type)

    return {
        "group_types": group_types,
        "group_membership": group_membership,
        "group_mask": group_mask,
        "member_counts": member_counts,
        "group_weights": group_weights,
        "proposal_keep": proposal_keep,
        "proposal_types": proposal_types,
        "source_group_ids": source_group_ids,
        "known_counts": known_counts,
    }


def oracle_keep_type_coverage(
    batch: dict[str, Any],
    targets: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Report matched, usable and correctly typed coverage without text."""

    primitives = batch.get("primitives")
    if not isinstance(primitives, list):
        raise ValueError("Coverage requires raw primitive sidecars")
    batch_size, proposals, device = _validate_proposal_batch(batch)
    if len(primitives) != batch_size:
        raise ValueError("Primitive sidecars and proposal batch sizes differ")
    expected = torch.tensor(
        [
            sum(
                primitive.get("kind") != "text_box"
                for primitive in sidecar.get("primitives", [])
            )
            for sidecar in primitives
        ],
        device=device,
        dtype=torch.long,
    )
    matched = batch["proposal_mask"].sum(dim=-1).long()
    kept = targets["proposal_keep"].sum(dim=-1).long()
    correct_type = (
        targets["proposal_keep"].bool()
        & (batch["proposal_type"].long() == targets["proposal_types"].long())
    ).sum(dim=-1).long()
    denominator = expected.clamp_min(1).float()
    return {
        "expected_physical": expected,
        "oracle_matched": matched,
        "oracle_keep": kept,
        "oracle_type_correct": correct_type,
        "oracle_matched_recall": matched.float() / denominator,
        "oracle_keep_recall": kept.float() / denominator,
        "oracle_type_recall": correct_type.float() / denominator,
        "ignored_matched": (matched - kept).clamp_min(0),
    }


def prepare_proposal_membership_pipeline(
    batch: dict[str, Any],
    flat_output: dict[str, torch.Tensor],
    query_reference_points: torch.Tensor,
    *,
    group_slots: int = DEFAULT_GROUP_SLOTS,
) -> dict[str, Any]:
    """Prepare exact model inputs, slot targets, counts and oracle coverage."""

    if "type_logits" not in flat_output:
        raise ValueError("Frozen flat output lacks type_logits")
    descriptors = build_proposal_descriptors(batch, query_reference_points)
    flat_type_logits = gather_flat_type_logits(
        flat_output["type_logits"],
        batch["proposal_query_index"],
        batch["proposal_mask"],
    )
    targets = aggregate_group_slot_targets(batch, group_slots=group_slots)
    model_inputs = {
        "proposal_features": batch["proposal_features"].detach(),
        "proposal_descriptors": descriptors,
        "flat_type_logits": flat_type_logits,
        "proposal_scores": batch["proposal_score"].detach(),
        "proposal_mask": batch["proposal_mask"].bool(),
    }
    return {
        "model_inputs": model_inputs,
        "targets": targets,
        "known_counts": targets["known_counts"],
        "oracle_coverage": oracle_keep_type_coverage(batch, targets),
    }


def _coordinate(value: float) -> int:
    if not math.isfinite(float(value)):
        raise ValueError("Proposal coordinate is not finite")
    return int(round(min(max(float(value), 0.0), 1.0) * 2047.0))


def _stroke(value: float) -> int:
    if not math.isfinite(float(value)):
        raise ValueError("Proposal stroke is not finite")
    return min(max(int(round(min(max(float(value), 0.0), 1.0) * 63.0)), 1), 63)


def _shared_stroke(values: Iterable[float]) -> int:
    widths = sorted(_stroke(value) for value in values)
    if not widths:
        raise ValueError("Carrier fitting requires at least one proposal stroke")
    middle = len(widths) // 2
    return (
        widths[middle]
        if len(widths) % 2
        else int(round((widths[middle - 1] + widths[middle]) * 0.5))
    )


def _proposal_member(
    group_type: int,
    geometry: torch.Tensor,
    stroke: int,
) -> LineMemberV02 | TriangleMemberV02 | QuadMemberV02:
    values = geometry.detach().float().cpu().tolist()
    if len(values) != 8:
        raise ValueError("Frozen proposal geometry must contain eight values")
    if group_type in LINE_GROUP_TYPES:
        return LineMemberV02(*(_coordinate(value) for value in values[:4]), stroke)
    if group_type == SOLID_TRIANGLE:
        return TriangleMemberV02(*(_coordinate(value) for value in values[:6]), stroke)
    if group_type == SOLID_QUAD:
        return QuadMemberV02(*(_coordinate(value) for value in values[:8]), stroke)
    raise ValueError(f"Unsupported proposal group type: {group_type}")


def _invalid_evaluator_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "query_index": int(row.get("query_index", -1)),
        "score": float(row.get("score", 0.0)),
        "type": int(row.get("type", 0)),
        "kind": str(row.get("kind", "unknown")),
        "member_count": int(row.get("member_count", 0)),
        "member_geometry": torch.empty((0, 8), dtype=torch.float32),
        "member_stroke": torch.empty((0,), dtype=torch.float32),
    }


def membership_rows_to_evaluator_rows(
    decoded_rows: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    proposal_batch: dict[str, Any],
    *,
    batch_index: int,
    max_members: int = DEFAULT_MAX_MEMBERS,
) -> EvaluatorRowsResult:
    """Fit typed carriers from frozen proposal geometry and expand evaluator rows."""

    batch_size, proposals, _ = _validate_proposal_batch(proposal_batch)
    if not 0 <= batch_index < batch_size:
        raise ValueError("batch_index is outside the proposal batch")
    if max_members < 2:
        raise ValueError("max_members must be at least two")
    output_rows = []
    failures = []
    for row in decoded_rows:
        source = dict(row)
        try:
            if not bool(source.get("valid_membership", False)):
                raise ValueError("Decoded group has invalid proposal membership")
            group_type = int(source["type"])
            kind = str(source["kind"])
            if not 1 <= group_type < len(PHYSICAL_GROUP_NAMES):
                raise ValueError("Decoded group type is outside 1..6")
            if PHYSICAL_GROUP_NAMES[group_type] != kind:
                raise ValueError("Decoded group kind/type mismatch")
            proposal_indices = [int(value) for value in source["proposal_indices"]]
            if len(set(proposal_indices)) != len(proposal_indices):
                raise ValueError("Decoded group repeats a proposal index")
            if not proposal_indices:
                raise ValueError("Decoded group has no supporting proposal")
            index_tensor = torch.tensor(
                proposal_indices,
                device=proposal_batch["proposal_mask"].device,
                dtype=torch.long,
            )
            if bool(((index_tensor < 0) | (index_tensor >= proposals)).any()):
                raise ValueError("Decoded group proposal index is out of range")
            if not bool(
                proposal_batch["proposal_mask"][batch_index, index_tensor].all()
            ):
                raise ValueError("Decoded group references padded proposals")
            member_count = int(source["member_count"])
            support_count = len(proposal_indices)
            if group_type == HASHED_WEDGE:
                if support_count < 2 or member_count < support_count:
                    raise ValueError("Hashed carrier needs 2..member_count supports")
            else:
                expected = FIXED_MEMBER_COUNTS.get(group_type)
                if expected is None or support_count != expected or member_count != expected:
                    raise ValueError("Fixed carrier support/member count is invalid")
            strokes = proposal_batch["proposal_stroke"][batch_index, index_tensor]
            shared_stroke = _shared_stroke(strokes.detach().cpu().tolist())
            members = tuple(
                _proposal_member(
                    group_type,
                    proposal_batch["proposal_geometry"][batch_index, proposal_index],
                    shared_stroke,
                )
                for proposal_index in proposal_indices
            )
            group = canonicalize_visual_svg_v0_2(
                ParsedVisualSVGSequenceV02(
                    (BondGroupRecordV02(kind, members),),
                    (),
                )
            ).bond_groups[0]
            encoding = encode_bond_group_carrier(group)
            carrier_row = {
                "query_index": int(source.get("query_index", -1)),
                "score": float(source.get("score", 0.0)),
                "type": group_type,
                "kind": kind,
                "member_count": member_count,
                "carrier_points": encoding.carrier_points,
                "positive_params": encoding.positive_params,
                "signed_params": encoding.signed_params,
                "group_stroke": encoding.group_stroke,
            }
            expanded = carrier_rows_to_bond_group_rows(
                [carrier_row],
                max_members=max_members,
            )
            output_rows.extend(expanded.rows)
            failures.extend(expanded.invalid_objects)
        except (KeyError, TypeError, ValueError) as error:
            output_rows.append(_invalid_evaluator_row(source))
            failures.append(
                {
                    "query_index": int(source.get("query_index", -1)),
                    "kind": str(source.get("kind", "unknown")),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    return EvaluatorRowsResult(tuple(output_rows), tuple(failures))


def bond_only_expected_v0_2_sequence(primitive_sidecar: dict[str, Any]) -> str:
    """Serialize the exact expected v0.2 bond groups without OCR text boxes."""

    parsed = visual_primitives_v0_1_to_visual_svg_v0_2(primitive_sidecar)
    return serialize_parsed_visual_svg_v0_2(
        ParsedVisualSVGSequenceV02(parsed.bond_groups, ())
    )


def bond_only_expected_v0_2_sequences(
    primitive_sidecars: Iterable[dict[str, Any]],
) -> list[str]:
    return [
        bond_only_expected_v0_2_sequence(sidecar)
        for sidecar in primitive_sidecars
    ]


def _f1(true_positive: int, predicted: int, expected: int) -> dict[str, float | int]:
    if predicted == 0:
        precision = 1.0 if expected == 0 else 0.0
    else:
        precision = true_positive / predicted
    if expected == 0:
        recall = 1.0 if predicted == 0 else 0.0
    else:
        recall = true_positive / expected
    score = (
        1.0
        if predicted == expected == 0
        else 2.0 * precision * recall / max(precision + recall, 1e-12)
    )
    return {
        "true_positive": true_positive,
        "predicted": predicted,
        "expected": expected,
        "precision": precision,
        "recall": recall,
        "f1": score,
    }


def _pair_set(groups: Iterable[frozenset[int]]) -> set[tuple[int, int]]:
    result = set()
    for group in groups:
        for left, right in itertools.combinations(sorted(group), 2):
            result.add((left, right))
    return result


def summarize_restricted_partitions(
    decoded_rows: list[list[dict[str, Any]]],
    targets: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Summarize exact restricted partitions plus group, typed-group and pair F1."""

    required = (
        "group_types",
        "group_membership",
        "group_mask",
        "member_counts",
    )
    missing = [name for name in required if name not in targets]
    if missing:
        raise ValueError(f"Restricted partition targets are missing: {missing}")
    batch = targets["group_types"].size(0)
    if len(decoded_rows) != batch:
        raise ValueError("Decoded rows and target batch sizes differ")
    group_tp = typed_tp = pair_tp = 0
    predicted_groups = expected_groups = predicted_pairs = expected_pairs = 0
    exact_samples = typed_exact_samples = invalid_rows = 0
    records = []
    for batch_index, rows in enumerate(decoded_rows):
        target_indices = torch.where(targets["group_mask"][batch_index].bool())[0]
        target_group_rows = []
        for target_index in target_indices.tolist():
            members = frozenset(
                torch.where(
                    targets["group_membership"][batch_index, target_index].bool()
                )[0].tolist()
            )
            target_group_rows.append(
                (
                    members,
                    int(targets["group_types"][batch_index, target_index]),
                    int(targets["member_counts"][batch_index, target_index]),
                )
            )
        predicted_group_rows = []
        sample_invalid = 0
        for row in rows:
            members = frozenset(int(value) for value in row.get("proposal_indices", []))
            valid = bool(row.get("valid_membership", False)) and bool(members)
            if not valid:
                sample_invalid += 1
            predicted_group_rows.append(
                (
                    members,
                    int(row.get("type", 0)),
                    int(row.get("member_count", 0)),
                    valid,
                )
            )
        target_sets = Counter(value[0] for value in target_group_rows)
        predicted_sets = Counter(
            value[0] for value in predicted_group_rows if value[3]
        )
        target_typed = Counter(target_group_rows)
        predicted_typed = Counter(
            value[:3] for value in predicted_group_rows if value[3]
        )
        sample_group_tp = sum((target_sets & predicted_sets).values())
        sample_typed_tp = sum((target_typed & predicted_typed).values())
        target_pair_values = _pair_set(target_sets.elements())
        predicted_pair_values = _pair_set(predicted_sets.elements())
        sample_pair_tp = len(target_pair_values & predicted_pair_values)
        sample_exact = sample_invalid == 0 and predicted_sets == target_sets
        sample_typed_exact = sample_invalid == 0 and predicted_typed == target_typed

        group_tp += sample_group_tp
        typed_tp += sample_typed_tp
        pair_tp += sample_pair_tp
        predicted_groups += len(rows)
        expected_groups += len(target_group_rows)
        predicted_pairs += len(predicted_pair_values)
        expected_pairs += len(target_pair_values)
        invalid_rows += sample_invalid
        exact_samples += int(sample_exact)
        typed_exact_samples += int(sample_typed_exact)
        records.append(
            {
                "partition_exact": sample_exact,
                "typed_partition_exact": sample_typed_exact,
                "invalid_rows": sample_invalid,
                "groups": _f1(sample_group_tp, len(rows), len(target_group_rows)),
                "typed_groups": _f1(
                    sample_typed_tp,
                    len(rows),
                    len(target_group_rows),
                ),
                "pairs": _f1(
                    sample_pair_tp,
                    len(predicted_pair_values),
                    len(target_pair_values),
                ),
            }
        )
    return {
        "samples": batch,
        "partition_exact": exact_samples / max(batch, 1),
        "typed_partition_exact": typed_exact_samples / max(batch, 1),
        "invalid_rows": invalid_rows,
        "groups": _f1(group_tp, predicted_groups, expected_groups),
        "typed_groups": _f1(typed_tp, predicted_groups, expected_groups),
        "pairs": _f1(pair_tp, predicted_pairs, expected_pairs),
        "records": records,
    }


__all__ = [
    "DEFAULT_GROUP_SLOTS",
    "DEFAULT_MAX_MEMBERS",
    "FLAT_TYPE_CLASSES",
    "PROPOSAL_DESCRIPTOR_DIM",
    "PROPOSAL_DESCRIPTOR_FIELDS",
    "EvaluatorRowsResult",
    "aggregate_group_slot_targets",
    "bond_only_expected_v0_2_sequence",
    "bond_only_expected_v0_2_sequences",
    "build_proposal_descriptors",
    "gather_flat_type_logits",
    "membership_rows_to_evaluator_rows",
    "oracle_keep_type_coverage",
    "prepare_proposal_membership_pipeline",
    "summarize_restricted_partitions",
]
