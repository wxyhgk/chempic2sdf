#!/usr/bin/env python3
"""Evaluate Visual SVG v0.2 bond-group set checkpoints."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
import torch
from torch.utils.data import DataLoader

from Data.sdf.visual_svg_sequence_v0_1 import (
    LineRecordV01,
    ParsedVisualSVGSequenceV01,
    QuadRecordV01,
    TextBoxRecordV01,
    TriangleRecordV01,
)
from Data.sdf.visual_svg_v0_2 import (
    GROUP_KINDS,
    BondGroupRecordV02,
    LineMemberV02,
    ParsedVisualSVGSequenceV02,
    QuadMemberV02,
    TextBoxRecordV02,
    TriangleMemberV02,
    VisualSVGV02Error,
    canonicalize_visual_svg_v0_2,
    flatten_visual_svg_v0_2_to_v0_1,
    parse_visual_svg_v0_2,
    serialize_parsed_visual_svg_v0_2,
)
from chemvector.audit_primitive_set_geometry import (
    LINE_TOLERANCES_Q,
    distribution,
    line_geometry_error_records,
    summarize_line_geometry,
)
from chemvector.bond_group_set import (
    GROUP_ID_TO_KIND,
    LINE_GROUP_TYPES,
    TEXT_BOX,
    BondGroupSetConfig,
    BondGroupSetPredictor,
    bond_group_set_loss,
    decode_bond_group_set,
)
from chemvector.bond_group_set_data import (
    BondGroupSetCollator,
    target_to_visual_svg_v0_2_json,
)
from chemvector.evaluate_primitive_set import (
    build_decode_groups,
    normalize_decode_modes,
    normalize_thresholds,
    validate_shards,
)
from chemvector.evaluate_visual_svg import (
    bond_iou,
    box_iou,
    line_distance,
    line_true_positives,
    optimal_matches,
    polygon_distance,
    polygon_true_positives,
    text_box_true_positives,
)
from chemvector.model import ChemVectorConfig
from chemvector.train_bond_group_set import (
    CHECKPOINT_SCHEMA_VERSION,
    SEQUENCE_SCHEMA_VERSION,
    autocast_context,
    targets_to_device,
)
from chemvector.train_primitive_set import file_sha256
from chemvector.visual_svg_data import VisualSVGWebDataset


EVALUATION_SCHEMA_VERSION = "bond-group-set-evaluation-0.1"
TYPE_NAMES = (*GROUP_KINDS, "text_box")
LINE_GROUP_KINDS = tuple(GROUP_ID_TO_KIND[type_id] for type_id in LINE_GROUP_TYPES)
DECODE_MODES = ("threshold", "count_head")
FIXED_MEMBER_COUNTS = {
    "single_line": 1,
    "parallel_2": 2,
    "parallel_3": 3,
    "solid_triangle": 1,
    "solid_quad": 1,
}


class BondGroupOverflowError(ValueError):
    """Raised when decode tensors cannot represent the requested members."""


@dataclass(frozen=True, slots=True)
class DecodedGroupSetResult:
    parsed: ParsedVisualSVGSequenceV02 | None
    flattened: ParsedVisualSVGSequenceV01
    sequence: str | None
    decoded_counts: dict[str, int]
    decoded_member_counts: dict[str, tuple[int, ...]]
    renderable_counts: dict[str, int]
    invalid_objects: tuple[dict[str, Any], ...]
    overflow_objects: tuple[dict[str, Any], ...]
    serialization_error: str | None

    @property
    def valid(self) -> bool:
        return not self.invalid_objects and self.serialization_error is None


class BondGroupSetEvaluationCollator(BondGroupSetCollator):
    """Retain exact v0.2 and original v0.1 targets for evaluation."""

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        batch = super().__call__(samples)
        batch["target_v0_2_sequences"] = [
            target_to_visual_svg_v0_2_json(self._target_payload(sample))
            for sample in samples
        ]
        batch["target_v0_1_sequences"] = [
            sample["target_sequence"] for sample in samples
        ]
        return batch


def load_bond_group_set_model(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[BondGroupSetPredictor, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "sequence_schema_version": SEQUENCE_SCHEMA_VERSION,
        "model_type": BondGroupSetPredictor.model_type,
        "uses_internal_test": False,
    }
    mismatches = {
        name: (checkpoint.get(name), value)
        for name, value in expected.items()
        if checkpoint.get(name) != value
    }
    if mismatches:
        raise ValueError(f"Bond Group Set checkpoint metadata differs: {mismatches}")
    vision_payload = checkpoint.get("vision_config")
    config_payload = checkpoint.get("bond_group_set_config")
    state = checkpoint.get("model")
    if not isinstance(vision_payload, dict) or not isinstance(config_payload, dict):
        raise ValueError("Bond Group Set checkpoint is missing model configurations")
    if not isinstance(state, dict):
        raise ValueError("Bond Group Set checkpoint is missing model state")
    model = BondGroupSetPredictor(
        ChemVectorConfig(**vision_payload),
        BondGroupSetConfig(**config_payload),
    )
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), checkpoint


def _clamped_unit(value: Any, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} is not finite: {result!r}")
    return min(max(result, 0.0), 1.0)


def _coordinate(value: Any, field: str) -> int:
    return int(round(_clamped_unit(value, field) * 2047.0))


def _stroke(value: Any) -> int:
    result = int(round(_clamped_unit(value, "stroke") * 63.0))
    return min(max(result, 1), 63)


def _positive_interval(center: Any, size: Any, axis: str) -> tuple[int, int]:
    center_value = _clamped_unit(center, f"text_{axis}_center")
    size_value = _clamped_unit(size, f"text_{axis}_size")
    if size_value <= 0.0:
        raise ValueError(f"text_{axis}_size must be positive")
    lower = int(round(max(center_value - size_value * 0.5, 0.0) * 2047.0))
    upper = int(round(min(center_value + size_value * 0.5, 1.0) * 2047.0))
    if upper <= lower:
        anchor = int(round(center_value * 2047.0))
        lower, upper = (
            (anchor, anchor + 1) if anchor < 2047 else (2046, 2047)
        )
    return lower, upper


def _tensor_list(value: Any, field: str) -> list[Any]:
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a tensor or list")
    return value


def _invalid_payload(
    row: dict[str, Any],
    error: Exception,
    *,
    overflow: bool,
) -> dict[str, Any]:
    return {
        "query_index": int(row.get("query_index", -1)),
        "score": float(row.get("score", 0.0)),
        "kind": str(row.get("kind", "unknown")),
        "member_count": int(row.get("member_count", 0)),
        "overflow": overflow,
        "error": f"{type(error).__name__}: {error}",
    }


def _member_from_prediction(
    kind: str,
    geometry: list[float],
    stroke: float,
) -> LineMemberV02 | TriangleMemberV02 | QuadMemberV02:
    point_count = {
        "line": 2,
        "triangle": 3,
        "quad": 4,
    }[kind]
    if len(geometry) != 8:
        raise ValueError("member geometry must contain eight normalized coordinates")
    values = [
        _coordinate(geometry[index], f"geometry[{index}]")
        for index in range(point_count * 2)
    ]
    width = _stroke(stroke)
    if kind == "line":
        return LineMemberV02(*values, width)
    if kind == "triangle":
        return TriangleMemberV02(*values, width)
    return QuadMemberV02(*values, width)


def _expected_member_kind(group_kind: str) -> str:
    if group_kind in LINE_GROUP_KINDS:
        return "line"
    if group_kind == "solid_triangle":
        return "triangle"
    if group_kind == "solid_quad":
        return "quad"
    raise ValueError(f"Unsupported group kind: {group_kind!r}")


def _validate_member_count(kind: str, count: int) -> None:
    fixed = FIXED_MEMBER_COUNTS.get(kind)
    if fixed is not None and count != fixed:
        raise ValueError(f"{kind} requires exactly {fixed} member(s), got {count}")
    if kind == "hashed_wedge" and count < 2:
        raise ValueError(f"hashed_wedge requires at least two members, got {count}")


def _flatten_renderable(
    groups: tuple[BondGroupRecordV02, ...],
    text_boxes: tuple[TextBoxRecordV02, ...],
) -> ParsedVisualSVGSequenceV01:
    if groups:
        return flatten_visual_svg_v0_2_to_v0_1(
            canonicalize_visual_svg_v0_2(
                ParsedVisualSVGSequenceV02(groups, text_boxes)
            )
        )
    return ParsedVisualSVGSequenceV01(
        (),
        (),
        (),
        tuple(TextBoxRecordV01(box.x0, box.y0, box.x1, box.y1) for box in text_boxes),
    )


def decoded_rows_to_visual_svg_v0_2(
    rows: Iterable[dict[str, Any]],
    *,
    max_members: int,
) -> DecodedGroupSetResult:
    """Quantize decoded groups without repairing type/cardinality mistakes."""

    decoded_counts = Counter({kind: 0 for kind in TYPE_NAMES})
    decoded_member_counts: dict[str, list[int]] = {kind: [] for kind in TYPE_NAMES}
    valid_groups: list[BondGroupRecordV02] = []
    valid_text: list[TextBoxRecordV02] = []
    invalid: list[dict[str, Any]] = []
    overflow: list[dict[str, Any]] = []
    for row in rows:
        kind = str(row.get("kind", "unknown"))
        if kind in decoded_counts:
            decoded_counts[kind] += 1
        count = int(row.get("member_count", 0))
        if kind in decoded_member_counts:
            decoded_member_counts[kind].append(count)
        try:
            if kind == "text_box":
                if count != 0:
                    raise ValueError(f"text_box requires member_count=0, got {count}")
                geometry = _tensor_list(row.get("text_geometry"), "text_geometry")
                if len(geometry) != 4:
                    raise ValueError("text geometry must contain center-x/y and width/height")
                x0, x1 = _positive_interval(geometry[0], geometry[2], "x")
                y0, y1 = _positive_interval(geometry[1], geometry[3], "y")
                valid_text.append(TextBoxRecordV02(x0, y0, x1, y1))
                continue
            if kind not in GROUP_KINDS:
                raise ValueError(f"Unsupported decoded group kind: {kind!r}")
            _validate_member_count(kind, count)
            geometry = _tensor_list(row.get("member_geometry"), "member_geometry")
            strokes = _tensor_list(row.get("member_stroke"), "member_stroke")
            if count > max_members or count > len(geometry) or count > len(strokes):
                raise BondGroupOverflowError(
                    f"decoded member_count={count} exceeds available capacity "
                    f"max_members={max_members}, geometry={len(geometry)}, strokes={len(strokes)}"
                )
            member_kind = _expected_member_kind(kind)
            members = tuple(
                _member_from_prediction(member_kind, geometry[index], strokes[index])
                for index in range(count)
            )
            candidate = canonicalize_visual_svg_v0_2(
                ParsedVisualSVGSequenceV02(
                    (BondGroupRecordV02(kind, members),),
                    (),
                )
            )
            valid_groups.append(candidate.bond_groups[0])
        except (TypeError, ValueError, VisualSVGV02Error) as error:
            is_overflow = isinstance(error, BondGroupOverflowError)
            payload = _invalid_payload(row, error, overflow=is_overflow)
            invalid.append(payload)
            if is_overflow:
                overflow.append(payload)

    parsed = None
    sequence = None
    serialization_error = None
    if valid_groups:
        try:
            parsed = canonicalize_visual_svg_v0_2(
                ParsedVisualSVGSequenceV02(tuple(valid_groups), tuple(valid_text))
            )
            sequence = serialize_parsed_visual_svg_v0_2(parsed)
            parsed = parse_visual_svg_v0_2(sequence)
        except (TypeError, ValueError, VisualSVGV02Error) as error:
            serialization_error = f"{type(error).__name__}: {error}"
    else:
        serialization_error = "VisualSVGV02Error: no renderable bond group"
    flattened = _flatten_renderable(tuple(valid_groups), tuple(valid_text))
    renderable_counts = Counter({kind: 0 for kind in TYPE_NAMES})
    for group in valid_groups:
        renderable_counts[group.kind] += 1
    renderable_counts["text_box"] = len(valid_text)
    return DecodedGroupSetResult(
        parsed=parsed,
        flattened=flattened,
        sequence=sequence,
        decoded_counts={kind: int(decoded_counts[kind]) for kind in TYPE_NAMES},
        decoded_member_counts={
            kind: tuple(decoded_member_counts[kind]) for kind in TYPE_NAMES
        },
        renderable_counts={kind: int(renderable_counts[kind]) for kind in TYPE_NAMES},
        invalid_objects=tuple(invalid),
        overflow_objects=tuple(overflow),
        serialization_error=serialization_error,
    )


def _member_to_v0_1(member):
    if isinstance(member, LineMemberV02):
        return LineRecordV01(
            member.x0,
            member.y0,
            member.x1,
            member.y1,
            member.stroke_width,
        )
    if isinstance(member, TriangleMemberV02):
        return TriangleRecordV01(
            member.x0,
            member.y0,
            member.x1,
            member.y1,
            member.x2,
            member.y2,
            member.stroke_width,
        )
    return QuadRecordV01(
        member.x0,
        member.y0,
        member.x1,
        member.y1,
        member.x2,
        member.y2,
        member.x3,
        member.y3,
        member.stroke_width,
    )


def _groups_of_kind(parsed: ParsedVisualSVGSequenceV02 | None, kind: str) -> tuple:
    if parsed is None:
        return ()
    if kind == "text_box":
        return parsed.text_boxes
    return tuple(group for group in parsed.bond_groups if group.kind == kind)


def _group_distance(left: BondGroupRecordV02, right: BondGroupRecordV02) -> float:
    if left.kind != right.kind or len(left.members) != len(right.members):
        return 1e9
    left_members = tuple(_member_to_v0_1(member) for member in left.members)
    right_members = tuple(_member_to_v0_1(member) for member in right.members)
    if left.kind in LINE_GROUP_KINDS:
        costs = np.asarray(
            [
                [line_distance(a, b) for b in right_members]
                for a in left_members
            ],
            dtype=np.float64,
        )
        rows, columns = linear_sum_assignment(costs)
        return max((float(costs[row, column]) for row, column in zip(rows, columns)), default=0.0)
    return polygon_distance(left_members[0], right_members[0])


def _group_true_positives(
    predicted: tuple,
    expected: tuple,
    *,
    tolerance: float,
) -> int:
    costs = np.asarray(
        [[_group_distance(left, right) for right in expected] for left in predicted],
        dtype=np.float64,
    )
    return optimal_matches(costs, tolerance)


def _exact_true_positives(predicted: tuple, expected: tuple) -> int:
    return sum((Counter(predicted) & Counter(expected)).values())


def _member_count_true_positives(
    predicted_counts: tuple[int, ...],
    expected_counts: tuple[int, ...],
) -> int:
    return sum((Counter(predicted_counts) & Counter(expected_counts)).values())


def _physical_decoded_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts = {"line": 0, "triangle": 0, "quad": 0, "text_box": 0}
    for row in rows:
        kind = str(row.get("kind", "unknown"))
        count = max(int(row.get("member_count", 0)), 0)
        if kind in LINE_GROUP_KINDS:
            counts["line"] += count
        elif kind == "solid_triangle":
            counts["triangle"] += count
        elif kind == "solid_quad":
            counts["quad"] += count
        elif kind == "text_box":
            counts["text_box"] += 1
    return counts


def _physical_counts(parsed: ParsedVisualSVGSequenceV01) -> dict[str, int]:
    return {
        "line": len(parsed.lines),
        "triangle": len(parsed.triangles),
        "quad": len(parsed.quads),
        "text_box": len(parsed.text_boxes),
    }


def _only_polygon_type(
    parsed: ParsedVisualSVGSequenceV01,
    kind: str,
) -> ParsedVisualSVGSequenceV01:
    return ParsedVisualSVGSequenceV01(
        (),
        parsed.triangles if kind == "triangle" else (),
        parsed.quads if kind == "quad" else (),
        (),
    )


def _safe_bond_iou(
    predicted: ParsedVisualSVGSequenceV01,
    target: Image.Image,
) -> tuple[float, str | None]:
    if target.width != target.height:
        return 0.0, "ValueError: bond target must be square"
    target_mask = np.asarray(target.convert("L")) > 0
    if not bool(target_mask.any()) or bool(target_mask.all()):
        return 0.0, "ValueError: bond target mask must contain foreground and background"
    if not (predicted.lines or predicted.triangles or predicted.quads):
        return 0.0, None
    try:
        return bond_iou(predicted, target), None
    except Exception as error:
        return 0.0, f"{type(error).__name__}: {error}"


def _group_aligned_line_metrics(
    predicted: ParsedVisualSVGSequenceV02 | None,
    expected: ParsedVisualSVGSequenceV02,
) -> tuple[int, dict[float, int], list[dict[str, float]]]:
    expected_lines = sum(
        len(group.members)
        for group in expected.bond_groups
        if group.kind in LINE_GROUP_KINDS
    )
    hits = {tolerance: 0 for tolerance in LINE_TOLERANCES_Q}
    errors: list[dict[str, float]] = []
    for kind in LINE_GROUP_KINDS:
        predicted_groups = _groups_of_kind(predicted, kind)
        expected_groups = _groups_of_kind(expected, kind)
        costs = np.asarray(
            [
                [_group_distance(left, right) for right in expected_groups]
                for left in predicted_groups
            ],
            dtype=np.float64,
        )
        if not costs.size:
            continue
        rows, columns = linear_sum_assignment(costs)
        for row, column in zip(rows, columns):
            left = ParsedVisualSVGSequenceV01(
                tuple(_member_to_v0_1(member) for member in predicted_groups[row].members),
                (),
                (),
                (),
            )
            right = ParsedVisualSVGSequenceV01(
                tuple(_member_to_v0_1(member) for member in expected_groups[column].members),
                (),
                (),
                (),
            )
            errors.extend(line_geometry_error_records(left, right))
            for tolerance in LINE_TOLERANCES_Q:
                hits[tolerance] += line_true_positives(left, right, tolerance)
    return expected_lines, hits, errors


def evaluate_decoded_sample(
    *,
    sample_key: str,
    decode_mode: str,
    threshold: float | None,
    decoded_rows: list[dict[str, Any]],
    expected_v0_2_sequence: str,
    bond_layer: Image.Image,
    line_tolerance: float,
    text_box_iou: float,
    max_members: int,
) -> dict[str, Any]:
    expected = parse_visual_svg_v0_2(expected_v0_2_sequence)
    expected_flat = flatten_visual_svg_v0_2_to_v0_1(expected)
    converted = decoded_rows_to_visual_svg_v0_2(
        decoded_rows,
        max_members=max_members,
    )
    predicted = converted.parsed
    predicted_flat = converted.flattened
    expected_counts = {
        kind: len(_groups_of_kind(expected, kind)) for kind in TYPE_NAMES
    }
    group_tolerance_tp = {}
    group_exact_tp = {}
    member_count_tp = {}
    member_count_sample_exact = {}
    for kind in TYPE_NAMES:
        predicted_objects = _groups_of_kind(predicted, kind)
        expected_objects = _groups_of_kind(expected, kind)
        if kind == "text_box":
            group_tolerance_tp[kind] = text_box_true_positives(
                predicted_flat,
                expected_flat,
                text_box_iou,
            )
        else:
            group_tolerance_tp[kind] = _group_true_positives(
                predicted_objects,
                expected_objects,
                tolerance=line_tolerance,
            )
        group_exact_tp[kind] = _exact_true_positives(
            predicted_objects,
            expected_objects,
        )
        expected_member_counts = tuple(
            0 if kind == "text_box" else len(value.members)
            for value in expected_objects
        )
        predicted_member_counts = converted.decoded_member_counts[kind]
        member_count_tp[kind] = _member_count_true_positives(
            predicted_member_counts,
            expected_member_counts,
        )
        member_count_sample_exact[kind] = (
            sorted(predicted_member_counts) == sorted(expected_member_counts)
        )

    decoded_physical = _physical_decoded_counts(decoded_rows)
    expected_physical = _physical_counts(expected_flat)
    physical_tp = {
        "line": line_true_positives(predicted_flat, expected_flat, line_tolerance),
        "triangle": polygon_true_positives(
            _only_polygon_type(predicted_flat, "triangle"),
            _only_polygon_type(expected_flat, "triangle"),
            line_tolerance,
        ),
        "quad": polygon_true_positives(
            _only_polygon_type(predicted_flat, "quad"),
            _only_polygon_type(expected_flat, "quad"),
            line_tolerance,
        ),
        "text_box": text_box_true_positives(
            predicted_flat,
            expected_flat,
            text_box_iou,
        ),
    }
    line_hits = {
        tolerance: line_true_positives(predicted_flat, expected_flat, tolerance)
        for tolerance in LINE_TOLERANCES_Q
    }
    line_errors = line_geometry_error_records(predicted_flat, expected_flat)
    grouped_expected, grouped_hits, grouped_errors = _group_aligned_line_metrics(
        predicted,
        expected,
    )
    visual_iou, render_error = _safe_bond_iou(predicted_flat, bond_layer)
    return {
        "sample_key": sample_key,
        "decode_mode": decode_mode,
        "threshold": threshold,
        "group_set_exact": converted.valid and predicted == expected,
        "flatten_exact": converted.valid and predicted_flat == expected_flat,
        "expected_v0_2_sequence": expected_v0_2_sequence,
        "predicted_v0_2_sequence": converted.sequence,
        "expected_counts": expected_counts,
        "decoded_counts": converted.decoded_counts,
        "renderable_counts": converted.renderable_counts,
        "group_tolerance_true_positives": group_tolerance_tp,
        "group_exact_true_positives": group_exact_tp,
        "member_count_true_positives": member_count_tp,
        "member_count_sample_exact": member_count_sample_exact,
        "physical": {
            "decoded_counts": decoded_physical,
            "expected_counts": expected_physical,
            "true_positives": physical_tp,
        },
        "line_geometry": {
            "expected": expected_physical["line"],
            "hits": {f"{int(key)}q": value for key, value in line_hits.items()},
            "errors": line_errors,
        },
        "group_aligned_line_geometry": {
            "expected": grouped_expected,
            "hits": {f"{int(key)}q": value for key, value in grouped_hits.items()},
            "errors": grouped_errors,
        },
        "invalid_objects": list(converted.invalid_objects),
        "overflow_objects": list(converted.overflow_objects),
        "invalid_member_count": sum(
            max(int(row.get("member_count", 0)), 0)
            for row in converted.invalid_objects
        ),
        "serialization_error": converted.serialization_error,
        "render_error": render_error,
        "conditional_valid": converted.valid and render_error is None,
        "bond_iou": visual_iou,
    }


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / max(float(denominator), 1.0)


def _f1(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / max(precision + recall, 1e-8)


def _detection_metrics(tp: int, predicted: int, expected: int) -> dict[str, Any]:
    precision = _ratio(tp, predicted)
    recall = _ratio(tp, expected)
    return {
        "true_positive": tp,
        "predicted": predicted,
        "expected": expected,
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
    }


def _summarize_line_rows(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    expected = sum(row[field]["expected"] for row in rows)
    hits = {
        tolerance: sum(row[field]["hits"][f"{int(tolerance)}q"] for row in rows)
        for tolerance in LINE_TOLERANCES_Q
    }
    errors = [error for row in rows for error in row[field]["errors"]]
    return summarize_line_geometry(
        expected_lines=expected,
        tolerance_hits=hits,
        errors=errors,
    )


def summarize_prediction_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    samples = len(rows)
    per_type = {}
    for kind in TYPE_NAMES:
        predicted = sum(row["decoded_counts"][kind] for row in rows)
        expected = sum(row["expected_counts"][kind] for row in rows)
        tolerance_tp = sum(
            row["group_tolerance_true_positives"][kind] for row in rows
        )
        exact_tp = sum(row["group_exact_true_positives"][kind] for row in rows)
        member_tp = sum(row["member_count_true_positives"][kind] for row in rows)
        count_errors = [
            abs(row["decoded_counts"][kind] - row["expected_counts"][kind])
            for row in rows
        ]
        per_type[kind] = {
            "group_at_tolerance": _detection_metrics(
                tolerance_tp,
                predicted,
                expected,
            ),
            "group_exact": _detection_metrics(exact_tp, predicted, expected),
            "count_exact": _ratio(sum(error == 0 for error in count_errors), samples),
            "count_mae": _ratio(sum(count_errors), samples),
            "member_count": {
                **_detection_metrics(member_tp, predicted, expected),
                "sample_exact": _ratio(
                    sum(row["member_count_sample_exact"][kind] for row in rows),
                    samples,
                ),
            },
        }

    object_errors = [
        abs(sum(row["decoded_counts"].values()) - sum(row["expected_counts"].values()))
        for row in rows
    ]
    physical = {}
    for kind in ("line", "triangle", "quad", "text_box"):
        predicted = sum(row["physical"]["decoded_counts"][kind] for row in rows)
        expected = sum(row["physical"]["expected_counts"][kind] for row in rows)
        true_positive = sum(row["physical"]["true_positives"][kind] for row in rows)
        count_errors = [
            abs(
                row["physical"]["decoded_counts"][kind]
                - row["physical"]["expected_counts"][kind]
            )
            for row in rows
        ]
        physical[kind] = {
            **_detection_metrics(true_positive, predicted, expected),
            "count_exact": _ratio(sum(error == 0 for error in count_errors), samples),
            "count_mae": _ratio(sum(count_errors), samples),
        }
    polygon_tp = physical["triangle"]["true_positive"] + physical["quad"]["true_positive"]
    polygon_predicted = physical["triangle"]["predicted"] + physical["quad"]["predicted"]
    polygon_expected = physical["triangle"]["expected"] + physical["quad"]["expected"]
    ious = [float(row["bond_iou"]) for row in rows]
    return {
        "samples": samples,
        "object_count": {
            "exact": _ratio(sum(error == 0 for error in object_errors), samples),
            "mae": _ratio(sum(object_errors), samples),
        },
        "type_count_set_exact": _ratio(
            sum(
                all(
                    row["decoded_counts"][kind] == row["expected_counts"][kind]
                    for kind in TYPE_NAMES
                )
                for row in rows
            ),
            samples,
        ),
        "member_count_set_exact": _ratio(
            sum(
                all(row["member_count_sample_exact"][kind] for kind in TYPE_NAMES)
                for row in rows
            ),
            samples,
        ),
        "group_set_exact": _ratio(sum(row["group_set_exact"] for row in rows), samples),
        "flatten_exact": _ratio(sum(row["flatten_exact"] for row in rows), samples),
        "types": per_type,
        "member_line_geometry": _summarize_line_rows(rows, "line_geometry"),
        "group_aligned_member_line_geometry": _summarize_line_rows(
            rows,
            "group_aligned_line_geometry",
        ),
        "flattened_v0_1": {
            "classes": physical,
            "polygon": _detection_metrics(
                polygon_tp,
                polygon_predicted,
                polygon_expected,
            ),
        },
        "bond_iou": {
            "mean": _ratio(sum(ious), samples),
            "min": min(ious) if ious else 0.0,
        },
        "invalid": {
            "objects": sum(len(row["invalid_objects"]) for row in rows),
            "members": sum(row["invalid_member_count"] for row in rows),
            "overflow_objects": sum(len(row["overflow_objects"]) for row in rows),
            "samples": sum(bool(row["invalid_objects"]) for row in rows),
            "serialization_errors": sum(
                row["serialization_error"] is not None for row in rows
            ),
            "render_errors": sum(row["render_error"] is not None for row in rows),
        },
    }


def _ensure_not_internal(meta: dict[str, Any], sample_key: str) -> None:
    serialized = json.dumps(meta, ensure_ascii=True, sort_keys=True).lower()
    if "internal_test" in sample_key.lower() or "internal_test" in serialized:
        raise RuntimeError(f"Refusing sealed internal_test sample: {sample_key}")


def _white_comparison(
    real: dict[str, torch.Tensor],
    white: dict[str, torch.Tensor],
) -> dict[str, Any]:
    comparisons = {}
    member_delta = (
        real["member_geometry"].float() - white["member_geometry"].float()
    ).abs() * 2047.0
    comparisons["member_geometry_all_slots_abs_q"] = distribution(
        member_delta.reshape(-1).cpu().tolist()
    )
    real_types = real["type_logits"].argmax(dim=-1)
    real_member_counts = real["member_count_logits"].argmax(dim=-1)
    member_indices = torch.arange(member_delta.size(2), device=member_delta.device)
    active = torch.zeros_like(real_member_counts[..., None], dtype=torch.bool).expand(
        -1,
        -1,
        member_delta.size(2),
    ).clone()
    bond_type = (real_types != 0) & (real_types != TEXT_BOX)
    active = bond_type[..., None] & (member_indices.view(1, 1, -1) < real_member_counts[..., None])
    if bool(active.any()):
        comparisons["member_geometry_predicted_active_abs_q"] = distribution(
            member_delta[active].reshape(-1).cpu().tolist()
        )
    else:
        comparisons["member_geometry_predicted_active_abs_q"] = distribution([])
    text_delta = (real["text_geometry"].float() - white["text_geometry"].float()).abs() * 2047.0
    comparisons["text_geometry_abs_q"] = distribution(
        text_delta.reshape(-1).cpu().tolist()
    )
    comparisons["type_argmax_same_rate"] = float(
        (real_types == white["type_logits"].argmax(dim=-1)).float().mean()
    )
    comparisons["member_count_argmax_same_rate"] = float(
        (
            real_member_counts
            == white["member_count_logits"].argmax(dim=-1)
        ).float().mean()
    )
    comparisons["total_count_argmax_same_rate"] = float(
        (
            real["count_logits"].argmax(dim=-1)
            == white["count_logits"].argmax(dim=-1)
        ).float().mean()
    )
    return comparisons


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--shard", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold", action="append", type=float)
    parser.add_argument(
        "--decode-mode",
        action="append",
        choices=DECODE_MODES,
        help="Repeat to select modes; default evaluates threshold and count_head",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--line-tolerance", type=float, default=8.0)
    parser.add_argument("--text-box-iou", type=float, default=0.5)
    parser.add_argument("--compare-white", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    decode_modes = (
        normalize_decode_modes(args.decode_mode)
        if args.decode_mode is not None
        else DECODE_MODES
    )
    thresholds = normalize_thresholds(args.threshold)
    if args.threshold is not None and "threshold" not in decode_modes:
        raise ValueError("--threshold is only meaningful with threshold decoding")
    decode_groups = build_decode_groups(decode_modes, thresholds)
    shards = validate_shards(args.shard)
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.batch_size < 1 or args.workers < 0 or args.image_size < 1:
        raise ValueError("Batch/image size must be positive and workers nonnegative")
    if args.line_tolerance < 0:
        raise ValueError("--line-tolerance must be nonnegative")
    if not 0.0 <= args.text_box_iou <= 1.0:
        raise ValueError("--text-box-iou must be in [0, 1]")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model, checkpoint = load_bond_group_set_model(args.checkpoint, device)
    loader = DataLoader(
        VisualSVGWebDataset(list(shards), limit=args.limit),
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=BondGroupSetEvaluationCollator(
            image_size=args.image_size,
            query_count=model.config.queries,
            max_members=model.config.max_members,
        ),
    )
    rows_by_group = {group: [] for group in decode_groups}
    white_rows = []
    real_loss_total = 0.0
    white_loss_total = 0.0
    samples_seen = 0
    with torch.inference_mode():
        for batch in loader:
            for key, meta in zip(batch["keys"], batch["meta"]):
                _ensure_not_internal(meta, key)
            pixels = batch["pixel_values"].to(device, non_blocking=True)
            targets = targets_to_device(batch, device)
            with autocast_context(device):
                output = model(pixels)
                if args.compare_white:
                    white_output = model(torch.ones_like(pixels))
                    real_losses = bond_group_set_loss(
                        output,
                        targets,
                        image_size=args.image_size,
                    )
                    white_losses = bond_group_set_loss(
                        white_output,
                        targets,
                        image_size=args.image_size,
                    )
            if args.compare_white:
                white_rows.append(_white_comparison(output, white_output))
                real_loss_total += float(real_losses["loss"]) * pixels.size(0)
                white_loss_total += float(white_losses["loss"]) * pixels.size(0)
            for decode_mode, threshold in decode_groups:
                decoded = decode_bond_group_set(
                    output,
                    threshold=threshold if threshold is not None else 0.5,
                    mode=decode_mode,
                )
                for index, decoded_rows in enumerate(decoded):
                    rows_by_group[(decode_mode, threshold)].append(
                        evaluate_decoded_sample(
                            sample_key=batch["keys"][index],
                            decode_mode=decode_mode,
                            threshold=threshold,
                            decoded_rows=decoded_rows,
                            expected_v0_2_sequence=batch["target_v0_2_sequences"][index],
                            bond_layer=batch["bond_layers"][index],
                            line_tolerance=args.line_tolerance,
                            text_box_iou=args.text_box_iou,
                            max_members=model.config.max_members,
                        )
                    )
            samples_seen += pixels.size(0)
            if samples_seen % 10 == 0:
                print(f"evaluated={samples_seen}", flush=True)
    if not samples_seen:
        raise RuntimeError("Bond Group Set evaluator read no samples")

    white_summary = None
    if args.compare_white:
        fields = white_rows[0].keys() if white_rows else ()
        white_summary = {}
        for field in fields:
            values = [row[field] for row in white_rows]
            if isinstance(values[0], dict):
                flattened = []
                for value in values:
                    # The per-batch distributions cannot be merged from quantiles;
                    # keep batch summaries and expose a count-weighted mean only.
                    flattened.append(value)
                white_summary[field] = {"batches": flattened}
            else:
                white_summary[field] = float(sum(values) / len(values))
        mean_real_loss = real_loss_total / samples_seen
        mean_white_loss = white_loss_total / samples_seen
        white_summary["real_loss"] = mean_real_loss
        white_summary["white_loss"] = mean_white_loss
        white_summary["white_image_loss_ratio"] = mean_white_loss / max(
            mean_real_loss,
            1e-8,
        )

    results = []
    for decode_mode, threshold in decode_groups:
        rows = rows_by_group[(decode_mode, threshold)]
        results.append(
            {
                "decode_mode": decode_mode,
                "threshold": threshold,
                **summarize_prediction_rows(rows),
            }
        )
    metrics = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": file_sha256(Path(args.checkpoint)),
        "checkpoint_global_step": int(checkpoint.get("global_step", -1)),
        "model_type": model.model_type,
        "config": {
            "queries": model.config.queries,
            "max_members": model.config.max_members,
            "line_tolerance_q": args.line_tolerance,
            "text_box_iou": args.text_box_iou,
        },
        "shards": list(shards),
        "samples": samples_seen,
        "decode_results": results,
        "white_image_comparison": white_summary,
        "uses_internal_test": False,
    }
    (output_dir / "evaluation_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for group in decode_groups:
            for row in rows_by_group[group]:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
