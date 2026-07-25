#!/usr/bin/env python3
"""Read-only oracle audit from flat physical proposals to typed carriers.

The audit deliberately grants the assembler the GT group count, group type and
physical-member partition.  Geometry must still come from the flat predictor:
one-to-one primitive matches outside the requested q tolerance are discarded,
and missing members are never replaced with GT coordinates.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from Data.sdf.visual_svg_sequence_v0_1 import (
    LineRecordV01,
    QuadRecordV01,
    TextBoxRecordV01,
    TriangleRecordV01,
)
from Data.sdf.visual_svg_v0_2 import (
    BondGroupRecordV02,
    LineMemberV02,
    ParsedVisualSVGSequenceV02,
    QuadMemberV02,
    TextBoxRecordV02,
    TriangleMemberV02,
    canonicalize_visual_svg_v0_2,
    serialize_parsed_visual_svg_v0_2,
    visual_primitives_v0_1_to_visual_svg_v0_2,
)
from chemvector.bond_group_carrier_geometry import encode_bond_group_carrier
from chemvector.bond_group_set import GROUP_KIND_TO_ID
from chemvector.evaluate_bond_group_carrier_set import (
    carrier_rows_to_bond_group_rows,
)
from chemvector.evaluate_bond_group_set import (
    TYPE_NAMES,
    evaluate_decoded_sample,
    summarize_prediction_rows,
)
from chemvector.evaluate_primitive_set import validate_shards
from chemvector.evaluate_visual_svg import box_iou, line_distance, polygon_distance
from chemvector.train_primitive_set import file_sha256
from chemvector.visual_svg_data import VisualSVGWebDataset


AUDIT_SCHEMA_VERSION = "flat-proposal-carrier-oracle-audit-0.1"
PHYSICAL_KINDS = ("line", "triangle", "quad", "text_box")
DEFAULT_TOLERANCES_Q = (8.0, 32.0)


PhysicalMember = LineMemberV02 | TriangleMemberV02 | QuadMemberV02
PredictionObject = PhysicalMember | TextBoxRecordV02
DistanceFunction = Callable[[PredictionObject, PredictionObject], float]


@dataclass(frozen=True, slots=True)
class TargetMemberRef:
    group_index: int | None
    member_index: int
    value: PredictionObject


@dataclass(frozen=True, slots=True)
class OracleAssembly:
    rows: tuple[dict[str, Any], ...]
    statistics: dict[str, Any]


def _points(payload: dict[str, Any], expected: int) -> tuple[tuple[int, int], ...]:
    values = payload.get("quantized_points")
    if not isinstance(values, list) or len(values) != expected:
        raise ValueError(f"prediction requires {expected} quantized points")
    result = []
    for point in values:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError("prediction point must contain x and y")
        x, y = point
        if isinstance(x, bool) or isinstance(y, bool):
            raise ValueError("prediction coordinates must be integers")
        x, y = int(x), int(y)
        if not 0 <= x <= 2047 or not 0 <= y <= 2047:
            raise ValueError("prediction coordinate is outside q2047")
        result.append((x, y))
    return tuple(result)


def _stroke(payload: dict[str, Any]) -> int:
    value = payload.get("stroke_width_q")
    if isinstance(value, bool):
        raise ValueError("prediction stroke width must be numeric")
    value = int(value)
    if not 1 <= value <= 63:
        raise ValueError("prediction stroke width is outside 1..63")
    return value


def _canonical_predicted_member(payload: dict[str, Any]) -> PredictionObject:
    kind = str(payload.get("kind", ""))
    if kind == "text_box":
        points = _points(payload, 2)
        x0, y0 = points[0]
        x1, y1 = points[1]
        if x1 <= x0 or y1 <= y0:
            raise ValueError("predicted text box must have positive area")
        return TextBoxRecordV02(x0, y0, x1, y1)
    if kind == "line":
        points = _points(payload, 2)
        member: PhysicalMember = LineMemberV02(
            *points[0], *points[1], _stroke(payload)
        )
        group_kind = "single_line"
    elif kind == "triangle":
        points = _points(payload, 3)
        member = TriangleMemberV02(
            *(coordinate for point in points for coordinate in point),
            _stroke(payload),
        )
        group_kind = "solid_triangle"
    elif kind == "quad":
        points = _points(payload, 4)
        member = QuadMemberV02(
            *(coordinate for point in points for coordinate in point),
            _stroke(payload),
        )
        group_kind = "solid_quad"
    else:
        raise ValueError(f"unsupported flat prediction kind: {kind!r}")
    parsed = canonicalize_visual_svg_v0_2(
        ParsedVisualSVGSequenceV02(
            (BondGroupRecordV02(group_kind, (member,)),),
            (),
        )
    )
    return parsed.bond_groups[0].members[0]


def prediction_objects_by_kind(
    values: list[dict[str, Any]],
) -> tuple[dict[str, list[PredictionObject]], list[dict[str, str]]]:
    """Parse the saved flat evaluator objects and expose rejected entries."""

    result: dict[str, list[PredictionObject]] = {kind: [] for kind in PHYSICAL_KINDS}
    invalid = []
    for index, payload in enumerate(values):
        try:
            member = _canonical_predicted_member(payload)
            if isinstance(member, LineMemberV02):
                kind = "line"
            elif isinstance(member, TriangleMemberV02):
                kind = "triangle"
            elif isinstance(member, QuadMemberV02):
                kind = "quad"
            else:
                kind = "text_box"
            result[kind].append(member)
        except (TypeError, ValueError) as error:
            invalid.append(
                {
                    "index": str(index),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    return result, invalid


def target_members_by_kind(
    expected: ParsedVisualSVGSequenceV02,
) -> dict[str, list[TargetMemberRef]]:
    result: dict[str, list[TargetMemberRef]] = {kind: [] for kind in PHYSICAL_KINDS}
    for group_index, group in enumerate(expected.bond_groups):
        for member_index, member in enumerate(group.members):
            if isinstance(member, LineMemberV02):
                kind = "line"
            elif isinstance(member, TriangleMemberV02):
                kind = "triangle"
            else:
                kind = "quad"
            result[kind].append(TargetMemberRef(group_index, member_index, member))
    for text_index, text_box in enumerate(expected.text_boxes):
        result["text_box"].append(TargetMemberRef(None, text_index, text_box))
    return result


def _as_v0_1(value: PredictionObject):
    if isinstance(value, LineMemberV02):
        return LineRecordV01(
            value.x0, value.y0, value.x1, value.y1, value.stroke_width
        )
    if isinstance(value, TriangleMemberV02):
        return TriangleRecordV01(
            value.x0,
            value.y0,
            value.x1,
            value.y1,
            value.x2,
            value.y2,
            value.stroke_width,
        )
    if isinstance(value, QuadMemberV02):
        return QuadRecordV01(
            value.x0,
            value.y0,
            value.x1,
            value.y1,
            value.x2,
            value.y2,
            value.x3,
            value.y3,
            value.stroke_width,
        )
    return TextBoxRecordV01(value.x0, value.y0, value.x1, value.y1)


def _distance(kind: str, left: PredictionObject, right: PredictionObject) -> float:
    if kind == "line":
        return line_distance(_as_v0_1(left), _as_v0_1(right))
    if kind in {"triangle", "quad"}:
        return polygon_distance(_as_v0_1(left), _as_v0_1(right))
    return 1.0 - box_iou(_as_v0_1(left), _as_v0_1(right))


def oracle_assignments(
    predicted: dict[str, list[PredictionObject]],
    expected: dict[str, list[TargetMemberRef]],
    *,
    tolerance_q: float,
    text_box_iou: float,
) -> tuple[dict[tuple[int | None, int], PredictionObject], dict[str, dict[str, int]]]:
    """Globally associate same-type proposals, then enforce hard tolerances."""

    assignments: dict[tuple[int | None, int], PredictionObject] = {}
    stats: dict[str, dict[str, int]] = {}
    for kind in PHYSICAL_KINDS:
        candidates = predicted[kind]
        targets = expected[kind]
        accepted = 0
        if candidates and targets:
            costs = np.asarray(
                [
                    [_distance(kind, candidate, target.value) for target in targets]
                    for candidate in candidates
                ],
                dtype=np.float64,
            )
            rows, columns = linear_sum_assignment(costs)
            maximum = 1.0 - text_box_iou if kind == "text_box" else tolerance_q
            for row, column in zip(rows, columns):
                if float(costs[row, column]) > maximum:
                    continue
                target = targets[int(column)]
                assignments[(target.group_index, target.member_index)] = candidates[
                    int(row)
                ]
                accepted += 1
        stats[kind] = {
            "predicted": len(candidates),
            "expected": len(targets),
            "accepted": accepted,
        }
    return assignments, stats


def _shared_stroke(members: list[PhysicalMember]) -> int:
    widths = sorted(int(member.stroke_width) for member in members)
    middle = len(widths) // 2
    value = (
        widths[middle]
        if len(widths) % 2
        else int(round((widths[middle - 1] + widths[middle]) * 0.5))
    )
    return min(max(value, 1), 63)


def _with_stroke(member: PhysicalMember, stroke: int) -> PhysicalMember:
    values = [
        getattr(member, name)
        for name in member.__dataclass_fields__
        if name != "stroke_width"
    ]
    return type(member)(*values, stroke)


def _missing_row(kind: str, member_count: int, query_index: int) -> dict[str, Any]:
    base = {
        "query_index": query_index,
        "score": 1.0,
        "type": GROUP_KIND_TO_ID[kind],
        "kind": kind,
        "member_count": member_count,
    }
    if kind == "text_box":
        base["text_geometry"] = []
    else:
        base["member_geometry"] = []
        base["member_stroke"] = []
    return base


def _carrier_row(
    group: BondGroupRecordV02,
    selected: list[PhysicalMember],
    query_index: int,
) -> dict[str, Any]:
    stroke = _shared_stroke(selected)
    fitted = BondGroupRecordV02(
        group.kind,
        tuple(_with_stroke(member, stroke) for member in selected),
    )
    encoded = encode_bond_group_carrier(fitted)
    return {
        "query_index": query_index,
        "score": 1.0,
        "type": int(encoded.kind_id),
        "kind": group.kind,
        # A hashed envelope is expanded at the oracle GT cardinality.  No GT
        # member coordinate enters the fit.
        "member_count": (
            len(group.members)
            if group.kind == "hashed_wedge"
            else int(encoded.member_count)
        ),
        "carrier_points": encoded.carrier_points,
        "positive_params": encoded.positive_params,
        "signed_params": encoded.signed_params,
        "group_stroke": encoded.group_stroke,
    }


def assemble_oracle_carriers(
    expected: ParsedVisualSVGSequenceV02,
    predicted_objects: list[dict[str, Any]],
    *,
    tolerance_q: float,
    text_box_iou: float = 0.5,
    max_members: int = 143,
) -> OracleAssembly:
    """Build known-type group slots using only tolerance-approved proposals."""

    predicted, invalid_predictions = prediction_objects_by_kind(predicted_objects)
    targets = target_members_by_kind(expected)
    assignments, member_stats = oracle_assignments(
        predicted,
        targets,
        tolerance_q=tolerance_q,
        text_box_iou=text_box_iou,
    )
    rows: list[dict[str, Any]] = []
    group_stats = {
        kind: Counter(expected=0, eligible=0, emitted=0, fit_failed=0)
        for kind in TYPE_NAMES
    }
    fit_failures: list[dict[str, str | int]] = []
    for group_index, group in enumerate(expected.bond_groups):
        stats = group_stats[group.kind]
        stats["expected"] += 1
        selected = [
            assignments[(group_index, member_index)]
            for member_index in range(len(group.members))
            if (group_index, member_index) in assignments
        ]
        required = 2 if group.kind == "hashed_wedge" else len(group.members)
        if len(selected) < required:
            rows.append(_missing_row(group.kind, len(group.members), group_index))
            continue
        stats["eligible"] += 1
        try:
            carrier = _carrier_row(group, selected, group_index)
            expanded = carrier_rows_to_bond_group_rows(
                [carrier],
                max_members=max_members,
            )
            if expanded.invalid_objects:
                raise ValueError(expanded.invalid_objects[0]["error"])
            rows.extend(expanded.rows)
            stats["emitted"] += 1
        except (RuntimeError, TypeError, ValueError) as error:
            stats["fit_failed"] += 1
            fit_failures.append(
                {
                    "group_index": group_index,
                    "kind": group.kind,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            rows.append(_missing_row(group.kind, len(group.members), group_index))

    text_offset = len(expected.bond_groups)
    for text_index, _ in enumerate(expected.text_boxes):
        stats = group_stats["text_box"]
        stats["expected"] += 1
        query_index = text_offset + text_index
        match = assignments.get((None, text_index))
        if not isinstance(match, TextBoxRecordV02):
            rows.append(_missing_row("text_box", 0, query_index))
            continue
        stats["eligible"] += 1
        stats["emitted"] += 1
        rows.append(
            {
                "query_index": query_index,
                "score": 1.0,
                "type": GROUP_KIND_TO_ID["text_box"],
                "kind": "text_box",
                "member_count": 0,
                "text_geometry": torch.tensor(
                    (
                        (match.x0 + match.x1) / (2.0 * 2047.0),
                        (match.y0 + match.y1) / (2.0 * 2047.0),
                        (match.x1 - match.x0) / 2047.0,
                        (match.y1 - match.y0) / 2047.0,
                    ),
                    dtype=torch.float32,
                ),
            }
        )

    return OracleAssembly(
        rows=tuple(rows),
        statistics={
            "member_association": member_stats,
            "groups": {
                kind: {name: int(value) for name, value in stats.items()}
                for kind, stats in group_stats.items()
            },
            "invalid_flat_predictions": invalid_predictions,
            "fit_failures": fit_failures,
        },
    )


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _load_prediction_rows(
    path: str | Path,
    *,
    decode_mode: str,
    threshold: float | None,
) -> dict[str, dict[str, Any]]:
    result = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            if row.get("decode_mode") != decode_mode:
                continue
            actual_threshold = row.get("threshold")
            if threshold is None:
                if actual_threshold is not None:
                    continue
            elif actual_threshold is None or not math.isclose(
                float(actual_threshold), threshold, abs_tol=1e-9
            ):
                continue
            key = str(row.get("sample_key", ""))
            if not key or key in result:
                raise ValueError(
                    f"Missing/duplicate sample key at {path}:{line_number}: {key!r}"
                )
            result[key] = row
    if not result:
        raise ValueError("No prediction rows matched the requested decode configuration")
    return result


def _not_internal(meta: dict[str, Any], key: str) -> None:
    split = str(meta.get("split", meta.get("source_split", ""))).lower().replace(
        "-", "_"
    )
    output_shard = str(meta.get("output_shard", "")).lower().replace("-", "_")
    if split == "internal_test" or "internal_test" in output_shard:
        raise RuntimeError(f"Refusing sealed internal_test sample: {key}")


def _selected_decode_result(
    payload: dict[str, Any], decode_mode: str, threshold: float | None
) -> dict[str, Any]:
    for row in payload.get("decode_results", []):
        if row.get("decode_mode") != decode_mode:
            continue
        actual = row.get("threshold")
        if threshold is None and actual is None:
            return row
        if actual is not None and threshold is not None and math.isclose(
            float(actual), threshold, abs_tol=1e-9
        ):
            return row
    raise ValueError("Baseline metrics lack the requested decode result")


def _compact_baselines(
    flat_metrics: dict[str, Any] | None,
    flat_geometry: dict[str, Any] | None,
    carrier_metrics: dict[str, Any] | None,
    *,
    decode_mode: str,
    threshold: float | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if flat_metrics is not None:
        row = _selected_decode_result(flat_metrics, decode_mode, threshold)
        result["flat_original"] = {
            "primitive_exact": row["primitive_exact"],
            "line_f1_at_evaluator_tolerance": row["classes"]["line"]["f1"],
            "text_box_f1": row["classes"]["text_box"]["f1"],
            "bond_iou": row["bond_iou"],
        }
    if flat_geometry is not None:
        row = _selected_decode_result(flat_geometry, decode_mode, threshold)
        result["flat_geometry"] = {
            "line_recall": row["line_geometry"]["recall"],
            "bond_iou": row["bond_iou"],
        }
    if carrier_metrics is not None:
        primary = str(carrier_metrics.get("primary_decode_mode", "count_head"))
        row = _selected_decode_result(carrier_metrics, primary, None)
        result["carrier_direct"] = {
            "checkpoint_global_step": carrier_metrics.get("checkpoint_global_step"),
            "group_set_exact": row["group_set_exact"],
            "flatten_exact": row["flatten_exact"],
            "line_recall": row["member_line_geometry"]["recall"],
            "bond_iou": row["bond_iou"],
        }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--shard", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--flat-metrics")
    parser.add_argument("--flat-geometry")
    parser.add_argument("--carrier-metrics")
    parser.add_argument("--decode-mode", choices=("threshold", "count_head"), default="threshold")
    parser.add_argument("--prediction-threshold", type=float, default=0.5)
    parser.add_argument("--tolerance-q", action="append", type=float)
    parser.add_argument("--text-box-iou", type=float, default=0.5)
    parser.add_argument("--max-members", type=int, default=143)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tolerances = tuple(
        sorted(set(args.tolerance_q if args.tolerance_q else DEFAULT_TOLERANCES_Q))
    )
    if not tolerances or any(value < 0.0 for value in tolerances):
        raise ValueError("Oracle q tolerances must be nonnegative")
    if not 0.0 <= args.text_box_iou <= 1.0:
        raise ValueError("--text-box-iou must be in [0, 1]")
    if args.max_members < 2:
        raise ValueError("--max-members must be at least 2")
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    predictions_path = Path(args.predictions).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not checkpoint.is_file() or not predictions_path.is_file():
        raise FileNotFoundError("Checkpoint and predictions must exist")
    if output_path.exists():
        raise RuntimeError(f"Audit output already exists: {output_path}")
    shards = validate_shards(args.shard)
    prediction_threshold = (
        None if args.decode_mode == "count_head" else args.prediction_threshold
    )
    predictions = _load_prediction_rows(
        predictions_path,
        decode_mode=args.decode_mode,
        threshold=prediction_threshold,
    )
    primitive_metrics = _load_json(args.flat_metrics) if args.flat_metrics else None
    if primitive_metrics is not None:
        expected_checkpoint = Path(primitive_metrics["checkpoint"]).resolve()
        if expected_checkpoint != checkpoint:
            raise ValueError(
                f"Flat metrics checkpoint mismatch: {expected_checkpoint} != {checkpoint}"
            )
    geometry_metrics = _load_json(args.flat_geometry) if args.flat_geometry else None
    carrier_metrics = _load_json(args.carrier_metrics) if args.carrier_metrics else None

    rows_by_tolerance: dict[float, list[dict[str, Any]]] = {
        tolerance: [] for tolerance in tolerances
    }
    aggregate_stats = {
        tolerance: {
            "member_association": {
                kind: Counter(predicted=0, expected=0, accepted=0)
                for kind in PHYSICAL_KINDS
            },
            "groups": {
                kind: Counter(expected=0, eligible=0, emitted=0, fit_failed=0)
                for kind in TYPE_NAMES
            },
            "invalid_flat_predictions": 0,
            "fit_failures": [],
        }
        for tolerance in tolerances
    }
    seen = set()
    dataset = VisualSVGWebDataset(list(shards))
    for sample in dataset:
        key = str(sample["key"])
        _not_internal(sample["meta"], key)
        if key not in predictions:
            raise ValueError(f"Missing flat prediction for shard sample {key}")
        seen.add(key)
        expected = visual_primitives_v0_1_to_visual_svg_v0_2(sample["primitives"])
        expected_sequence = serialize_parsed_visual_svg_v0_2(expected)
        objects = predictions[key].get("renderable_objects")
        if not isinstance(objects, list):
            raise ValueError(f"Prediction {key} lacks renderable_objects")
        for tolerance in tolerances:
            assembled = assemble_oracle_carriers(
                expected,
                objects,
                tolerance_q=tolerance,
                text_box_iou=args.text_box_iou,
                max_members=args.max_members,
            )
            rows_by_tolerance[tolerance].append(
                evaluate_decoded_sample(
                    sample_key=key,
                    decode_mode=f"oracle_{tolerance:g}q",
                    threshold=None,
                    decoded_rows=list(assembled.rows),
                    expected_v0_2_sequence=expected_sequence,
                    bond_layer=sample["bond_layer"],
                    line_tolerance=tolerance,
                    text_box_iou=args.text_box_iou,
                    max_members=args.max_members,
                )
            )
            target = aggregate_stats[tolerance]
            for kind, values in assembled.statistics["member_association"].items():
                target["member_association"][kind].update(values)
            for kind, values in assembled.statistics["groups"].items():
                target["groups"][kind].update(values)
            target["invalid_flat_predictions"] += len(
                assembled.statistics["invalid_flat_predictions"]
            )
            for failure in assembled.statistics["fit_failures"]:
                if len(target["fit_failures"]) < 50:
                    target["fit_failures"].append({"sample_key": key, **failure})
    extras = sorted(set(predictions) - seen)
    if extras:
        raise ValueError(f"Predictions contain keys absent from shards: {extras[:10]}")
    if not seen:
        raise RuntimeError("Oracle audit read no samples")

    modes = []
    for tolerance in tolerances:
        summary = summarize_prediction_rows(rows_by_tolerance[tolerance])
        stats = aggregate_stats[tolerance]
        modes.append(
            {
                "association_tolerance_q": tolerance,
                "oracle_inputs": {
                    "gt_group_count": True,
                    "gt_group_type": True,
                    "gt_group_partition_for_association_only": True,
                    "gt_geometry_used_for_output": False,
                    "hashed_minimum_matched_lines": 2,
                    "parallel_2_requires_both_lines": True,
                    "parallel_3_requires_all_three_lines": True,
                },
                "proposal_coverage": {
                    "member_association": {
                        kind: dict(values)
                        for kind, values in stats["member_association"].items()
                    },
                    "groups": {
                        kind: dict(values) for kind, values in stats["groups"].items()
                    },
                    "invalid_flat_predictions": stats["invalid_flat_predictions"],
                    "fit_failure_examples": stats["fit_failures"],
                },
                "evaluation": summary,
            }
        )

    payload = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "predictions": str(predictions_path),
        "predictions_sha256": file_sha256(predictions_path),
        "shards": list(shards),
        "samples": len(seen),
        "decode_mode": args.decode_mode,
        "prediction_threshold": prediction_threshold,
        "baselines": _compact_baselines(
            primitive_metrics,
            geometry_metrics,
            carrier_metrics,
            decode_mode=args.decode_mode,
            threshold=prediction_threshold,
        ),
        "oracle_modes": modes,
        "uses_internal_test": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
