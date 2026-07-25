#!/usr/bin/env python3
"""Read-only residual forensics for Primitive Set line predictions."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch.utils.data import DataLoader

from Data.sdf.visual_svg_sequence_v0_1 import (
    COORDINATE_MAX,
    LineRecordV01,
    parse_visual_svg_v0_1,
)
from chemvector.model import ChemVectorConfig
from chemvector.primitive_set import PrimitiveSetConfig, PrimitiveSetPredictor
from chemvector.primitive_set_data import PrimitiveSetCollator
from chemvector.visual_svg_data import VisualSVGWebDataset


AUDIT_SCHEMA_VERSION = "primitive-set-residual-audit-0.1"
ENDPOINT_REFINER_PREFIXES = (
    "endpoint_refine_projection.",
    "endpoint_refine_depthwise.",
    "endpoint_point_embedding.",
    "endpoint_refine_mlp.",
)


def _ensure_not_internal(meta: dict[str, Any], sample_key: str) -> None:
    split = str(meta.get("split", meta.get("source_split", ""))).lower().replace(
        "-", "_"
    )
    output_shard = str(meta.get("output_shard", "")).lower().replace("-", "_")
    if split == "internal_test" or "internal_test" in output_shard:
        raise RuntimeError(f"Refusing sealed internal_test sample: {sample_key}")


def _named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected LABEL=PATH")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    path = Path(raw_path).expanduser().resolve()
    if not label or not path.is_file():
        raise argparse.ArgumentTypeError(f"Invalid named path: {value}")
    if "internal_test" in str(path).lower().replace("-", "_"):
        raise argparse.ArgumentTypeError(f"internal_test is sealed: {path}")
    return label, path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_shards(values: list[str]) -> tuple[str, ...]:
    resolved = []
    for value in values:
        if any(character in value for character in "*?[]{}"):
            raise ValueError(f"Shard patterns are not allowed: {value}")
        path = Path(value).expanduser().resolve()
        if "internal_test" in str(path).lower().replace("-", "_"):
            raise ValueError(f"internal_test is sealed: {path}")
        if not path.is_file():
            raise ValueError(f"Shard does not exist: {path}")
        resolved.append(str(path))
    if not resolved:
        raise ValueError("At least one shard is required")
    return tuple(dict.fromkeys(resolved))


def _checkpoint_configs(checkpoint: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    nested = checkpoint.get("config")
    nested = nested if isinstance(nested, dict) else {}
    vision = checkpoint.get("vision_config", nested.get("vision_config"))
    primitive = checkpoint.get(
        "primitive_set_config", nested.get("primitive_set_config")
    )
    if not isinstance(vision, dict) or not isinstance(primitive, dict):
        raise ValueError("Primitive Set checkpoint lacks model configurations")
    return vision, primitive


def _load_model(checkpoint_path: Path, device: torch.device) -> PrimitiveSetPredictor:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_type") != PrimitiveSetPredictor.model_type:
        raise ValueError(
            f"Unsupported checkpoint model_type: {checkpoint.get('model_type')!r}"
        )
    vision, primitive = _checkpoint_configs(checkpoint)
    model = PrimitiveSetPredictor(
        ChemVectorConfig(**vision), PrimitiveSetConfig(**primitive)
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval()


def _line_distance(left: LineRecordV01, right: LineRecordV01) -> float:
    direct = max(
        math.hypot(left.x0 - right.x0, left.y0 - right.y0),
        math.hypot(left.x1 - right.x1, left.y1 - right.y1),
    )
    reversed_distance = max(
        math.hypot(left.x0 - right.x1, left.y0 - right.y1),
        math.hypot(left.x1 - right.x0, left.y1 - right.y0),
    )
    return min(direct, reversed_distance)


def _line_from_points(points: list[list[int]], stroke_width: int = 1) -> LineRecordV01:
    if len(points) != 2 or any(len(point) != 2 for point in points):
        raise ValueError("A line requires two 2D points")
    return LineRecordV01(
        int(points[0][0]),
        int(points[0][1]),
        int(points[1][0]),
        int(points[1][1]),
        int(stroke_width),
    )


def _line_length(line: LineRecordV01) -> float:
    return math.hypot(line.x1 - line.x0, line.y1 - line.y0)


def _line_orientation_degrees(line: LineRecordV01) -> float:
    angle = math.degrees(math.atan2(line.y1 - line.y0, line.x1 - line.x0)) % 180.0
    return min(angle, 180.0 - angle)


def _point_box_distance(x: float, y: float, box) -> float:
    dx = max(float(box.x0) - x, 0.0, x - float(box.x1))
    dy = max(float(box.y0) - y, 0.0, y - float(box.y1))
    return math.hypot(dx, dy)


def nearest_text_endpoint_distance(
    line: LineRecordV01,
    text_boxes: Iterable[Any],
) -> float | None:
    boxes = tuple(text_boxes)
    if not boxes:
        return None
    return min(
        _point_box_distance(x, y, box)
        for x, y in ((line.x0, line.y0), (line.x1, line.y1))
        for box in boxes
    )


def classify_bond_group(group: dict[str, Any] | None) -> str:
    if group is None:
        return "unmapped"
    direction = str(group.get("direction", "NONE")).upper()
    display = str(group.get("display_type", group.get("chemical_type", "OTHER"))).upper()
    if "DASH" in direction:
        return "hashed_wedge"
    if "WEDGE" in direction:
        return "solid_wedge"
    if display == "TRIPLE":
        return "triple"
    if display == "DOUBLE":
        return "double"
    if display == "AROMATIC":
        return "aromatic"
    if display == "SINGLE":
        return "single"
    return "other"


def length_bin(value: float) -> str:
    if value < 16.0:
        return "00:<16q"
    if value < 32.0:
        return "01:16-32q"
    if value < 64.0:
        return "02:32-64q"
    if value < 128.0:
        return "03:64-128q"
    if value < 256.0:
        return "04:128-256q"
    return "05:>=256q"


def orientation_bin(value: float) -> str:
    if value < 15.0:
        return "00:horizontal_0-15deg"
    if value < 45.0:
        return "01:shallow_15-45deg"
    if value < 75.0:
        return "02:steep_45-75deg"
    return "03:vertical_75-90deg"


def text_distance_bin(value: float | None) -> str:
    if value is None:
        return "04:no_text"
    if value <= 4.0:
        return "00:touching_0-4q"
    if value <= 16.0:
        return "01:near_4-16q"
    if value <= 32.0:
        return "02:near_16-32q"
    return "03:far_>32q"


def coarse_error_bin(value: float | None) -> str:
    if value is None:
        return "06:unavailable"
    if value <= 32.0:
        return "00:<=32q"
    if value <= 64.0:
        return "01:32-64q"
    if value <= 128.0:
        return "02:64-128q"
    if value <= 256.0:
        return "03:128-256q"
    if value <= 512.0:
        return "04:256-512q"
    return "05:>512q"


def group_line_count_bin(value: int) -> str:
    if value <= 1:
        return "00:1_line"
    if value == 2:
        return "01:2_lines"
    if value == 3:
        return "02:3_lines"
    if value <= 8:
        return "03:4-8_lines"
    return "04:>=9_lines"


def bond_length_signature(category: str, length: float) -> str:
    if length < 32.0:
        scale = "short_<32q"
    elif length < 64.0:
        scale = "medium_32-64q"
    else:
        scale = "normal_>=64q"
    return f"{category}|{scale}"


def _distribution(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.5)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(array.max()),
    }


def _aligned_endpoint_errors(
    predicted: LineRecordV01,
    expected: LineRecordV01,
) -> tuple[float, float]:
    direct = (
        math.hypot(predicted.x0 - expected.x0, predicted.y0 - expected.y0),
        math.hypot(predicted.x1 - expected.x1, predicted.y1 - expected.y1),
    )
    reverse = (
        math.hypot(predicted.x0 - expected.x1, predicted.y0 - expected.y1),
        math.hypot(predicted.x1 - expected.x0, predicted.y1 - expected.y0),
    )
    return direct if max(direct) <= max(reverse) else reverse


def _geometry_errors(
    predicted: LineRecordV01,
    expected: LineRecordV01,
) -> dict[str, float]:
    endpoint_errors = _aligned_endpoint_errors(predicted, expected)
    predicted_center = ((predicted.x0 + predicted.x1) * 0.5, (predicted.y0 + predicted.y1) * 0.5)
    expected_center = ((expected.x0 + expected.x1) * 0.5, (expected.y0 + expected.y1) * 0.5)
    predicted_vector = (predicted.x1 - predicted.x0, predicted.y1 - predicted.y0)
    expected_vector = (expected.x1 - expected.x0, expected.y1 - expected.y0)
    denominator = math.hypot(*predicted_vector) * math.hypot(*expected_vector)
    if denominator <= 0.0:
        angle_error = 90.0
    else:
        cosine = abs(
            (
                predicted_vector[0] * expected_vector[0]
                + predicted_vector[1] * expected_vector[1]
            )
            / denominator
        )
        angle_error = math.degrees(math.acos(min(max(cosine, 0.0), 1.0)))
    return {
        "max_endpoint_error_q": max(endpoint_errors),
        "endpoint0_error_q": endpoint_errors[0],
        "endpoint1_error_q": endpoint_errors[1],
        "center_error_q": math.hypot(
            predicted_center[0] - expected_center[0],
            predicted_center[1] - expected_center[1],
        ),
        "length_error_q": abs(_line_length(predicted) - _line_length(expected)),
        "angle_error_degrees": angle_error,
    }


def _expected_line_metadata(
    row: dict[str, Any],
    primitives_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    expected = parse_visual_svg_v0_1(row["expected_sequence"])
    line_primitives = sorted(
        (
            primitive
            for primitive in primitives_payload.get("primitives", [])
            if primitive.get("kind") == "line"
        ),
        key=lambda primitive: int(primitive["canonical_index"]),
    )
    if len(line_primitives) != len(expected.lines):
        raise ValueError(
            f"{row['sample_key']} line sidecar mismatch: "
            f"{len(line_primitives)} != {len(expected.lines)}"
        )
    group_by_primitive = {}
    primitive_by_id = {
        int(primitive["primitive_id"]): primitive
        for primitive in primitives_payload.get("primitives", [])
    }
    for group in primitives_payload.get("bond_groups", []):
        for primitive_id in group.get("primitive_ids", []):
            group_by_primitive[int(primitive_id)] = group
    metadata = []
    for expected_index, (line, primitive) in enumerate(
        zip(expected.lines, line_primitives)
    ):
        sidecar_line = _line_from_points(
            primitive["quantized_points"],
            primitive.get("stroke_width_q", 1),
        )
        if (
            _line_distance(sidecar_line, line) != 0.0
            or sidecar_line.stroke_width != line.stroke_width
        ):
            raise ValueError(
                f"{row['sample_key']} line {expected_index} does not match sidecar"
            )
        primitive_id = int(primitive["primitive_id"])
        group = group_by_primitive.get(primitive_id)
        group_line_count = 0
        if group is not None:
            group_line_count = sum(
                primitive_by_id.get(int(member), {}).get("kind") == "line"
                for member in group.get("primitive_ids", [])
            )
        metadata.append(
            {
                "expected_index": expected_index,
                "line": line,
                "primitive_id": primitive_id,
                "canonical_index": int(primitive["canonical_index"]),
                "bond_group": group,
                "bond_id": int(group["bond_id"]) if group is not None else None,
                "bond_category": classify_bond_group(group),
                "display_type": str(group.get("display_type")) if group else None,
                "direction": str(group.get("direction")) if group else None,
                "group_line_count": group_line_count,
            }
        )
    return metadata


def _prediction_lines(row: dict[str, Any]) -> list[dict[str, Any]]:
    lines = []
    for obj in row.get("renderable_objects", []):
        if obj.get("kind") != "line":
            continue
        lines.append(
            {
                "query_index": int(obj["query_index"]),
                "score": float(obj.get("score", 0.0)),
                "line": _line_from_points(
                    obj["quantized_points"], obj.get("stroke_width_q", 1)
                ),
            }
        )
    return lines


def _coarse_line(values: list[float]) -> LineRecordV01:
    coordinates = [
        min(max(int(round(float(value) * COORDINATE_MAX)), 0), COORDINATE_MAX)
        for value in values[:4]
    ]
    return LineRecordV01(*coordinates, 1)


def build_residual_records(
    row: dict[str, Any],
    primitives_payload: dict[str, Any],
    coarse_by_query: dict[int, list[float]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Match one predicted line set to GT and retain sidecar-backed attributes."""

    expected_metadata = _expected_line_metadata(row, primitives_payload)
    expected = [item["line"] for item in expected_metadata]
    predicted_metadata = _prediction_lines(row)
    predicted = [item["line"] for item in predicted_metadata]
    costs = np.asarray(
        [[_line_distance(left, right) for right in expected] for left in predicted],
        dtype=np.float64,
    )
    matched_by_expected: dict[int, int] = {}
    if costs.size:
        rows, columns = linear_sum_assignment(costs)
        matched_by_expected = {
            int(column): int(predicted_index)
            for predicted_index, column in zip(rows, columns)
        }
    text_boxes = parse_visual_svg_v0_1(row["expected_sequence"]).text_boxes
    records = []
    for item in expected_metadata:
        expected_index = item["expected_index"]
        target = item["line"]
        predicted_index = matched_by_expected.get(expected_index)
        record = {
            "sample_key": str(row["sample_key"]),
            "primitive_id": item["primitive_id"],
            "canonical_index": item["canonical_index"],
            "expected_index": expected_index,
            "bond_id": item["bond_id"],
            "bond_category": item["bond_category"],
            "display_type": item["display_type"],
            "direction": item["direction"],
            "group_line_count": item["group_line_count"],
            "target_points": [[target.x0, target.y0], [target.x1, target.y1]],
            "target_length_q": _line_length(target),
            "target_orientation_degrees": _line_orientation_degrees(target),
            "text_endpoint_distance_q": nearest_text_endpoint_distance(
                target, text_boxes
            ),
            "matched": predicted_index is not None,
            "query_index": None,
            "score": None,
            "predicted_points": None,
            "final_error_q": None,
            "coarse_points": None,
            "coarse_error_q": None,
            "refinement_gain_q": None,
        }
        if predicted_index is not None:
            prediction = predicted_metadata[predicted_index]
            predicted_line = prediction["line"]
            geometry = _geometry_errors(predicted_line, target)
            record.update(geometry)
            record["query_index"] = prediction["query_index"]
            record["score"] = prediction["score"]
            record["predicted_points"] = [
                [predicted_line.x0, predicted_line.y0],
                [predicted_line.x1, predicted_line.y1],
            ]
            record["final_error_q"] = geometry["max_endpoint_error_q"]
            if coarse_by_query is not None:
                coarse_values = coarse_by_query.get(prediction["query_index"])
                if coarse_values is None:
                    raise ValueError(
                        f"Missing coarse geometry for query {prediction['query_index']}"
                    )
                coarse_line = _coarse_line(coarse_values)
                coarse_error = _line_distance(coarse_line, target)
                record["coarse_points"] = [
                    [coarse_line.x0, coarse_line.y0],
                    [coarse_line.x1, coarse_line.y1],
                ]
                record["coarse_error_q"] = coarse_error
                record["refinement_gain_q"] = coarse_error - record["final_error_q"]
        record["length_bin"] = length_bin(record["target_length_q"])
        record["orientation_bin"] = orientation_bin(
            record["target_orientation_degrees"]
        )
        record["text_distance_bin"] = text_distance_bin(
            record["text_endpoint_distance_q"]
        )
        record["coarse_error_bin"] = coarse_error_bin(record["coarse_error_q"])
        record["group_line_count_bin"] = group_line_count_bin(
            record["group_line_count"]
        )
        record["bond_length_signature"] = bond_length_signature(
            record["bond_category"], record["target_length_q"]
        )
        if record["final_error_q"] is None:
            record["endpoint_failure_mode"] = "03:unmatched"
        elif record["final_error_q"] <= 8.0:
            record["endpoint_failure_mode"] = "00:both_within_8q"
        elif min(record["endpoint0_error_q"], record["endpoint1_error_q"]) <= 8.0:
            record["endpoint_failure_mode"] = "01:one_endpoint_outside_8q"
        else:
            record["endpoint_failure_mode"] = "02:both_outside_8q"
        records.append(record)
    return records, {
        "expected_lines": len(expected),
        "predicted_lines": len(predicted),
        "matched_lines": len(matched_by_expected),
        "extra_predicted_lines": max(len(predicted) - len(matched_by_expected), 0),
        "unmatched_expected_lines": max(len(expected) - len(matched_by_expected), 0),
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    matched_errors = [
        float(record["final_error_q"])
        for record in records
        if record["final_error_q"] is not None
    ]
    failures = [
        record
        for record in records
        if record["final_error_q"] is None or record["final_error_q"] > 8.0
    ]
    success_8q = len(records) - len(failures)
    success_32q = sum(
        record["final_error_q"] is not None and record["final_error_q"] <= 32.0
        for record in records
    )
    coarse = [
        float(record["coarse_error_q"])
        for record in records
        if record["coarse_error_q"] is not None
    ]
    gains = [
        float(record["refinement_gain_q"])
        for record in records
        if record["refinement_gain_q"] is not None
    ]
    return {
        "lines": len(records),
        "matched_lines": len(matched_errors),
        "unmatched_lines": len(records) - len(matched_errors),
        "success_8q": success_8q,
        "failure_8q": len(failures),
        "recall_8q": success_8q / max(len(records), 1),
        "recall_32q": success_32q / max(len(records), 1),
        "final_error_q": _distribution(matched_errors),
        "coarse_error_q": _distribution(coarse),
        "refinement_gain_q": _distribution(gains),
    }


def summarize_buckets(
    records: list[dict[str, Any]],
    field: str,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record[field])].append(record)
    total_failures = sum(
        record["final_error_q"] is None or record["final_error_q"] > 8.0
        for record in records
    )
    result = {}
    for name in sorted(grouped):
        summary = summarize_records(grouped[name])
        summary["failure_share"] = summary["failure_8q"] / max(total_failures, 1)
        result[name] = summary
    return result


def _load_prediction_rows(
    path: Path,
    *,
    decode_mode: str,
    threshold: float,
) -> dict[str, dict[str, Any]]:
    selected = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            if row.get("decode_mode") != decode_mode:
                continue
            if decode_mode == "threshold" and not math.isclose(
                float(row.get("threshold")), threshold, abs_tol=1e-9
            ):
                continue
            key = str(row["sample_key"])
            if key in selected:
                raise ValueError(f"Duplicate prediction row for {key} in {path}")
            selected[key] = row
    if not selected:
        raise ValueError(f"No matching prediction rows in {path}")
    return selected


def _load_sidecars(shards: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    samples = {}
    for sample in VisualSVGWebDataset(list(shards)):
        _ensure_not_internal(sample["meta"], sample["key"])
        samples[str(sample["key"])] = sample["primitives"]
    if not samples:
        raise RuntimeError("Residual audit read no sidecars")
    return samples


def _collect_coarse_geometry(
    checkpoint: Path,
    shards: tuple[str, ...],
    *,
    device: torch.device,
    batch_size: int,
    workers: int,
    image_size: int,
) -> dict[str, dict[int, list[float]]]:
    model = _load_model(checkpoint, device)
    loader = DataLoader(
        VisualSVGWebDataset(list(shards)),
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        collate_fn=PrimitiveSetCollator(image_size=image_size),
    )
    result = {}
    with torch.inference_mode():
        for batch in loader:
            for key, meta in zip(batch["keys"], batch["meta"]):
                _ensure_not_internal(meta, key)
            pixels = batch["pixel_values"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                output = model(pixels)
            if "coarse_geometry" not in output:
                raise ValueError(f"Checkpoint has no endpoint refiner: {checkpoint}")
            coarse = output["coarse_geometry"].detach().float().cpu()
            for sample_index, key in enumerate(batch["keys"]):
                result[str(key)] = {
                    query_index: coarse[sample_index, query_index].tolist()
                    for query_index in range(coarse.size(1))
                }
    return result


def _checkpoint_parameter_comparison(
    first: Path,
    second: Path,
) -> dict[str, Any]:
    left = torch.load(first, map_location="cpu", weights_only=False).get("model", {})
    right = torch.load(second, map_location="cpu", weights_only=False).get("model", {})
    if left.keys() != right.keys():
        return {"same_parameter_names": False}
    changed_refiner = []
    changed_non_refiner = []
    for name in left:
        if torch.equal(left[name], right[name]):
            continue
        target = (
            changed_refiner
            if name.startswith(ENDPOINT_REFINER_PREFIXES)
            else changed_non_refiner
        )
        target.append(name)
    return {
        "same_parameter_names": True,
        "changed_refiner_parameters": changed_refiner,
        "changed_non_refiner_parameters": changed_non_refiner,
        "endpoint_refiner_only_confirmed": bool(changed_refiner)
        and not changed_non_refiner,
    }


def compare_coarse_outputs(
    left: dict[str, dict[int, list[float]]],
    right: dict[str, dict[int, list[float]]],
) -> dict[str, Any]:
    if left.keys() != right.keys():
        raise ValueError("Coarse outputs do not contain the same samples")
    coordinate_errors = []
    query_errors = []
    for sample_key in left:
        if left[sample_key].keys() != right[sample_key].keys():
            raise ValueError(f"Coarse query sets differ for {sample_key}")
        for query_index in left[sample_key]:
            errors = [
                abs(float(a) - float(b)) * COORDINATE_MAX
                for a, b in zip(
                    left[sample_key][query_index], right[sample_key][query_index]
                )
            ]
            coordinate_errors.extend(errors)
            query_errors.append(max(errors))
    return {
        "coordinate_abs_q": _distribution(coordinate_errors),
        "query_max_abs_q": _distribution(query_errors),
        "exact_query_rate": sum(error == 0.0 for error in query_errors)
        / max(len(query_errors), 1),
    }


def compare_runs(
    left_label: str,
    left_records: list[dict[str, Any]],
    right_label: str,
    right_records: list[dict[str, Any]],
) -> dict[str, Any]:
    def keyed(records):
        return {
            (record["sample_key"], int(record["primitive_id"])): record
            for record in records
        }

    left = keyed(left_records)
    right = keyed(right_records)
    if left.keys() != right.keys():
        raise ValueError("Compared runs do not contain the same GT line set")
    transitions = defaultdict(int)
    improvements = []
    right_failures = []
    for key in left:
        left_error = left[key]["final_error_q"]
        right_error = right[key]["final_error_q"]
        left_pass = left_error is not None and left_error <= 8.0
        right_pass = right_error is not None and right_error <= 8.0
        transitions[f"{'pass' if left_pass else 'fail'}_to_{'pass' if right_pass else 'fail'}"] += 1
        if left_error is not None and right_error is not None:
            improvements.append(float(left_error) - float(right_error))
        if not right_pass:
            right_failures.append(right[key])
    return {
        "from": left_label,
        "to": right_label,
        "recall_8q_delta": summarize_records(right_records)["recall_8q"]
        - summarize_records(left_records)["recall_8q"],
        "transitions": dict(sorted(transitions.items())),
        "final_error_improvement_q": _distribution(improvements),
        "remaining_failures": {
            "lines": len(right_failures),
            "by_bond_category": summarize_buckets(right_failures, "bond_category"),
            "by_length": summarize_buckets(right_failures, "length_bin"),
            "by_orientation": summarize_buckets(right_failures, "orientation_bin"),
            "by_text_distance": summarize_buckets(right_failures, "text_distance_bin"),
            "by_coarse_error": summarize_buckets(right_failures, "coarse_error_bin"),
            "by_endpoint_failure_mode": summarize_buckets(
                right_failures, "endpoint_failure_mode"
            ),
            "by_bond_length_signature": summarize_buckets(
                right_failures, "bond_length_signature"
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        type=_named_path,
        help="Repeat LABEL=predictions.jsonl",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=_named_path,
        help="Optional matching LABEL=checkpoint.pt for coarse-error replay",
    )
    parser.add_argument("--shard", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--decode-mode", choices=("threshold", "count_head"), default="threshold")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be in [0, 1]")
    if args.batch_size < 1 or args.workers < 0 or args.image_size < 1:
        raise ValueError("Batch size/image size must be positive and workers nonnegative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    shards = _validate_shards(args.shard)
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise RuntimeError(f"Audit output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    predictions = dict(args.prediction)
    if len(predictions) != len(args.prediction):
        raise ValueError("Prediction labels must be unique")
    checkpoints = dict(args.checkpoint or [])
    if len(checkpoints) != len(args.checkpoint or []):
        raise ValueError("Checkpoint labels must be unique")
    unknown_checkpoints = set(checkpoints) - set(predictions)
    if unknown_checkpoints:
        raise ValueError(f"Checkpoints lack matching predictions: {unknown_checkpoints}")

    sidecars = _load_sidecars(shards)
    device = torch.device(args.device)
    coarse = {}
    for label, checkpoint in checkpoints.items():
        coarse[label] = _collect_coarse_geometry(
            checkpoint,
            shards,
            device=device,
            batch_size=args.batch_size,
            workers=args.workers,
            image_size=args.image_size,
        )

    run_payloads = []
    all_records = {}
    for label, prediction_path in predictions.items():
        rows = _load_prediction_rows(
            prediction_path,
            decode_mode=args.decode_mode,
            threshold=args.threshold,
        )
        if rows.keys() != sidecars.keys():
            missing = sorted(sidecars.keys() - rows.keys())
            extra = sorted(rows.keys() - sidecars.keys())
            raise ValueError(
                f"{label} prediction/sidecar keys differ: missing={missing[:5]} extra={extra[:5]}"
            )
        records = []
        count_totals = defaultdict(int)
        for sample_key, row in rows.items():
            sample_records, counts = build_residual_records(
                row,
                sidecars[sample_key],
                coarse.get(label, {}).get(sample_key),
            )
            records.extend(sample_records)
            for name, value in counts.items():
                count_totals[name] += value
        all_records[label] = records
        summary = summarize_records(records)
        run_payloads.append(
            {
                "label": label,
                "prediction": str(prediction_path),
                "prediction_sha256": _sha256(prediction_path),
                "checkpoint": str(checkpoints[label]) if label in checkpoints else None,
                "checkpoint_sha256": _sha256(checkpoints[label]) if label in checkpoints else None,
                "sample_count": len(rows),
                "line_counts": dict(count_totals),
                "overall": summary,
                "buckets": {
                    "bond_category": summarize_buckets(records, "bond_category"),
                    "line_length": summarize_buckets(records, "length_bin"),
                    "orientation": summarize_buckets(records, "orientation_bin"),
                    "text_distance": summarize_buckets(records, "text_distance_bin"),
                    "coarse_error": summarize_buckets(records, "coarse_error_bin"),
                    "group_line_count": summarize_buckets(
                        records, "group_line_count_bin"
                    ),
                    "endpoint_failure_mode": summarize_buckets(
                        records, "endpoint_failure_mode"
                    ),
                    "bond_length_signature": summarize_buckets(
                        records, "bond_length_signature"
                    ),
                },
                "top_failures": sorted(
                    (
                        record
                        for record in records
                        if record["final_error_q"] is None
                        or record["final_error_q"] > 8.0
                    ),
                    key=lambda record: (
                        record["final_error_q"] is not None,
                        -(record["final_error_q"] or 0.0),
                    ),
                )[:50],
            }
        )

    labels = list(predictions)
    comparison = None
    parameter_comparison = None
    coarse_output_comparison = None
    if len(labels) == 2:
        comparison = compare_runs(
            labels[0], all_records[labels[0]], labels[1], all_records[labels[1]]
        )
        if all(label in checkpoints for label in labels):
            parameter_comparison = _checkpoint_parameter_comparison(
                checkpoints[labels[0]], checkpoints[labels[1]]
            )
            coarse_output_comparison = compare_coarse_outputs(
                coarse[labels[0]], coarse[labels[1]]
            )
    payload = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "decode_mode": args.decode_mode,
        "threshold": args.threshold if args.decode_mode == "threshold" else None,
        "shards": list(shards),
        "runs": run_payloads,
        "comparison": comparison,
        "checkpoint_parameter_comparison": parameter_comparison,
        "coarse_output_comparison": coarse_output_comparison,
        "uses_internal_test": False,
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "runs": [
                    {"label": run["label"], "overall": run["overall"]}
                    for run in run_payloads
                ],
                "comparison": (
                    {
                        "from": comparison["from"],
                        "to": comparison["to"],
                        "recall_8q_delta": comparison["recall_8q_delta"],
                        "transitions": comparison["transitions"],
                        "final_error_improvement_q": comparison[
                            "final_error_improvement_q"
                        ],
                    }
                    if comparison is not None
                    else None
                ),
                "checkpoint_parameter_comparison": parameter_comparison,
                "coarse_output_comparison": coarse_output_comparison,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
