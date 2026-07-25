#!/usr/bin/env python3
"""Read-only geometry audit for Primitive Set checkpoints."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch.utils.data import DataLoader

from Data.sdf.visual_svg_sequence_v0_1 import (
    ParsedVisualSVGSequenceV01,
    parse_visual_svg_v0_1,
)
from chemvector.evaluate_primitive_set import (
    PrimitiveSetEvaluationCollator,
    build_decode_groups,
    evaluate_decoded_sample,
    load_primitive_set_model,
    normalize_thresholds,
    summarize_prediction_rows,
    validate_shards,
)
from chemvector.evaluate_visual_svg import line_distance, line_true_positives
from chemvector.primitive_set import decode_primitive_set
from chemvector.visual_svg_data import VisualSVGWebDataset


AUDIT_SCHEMA_VERSION = "primitive-set-geometry-audit-0.1"
LINE_TOLERANCES_Q = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)


def _line_center(record) -> tuple[float, float]:
    return ((record.x0 + record.x1) * 0.5, (record.y0 + record.y1) * 0.5)


def _line_vector(record) -> tuple[float, float]:
    return (float(record.x1 - record.x0), float(record.y1 - record.y0))


def _line_length(record) -> float:
    x, y = _line_vector(record)
    return math.hypot(x, y)


def _undirected_angle_error_degrees(left, right) -> float:
    left_x, left_y = _line_vector(left)
    right_x, right_y = _line_vector(right)
    denominator = math.hypot(left_x, left_y) * math.hypot(right_x, right_y)
    if denominator <= 0.0:
        raise ValueError("Line angle requires non-degenerate records")
    cosine = abs((left_x * right_x + left_y * right_y) / denominator)
    return math.degrees(math.acos(min(max(cosine, 0.0), 1.0)))


def line_geometry_error_records(
    predicted: ParsedVisualSVGSequenceV01,
    expected: ParsedVisualSVGSequenceV01,
) -> list[dict[str, float]]:
    """Match unordered lines once, then expose interpretable geometry errors."""

    costs = np.asarray(
        [
            [line_distance(left, right) for right in expected.lines]
            for left in predicted.lines
        ],
        dtype=np.float64,
    )
    if not costs.size:
        return []
    rows, columns = linear_sum_assignment(costs)
    errors = []
    for row, column in zip(rows, columns):
        left = predicted.lines[int(row)]
        right = expected.lines[int(column)]
        left_center = _line_center(left)
        right_center = _line_center(right)
        errors.append(
            {
                "center_error_q": math.hypot(
                    left_center[0] - right_center[0],
                    left_center[1] - right_center[1],
                ),
                "max_endpoint_error_q": float(costs[row, column]),
                "length_error_q": abs(_line_length(left) - _line_length(right)),
                "angle_error_degrees": _undirected_angle_error_degrees(left, right),
            }
        )
    return errors


def distribution(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.5)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(array.max()),
    }


def summarize_line_geometry(
    *,
    expected_lines: int,
    tolerance_hits: dict[float, int],
    errors: list[dict[str, float]],
) -> dict[str, Any]:
    return {
        "expected_lines": expected_lines,
        "matched_pairs": len(errors),
        "unmatched_expected_lines": max(expected_lines - len(errors), 0),
        "recall": {
            f"{int(tolerance)}q": tolerance_hits.get(tolerance, 0)
            / max(expected_lines, 1)
            for tolerance in LINE_TOLERANCES_Q
        },
        "center_error_q": distribution(
            [row["center_error_q"] for row in errors]
        ),
        "max_endpoint_error_q": distribution(
            [row["max_endpoint_error_q"] for row in errors]
        ),
        "length_error_q": distribution(
            [row["length_error_q"] for row in errors]
        ),
        "angle_error_degrees": distribution(
            [row["angle_error_degrees"] for row in errors]
        ),
    }


def compare_geometry_outputs(
    real: dict[str, torch.Tensor],
    white: dict[str, torch.Tensor],
) -> dict[str, Any]:
    comparisons = {}
    for name in ("geometry", "text_geometry"):
        if name not in real or name not in white or real[name].shape != white[name].shape:
            raise ValueError(f"Real/white outputs lack matching {name}")
        absolute_q = (
            real[name].detach().float() - white[name].detach().float()
        ).abs() * 2047.0
        comparisons[f"{name}_coordinate_abs_q"] = distribution(
            absolute_q.reshape(-1).cpu().tolist()
        )
        query_max = absolute_q.amax(dim=-1)
        comparisons[f"{name}_query_max_abs_q"] = distribution(
            query_max.reshape(-1).cpu().tolist()
        )
    real_types = real["type_logits"].argmax(dim=-1)
    white_types = white["type_logits"].argmax(dim=-1)
    comparisons["type_argmax_same_rate"] = float(
        (real_types == white_types).float().mean()
    )
    real_counts = real["count_logits"].argmax(dim=-1)
    white_counts = white["count_logits"].argmax(dim=-1)
    comparisons["count_argmax_same_rate"] = float(
        (real_counts == white_counts).float().mean()
    )
    return comparisons


def _ensure_not_internal(meta: dict[str, Any], sample_key: str) -> None:
    split = str(meta.get("split", meta.get("source_split", ""))).lower().replace("-", "_")
    output_shard = str(meta.get("output_shard", "")).lower().replace("-", "_")
    if split == "internal_test" or "internal_test" in output_shard:
        raise RuntimeError(f"Refusing sealed internal_test sample: {sample_key}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--shard", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", action="append", type=float)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--text-box-iou", type=float, default=0.5)
    parser.add_argument("--compare-white", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    thresholds = normalize_thresholds(args.threshold)
    shards = validate_shards(args.shard)
    decode_groups = build_decode_groups(("threshold", "count_head"), thresholds)
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.batch_size < 1 or args.workers < 0 or args.image_size < 1:
        raise ValueError("Batch size/image size must be positive and workers nonnegative")
    if not 0.0 <= args.text_box_iou <= 1.0:
        raise ValueError("--text-box-iou must be in [0, 1]")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    output_path = Path(args.output)
    if output_path.exists():
        raise RuntimeError(f"Audit output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model, _ = load_primitive_set_model(args.checkpoint, device)
    loader = DataLoader(
        VisualSVGWebDataset(list(shards), limit=args.limit),
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=PrimitiveSetEvaluationCollator(image_size=args.image_size),
    )
    rows_by_group = {group: [] for group in decode_groups}
    line_errors = {group: [] for group in decode_groups}
    line_expected = {group: 0 for group in decode_groups}
    line_hits = {
        group: {tolerance: 0 for tolerance in LINE_TOLERANCES_Q}
        for group in decode_groups
    }
    white_values: dict[str, list[float]] = {
        "geometry_coordinate_abs_q": [],
        "geometry_query_max_abs_q": [],
        "text_geometry_coordinate_abs_q": [],
        "text_geometry_query_max_abs_q": [],
    }
    white_type_same = 0
    white_type_total = 0
    white_count_same = 0
    white_count_total = 0
    samples_seen = 0

    model.eval()
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
                real_output = model(pixels)
                white_output = model(torch.ones_like(pixels)) if args.compare_white else None
            if white_output is not None:
                for name in ("geometry", "text_geometry"):
                    absolute_q = (
                        real_output[name].float() - white_output[name].float()
                    ).abs() * 2047.0
                    white_values[f"{name}_coordinate_abs_q"].extend(
                        absolute_q.reshape(-1).cpu().tolist()
                    )
                    white_values[f"{name}_query_max_abs_q"].extend(
                        absolute_q.amax(dim=-1).reshape(-1).cpu().tolist()
                    )
                real_types = real_output["type_logits"].argmax(dim=-1)
                white_types = white_output["type_logits"].argmax(dim=-1)
                white_type_same += int((real_types == white_types).sum())
                white_type_total += real_types.numel()
                real_counts = real_output["count_logits"].argmax(dim=-1)
                white_counts = white_output["count_logits"].argmax(dim=-1)
                white_count_same += int((real_counts == white_counts).sum())
                white_count_total += real_counts.numel()

            for decode_mode, threshold in decode_groups:
                decoded = decode_primitive_set(
                    real_output,
                    threshold=threshold if threshold is not None else 0.5,
                    mode=decode_mode,
                )
                for index, decoded_rows in enumerate(decoded):
                    row = evaluate_decoded_sample(
                        sample_key=batch["keys"][index],
                        decode_mode=decode_mode,
                        threshold=threshold,
                        decoded_rows=decoded_rows,
                        expected_sequence=batch["target_sequences"][index],
                        bond_layer=batch["bond_layers"][index],
                        line_tolerance=8.0,
                        text_box_iou=args.text_box_iou,
                    )
                    rows_by_group[(decode_mode, threshold)].append(row)
                    expected = parse_visual_svg_v0_1(
                        batch["target_sequences"][index]
                    )
                    predicted = (
                        parse_visual_svg_v0_1(row["predicted_sequence"])
                        if row["predicted_sequence"] is not None
                        else ParsedVisualSVGSequenceV01((), (), (), ())
                    )
                    line_expected[(decode_mode, threshold)] += len(expected.lines)
                    line_errors[(decode_mode, threshold)].extend(
                        line_geometry_error_records(predicted, expected)
                    )
                    for tolerance in LINE_TOLERANCES_Q:
                        line_hits[(decode_mode, threshold)][tolerance] += (
                            line_true_positives(predicted, expected, tolerance)
                        )
            samples_seen += len(batch["keys"])
            if samples_seen % 10 == 0:
                print(f"audited={samples_seen}", flush=True)

    if not samples_seen:
        raise RuntimeError("Primitive Set geometry audit read no samples")
    decode_results = []
    for group in decode_groups:
        decode_mode, threshold = group
        evaluation = summarize_prediction_rows(rows_by_group[group])
        decode_results.append(
            {
                "decode_mode": decode_mode,
                "threshold": threshold,
                "samples": evaluation["samples"],
                "primitive_exact": evaluation["primitive_exact"],
                "count_exact": evaluation["primitive_count"]["exact"],
                "count_mae": evaluation["primitive_count"]["mae"],
                "class_count_exact": {
                    name: values["count_exact"]
                    for name, values in evaluation["classes"].items()
                },
                "line_geometry": summarize_line_geometry(
                    expected_lines=line_expected[group],
                    tolerance_hits=line_hits[group],
                    errors=line_errors[group],
                ),
                "bond_iou": evaluation["bond_iou"],
                "invalid": evaluation["invalid"],
                "conditional_on_valid": {
                    "samples": evaluation["conditional_on_valid"]["samples"],
                    "bond_iou": evaluation["conditional_on_valid"]["bond_iou"],
                },
            }
        )
    white_comparison = None
    if args.compare_white:
        white_comparison = {
            name: distribution(values) for name, values in white_values.items()
        }
        white_comparison["type_argmax_same_rate"] = white_type_same / max(
            white_type_total, 1
        )
        white_comparison["count_argmax_same_rate"] = white_count_same / max(
            white_count_total, 1
        )
    payload = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "shards": list(shards),
        "samples": samples_seen,
        "decode_results": decode_results,
        "real_white_geometry": white_comparison,
        "uses_internal_test": False,
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
