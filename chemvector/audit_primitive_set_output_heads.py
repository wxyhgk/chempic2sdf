#!/usr/bin/env python3
"""Read-only selector and text-box audit for Primitive Set checkpoints."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch.utils.data import DataLoader

from Data.sdf.visual_svg_sequence_v0_1 import COORDINATE_MAX
from chemvector.evaluate_primitive_set import (
    PrimitiveSetEvaluationCollator,
    evaluate_decoded_sample,
    load_primitive_set_model,
    summarize_prediction_rows,
    validate_shards,
)
from chemvector.primitive_set import (
    LINE,
    NO_OBJECT,
    PRIMITIVE_TYPES,
    QUAD,
    TEXT_BOX,
    TRIANGLE,
    primitive_set_assignments,
)
from chemvector.train_primitive_set import TARGET_FIELDS
from chemvector.visual_svg_data import VisualSVGWebDataset


AUDIT_SCHEMA_VERSION = "primitive-set-output-head-audit-0.1"
CLASS_IDS = (LINE, TRIANGLE, QUAD, TEXT_BOX)
CLASS_NAMES = {type_id: PRIMITIVE_TYPES[type_id] for type_id in CLASS_IDS}


def distribution(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size:
        return {
            "count": 0,
            "mean": 0.0,
            "min": 0.0,
            "p10": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "max": 0.0,
        }
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "min": float(array.min()),
        "p10": float(np.quantile(array, 0.1)),
        "p50": float(np.quantile(array, 0.5)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(array.max()),
    }


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / max(float(denominator), 1.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def named_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Checkpoint must use LABEL=PATH")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    path = Path(raw_path).expanduser().resolve()
    if not label or not path.is_file():
        raise argparse.ArgumentTypeError(f"Invalid checkpoint: {value}")
    if "internal_test" in str(path).lower().replace("-", "_"):
        raise argparse.ArgumentTypeError(f"internal_test is sealed: {path}")
    return label, path


def targets_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        name: batch[name].to(device, non_blocking=True)
        for name in TARGET_FIELDS
    }


def global_topk_indices(
    probabilities: torch.Tensor,
    count: int,
) -> torch.Tensor:
    """Reproduce the production total-count top-k ranking."""

    if probabilities.ndim != 2 or probabilities.size(-1) != len(PRIMITIVE_TYPES):
        raise ValueError("Probabilities must have shape [queries, primitive_types]")
    count = min(max(int(count), 0), probabilities.size(0))
    object_scores = probabilities[:, 1:].amax(dim=-1)
    margins = object_scores - probabilities[:, NO_OBJECT]
    return torch.argsort(margins, descending=True, stable=True)[:count]


def class_count_oracle_indices(
    probabilities: torch.Tensor,
    class_counts: dict[int, int],
) -> list[tuple[int, int]]:
    """Select unique queries using GT class counts but no target geometry.

    Repeated class slots are jointly assigned to queries, so a query cannot be
    reused for two primitive classes. This is an upper bound for a per-class
    count decoder, not a deployable decode mode.
    """

    if probabilities.ndim != 2 or probabilities.size(-1) != len(PRIMITIVE_TYPES):
        raise ValueError("Probabilities must have shape [queries, primitive_types]")
    slots = [
        type_id
        for type_id in CLASS_IDS
        for _ in range(max(int(class_counts.get(type_id, 0)), 0))
    ]
    if len(slots) > probabilities.size(0):
        raise ValueError("Requested class counts exceed query capacity")
    if not slots:
        return []
    no_object = probabilities[:, NO_OBJECT].detach().float().cpu().numpy()
    object_probabilities = probabilities.detach().float().cpu().numpy()
    costs = np.stack(
        [-(object_probabilities[:, type_id] - no_object) for type_id in slots],
        axis=1,
    )
    rows, columns = linear_sum_assignment(costs)
    selected = [(int(row), int(slots[int(column)])) for row, column in zip(rows, columns)]
    selected.sort(
        key=lambda item: (
            -float(probabilities[item[0], item[1]]),
            item[0],
            item[1],
        )
    )
    return selected


def text_box_iou_from_geometry(
    predicted: torch.Tensor,
    expected: torch.Tensor,
) -> float:
    if predicted.numel() < 4 or expected.numel() < 4:
        raise ValueError("Text geometry requires center x/y and width/height")
    predicted = predicted.detach().float().cpu()
    expected = expected.detach().float().cpu()

    def corners(values: torch.Tensor) -> tuple[float, float, float, float]:
        center_x, center_y, width, height = [float(value) for value in values[:4]]
        return (
            center_x - width * 0.5,
            center_y - height * 0.5,
            center_x + width * 0.5,
            center_y + height * 0.5,
        )

    left = corners(predicted)
    right = corners(expected)
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    intersection = max(x1 - x0, 0.0) * max(y1 - y0, 0.0)
    left_area = max(left[2] - left[0], 0.0) * max(left[3] - left[1], 0.0)
    right_area = max(right[2] - right[0], 0.0) * max(right[3] - right[1], 0.0)
    return intersection / max(left_area + right_area - intersection, 1e-12)


def text_box_error_record(
    predicted: torch.Tensor,
    expected: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, float]:
    predicted = predicted.detach().float().cpu()
    expected = expected.detach().float().cpu()
    reference = reference.detach().float().cpu()
    predicted_center = predicted[:2]
    expected_center = expected[:2]
    center_error = torch.linalg.vector_norm(predicted_center - expected_center)
    reference_distance = torch.linalg.vector_norm(expected_center - reference[:2])
    predicted_reference_distance = torch.linalg.vector_norm(
        predicted_center - reference[:2]
    )
    return {
        "iou": text_box_iou_from_geometry(predicted, expected),
        "center_error_q": float(center_error) * COORDINATE_MAX,
        "width_error_q": abs(float(predicted[2] - expected[2])) * COORDINATE_MAX,
        "height_error_q": abs(float(predicted[3] - expected[3])) * COORDINATE_MAX,
        "target_center_to_reference_q": float(reference_distance) * COORDINATE_MAX,
        "predicted_center_to_reference_q": float(predicted_reference_distance)
        * COORDINATE_MAX,
        "predicted_width_q": float(predicted[2]) * COORDINATE_MAX,
        "predicted_height_q": float(predicted[3]) * COORDINATE_MAX,
        "target_width_q": float(expected[2]) * COORDINATE_MAX,
        "target_height_q": float(expected[3]) * COORDINATE_MAX,
    }


def _correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.std() == 0.0 or right_array.std() == 0.0:
        return None
    return float(np.corrcoef(left_array, right_array)[0, 1])


def summarize_text_boxes(records: list[dict[str, float]]) -> dict[str, Any]:
    ious = [record["iou"] for record in records]
    reference_distances = [
        record["target_center_to_reference_q"] for record in records
    ]
    center_errors = [record["center_error_q"] for record in records]
    return {
        "boxes": len(records),
        "iou": distribution(ious),
        "iou_recall": {
            f"at_{threshold:g}": _ratio(sum(value >= threshold for value in ious), len(ious))
            for threshold in (0.1, 0.25, 0.5, 0.75)
        },
        "center_error_q": distribution(center_errors),
        "width_error_q": distribution(record["width_error_q"] for record in records),
        "height_error_q": distribution(record["height_error_q"] for record in records),
        "target_center_to_reference_q": distribution(reference_distances),
        "predicted_center_to_reference_q": distribution(
            record["predicted_center_to_reference_q"] for record in records
        ),
        "predicted_width_q": distribution(
            record["predicted_width_q"] for record in records
        ),
        "predicted_height_q": distribution(
            record["predicted_height_q"] for record in records
        ),
        "target_width_q": distribution(record["target_width_q"] for record in records),
        "target_height_q": distribution(
            record["target_height_q"] for record in records
        ),
        "reference_distance_center_error_correlation": _correlation(
            reference_distances, center_errors
        ),
        "reference_distance_iou_correlation": _correlation(reference_distances, ious),
    }


def _row_from_query(
    output: dict[str, torch.Tensor],
    probabilities: torch.Tensor,
    batch_index: int,
    query_index: int,
    primitive_type: int,
) -> dict[str, Any]:
    if primitive_type == TEXT_BOX:
        geometry = output.get("text_geometry", output["geometry"])[
            batch_index, query_index
        ]
    elif primitive_type == LINE and "line_geometry" in output:
        geometry = output["line_geometry"][batch_index, query_index]
    else:
        geometry = output["geometry"][batch_index, query_index]
    return {
        "query_index": int(query_index),
        "score": float(probabilities[query_index, primitive_type]),
        "type": int(primitive_type),
        "geometry": geometry.detach(),
        "stroke": float(output["stroke"][batch_index, query_index]),
    }


def _predicted_type_rows(
    output: dict[str, torch.Tensor],
    probabilities: torch.Tensor,
    batch_index: int,
    query_indices: Iterable[int],
) -> list[dict[str, Any]]:
    rows = []
    for query_index in query_indices:
        primitive_type = int(probabilities[int(query_index), 1:].argmax()) + 1
        rows.append(
            _row_from_query(
                output,
                probabilities,
                batch_index,
                int(query_index),
                primitive_type,
            )
        )
    return rows


def _assignment_rows(
    output: dict[str, torch.Tensor],
    probabilities: torch.Tensor,
    targets: dict[str, torch.Tensor],
    batch_index: int,
    assignment: tuple[torch.Tensor, torch.Tensor],
) -> list[dict[str, Any]]:
    query_indices, target_indices = assignment
    return [
        _row_from_query(
            output,
            probabilities,
            batch_index,
            int(query_index),
            int(targets["types"][batch_index, target_index]),
        )
        for query_index, target_index in zip(query_indices, target_indices)
    ]


def _evaluate_rows(
    batch: dict[str, Any],
    batch_index: int,
    rows: list[dict[str, Any]],
    decode_mode: str,
) -> dict[str, Any]:
    return evaluate_decoded_sample(
        sample_key=batch["keys"][batch_index],
        decode_mode=decode_mode,
        threshold=None,
        decoded_rows=rows,
        expected_sequence=batch["target_sequences"][batch_index],
        bond_layer=batch["bond_layers"][batch_index],
        line_tolerance=8.0,
        text_box_iou=0.5,
    )


def _matcher_accuracy(
    output: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    assignments: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[int, int, Counter]:
    predicted_types = output["type_logits"].argmax(dim=-1)
    correct = 0
    total = 0
    by_type: Counter = Counter()
    for batch_index, (query_indices, target_indices) in enumerate(assignments):
        expected = targets["types"][batch_index, target_indices].long()
        predicted = predicted_types[batch_index, query_indices]
        correct_mask = predicted == expected
        correct += int(correct_mask.sum())
        total += len(expected)
        for type_id in CLASS_IDS:
            mask = expected == type_id
            by_type[(type_id, "correct")] += int((correct_mask & mask).sum())
            by_type[(type_id, "total")] += int(mask.sum())
    return correct, total, by_type


def _summarize_matcher(correct: int, total: int, by_type: Counter) -> dict[str, Any]:
    return {
        "matched_type_accuracy": _ratio(correct, total),
        "matched_type_correct": correct,
        "matched_type_total": total,
        "matched_type_accuracy_by_type": {
            CLASS_NAMES[type_id]: _ratio(
                by_type[(type_id, "correct")], by_type[(type_id, "total")]
            )
            for type_id in CLASS_IDS
        },
    }


def _score_summary(score_groups: dict[str, list[float]]) -> dict[str, Any]:
    return {name: distribution(values) for name, values in score_groups.items()}


def audit_checkpoint(
    *,
    label: str,
    checkpoint_path: Path,
    shards: tuple[str, ...],
    device: torch.device,
    limit: int | None,
    batch_size: int,
    workers: int,
    image_size: int,
) -> dict[str, Any]:
    model, checkpoint = load_primitive_set_model(checkpoint_path, device)
    training_args = checkpoint.get("training_args")
    training_args = training_args if isinstance(training_args, dict) else {}
    class_cost_weight = float(training_args.get("class_cost_weight", 2.0))
    line_geometry_objective = bool(training_args.get("line_geometry_objective", False))
    dataset = VisualSVGWebDataset(list(shards), limit=limit)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        collate_fn=PrimitiveSetEvaluationCollator(image_size=image_size),
    )

    selector_counts: Counter = Counter()
    selector_by_type: Counter = Counter()
    score_groups: dict[str, list[float]] = defaultdict(list)
    rank_groups: dict[int, list[float]] = defaultdict(list)
    class_count_errors: dict[int, list[float]] = defaultdict(list)
    text_records: list[dict[str, float]] = []
    standard_rows: list[dict[str, Any]] = []
    class_oracle_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    matcher_totals = Counter()
    matcher_by_type = Counter()
    geometry_matcher_totals = Counter()
    geometry_matcher_by_type = Counter()
    assignment_query_overlap = 0
    assignment_query_total = 0
    assignment_target_agreement = 0
    samples = 0

    with torch.inference_mode():
        for batch in loader:
            for key, meta in zip(batch["keys"], batch["meta"]):
                split = str(meta.get("split", meta.get("source_split", ""))).lower().replace(
                    "-", "_"
                )
                output_shard = str(meta.get("output_shard", "")).lower().replace(
                    "-", "_"
                )
                if split == "internal_test" or "internal_test" in output_shard:
                    raise RuntimeError(f"Refusing sealed internal_test sample: {key}")
            pixels = batch["pixel_values"].to(device, non_blocking=True)
            targets = targets_to_device(batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                output = model(pixels)
            probabilities_batch = output["type_logits"].float().softmax(dim=-1)
            assignments = primitive_set_assignments(
                output,
                targets,
                class_cost_weight=class_cost_weight,
                line_geometry_objective=line_geometry_objective,
            )
            geometry_assignments = primitive_set_assignments(
                output,
                targets,
                class_cost_weight=0.0,
                line_geometry_objective=line_geometry_objective,
            )
            correct, total, by_type = _matcher_accuracy(output, targets, assignments)
            matcher_totals["correct"] += correct
            matcher_totals["total"] += total
            matcher_by_type.update(by_type)
            correct, total, by_type = _matcher_accuracy(
                output, targets, geometry_assignments
            )
            geometry_matcher_totals["correct"] += correct
            geometry_matcher_totals["total"] += total
            geometry_matcher_by_type.update(by_type)

            for batch_index, (assignment, geometry_assignment) in enumerate(
                zip(assignments, geometry_assignments)
            ):
                probabilities = probabilities_batch[batch_index]
                object_scores = probabilities[:, 1:].amax(dim=-1)
                no_object_scores = probabilities[:, NO_OBJECT]
                margins = object_scores - no_object_scores
                ranking = torch.argsort(margins, descending=True, stable=True)
                inverse_rank = torch.empty_like(ranking)
                inverse_rank[ranking] = torch.arange(
                    1, len(ranking) + 1, device=ranking.device
                )
                predicted_count = int(output["count_logits"][batch_index].argmax())
                target_indices = torch.where(targets["target_mask"][batch_index])[0]
                target_count = len(target_indices)
                selected = global_topk_indices(probabilities, predicted_count)
                selected_set = set(int(index) for index in selected)
                query_indices, matched_target_indices = assignment
                matched_set = set(int(index) for index in query_indices)
                geometry_queries, geometry_targets = geometry_assignment
                geometry_mapping = {
                    int(target_index): int(query_index)
                    for query_index, target_index in zip(
                        geometry_queries, geometry_targets
                    )
                }
                assignment_mapping = {
                    int(target_index): int(query_index)
                    for query_index, target_index in zip(
                        query_indices, matched_target_indices
                    )
                }
                assignment_query_overlap += len(
                    set(assignment_mapping.values()) & set(geometry_mapping.values())
                )
                assignment_query_total += len(assignment_mapping)
                assignment_target_agreement += sum(
                    assignment_mapping.get(target_index)
                    == geometry_mapping.get(target_index)
                    for target_index in assignment_mapping
                )

                selector_counts["samples"] += 1
                selector_counts["target_total"] += target_count
                selector_counts["predicted_total"] += predicted_count
                selector_counts["count_exact"] += predicted_count == target_count
                selector_counts["matched_total"] += len(matched_set)
                retained = matched_set & selected_set
                dropped = matched_set - selected_set
                false_selected = selected_set - matched_set
                selector_counts["matched_retained"] += len(retained)
                selector_counts["matched_dropped"] += len(dropped)
                selector_counts["unmatched_selected"] += len(false_selected)
                selector_counts["selection_set_exact"] += selected_set == matched_set

                predicted_types = probabilities[:, 1:].argmax(dim=-1) + 1
                class_counts = Counter(
                    int(targets["types"][batch_index, index])
                    for index in target_indices
                )
                selected_class_counts = Counter(
                    int(predicted_types[index]) for index in selected
                )
                for type_id in CLASS_IDS:
                    class_count_errors[type_id].append(
                        abs(selected_class_counts[type_id] - class_counts[type_id])
                    )

                for query_index, target_index in zip(
                    query_indices, matched_target_indices
                ):
                    query = int(query_index)
                    target = int(target_index)
                    type_id = int(targets["types"][batch_index, target])
                    kept = query in selected_set
                    selector_by_type[(type_id, "total")] += 1
                    selector_by_type[(type_id, "retained")] += kept
                    selector_by_type[(type_id, "dropped")] += not kept
                    rank_groups[type_id].append(float(inverse_rank[query]))
                    score_groups["matched_all_margin"].append(float(margins[query]))
                    score_groups["matched_all_object_probability"].append(
                        float(object_scores[query])
                    )
                    score_groups["matched_retained_margin" if kept else "matched_dropped_margin"].append(
                        float(margins[query])
                    )
                    score_groups[
                        f"matched_{CLASS_NAMES[type_id]}_margin"
                    ].append(float(margins[query]))
                    if type_id == TEXT_BOX:
                        text_records.append(
                            text_box_error_record(
                                output["text_geometry"][batch_index, query],
                                targets["geometry"][batch_index, target],
                                model.reference_points[query],
                            )
                        )

                for query in false_selected:
                    score_groups["unmatched_selected_margin"].append(
                        float(margins[query])
                    )
                    predicted_type = int(predicted_types[query])
                    selector_by_type[(predicted_type, "unmatched_selected")] += 1
                for query in set(range(probabilities.size(0))) - matched_set - selected_set:
                    score_groups["unmatched_unselected_margin"].append(
                        float(margins[query])
                    )

                standard = _predicted_type_rows(
                    output,
                    probabilities,
                    batch_index,
                    [int(index) for index in selected],
                )
                standard_rows.append(
                    _evaluate_rows(batch, batch_index, standard, "total_count_topk")
                )
                class_oracle = class_count_oracle_indices(probabilities, class_counts)
                class_oracle_decoded = [
                    _row_from_query(
                        output,
                        probabilities,
                        batch_index,
                        query_index,
                        type_id,
                    )
                    for query_index, type_id in class_oracle
                ]
                class_oracle_rows.append(
                    _evaluate_rows(
                        batch,
                        batch_index,
                        class_oracle_decoded,
                        "oracle_class_count",
                    )
                )
                assignment_decoded = _assignment_rows(
                    output,
                    probabilities,
                    targets,
                    batch_index,
                    assignment,
                )
                assignment_rows.append(
                    _evaluate_rows(
                        batch,
                        batch_index,
                        assignment_decoded,
                        "oracle_hungarian_assignment",
                    )
                )
                samples += 1
            if samples % 20 == 0:
                print(f"{label}: audited={samples}", flush=True)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if not samples:
        raise RuntimeError("Primitive Set output-head audit read no samples")

    return {
        "label": label,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "global_step": int(checkpoint.get("global_step", 0)),
        "samples": samples,
        "training_assignment": _summarize_matcher(
            matcher_totals["correct"], matcher_totals["total"], matcher_by_type
        ),
        "geometry_only_assignment": {
            **_summarize_matcher(
                geometry_matcher_totals["correct"],
                geometry_matcher_totals["total"],
                geometry_matcher_by_type,
            ),
            "query_set_overlap_with_training_assignment": _ratio(
                assignment_query_overlap, assignment_query_total
            ),
            "same_query_for_same_target_rate": _ratio(
                assignment_target_agreement, assignment_query_total
            ),
        },
        "total_count_topk_selector": {
            "count_head_total_exact_rate": _ratio(
                selector_counts["count_exact"], selector_counts["samples"]
            ),
            "target_count_mean": _ratio(
                selector_counts["target_total"], selector_counts["samples"]
            ),
            "predicted_count_mean": _ratio(
                selector_counts["predicted_total"], selector_counts["samples"]
            ),
            "matched_query_recall": _ratio(
                selector_counts["matched_retained"], selector_counts["matched_total"]
            ),
            "matched_queries_dropped": selector_counts["matched_dropped"],
            "unmatched_queries_selected": selector_counts["unmatched_selected"],
            "exact_assignment_query_set_rate": _ratio(
                selector_counts["selection_set_exact"], selector_counts["samples"]
            ),
            "by_type": {
                CLASS_NAMES[type_id]: {
                    "matched": selector_by_type[(type_id, "total")],
                    "retained": selector_by_type[(type_id, "retained")],
                    "dropped": selector_by_type[(type_id, "dropped")],
                    "retention_rate": _ratio(
                        selector_by_type[(type_id, "retained")],
                        selector_by_type[(type_id, "total")],
                    ),
                    "unmatched_selected_as_type": selector_by_type[
                        (type_id, "unmatched_selected")
                    ],
                    "class_count_exact_rate": _ratio(
                        sum(error == 0 for error in class_count_errors[type_id]),
                        len(class_count_errors[type_id]),
                    ),
                    "class_count_mae": _ratio(
                        sum(class_count_errors[type_id]),
                        len(class_count_errors[type_id]),
                    ),
                    "global_rank": distribution(rank_groups[type_id]),
                }
                for type_id in CLASS_IDS
            },
            "score_distributions": _score_summary(score_groups),
        },
        "text_box_assignment_geometry": summarize_text_boxes(text_records),
        "decode_comparison": {
            "total_count_topk": summarize_prediction_rows(standard_rows),
            "oracle_class_count": summarize_prediction_rows(class_oracle_rows),
            "oracle_hungarian_assignment": summarize_prediction_rows(assignment_rows),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=named_checkpoint,
        required=True,
        help="Repeat LABEL=PATH to compare checkpoints",
    )
    parser.add_argument("--shard", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.batch_size < 1 or args.workers < 0 or args.image_size < 1:
        raise ValueError("Batch/image size must be positive and workers nonnegative")
    labels = [label for label, _ in args.checkpoint]
    if len(labels) != len(set(labels)):
        raise ValueError("Checkpoint labels must be unique")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    shards = validate_shards(args.shard)
    output_path = Path(args.output).expanduser().resolve()
    if "internal_test" in str(output_path).lower().replace("-", "_"):
        raise ValueError(f"internal_test is sealed: {output_path}")
    if output_path.exists():
        raise RuntimeError(f"Refusing to overwrite existing audit: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    results = [
        audit_checkpoint(
            label=label,
            checkpoint_path=path,
            shards=shards,
            device=device,
            limit=args.limit,
            batch_size=args.batch_size,
            workers=args.workers,
            image_size=args.image_size,
        )
        for label, path in args.checkpoint
    ]
    payload = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "shards": list(shards),
        "uses_internal_test": False,
        "results": results,
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
