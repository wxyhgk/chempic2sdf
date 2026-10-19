"""Text-blind bbox assignment and shared-OCR metrics; no model execution."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def box_arrays(boxes):
    array = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if not np.isfinite(array).all() or np.any(array[:, 2:] <= array[:, :2]):
        raise ValueError("Boxes must be finite and have positive extent")
    return array


def overlap_matrices(predicted, targets):
    predicted, targets = box_arrays(predicted), box_arrays(targets)
    low = np.maximum(predicted[:, None, :2], targets[None, :, :2])
    high = np.minimum(predicted[:, None, 2:], targets[None, :, 2:])
    intersection = np.maximum(high - low, 0).prod(axis=-1)
    predicted_area = (predicted[:, 2:] - predicted[:, :2]).prod(axis=-1)
    target_area = (targets[:, 2:] - targets[:, :2]).prod(axis=-1)
    union = predicted_area[:, None] + target_area[None, :] - intersection
    return intersection / union, intersection / target_area[None, :]


def match_boxes(predicted, targets, threshold=0.5):
    if not 0 < threshold <= 1:
        raise ValueError("IoU threshold must be in (0, 1]")
    overlaps, coverage = overlap_matrices(predicted, targets)
    if not overlaps.size:
        return {}, overlaps
    feasible = overlaps >= threshold
    penalty = min(overlaps.shape) + 1.0
    costs = np.where(feasible, 1.0 - overlaps, penalty)
    candidate_ids, target_ids = linear_sum_assignment(costs)
    matches = {
        int(target): {"prediction_index": int(candidate),
                      "iou": float(overlaps[candidate, target]),
                      "coverage": float(coverage[candidate, target])}
        for candidate, target in zip(candidate_ids, target_ids)
        if feasible[candidate, target]
    }
    return matches, overlaps


def evaluate_image(sample_key, variant, predictions, targets, threshold=0.5):
    matches, overlaps = match_boxes(
        [row["bbox"] for row in predictions], [row["bbox"] for row in targets],
        threshold,
    )
    target_rows = []
    for index, target in enumerate(targets):
        match = matches.get(index)
        prediction = predictions[match["prediction_index"]] if match else None
        target_rows.append({
            "gt_key": target["gt_key"], "truth": target["truth"],
            "matched": match is not None,
            "prediction_key": prediction["object_key"] if prediction else None,
            "predicted_text": prediction["text"] if prediction else None,
            "grounded_exact": bool(prediction and prediction["text"] == target["truth"]),
            "iou": match["iou"] if match else None,
            "coverage": match["coverage"] if match else None,
        })
    assigned = {match["prediction_index"] for match in matches.values()}
    negatives = [index for index in range(len(predictions)) if index not in assigned]
    duplicates = sum(bool(overlaps.shape[1] and overlaps[index].max() >= threshold)
                     for index in negatives)
    sample = {
        "sample_key": sample_key, "variant": variant, "targets": len(targets),
        "predictions": len(predictions), "matched": len(matches),
        "grounded_exact": sum(row["grounded_exact"] for row in target_rows),
        "covered_90": sum(match["coverage"] >= 0.9 for match in matches.values()),
        "unmatched_predictions": len(negatives), "duplicate_predictions": duplicates,
        "unmatched_nonempty": sum(bool(predictions[index]["text"]) for index in negatives),
        "all_text_exact": (len(matches) == len(targets) == len(predictions)
                           and all(row["grounded_exact"] for row in target_rows)),
    }
    return sample, target_rows


def summarize(rows):
    counts = {key: sum(row[key] for row in rows) for key in (
        "targets", "predictions", "matched", "grounded_exact", "covered_90",
        "unmatched_predictions", "duplicate_predictions", "unmatched_nonempty",
        "all_text_exact",
    )}
    def divide(numerator, denominator):
        return numerator / denominator if denominator else None
    return {**counts, "samples": len(rows),
            "box_recall": divide(counts["matched"], counts["targets"]),
            "box_precision": divide(counts["matched"], counts["predictions"]),
            "grounded_exact_rate": divide(counts["grounded_exact"], counts["targets"]),
            "conditional_ocr_exact": divide(counts["grounded_exact"], counts["matched"]),
            "predictions_per_image": divide(counts["predictions"], len(rows)),
            "unmatched_per_image": divide(counts["unmatched_predictions"], len(rows)),
            "all_text_exact_rate": divide(counts["all_text_exact"], len(rows))}


def decide(arms, variants, gates):
    baseline, independent = arms["atom_conditioned"], arms["independent_text"]
    if "stress" not in variants or any(
        summary[field] is None
        for summary in (baseline, independent, *variants["stress"].values())
        for field in ("grounded_exact_rate", "unmatched_per_image")
    ):
        raise ValueError("Diagnostic gates require nonempty overall and stress targets")
    delta = 100 * (independent["grounded_exact_rate"] - baseline["grounded_exact_rate"])
    precision_ok = (independent["box_precision"] is not None
                    and baseline["box_precision"] is not None
                    and independent["box_precision"] >= baseline["box_precision"]
                    - gates["precision_regression_pp_maximum"] / 100)
    extra = independent["unmatched_per_image"] - baseline["unmatched_per_image"]
    stress = variants["stress"]
    stress_ok = (stress["independent_text"]["grounded_exact_rate"]
                 >= stress["atom_conditioned"]["grounded_exact_rate"])
    passed = (delta >= gates["grounded_exact_delta_pp_minimum"] and precision_ok
              and extra <= gates["extra_boxes_per_image_increase_maximum"] and stress_ok)
    return {"grounded_exact_delta_pp": delta, "precision_guard_passed": precision_ok,
            "extra_unmatched_per_image": extra, "stress_guard_passed": stress_ok,
            "diagnostic_gate_passed": passed,
            "conclusion": ("independent_checkpoint_warrants_unseen_confirm" if passed
                           else "historical_independent_checkpoint_not_sufficient"),
            "architecture_verdict": "not_identified_by_different_checkpoint_comparison",
            "training_authorized": False, "promotion_authorized": False}
