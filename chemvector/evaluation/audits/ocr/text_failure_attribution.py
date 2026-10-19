"""Pure, identity-aware failure attribution; no model or dataset execution."""

from collections import Counter

from chemvector.text_entry_metrics import overlap_matrices


ARM = "atom_conditioned"


def geometry_evidence(predicted, target, context=4.0):
    iou, coverage = overlap_matrices([predicted], [target])
    expanded = [predicted[0] - context, predicted[1] - context,
                predicted[2] + context, predicted[3] + context]
    _, crop_coverage = overlap_matrices([expanded], [target])
    return {"iou": float(iou[0, 0]), "gt_box_coverage": float(coverage[0, 0]),
            "context_gt_box_coverage": float(crop_coverage[0, 0]),
            "context_misses_gt_box_area": bool(crop_coverage[0, 0] < 0.99)}


def locate_box_failure(target, ladder, atoms, predictions, threshold=-0.25):
    """Report the last observed reachable stage, not a causal intervention result."""
    stages = ladder["stages"]
    sample = target["sample_key"]
    deployed = stages["deployed"]
    if deployed["matched"]:
        key = (sample, deployed["proposal_index"])
        if key not in atoms:
            raise ValueError("Deployed proposal identity missing from frozen atoms")
        atom = atoms[key]
        if not atom["selected_as_text"]:
            return "role_rejected", None
        prediction_key = "A:" + atom["crop_object_key"]
        if prediction_key not in predictions:
            raise ValueError("Selected TEXT proposal missing its frozen crop")
        proposal = predictions[prediction_key]
        evidence = geometry_evidence(proposal["bbox"], target["bbox"])
        reason = ("bbox_misaligned" if evidence["iou"] < 0.5
                  else "bbox_assignment_competition")
        return reason, proposal
    decoded = stages["decoded"]
    if decoded["matched"]:
        survives = (sample, decoded["proposal_index"]) in atoms
        if decoded["presence_logit"] < threshold:
            if survives:
                raise ValueError("Below-threshold decoded candidate unexpectedly deployed")
            return "set_rejected", None
        if not survives:
            raise ValueError("Above-threshold decoded candidate absent from deployment")
        return "anchor_assignment_competition", None
    if stages["raw_top128"]["matched"]:
        return "decode_or_assignment_loss", None
    if stages["all_local_peaks"]["matched"]:
        return "raw_capacity_or_assignment_loss", None
    if stages["dense_anchors"]["matched"]:
        return "local_peak_unreachable", None
    return "anchor_field_unreachable", None


def attribute_failures(targets, ladder, atoms, predictions, peaks):
    failures = []
    for target in targets:
        current = target["arms"][ARM]
        if current["grounded_exact"]:
            continue
        key = target["gt_key"]
        reach = ladder[key]
        if any(target[field] != reach[field] for field in
               ("gt_key", "sample_key", "molecule_id", "variant", "truth")):
            raise ValueError("Comparison/ladder target identity differs")
        if reach["role"] != "text":
            raise ValueError("Visible text target has non-text ladder role")
        if current["matched"]:
            category = "matched_ocr_error"
            proposal = predictions[current["prediction_key"]]
        else:
            category, proposal = locate_box_failure(target, reach, atoms, predictions)
        geometry = geometry_evidence(proposal["bbox"], target["bbox"]) if proposal else None
        if current["matched"] and (proposal["text"] != current["predicted_text"]
                                   or abs(geometry["iou"] - current["iou"]) > 1e-9):
            raise ValueError("Frozen matched OCR/geometry differs")
        failures.append({
            "gt_key": key, "sample_key": target["sample_key"],
            "molecule_id": target["molecule_id"], "variant": target["variant"],
            "truth": target["truth"], "gt_bbox": target["bbox"],
            "category": category, "baseline": current,
            "associated_prediction": proposal, "geometry": geometry,
            "ladder": reach["stages"], "anchor": reach["anchor"],
            "peak_diagnostics": peaks[key]["stages"],
            "independent_grounded_exact": target["arms"]["independent_text"]["grounded_exact"],
        })
    return sorted(failures, key=lambda row: row["gt_key"])


def summarize_failures(rows, oracle):
    expected = {r["gt_key"] for r in rows if r["category"] == "matched_ocr_error"}
    if set(oracle) != expected:
        raise ValueError("Oracle crop population differs from matched OCR failures")
    for row in rows:
        if row["gt_key"] in oracle:
            result = oracle[row["gt_key"]]
            if result["baseline_text"] != row["baseline"]["predicted_text"]:
                raise ValueError("Baseline crop replay differs")
            if result["truth"] != row["truth"]:
                raise ValueError("Oracle truth identity differs")
    ocr_rows = [r for r in rows if r["gt_key"] in oracle]
    fixed = [r for r in ocr_rows if oracle[r["gt_key"]]["oracle_text"] == r["truth"]]
    return {
        "failures": len(rows),
        "category_counts": dict(sorted(Counter(r["category"] for r in rows).items())),
        "by_variant": {variant: dict(sorted(Counter(
            r["category"] for r in rows if r["variant"] == variant).items()))
            for variant in sorted({r["variant"] for r in rows})},
        "ocr_counterfactual": {
            "targets": len(ocr_rows), "oracle_exact": len(fixed),
            "oracle_still_wrong": len(ocr_rows) - len(fixed),
            "context_misses_gt_box_area": sum(r["geometry"]["context_misses_gt_box_area"]
                                                for r in ocr_rows),
            "fixed_with_context_missing_gt_box_area": sum(
                r["geometry"]["context_misses_gt_box_area"] for r in fixed),
            "changed_string": sum(oracle[r["gt_key"]]["oracle_text"]
                                  != r["baseline"]["predicted_text"] for r in ocr_rows),
        },
        "confusions": [{"truth": truth, "predicted": predicted, "count": count}
                       for (truth, predicted), count in Counter(
                           (r["truth"], r["baseline"]["predicted_text"])
                           for r in ocr_rows).most_common()],
        "local_peak_misses_dense384_reachable": sum(
            r["category"] == "local_peak_unreachable"
            and r["peak_diagnostics"]["a3_bf16_dense_top384"]["matched"] for r in rows),
        "training_authorized": False, "promotion_authorized": False,
        "limitations": [
            "last observed failure stage is not an isolated causal effect",
            "6px anchor matching and IoU>=0.5 bbox matching are different contracts",
            "GT crops are diagnostic only, never deployment proposals",
            "GT crop response changes crop scale and context as well as truncation",
            "GT bbox coverage is not measured glyph-ink or neighboring-bond coverage",
            "persistent oracle errors do not prove recognizer-only causation",
            "no SDF graph recovery claim; no new OCR model comparison",
            "reused fresh-dev failures, no unseen confirmation",
        ],
    }
