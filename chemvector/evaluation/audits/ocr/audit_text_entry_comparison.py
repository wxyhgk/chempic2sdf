"""Freeze, run, or score a small independent-text-entry checkpoint comparison."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import chemvector.external_ocr_r0_contracts as contract
from chemvector.text_entry_metrics import box_arrays, decide, evaluate_image, summarize
from chemvector.text_entry_protocol import ARMS, freeze, load_protocol


def validate_predictions(protocol, receipt, predictions):
    if len(predictions) != receipt["rows"]:
        raise ValueError("Prediction row count differs")
    if receipt["sample_keys"] != sorted(protocol["sample_metadata"]):
        raise ValueError("Prediction sample population differs")
    if (not receipt["detector_repeat_exact"] or receipt["training_performed"]
            or receipt["scoring_truth_used_for_proposals"]
            or receipt["models_before_sha256"] != receipt["models_after_sha256"]):
        raise ValueError("Prediction receipt violates frozen inference contract")
    seen = set()
    for row in predictions:
        if row["object_key"] in seen:
            raise ValueError("Duplicate prediction identity")
        seen.add(row["object_key"])
        if row["sample_key"] not in protocol["sample_metadata"] or row["arm"] not in ARMS:
            raise ValueError("Prediction outside frozen population or arms")
        box_arrays([row["bbox"]])
        if not isinstance(row["text"], str):
            raise ValueError("Prediction text must be a string")


def score(output_dir):
    from chemvector.external_ocr_r0_truth import rendered_truth_targets
    from chemvector.text_entry_predictions import selected_images

    output = Path(output_dir)
    if any((output / name).exists() for name in ("targets.jsonl", "samples.jsonl", "report.json")):
        raise FileExistsError("Scoring outputs already exist")
    protocol = load_protocol(output)
    receipt = json.loads((output / "prediction_receipt.json").read_text())
    contract.validate_frozen_file(
        output / "protocol.json", receipt["protocol_sha256"], "prediction protocol"
    )
    contract.validate_frozen_file(
        output / "predictions.jsonl", receipt["predictions_sha256"], "predictions"
    )
    predictions = contract.read_jsonl(output / "predictions.jsonl")
    validate_predictions(protocol, receipt, predictions)
    grouped = defaultdict(list)
    for row in predictions:
        grouped[(row["sample_key"], row["arm"])].append(row)
    records, targets_out, sensitivity = [], [], defaultdict(list)
    for sample in selected_images(protocol):
        key = sample["key"]
        metadata = protocol["sample_metadata"][key]
        layers = {int(atom["atom_id"]): atom for atom in sample["layers"]["atoms"]}
        targets = []
        for index, target in enumerate(rendered_truth_targets(sample)):
            if target["visible"]:
                targets.append({"gt_key": f"{key}:t{index:03d}",
                                "truth": target["truth"],
                                "bbox": layers[target["source_atom_id"]]["bbox"]})
        paired = {}
        for arm in ARMS:
            current, per_target = evaluate_image(
                key, metadata["variant"], grouped[(key, arm)], targets,
                protocol["metrics"]["primary_iou"],
            )
            current.update(arm=arm, molecule_id=metadata["molecule_id"])
            records.append(current)
            paired[arm] = per_target
            for threshold in protocol["metrics"]["sensitivity_iou"]:
                summary, _ = evaluate_image(
                    key, metadata["variant"], grouped[(key, arm)], targets, threshold
                )
                sensitivity[(arm, threshold)].append(summary)
        for index, target in enumerate(targets):
            targets_out.append({**target, "sample_key": key,
                                "molecule_id": metadata["molecule_id"],
                                "variant": metadata["variant"],
                                "arms": {arm: paired[arm][index] for arm in ARMS}})
    arms = {arm: summarize([row for row in records if row["arm"] == arm]) for arm in ARMS}
    variants = {variant: {
        arm: summarize([row for row in records if row["arm"] == arm
                        and row["variant"] == variant]) for arm in ARMS
    } for variant in sorted({row["variant"] for row in records})}
    transitions = {}
    for metric in ("matched", "grounded_exact"):
        transitions[metric] = {
            "both": sum(row["arms"][ARMS[0]][metric] and row["arms"][ARMS[1]][metric]
                        for row in targets_out),
            "independent_only": sum(not row["arms"][ARMS[0]][metric]
                                    and row["arms"][ARMS[1]][metric] for row in targets_out),
            "baseline_only": sum(row["arms"][ARMS[0]][metric]
                                 and not row["arms"][ARMS[1]][metric] for row in targets_out),
            "neither": sum(not row["arms"][ARMS[0]][metric]
                           and not row["arms"][ARMS[1]][metric] for row in targets_out),
        }
    targets_out.sort(key=lambda row: row["gt_key"])
    records.sort(key=lambda row: (row["sample_key"], row["arm"]))
    for filename, rows in (("targets.jsonl", targets_out), ("samples.jsonl", records)):
        contract.write_new_jsonl(output / filename, rows)
    report = {
        "schema_version": protocol["schema_version"], "arms": arms, "variants": variants,
        "paired_transitions": transitions,
        "iou_sensitivity": {str(threshold): {
            arm: summarize(sensitivity[(arm, threshold)]) for arm in ARMS
        } for threshold in protocol["metrics"]["sensitivity_iou"]},
        "decision": decide(arms, variants, protocol["gates"]),
        "limitations": protocol["limitations"],
        "population": {"molecules": protocol["molecules"],
                       "images": len(protocol["sample_metadata"]),
                       "text_objects": len(targets_out)},
        "artifacts": {filename: contract.sha256_file(output / filename) for filename in (
            "protocol.json", "prediction_receipt.json", "predictions.jsonl",
            "targets.jsonl", "samples.jsonl",
        )},
        "finished_utc": contract.utc_now(),
    }
    contract.write_new_json(output / "report.json", report)
    return {"report": str(output / "report.json"), "decision": report["decision"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "predict", "summarize"))
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.phase == "freeze":
        result = freeze(args.output_dir)
    elif args.phase == "predict":
        from chemvector.text_entry_predictions import predict
        result = predict(args.output_dir)
    else:
        result = score(args.output_dir)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
