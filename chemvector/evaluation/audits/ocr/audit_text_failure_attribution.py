"""Attribute frozen text-entry failures and run a diagnostic GT-crop intervention."""

import argparse
import json
from pathlib import Path

import chemvector.external_ocr_r0_contracts as contract
from chemvector.text_entry_protocol import load_protocol, source_files
from chemvector.text_failure_attribution import attribute_failures, summarize_failures


COMPARISON = Path("outputs/evaluations/text-entry-comparison-20260909")
R0 = Path("outputs/evaluations/external-ocr-r0-predicted-bbox")
REPORTS = {
    str(COMPARISON / "report.json"): "a3fed25e12a96d8da7e0e466650bf2f7339af5ad380aab1c3221afb49b5f4f4a",
    str(R0 / "proposal-ladder/report.json"): "2948274057d39e1dd44aac9d8ed469990b6c5f4529e4357e569432c02f4dd16a",
    str(R0 / "peak-precision/report.json"): "8c62a8ff8ea5dc38f0d30b4aa71349e17f74b0f6076c86ada034e6afa66fd0f2",
}


def unique_index(rows, key):
    result = {}
    for row in rows:
        identity = key(row)
        if identity in result:
            raise ValueError("Duplicate record identity")
        result[identity] = row
    return result


def selected_records(path, keys, expected_rows):
    result, seen = {}, set()
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            identity = row["gt_key"]
            if identity in seen:
                raise ValueError("Duplicate audit target identity")
            seen.add(identity)
            if identity in keys:
                result[identity] = row
    if len(seen) != expected_rows or set(result) != keys:
        raise ValueError("Historical audit population differs")
    return result


def load_inputs():
    from chemvector.external_ocr_r0_crop_bundle import load_crop_manifest

    protocol = load_protocol(COMPARISON)
    inputs = dict(REPORTS)
    reports = []
    for path, digest in REPORTS.items():
        contract.validate_frozen_file(path, digest, "input report")
        report = json.loads(Path(path).read_text())
        reports.append(report)
        for name, artifact in report["artifacts"].items():
            source = str(COMPARISON / name) if isinstance(artifact, str) else artifact["path"]
            sha = artifact if isinstance(artifact, str) else artifact["sha256"]
            contract.validate_frozen_file(source, sha, "input artifact")
            inputs[source] = sha
    comparison, ladder_report, peak_report = reports
    root, manifest, _ = load_crop_manifest(R0 / "dataset")
    atom_path = root / manifest["artifacts"]["atom_objects"]["path"]
    inputs[str(atom_path)] = contract.sha256_file(atom_path)
    targets = contract.read_jsonl(COMPARISON / "targets.jsonl")
    target_index = unique_index(targets, lambda row: row["gt_key"])
    if len(targets) != comparison["population"]["text_objects"]:
        raise ValueError("Comparison target count differs")
    keys = set(target_index)
    ladder = selected_records(ladder_report["artifacts"]["targets"]["path"], keys,
                              ladder_report["artifacts"]["targets"]["rows"])
    peaks = selected_records(peak_report["artifacts"]["targets"]["path"], keys,
                             peak_report["artifacts"]["targets"]["rows"])
    for key in keys:
        if any(peaks[key][field] != ladder[key][field]
               for field in ("anchor", "truth", "source_atom_id", "role", "sample_key")):
            raise ValueError("Peak/ladder target identity differs")
    atoms = unique_index(contract.read_jsonl(atom_path),
                         lambda row: (row["sample_key"], row["predicted_atom_index"]))
    predictions = unique_index(contract.read_jsonl(COMPARISON / "predictions.jsonl"),
                               lambda row: row["object_key"])
    return protocol, inputs, comparison, targets, ladder, atoms, predictions, peaks


def run(output_dir):
    from chemvector.text_failure_review import crop_intervention, write_gallery

    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Attribution requires an empty output directory")
    protocol, inputs, baseline, targets, ladder, atoms, predictions, peaks = load_inputs()
    frozen = {
        "schema_version": "text-failure-attribution-v1", "frozen_utc": contract.utc_now(),
        "inputs": inputs, "sources": source_files(),
        "selection": "all A failures in the frozen 200-image comparison; no new selection",
        "expected_population": {"targets": 2147, "failures": 211,
                                "box_unmatched": 114, "matched_ocr_errors": 97},
        "set_logit_threshold": -0.25, "bbox_iou_threshold": 0.5,
        "anchor_radius_px": 6.0, "context_px": 4.0,
        "ocr_intervention": "matched OCR failures only: frozen A crop vs GT bbox +4px",
        "ocr": protocol["ocr"], "crop_size": [192, 48],
        "coverage_flag": "GT bbox area coverage after context <0.99; not ink coverage",
        "training_authorized": False, "promotion_authorized": False,
        "GT_used_for_diagnostic_crops": True, "deployment_predictions_modified": False,
    }
    contract.write_new_json(output / "protocol.json", frozen)
    rows = attribute_failures(targets, ladder, atoms, predictions, peaks)
    expected = frozen["expected_population"]
    actual = {"targets": len(targets), "failures": len(rows),
              "box_unmatched": sum(not r["baseline"]["matched"] for r in rows),
              "matched_ocr_errors": sum(r["baseline"]["matched"] for r in rows)}
    if actual != expected:
        raise ValueError("Failure population differs from frozen comparison")
    print(f"attributed={len(rows)}; categories pending independent OCR crop intervention", flush=True)
    contract.write_new_jsonl(output / "failures.jsonl", rows)
    oracle, previews, receipt = crop_intervention(output, rows, protocol)
    summary = summarize_failures(rows, oracle)
    write_gallery(output, rows, oracle, previews)
    # Revalidate both snapshots after execution, including the unchanged CTC/data chain.
    load_protocol(COMPARISON)
    for path, digest in {**inputs, **frozen["sources"]}.items():
        contract.validate_frozen_file(path, digest, "final frozen input")
    contract.write_new_json(output / "receipt.json", receipt)
    artifacts = {str(path.relative_to(output)): contract.sha256_file(path)
                 for path in sorted(output.rglob("*")) if path.is_file()}
    report = {"schema_version": frozen["schema_version"], "population": actual,
              "baseline_grounded_exact": baseline["arms"]["atom_conditioned"]["grounded_exact"],
              **summary, "artifacts": artifacts, "finished_utc": contract.utc_now()}
    contract.write_new_json(output / "report.json", report)
    return {"report": str(output / "report.json"), "categories": summary["category_counts"],
            "ocr_counterfactual": summary["ocr_counterfactual"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
