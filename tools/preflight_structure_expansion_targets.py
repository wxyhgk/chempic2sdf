"""All retained styles: CPU-only diagnostic target compatibility, no model.

Consumes the saved identity partition without a new identity query. Reuses
the unchanged loader, collator and independent target validator. Not admission.
"""

import argparse
from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.training.curriculum.data import CurriculumCollator
from chemvector.training.curriculum.runtime import ResourceBudget, run_budgeted
from tools import preflight_curriculum_v02_targets as validator
from tools import review_structure_expansion_identity as fixed

OUTPUT = fixed.REPO / "outputs/audits/structure-expansion-all-style-targets-20260923"
PARTITION_SHA = "4b9b2b6b75c58d5898065b197cad0b8f910fe7298c3936e31bbd89f3df7d44e5"


def select_rows(samples, candidates):
    by_key = {row["sample_key"]: row for row in samples}
    fixed.require(len(by_key) == len(samples), "Duplicate sample index key")
    selected, seen, molecule_ids = [], set(), set()
    for candidate in candidates:
        mid = candidate["molecule_id"]
        fixed.require(mid not in molecule_ids and candidate["source"] == "curriculum",
                      "Duplicate/foreign candidate identity")
        molecule_ids.add(mid)
        keys = candidate["sample_keys"]
        fixed.require(len(keys) == candidate["image_count"] and keys, "Candidate image count mismatch")
        for key in keys:
            fixed.require(key not in seen and key in by_key and by_key[key]["molecule_id"] == mid,
                          "Missing/foreign/duplicate selected sample")
            seen.add(key)
            row = by_key[key]
            fixed.require(row["profile_id"] in fixed.PROFILES and key == f"{mid}__{row['profile_id']}",
                          "Invalid selected profile")
            selected.append({field: row[field] for field in
                             ("sample_key", "molecule_id", "profile_id", "png_sha256")}
                            | {"decoded_rgb_sha256": row["rgb_sha256"]})
    return sorted(selected, key=lambda row: row["sample_key"])


def summarize_journal(selection, rows):
    fixed.require(len(rows) == len(selection)
                  and [r["sample_key"] for r in rows] == [r["sample_key"] for r in selection],
                  "Incomplete or changed target journal")
    by_profile, passed_profiles = Counter(), Counter()
    for expected, row in zip(selection, rows, strict=True):
        fixed.require(all(row[field] == expected[field] for field in ("molecule_id", "profile_id"))
                      and type(row["compatible"]) is bool, "Journal identity/status mismatch")
        by_profile[row["profile_id"]] += 1
        passed_profiles[row["profile_id"]] += row["compatible"]
    passed = sum(row["compatible"] for row in rows)
    return dict(checked_images=len(rows), compatible_images=passed, failed_images=len(rows)-passed,
                checked_molecules=len({r["molecule_id"] for r in rows}),
                checked_images_by_profile=dict(by_profile), compatible_images_by_profile=dict(passed_profiles))


def run():
    fixed.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only preflight required")
    torch.set_num_threads(1)
    fixed.require(OUTPUT.resolve() == OUTPUT and not (OUTPUT / "plan.json").exists(), "Fresh preflight required")
    fixed.require(fixed.digest(fixed.AUDIT) == fixed.AUDIT_SHA
                  and fixed.digest(fixed.PARTITION / "report.json") == PARTITION_SHA,
                  "Fixed audit/identity partition changed")
    audit = json.loads(fixed.read(fixed.AUDIT))
    partition = json.loads(fixed.read(fixed.PARTITION / "report.json"))
    bindings = {str(Path(path).absolute()): value for path, value in audit["input_sha256"].items()}
    bindings.update({str(fixed.DATASET / name): value for name, value in audit["manifest_sha256"].items()})
    bindings.update({str(fixed.AUDIT): fixed.AUDIT_SHA,
        str(fixed.PARTITION / "report.json"): PARTITION_SHA,
        str(fixed.PARTITION / "remaining-candidates.json"): partition["remaining_candidates_sha256"]})
    for path in validator.source_paths() | {Path(__file__).absolute(), Path(fixed.__file__).absolute(),
            fixed.REPO / "tests/data_sdf/test_structure_target_preflight.py"}:
        path = str(path)
        actual = fixed.digest(path)
        fixed.require(path not in bindings or bindings[path] == actual, "Validator source changed")
        bindings[path] = actual
    fixed.verify_bindings(bindings)
    fixed.require(partition["training_authorized"] is False and partition["excluded_molecules"] == 0
                  and (partition["remaining_molecules"], partition["remaining_images"]) == fixed.COUNTS,
                  "Expected unchanged full fixed batch")
    selection = select_rows(json.loads(fixed.read(fixed.DATASET / "samples.json")),
        json.loads(fixed.read(fixed.PARTITION / "remaining-candidates.json")))
    fixed.require(len(selection) == fixed.COUNTS[1], "Selected count mismatch")
    hashes = json.loads(fixed.read(fixed.DATASET / "sample-sha256.json"))
    config = AtomBondSceneConfig()
    plan = dict(schema_version="structure-expansion-all-style-preflight-1", selection=selection,
        selected_images=len(selection), config=asdict(config), input_sha256=bindings,
        training_authorized=False, training_performed=False, model_forwards=0, gpu_used=False,
        external_identity_queries=0, wall_budget_seconds=600)
    fixed.write_new(OUTPUT / "plan.json", plan)
    collate = CurriculumCollator(config, purpose="diagnostic")
    start = time.monotonic()
    report = dict(schema_version=plan["schema_version"], complete=False, status="incomplete",
        plan_sha256=fixed.digest(OUTPUT / "plan.json"), training_authorized=False, training_performed=False,
        gpu_used=False, model_forwards=0, optimizer_steps=0, external_identity_queries=0,
        all_retained_styles_collated=False, predicted_candidate_reachability="not_measured",
        not_supervised_by_adapter=["OCR transcription", "element/charge/isotope classification",
                                   "functional-group classification"],
        limitations=["Target compatibility is not recognition accuracy or predicted endpoint/bond reachability.",
                     "Identity partition is only against the fixed private evaluation; public review still pending.",
                     "No training release or split; all source samples remain quarantined."])
    chemical, direction = Counter(), Counter()
    try:
        with (OUTPUT / "samples.jsonl").open("x", encoding="utf-8") as stream:
            for index, row in enumerate(selection, 1):
                result = {field: row[field] for field in ("sample_key", "molecule_id", "profile_id")}
                try:
                    sample = validator.verify_sample(fixed.DATASET, row, hashes)
                    target = validator.validate_targets(sample, collate([sample]), config)
                    result.update(compatible=True, **target)
                    chemical.update(bond["chemical_type"] for bond in sample["graph"]["bonds"])
                    direction.update(bond["drawing"]["direction"] for bond in sample["layers"]["bonds"])
                except ValueError as exc:
                    result.update(compatible=False, reason=str(exc))
                stream.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
                if index % 256 == 0:
                    stream.flush()
                    print(json.dumps({"checked_images": index, "total": len(selection)}), flush=True)
        journal = [json.loads(line) for line in fixed.read(OUTPUT / "samples.jsonl").splitlines()]
        report.update(summarize_journal(selection, journal))
        fixed.verify_bindings(bindings)
        report.update(complete=True, status="passed_target_compatibility_only" if not report["failed_images"] else "failed_targets",
            all_retained_styles_collated=report["failed_images"] == 0,
            journal_sha256=fixed.digest(OUTPUT / "samples.jsonl"), chemical_bond_instances=dict(chemical),
            drawing_direction_instances=dict(direction),
            training_loader_and_collator_rejected_quarantine=report["compatible_images"])
    finally:
        report["elapsed_seconds"] = time.monotonic() - start
        fixed.write_new(OUTPUT / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args(argv)
    if args.worker:
        result = run()
        return 0 if result["status"] == "passed_target_compatibility_only" else 1
    result = run_budgeted([sys.executable, "-m", "tools.preflight_structure_expansion_targets", "--worker"],
                          output=OUTPUT, budget=ResourceBudget(600))
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
