"""Public identity screening for the fixed structure batch; no private query.

Reuse the immutable scanner and whole-candidate union. Bind public inputs in a
sidecar; never rewrite the already reviewed private request or its candidates.
"""

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time

from rdkit import rdBase

from chemvector.training.curriculum import public_review, private_review
from chemvector.training.curriculum.runtime import ResourceBudget, run_budgeted
from tools import review_structure_expansion_identity as fixed
from tools.review_curriculum_v02_public_identity import combine

BENCHMARK = fixed.REPO / "Data/benchmarks/public_ocsr_v1"
CACHE = fixed.REPO / "Data/sdf/chemvector_joint_v1/work/old_overlap_index.json"
OUTPUT = fixed.REPO / "outputs/audits/structure-expansion-public-identity-20260923"
REQUEST_SHA = "eae7f532ff6e5df14b12d36dc7b34530441466f1a6611ee5b6992d633cecd16d"
CANDIDATES_SHA = "c05d624a0791b0a6e33a90d88e7d04719d6cf7e00de1279e567853a6443efe6b"
RECEIPT_SHA = "a96f49cd55425e4456638d9084d6b40534409e1e2d41540055fd20592a1b5f16"


def bound_inputs():
    request_bytes, candidate_bytes, request, rows = fixed.validate_fixed_request()
    fixed.require(sha256(request_bytes).hexdigest() == REQUEST_SHA
                  and sha256(candidate_bytes).hexdigest() == CANDIDATES_SHA,
                  "Fixed reviewed request mismatch")
    receipt_path = fixed.OUTPUT / "receipt.json"
    fixed.require(fixed.digest(receipt_path) == RECEIPT_SHA, "Saved private receipt changed")
    receipt = private_review.validate_receipt(json.loads(fixed.read(receipt_path)),
        request_bytes, candidate_bytes, fixed.WORKER_SHA)
    fixed.require(receipt["status"] in {"checked", "checked_conservative"}, "Incomplete saved receipt")
    bindings = {**request["input_sha256"], str(fixed.REQUEST / "request.json"): REQUEST_SHA,
                str(fixed.REQUEST / "candidate-identities.json"): CANDIDATES_SHA, str(receipt_path): RECEIPT_SHA}
    return rows, receipt, bindings


def run():
    fixed.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only public review required")
    fixed.require(not OUTPUT.exists() and OUTPUT.resolve() == OUTPUT, "Fresh public output required")
    rows, receipt, bindings = bound_inputs()
    for path in (CACHE, Path(__file__).absolute(), Path(public_review.__file__),
                 Path(public_review.__file__).with_name("ctab_inventory.py"),
                 fixed.REPO / "tools/review_curriculum_v02_public_identity.py",
                 fixed.REPO / "tests/data_sdf/test_structure_public_review.py",
                 *(BENCHMARK / "manifests" / (name + ".csv") for name in public_review.SUBSETS)):
        bindings[str(path)] = fixed.digest(path)
    cache = json.loads(fixed.read(CACHE))
    fixed.require(cache.get("schema_version") == "chemvector-joint-overlap-index-0.1", "Unsupported public cache")
    keys, cache_audit = public_review._cached_keys(cache, "benchmark_inchikeys")
    fixed.require(keys, "Missing valid public cache keys")
    fixed.verify_bindings(bindings)
    plan = dict(schema_version="structure-expansion-public-plan-1", input_sha256=bindings,
        request_sha256=REQUEST_SHA, candidate_identities_sha256=CANDIDATES_SHA,
        candidate_molecules=len(rows), candidate_images=sum(r["image_count"] for r in rows),
        cache_binding="separate_public_sidecar_no_private_request_changes",
        private_identity_queries=0, training_authorized=False, model_forwards=0, gpu_used=False,
        wall_budget_seconds=180, automatic_retries=0)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    fixed.write_new(OUTPUT / "plan.json", plan)
    started = time.monotonic()
    report = dict(schema_version="chemvector-public-candidate-exclusion-0.1", status="incomplete", complete=False,
        plan_sha256=fixed.digest(OUTPUT / "plan.json"), private_identity_queries=0,
        training_authorized=False, training_performed=False, gpu_used=False, model_forwards=0, optimizer_steps=0)
    try:
        with rdBase.BlockLogs():
            candidates, inventories = public_review.candidate_inventory(rows)
        index, unresolved, evidence, public_excluded, manifests = public_review.scan_public(BENCHMARK, candidates, inventories)
        fixed.require(all(value == bindings[str(BENCHMARK / "manifests" / (name + ".csv"))]
                          for name, value in manifests.items()), "Public manifest changed during scan")
        remaining, rejected, reasons = combine(candidates, public_excluded, {key[:14] for key in keys}, receipt)
        risk = sum(row["excluded_conservatively"] and row["candidate_ref"] not in reasons for row in evidence)
        fixed.require(len(remaining) + len(rejected) == len(rows) and risk == 0, "Incomplete conservative partition")
        fixed.verify_bindings(bindings)
        for name, value in (("remaining-candidates.json", remaining), ("excluded-candidates.json", rejected),
                            ("unresolved-records.json", unresolved), ("candidate-element-inventories.json", inventories)):
            fixed.write_new(OUTPUT / name, value)
        for name, records in (("public-records.jsonl", index), ("unresolved-pair-evidence.jsonl", evidence)):
            with (OUTPUT / name).open("x", encoding="utf-8") as stream:
                for row in records:
                    stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        report.update(status="screened_with_conservative_exclusions", complete=True,
            request_sha256=REQUEST_SHA, candidate_identities_sha256=CANDIDATES_SHA,
            public_manifest_sha256=manifests, cached_index_sha256=bindings[str(CACHE)], cached_key_audit=cache_audit,
            rdkit_version=rdBase.rdkitVersion, public_source_hashes_verified=len(index),
            normal_identity_records=len(index)-len(unresolved), unresolved_global_identity_records=len(unresolved),
            supplementary_native_keys_generated=sum(bool(row["native_probe"]["key"]) for row in unresolved),
            unsupported_inventory_records=sum(row["unsupported_inventory"] for row in unresolved),
            unresolved_candidate_pairs_checked=len(evidence),
            deficiency_witness_pairs=sum(row["deficiency_witness"] is not None for row in evidence),
            unresolved_risk_pairs=sum(row["excluded_conservatively"] for row in evidence), remaining_ambiguous_pairs=risk,
            candidate_molecules=len(rows), candidate_images=sum(r["image_count"] for r in rows),
            conservatively_excluded_molecules=len(rejected), conservatively_excluded_images=sum(r["image_count"] for r in rejected),
            remaining_molecules=len(remaining), remaining_images=sum(r["image_count"] for r in remaining),
            remaining_candidates_sha256=fixed.digest(OUTPUT / "remaining-candidates.json"),
            excluded_candidates_by_reason=dict(Counter(reason for row in rejected for reason in
                {item["reason"] for item in row["exclusion_reasons"]})),
            private_receipt_inheritance=dict(receipt_sha256=RECEIPT_SHA, status=receipt["status"],
                evaluation_commitment_sha256=receipt["evaluation_commitment_sha256"], method="saved allowlisted receipt only; zero new private queries"),
            images_read=False, private_sample_data_read=False, sources_modified=False, cache_rewritten=False,
            artifact_sha256={path.name: fixed.digest(path) for path in OUTPUT.iterdir() if path.is_file()},
            limitations=["Three USPTO public subsets plus positive-only legacy cache; not all historical evaluation sources.",
                "Unresolved global identities are not repaired; possible candidate matches are excluded.",
                "Public full-key/connectivity screening does not prove public scaffold disjointness.",
                "Identity screening is not training admission or a source-license grant."],
            remaining_blockers=["Fixed-model endpoint/bond diagnosis, grouped data release, sample mixture and new training budget."])
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic()-started
        fixed.write_new(OUTPUT / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args(argv)
    if args.worker:
        result = run()
        print(json.dumps({k: result[k] for k in ("status", "remaining_molecules", "remaining_images",
            "conservatively_excluded_molecules", "remaining_ambiguous_pairs", "training_authorized")}))
        return 0 if result["complete"] else 1
    result = run_budgeted([sys.executable, "-m", "tools.review_structure_expansion_public", "--worker"],
        output=OUTPUT.with_name(OUTPUT.name + "-runtime"), budget=ResourceBudget(180))
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
