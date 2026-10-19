#!/usr/bin/env python3
"""Read-only training-readiness receipt for the current public artifacts."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / os.environ.get("CHEMVECTOR_READINESS_OUT", "outputs/audits/chemvector-training-readiness-20261008-r1")
INITIAL = ROOT / "outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"


def read(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text())


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run() -> dict:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    release = read("outputs/audits/chemvector-structure-expansion-10k-release-plan-20261016-r1/report.json")
    replay = read("outputs/audits/chemvector-structure-expansion-10k-frozen-replay-20261016-r3/report.json")
    raw_gpu = read("outputs/audits/chemvector-style-factorial-raw-objectness-gpu-shadow-20261010-r1/final-audit.json")
    graph_contract = read("outputs/audits/chemvector-style-factorial-raw-objectness-graph-downstream-contract-20261016-r1/report.json")
    overlap = read("outputs/audits/chemvector-r17-release-slice80-identity-overlap-v3/report.json")
    tx = read("outputs/audits/chemvector-aromatic-transaction-target-free-cpu-20261018/report.json")
    m128_type = read("outputs/audits/chemvector-r17-type-transaction-m128-v1/report.json")
    m128_type_identity = read("outputs/audits/chemvector-r17-type-transaction-m128-v1-identity-v1/report.json")
    m128_cycle = read("outputs/audits/chemvector-r17-target-free-rollback-m128-connected-cycle-v1/report.json")
    context_scope = read("outputs/audits/chemvector-r17-element-context-scope-cpu-20261008-v1/report.json")
    strict_replay = read("outputs/audits/chemvector-r17-selector-graph-replay-m128-aromatic-v4/report.json")
    clean_cohort = read("outputs/audits/chemvector-r17-selector-known-clean-shadow-cohort-v1/report.json")
    clean_sidecar = read("outputs/audits/chemvector-r17-selector-retention-known-clean-cpu-v1/report.json")
    clean_graph = read("outputs/audits/chemvector-r17-selector-graph-replay-known-clean-v1/report.json")
    core_profiles = ("oa00", "oa02", "oa11", "oa20", "oa22", "oa31")
    raw_by_profile = replay["by_profile"]
    checks = {
        "production_initial_sha_matches": digest(INITIAL) == INITIAL_SHA,
        "release_metadata_complete": bool(release.get("complete")),
        "release_historical_prior_use_ledger_complete": bool(release.get("identity_split", {}).get("complete_historical_prior_use_ledger", False)),
        "release_identity_molecule_connection_scaffold_disjoint": all(release.get("identity_split", {}).get(k) == 0 for k in ("molecule_overlap", "connection_overlap", "scaffold_overlap")),
        "aggregate_raw_reachability_ge_098": replay["totals"]["raw_reachability"] >= 0.98,
        "each_core_profile_raw_reachability_ge_098": all(raw_by_profile[p]["raw_reachability"] >= 0.98 for p in core_profiles),
        "stress_profiles_raw_reachability_ge_098": all(raw_by_profile[p]["raw_reachability"] >= 0.98 for p in ("oa13", "oa33")),
        "selected_reachability_ge_098": replay["totals"]["selected_reachability"] >= 0.98,
        "raw_objectness_frozen_gpu_shadow_checks_pass": bool(raw_gpu.get("pass")),
        "graph_downstream_artifact_available": bool(graph_contract.get("graph_artifact", {}).get("provided", False)),
        "slice80_disjoint_from_r17_plan": bool(overlap["gate"].get("release_slice_disjoint_from_r17_plan", False)),
        "slice80_disjoint_from_materialized_r17_cache": bool(overlap["gate"].get("release_slice_disjoint_from_materialized_r17_cache", False)),
        "transaction_layer_worth_training": bool(tx.get("gate", {}).get("worth_training_transaction_layer", False)),
        "m128_type_transaction_gate": bool(m128_type.get("gate", {}).get("fresh_sanitize_gain_ge_0.03", False)) and bool(m128_type_identity.get("gate", {}).get("fresh_changed_type_accuracy_non_decrease", False)) and bool(m128_type_identity.get("gate", {}).get("fresh_overall_type_accuracy_non_decrease", False)),
        "m128_cycle_transaction_gate": bool(m128_cycle.get("gate", {}).get("fresh_sanitize_gain_ge_0.03", False)),
        "element_context_scope_contract_consistent": strict_replay.get("context_scope") == "presence_admitted_edges",
        "known_clean_cohort_identity_gate": bool(clean_cohort.get("gate", {}).get("known_plan_and_cache_excluded")) and bool(clean_cohort.get("gate", {}).get("train_holdout_identity_disjoint")),
        "known_clean_sidecar_precision_gate": bool(clean_sidecar.get("gate", {}).get("fresh_precision_ge_0.85")),
        "known_clean_graph_strict_gate": all(bool(clean_graph.get("gate", {}).get(k)) for k in ("fresh_strict_edge_precision_no_drop_gt_0.01", "fresh_strict_typed_precision_no_drop_gt_0.01", "fresh_strict_typed_recall_no_drop", "fresh_sanitize_no_drop")),
    }
    result = {
        "schema_version": "chemvector-training-readiness-20261008-0.1",
        "read_only": True,
        "training_performed": False,
        "gpu_used": False,
        "sealed_test_read": False,
        "production_checkpoint_written": False,
        "inputs": {
            "release_plan": "outputs/audits/chemvector-structure-expansion-10k-release-plan-20261016-r1/report.json",
            "frozen_replay": "outputs/audits/chemvector-structure-expansion-10k-frozen-replay-20261016-r3/report.json",
            "raw_objectness_gpu_final_audit": "outputs/audits/chemvector-style-factorial-raw-objectness-gpu-shadow-20261010-r1/final-audit.json",
            "graph_downstream_contract": "outputs/audits/chemvector-style-factorial-raw-objectness-graph-downstream-contract-20261016-r1/report.json",
            "slice80_overlap": "outputs/audits/chemvector-r17-release-slice80-identity-overlap-v3/report.json",
            "aromatic_transaction": "outputs/audits/chemvector-aromatic-transaction-target-free-cpu-20261018/report.json",
            "m128_type_transaction": "outputs/audits/chemvector-r17-type-transaction-m128-v1/report.json",
            "m128_type_transaction_identity": "outputs/audits/chemvector-r17-type-transaction-m128-v1-identity-v1/report.json",
            "m128_cycle_transaction": "outputs/audits/chemvector-r17-target-free-rollback-m128-connected-cycle-v1/report.json",
            "element_context_scope_probe": "outputs/audits/chemvector-r17-element-context-scope-cpu-20261008-v1/report.json",
            "strict_replay": "outputs/audits/chemvector-r17-selector-graph-replay-m128-aromatic-v4/report.json",
            "known_clean_cohort": "outputs/audits/chemvector-r17-selector-known-clean-shadow-cohort-v1/report.json",
            "known_clean_sidecar": "outputs/audits/chemvector-r17-selector-retention-known-clean-cpu-v1/report.json",
            "known_clean_graph_replay": "outputs/audits/chemvector-r17-selector-graph-replay-known-clean-v1/report.json",
        },
        "checks": checks,
        "decision": {
            "full_10k_training_go": False,
            "gpu_training_go": False,
            "graph_transaction_training_go": False,
            "current_existing_cohort_training_go": False,
            "bounded_cpu_sidecar_protocol_defined": True,
            "bounded_cpu_sidecar_shadow_completed": checks["known_clean_cohort_identity_gate"] and checks["known_clean_sidecar_precision_gate"] and checks["known_clean_graph_strict_gate"],
            "bounded_cpu_sidecar_requires_new_identity_clean_cohort": True,
            "reason": "scope contract is repaired and the known-clean sidecar shadow passes, but full GPU/production training remains blocked by stress-profile coverage, incomplete historical prior-use ledger, missing graph downstream artifact, and the fact that the current shadow is not a deployment-authorized checkpoint",
        },
        "minimum_bounded_protocol": {
            "scope": "one isolated CPU sidecar only; no production checkpoint",
            "freeze": ["production initial", "vision", "AtomObject", "AtomPair/bond", "OCR"],
            "budget": {"max_steps": 1200, "max_wall_minutes": 30, "seeds": 1},
            "data": "new molecule/connection/scaffold-disjoint cohort after excluding release slice80 and every materialized/planned r17 identity",
            "objective": "candidate-observable raw objectness or visible O/N context; no sanitize/rollback target",
            "stress": "oa13/oa33 and stress rows evaluation-only until their raw reachability gate passes",
            "gates": ["per-core raw reachability >=0.98", "holdout precision >=0.85", "holdout strict recall >=0.50", "protected retention=1.0", "graph artifact schema present", "initial SHA unchanged"],
        },
    }
    OUT.mkdir(parents=False)
    (OUT / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT / "report.json"), "checks": checks, "decision": result["decision"]}, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    run()
