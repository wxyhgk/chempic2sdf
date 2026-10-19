#!/usr/bin/env python3
"""Read-only receipt for the r17 rollback CPU shadow.

This does not reroute predictions or train.  It joins the already completed
normal and target-perturbed rollback journals, verifies decision invariance,
and exposes per-profile graph deltas for a later integration review.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.train_direct_graph_pair_scorer_cpu_r9 import SHA

NORMAL = ROOT / "outputs/audits/chemvector-r17-target-free-rollback-m64-v2/report.json"
PERTURBED = ROOT / "outputs/audits/chemvector-r17-target-free-rollback-m64-v3-perturb/report.json"
INITIAL = ROOT / "outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt"
OUT = ROOT / "outputs/audits/chemvector-r17-rollback-shadow-receipt-m64-v2"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decision_journal(report: dict) -> dict[str, tuple]:
    return {
        row["sample_key"]: (
            row["split"], row["baseline_error"], row["rollback_error"],
            row["attempts"], row["transactions"], row["abstain"],
        )
        for row in report["rows"]
    }


def journal_hash(journal: dict[str, tuple]) -> str:
    payload = json.dumps(journal, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def run() -> dict:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    if digest(INITIAL) != SHA:
        raise RuntimeError("production checkpoint changed")
    normal = json.loads(NORMAL.read_text())
    perturbed = json.loads(PERTURBED.read_text())
    first = decision_journal(normal); second = decision_journal(perturbed)
    if first != second:
        raise RuntimeError("target perturbation changed rollback decisions")
    if not perturbed.get("target_perturbation_sentinel"):
        raise RuntimeError("perturbed receipt is not marked as a sentinel")
    profile_delta = {}
    for split, profiles in normal["profile_summary"].items():
        profile_delta[split] = {}
        for profile, value in profiles.items():
            base = value["baseline_edges"]; after = value["rollback_edges"]
            profile_delta[split][profile] = {
                "images": value["images"],
                "baseline_sanitize_rate": value["baseline_sanitize_rate"],
                "rollback_sanitize_rate": value["rollback_sanitize_rate"],
                "delta_sanitize": value["rollback_sanitize_rate"] - value["baseline_sanitize_rate"],
                "abstain_rate": value["abstain_rate"],
                "removed_edges": value.get("removed_edges", 0),
                "edge_precision_delta": after["edge_precision"] - base["edge_precision"],
                "edge_recall_delta": after["edge_recall"] - base["edge_recall"],
                "aromatic_precision_delta": after["aromatic_edge_precision"] - base["aromatic_edge_precision"],
                "aromatic_recall_delta": after["aromatic_edge_recall"] - base["aromatic_edge_recall"],
            }
    report = {
        "schema_version": "chemvector-r17-rollback-shadow-receipt-m64-0.1",
        "complete": True, "read_only": True, "training_performed": False,
        "gpu_used": False, "sealed_test_read": False,
        "production_checkpoint_written": False,
        "normal_report_sha256": digest(NORMAL), "perturbed_report_sha256": digest(PERTURBED),
        "decision_journal_sha256": journal_hash(first), "decision_rows": len(first),
        "profile_delta": profile_delta, "summary": normal["summary"],
        "policy": {"candidate_edge_set_frozen": True, "rollback_only_on_invalid": True,
                   "target_free_routing": True, "journal_exposed": True,
                   "production_integration": "not_authorized"},
        "initial_checkpoint_sha256": SHA,
        "initial_checkpoint_sha256_after": digest(INITIAL),
        "gate": {"checkpoint_unchanged": True, "sealed_test_read": False,
                 "target_perturbation_decisions_equal": first == second,
                 "journal_rows_768": len(first) == 768,
                 "production_go": False,
                 "next": "review shadow receipt; do not merge or start GPU"},
    }
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT / "report.json"), "journal_sha256": report["decision_journal_sha256"], "gate": report["gate"]}, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    run()
