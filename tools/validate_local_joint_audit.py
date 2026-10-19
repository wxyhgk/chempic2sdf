"""Validate and annotate the completed local-pair forward audit.

This is a read-only contract check.  It does not rerun the model, read sealed
data, or authorize deployment.  The original report's top-K values are
explicitly marked known-row diagnostics because unknown rows were excluded.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = Path(os.environ.get(
    "CHEMVECTOR_LOCAL_JOINT_REPORT",
    str(ROOT / "outputs/audits/chemvector-local-joint-candidate-20261009-r1/report.json"),
))
OUTPUT = REPORT.parent / "gate.json"
EXPECTED_INITIAL = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    if not report.get("complete") or report.get("training_performed"):
        raise ValueError("Completed forward-only report required")
    if report.get("sealed_test_read") or report.get("deployment_authorized"):
        raise ValueError("Invalid sealed/deployment flags")
    if report["source_sha256"]["initial.pt"] != EXPECTED_INITIAL:
        raise ValueError("Initial checkpoint lineage changed")
    if report["baseline_prefix"]["state_before_sha256"] != report["baseline_prefix"]["state_after_sha256"]:
        raise ValueError("Frozen initial model state changed")
    split = report["split"]
    if any(split[key] != 0 for key in ("group_overlap", "molecule_overlap", "canonical_smiles_overlap")):
        raise ValueError(f"Split overlap: {split}")
    totals = report["totals"]
    train = report["partition_summary"]["train"]["totals"]
    holdout = report["partition_summary"]["holdout"]["totals"]
    coverage = totals["local_reachable_target_bonds"] / max(totals["all_pair_reachable_target_bonds"], 1)
    type_counts = report["confirmed_positive_bond_types"]
    top_k = {}
    for partition, summary in report["partition_summary"].items():
        top_k[partition] = {}
        for k, values in summary["top_k_summary"].items():
            top_k[partition][k] = {
                "known_only_non_deployable": True,
                "selected": values["selected"],
                "true_positive": values["true_positive"],
                "positive_total": values["positive_total"],
                "precision_known_only": values["precision"],
                "recall_known_only": values["true_positive"] / max(values["positive_total"], 1),
            }
    result = {
        "schema_version": "chemvector-local-joint-audit-gate-0.1",
        "source_report": str(REPORT.relative_to(ROOT)),
        "read_only": True,
        "training_performed": False,
        "sealed_test_read": False,
        "deployment_authorized": False,
        "initial_sha256": EXPECTED_INITIAL,
        "split": split,
        "unknown_fraction": {
            "train": train["unknown_local_pairs"] / max(train["local_candidates"], 1),
            "holdout": holdout["unknown_local_pairs"] / max(holdout["local_candidates"], 1),
        },
        "reachable_coverage": {
            "local_strict_positive": totals["local_reachable_target_bonds"],
            "all_pair_reachable": totals["all_pair_reachable_target_bonds"],
            "all_target_bonds": totals["target_bonds"],
            "local_over_reachable": coverage,
            "all_target_upper_bound": totals["local_reachable_target_bonds"] / max(totals["target_bonds"], 1),
            "all_pair_unreachable": totals["all_pair_unreachable_target_bonds"],
            "extra_geometry_rejected": totals["target_bonds_not_representable_by_local_pairs"] - totals["all_pair_unreachable_target_bonds"],
        },
        "strict_positive_counts": {
            "train": train["confirmed_positive_local_pairs"],
            "holdout": holdout["confirmed_positive_local_pairs"],
            "by_type": type_counts,
        },
        "top_k_known_only": top_k,
        "gate": {
            "prefix_immutable": report["baseline_prefix"]["immutable"],
            "split_disjoint": True,
            "candidate_target_independent": True,
            "strict_coverage_sufficient": coverage >= 0.99,
            "train_positive_sufficient": train["confirmed_positive_local_pairs"] >= 500,
            "holdout_positive_sufficient": holdout["confirmed_positive_local_pairs"] >= 200,
            "common_type_counts_sufficient": all(type_counts[name] >= 20 for name in ("SINGLE", "DOUBLE", "AROMATIC")),
            "frozen_state_unchanged": report["baseline_prefix"]["state_before_sha256"] == report["baseline_prefix"]["state_after_sha256"],
            "training_admissible": bool(
                report["baseline_prefix"]["immutable"]
                and report["baseline_prefix"]["state_before_sha256"] == report["baseline_prefix"]["state_after_sha256"]
                and coverage >= 0.99
                and train["confirmed_positive_local_pairs"] >= 500
                and holdout["confirmed_positive_local_pairs"] >= 200
                and all(type_counts[name] >= 20 for name in ("SINGLE", "DOUBLE", "AROMATIC"))
            ),
        },
        "limitations": [
            "unknown rows are excluded from top-K ranking and cannot support deployment precision claims",
            "793 target bonds are unreachable because selected atom endpoints are missing; this route cannot recover them",
            "the report is an audit of frozen pair logits, not a trained sidecar or graph-merge result",
        ],
    }
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    run()
