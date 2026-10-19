"""Verify the minimal immutable-baseline/residual-ownership contract.

This is a report-only check over the current-initial forward trace produced by
``audit_highres_candidate_parity.py``.  It does not load a model or images,
and it never reads sealed identities.  The check separates *interface safety*
from *candidate quality*: a safe ownership interface may still be far too
noisy to justify training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CURRENT_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
DEFAULT_TRACE = ROOT / "outputs/audits/chemvector-highres-candidate-parity-20261006-r2/report.json"
DEFAULT_OUTPUT = ROOT / "outputs/audits/chemvector-highres-ownership-contract-20261006"


def _check_row(row: dict) -> dict:
    raw = set(int(value) for value in row["raw_baseline_ids"])
    selected = set(int(value) for value in row["selected_baseline_ids"])
    side = set(int(value) for value in row["selected_locked_sidecar_matched"])
    raw_side = set(int(value) for value in row["raw_locked_sidecar_matched"])
    candidate_count = int(row["final_candidate_count"])
    guarded = int(row["guarded_peak_count"])
    rejected = int(row["guard_rejected_peak_count"])
    peaks = int(row["raw_peak_count"])
    checks = {
        "selected_baseline_subset_raw": selected <= raw,
        "selected_sidecar_disjoint_baseline": not (side & selected),
        "raw_sidecar_disjoint_baseline": not (raw_side & raw),
        "candidate_ladder_conserved": guarded + rejected == peaks and candidate_count <= guarded,
        "selected_sidecar_within_candidate_budget": len(side) <= candidate_count,
        "false_positive_accounting": int(row["selected_sidecar_false_positive_count"]) == candidate_count - len(side),
        "residual_count_consistent": int(row["selected_residual_target_count"]) == int(row["targets"]) - len(selected),
    }
    return {"sample_key": row["sample_key"], "checks": checks, "pass": all(checks.values())}


def run(trace_path: Path = DEFAULT_TRACE, output: Path = DEFAULT_OUTPUT) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    trace = json.loads(trace_path.read_text())
    rows = trace.get("rows", [])
    if trace.get("initial_sha256") != CURRENT_SHA:
        raise ValueError("Trace is not bound to the frozen current initial checkpoint")
    if not rows:
        raise ValueError("Trace contains no rows")
    row_checks = [_check_row(row) for row in rows]
    interface_pass = all(item["pass"] for item in row_checks)
    summary = trace["summary"]
    candidates = int(summary["final_candidate_count"])
    positives = int(summary["selected_locked_sidecar_matched"])
    precision = positives / max(candidates, 1)
    result = {
        "schema_version": "chemvector-highres-ownership-contract-0.1",
        "complete": True,
        "read_only": True,
        "training_performed": False,
        "sealed_test_read": False,
        "deployment_authorized": False,
        "trace": str(trace_path),
        "initial_sha256": CURRENT_SHA,
        "contract": {
            "baseline": "selected baseline target IDs are immutable",
            "residual": "sidecar may match only targets outside selected baseline IDs",
            "candidate_order": "raw peaks -> baseline guard -> post-guard top-K -> residual Hungarian",
        },
        "interface": {
            "images": len(rows),
            "all_rows_pass": interface_pass,
            "failed_rows": [item["sample_key"] for item in row_checks if not item["pass"]],
        },
        "quality_diagnostic": {
            "candidate_count": candidates,
            "positive_sidecar_matches": positives,
            "precision": precision,
            "minimum_candidate_precision_for_training": 0.85,
            "quality_gate_pass": positives >= 50 and precision >= 0.85,
        },
        "gate": {
            "ownership_interface_pass": interface_pass,
            "quality_gate_pass": positives >= 50 and precision >= 0.85,
            "bounded_training_admissible": False,
            "reason": "interface is auditable but frozen-field candidate precision is insufficient; regenerate/train sidecar only after current-contract labels pass strict readiness",
        },
        "row_checks": row_checks,
    }
    output.mkdir(parents=True)
    (output / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "interface": result["interface"], "quality": result["quality_diagnostic"], "gate": result["gate"]}, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", default=str(DEFAULT_TRACE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    run(Path(args.trace), Path(args.output))

