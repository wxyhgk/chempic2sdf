"""Offline aromatic-promotion sensitivity audit for the saved 15-case trace.

This deliberately reuses only saved pair presence/chemical logits.  It does
not instantiate a model, run inference, read new images, access sealed data,
train, or modify production thresholds.  The result is a sensitivity report,
not a deployment recommendation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from chemvector.bond_object import BOND_TYPES
from chemvector.components.atoms.chemistry.direct_atom_pair_chemistry import (
    promote_complete_aromatic_rings,
)


ROOT = Path(__file__).resolve().parents[1]
TRACE = ROOT / "outputs/audits/chemvector-pair-scale-trace-20260925-gpu"
DEFAULT_OUTPUT = ROOT / "outputs/audits/chemvector-pair-scale-aromatic-thresholds-20260925"
MINIMUMS = (0.0, 0.02, 0.03, 0.05, 0.08)
MEAN = 0.40
PAIR_THRESHOLD = 0.85


def _score(rows: list[dict], minimum: float) -> dict:
    pairs = torch.tensor([row["atom_candidates"] for row in rows], dtype=torch.long)
    presence = torch.tensor([row["presence_probability"] for row in rows], dtype=torch.float32)
    logits = torch.tensor([row["chemical_logits"] for row in rows], dtype=torch.float32)
    chemistry, details = promote_complete_aromatic_rings(
        pairs,
        presence,
        logits,
        presence_threshold=PAIR_THRESHOLD,
        mean_threshold=MEAN,
        minimum_threshold=max(minimum, 1e-6),
    )
    return {
        "minimum_threshold": minimum,
        "selected_pairs": int((presence >= PAIR_THRESHOLD).sum()),
        "candidate_cycles": int(details["candidate_cycles"]),
        "accepted_cycles": int(details["accepted_cycles"]),
        "promoted_bonds": int(details["promoted_bonds"]),
        "raw_aromatic_selected": int(
            (((presence >= PAIR_THRESHOLD)
              & (logits.argmax(-1) == BOND_TYPES.index("AROMATIC"))).sum())
        ),
        "final_aromatic_selected": int(
            (((presence >= PAIR_THRESHOLD)
              & (chemistry == BOND_TYPES.index("AROMATIC"))).sum())
        ),
        "accepted_cycle_details": details["accepted_cycle_details"],
    }


def run(trace: Path = TRACE, output: Path = DEFAULT_OUTPUT) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    report = json.loads((trace / "report.json").read_text(encoding="utf-8"))
    if not report.get("complete") or report.get("training_performed"):
        raise ValueError("A complete read-only trace is required")
    results = []
    for case in report["results"]:
        case_out = {"key": case["key"], "class": case["class"], "arms": {}}
        for arm in ("initial", "pair-015443"):
            rows = case["arms"][arm]["pair_rows"]
            case_out["arms"][arm] = {str(m): _score(rows, m) for m in MINIMUMS}
        results.append(case_out)

    aggregate = {}
    for arm in ("initial", "pair-015443"):
        aggregate[arm] = {}
        for minimum in MINIMUMS:
            key = str(minimum)
            aggregate[arm][key] = {
                "broken_accepted_cycles": sum(
                    c["arms"][arm][key]["accepted_cycles"]
                    for c in results if c["class"] == "broken"
                ),
                "rescued_accepted_cycles": sum(
                    c["arms"][arm][key]["accepted_cycles"]
                    for c in results if c["class"] == "rescued"
                ),
                "broken_promoted_bonds": sum(
                    c["arms"][arm][key]["promoted_bonds"]
                    for c in results if c["class"] == "broken"
                ),
                "rescued_promoted_bonds": sum(
                    c["arms"][arm][key]["promoted_bonds"]
                    for c in results if c["class"] == "rescued"
                ),
                "broken_cases_with_more_accepted_cycles_than_production": sum(
                    c["arms"][arm][key]["accepted_cycles"]
                    > c["arms"][arm]["0.05"]["accepted_cycles"]
                    for c in results if c["class"] == "broken"
                ),
                "rescued_cases_with_fewer_accepted_cycles_than_production": sum(
                    c["arms"][arm][key]["accepted_cycles"]
                    < c["arms"][arm]["0.05"]["accepted_cycles"]
                    for c in results if c["class"] == "rescued"
                ),
            }
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "schema_version": "chemvector-pair-scale-aromatic-threshold-audit-0.1",
        "complete": True,
        "trace_report_sha256": __import__("hashlib").sha256(
            (trace / "report.json").read_bytes()).hexdigest(),
        "trace_scope": {"cases": 15, "model_forwards": 0, "training_performed": False,
                        "sealed_test_read": False, "production_modified": False},
        "fixed_thresholds": {"pair": PAIR_THRESHOLD, "mean": MEAN,
                             "minimums": list(MINIMUMS)},
        "results": results,
        "aggregate": aggregate,
        "limitations": [
            "This recomputes promotion from saved logits only.",
            "It does not recompile scenes or measure SDF validity/exactness.",
            "No threshold is changed in production.",
        ],
    }
    (output / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, default=TRACE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.trace, args.output)
    print(json.dumps({"status": "completed", "output": str(args.output),
                      "cases": len(result["results"]), "model_forwards": 0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
