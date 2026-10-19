"""Compare the extension CPU preflight with the frozen full-holdout shadow.

The preflight's ``recall`` is candidate-positive recall.  This report adds
the fixed full-holdout denominator from the frozen shadow so a seemingly
perfect candidate recall cannot hide missing graph bonds.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-valence-extension-edge-r5-cpu/report.json"
BASELINE = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-edge-bond-shadow-r5-full/report.json"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-valence-extension-comparison-r5/report.json"


def main() -> dict:
    new = json.loads(PREFLIGHT.read_text())
    old = json.loads(BASELINE.read_text())
    nm, om = new["holdout"], old["holdout"]["metrics"]
    report = {
        "schema_version": "chemvector-identity-aligned-line-first-valence-extension-comparison-r5-0.1",
        "complete": True, "read_only": True, "training_performed": True, "gpu_used": False,
        "sealed_test_read": False, "production_checkpoint_written": False,
        "new_preflight": {"precision": nm["precision"], "candidate_positive_recall": nm["recall"],
                           "auprc": nm["auprc"], "candidates": nm["candidates"], "positives": nm["positives"]},
        "frozen_r5_full_holdout": {"precision": om["edge_precision"],
                                    "graph_bond_recall": om["candidate_edge_recall"],
                                    "type_accuracy": om["conditional_bond_type_accuracy"]},
        "delta_precision": float(nm["precision"] - om["edge_precision"]),
        "interpretation": "extension preflight selects every candidate; candidate-positive recall is not graph-bond recall",
        "gate": {
            "new_precision_ge_frozen": bool(nm["precision"] >= om["edge_precision"]),
            "new_precision_strictly_improves": bool(nm["precision"] > om["edge_precision"]),
            "gpu_training_admissible": False,
            "production_go": False,
        },
        "next": "do not spend GPU on this data-only sidecar recipe; focus on ownership/graph topology CPU route",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
