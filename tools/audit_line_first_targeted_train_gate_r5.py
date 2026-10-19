"""CPU integrity and line-reachability gate for the cov13 train-only panel."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chemvector.training.curriculum.data import load_sample  # noqa: E402
from tools import audit_line_first_graph_traversal_r5 as traversal  # noqa: E402

DATA = ROOT / "outputs/datasets/chemvector-identity-aligned-line-first-targeted-train-r5"
SOURCE = ROOT / "outputs/datasets/chemvector-identity-aligned-visible-text-20261017-r5-render"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-targeted-train-gate-r5/report.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> dict:
    rows = [json.loads(line) for line in (DATA / "samples.jsonl").read_text().splitlines() if line.strip()]
    if len(rows) != 348 or any(row.get("split") != "train" for row in rows):
        raise ValueError("target panel must contain exactly 348 train rows")
    if len({row["molecule_id"] for row in rows}) != 116 or len({row["profile_id"] for row in rows}) != 3:
        raise ValueError("target identity/profile accounting failed")
    source_holdout = {json.loads(line)["molecule_id"] for line in (SOURCE / "samples.jsonl").read_text().splitlines()
                      if line.strip() and json.loads(line).get("split") == "holdout"}
    target_ids = {row["molecule_id"] for row in rows}
    if target_ids & source_holdout:
        raise ValueError("target panel overlaps fixed holdout")
    # Verify every accepted row through the normal diagnostic loader and retain
    # the same target-free LSD proposal contract used by the r5 route gate.
    for row in rows:
        load_sample(DATA, row["sample_key"], purpose="diagnostic")
    traversal.DATA = DATA
    by_profile = defaultdict(list)
    for row in sorted(rows, key=lambda item: item["sample_key"]):
        result = traversal.audit_sample(row["sample_key"], "train", cluster_radius=10.0,
                                        target_radius=24.0, max_segments=512)
        by_profile[row["profile_id"]].append(result)
    profile_report = {}
    for profile, values in sorted(by_profile.items()):
        profile_report[profile] = {
            "images": len(values),
            "mean_edge_precision": float(np.mean([v["edge_precision"] for v in values])),
            "mean_bond_recall": float(np.mean([v["bond_recall"] for v in values])),
            "images_all_bonds_recovered": int(sum(v["all_bonds_recovered"] for v in values)),
        }
    all_values = [value for values in by_profile.values() for value in values]
    aggregate = {
        "images": len(all_values),
        "mean_edge_precision": float(np.mean([v["edge_precision"] for v in all_values])),
        "mean_bond_recall": float(np.mean([v["bond_recall"] for v in all_values])),
        "images_all_bonds_recovered": int(sum(v["all_bonds_recovered"] for v in all_values)),
    }
    weak = [profile_report[name]["mean_bond_recall"] for name in ("cov13_base48", "cov13_stroke16_48")]
    gate = {
        "row_accounting": True, "loader_roundtrip": True, "identity_disjoint": True,
        "weak_profile_edge_reachability_ge_0_90": all(value >= 0.90 for value in weak),
        "aggregate_edge_precision_ge_0_90": aggregate["mean_edge_precision"] >= 0.90,
        "training_authorized": False, "sealed_test_read": False,
        "gpu_training_admissible": False,
    }
    report = {
        "schema_version": "chemvector-identity-aligned-line-first-targeted-train-gate-r5-0.1",
        "complete": True, "read_only": True, "training_performed": False, "gpu_used": False,
        "sealed_test_read": False, "production_checkpoint_written": False,
        "dataset": str(DATA), "source_summary_sha256": digest(SOURCE / "summary.json"),
        "target_summary_sha256": digest(DATA / "summary.json"), "aggregate": aggregate,
        "by_profile": profile_report, "gate": gate,
        "next": "do not start GPU shadow; repair endpoint/ownership and aromatic closure coverage first",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
