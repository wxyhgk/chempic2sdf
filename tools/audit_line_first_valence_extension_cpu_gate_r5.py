"""CPU-only loader and line-reachability gate for the new valence extension."""
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

DATA = ROOT / "outputs/datasets/chemvector-identity-aligned-line-first-valence-extension-r5"
SOURCE = ROOT / "outputs/datasets/chemvector-identity-aligned-visible-text-20261017-r5-render"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-valence-extension-cpu-gate-r5/report.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> dict:
    rows = [json.loads(line) for line in (DATA / "samples.jsonl").read_text().splitlines() if line.strip()]
    if len(rows) != 216 or any(row.get("split") != "train" for row in rows):
        raise ValueError("extension must contain exactly 216 train rows")
    if len({row["molecule_id"] for row in rows}) != 72 or len({row["profile_id"] for row in rows}) != 3:
        raise ValueError("extension identity/profile accounting failed")
    fixed_ids = {json.loads(line)["molecule_id"] for line in (SOURCE / "samples.jsonl").read_text().splitlines() if line.strip()}
    if {row["molecule_id"] for row in rows} & fixed_ids:
        raise ValueError("extension overlaps fixed source identities")
    # Normal diagnostic loader validates all sidecars and same-source labels.
    for row in rows:
        load_sample(DATA, row["sample_key"], purpose="diagnostic")
    traversal.DATA = DATA
    by_profile = defaultdict(list)
    for row in sorted(rows, key=lambda item: item["sample_key"]):
        result = traversal.audit_sample(row["sample_key"], "train", cluster_radius=10.0,
                                        target_radius=24.0, max_segments=512)
        by_profile[row["profile_id"]].append(result)
    def summarize(values: list[dict]) -> dict:
        return {
            "images": len(values),
            "mean_edge_precision": float(np.mean([v["edge_precision"] for v in values])),
            "mean_bond_recall": float(np.mean([v["bond_recall"] for v in values])),
            "images_all_bonds_recovered": int(sum(v["all_bonds_recovered"] for v in values)),
            "mean_gt_bonds": float(np.mean([v["gt_bonds"] for v in values])),
        }
    all_values = [v for values in by_profile.values() for v in values]
    aggregate = summarize(all_values)
    profile = {key: summarize(value) for key, value in sorted(by_profile.items())}
    weak = [profile[key]["mean_bond_recall"] for key in ("cov13_base48", "cov13_stroke16_48")]
    gate = {
        "row_accounting": True, "loader_roundtrip": True, "identity_disjoint": True,
        "aggregate_edge_precision_ge_0_90": aggregate["mean_edge_precision"] >= 0.90,
        "weak_profile_reachability_ge_0_75": all(value >= 0.75 for value in weak),
        "training_performed": False, "gpu_used": False, "sealed_test_read": False,
        "gpu_training_admissible": False,
    }
    report = {
        "schema_version": "chemvector-identity-aligned-line-first-valence-extension-cpu-gate-r5-0.1",
        "complete": True, "read_only": True, "training_performed": False, "gpu_used": False,
        "sealed_test_read": False, "production_checkpoint_written": False,
        "dataset": str(DATA), "dataset_summary_sha256": digest(DATA / "summary.json"),
        "source_summary_sha256": digest(SOURCE / "summary.json"),
        "aggregate": aggregate, "by_profile": profile, "gate": gate,
        "next": "compare against fixed train/holdout CPU baselines; no GPU admission from extension alone",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
