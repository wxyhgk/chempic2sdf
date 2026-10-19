"""Attribute targeted train round-trip failures by graph/renderer strata."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import torch
from rdkit import rdBase

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import audit_line_first_graph_chemistry_roundtrip_r5 as audit  # noqa: E402

DATA = ROOT / "outputs/datasets/chemvector-identity-aligned-line-first-targeted-train-r5"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-targeted-failure-strata-r5/report.json"


def tags(graph: dict) -> list[str]:
    kinds = {str(b.get("chemical_type", "OTHER")).upper() for b in graph.get("bonds", [])}
    degree = Counter()
    for bond in graph.get("bonds", []):
        degree[int(bond["begin_atom"])] += 1
        degree[int(bond["end_atom"])] += 1
    result = ["all"]
    if "AROMATIC" in kinds:
        result.append("aromatic_ring")
    if {"DOUBLE", "TRIPLE"} & kinds:
        result.append("multiple_bond")
    if any(value >= 3 for value in degree.values()):
        result.append("junction_degree3plus")
    if "SINGLE" in kinds and not ({"DOUBLE", "TRIPLE", "AROMATIC"} & kinds):
        result.append("single_only")
    return result


def finish(counter: Counter) -> dict:
    images = counter["images"]
    return {
        "images": images,
        "edge_precision": counter["true_edges"] / max(counter["mapped_edges"], 1),
        "edge_recall": counter["true_edges"] / max(counter["gt_bonds"], 1),
        "bond_type_accuracy": counter["type_correct"] / max(counter["type_total"], 1),
        "dangling_rate": counter["dangling_edges"] / max(counter["selected_edges"], 1),
        "oracle_sanitize_rate": counter["oracle_sanitize_ok"] / max(images, 1),
        "oracle_aromatic_error_rate": counter["oracle_aromatic_errors"] / max(images, 1),
        "oracle_valence_error_rate": counter["oracle_valence_errors"] / max(images, 1),
    }


def main() -> dict:
    rdBase.DisableLog("rdApp.error")
    audit.DATA = DATA
    candidates = audit._candidate_rows()
    state = torch.load(audit.STATE, map_location="cpu", weights_only=False)
    model = audit.EdgeBondMLP(len(state["feature_names"]))
    model.load_state_dict(state["model"])
    model.eval()
    mean, std = np.asarray(state["mean"], np.float32), np.asarray(state["std"], np.float32)
    threshold = float(state["threshold"])
    rows = [json.loads(line) for line in (DATA / "samples.jsonl").read_text().splitlines() if line.strip()]
    by_stratum = defaultdict(Counter)
    for row in sorted(rows, key=lambda item: item["sample_key"]):
        base = DATA / "samples" / row["sample_key"]
        graph = json.loads(base.with_suffix(".graph.json").read_text())
        result = audit._evaluate(row, candidates, model, mean, std, threshold, torch.device("cpu"))
        for label in tags(graph) + ["profile:" + row["profile_id"]]:
            by_stratum[label].update(result)
    report = {
        "schema_version": "chemvector-identity-aligned-line-first-targeted-failure-strata-r5-0.1",
        "complete": True, "read_only": True, "training_performed": False, "gpu_used": False,
        "sealed_test_read": False, "production_checkpoint_written": False,
        "dataset": str(DATA), "by_stratum": {key: finish(value) for key, value in sorted(by_stratum.items())},
        "interpretation": {
            "next": "repair target-free aromatic closure/ownership before any GPU shadow",
            "gpu_training_admissible": False,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
