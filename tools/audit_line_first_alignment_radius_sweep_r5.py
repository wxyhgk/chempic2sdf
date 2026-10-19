"""Read-only candidate alignment radius sweep for the line-first graph route."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import torch
from rdkit import rdBase

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import audit_line_first_graph_chemistry_roundtrip_r5 as audit  # noqa: E402

SOURCE = ROOT / "outputs/datasets/chemvector-identity-aligned-visible-text-20261017-r5-render"
TARGET = ROOT / "outputs/datasets/chemvector-identity-aligned-line-first-targeted-train-r5"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-alignment-radius-sweep-r5/report.json"


def summarize(values):
    c = Counter()
    for value in values:
        c.update(value)
    return {"images": c["images"], "edge_precision": c["true_edges"] / max(c["mapped_edges"], 1), "edge_recall": c["true_edges"] / max(c["gt_bonds"], 1), "bond_type_accuracy": c["type_correct"] / max(c["type_total"], 1), "dangling_rate": c["dangling_edges"] / max(c["selected_edges"], 1), "oracle_sanitize_rate": c["oracle_sanitize_ok"] / max(c["images"], 1), "candidate_sanitize_rate": c["candidate_sanitize_ok"] / max(c["images"], 1)}


def run_dataset(data, rows, candidates, model, mean, std, threshold, radius):
    audit.DATA = data
    audit.ALIGN_RADIUS = radius
    return summarize([audit._evaluate(row, candidates, model, mean, std, threshold, torch.device("cpu")) for row in rows])


def main():
    rdBase.DisableLog("rdApp.error")
    audit.DATA = TARGET
    candidates = audit._candidate_rows()
    state = torch.load(audit.STATE, map_location="cpu", weights_only=False)
    model = audit.EdgeBondMLP(len(state["feature_names"])); model.load_state_dict(state["model"]); model.eval()
    mean, std = np.asarray(state["mean"], np.float32), np.asarray(state["std"], np.float32); threshold = float(state["threshold"])
    train_rows = [json.loads(line) for line in (TARGET / "samples.jsonl").read_text().splitlines() if line.strip()]
    hold_rows = [json.loads(line) for line in (SOURCE / "samples.jsonl").read_text().splitlines() if line.strip() and json.loads(line).get("split") == "holdout"]
    results = {}
    for radius in (8.0, 10.0, 12.0, 16.0, 20.0, 24.0):
        results[str(radius)] = {"targeted_train": run_dataset(TARGET, train_rows, candidates, model, mean, std, threshold, radius), "fixed_holdout": run_dataset(SOURCE, hold_rows, candidates, model, mean, std, threshold, radius)}
    report = {"schema_version": "chemvector-identity-aligned-line-first-alignment-radius-sweep-r5-0.1", "complete": True, "read_only": True, "training_performed": False, "gpu_used": False, "sealed_test_read": False, "production_checkpoint_written": False, "results": results, "decision": {"production_authorized": False, "gpu_training_admissible": False, "next": "keep alignment frozen; radius-only repair is insufficient if oracle sanitize remains below 0.90"}}
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n"); print(json.dumps(report, indent=2, ensure_ascii=False)); return report


if __name__ == "__main__":
    main()
