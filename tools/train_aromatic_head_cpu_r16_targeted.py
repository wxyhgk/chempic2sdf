#!/usr/bin/env python3
"""Train/evaluate an isolated aromatic-atom head on the r16 targeted split."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from tools.probe_atom_aromatic_head_cpu_r16 import extract
from tools.train_direct_graph_pair_scorer_cpu_r9 import DATA, SHA

SPLIT = ROOT / "outputs/audits/chemvector-aromatic-probe-split-r16/manifest.json"
OUT = ROOT / "outputs/audits/chemvector-atom-aromatic-head-cpu-r16-targeted"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def metrics(head, data, threshold: float) -> dict:
    xs = [x for x, _, _ in data if len(x)]
    ys = [y for _, y, _ in data if len(y)]
    if not xs:
        return {"rows": 0, "accuracy": None, "positive_recall": None, "negative_specificity": None}
    features, truth = torch.cat(xs), torch.cat(ys)
    pred = head(features).squeeze(-1).sigmoid() >= threshold
    positive = truth.bool(); negative = ~positive
    return {
        "rows": int(len(truth)),
        "positive_rows": int(positive.sum()),
        "accuracy": float((pred == positive).float().mean()),
        "positive_recall": float(pred[positive].float().mean()) if bool(positive.any()) else None,
        "negative_specificity": float((~pred[negative]).float().mean()) if bool(negative.any()) else None,
        "threshold": threshold,
    }


def run() -> dict:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    if digest(INITIAL) != SHA:
        raise RuntimeError("production checkpoint changed")
    manifest = json.loads(SPLIT.read_text())["rows"]
    train_rows = [row for row in manifest if row["split"] == "train"]
    holdout_rows = [row for row in manifest if row["split"] == "holdout"]
    train_ids = {row["molecule_id"] for row in train_rows}
    holdout_ids = {row["molecule_id"] for row in holdout_rows}
    if train_ids & holdout_ids:
        raise RuntimeError("targeted aromatic split identity overlap")
    model, config, _ = load_bundle_for_probe(INITIAL, SHA)
    model.eval().requires_grad_(False)
    train_data = [extract(row, model, config) for row in train_rows]
    holdout_data = [extract(row, model, config) for row in holdout_rows]
    features = torch.cat([x for x, _, _ in train_data if len(x)])
    labels = torch.cat([y for _, y, _ in train_data if len(y)])
    head = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 1))
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=1e-4)
    pos_weight = ((len(labels) - labels.sum()) / labels.sum().clamp_min(1.0)).clamp(1.0, 16.0)
    events = []
    for step in range(1, 301):
        head.train(); optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(head(features).squeeze(-1), labels, pos_weight=pos_weight)
        loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); optimizer.step()
        if step in (1, 100, 200, 300):
            events.append({"step": step, "loss": float(loss.detach())})
    head.eval()
    # Threshold is selected on train only; holdout is never used for selection.
    candidates = [round(x, 2) for x in torch.arange(0.30, 0.81, 0.02).tolist()]
    train_sweep = [(metrics(head, train_data, t), t) for t in candidates]
    threshold = max(train_sweep, key=lambda pair: (pair[0]["accuracy"], pair[0]["positive_recall"], -abs(pair[1] - 0.5)))[1]
    train_metrics = metrics(head, train_data, threshold)
    holdout_metrics = metrics(head, holdout_data, threshold)
    after = digest(INITIAL)
    result = {
        "schema_version": "chemvector-atom-aromatic-head-cpu-r16-targeted-0.1",
        "complete": True, "read_only": False, "training_performed": True,
        "gpu_used": False, "sealed_test_read": False, "production_checkpoint_written": False,
        "source_split": str(SPLIT.relative_to(ROOT)),
        "train_images": len(train_rows), "holdout_images": len(holdout_rows),
        "train_molecules": len(train_ids), "holdout_molecules": len(holdout_ids),
        "train_atom_rows": int(len(labels)), "train_positive_rows": int(labels.sum()),
        "events": events, "train_threshold_sweep": train_sweep,
        "selected_threshold": threshold, "train": train_metrics, "holdout": holdout_metrics,
        "initial_checkpoint_sha256": SHA, "initial_checkpoint_sha256_after": after,
        "gate": {
            "identity_disjoint": not bool(train_ids & holdout_ids),
            "train_threshold_only": True, "checkpoint_unchanged": after == SHA,
            "sealed_test_read": False,
            "holdout_accuracy_ge_0.90": holdout_metrics["accuracy"] >= 0.90,
            "holdout_positive_recall_ge_0.90": holdout_metrics["positive_recall"] is not None and holdout_metrics["positive_recall"] >= 0.90,
            "production_go": False,
        },
    }
    OUT.mkdir(parents=True, exist_ok=False)
    torch.save({"schema_version": result["schema_version"], "state_dict": head.state_dict(), "initial_checkpoint_sha256": SHA, "threshold": threshold}, OUT / "head-state.pt")
    (OUT / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT / "report.json"), "selected_threshold": threshold, "train": train_metrics, "holdout": holdout_metrics, "gate": result["gate"]}, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    run()
