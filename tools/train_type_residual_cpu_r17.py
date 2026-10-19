#!/usr/bin/env python3
"""Train an isolated bond-type residual on the r17 graph candidate cache."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.train_direct_graph_pair_scorer_cpu_r9 import SHA

CACHE = ROOT / os.environ.get("R17_TYPE_CACHE", "outputs/audits/chemvector-graph-context-aromatic-cpu-r17-all-m16-t0p72-c1-cache3/feature-cache.pt")
OUT = ROOT / os.environ.get("R17_TYPE_OUT", "outputs/audits/chemvector-type-residual-cpu-r17-all-m16")


class TypeContextHead(nn.Module):
    def __init__(self, input_dim: int = 262, classes: int = 4):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, 64), nn.GELU(), nn.Linear(64, classes))

    def forward(self, x):
        return self.net(x)


def score(head, records) -> dict:
    features = torch.cat([r["type_features"] for r in records if len(r["type_features"])])
    truth = torch.cat([r["type_labels"] for r in records if len(r["type_labels"])])
    pred = head(features).argmax(-1)
    confusion = [[0] * 4 for _ in range(4)]
    for t, p in zip(truth.tolist(), pred.tolist()): confusion[t][p] += 1
    return {"rows": len(truth), "accuracy": float((pred == truth).float().mean()), "confusion": confusion, "target_counts": [int((truth == i).sum()) for i in range(4)]}


def run() -> dict:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    cache = torch.load(CACHE, map_location="cpu", weights_only=False)
    if cache.get("initial_checkpoint_sha256") != SHA:
        raise RuntimeError("cache checkpoint binding mismatch")
    records = cache["records"]
    train_features = torch.cat([r["type_features"] for r in records["train"] if len(r["type_features"])])
    train_labels = torch.cat([r["type_labels"] for r in records["train"] if len(r["type_labels"])])
    head = TypeContextHead(train_features.shape[-1])
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=1e-4)
    counts = torch.bincount(train_labels, minlength=4).float()
    weights = (len(train_labels) / counts.clamp_min(1.0)).sqrt()
    events = []
    for step in range(1, 501):
        head.train(); optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(head(train_features), train_labels, weight=weights)
        loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); optimizer.step()
        if step in (1, 100, 300, 500): events.append({"step": step, "loss": float(loss.detach())})
    head.eval()
    metrics = {split: score(head, records[split]) for split in records}
    result = {"schema_version": "chemvector-type-residual-cpu-r17-0.1", "complete": True, "read_only": False, "training_performed": True, "gpu_used": False, "sealed_test_read": False, "production_checkpoint_written": False, "cache": str(CACHE.relative_to(ROOT)), "head_parameters": sum(p.numel() for p in head.parameters()), "events": events, "metrics": metrics, "initial_checkpoint_sha256": SHA, "gate": {"target_free_features": True, "edge_set_frozen": True, "dev_type_accuracy_ge_0.90": metrics["dev"]["accuracy"] >= .90, "fresh_type_accuracy_ge_0.90": metrics["fresh_holdout"]["accuracy"] >= .90, "production_go": False}}
    OUT.mkdir(parents=True, exist_ok=False)
    torch.save({"schema_version": result["schema_version"], "state_dict": head.state_dict(), "initial_checkpoint_sha256": SHA}, OUT / "sidecar-state.pt")
    (OUT / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT / "report.json"), "metrics": metrics, "gate": result["gate"]}, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__": run()
