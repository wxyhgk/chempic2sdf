#!/usr/bin/env python3
"""Bounded CPU probe for target-free graph-context element classification."""
from __future__ import annotations
import json
import os
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / os.environ.get("R17_ELEMENT_CONTEXT_CACHE", "outputs/audits/chemvector-graph-context-aromatic-cpu-r17-all-m16-t0p72-c1-cache3/feature-cache.pt")
OUT = ROOT / os.environ.get("R17_ELEMENT_CONTEXT_OUT", "outputs/audits/chemvector-element-context-head-cpu-r17-all-m16")
ELEMENTS = ("C", "O", "N", "S", "F", "Cl", "Br", "I", "P", "B", "Si")
SHA = ""


def rows(records):
    xs, ys = [], []
    for record in records:
        if not len(record["pairs"]):
            continue
        node_ids = sorted({int(a) for a, _, _ in record["pairs"]} | {int(b) for _, b, _ in record["pairs"]})
        by_node = {node: [] for node in node_ids}
        for index, (left, right, _) in enumerate(record["pairs"]):
            by_node[left].append(record["type_features"][index])
            by_node[right].append(record["type_features"][index])
        for node in node_ids:
            # Frozen pair context only: mean context, max context, and degree.
            context = torch.stack(by_node[node])
            degree = torch.tensor([float(len(by_node[node])) / 8.0], dtype=torch.float32)
            feature = torch.cat((record["atom_features"][node].float(), context.mean(0), context.max(0).values, degree))
            target = record["target_elements"].get(int(record["mapping"].get(node, -1)), "C")
            if target in ELEMENTS:
                xs.append(feature); ys.append(ELEMENTS.index(target))
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)


class ElementContextHead(nn.Module):
    def __init__(self, input_dim=653):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, 96), nn.GELU(), nn.Linear(96, len(ELEMENTS)))
    def forward(self, x): return self.net(x)


def score(head, data):
    x, y = data
    with torch.inference_mode():
        pred = head(x).argmax(-1)
    return {"rows": len(y), "accuracy": float((pred == y).float().mean()), "confusion": torch.bincount(y * len(ELEMENTS) + pred, minlength=len(ELEMENTS) ** 2).reshape(len(ELEMENTS), len(ELEMENTS)).tolist()}


def run():
    global SHA
    if OUT.exists(): raise FileExistsError(OUT)
    cache = torch.load(CACHE, map_location="cpu", weights_only=False)
    SHA = str(cache["initial_checkpoint_sha256"])
    train = rows(cache["records"]["train"])
    head = ElementContextHead(train[0].shape[-1])
    opt = torch.optim.AdamW(head.parameters(), lr=1.5e-3, weight_decay=1e-4)
    counts = torch.bincount(train[1], minlength=len(ELEMENTS)).float(); weights = (len(train[1]) / counts.clamp_min(1)).sqrt()
    events = []
    for step in range(1, 501):
        opt.zero_grad(set_to_none=True); loss = F.cross_entropy(head(train[0]), train[1], weight=weights); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step()
        if step in (1, 100, 300, 500): events.append({"step": step, "loss": float(loss.detach())})
    head.eval(); metrics = {split: score(head, rows(values)) for split, values in cache["records"].items()}
    result = {"schema_version": "chemvector-element-context-head-cpu-r17-0.1", "complete": True, "read_only": False, "training_performed": True, "gpu_used": False, "sealed_test_read": False, "production_checkpoint_written": False, "input_dim": train[0].shape[-1], "head_parameters": sum(p.numel() for p in head.parameters()), "events": events, "metrics": metrics, "initial_checkpoint_sha256": SHA, "gate": {"target_free_features": True, "dev_accuracy_ge_0.90": metrics["dev"]["accuracy"] >= .90, "fresh_accuracy_ge_0.90": metrics["fresh_holdout"]["accuracy"] >= .90, "production_go": False}}
    OUT.mkdir(parents=True, exist_ok=False); torch.save({"schema_version": result["schema_version"], "state_dict": head.state_dict(), "initial_checkpoint_sha256": SHA}, OUT / "head-state.pt"); (OUT / "report.json").write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps({"output": str(OUT / "report.json"), "metrics": metrics, "gate": result["gate"]}, indent=2))


if __name__ == "__main__": run()
