#!/usr/bin/env python3
"""CPU probe for aromatic-atom recoverability from frozen AtomObject features.

The probe is isolated: it trains only a tiny binary head, never changes the
production checkpoint, and reports dev/fresh atom and molecule-level aromatic
labels.  It is intended to decide whether the remaining r15 failures are an
atom-aromaticity problem before spending GPU time on bond type residuals.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.pixels import image_tensor
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from tools.train_direct_graph_pair_scorer_cpu_r9 import DATA, SHA, _map_atoms, _target, rows, ATOM_THRESHOLD

OUT = ROOT / "outputs/audits/chemvector-atom-aromatic-head-cpu-r16"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def train_rows(all_rows: list[dict]) -> list[dict]:
    by_profile: dict[str, list[dict]] = defaultdict(list)
    for row in sorted((r for r in all_rows if r.get("split") == "train"), key=lambda r: r["sample_key"]):
        by_profile[row["profile_id"]].append(row)
    return [row for profile in sorted(by_profile) for row in by_profile[profile][:24]]


def extract(row: dict, model, config) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    base = DATA / "samples" / row["sample_key"]
    with torch.inference_mode():
        output = structure_forward(model, image_tensor(Image.open(base.with_suffix(".png")).convert("RGB"))[None], config, ATOM_THRESHOLD)
    selected = output[6]
    slots = torch.where(selected["atom_mask"][0])[0]
    points = selected["atom_geometry"][0, slots, :2].float()
    targets, _, _ = _target(row["sample_key"])
    mapping = _map_atoms(points, targets)
    layers = json.loads(base.with_suffix(".layers.json").read_text())
    atoms = sorted(layers["atoms"], key=lambda atom: int(atom["atom_id"]))
    features, labels, elements = [], [], []
    for compact, target in mapping.items():
        features.append(selected["atom_features"][0, slots[compact]].float())
        labels.append(float(bool(atoms[target].get("aromatic", False))))
        elements.append(str(atoms[target].get("element") or "C"))
    if not features:
        return torch.empty((0, 128)), torch.empty((0,)), []
    return torch.stack(features), torch.tensor(labels, dtype=torch.float32), elements


def score(head, data: list[tuple[torch.Tensor, torch.Tensor, list[str]]]) -> dict:
    xs = [x for x, _, _ in data if len(x)]
    ys = [y for _, y, _ in data if len(y)]
    es = [e for _, _, e in data]
    features, truth = torch.cat(xs), torch.cat(ys)
    probability = head(features).squeeze(-1).sigmoid()
    pred = probability >= 0.5
    by_element = {}
    offset = 0
    for elements in es:
        for element in elements:
            value = int(pred[offset]); target = int(truth[offset]); offset += 1
            slot = by_element.setdefault(element, {"rows": 0, "correct": 0, "positive": 0, "predicted_positive": 0})
            slot["rows"] += 1; slot["correct"] += int(value == target)
            slot["positive"] += target; slot["predicted_positive"] += value
    for value in by_element.values():
        value["accuracy"] = value["correct"] / max(value["rows"], 1)
    return {
        "rows": int(len(truth)),
        "positive_rows": int(truth.sum()),
        "accuracy": float((pred == truth.bool()).float().mean()),
        "positive_recall": float((pred[truth.bool()]).float().mean()) if bool(truth.bool().any()) else None,
        "negative_specificity": float((~pred[~truth.bool()]).float().mean()) if bool((~truth.bool()).any()) else None,
        "by_element": by_element,
    }


def run() -> dict:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    if digest(INITIAL) != SHA:
        raise RuntimeError("production checkpoint changed")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20261016)
    all_rows = rows()
    train = train_rows(all_rows)
    dev = sorted((r for r in all_rows if r.get("split") == "dev"), key=lambda r: r["sample_key"])
    fresh = sorted((r for r in all_rows if r.get("split") == "fresh_holdout"), key=lambda r: r["sample_key"])
    identities = {name: {r["molecule_id"] for r in value} for name, value in (("train", train), ("dev", dev), ("fresh", fresh))}
    if any(identities[a] & identities[b] for a, b in (("train", "dev"), ("train", "fresh"), ("dev", "fresh"))):
        raise RuntimeError("molecule identity overlap")
    model, config, _ = load_bundle_for_probe(INITIAL, SHA)
    model.eval().requires_grad_(False)
    train_data = [extract(row, model, config) for row in train]
    dev_data = [extract(row, model, config) for row in dev]
    fresh_data = [extract(row, model, config) for row in fresh]
    features = torch.cat([x for x, _, _ in train_data if len(x)])
    labels = torch.cat([y for _, y, _ in train_data if len(y)])
    head = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 1))
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=1e-4)
    positive_weight = ((len(labels) - labels.sum()) / labels.sum().clamp_min(1.0)).clamp(1.0, 16.0)
    events = []
    for step in range(1, 301):
        head.train(); optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(head(features).squeeze(-1), labels, pos_weight=positive_weight)
        loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); optimizer.step()
        if step in (1, 100, 200, 300):
            events.append({"step": step, "loss": float(loss.detach())})
    head.eval()
    dev_metrics = score(head, dev_data)
    fresh_metrics = score(head, fresh_data)
    after = digest(INITIAL)
    result = {
        "schema_version": "chemvector-atom-aromatic-head-cpu-r16-0.1",
        "complete": True, "read_only": False, "training_performed": True,
        "gpu_used": False, "sealed_test_read": False, "production_checkpoint_written": False,
        "train_images": len(train), "dev_images": len(dev), "fresh_images": len(fresh),
        "train_rows": int(len(labels)), "train_positive_rows": int(labels.sum()),
        "head_parameters": sum(p.numel() for p in head.parameters()),
        "events": events, "dev": dev_metrics, "fresh_holdout": fresh_metrics,
        "initial_checkpoint_sha256": SHA, "initial_checkpoint_sha256_after": after,
        "gate": {
            "identity_disjoint": True, "checkpoint_unchanged": after == SHA,
            "sealed_test_read": False,
            "dev_accuracy_ge_0.90": dev_metrics["accuracy"] >= 0.90,
            "fresh_accuracy_ge_0.90": fresh_metrics["accuracy"] >= 0.90,
            "positive_recall_ge_0.90_both": all(x["positive_recall"] is not None and x["positive_recall"] >= 0.90 for x in (dev_metrics, fresh_metrics)),
            "production_go": False,
        },
    }
    OUT.mkdir(parents=True, exist_ok=False)
    torch.save({"schema_version": result["schema_version"], "state_dict": head.state_dict(), "initial_checkpoint_sha256": SHA}, OUT / "head-state.pt")
    (OUT / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT / "report.json"), "dev": dev_metrics, "fresh": fresh_metrics, "gate": result["gate"]}, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
