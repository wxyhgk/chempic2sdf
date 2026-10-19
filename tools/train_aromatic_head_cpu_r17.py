#!/usr/bin/env python3
"""Train an isolated aromatic-atom head on the r17 scaffold-disjoint slice."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
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
from tools.train_direct_graph_pair_scorer_cpu_r9 import SHA, _map_atoms

PLAN = ROOT / "outputs/audits/chemvector-aromatic-specialist-plan-r17-v2/rows.jsonl"
SOURCE = ROOT / "outputs/datasets/structure-expansion-batch-12272-20260923"
OUT = ROOT / "outputs/audits/chemvector-atom-aromatic-head-cpu-r17"
ATOM_THRESHOLD = -0.25
LIMIT_MOLECULES = int(os.environ.get("R17_AROMATIC_LIMIT_MOLECULES", "0"))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def extract(row: dict, model, config) -> tuple[torch.Tensor, torch.Tensor, list[str], list[str]]:
    base = SOURCE / "samples" / row["sample_key"]
    with torch.inference_mode():
        output = structure_forward(model, image_tensor(Image.open(base.with_suffix(".png")).convert("RGB"))[None], config, ATOM_THRESHOLD)
    selected = output[6]
    slots = torch.where(selected["atom_mask"][0])[0]
    points = selected["atom_geometry"][0, slots, :2].float()
    layers = json.loads(base.with_suffix(".layers.json").read_text())
    atoms = sorted(layers["atoms"], key=lambda atom: int(atom["atom_id"]))
    targets = torch.tensor([atom["center"] for atom in atoms], dtype=torch.float32)
    mapping = _map_atoms(points, targets)
    features, labels, elements = [], [], []
    for compact, target in mapping.items():
        atom = atoms[target]
        features.append(selected["atom_features"][0, slots[compact]].float())
        labels.append(float(bool(atom.get("aromatic", False))))
        elements.append(str(atom.get("element") or "C"))
    if not features:
        return torch.empty((0, 128)), torch.empty((0,)), [], []
    return torch.stack(features), torch.tensor(labels, dtype=torch.float32), elements, row.get("features", [])


def score(head, data: list[tuple[torch.Tensor, torch.Tensor, list[str], list[str]]], threshold: float) -> dict:
    xs = [x for x, _, _, _ in data if len(x)]
    ys = [y for _, y, _, _ in data if len(y)]
    if not xs:
        return {"rows": 0, "positive_rows": 0, "accuracy": None, "positive_recall": None, "negative_specificity": None}
    features, truth = torch.cat(xs), torch.cat(ys)
    pred = head(features).squeeze(-1).sigmoid() >= threshold
    positive, negative = truth.bool(), ~truth.bool()
    return {
        "rows": int(len(truth)), "positive_rows": int(positive.sum()),
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
    rows = [json.loads(line) for line in PLAN.read_text().splitlines() if line.strip()]
    splits = {split: [row for row in rows if row["split"] == split] for split in ("train", "dev", "fresh_holdout")}
    if LIMIT_MOLECULES:
        for split in splits:
            ids = sorted({row["molecule_id"] for row in splits[split]})[:LIMIT_MOLECULES]
            splits[split] = [row for row in splits[split] if row["molecule_id"] in set(ids)]
    id_sets = {split: {row["molecule_id"] for row in values} for split, values in splits.items()}
    overlaps = {f"{a}:{b}": len(id_sets[a] & id_sets[b]) for a, b in (("train", "dev"), ("train", "fresh_holdout"), ("dev", "fresh_holdout"))}
    if any(overlaps.values()):
        raise RuntimeError(f"identity overlap: {overlaps}")
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); torch.manual_seed(20261017)
    model, config, _ = load_bundle_for_probe(INITIAL, SHA)
    model.eval().requires_grad_(False)
    data = {}
    for split, values in splits.items():
        print(json.dumps({"phase": "extract", "split": split, "images": len(values)}), flush=True)
        data[split] = [extract(row, model, config) for row in values]
    features = torch.cat([x for x, _, _, _ in data["train"] if len(x)])
    labels = torch.cat([y for _, y, _, _ in data["train"] if len(y)])
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
    candidates = [round(x, 2) for x in torch.arange(0.30, 0.81, 0.02).tolist()]
    train_sweep = [(score(head, data["train"], threshold), threshold) for threshold in candidates]
    threshold = max(train_sweep, key=lambda pair: (pair[0]["accuracy"], pair[0]["positive_recall"], -abs(pair[1] - 0.5)))[1]
    metrics = {split: score(head, data[split], threshold) for split in splits}
    coverage = {split: {feature: sum(feature in row["labels"] for row in values) for feature in ("aromatic_C", "aromatic_N", "aromatic_S", "fused", "heteroaromatic", "control_DOUBLE", "control_TRIPLE")} for split, values in splits.items()}
    after = digest(INITIAL)
    result = {
        "schema_version": "chemvector-atom-aromatic-head-cpu-r17-0.1", "complete": True,
        "read_only": False, "training_performed": True, "gpu_used": False,
        "sealed_test_read": False, "production_checkpoint_written": False,
        "source_plan": str(PLAN.relative_to(ROOT)), "source_root": str(SOURCE.relative_to(ROOT)),
        "image_counts": {split: len(values) for split, values in splits.items()},
        "molecule_counts": {split: len(ids) for split, ids in id_sets.items()}, "identity_overlaps": overlaps,
        "coverage": coverage, "train_atom_rows": int(len(labels)), "train_positive_rows": int(labels.sum()),
        "head_parameters": sum(parameter.numel() for parameter in head.parameters()),
        "events": events, "selected_threshold": threshold, "train_threshold_sweep": train_sweep,
        "metrics": metrics, "initial_checkpoint_sha256": SHA, "initial_checkpoint_sha256_after": after,
        "gate": {
            "identity_disjoint": all(value == 0 for value in overlaps.values()), "train_threshold_only": True,
            "fresh_has_aromatic_positive": metrics["fresh_holdout"]["positive_rows"] > 0,
            "fresh_accuracy_ge_0.90": metrics["fresh_holdout"]["accuracy"] >= 0.90,
            "fresh_positive_recall_ge_0.90": metrics["fresh_holdout"]["positive_recall"] is not None and metrics["fresh_holdout"]["positive_recall"] >= 0.90,
            "checkpoint_unchanged": after == SHA, "sealed_test_read": False, "production_go": False,
        },
    }
    OUT.mkdir(parents=True, exist_ok=False)
    torch.save({"schema_version": result["schema_version"], "state_dict": head.state_dict(), "initial_checkpoint_sha256": SHA, "threshold": threshold}, OUT / "head-state.pt")
    (OUT / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT / "report.json"), "threshold": threshold, "metrics": metrics, "coverage": coverage, "gate": result["gate"]}, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    run()
