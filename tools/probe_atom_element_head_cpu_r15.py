#!/usr/bin/env python3
"""CPU-only probe for element recoverability from frozen AtomObject features."""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib, json, sys
from pathlib import Path
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.pixels import image_tensor
from tools.train_direct_graph_pair_scorer_cpu_r9 import DATA, SHA, _map_atoms, _target, ATOM_THRESHOLD, rows

OUT = ROOT / "outputs/audits/chemvector-atom-element-head-cpu-r15"
ELEMENTS = ("C", "O", "N", "S", "F", "Cl", "Br", "I", "P", "B", "Si")

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def train_rows(all_rows):
    by = defaultdict(list)
    for row in sorted((r for r in all_rows if r.get("split") == "train"), key=lambda r: r["sample_key"]):
        by[row["profile_id"]].append(row)
    return [row for profile in sorted(by) for row in by[profile][:96]]

def extract(row, model, config):
    base = DATA / "samples" / row["sample_key"]
    image = Image.open(base.with_suffix(".png")).convert("RGB")
    with torch.inference_mode():
        output = structure_forward(model, image_tensor(image)[None], config, ATOM_THRESHOLD)
    selected = output[6]
    slots = torch.where(selected["atom_mask"][0])[0]
    points = selected["atom_geometry"][0, slots, :2].float()
    targets, _, _ = _target(row["sample_key"])
    mapping = _map_atoms(points, targets)
    layers = json.loads(base.with_suffix(".layers.json").read_text())
    atoms = sorted(layers["atoms"], key=lambda atom: int(atom["atom_id"]))
    features, labels = [], []
    for compact, target in mapping.items():
        element = str(atoms[target].get("element") or "C")
        if element in ELEMENTS:
            features.append(selected["atom_features"][0, slots[compact]].float())
            labels.append(ELEMENTS.index(element))
    if not features:
        return torch.empty((0, 128)), torch.empty((0,), dtype=torch.long)
    return torch.stack(features), torch.tensor(labels, dtype=torch.long)

def run():
    if digest(INITIAL) != SHA:
        raise RuntimeError("production checkpoint changed")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20261015)
    all_rows = rows()
    train = train_rows(all_rows)
    dev = sorted((r for r in all_rows if r.get("split") == "dev"), key=lambda r: r["sample_key"])
    if {r["molecule_id"] for r in train} & {r["molecule_id"] for r in dev}:
        raise RuntimeError("train/dev identity overlap")
    model, config, _ = load_bundle_for_probe(INITIAL, SHA)
    model.eval().requires_grad_(False)
    train_data = [extract(row, model, config) for row in train]
    dev_data = [extract(row, model, config) for row in dev]
    features = torch.cat([x for x, _ in train_data if len(x)])
    labels = torch.cat([y for _, y in train_data if len(y)])
    head = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, len(ELEMENTS)))
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=1e-4)
    counts = Counter(labels.tolist())
    weights = torch.tensor([(len(labels) / max(counts[i], 1)) ** 0.5 for i in range(len(ELEMENTS))])
    events = []
    for step in range(1, 301):
        head.train(); optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(head(features), labels, weight=weights)
        loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); optimizer.step()
        if step in (1, 100, 200, 300):
            events.append({"step": step, "loss": float(loss.detach())})
    head.eval()
    predictions = torch.cat([head(x).argmax(-1) for x, _ in dev_data if len(x)])
    truth = torch.cat([y for _, y in dev_data if len(y)])
    by_element = {}
    for index, element in enumerate(ELEMENTS):
        mask = truth == index
        by_element[element] = {"rows": int(mask.sum()), "accuracy": float((predictions[mask] == index).float().mean()) if bool(mask.any()) else None}
    accuracy = float((predictions == truth).float().mean())
    after = digest(INITIAL)
    result = {
        "schema_version": "chemvector-atom-element-head-cpu-r15-0.1", "complete": True,
        "read_only": False, "training_performed": True, "gpu_used": False,
        "sealed_test_read": False, "production_checkpoint_written": False,
        "train_images": len(train), "dev_images": len(dev), "train_rows": len(labels), "dev_rows": len(truth),
        "elements": ELEMENTS, "train_counts": {ELEMENTS[i]: counts[i] for i in range(len(ELEMENTS))},
        "head_parameters": sum(p.numel() for p in head.parameters()), "events": events,
        "dev_accuracy": accuracy, "dev_by_element": by_element,
        "initial_checkpoint_sha256": SHA, "initial_checkpoint_sha256_after": after,
        "gate": {"checkpoint_unchanged": after == SHA, "sealed_test_read": False,
                 "dev_accuracy_ge_0.90": accuracy >= .90,
                 "hetero_O_N_accuracy_ge_0.85": all(by_element[e]["accuracy"] is not None and by_element[e]["accuracy"] >= .85 for e in ("O", "N")),
                 "production_go": False},
    }
    OUT.mkdir(parents=True, exist_ok=False)
    torch.save({"schema_version": result["schema_version"], "state_dict": head.state_dict(), "initial_checkpoint_sha256": SHA}, OUT / "head-state.pt")
    (OUT / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT / "report.json"), "dev_accuracy": accuracy, "by_element": by_element, "gate": result["gate"]}, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    run()
