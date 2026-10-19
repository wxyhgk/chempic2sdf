#!/usr/bin/env python3
"""CPU graph-context aromatic edge sidecar on the admitted r17 slice.

The sidecar sees only frozen AtomObject/AtomPair outputs, pair geometry and a
target-free cycle-support bit.  Targets are joined after feature extraction for
loss/evaluation only.  It never changes production weights or the candidate
edge set; aromatic atoms are derived from accepted aromatic edges in audit code.
"""
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
from chemvector.components.graph.joint.direct_atom_pair_bond import enumerate_unordered_atom_pairs
from tools.train_direct_graph_pair_scorer_cpu_r9 import SHA, _map_atoms

PLAN = ROOT / "outputs/audits/chemvector-aromatic-specialist-plan-r17-v2/rows.jsonl"
SOURCE = ROOT / "outputs/datasets/structure-expansion-batch-12272-20260923"
PROFILE = os.environ.get("R17_GRAPH_PROFILE", "oa00")
LIMIT = int(os.environ.get("R17_GRAPH_LIMIT_MOLECULES", "16"))
ATOM_THRESHOLD = -0.25
PAIR_THRESHOLD = float(os.environ.get("R17_GRAPH_PAIR_THRESHOLD", "0.72"))
REQUIRE_CYCLE = os.environ.get("R17_GRAPH_REQUIRE_CYCLE", "1") != "0"
TAG = os.environ.get("R17_GRAPH_RUN_TAG", "")
OUT = ROOT / f"outputs/audits/chemvector-graph-context-aromatic-cpu-r17-{PROFILE}-m{LIMIT}-t{str(PAIR_THRESHOLD).replace('.', 'p')}-c{int(REQUIRE_CYCLE)}{('-' + TAG) if TAG else ''}"
IMAGE_SIZE = 768.0


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def alternate_path(adjacency: dict[int, set[int]], left: int, right: int, skip: tuple[int, int]) -> bool:
    seen = {left}; stack = [left]
    while stack:
        node = stack.pop()
        for nxt in adjacency.get(node, ()):
            if tuple(sorted((node, nxt))) == skip:
                continue
            if nxt == right:
                return True
            if nxt not in seen:
                seen.add(nxt); stack.append(nxt)
    return False


def target(row: dict):
    base = SOURCE / "samples" / row["sample_key"]
    layers = json.loads(base.with_suffix(".layers.json").read_text())
    graph = json.loads(base.with_suffix(".graph.json").read_text())
    atoms = sorted(layers["atoms"], key=lambda atom: int(atom["atom_id"]))
    centers = torch.tensor([atom["center"] for atom in atoms], dtype=torch.float32)
    edges = {}
    for bond in graph["bonds"]:
        edge = tuple(sorted((int(bond["begin_atom"]), int(bond["end_atom"]))))
        edges[edge] = str(bond.get("chemical_type", "SINGLE")).upper()
    aromatic_atoms = {index for index, atom in enumerate(atoms) if bool(atom.get("aromatic", False))}
    return centers, edges, aromatic_atoms


def extract(row: dict, model, config) -> dict:
    base = SOURCE / "samples" / row["sample_key"]
    with torch.inference_mode():
        outputs = structure_forward(model, image_tensor(Image.open(base.with_suffix(".png")).convert("RGB"))[None], config, ATOM_THRESHOLD)
        selected = outputs[6]
        slots = torch.where(selected["atom_mask"][0])[0]
        pairs, sample_indices = enumerate_unordered_atom_pairs(selected["atom_mask"])
        direct = model.pair(outputs[0], selected["atom_features"], selected["atom_geometry"][..., :2], selected["atom_scores"], pairs, sample_indices)
    if not len(pairs):
        return {"sample_key": row["sample_key"], "features": torch.empty((0, 526)), "labels": torch.empty((0,)), "type_features": torch.empty((0, 262)), "type_labels": torch.empty((0,), dtype=torch.long), "candidate_keys": set(), "target_aromatic_atoms": set(), "target_elements": {}, "mapping": {}, "pairs": [], "atom_features": torch.empty((0, 128)), "edge_chemical_logits": torch.empty((0, 4))}
    all_points = selected["atom_geometry"][0, :, :2].float()
    pair_points = all_points[pairs]
    delta = pair_points[:, 1] - pair_points[:, 0]
    distance = torch.linalg.vector_norm(delta * IMAGE_SIZE, dim=-1)
    midpoint = pair_points.mean(dim=1)
    scores = selected["atom_scores"][0, pairs].float()
    presence_logits = direct["presence_logits"].float()
    chosen = torch.where(presence_logits.sigmoid() >= PAIR_THRESHOLD)[0].tolist()
    compact = {int(slot): index for index, slot in enumerate(slots.tolist())}
    edges = []
    adjacency: dict[int, set[int]] = {}
    for index in chosen:
        left, right = (int(value) for value in pairs[index].tolist())
        if left not in compact or right not in compact or compact[left] == compact[right]:
            continue
        a, b = compact[left], compact[right]
        edges.append((a, b, index)); adjacency.setdefault(a, set()).add(b); adjacency.setdefault(b, set()).add(a)
    centers, target_edges, target_aromatic_atoms = target(row)
    layers = json.loads((base.with_suffix(".layers.json")).read_text())
    target_elements = {int(atom["atom_id"]): str(atom.get("element") or "C") for atom in layers["atoms"]}
    points = selected["atom_geometry"][0, slots, :2].float()
    mapping = _map_atoms(points, centers)
    feature_rows, labels, type_labels = [], [], []
    candidate_keys = set()
    for a, b, index in edges:
        left, right = (int(value) for value in pairs[index].tolist())
        cycle = float(alternate_path(adjacency, a, b, tuple(sorted((a, b)))))
        endpoint_features = torch.cat((selected["atom_features"][0, left].float(), selected["atom_features"][0, right].float()))
        geometry = torch.cat((midpoint[index], delta[index], distance[index:index + 1], scores[index]))
        graph_context = torch.tensor([cycle, float(len(adjacency.get(a, ()))), float(len(adjacency.get(b, ())))], dtype=torch.float32)
        feat = torch.cat((direct["hidden"][index].float(), direct["chemical_logits"][index].float(), endpoint_features, geometry, graph_context))
        feature_rows.append(feat)
        target_edge = tuple(sorted((mapping[a], mapping[b]))) if a in mapping and b in mapping else None
        labels.append(float(target_edges.get(target_edge) == "AROMATIC") if target_edge is not None else 0.0)
        type_names = {"SINGLE": 0, "DOUBLE": 1, "TRIPLE": 2, "AROMATIC": 3}
        type_labels.append(type_names.get(target_edges.get(target_edge, "SINGLE"), 0) if target_edge is not None else 0)
        candidate_keys.add(tuple(sorted((a, b))))
    if not feature_rows:
        return {"sample_key": row["sample_key"], "features": torch.empty((0, 526)), "labels": torch.empty((0,)), "type_features": torch.empty((0, 262)), "type_labels": torch.empty((0,), dtype=torch.long), "candidate_keys": candidate_keys, "target_aromatic_atoms": target_aromatic_atoms, "target_elements": target_elements, "mapping": mapping, "pairs": edges, "atom_features": selected["atom_features"][0, slots].float(), "edge_chemical_logits": direct["chemical_logits"][torch.tensor([i for _, _, i in edges], dtype=torch.long)].float() if edges else torch.empty((0, 4))}
    edge_indices = torch.tensor([i for _, _, i in edges], dtype=torch.long)
    type_base_features = torch.cat((direct["hidden"].float(), presence_logits[:, None], midpoint, distance[:, None], scores), dim=-1)[edge_indices]
    return {"sample_key": row["sample_key"], "features": torch.stack(feature_rows), "labels": torch.tensor(labels, dtype=torch.float32), "type_features": type_base_features, "type_labels": torch.tensor(type_labels, dtype=torch.long), "candidate_keys": candidate_keys, "target_aromatic_atoms": target_aromatic_atoms, "target_elements": target_elements, "mapping": mapping, "pairs": edges, "atom_features": selected["atom_features"][0, slots].float(), "edge_chemical_logits": direct["chemical_logits"][edge_indices].float()}


class AromaticContextHead(nn.Module):
    def __init__(self, input_dim: int = 526):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, features):
        return self.net(features).squeeze(-1)


def edge_metrics(head, records, threshold: float) -> dict:
    rows = []
    for record in records:
        if len(record["features"]):
            probabilities = head(record["features"]).sigmoid()
            rows.append((record, probabilities))
    total = positive = predicted = correct = tp = 0
    for record, probabilities in rows:
        labels = record["labels"].bool(); pred = (probabilities >= threshold) & torch.tensor([bool(a and b) for a, b, _ in record["pairs"]], dtype=torch.bool)
        # cycle mask above is already represented in feature column; use the
        # last three feature values: cycle, endpoint degree, endpoint degree.
        cycle = record["features"][:, -3] > 0.5
        pred = (probabilities >= threshold) & (cycle if REQUIRE_CYCLE else torch.ones_like(cycle, dtype=torch.bool))
        total += len(labels); positive += int(labels.sum()); predicted += int(pred.sum()); correct += int((pred == labels).sum()); tp += int((pred & labels).sum())
    return {"candidate_edges": total, "target_aromatic_edges": positive, "predicted_aromatic_edges": predicted, "true_aromatic_edges": tp, "edge_accuracy": correct / max(total, 1), "aromatic_precision": tp / max(predicted, 1), "aromatic_recall": tp / max(positive, 1)}


def run() -> dict:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    if digest(INITIAL) != SHA:
        raise RuntimeError("production checkpoint changed")
    rows = [json.loads(line) for line in PLAN.read_text().splitlines() if line.strip() and (PROFILE == "all" or json.loads(line)["profile_id"] == PROFILE)]
    splits = {split: [row for row in rows if row["split"] == split] for split in ("train", "dev", "fresh_holdout")}
    for split in splits:
        ids = sorted({row["molecule_id"] for row in splits[split]})[:LIMIT]
        splits[split] = [row for row in splits[split] if row["molecule_id"] in set(ids)]
    ids = {split: {row["molecule_id"] for row in values} for split, values in splits.items()}
    overlaps = {f"{a}:{b}": len(ids[a] & ids[b]) for a, b in (("train", "dev"), ("train", "fresh_holdout"), ("dev", "fresh_holdout"))}
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); torch.manual_seed(20261018)
    model, config, _ = load_bundle_for_probe(INITIAL, SHA); model.eval().requires_grad_(False)
    records = {}
    for split, values in splits.items():
        print(json.dumps({"phase": "extract", "split": split, "images": len(values)}), flush=True)
        records[split] = [extract(row, model, config) for row in values]
    OUT.mkdir(parents=True, exist_ok=False)
    torch.save({"schema_version": "chemvector-graph-context-aromatic-cache-r17-0.1", "initial_checkpoint_sha256": SHA, "profile": PROFILE, "pair_threshold": PAIR_THRESHOLD, "records": records}, OUT / "feature-cache.pt")
    train_features = torch.cat([record["features"] for record in records["train"] if len(record["features"])])
    train_labels = torch.cat([record["labels"] for record in records["train"] if len(record["labels"])])
    head = AromaticContextHead(train_features.shape[-1])
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=1e-4)
    pos_weight = ((len(train_labels) - train_labels.sum()) / train_labels.sum().clamp_min(1.0)).clamp(1.0, 16.0)
    events = []
    for step in range(1, 501):
        head.train(); optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(head(train_features), train_labels, pos_weight=pos_weight)
        loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); optimizer.step()
        if step in (1, 100, 300, 500): events.append({"step": step, "loss": float(loss.detach())})
    head.eval()
    candidates = [round(x, 2) for x in torch.arange(0.30, 0.81, 0.02).tolist()]
    sweep = [(edge_metrics(head, records["train"], threshold), threshold) for threshold in candidates]
    threshold = max(sweep, key=lambda pair: (pair[0]["aromatic_precision"], pair[0]["aromatic_recall"], -abs(pair[1] - 0.5)))[1]
    metrics = {split: edge_metrics(head, records[split], threshold) for split in records}
    after = digest(INITIAL)
    result = {
        "schema_version": "chemvector-graph-context-aromatic-cpu-r17-0.1", "complete": True,
        "read_only": False, "training_performed": True, "gpu_used": False,
        "sealed_test_read": False, "production_checkpoint_written": False,
        "source_plan": str(PLAN.relative_to(ROOT)), "profile": PROFILE, "molecule_limit_per_split": LIMIT,
        "pair_threshold": PAIR_THRESHOLD, "require_cycle": REQUIRE_CYCLE,
        "image_counts": {split: len(values) for split, values in splits.items()}, "molecule_counts": {split: len(value) for split, value in ids.items()},
        "identity_overlaps": overlaps, "head_parameters": sum(p.numel() for p in head.parameters()),
        "events": events, "selected_threshold": threshold, "train_threshold_sweep": sweep, "metrics": metrics,
        "target_free_features": True, "edge_set_frozen": True, "initial_checkpoint_sha256": SHA, "initial_checkpoint_sha256_after": after,
        "gate": {
            "identity_disjoint": all(value == 0 for value in overlaps.values()), "fresh_aromatic_positive": metrics["fresh_holdout"]["target_aromatic_edges"] > 0,
            "fresh_aromatic_precision_ge_0.90": metrics["fresh_holdout"]["aromatic_precision"] >= 0.90,
            "fresh_aromatic_recall_ge_0.90": metrics["fresh_holdout"]["aromatic_recall"] >= 0.90,
            "checkpoint_unchanged": after == SHA, "sealed_test_read": False, "production_go": False,
        },
    }
    torch.save({"schema_version": result["schema_version"], "state_dict": head.state_dict(), "initial_checkpoint_sha256": SHA, "threshold": threshold}, OUT / "sidecar-state.pt")
    (OUT / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT / "report.json"), "threshold": threshold, "metrics": metrics, "gate": result["gate"]}, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    run()
