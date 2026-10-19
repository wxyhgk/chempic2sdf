#!/usr/bin/env python3
"""Frozen-initial raw-vs-selected atom endpoint coverage audit for r17 m128.

The model is only used for inference.  Target geometry is joined after the
raw/selected proposal masks are produced, so it cannot influence decisions.
This separates missing raw proposals from selector rejection before any atom
head training is considered.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.pixels import image_tensor
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from tools.train_direct_graph_pair_scorer_cpu_r9 import SHA

CACHE = ROOT / "outputs/audits/chemvector-graph-context-aromatic-cpu-r17-all-m128-t0p72-c1-panel128/feature-cache.pt"
SOURCE = ROOT / "outputs/datasets/structure-expansion-batch-12272-20260923/samples"
OUT = ROOT / os.environ.get("R17_RAW_ENDPOINT_OUT", "outputs/audits/chemvector-r17-raw-selected-endpoint-m128")
ATOM_THRESHOLD = -0.25
MATCH_RADIUS = float(os.environ.get("R17_RAW_MATCH_RADIUS", "16.0"))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def target_layers(sample_key: str):
    layers = json.loads((SOURCE / f"{sample_key}.layers.json").read_text())
    graph = json.loads((SOURCE / f"{sample_key}.graph.json").read_text())
    atoms = sorted(layers["atoms"], key=lambda x: int(x["atom_id"]))
    centers = torch.tensor([a["center"] for a in atoms], dtype=torch.float32) / 768.0
    bonds = {tuple(sorted((int(b["begin_atom"]), int(b["end_atom"])))) for b in graph["bonds"]}
    return atoms, centers, bonds


def match(points: torch.Tensor, targets: torch.Tensor) -> dict[int, int]:
    if not len(points) or not len(targets):
        return {}
    distances = torch.cdist(points.float() * 768.0, targets.float() * 768.0)
    candidates = [(float(distances[i, j]), i, j)
                  for i in range(distances.size(0)) for j in range(distances.size(1))
                  if float(distances[i, j]) <= MATCH_RADIUS]
    result, used = {}, set()
    for _, i, j in sorted(candidates):
        if i not in result and j not in used:
            result[i] = j; used.add(j)
    return result


def run() -> dict:
    if OUT.exists():
        raise FileExistsError(OUT)
    if digest(INITIAL) != SHA:
        raise RuntimeError("production initial checkpoint changed")
    cache = torch.load(CACHE, map_location="cpu", weights_only=False)
    if cache.get("initial_checkpoint_sha256") != SHA:
        raise RuntimeError("cache/checkpoint binding mismatch")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(1)
    model, config, _ = load_bundle_for_probe(INITIAL, SHA)
    model.to(device).eval().requires_grad_(False)
    rows = []
    with torch.inference_mode():
        for split, records in cache["records"].items():
            for record in records:
                sample_key = record["sample_key"]
                atoms, centers, target_bonds = target_layers(sample_key)
                pixels = image_tensor(Image.open(SOURCE / f"{sample_key}.png").convert("RGB"))[None].to(device)
                outputs = structure_forward(model, pixels, config, ATOM_THRESHOLD)
                raw, selected = outputs[3], outputs[6]
                raw_points = raw["atom_geometry"][0, raw["atom_mask"][0], :2].detach().cpu()
                selected_points = selected["atom_geometry"][0, selected["atom_mask"][0], :2].detach().cpu()
                raw_map = match(raw_points, centers)
                selected_map = match(selected_points, centers)
                raw_target_ids = set(raw_map.values()); selected_target_ids = set(selected_map.values())
                category_counts = defaultdict(Counter)
                for atom in atoms:
                    aid = int(atom["atom_id"]); element = str(atom.get("element") or "C").upper()
                    if bool(atom.get("visible_text")) and element in {"O", "N"}:
                        category = f"visible_{element}"
                    elif not bool(atom.get("visible_text")) and element == "C":
                        category = "implicit_C"
                    elif bool(atom.get("visible_text")):
                        category = "visible_other"
                    else:
                        category = "implicit_other"
                    state = "selected" if aid in selected_target_ids else ("selector_drop" if aid in raw_target_ids else "raw_missing")
                    category_counts[category][state] += 1
                def reachable(mapping):
                    ids = set(mapping.values())
                    return sum(int(a in ids and b in ids) for a, b in target_bonds)
                profile = sample_key.rsplit("__", 1)[-1]
                row = {"split": split, "profile": profile, "sample_key": sample_key,
                       "target_atoms": len(atoms), "raw_candidates": len(raw_points), "selected_candidates": len(selected_points),
                       "raw_matched_atoms": len(raw_target_ids), "selected_matched_atoms": len(selected_target_ids),
                       "target_bonds": len(target_bonds), "raw_reachable_bonds": reachable(raw_map),
                       "selected_reachable_bonds": reachable(selected_map),
                       "categories": {key: dict(value) for key, value in category_counts.items()}}
                rows.append(row)
    summary = {}
    for split in ("train", "dev", "fresh_holdout"):
        subset = [r for r in rows if r["split"] == split]
        summary[split] = {"images": len(subset), "target_atoms": sum(r["target_atoms"] for r in subset),
                          "raw_candidates": sum(r["raw_candidates"] for r in subset), "selected_candidates": sum(r["selected_candidates"] for r in subset),
                          "raw_matched_atoms": sum(r["raw_matched_atoms"] for r in subset), "selected_matched_atoms": sum(r["selected_matched_atoms"] for r in subset),
                          "target_bonds": sum(r["target_bonds"] for r in subset), "raw_reachable_bonds": sum(r["raw_reachable_bonds"] for r in subset),
                          "selected_reachable_bonds": sum(r["selected_reachable_bonds"] for r in subset),
                          "profiles": {}}
        for profile in ("oa00", "oa11", "oa20", "oa33"):
            part = [r for r in subset if r["profile"] == profile]
            summary[split]["profiles"][profile] = {"images": len(part), "target_atoms": sum(r["target_atoms"] for r in part),
                "raw_matched_atoms": sum(r["raw_matched_atoms"] for r in part), "selected_matched_atoms": sum(r["selected_matched_atoms"] for r in part),
                "target_bonds": sum(r["target_bonds"] for r in part), "raw_reachable_bonds": sum(r["raw_reachable_bonds"] for r in part),
                "selected_reachable_bonds": sum(r["selected_reachable_bonds"] for r in part)}
    result = {"schema_version": "chemvector-r17-raw-selected-endpoint-m128-0.1", "complete": True,
              "read_only": True, "training_performed": False, "gpu_used": device.type == "cuda",
              "sealed_test_read": False, "production_checkpoint_written": False, "device": str(device),
              "initial_checkpoint_sha256": SHA, "initial_checkpoint_sha256_after": digest(INITIAL),
              "cache": str(CACHE.relative_to(ROOT)), "summary": summary, "rows": rows,
              "gate": {"checkpoint_unchanged": digest(INITIAL) == SHA, "sealed_test_read": False,
                       "raw_reachability_reported": True, "selected_vs_raw_separated": True, "production_go": False}}
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT / "report.json"), "device": str(device), "summary": summary, "gate": result["gate"]}, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    run()
