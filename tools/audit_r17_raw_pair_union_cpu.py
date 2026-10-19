#!/usr/bin/env python3
"""CPU-only raw/selected AtomPair union diagnostic for r17 t2.

This is deliberately a read-only reachability audit.  The frozen decoder is
run twice per image: once on the selector-retained atom set and once on the
raw atom proposal set.  Pair predictions are never fed back into production.
Target coordinates are joined only after inference to answer one question:
are oa13/oa33 misses caused by pair-presence routing, or are the endpoints
already absent from the selected set?

No sealed artifact, optimizer, or production checkpoint is touched.
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

from chemvector.components.graph.joint.direct_atom_pair_bond import enumerate_unordered_atom_pairs
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.pixels import image_tensor
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from tools.train_direct_graph_pair_scorer_cpu_r9 import SHA, _map_atoms

MANIFEST = ROOT / os.environ.get(
    "R17_RAW_PAIR_MANIFEST",
    "outputs/datasets/chemvector-r17-targeted-source-20261018-t2/manifest.json",
)
SOURCE = ROOT / os.environ.get(
    "R17_RAW_PAIR_SOURCE",
    "outputs/datasets/chemvector-r17-targeted-source-20261018-t2/samples",
)
OUT = ROOT / os.environ.get(
    "R17_RAW_PAIR_OUT",
    "outputs/audits/chemvector-r17-raw-pair-union-cpu-20261018-t2",
)
PROFILES = tuple(os.environ.get("R17_RAW_PAIR_PROFILES", "oa13,oa33").split(","))
MOLECULES_PER_SPLIT = int(os.environ.get("R17_RAW_PAIR_MOLECULES", "2"))
PAIR_THRESHOLD = float(os.environ.get("R17_RAW_PAIR_THRESHOLD", "0.72"))
ATOM_THRESHOLD = -0.25
IMAGE_SIZE = 768.0


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def choose_rows() -> list[dict]:
    rows = json.loads(MANIFEST.read_text())["rows"]
    out = []
    for split in ("train", "holdout"):
        mids = sorted({str(r["molecule_id"]) for r in rows if r["split"] == split})
        mids = set(mids[:MOLECULES_PER_SPLIT])
        out.extend(r for r in rows if r["split"] == split
                   and str(r.get("profile_id")) in PROFILES
                   and str(r["molecule_id"]) in mids)
    out.sort(key=lambda r: str(r["sample_key"]))
    return out


def target(key: str):
    layers = json.loads((SOURCE / f"{key}.layers.json").read_text())
    graph = json.loads((SOURCE / f"{key}.graph.json").read_text())
    atoms = sorted(layers["atoms"], key=lambda a: int(a["atom_id"]))
    centers = torch.tensor([a["center"] for a in atoms], dtype=torch.float32)
    edge_types = {}
    for bond in graph["bonds"]:
        edge = tuple(sorted((int(bond["begin_atom"]), int(bond["end_atom"]))))
        edge_types[edge] = str(bond.get("chemical_type", "SINGLE")).upper()
    return centers, edge_types


def extend_mapping(raw_slots: torch.Tensor, selected_slots: torch.Tensor,
                   selected_mapping: dict[int, int], raw_points: torch.Tensor,
                   targets: torch.Tensor) -> dict[int, int]:
    """Map raw slots while preserving selected-slot identities.

    A plain greedy match over all raw proposals can let a distractor steal a
    target already owned by a selected slot.  That would make raw and selected
    pair sets incomparable.  Seed the raw map with the selected assignment,
    then greedily assign only remaining slots/targets.
    """
    selected_by_slot = {int(s): int(selected_mapping[i])
                        for i, s in enumerate(selected_slots.tolist())
                        if i in selected_mapping}
    used_targets = set(selected_by_slot.values())
    result = dict(selected_by_slot)
    candidates = []
    for i, slot in enumerate(raw_slots.tolist()):
        if int(slot) in result:
            continue
        for j in range(len(targets)):
            if j in used_targets:
                continue
            distance = float(torch.linalg.vector_norm(
                (raw_points[i] - targets[j] / IMAGE_SIZE) * IMAGE_SIZE))
            if distance <= 16.0:
                candidates.append((distance, int(slot), j))
    for _, slot, target_id in sorted(candidates):
        if slot not in result and target_id not in used_targets:
            result[slot] = target_id
            used_targets.add(target_id)
    return {i: result[int(slot)] for i, slot in enumerate(raw_slots.tolist()) if int(slot) in result}


def run() -> dict:
    if OUT.exists():
        raise FileExistsError(OUT)
    if digest(INITIAL) != SHA:
        raise RuntimeError("production checkpoint SHA changed")
    rows = choose_rows()
    if not rows:
        raise RuntimeError("empty diagnostic slice")
    model, config, _ = load_bundle_for_probe(INITIAL, SHA)
    model.cpu().eval().requires_grad_(False)
    torch.set_num_threads(1)
    totals = defaultdict(Counter)
    records = []
    with torch.inference_mode():
        for row in rows:
            key = str(row["sample_key"])
            centers, truth = target(key)
            pixels = image_tensor(Image.open(SOURCE / f"{key}.png").convert("RGB"))[None]
            outputs = structure_forward(model, pixels, config, ATOM_THRESHOLD)
            vision, raw, selected = outputs[0], outputs[3], outputs[6]
            raw_slots = torch.where(raw["atom_mask"][0])[0]
            sel_slots = torch.where(selected["atom_mask"][0])[0]
            sel_map = _map_atoms(selected["atom_geometry"][0, sel_slots, :2].cpu(), centers)
            # Preserve selected assignments when extending to raw proposals;
            # see extend_mapping() for why this matters.
            raw_slot_map = extend_mapping(
                raw_slots, sel_slots, sel_map,
                raw["atom_geometry"][0, raw_slots, :2].cpu(), centers,
            )
            selected_slot_map = {
                int(slot): int(target_id)
                for compact, slot in enumerate(sel_slots.tolist())
                if compact in sel_map
                for target_id in [sel_map[compact]]
            }
            raw_pairs, raw_rows = enumerate_unordered_atom_pairs(raw["atom_mask"])
            sel_pairs, sel_rows = enumerate_unordered_atom_pairs(selected["atom_mask"])
            raw_direct = model.pair(vision, raw["atom_features"], raw["atom_geometry"][..., :2], raw["atom_scores"], raw_pairs, raw_rows)
            sel_direct = model.pair(vision, selected["atom_features"], selected["atom_geometry"][..., :2], selected["atom_scores"], sel_pairs, sel_rows)
            raw_score = raw_direct["presence_logits"].float().sigmoid()
            sel_score = sel_direct["presence_logits"].float().sigmoid()
            raw_above = raw_score >= PAIR_THRESHOLD
            sel_above = sel_score >= PAIR_THRESHOLD
            # The union's useful part is raw pairs absent from the selected
            # route.  The set comparison is by target identity, post-hoc.
            def true_edges(pairs, slots, mapping, above):
                sm = {int(s): i for i, s in enumerate(slots.tolist())}
                out = set()
                for i in torch.where(above)[0].tolist():
                    a, b = (int(x) for x in pairs[i].tolist())
                    if a in sm and b in sm and int(slots[sm[a]]) in mapping and int(slots[sm[b]]) in mapping:
                        edge = tuple(sorted((mapping[int(slots[sm[a]])], mapping[int(slots[sm[b]])])))
                        if edge in truth: out.add(edge)
                return out
            raw_true = true_edges(raw_pairs, raw_slots, raw_slot_map, raw_above)
            sel_true = true_edges(sel_pairs, sel_slots, selected_slot_map, sel_above)
            raw_reachable = set()
            raw_slot_lookup = set(int(s) for s in raw_slots.tolist())
            for i, (a, b) in enumerate(raw_pairs.tolist()):
                if a in raw_slot_lookup and b in raw_slot_lookup and a in raw_slot_map and b in raw_slot_map:
                    edge = tuple(sorted((raw_slot_map[a], raw_slot_map[b])))
                    if edge in truth: raw_reachable.add(edge)
            rowrec = {"split": row["split"], "profile": row.get("profile_id"), "sample_key": key,
                      "raw_atoms": len(raw_slots), "selected_atoms": len(sel_slots),
                      "raw_mapped_atoms": len(raw_slot_map), "selected_mapped_atoms": len(sel_map),
                      "target_edges": len(truth), "raw_pair_rows": len(raw_pairs),
                      "selected_pair_rows": len(sel_pairs), "raw_reachable_true_edges": len(raw_reachable),
                      "raw_route_true_edges": len(raw_true), "selected_route_true_edges": len(sel_true),
                      "raw_route_edge_gain": len(raw_true - sel_true),
                      "raw_reachable_edge_gain": len(raw_reachable - sel_true),
                      "raw_only_route_true_edges": len(raw_true - sel_true),
                      "raw_only_route_pairs": int(raw_above.sum()) - int(sel_above.sum()),
                      "selector_dropped_true_edges": len(raw_reachable - set(sel_true))}
            records.append(rowrec)
            c = totals[(str(row["split"]), str(row.get("profile_id")))]
            for k, v in rowrec.items():
                if isinstance(v, int): c[k] += v
    summary = {f"{s}|{p}": dict(c) for (s, p), c in sorted(totals.items())}
    result = {"schema_version": "chemvector-r17-raw-pair-union-cpu-0.1", "complete": True,
              "read_only": True, "training_performed": False, "gpu_used": False,
              "sealed_test_read": False, "production_checkpoint_written": False,
              "manifest": str(MANIFEST.relative_to(ROOT)), "source": str(SOURCE.relative_to(ROOT)),
              "initial_checkpoint_sha256": SHA, "initial_checkpoint_sha256_after": digest(INITIAL),
              "pair_threshold": PAIR_THRESHOLD, "profiles": list(PROFILES),
              "molecules_per_split": MOLECULES_PER_SPLIT, "summary": summary, "rows": records,
              "gate": {"checkpoint_unchanged": digest(INITIAL) == SHA, "target_free_routing": True,
                       "raw_union_only_diagnostic": True, "sealed_test_read": False,
                       "production_go": False}}
    OUT.mkdir(parents=True)
    (OUT / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(OUT / "report.json"), "summary": summary, "gate": result["gate"]}, indent=2))
    return result


if __name__ == "__main__":
    run()
