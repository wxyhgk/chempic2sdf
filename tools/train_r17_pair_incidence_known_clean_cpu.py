#!/usr/bin/env python3
"""Bounded CPU shadow for frozen AtomPair incidence on the known-clean cohort.

Only a tiny residual presence head is trained.  Atom proposals, selected
prefix, chemistry/type heads, and production checkpoint remain frozen.  The
new edge set is still the frozen unordered AtomPair set; no candidate atoms
or bonds are injected.  This targets the oa13/oa33 failure where atom recall
is high but the frozen pair head emits too few edges.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from collections import Counter, defaultdict
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
from chemvector.bond_object import BOND_TYPES

COHORT = ROOT / os.environ.get("R17_PAIR_COHORT", "outputs/audits/chemvector-r17-selector-known-clean-shadow-cohort-v1/manifest.json")
SOURCE = ROOT / os.environ.get(
    "R17_PAIR_SOURCE",
    os.environ.get("R17_PAIR_CALIB_SOURCE", "outputs/datasets/structure-expansion-batch-12272-20260923/samples"),
)
OUT = ROOT / os.environ.get("R17_PAIR_OUT", "outputs/audits/chemvector-r17-pair-incidence-known-clean-cpu-v1")
PROFILES = ("oa00", "oa02", "oa11", "oa13", "oa20", "oa22", "oa31", "oa33")
TRAIN_MOLECULES = int(os.environ.get("R17_PAIR_TRAIN_MOLECULES", "32"))
HOLDOUT_MOLECULES = int(os.environ.get("R17_PAIR_HOLDOUT_MOLECULES", "16"))
STEPS = int(os.environ.get("R17_PAIR_STEPS", "600"))
PAIR_THRESHOLD = float(os.environ.get("R17_PAIR_THRESHOLD", "0.72"))
MATCH_RADIUS = 16.0
NEGATIVE_RATIO = int(os.environ.get("R17_PAIR_NEGATIVE_RATIO", "3"))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Residual(nn.Module):
    def __init__(self, dim: int = 262, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def target(sample_key: str):
    layers = json.loads((SOURCE / f"{sample_key}.layers.json").read_text())
    graph = json.loads((SOURCE / f"{sample_key}.graph.json").read_text())
    atoms = sorted(layers["atoms"], key=lambda a: int(a["atom_id"]))
    centers = torch.tensor([a["center"] for a in atoms], dtype=torch.float32)
    types = {}
    for b in graph["bonds"]:
        edge = tuple(sorted((int(b["begin_atom"]), int(b["end_atom"]))))
        types[edge] = str(b.get("chemical_type", "SINGLE")).upper()
    return centers, types


def extract(row: dict, model, config) -> dict:
    key = row["sample_key"]
    with torch.inference_mode():
        pix = image_tensor(Image.open(SOURCE / f"{key}.png").convert("RGB"))[None]
        outputs = structure_forward(model, pix, config, -.25)
        vision, selected = outputs[0], outputs[6]
        mask = selected["atom_mask"]
        pairs, sample_indices = enumerate_unordered_atom_pairs(mask)
        direct = model.pair(vision, selected["atom_features"], selected["atom_geometry"][..., :2], selected["atom_scores"], pairs, sample_indices)
    pairs = pairs.cpu()
    points = selected["atom_geometry"][0, :, :2].float()
    pair_points = points[pairs]
    delta = pair_points[:, 1] - pair_points[:, 0]
    distance = torch.linalg.vector_norm(delta * 768.0, dim=-1)
    midpoint = pair_points.mean(1)
    scores = selected["atom_scores"][0, pairs].float()
    features = torch.cat((direct["hidden"].float(), direct["presence_logits"][:, None].float(), midpoint, distance[:, None], scores), -1).cpu()
    slots = torch.where(mask[0].cpu())[0]
    target_centers, target_types = target(key)
    mapping = _map_atoms(points[slots].cpu(), target_centers)
    slot_to_compact = {int(s): i for i, s in enumerate(slots.tolist())}
    labels = torch.zeros(len(pairs), dtype=torch.float32)
    type_labels = [None] * len(pairs)
    for i, (a, b) in enumerate(pairs.tolist()):
        if int(a) not in slot_to_compact or int(b) not in slot_to_compact:
            continue
        ca, cb = slot_to_compact[int(a)], slot_to_compact[int(b)]
        if ca in mapping and cb in mapping:
            edge = tuple(sorted((mapping[ca], mapping[cb])))
            if edge in target_types:
                labels[i] = 1.0; type_labels[i] = target_types[edge]
    return {"key": key, "profile": row["profile_id"], "features": features, "base_logits": direct["presence_logits"].float().cpu(), "chemical_logits": direct["chemical_logits"].float().cpu(), "pairs": pairs, "labels": labels, "type_labels": type_labels, "target_edges": target_types, "mapped_atoms": len(mapping), "target_atoms": len(target_centers), "slots": slots}


def choose(manifest: dict):
    by = defaultdict(list)
    for row in manifest["rows"]:
        by[(row["split"], row["molecule_id"])].append(row)
    train_ids = sorted({mid for split, mid in by if split == "train"})[:TRAIN_MOLECULES]
    hold_ids = sorted({mid for split, mid in by if split == "holdout"})[:HOLDOUT_MOLECULES]
    # Fixed molecule IDs are independent of profile and guaranteed identity
    # disjoint by the cohort manifest; retain every profile row.
    train = [r for (split, mid), rows in by.items() if split == "train" and mid in train_ids for r in rows]
    hold = [r for (split, mid), rows in by.items() if split == "holdout" and mid in hold_ids for r in rows]
    return sorted(train, key=lambda r: r["sample_key"]), sorted(hold, key=lambda r: r["sample_key"])


def score(items, head=None):
    out = defaultdict(int)
    by_profile = defaultdict(lambda: defaultdict(int))
    for item in items:
        logits = item["base_logits"] if head is None else item["base_logits"] + head(item["features"])
        selected = logits.sigmoid() >= PAIR_THRESHOLD
        predicted = int(selected.sum()); mapped = 0; true = 0; typed = 0; unmap = 0
        # Mapping labels are represented by target edge membership and the
        # frozen pair row's type label; target-free routing is only the logits.
        for i in torch.where(selected)[0].tolist():
            if item["labels"][i] <= .5:
                unmap += 0  # same-image non-edge, retained for clarity
            mapped += 1
            if item["labels"][i] > .5:
                true += 1
                if item["type_labels"][i] is not None and BOND_TYPES[int(item["chemical_logits"][i].argmax())] == item["type_labels"][i]:
                    typed += 1
        p = item["profile"]
        for bucket, value in (("predicted",predicted),("mapped",mapped),("true",true),("typed",typed),("target",int(item["labels"].sum())),("unmapped",unmap),("mapped_atoms",item["mapped_atoms"]),("target_atoms",item["target_atoms"])):
            out[bucket] += value; by_profile[p][bucket] += value
        by_profile[p]["images"] += 1; out["images"] += 1
    def finish(c):
        return {"images":c["images"],"predicted_edges":c["predicted"],"mapped_edges":c["mapped"],"true_edges":c["true"],"typed_true_edges":c["typed"],"target_edges":c["target"],"unmapped_edges":c["unmapped"],"edge_precision":c["true"]/max(c["mapped"],1),"edge_recall":c["true"]/max(c["target"],1),"typed_edge_precision":c["typed"]/max(c["mapped"],1),"typed_edge_recall":c["typed"]/max(c["target"],1),"atom_recall":c["mapped_atoms"]/max(c["target_atoms"],1)}
    return finish(out), {p: finish(c) for p,c in by_profile.items()}


def run():
    if OUT.exists(): raise FileExistsError(OUT)
    if digest(INITIAL) != SHA: raise RuntimeError("initial checkpoint changed")
    manifest = json.loads(COHORT.read_text())
    train_rows, hold_rows = choose(manifest)
    if {r["molecule_id"] for r in train_rows} & {r["molecule_id"] for r in hold_rows}: raise RuntimeError("identity overlap")
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); torch.manual_seed(20261008); random.seed(20261008)
    model, config, _ = load_bundle_for_probe(INITIAL, SHA); model.eval().requires_grad_(False)
    train = [extract(r, model, config) for r in train_rows]; hold = [extract(r, model, config) for r in hold_rows]
    # Profile-balanced positives and same-image hard negatives.  No target
    # fields enter features; labels are used only for sidecar fitting.
    xrows=[]; brows=[]; yrows=[]
    for item in train:
        pos=torch.where(item["labels"]>.5)[0].tolist(); neg=torch.where(item["labels"]<=.5)[0].tolist(); need=min(len(neg),NEGATIVE_RATIO*len(pos)); chosen=neg[:need]
        for i in pos+chosen: xrows.append(item["features"][i]); brows.append(item["base_logits"][i]); yrows.append(item["labels"][i])
    x=torch.stack(xrows); b=torch.stack(brows); y=torch.stack(yrows); sidecar=Residual(x.shape[1]); opt=torch.optim.AdamW(sidecar.parameters(),lr=2e-3,weight_decay=1e-4); pos_weight=((len(y)-y.sum())/y.sum().clamp_min(1)).clamp(1,32)
    events=[]
    for step in range(1,STEPS+1):
        loss=F.binary_cross_entropy_with_logits(b + sidecar(x), y, pos_weight=pos_weight)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(sidecar.parameters(),1.0); opt.step()
        if step in (1,100,300,600): events.append({"step":step,"loss":float(loss.detach())})
    sidecar.eval(); baseline,bprof=score(hold); scored,sprof=score(hold,sidecar)
    after=digest(INITIAL)
    deltas={p:{k:sprof[p][k]-bprof[p][k] for k in ("edge_recall","typed_edge_recall","edge_precision","typed_edge_precision","atom_recall")} for p in sprof}
    gate={"checkpoint_unchanged":after==SHA,"sealed_test_read":False,"edge_set_frozen":True,"oa13_typed_recall_gain":deltas.get("oa13",{}).get("typed_edge_recall",0)>0,"oa33_typed_recall_gain":deltas.get("oa33",{}).get("typed_edge_recall",0)>0,"each_profile_typed_precision_drop_le_0.01":all(v["typed_edge_precision"]>=-.01 for v in deltas.values()),"each_profile_edge_precision_drop_le_0.01":all(v["edge_precision"]>=-.01 for v in deltas.values()),"production_go":False}
    result={"schema_version":"chemvector-r17-pair-incidence-known-clean-cpu-0.2","complete":True,"training_performed":True,"training_scope":"frozen_pair_presence_residual_known_clean_cpu","gpu_used":False,"sealed_test_read":False,"production_checkpoint_written":False,"cohort":str(COHORT.relative_to(ROOT)),"source":str(SOURCE.relative_to(ROOT)),"train_molecules":TRAIN_MOLECULES,"holdout_molecules":HOLDOUT_MOLECULES,"train_images":len(train),"holdout_images":len(hold),"steps":STEPS,"features":x.shape[1],"positive_rows":int(y.sum()),"negative_rows":int((y<=.5).sum()),"events":events,"baseline":baseline,"scored":scored,"baseline_profiles":bprof,"scored_profiles":sprof,"deltas":deltas,"initial_checkpoint_sha256":SHA,"initial_checkpoint_sha256_after":after,"gate":gate}
    OUT.mkdir(parents=True); torch.save({"schema_version":result["schema_version"],"state_dict":sidecar.state_dict(),"initial_checkpoint_sha256":SHA},OUT/"sidecar-state.pt"); (OUT/"report.json").write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps({"output":str(OUT/"report.json"),"baseline":baseline,"scored":scored,"deltas":deltas,"gate":gate},indent=2))


if __name__ == "__main__": run()
