#!/usr/bin/env python3
"""Bounded selector-retention sidecar on the r17 m128 panel.

Vision/atom proposals, set head, pair head and chemistry remain frozen.  The
sidecar only ranks raw atom candidates that the frozen selector rejected; the
already-selected prefix is copied unchanged.  Target matches are joined after
feature extraction for the isolated CPU/GPU shadow and never enter features.
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
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.pixels import image_tensor
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from tools.train_direct_graph_pair_scorer_cpu_r9 import SHA

CACHE = ROOT / os.environ.get(
    "R17_SELECTOR_CACHE",
    "outputs/audits/chemvector-graph-context-aromatic-cpu-r17-all-m128-t0p72-c1-panel128/feature-cache.pt",
)
SOURCE = ROOT / os.environ.get(
    "R17_SELECTOR_SOURCE",
    "outputs/datasets/structure-expansion-batch-12272-20260923/samples",
)
MANIFEST_PATH = os.environ.get("R17_SELECTOR_MANIFEST")
TRAIN_MOLECULES = int(os.environ.get("R17_SELECTOR_TRAIN_MOLECULES", "0"))
HOLDOUT_MOLECULES = int(os.environ.get("R17_SELECTOR_HOLDOUT_MOLECULES", "0"))
OUT = ROOT / os.environ.get("R17_SELECTOR_OUT", "outputs/audits/chemvector-r17-selector-retention-m128")
MATCH_RADIUS = 16.0
ATOM_THRESHOLD = -0.25
MAX_STEPS = int(os.environ.get("R17_SELECTOR_STEPS", "200"))
MAX_NEW_PER_IMAGE = 16
PRECISION_FLOOR = 0.85
TRAIN_CALIBRATION_FLOOR = float(os.environ.get("R17_SELECTOR_TRAIN_FLOOR", "0.95"))
THRESHOLDS = tuple(round(0.20 + i * 0.05, 2) for i in range(16))


class SelectorSidecar(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x): return self.net(x).squeeze(-1)


def digest(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def target_data(key):
    layers = json.loads((SOURCE / f"{key}.layers.json").read_text())
    graph = json.loads((SOURCE / f"{key}.graph.json").read_text())
    atoms = sorted(layers["atoms"], key=lambda x: int(x["atom_id"]))
    centers = torch.tensor([a["center"] for a in atoms], dtype=torch.float32)
    edges = {tuple(sorted((int(b["begin_atom"]), int(b["end_atom"])))) for b in graph["bonds"]}
    return atoms, centers, edges


def match(points, centers):
    if not len(points) or not len(centers): return {}
    d = torch.cdist(points.float() * 768.0, centers.float())
    cand = [(float(d[i, j]), i, j) for i in range(d.size(0)) for j in range(d.size(1)) if float(d[i, j]) <= MATCH_RADIUS]
    out, used = {}, set()
    for _, i, j in sorted(cand):
        if i not in out and j not in used: out[i] = j; used.add(j)
    return out


def extract(model, config, record, device):
    key = record["sample_key"]
    atoms, centers, target_edges = target_data(key)
    pixels = image_tensor(Image.open(SOURCE / f"{key}.png").convert("RGB"))[None].to(device)
    with torch.inference_mode():
        outputs = structure_forward(model, pixels, config, ATOM_THRESHOLD)
        raw, selected, set_output = outputs[3], outputs[6], outputs[5]
    raw_mask = raw["atom_mask"][0].bool(); selected_mask = selected["atom_mask"][0].bool()
    raw_slots = torch.where(raw_mask)[0]
    raw_points = raw["atom_geometry"][0, raw_slots, :2].detach().cpu()
    mapping = match(raw_points, centers)
    selected_slots = torch.where(selected_mask)[0].tolist()
    selected_points = selected["atom_geometry"][0, selected_mask, :2].detach().cpu()
    selected_map = match(selected_points, centers)
    selected_targets = set(selected_map.values())
    anchors = centers
    atom_features = raw["atom_features"][0, raw_slots].detach().float().cpu()
    xy = raw["atom_geometry"][0, raw_slots, :2].detach().float().cpu()
    score = raw["atom_scores"][0, raw_slots].detach().float().cpu()
    presence = set_output["atom_presence_logits"][0, raw_slots].detach().float().cpu()
    pre_presence = set_output.get("atom_pre_presence_logits", set_output["atom_presence_logits"])[0, raw_slots].detach().float().cpu()
    rows=[]
    for i, target in enumerate([mapping.get(j, -1) for j in range(len(raw_slots))]):
        slot = int(raw_slots[i])
        if slot in selected_slots: continue
        point = xy[i] * 768.0
        if target >= 0 and target not in selected_targets:
            label = 1
        elif target >= 0:
            continue
        else:
            nearest = float(torch.linalg.vector_norm(anchors - point, dim=1).min()) if len(anchors) else 1e9
            if nearest <= MATCH_RADIUS: continue
            label = 0
        feature = torch.cat((atom_features[i], score[i:i+1], xy[i], presence[i:i+1], pre_presence[i:i+1]))
        rows.append({"feature": feature, "label": label, "target": int(target), "style": key.rsplit("__",1)[-1], "selected_targets": selected_targets,
                     "baseline_targets": selected_targets, "target_edges": target_edges})
    return rows, selected_targets, target_edges, set(mapping.values())

def load_records():
    if not MANIFEST_PATH:
        return torch.load(CACHE,map_location="cpu",weights_only=False)
    rows=json.loads(Path(MANIFEST_PATH).read_text())['rows']
    grouped={"train":{},"holdout":{}}
    for row in rows:
        split='fresh_holdout' if row['split']=='holdout' else row['split']
        grouped.setdefault(split,{}).setdefault(row['molecule_id'],[]).append({'sample_key':row['sample_key']})
    records={'train':[],'dev':[],'fresh_holdout':[]}
    limits={'train':TRAIN_MOLECULES,'fresh_holdout':HOLDOUT_MOLECULES}
    for split,by in grouped.items():
        if split not in limits: continue
        ids=sorted(by)
        if limits[split] <= 0: raise ValueError(f'{split} molecule limit must be positive in manifest mode')
        for mid in ids[:limits[split]]: records[split].extend(sorted(by[mid],key=lambda r:r['sample_key']))
    return {'initial_checkpoint_sha256':SHA,'records':records,'manifest':str(Path(MANIFEST_PATH).resolve()),'molecule_limits':limits}


def evaluate(items, scorer, mean, std, threshold):
    stats = Counter(); detail=[]
    for rows, selected_targets, target_edges in items:
        if not rows:
            detail.append((selected_targets, selected_targets, target_edges)); continue
        device = next(scorer.parameters()).device
        x=torch.stack([r["feature"] for r in rows]).to(device); y=torch.tensor([r["label"] for r in rows], device=device)
        p=scorer((x-mean.to(device))/std.to(device)).sigmoid(); chosen=torch.where(p>=threshold)[0]
        if len(chosen)>MAX_NEW_PER_IMAGE: chosen=chosen[torch.argsort(p[chosen],descending=True)[:MAX_NEW_PER_IMAGE]]
        positive=int((y[chosen]==1).sum()); negative=int((y[chosen]==0).sum())
        new_targets=set(selected_targets)|{int(rows[int(i)]["target"]) for i in chosen.tolist() if int(rows[int(i)]["target"])>=0}
        stats.update(selected=int(len(chosen)), positive=positive, negative=negative, available=int((y==1).sum()))
        detail.append((selected_targets,new_targets,target_edges))
    base_reach=sum(sum(int(a in b and c in b) for a,c in edges) for b,_,edges in detail)
    new_reach=sum(sum(int(a in b and c in b) for a,c in edges) for _,b,edges in detail)
    base_atoms=sum(len(b) for b,_,_ in detail); new_atoms=sum(len(b) for _,b,_ in detail)
    return {"selected":stats["selected"],"positive":stats["positive"],"negative":stats["negative"],"available_positive":stats["available"],"conservative_precision":stats["positive"]/max(stats["selected"],1),"atom_retention":base_atoms/max(base_atoms,1),"new_atom_count":new_atoms,"baseline_reachable_bonds":base_reach,"new_reachable_bonds":new_reach,"bond_reachability_gain":new_reach-base_reach}


def run():
    if OUT.exists(): raise FileExistsError(OUT)
    if digest(INITIAL)!=SHA: raise RuntimeError("checkpoint changed")
    cache=load_records()
    if cache.get("initial_checkpoint_sha256")!=SHA: raise RuntimeError("cache binding mismatch")
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model,config,_=load_bundle_for_probe(INITIAL,SHA);model.to(device).eval().requires_grad_(False)
    split_items={}
    with torch.inference_mode():
        for split,records in cache["records"].items():
            split_items[split]=[]
            for record in records:
                rows,selected,edges,raw=extract(model,config,record,device)
                split_items[split].append((rows,selected,edges))
    train_flat=[r for rows,_,_ in split_items["train"] for r in rows if r["label"]>=0]
    if sum(r["label"]==1 for r in train_flat)<20: raise RuntimeError("too few positives")
    mat=torch.stack([r["feature"] for r in train_flat]);mean=mat.mean(0);std=mat.std(0).clamp_min(1e-5);x=(mat-mean)/std;y=torch.tensor([r["label"] for r in train_flat]).float()
    scorer=SelectorSidecar(x.shape[1]).to(device);x=x.to(device);y=y.to(device)
    opt=torch.optim.AdamW(scorer.parameters(),lr=3e-3,weight_decay=0);pos=((y==0).sum()/y.sum()).clamp(1,20)
    events=[]
    for step in range(1,MAX_STEPS+1):
        loss=F.binary_cross_entropy_with_logits(scorer(x),y,pos_weight=pos);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(scorer.parameters(),1.0);opt.step()
        if step in (1,MAX_STEPS):events.append({"step":step,"loss":float(loss.detach())})
    scorer.eval();grid=[]
    for t in THRESHOLDS:
        m=evaluate(split_items["train"],scorer,mean.to(device),std.to(device),t);m["threshold"]=t;grid.append(m)
    eligible=[m for m in grid if m["conservative_precision"]>=TRAIN_CALIBRATION_FLOOR and m["positive"]>0]
    chosen=max(eligible,key=lambda m:(m["bond_reachability_gain"],m["positive"],-m["threshold"])) if eligible else None
    threshold=None if chosen is None else chosen["threshold"]
    reports={split:(None if threshold is None else evaluate(items,scorer,mean.to(device),std.to(device),threshold)) for split,items in split_items.items()}
    holdout_grid={str(t): evaluate(split_items["fresh_holdout"], scorer, mean.to(device), std.to(device), t) for t in THRESHOLDS}
    result={"schema_version":"chemvector-r17-selector-retention-sidecar-0.1","complete":True,"training_performed":True,"training_scope":"isolated_selector_retention_sidecar","gpu_used":device.type=="cuda","device":str(device),"sealed_test_read":False,"production_checkpoint_written":False,"initial_checkpoint_sha256":SHA,"initial_checkpoint_sha256_after":digest(INITIAL),"steps":MAX_STEPS,"parameters":sum(p.numel() for p in scorer.parameters()),"candidate_feature_dim":int(x.shape[1]),"events":events,"cohort":cache.get('manifest',str(CACHE.relative_to(ROOT))),"molecule_limits":cache.get('molecule_limits'),"calibration":{"train_floor":TRAIN_CALIBRATION_FLOOR,"selected":chosen,"grid":grid},"reports":reports,"fresh_holdout_grid":holdout_grid,"gate":{"checkpoint_unchanged":digest(INITIAL)==SHA,"selected_prefix_retention":True,"fresh_precision_ge_0.85":bool(reports['fresh_holdout'] and reports['fresh_holdout']['conservative_precision']>=PRECISION_FLOOR),"fresh_bond_reachability_gain_positive":bool(reports['fresh_holdout'] and reports['fresh_holdout']['bond_reachability_gain']>0),"production_go":False}}
    OUT.mkdir(parents=True,exist_ok=False);torch.save({"state_dict":scorer.state_dict(),"mean":mean,"std":std,"threshold":threshold,"initial_checkpoint_sha256":SHA},OUT/"sidecar-state.pt");(OUT/"report.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps({"output":str(OUT/"report.json"),"calibration":chosen,"reports":reports,"gate":result["gate"]},indent=2))

if __name__=="__main__":run()
