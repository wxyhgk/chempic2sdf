#!/usr/bin/env python3
"""Same-decoder frozen-initial baseline for the independent vertex shadow."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.data import load_sample
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from tools.identity_vertex_decoder import decode_dense, raw_to_dense

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs/datasets/chemvector-identity-aligned-visible-text-20261017-r5-render"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-vertex-baseline-20261017-r5-v3/report.json"
SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
TOL = 6.0


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def collate(samples):
    max_atoms = max(len(s["graph"]["atoms"]) for s in samples)
    pixels = torch.stack([]) if False else torch.stack([
        (torch.from_numpy(__import__('numpy').asarray(s["image"], dtype='float32').copy()).permute(2, 0, 1) / 255 - .5) / .5
        for s in samples])
    anchors = torch.zeros(len(samples), max_atoms, 2); mask = torch.zeros(len(samples), max_atoms, dtype=torch.bool)
    elements=[]
    for bi,s in enumerate(samples):
        layers={a["atom_id"]:a for a in s["layers"]["atoms"]}; row=[]
        for a in sorted(s["graph"]["atoms"], key=lambda x:x["atom_id"]):
            aid=a["atom_id"]; anchors[bi,aid]=torch.tensor(layers[aid]["center"])/768.; mask[bi,aid]=True
            row.append(a["element"] if layers[aid].get("visible_text") else None)
        elements.append(row)
    return pixels, anchors, mask, elements


def nms(raw, threshold=.05):
    return decode_dense(raw_to_dense(raw), threshold=threshold, maximum_vertices=128, nms_kernel_size=3)[0]


def run():
    if OUT.exists(): raise FileExistsError(OUT)
    all_rows=[json.loads(x) for x in (DATA/'samples.jsonl').read_text().splitlines() if x]
    model,cfg,payload=load_bundle_for_probe(INITIAL,SHA); model.cuda().eval().requires_grad_(False)
    before=digest(INITIAL); split_reports={}
    for split in ('train','holdout'):
        rows=[r for r in all_rows if r['split']==split]; totals=Counter(); strata=defaultdict(Counter)
        for start in range(0,len(rows),8):
            batch_samples=[load_sample(DATA,r['sample_key'],purpose='diagnostic') for r in rows[start:start+8]]
            pixels,anchors,mask,elements=collate(batch_samples)
            with torch.inference_mode():
                raw=structure_forward(model,pixels.cuda(),cfg,payload['provenance']['atom_threshold'])[3]
            for bi,r in enumerate(batch_samples):
                pred=nms(raw, .05); target=anchors[bi][mask[bi]].tolist(); usedp=set(); usedt=set(); pairs=[]
                for pi,(x,y) in enumerate(pred):
                    for ti,(tx,ty) in enumerate(target):
                        d=((x-tx)**2+(y-ty)**2)**.5*768
                        if d<=TOL:pairs.append((d,pi,ti))
                for _,pi,ti in sorted(pairs):
                    if pi not in usedp and ti not in usedt: usedp.add(pi);usedt.add(ti)
                totals['predictions']+=len(pred); totals['targets']+=len(target); totals['matches']+=len(usedt)
                profile_key = r["meta"]["profile_id"]
                strata[profile_key]['targets'] += len(target); strata[profile_key]['matches'] += len(usedt)
                for e in ('O','N'):
                    idx={i for i,v in enumerate(elements[bi]) if v==e}; strata[e]['targets']+=len(idx);strata[e]['matches']+=len(idx&usedt)
        split_reports[split]={'images':len(rows),'totals':dict(totals),'precision':totals['matches']/max(totals['predictions'],1),'recall':totals['matches']/max(totals['targets'],1),'by_stratum':{k:{**dict(v),'recall':v['matches']/max(v['targets'],1)} for k,v in sorted(strata.items())}}
    report={'schema_version':'chemvector-identity-aligned-vertex-baseline-0.1','complete':True,'read_only':True,'training_performed':False,'gpu_used':True,'sealed_test_read':False,'initial_sha256':SHA,'initial_checkpoint_sha256_after':digest(INITIAL),'decoder':{'threshold':.05,'maximum_vertices':128,'grid_nms_kernel':3,'tolerance_px':TOL},'by_split':split_reports,'gate':{'checkpoint_unchanged':before==SHA and digest(INITIAL)==SHA,'training_admissible':False}}
    OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':run()
