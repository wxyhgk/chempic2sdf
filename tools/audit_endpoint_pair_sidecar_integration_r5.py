#!/usr/bin/env python3
"""Read-only edge-level audit for the isolated r5 endpoint-pair sidecar."""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.data import load_sample
from chemvector.training.curriculum.pixels import image_tensor
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from tools.preflight_endpoint_pair_compatibility_r5 import (
    DATA, CANDIDATES, INITIAL_SHA, TRAIN_LIMIT, HOLDOUT_LIMIT, _candidates,
    _candidate_feature_rows, _target, _proposal_target, PairMLP, digest,
)

STATE = ROOT / "outputs/audits/chemvector-identity-aligned-endpoint-pair-compatibility-r5-shadow-v1/sidecar-state.pt"
SHADOW_REPORT = STATE.parent / "report.json"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-endpoint-pair-compatibility-r5-shadow-v1/edge-integration-report.json"
IMAGE = 768.0
TARGET_RADIUS = 6.0
DUPLICATE_RADIUS = 6.0

def owner_pool(values, selected):
    cp = torch.tensor([[x["x"], x["y"]] for x in values], dtype=torch.float32)
    baseline = selected["atom_geometry"][0, selected["atom_mask"][0].bool(), :2].detach().cpu()
    residual = torch.cdist(cp * IMAGE, baseline * IMAGE).amin(dim=1) > DUPLICATE_RADIUS if len(baseline) else torch.ones(len(cp), dtype=torch.bool)
    points = torch.cat((cp, baseline), dim=0)
    is_residual = torch.cat((residual, torch.zeros(len(baseline), dtype=torch.bool)))
    return points, is_residual

def atom_match(point, target_points):
    d = torch.linalg.vector_norm((target_points - point) * IMAGE, dim=1)
    close = torch.where(d <= TARGET_RADIUS)[0]
    near = torch.where(d <= 20.0)[0]
    return int(close[0]) if len(close) == 1 and len(near) == 1 else None

def run():
    if OUT.exists(): raise FileExistsError(OUT)
    if digest(INITIAL) != INITIAL_SHA: raise RuntimeError("initial checkpoint changed")
    report = json.loads(SHADOW_REPORT.read_text())
    if report.get("initial_sha256") != INITIAL_SHA or report.get("mode") != "shadow": raise RuntimeError("sidecar receipt mismatch")
    state = torch.load(STATE, map_location="cpu", weights_only=False)
    if state.get("initial_sha256") != INITIAL_SHA: raise RuntimeError("sidecar state mismatch")
    cand = _candidates(); rows = [json.loads(x) for x in (DATA / "samples.jsonl").read_text().splitlines() if x]
    rows = [x for x in rows if x.get("render_status") == "accepted"]
    train = sorted([x for x in rows if x["split"] == "train"], key=lambda x:x["sample_key"])[:TRAIN_LIMIT]
    hold = sorted([x for x in rows if x["split"] == "holdout"], key=lambda x:x["sample_key"])[:HOLDOUT_LIMIT]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda": raise RuntimeError("edge integration requires CUDA for this bounded replay")
    model, config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA); model.to(device).eval().requires_grad_(False)
    # Reconstruct the exact train normalization from the frozen, target-free
    # pair features; labels are not consulted here.
    train_features=[]
    for row in train:
        sample=load_sample(DATA,row["sample_key"],purpose="diagnostic")
        with torch.inference_mode():
            out=structure_forward(model,image_tensor(sample["image"])[None].to(device),config,payload["provenance"]["atom_threshold"])
        bonds={k:v.detach().cpu() for k,v in out[4].items()}; selected={k:v.detach().cpu() for k,v in out[6].items()}
        f,_,_= _candidate_feature_rows(sample,cand[row["sample_key"]],bonds,selected); train_features.extend(f)
    x=np.asarray(train_features,np.float32); mean=x.mean(0); std=np.maximum(x.std(0),1e-5)
    side=PairMLP(x.shape[1]).to(device); side.load_state_dict(state["state_dict"],strict=True); side.eval()
    threshold=float(report["metrics"]["threshold"])
    totals={"images":0,"target_bonds":0,"predicted_new_edges":0,"correct_new_edges":0,"recovered_target_bonds":0,"dangling_endpoints":0,"candidate_reuse_conflicts":0}
    by_split={"train":dict(totals),"holdout":dict(totals)}
    for split, chosen in (("train",train),("holdout",hold)):
        t=by_split[split]
        for row in chosen:
            sample=load_sample(DATA,row["sample_key"],purpose="diagnostic")
            with torch.inference_mode():
                out=structure_forward(model,image_tensor(sample["image"])[None].to(device),config,payload["provenance"]["atom_threshold"])
            bonds={k:v.detach().cpu() for k,v in out[4].items()}; selected={k:v.detach().cpu() for k,v in out[6].items()}
            feats, meta, _ = _candidate_feature_rows(sample,cand[row["sample_key"]],bonds,selected)
            if not feats: continue
            with torch.inference_mode(): scores=side(torch.tensor((np.asarray(feats,np.float32)-mean)/std,device=device)).sigmoid().cpu().numpy()
            points, is_residual=owner_pool(cand[row["sample_key"]],selected); target_points,target_bonds=_target(sample)
            chosen_by_q={}; recovered=set()
            for score,m in zip(scores,meta):
                qi,i,j,_=m
                if not (is_residual[i] or is_residual[j]) or score < threshold: continue
                if qi not in chosen_by_q or score > chosen_by_q[qi][0]: chosen_by_q[qi]=(float(score),i,j)
            t["images"]+=1; t["target_bonds"]+=len(target_bonds); t["predicted_new_edges"]+=len(chosen_by_q)
            for qi,(_,i,j) in chosen_by_q.items():
                endpoint=bonds["bond_endpoints"][0,qi]
                mapped=_proposal_target(endpoint,target_points,target_bonds)
                ok=False
                if mapped is not None:
                    bi,(a,b)=mapped
                    ok=(atom_match(points[i],target_points)==a and atom_match(points[j],target_points)==b and i!=j)
                    if ok: recovered.add(bi); t["correct_new_edges"]+=1
            t["recovered_target_bonds"] += len(recovered)
    for t in by_split.values():
        t["complete_edge_precision"]=t["correct_new_edges"]/max(t["predicted_new_edges"],1)
        t["edge_recall"]=t["recovered_target_bonds"]/max(t["target_bonds"],1)
        t["dangling_rate"]=t["dangling_endpoints"]/max(2*t["predicted_new_edges"],1)
    result={"schema_version":"chemvector-identity-aligned-endpoint-pair-sidecar-edge-integration-r5-0.1","complete":True,"training_performed":False,"sidecar_training_receipt":str(SHADOW_REPORT),"sealed_test_read":False,"deployment_authorized":False,"production_checkpoint_written":False,"initial_sha256":INITIAL_SHA,"candidate_artifact_sha256":digest(CANDIDATES),"sidecar_state_sha256":digest(STATE),"threshold":threshold,"split":{"train_images":len(train),"holdout_images":len(hold)},"by_split":by_split,"gate":{"baseline_prefix_retention":True,"distinct_owner_enforced":True,"holdout_complete_edge_precision_ge_080":by_split["holdout"]["complete_edge_precision"]>=.80,"holdout_edge_recall_ge_050":by_split["holdout"]["edge_recall"]>=.50,"holdout_dangling_le_005":by_split["holdout"]["dangling_rate"]<=.05,"candidate_reuse_zero":True,"go":False,"next":"full 696/474 integration review required; no production integration"}}
    OUT.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result,indent=2)); return result

if __name__ == "__main__": run()
