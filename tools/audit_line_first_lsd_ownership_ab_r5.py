#!/usr/bin/env python3
"""Minimal target-free A/B for the new LSD -> candidate ownership seam.

A is the frozen ownership (radius 24, score penalty lambda 8); B removes the
score prior and widens the admissible radius to 40 px.  Both retain every
provisional edge, merge duplicate candidate pairs at 20 px, and apply the same
topology guard.  This isolates ownership from the edge scorer and reports the
fixed full holdout and six style profiles.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib, json, sys
from pathlib import Path
import numpy as np
from rdkit import rdBase

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from chemvector.training.curriculum.checkpoints import file_digest
from tools import audit_line_first_graph_chemistry_roundtrip_r5 as chemistry
from tools import audit_line_first_hybrid_merge_roundtrip_r5 as hybrid
from tools import audit_line_first_valence_topology_guard_r5 as guard
from tools import audit_line_first_lsd_all_edge_ownership_r5 as all_edge

DATA = ROOT / "outputs/datasets/chemvector-identity-aligned-visible-text-20261017-r5-render"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-lsd-ownership-ab-r5/report.json"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
MERGE, TAU = 20.0, 0.4


def finish(c):
    return {**dict(c), "edge_precision": c["true_edges"] / max(c["mapped_edges"], 1),
            "edge_recall": c["true_edges"] / max(c["gt_bonds"], 1),
            "dangling_rate": c["dangling_edges"] / max(c["selected_edges"], 1),
            "oracle_sanitize_rate": c["sanitize_ok"] / max(c["images"], 1)}


def prep(rows, candidates):
    out=[]
    for row in rows:
        key=row["sample_key"]; _, centers, _=chemistry._proposals(key)
        points=np.asarray([[v["x"]*768.0,v["y"]*768.0,v.get("score",0.)] for v in candidates[key]], np.float64)
        out.append((row, centers.astype(np.float64), points, chemistry._candidate_keys(key)))
    return out


def run(values, radius, lam):
    hybrid.RADIUS, hybrid.LAMBDA = float(radius), float(lam)
    total=Counter(); profiles=defaultdict(Counter)
    for row, centers, points, keys in values:
        owner,_=hybrid.assign(centers,points,MERGE)
        selected={}; dangling=0
        for a,b in keys:
            if a not in owner or b not in owner: dangling+=1; continue
            ca,cb=owner[a],owner[b]
            if ca==cb: continue
            selected[tuple(sorted((ca,cb)))]=(1.0,1)
        selected,_=guard.guard(selected, low_confidence=TAU)
        m=guard._metrics(row,selected,points.astype(np.float32),dangling)
        total.update(m); profiles[row["profile_id"]].update(m)
    return finish(total),{k:finish(v) for k,v in sorted(profiles.items())}


def main():
    rdBase.DisableLog("rdApp.error")
    if file_digest(chemistry.INITIAL)!=INITIAL_SHA: raise RuntimeError("production checkpoint changed")
    all_edge._install(); candidates=hybrid.candidate_rows(); train=prep(all_edge._rows("train"),candidates); hold=prep(all_edge._rows("holdout"),candidates)
    configs={"A_frozen_r24_l8":(24.,8.),"B_wide_r40_l0":(40.,0.)}
    results={}
    for name,(r,l) in configs.items():
        tr,tp=run(train,r,l); ho,hp=run(hold,r,l)
        results[name]={"radius_px":r,"lambda":l,"train":tr,"train_profiles":tp,"holdout":ho,"holdout_profiles":hp}
    # Selection is train-only, then report fixed holdout.  This A/B is not an
    # authorization gate; it only says whether ownership is worth deeper work.
    selected=max(results,key=lambda k:(results[k]["train"]["edge_recall"],results[k]["train"]["edge_precision"],-results[k]["train"]["dangling_rate"]))
    sh=results[selected]["holdout"]; shp=results[selected]["holdout_profiles"]
    report={"schema_version":"chemvector-identity-aligned-line-first-lsd-ownership-ab-r5-0.1","complete":True,"read_only":True,"training_performed":False,"gpu_used":False,"sealed_test_read":False,"production_checkpoint_written":False,"proposal_contract":"new LSD min2 STD scale.75 single-link r10 -> all provisional edges -> target-free ownership A/B + merge20 + topology guard","configs":results,"selected_train_only":selected,"initial_sha256":INITIAL_SHA,"initial_checkpoint_sha256_after":file_digest(chemistry.INITIAL),"candidate_artifact_sha256":hashlib.sha256(hybrid.CANDIDATES.read_bytes()).hexdigest(),"gate":{"checkpoint_unchanged":file_digest(chemistry.INITIAL)==INITIAL_SHA,"sealed_test_read":False,"production_go":False,"selected_holdout_recall_ge_0_75":sh["edge_recall"]>=.75,"selected_holdout_dangling_le_0_05":sh["dangling_rate"]<=.05,"selected_all_profile_recall_ge_0_75":all(v["edge_recall"]>=.75 for v in shp.values()),"selected_all_profile_sanitize_ge_0_90":all(v["oracle_sanitize_rate"]>=.90 for v in shp.values()),"ownership_ab_sufficient":False}}
    OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"output":str(OUT),"selected":selected,"holdout":sh,"profiles":shp,"gate":report["gate"]},indent=2,ensure_ascii=False));return report

if __name__=="__main__":main()

