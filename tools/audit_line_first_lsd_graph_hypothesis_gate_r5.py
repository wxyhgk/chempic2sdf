#!/usr/bin/env python3
"""CPU graph-hypothesis ceiling gate for the new-LSD proposal.

The proposal, candidate artifact, Hungarian ownership (r24/lambda8), and
audit-local edge/type sidecar are frozen.  Only graph decoding knobs are
swept: duplicate merge radius, topology confidence tau, aromatic cycle
demotion, and a small train-derived edge threshold perturbation.  Train-only
selection is followed by full holdout and six-profile reporting.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib, json, sys
from pathlib import Path
import numpy as np, torch
from rdkit import rdBase

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from chemvector.training.curriculum.checkpoints import file_digest
from tools import audit_line_first_graph_chemistry_roundtrip_r5 as chemistry
from tools import audit_line_first_hybrid_merge_roundtrip_r5 as hybrid
from tools import audit_line_first_lsd_edge_sidecar_cpu_r5 as sidecar
from tools import audit_line_first_lsd_all_edge_ownership_r5 as all_edge
from tools import audit_line_first_valence_topology_guard_r5 as guard
from tools import shadow_line_first_edge_bond_type_r5 as shadow

OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-lsd-graph-hypothesis-gate-r5/report.json"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
MERGES, TAUS, CYCLES, THRESHOLDS = (12.,16.,20.,24.), (.2,.4,.6), (False,True), (.004, .0062193297781050205, .010)


def finish(c):
    return {**dict(c), "edge_precision": c["true_edges"] / max(c["mapped_edges"], 1),
            "edge_recall": c["true_edges"] / max(c["gt_bonds"], 1),
            "bond_type_accuracy": c["type_correct"] / max(c["type_total"], 1),
            "dangling_rate": c["dangling_edges"] / max(c["selected_edges"], 1),
            "oracle_sanitize_rate": c["sanitize_ok"] / max(c["images"], 1)}


def install():
    all_edge._install()
    hybrid.RADIUS, hybrid.LAMBDA = 24.0, 8.0


def prepare(rows, candidates, model, mean, std):
    values=[]
    with torch.inference_mode():
        for row in rows:
            key=row["sample_key"]; features, centers, _=chemistry._proposals(key)
            points=np.asarray([[v["x"]*768.,v["y"]*768.,v.get("score",0.)] for v in candidates[key]],np.float64)
            edge_logits,type_logits=model(torch.from_numpy((features-mean)/std))
            scores=torch.sigmoid(edge_logits).numpy(); kinds=(torch.argmax(type_logits,dim=1)+1).numpy()
            values.append({"row":row,"centers":centers.astype(np.float64),"points":points,"keys":chemistry._candidate_keys(key),"scores":scores,"kinds":kinds})
    return values


def decode(v, merge, tau, cycle_on, threshold):
    owner,_=hybrid.assign(v["centers"],v["points"],merge)
    selected={}; dangling=0
    for i,(a,b) in enumerate(v["keys"]):
        if v["scores"][i] < threshold: continue
        if a not in owner or b not in owner: dangling+=1; continue
        ca,cb=owner[a],owner[b]
        if ca==cb: continue
        pair=tuple(sorted((ca,cb))); old=selected.get(pair)
        if old is None or float(v["scores"][i])>old[0]: selected[pair]=(float(v["scores"][i]),int(v["kinds"][i]))
    if cycle_on:
        ring=guard._cycles(selected)
        selected={p:(s,1 if t==guard.TYPE_AROMATIC and p not in ring else t) for p,(s,t) in selected.items()}
    selected,_=guard.guard(selected,low_confidence=tau)
    return selected,dangling


def evaluate(values, config):
    total=Counter(); profiles=defaultdict(Counter)
    for v in values:
        selected,dangling=decode(v,**config)
        m=guard._metrics(v["row"],selected,v["points"].astype(np.float32),dangling)
        total.update(m); profiles[v["row"]["profile_id"]].update(m)
    return finish(total),{k:finish(x) for k,x in sorted(profiles.items())}


def main():
    rdBase.DisableLog("rdApp.error")
    if file_digest(chemistry.INITIAL)!=INITIAL_SHA: raise RuntimeError("production checkpoint changed")
    install(); candidates=hybrid.candidate_rows(); state=torch.load(sidecar.STATE,map_location="cpu",weights_only=False)
    model=shadow.EdgeBondMLP(len(state["feature_names"])); model.load_state_dict(state["model"]); model.eval()
    mean,std=np.asarray(state["mean"],np.float32),np.asarray(state["std"],np.float32)
    train_rows,hold_rows=all_edge._rows("train"),all_edge._rows("holdout")
    train,hold=prepare(train_rows,candidates,model,mean,std),prepare(hold_rows,candidates,model,mean,std)
    configs=[]
    for merge in MERGES:
        for tau in TAUS:
            for cycle in CYCLES:
                for threshold in THRESHOLDS: configs.append({"merge":merge,"tau":tau,"cycle_on":cycle,"threshold":threshold})
    sweeps=[]
    for i,cfg in enumerate(configs,1):
        tr,tp=evaluate(train,cfg); sweeps.append({**cfg,"train":tr,"train_profiles":tp})
        print(f"train {i}/{len(configs)} {cfg} R={tr['edge_recall']:.4f} P={tr['edge_precision']:.4f}")
    eligible=[s for s in sweeps if s["train"]["edge_precision"]>=.90]
    selected=max(eligible or sweeps,key=lambda s:(s["train"]["edge_recall"],s["train"]["edge_precision"],-s["merge"],-s["tau"]))
    hold_sweeps=[]
    for i,s in enumerate(sweeps,1):
        cfg={k:s[k] for k in ("merge","tau","cycle_on","threshold")}; ho,hp=evaluate(hold,cfg); hold_sweeps.append({**s,"holdout":ho,"holdout_profiles":hp})
        print(f"holdout {i}/{len(configs)} {cfg} R={ho['edge_recall']:.4f} P={ho['edge_precision']:.4f}")
    chosen=next(x for x in hold_sweeps if all(x[k]==selected[k] for k in ("merge","tau","cycle_on","threshold")))
    sh,sp=chosen["holdout"],chosen["holdout_profiles"]
    report={"schema_version":"chemvector-identity-aligned-line-first-lsd-graph-hypothesis-gate-r5-0.1","complete":True,"read_only":True,"training_performed":False,"gpu_used":False,"sealed_test_read":False,"production_checkpoint_written":False,"proposal_contract":"new LSD min2 STD scale.75 -> fixed Hungarian r24/lambda8 -> graph hypotheses (merge/tau/cycle/threshold)","config":{"merges_px":MERGES,"taus":TAUS,"cycle_demotion":CYCLES,"thresholds":THRESHOLDS,"selection":"train-only precision >= .90 then recall"},"selected_train_only":selected,"selected_holdout":chosen,"sweeps":hold_sweeps,"initial_sha256":INITIAL_SHA,"initial_checkpoint_sha256_after":file_digest(chemistry.INITIAL),"sidecar_state_sha256":hashlib.sha256(sidecar.STATE.read_bytes()).hexdigest(),"gate":{"checkpoint_unchanged":file_digest(chemistry.INITIAL)==INITIAL_SHA,"sealed_test_read":False,"production_go":False,"selected_holdout_precision_ge_0.95852":sh["edge_precision"]>=.95852,"selected_holdout_recall_ge_0.85034":sh["edge_recall"]>=.85034,"selected_holdout_sanitize_ge_0.945":sh["oracle_sanitize_rate"]>=.945,"selected_all_profile_recall_ge_0.75":all(v["edge_recall"]>=.75 for v in sp.values()),"selected_all_profile_sanitize_ge_0.90":all(v["oracle_sanitize_rate"]>=.90 for v in sp.values()),"any_sweep_meets_aggregate_and_profile":any(x["holdout"]["edge_precision"]>=.95852 and x["holdout"]["edge_recall"]>=.85034 and x["holdout"]["oracle_sanitize_rate"]>=.945 and all(v["edge_recall"]>=.75 and v["oracle_sanitize_rate"]>=.90 for v in x["holdout_profiles"].values()) for x in hold_sweeps)}}
    OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"output":str(OUT),"selected":selected,"holdout":sh,"profiles":sp,"gate":report["gate"]},indent=2,ensure_ascii=False));return report

if __name__=="__main__":main()

