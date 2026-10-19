#!/usr/bin/env python3
"""Cached CPU sweep for legacy many-to-one ownership/guard parameters."""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib,json,sys
from pathlib import Path
import numpy as np, torch
from rdkit import rdBase
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from chemvector.training.curriculum.checkpoints import file_digest
from tools import audit_line_first_graph_chemistry_roundtrip_r5 as chemistry
from tools import audit_line_first_many_to_one_roundtrip_r5 as route
from tools import audit_line_first_valence_topology_guard_r5 as guard
DATA=route.DATA; OUT=ROOT/"outputs/audits/chemvector-identity-aligned-line-first-many-to-one-sweep-v2-r5/report.json"
SHA="09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"; RADII=(16.,20.,24.,28.); LAMS=(0.,4.,8.); TAUS=(.2,.4,.6)
def rows(split):return sorted([json.loads(x) for x in (DATA/"samples.jsonl").read_text().splitlines() if x.strip() and json.loads(x).get("split")==split and json.loads(x).get("render_status","accepted")=="accepted"],key=lambda r:r["sample_key"])
def fin(c):return {**dict(c),"edge_precision":c["true_edges"]/max(c["mapped_edges"],1),"edge_recall":c["true_edges"]/max(c["gt_bonds"],1),"bond_type_accuracy":c["type_correct"]/max(c["type_total"],1),"dangling_rate":c["dangling_edges"]/max(c["selected_edges"],1),"oracle_sanitize_rate":c["sanitize_ok"]/max(c["images"],1)}
def prepare(chosen,cands,model,mean,std):
 out=[]
 with torch.inference_mode():
  for i,row in enumerate(chosen,1):
   key=row["sample_key"]; f,cent,_=chemistry._proposals(key); pts=np.asarray([[v["x"]*768.,v["y"]*768.,v.get("score",0.)] for v in cands[key]],np.float64); e,t=model(torch.from_numpy((f-mean)/std)); out.append((row,cent.astype(np.float64),pts,chemistry._candidate_keys(key),torch.sigmoid(e).numpy(),(torch.argmax(t,dim=1)+1).numpy()))
   if i%200==0:print(f"prepare {i}/{len(chosen)}",flush=True)
 return out
def owner(cent,pts,r,lam):
 d=np.linalg.norm(cent[:,None]-pts[None,:,:2],axis=-1); cost=np.where(d<=r,d-lam*pts[None,:,2],1e5+d); return {int(i):int(np.argmin(cost[i])) for i in range(len(cent)) if float(d[i,int(np.argmin(cost[i]))])<=r}
def evaluate(vals,r,lam,tau):
 total=Counter(); prof=defaultdict(Counter)
 for row,cent,pts,keys,scores,kinds in vals:
  own=owner(cent,pts,r,lam); sel={};dang=0
  for i,(a,b) in enumerate(keys):
   if scores[i]<1.4e-8:continue
   if a not in own or b not in own:dang+=1;continue
   ca,cb=own[a],own[b]
   if ca==cb:continue
   p=tuple(sorted((ca,cb)));old=sel.get(p)
   if old is None or scores[i]>old[0]:sel[p]=(float(scores[i]),int(kinds[i]))
  ring=guard._cycles(sel);sel={p:(s,1 if t==guard.TYPE_AROMATIC and p not in ring else t) for p,(s,t) in sel.items()};sel,_=guard.guard(sel,low_confidence=tau);m=guard._metrics(row,sel,pts.astype(np.float32),dang);total.update(m);prof[row["profile_id"]].update(m)
 return fin(total),{k:fin(v) for k,v in sorted(prof.items())}
def main():
 rdBase.DisableLog("rdApp.error"); assert file_digest(chemistry.INITIAL)==SHA
 cands=chemistry._candidate_rows();st=torch.load(chemistry.STATE,map_location="cpu",weights_only=False);model=chemistry.EdgeBondMLP(len(st["feature_names"]));model.load_state_dict(st["model"]);model.eval();mean,std=np.asarray(st["mean"],np.float32),np.asarray(st["std"],np.float32)
 tr,ho=rows("train"),rows("holdout"); tv,hv=prepare(tr,cands,model,mean,std),prepare(ho,cands,model,mean,std); sweeps=[]
 cfgs=[(r,l,t) for r in RADII for l in LAMS for t in TAUS]
 for i,(r,l,t) in enumerate(cfgs,1):
  a,ap=evaluate(tv,r,l,t);s={"radius_px":r,"lambda":l,"tau":t,"train":a,"train_profiles":ap};sweeps.append(s);print(f"train {i}/{len(cfgs)} r{r} l{l} t{t} R={a['edge_recall']:.4f} P={a['edge_precision']:.4f}",flush=True)
 eligible=[s for s in sweeps if s["train"]["edge_precision"]>=.96592 and s["train"]["edge_recall"]>.85034]; chosen=max(eligible or sweeps,key=lambda s:(s["train"]["edge_recall"],s["train"]["edge_precision"],-s["radius_px"],-s["lambda"],-s["tau"]))
 allout=[]
 for i,s in enumerate(sweeps,1):
  a,ap=evaluate(hv,s["radius_px"],s["lambda"],s["tau"]);allout.append({**s,"holdout":a,"holdout_profiles":ap});print(f"holdout {i}/{len(cfgs)} R={a['edge_recall']:.4f} P={a['edge_precision']:.4f}",flush=True)
 selected=next(s for s in allout if all(s[k]==chosen[k] for k in ("radius_px","lambda","tau")));h=selected["holdout"];hp=selected["holdout_profiles"]
 report={"schema_version":"chemvector-identity-aligned-line-first-many-to-one-sweep-v2-r5-0.1","complete":True,"read_only":True,"training_performed":False,"gpu_used":False,"sealed_test_read":False,"production_checkpoint_written":False,"proposal_contract":"legacy LSD -> many-to-one score-distance ownership -> edge dedup -> cycle/valence guard","config":{"radii_px":RADII,"lambdas":LAMS,"taus":TAUS,"threshold":"frozen 1.4e-8","selection":"train-only P>=.96592 and R>.85034 then max recall"},"selected_train_only":chosen,"selected_holdout":selected,"sweeps":allout,"initial_sha256":SHA,"initial_checkpoint_sha256_after":file_digest(chemistry.INITIAL),"gate":{"checkpoint_unchanged":file_digest(chemistry.INITIAL)==SHA,"sealed_test_read":False,"production_go":False,"selected_holdout_precision_ge_0.96592":h["edge_precision"]>=.96592,"selected_holdout_recall_gt_0.85034":h["edge_recall"]>.85034,"selected_all_profile_recall_ge_0.75":all(v["edge_recall"]>=.75 for v in hp.values()),"selected_all_profile_sanitize_ge_0.90":all(v["oracle_sanitize_rate"]>=.90 for v in hp.values()),"any_sweep_meets_all":any(x["holdout"]["edge_precision"]>=.96592 and x["holdout"]["edge_recall"]>.85034 and all(v["edge_recall"]>=.75 and v["oracle_sanitize_rate"]>=.90 for v in x["holdout_profiles"].values()) for x in allout)}}
 OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n");print(json.dumps({"output":str(OUT),"selected":chosen,"holdout":h,"profiles":hp,"gate":report["gate"]},indent=2,ensure_ascii=False))
if __name__=="__main__":main()
