#!/usr/bin/env python3
"""CPU-only min-length/refine sweep for the legacy (scale=1) detector."""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib,json,sys
from pathlib import Path
import numpy as np,torch
from rdkit import rdBase
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from chemvector.training.curriculum.checkpoints import file_digest
from tools import audit_line_first_graph_chemistry_roundtrip_r5 as chemistry
from tools import audit_line_first_many_to_one_roundtrip_r5 as route
from tools import audit_line_first_lsd_config_sweep_r5 as detector
from tools import audit_line_first_valence_topology_guard_r5 as guard
DATA=route.DATA;OUT=ROOT/"outputs/audits/chemvector-identity-aligned-line-first-legacy-detector-sweep-r5/report.json";SHA="09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
LENS=(3.,5.,8.,12.);REFS=("STD","ADV");RADIUS,LAM,TAU=24.,8.,.4
def rows(s):return sorted([json.loads(x) for x in (DATA/"samples.jsonl").read_text().splitlines() if x.strip() and json.loads(x).get("split")==s and json.loads(x).get("render_status","accepted")=="accepted"],key=lambda x:x["sample_key"])
def install(length,refine):
 def d(image,*,config=None):return detector.detect_segments(image,minimum_length=length,refine=refine,scale=1.,border_margin=2.,maximum_candidates=int(getattr(config,"maximum_candidates",512)))
 import tools.preflight_line_first_edge_features_r5 as pre;import chemvector.components.vision.line_segment_rescue as res
 chemistry._proposals.__globals__["detect_raster_line_segments"]=d;chemistry._proposals.__globals__["CLUSTER_RADIUS"]=16.
 chemistry._candidate_keys.__globals__["CLUSTER_RADIUS"]=16.;pre.detect_raster_line_segments=d;pre.CLUSTER_RADIUS=16.;route.shadow.detect_raster_line_segments=d;route.shadow.CLUSTER_RADIUS=16.;res.detect_raster_line_segments=d
def prep(chosen,cands,model,mean,std):
 out=[]
 with torch.inference_mode():
  for row in chosen:
   k=row["sample_key"];f,c,_=chemistry._proposals(k);p=np.asarray([[v["x"]*768.,v["y"]*768.,v.get("score",0.)] for v in cands[k]],np.float64);e,t=model(torch.from_numpy((f-mean)/std));out.append((row,c.astype(np.float64),p,chemistry._candidate_keys(k),torch.sigmoid(e).numpy(),(torch.argmax(t,dim=1)+1).numpy()))
 return out
def fin(c):return {**dict(c),"edge_precision":c["true_edges"]/max(c["mapped_edges"],1),"edge_recall":c["true_edges"]/max(c["gt_bonds"],1),"bond_type_accuracy":c["type_correct"]/max(c["type_total"],1),"dangling_rate":c["dangling_edges"]/max(c["selected_edges"],1),"oracle_sanitize_rate":c["sanitize_ok"]/max(c["images"],1)}
def run(vals):
 tot=Counter();prof=defaultdict(Counter)
 for row,c,p,keys,scores,kinds in vals:
  own=route.assign_many(c,p);sel={};dang=0
  for i,(a,b) in enumerate(keys):
   if scores[i]<1.4e-8:continue
   if a not in own or b not in own:dang+=1;continue
   ca,cb=own[a],own[b]
   if ca==cb:continue
   q=tuple(sorted((ca,cb)));old=sel.get(q)
   if old is None or scores[i]>old[0]:sel[q]=(float(scores[i]),int(kinds[i]))
  ring=guard._cycles(sel);sel={q:(s,1 if t==guard.TYPE_AROMATIC and q not in ring else t) for q,(s,t) in sel.items()};sel,_=guard.guard(sel,low_confidence=TAU);m=guard._metrics(row,sel,p.astype(np.float32),dang);tot.update(m);prof[row["profile_id"]].update(m)
 return fin(tot),{k:fin(v) for k,v in sorted(prof.items())}
def main():
 rdBase.DisableLog("rdApp.error");assert file_digest(chemistry.INITIAL)==SHA;cands=chemistry._candidate_rows();st=torch.load(chemistry.STATE,map_location="cpu",weights_only=False);model=chemistry.EdgeBondMLP(len(st["feature_names"]));model.load_state_dict(st["model"]);model.eval();mean,std=np.asarray(st["mean"],np.float32),np.asarray(st["std"],np.float32);tr,ho=rows("train"),rows("holdout");sweeps=[]
 for le in LENS:
  for re in REFS:
   install(le,re);tv=prep(tr,cands,model,mean,std);hv=prep(ho,cands,model,mean,std);tm,tp=run(tv);sweeps.append({"minimum_length_px":le,"refine":re,"train":tm,"train_profiles":tp,"_hold":hv})
   print(f"train len{le} {re} R={tm['edge_recall']:.4f} P={tm['edge_precision']:.4f}",flush=True)
 eligible=[s for s in sweeps if s["train"]["edge_precision"]>=.96592];chosen=max(eligible or sweeps,key=lambda s:(s["train"]["edge_recall"],s["train"]["edge_precision"],-s["minimum_length_px"]))
 out=[]
 for s in sweeps:
  hm,hp=run(s.pop("_hold"));out.append({**s,"holdout":hm,"holdout_profiles":hp})
 selected=next(s for s in out if s["minimum_length_px"]==chosen["minimum_length_px"] and s["refine"]==chosen["refine"]);h=selected["holdout"];hp=selected["holdout_profiles"]
 report={"schema_version":"chemvector-identity-aligned-line-first-legacy-detector-sweep-r5-0.1","complete":True,"read_only":True,"training_performed":False,"gpu_used":False,"sealed_test_read":False,"production_checkpoint_written":False,"proposal_contract":"legacy scale=1 LSD -> cluster16 -> many-to-one r24/l8 -> cycle/guard tau.4","config":{"minimum_lengths_px":LENS,"refines":REFS,"radius_px":RADIUS,"lambda":LAM,"tau":TAU,"selection":"train-only precision >=.96592 then recall"},"selected_train_only":chosen,"selected_holdout":selected,"sweeps":out,"initial_sha256":SHA,"initial_checkpoint_sha256_after":file_digest(chemistry.INITIAL),"gate":{"checkpoint_unchanged":file_digest(chemistry.INITIAL)==SHA,"sealed_test_read":False,"production_go":False,"selected_holdout_precision_ge_0.96592":h["edge_precision"]>=.96592,"selected_holdout_recall_gt_0.85034":h["edge_recall"]>.85034,"selected_all_profile_recall_ge_0.75":all(v["edge_recall"]>=.75 for v in hp.values()),"selected_all_profile_sanitize_ge_0.90":all(v["oracle_sanitize_rate"]>=.90 for v in hp.values()),"any_sweep_meets_all":any(x["holdout"]["edge_precision"]>=.96592 and x["holdout"]["edge_recall"]>.85034 and all(v["edge_recall"]>=.75 and v["oracle_sanitize_rate"]>=.90 for v in x["holdout_profiles"].values()) for x in out)}}
 OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n");print(json.dumps({"output":str(OUT),"selected":selected,"holdout":h,"gate":report["gate"]},indent=2,ensure_ascii=False))
if __name__=="__main__":main()
