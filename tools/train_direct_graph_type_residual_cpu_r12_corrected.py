#!/usr/bin/env python3
"""Corrected profile×type residual: CE(base chemical logits + delta, target)."""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib, json, sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from chemvector.bond_object import BOND_TYPES
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from tools.train_direct_graph_pair_scorer_cpu_r9 import DATA,SHA,_extract,rows,DirectPairResidual
from tools.audit_direct_graph_pair_scorer_chemistry_r9 import one,finish

TRIAL=ROOT/"outputs/audits/chemvector-identity-aligned-direct-graph-pair-scorer-cpu-r10-coverage"
OUT=ROOT/"outputs/audits/chemvector-identity-aligned-direct-graph-type-residual-cpu-r12-corrected"
REPORT=OUT/"report.json"; STATE=TRIAL/"scorer-state.pt"; TYPE_STATE=OUT/"type-state.pt"
STEPS=300; THRESHOLD=.72; CAP=64; RESIDUAL_SCALE=.5

def digest(path:Path)->str:
 h=hashlib.sha256();
 with path.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()

class TypeResidual(nn.Module):
 def __init__(self,dim=262):
  super().__init__(); self.norm=nn.LayerNorm(dim); self.out=nn.Linear(dim,len(BOND_TYPES)); nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
 def forward(self,x): return self.out(self.norm(x))

def train_rows(raw):
 by=defaultdict(list)
 for r in sorted((x for x in raw if x.get("split")=="train"),key=lambda x:x["sample_key"]): by[r["profile_id"]].append(r)
 out=[]
 for p in sorted(by): out.extend(by[p] if p.endswith("r30_w16_serif") else by[p][:96])
 if len(out)!=474: raise RuntimeError(len(out))
 return sorted(out,key=lambda x:x["sample_key"])

def collect(items):
 buckets=defaultdict(list); all_counts=Counter()
 for item in items:
  pairs=item["pairs"]; slots=item["slots"]; mapping=item["mapping"]; types=item["target_type_by_edge"]; profile=item["row"]["profile_id"]
  for i in torch.where(item["labels"]>.5)[0].tolist():
   l,r=(int(v) for v in pairs[i].tolist()); li=int((slots==l).nonzero(as_tuple=False)[0]); ri=int((slots==r).nonzero(as_tuple=False)[0])
   if li not in mapping or ri not in mapping: continue
   name=types.get(tuple(sorted((mapping[li],mapping[ri]))));
   if name not in BOND_TYPES: continue
   all_counts[(profile,name)]+=1; buckets[(profile,name)].append((item["features"][i],item["chemical_logits"][i],BOND_TYPES.index(name)))
 selected=[]
 for k in sorted(buckets): selected.extend(buckets[k][:CAP])
 return (torch.stack([z[0] for z in selected]),torch.stack([z[1] for z in selected]),torch.tensor([z[2] for z in selected],dtype=torch.long),all_counts,{str(k):len(v) for k,v in buckets.items()})

def run():
 if OUT.exists(): raise FileExistsError(OUT)
 if digest(INITIAL)!=SHA: raise RuntimeError("production checkpoint changed")
 torch.set_num_threads(1); torch.use_deterministic_algorithms(True); torch.manual_seed(20261011)
 raw=rows(); trs=train_rows(raw); hrs=sorted((r for r in raw if r.get("split")=="fresh_holdout"),key=lambda x:x["sample_key"])
 if {r["molecule_id"] for r in trs}&{r["molecule_id"] for r in hrs}: raise RuntimeError("identity overlap")
 ps=torch.load(STATE,map_location="cpu",weights_only=False); presence=DirectPairResidual(262); presence.load_state_dict(ps["state_dict"]); presence.eval()
 model,config,_=load_bundle_for_probe(INITIAL,SHA); model.eval().requires_grad_(False)
 train=[_extract(r,model,config,need_labels=True) for r in trs]; hold=[_extract(r,model,config,need_labels=True) for r in hrs]
 x,basechem,y,all_counts,bcounts=collect(train); typer=TypeResidual(x.size(1)); opt=torch.optim.AdamW(typer.parameters(),lr=2e-3,weight_decay=1e-4)
 counts=Counter(y.tolist()); weights=torch.tensor([(len(y)/max(counts[i],1))**.5 for i in range(len(BOND_TYPES))]); events=[]
 for step in range(1,STEPS+1):
  typer.train(); opt.zero_grad(set_to_none=True); delta=RESIDUAL_SCALE*typer(x); loss=F.cross_entropy(basechem+delta,y,weight=weights); loss.backward(); torch.nn.utils.clip_grad_norm_(typer.parameters(),1.0); opt.step()
  if step in (1,100,200,300): events.append({"step":step,"loss":float(loss.detach()),"delta_abs_mean":float(delta.detach().abs().mean())})
 typer.eval(); b=Counter(); t=Counter(); bp={}; tp={}
 def typed_item(item):
  q=dict(item); q["chemical_logits"]=item["chemical_logits"]+RESIDUAL_SCALE*typer(item["features"]); return q
 for item in hold: b.update(one(item,presence,THRESHOLD)); t.update(one(typed_item(item),presence,THRESHOLD))
 for p in sorted({i["row"]["profile_id"] for i in hold}):
  bc=Counter();tc=Counter()
  for item in hold:
   if item["row"]["profile_id"]==p: bc.update(one(item,presence,THRESHOLD));tc.update(one(typed_item(item),presence,THRESHOLD))
  bp[p]=finish(bc);tp[p]=finish(tc)
 after=digest(INITIAL); result={"schema_version":"chemvector-identity-aligned-direct-graph-type-residual-cpu-r12-corrected-0.1","complete":True,"read_only":False,"training_performed":True,"gpu_used":False,"sealed_test_read":False,"production_checkpoint_written":False,"train_images":len(trs),"holdout_images":len(hrs),"train_type_rows":len(y),"bucket_cap":CAP,"residual_scale":RESIDUAL_SCALE,"type_counts":dict((BOND_TYPES[i],counts[i]) for i in range(len(BOND_TYPES))),"all_type_counts":{str(k):v for k,v in all_counts.items()},"bucket_counts":bcounts,"type_sidecar":{"class":"TypeResidual","parameters":sum(p.numel() for p in typer.parameters()),"steps":STEPS,"loss_events":events,"class_weights":weights.tolist(),"objective":"cross_entropy(base_chemical_logits + residual_scale * delta, target_type)"},"baseline_holdout":finish(b),"typed_holdout":finish(t),"baseline_profiles":bp,"typed_profiles":tp,"initial_checkpoint_sha256":SHA,"initial_checkpoint_sha256_after":after,"gate":{"checkpoint_unchanged":after==SHA,"sealed_test_read":False,"strict_sanitize_improvement":finish(t)["oracle_sanitize_rate"]>finish(b)["oracle_sanitize_rate"],"oracle_sanitize_ge_0.90":finish(t)["oracle_sanitize_rate"]>=.90,"all_profile_sanitize_ge_0.90":all(v["oracle_sanitize_rate"]>=.90 for v in tp.values()),"valence_not_increased":finish(t)["oracle_valence_errors"]<=finish(b)["oracle_valence_errors"],"edge_set_frozen":True,"production_go":False,"next":"CPU-only corrected residual; no GPU unless all chemistry gates pass"}}
 OUT.mkdir(parents=True); torch.save({"schema_version":result["schema_version"],"state_dict":typer.state_dict(),"initial_checkpoint_sha256":SHA},TYPE_STATE); REPORT.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n"); print(json.dumps({"output":str(REPORT),"baseline":result["baseline_holdout"],"typed":result["typed_holdout"],"gate":result["gate"]},indent=2,ensure_ascii=False))

if __name__=="__main__": run()
