#!/usr/bin/env python3
"""Profile×bond-type balanced CPU trial for the direct graph type sidecar."""
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
OUT=ROOT/"outputs/audits/chemvector-identity-aligned-direct-graph-type-sidecar-cpu-r11-profile"
REPORT=OUT/"report.json"; STATE=TRIAL/"scorer-state.pt"; TYPE_STATE=OUT/"type-state.pt"
STEPS=300; THRESHOLD=.72; CAP_PER_BUCKET=64

def digest(path:Path)->str:
 h=hashlib.sha256();
 with path.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()

class TypeResidual(nn.Module):
 def __init__(self,dim=262):
  super().__init__(); self.norm=nn.LayerNorm(dim); self.out=nn.Linear(dim,len(BOND_TYPES)); nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
 def forward(self,x): return self.out(self.norm(x))

def extended_rows(all_rows):
 # All 186 serif30 images; 96 images in each remaining profile.  This adds
 # the missing serif30 style coverage without changing the holdout identities.
 by=defaultdict(list)
 for r in sorted((x for x in all_rows if x.get("split")=="train"),key=lambda x:x["sample_key"]): by[r["profile_id"]].append(r)
 out=[]
 for p in sorted(by): out.extend(by[p] if p.endswith("r30_w16_serif") else by[p][:96])
 if len(out)!=474: raise RuntimeError(f"expected 474 rows, got {len(out)}")
 return sorted(out,key=lambda x:x["sample_key"])

def type_rows(items):
 buckets=defaultdict(list); all_counts=Counter()
 for item in items:
  pairs=item["pairs"]; slots=item["slots"]; mapping=item["mapping"]; types=item["target_type_by_edge"]; profile=item["row"]["profile_id"]
  for i in torch.where(item["labels"]>.5)[0].tolist():
   l,r=(int(v) for v in pairs[i].tolist()); li=int((slots==l).nonzero(as_tuple=False)[0]); ri=int((slots==r).nonzero(as_tuple=False)[0])
   if li not in mapping or ri not in mapping: continue
   name=types.get(tuple(sorted((mapping[li],mapping[ri]))));
   if name not in BOND_TYPES: continue
   all_counts[(profile,name)]+=1; buckets[(profile,name)].append((item["features"][i],BOND_TYPES.index(name)))
 selected=[]
 for key in sorted(buckets): selected.extend(buckets[key][:CAP_PER_BUCKET])
 return torch.stack([x[0] for x in selected]),torch.tensor([x[1] for x in selected],dtype=torch.long),all_counts,{str(k):len(v) for k,v in buckets.items()}

def run():
 if OUT.exists(): raise FileExistsError(OUT)
 if digest(INITIAL)!=SHA: raise RuntimeError("production checkpoint changed")
 torch.set_num_threads(1); torch.use_deterministic_algorithms(True); torch.manual_seed(20261010)
 raw=rows(); train_rows=extended_rows(raw); hold_rows=sorted((r for r in raw if r.get("split")=="fresh_holdout"),key=lambda x:x["sample_key"])
 if {r["molecule_id"] for r in train_rows}&{r["molecule_id"] for r in hold_rows}: raise RuntimeError("identity overlap")
 presence_state=torch.load(STATE,map_location="cpu",weights_only=False); presence=DirectPairResidual(262); presence.load_state_dict(presence_state["state_dict"]); presence.eval()
 model,config,_=load_bundle_for_probe(INITIAL,SHA); model.eval().requires_grad_(False)
 train=[_extract(r,model,config,need_labels=True) for r in train_rows]; hold=[_extract(r,model,config,need_labels=True) for r in hold_rows]
 x,y,all_counts,bucket_counts=type_rows(train); typer=TypeResidual(x.size(1)); opt=torch.optim.AdamW(typer.parameters(),lr=2e-3,weight_decay=1e-4)
 class_counts=Counter(int(v) for v in y.tolist()); weights=torch.tensor([(len(y)/max(class_counts[i],1))**.5 for i in range(len(BOND_TYPES))])
 events=[]
 for step in range(1,STEPS+1):
  typer.train(); opt.zero_grad(set_to_none=True); loss=F.cross_entropy(typer(x),y,weight=weights); loss.backward(); torch.nn.utils.clip_grad_norm_(typer.parameters(),1.0); opt.step()
  if step in (1,100,200,300): events.append({"step":step,"loss":float(loss.detach())})
 typer.eval(); base=Counter(); typed=Counter(); bp={}; tp={}
 def typed_item(item):
  q=dict(item); q["chemical_logits"]=item["chemical_logits"]+typer(item["features"]); return q
 for item in hold: base.update(one(item,presence,THRESHOLD)); typed.update(one(typed_item(item),presence,THRESHOLD))
 for p in sorted({i["row"]["profile_id"] for i in hold}):
  b=Counter(); t=Counter()
  for item in hold:
   if item["row"]["profile_id"]==p: b.update(one(item,presence,THRESHOLD)); t.update(one(typed_item(item),presence,THRESHOLD))
  bp[p]=finish(b); tp[p]=finish(t)
 after=digest(INITIAL); result={"schema_version":"chemvector-identity-aligned-direct-graph-type-sidecar-cpu-r11-profile-0.1","complete":True,"read_only":False,"training_performed":True,"gpu_used":False,"sealed_test_read":False,"production_checkpoint_written":False,"train_images":len(train_rows),"holdout_images":len(hold_rows),"train_profile_counts":dict(Counter(r["profile_id"] for r in train_rows)),"bucket_cap":CAP_PER_BUCKET,"type_rows":len(y),"type_counts":dict((BOND_TYPES[i],class_counts[i]) for i in range(len(BOND_TYPES))),"all_type_counts":{str(k):v for k,v in all_counts.items()},"bucket_counts":bucket_counts,"type_sidecar":{"class":"TypeResidual","parameters":sum(p.numel() for p in typer.parameters()),"steps":STEPS,"loss_events":events,"class_weights":weights.tolist()},"baseline_holdout":finish(base),"typed_holdout":finish(typed),"baseline_profiles":bp,"typed_profiles":tp,"initial_checkpoint_sha256":SHA,"initial_checkpoint_sha256_after":after,"gate":{"checkpoint_unchanged":after==SHA,"sealed_test_read":False,"strict_sanitize_improvement":finish(typed)["oracle_sanitize_rate"]>finish(base)["oracle_sanitize_rate"],"oracle_sanitize_ge_0.90":finish(typed)["oracle_sanitize_rate"]>=.90,"all_profile_sanitize_ge_0.90":all(v["oracle_sanitize_rate"]>=.90 for v in tp.values()),"edge_set_frozen":True,"production_go":False,"next":"CPU-only diagnostic; no GPU unless all chemistry gates pass"}}
 OUT.mkdir(parents=True); torch.save({"schema_version":result["schema_version"],"state_dict":typer.state_dict(),"initial_checkpoint_sha256":SHA},TYPE_STATE); REPORT.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n"); print(json.dumps({"output":str(REPORT),"baseline":result["baseline_holdout"],"typed":result["typed_holdout"],"gate":result["gate"]},indent=2,ensure_ascii=False))

if __name__=="__main__": run()
