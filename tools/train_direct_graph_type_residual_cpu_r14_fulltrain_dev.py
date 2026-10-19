#!/usr/bin/env python3
"""Full-train shared bond-type residual, evaluated once on independent dev."""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib,json,sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from chemvector.bond_object import BOND_TYPES
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from tools.train_direct_graph_pair_scorer_cpu_r9 import DATA,SHA,_extract,rows,DirectPairResidual
from tools.diagnose_shared_type_cycle_valence_r12 import transform
from tools.audit_direct_graph_pair_scorer_chemistry_r9 import one,finish
TRIAL=ROOT/"outputs/audits/chemvector-identity-aligned-direct-graph-pair-scorer-cpu-r10-coverage";OUT=ROOT/"outputs/audits/chemvector-identity-aligned-direct-graph-type-residual-cpu-r14-fulltrain-dev";STATE=TRIAL/"scorer-state.pt";REPORT=OUT/"report.json";TYPE_STATE=OUT/"type-state.pt"
STEPS=300;CAP=64;SCALE=.5;REG=.05;THRESHOLD=.72
def dig(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
class TypeResidual(nn.Module):
 def __init__(self,dim=262):
  super().__init__();self.norm=nn.LayerNorm(dim);self.out=nn.Linear(dim,len(BOND_TYPES));nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
 def forward(self,x):return self.out(self.norm(x))
def collect(items):
 buckets=defaultdict(list);counts=Counter()
 for item in items:
  p=item['row']['profile_id'];pairs=item['pairs'];slots=item['slots'];mapping=item['mapping'];types=item['target_type_by_edge']
  for i in torch.where(item['labels']>.5)[0].tolist():
   l,r=(int(v) for v in pairs[i].tolist());li=int((slots==l).nonzero(as_tuple=False)[0]);ri=int((slots==r).nonzero(as_tuple=False)[0])
   if li not in mapping or ri not in mapping:continue
   n=types.get(tuple(sorted((mapping[li],mapping[ri]))));
   if n not in BOND_TYPES:continue
   buckets[(p,n)].append((item['features'][i],item['chemical_logits'][i],BOND_TYPES.index(n)));counts[(p,n)]+=1
 selected=[]
 for k in sorted(buckets):selected.extend(buckets[k][:CAP])
 return torch.stack([z[0] for z in selected]),torch.stack([z[1] for z in selected]),torch.tensor([z[2] for z in selected],dtype=torch.long),counts
def run():
 if OUT.exists():raise FileExistsError(OUT)
 if dig(INITIAL)!=SHA:raise RuntimeError('production checkpoint changed')
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(20261013)
 raw=rows();trs=sorted((r for r in raw if r.get('split')=='train'),key=lambda x:x['sample_key']);ev=sorted((r for r in raw if r.get('split')=='dev'),key=lambda x:x['sample_key'])
 if {r['molecule_id'] for r in trs}&{r['molecule_id'] for r in ev}:raise RuntimeError('train/dev identity overlap')
 ps=torch.load(STATE,map_location='cpu',weights_only=False);presence=DirectPairResidual(262);presence.load_state_dict(ps['state_dict']);presence.eval();model,cfg,_=load_bundle_for_probe(INITIAL,SHA);model.eval().requires_grad_(False)
 train=[_extract(r,model,cfg,need_labels=True) for r in trs];dev=[_extract(r,model,cfg,need_labels=True) for r in ev];x,b,y,counts=collect(train);typer=TypeResidual(262);opt=torch.optim.AdamW(typer.parameters(),lr=2e-3,weight_decay=1e-4);cc=Counter(y.tolist());w=torch.tensor([(len(y)/max(cc[i],1))**.5 for i in range(len(BOND_TYPES))]);events=[]
 for step in range(1,STEPS+1):
  typer.train();opt.zero_grad(set_to_none=True);delta=SCALE*typer(x);loss=F.cross_entropy(b+delta,y,weight=w)+REG*delta.square().mean();loss.backward();torch.nn.utils.clip_grad_norm_(typer.parameters(),1.0);opt.step()
  if step in (1,100,200,300):events.append({'step':step,'loss':float(loss.detach()),'delta_abs_mean':float(delta.detach().abs().mean())})
 typer.eval();baseline=Counter();scored=Counter();profiles={}
 for item in dev:baseline.update(one(item,presence,THRESHOLD));scored.update(one(transform(item,presence,typer),presence,THRESHOLD))
 for p in sorted({i['row']['profile_id'] for i in dev}):
  c=Counter()
  for item in dev:
   if item['row']['profile_id']==p:c.update(one(transform(item,presence,typer),presence,THRESHOLD))
  profiles[p]=finish(c)
 after=dig(INITIAL);result={'schema_version':'chemvector-identity-aligned-direct-graph-type-residual-cpu-r14-fulltrain-dev-0.1','complete':True,'read_only':False,'training_performed':True,'gpu_used':False,'sealed_test_read':False,'production_checkpoint_written':False,'train_images':len(trs),'dev_images':len(ev),'type_rows':len(y),'bucket_cap':CAP,'residual_scale':SCALE,'trust_region_lambda':REG,'type_counts':dict((BOND_TYPES[i],cc[i]) for i in range(len(BOND_TYPES))), 'train_bucket_counts':{str(k):v for k,v in counts.items()},'events':events,'baseline_dev':finish(baseline),'scored_dev':finish(scored),'dev_profiles':profiles,'initial_checkpoint_sha256':SHA,'initial_checkpoint_sha256_after':after,'gate':{'checkpoint_unchanged':after==SHA,'sealed_test_read':False,'strict_dev_sanitize_improvement':finish(scored)['oracle_sanitize_rate']>finish(baseline)['oracle_sanitize_rate'],'dev_oracle_sanitize_ge_0.90':finish(scored)['oracle_sanitize_rate']>=.90,'all_profile_dev_sanitize_ge_0.85':all(v['oracle_sanitize_rate']>=.85 for v in profiles.values()),'edge_set_frozen':True,'production_go':False,'next':'independent dev diagnostic; no GPU or production merge'}}
 OUT.mkdir(parents=True);torch.save({'schema_version':result['schema_version'],'state_dict':typer.state_dict(),'initial_checkpoint_sha256':SHA},TYPE_STATE);REPORT.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n');print(json.dumps({'output':str(REPORT),'baseline':result['baseline_dev'],'scored':result['scored_dev'],'gate':result['gate']},indent=2,ensure_ascii=False))
if __name__=='__main__':run()
