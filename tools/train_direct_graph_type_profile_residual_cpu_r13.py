#!/usr/bin/env python3
"""Profile-specific, trust-region bond-type residual CPU experiment."""
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
from tools.audit_direct_graph_pair_scorer_chemistry_r9 import one,finish
TRIAL=ROOT/"outputs/audits/chemvector-identity-aligned-direct-graph-pair-scorer-cpu-r10-coverage"; OUT=ROOT/"outputs/audits/chemvector-identity-aligned-direct-graph-type-profile-residual-cpu-r13"; REPORT=OUT/"report.json"; STATE=TRIAL/"scorer-state.pt"; TYPE_STATE=OUT/"type-state.pt"
STEPS=300; CAP=64; SCALE=.5; REG=.05; THRESHOLD=.72
def digest(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
class TypeResidual(nn.Module):
 def __init__(self,dim=262):
  super().__init__();self.norm=nn.LayerNorm(dim);self.out=nn.Linear(dim,len(BOND_TYPES));nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
 def forward(self,x):return self.out(self.norm(x))
def train_rows(raw):
 by=defaultdict(list)
 for r in sorted((x for x in raw if x.get('split')=='train'),key=lambda x:x['sample_key']):by[r['profile_id']].append(r)
 out=[]
 for p in sorted(by):out.extend(by[p] if p.endswith('r30_w16_serif') else by[p][:96])
 return sorted(out,key=lambda x:x['sample_key'])
def collect(items):
 buckets=defaultdict(list); counts=Counter()
 for item in items:
  p=item['row']['profile_id'];pairs=item['pairs'];slots=item['slots'];mapping=item['mapping'];types=item['target_type_by_edge']
  for i in torch.where(item['labels']>.5)[0].tolist():
   l,r=(int(v) for v in pairs[i].tolist());li=int((slots==l).nonzero(as_tuple=False)[0]);ri=int((slots==r).nonzero(as_tuple=False)[0])
   if li not in mapping or ri not in mapping:continue
   n=types.get(tuple(sorted((mapping[li],mapping[ri]))));
   if n not in BOND_TYPES:continue
   buckets[p].append((item['features'][i],item['chemical_logits'][i],BOND_TYPES.index(n),n));counts[(p,n)]+=1
 out={}
 for p in sorted({x[0] for x in counts}):
  b=[z for n in BOND_TYPES for z in [x for x in buckets[p] if x[3]==n][:CAP]]
  out[p]=(torch.stack([z[0] for z in b]),torch.stack([z[1] for z in b]),torch.tensor([z[2] for z in b],dtype=torch.long))
 return out,counts
def run():
 if OUT.exists():raise FileExistsError(OUT)
 if digest(INITIAL)!=SHA:raise RuntimeError('production checkpoint changed')
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(20261012)
 raw=rows();trs=train_rows(raw);hrs=sorted((r for r in raw if r.get('split')=='fresh_holdout'),key=lambda x:x['sample_key'])
 if {r['molecule_id'] for r in trs}&{r['molecule_id'] for r in hrs}:raise RuntimeError('identity overlap')
 ps=torch.load(STATE,map_location='cpu',weights_only=False);presence=DirectPairResidual(262);presence.load_state_dict(ps['state_dict']);presence.eval();model,cfg,_=load_bundle_for_probe(INITIAL,SHA);model.eval().requires_grad_(False)
 train=[_extract(r,model,cfg,need_labels=True) for r in trs];hold=[_extract(r,model,cfg,need_labels=True) for r in hrs];data,counts=collect(train);heads={p:TypeResidual(262) for p in data};opts={p:torch.optim.AdamW(h.parameters(),lr=2e-3,weight_decay=1e-4) for p,h in heads.items()};events=[]
 for step in range(1,STEPS+1):
  row_losses=[]
  for p,(x,b,y) in data.items():
   h=heads[p];h.train();opts[p].zero_grad(set_to_none=True);delta=SCALE*h(x); cc=Counter(y.tolist());w=torch.tensor([(len(y)/max(cc[i],1))**.5 for i in range(len(BOND_TYPES))]);loss=F.cross_entropy(b+delta,y,weight=w)+REG*delta.square().mean();loss.backward();torch.nn.utils.clip_grad_norm_(h.parameters(),1.0);opts[p].step();row_losses.append(float(loss.detach()))
  if step in (1,100,200,300):events.append({'step':step,'loss_mean':sum(row_losses)/len(row_losses)})
 for h in heads.values():h.eval()
 def typed(item):
  q=dict(item);q['chemical_logits']=item['chemical_logits']+SCALE*heads[item['row']['profile_id']](item['features']);return q
 b=Counter();t=Counter();bp={};tp={}
 for item in hold:b.update(one(item,presence,THRESHOLD));t.update(one(typed(item),presence,THRESHOLD))
 for p in sorted({i['row']['profile_id'] for i in hold}):
  bc=Counter();tc=Counter()
  for item in hold:
   if item['row']['profile_id']==p:bc.update(one(item,presence,THRESHOLD));tc.update(one(typed(item),presence,THRESHOLD))
  bp[p]=finish(bc);tp[p]=finish(tc)
 after=digest(INITIAL);result={'schema_version':'chemvector-identity-aligned-direct-graph-type-profile-residual-cpu-r13-0.1','complete':True,'read_only':False,'training_performed':True,'gpu_used':False,'sealed_test_read':False,'production_checkpoint_written':False,'train_images':len(trs),'holdout_images':len(hrs),'train_profile_counts':dict(Counter(r['profile_id'] for r in trs)),'bucket_cap':CAP,'residual_scale':SCALE,'trust_region_lambda':REG,'type_counts':{str(k):v for k,v in counts.items()},'heads':{p:sum(q.numel() for q in h.parameters()) for p,h in heads.items()},'events':events,'baseline_holdout':finish(b),'typed_holdout':finish(t),'baseline_profiles':bp,'typed_profiles':tp,'initial_checkpoint_sha256':SHA,'initial_checkpoint_sha256_after':after,'gate':{'checkpoint_unchanged':after==SHA,'sealed_test_read':False,'strict_sanitize_improvement':finish(t)['oracle_sanitize_rate']>finish(b)['oracle_sanitize_rate'],'oracle_sanitize_ge_0.90':finish(t)['oracle_sanitize_rate']>=.90,'all_profile_sanitize_ge_0.90':all(v['oracle_sanitize_rate']>=.90 for v in tp.values()),'valence_not_increased':finish(t)['oracle_valence_errors']<=finish(b)['oracle_valence_errors'],'edge_set_frozen':True,'production_go':False,'next':'CPU-only diagnostic; no GPU unless all chemistry gates pass'}}
 OUT.mkdir(parents=True);torch.save({'schema_version':result['schema_version'],'heads':{p:h.state_dict() for p,h in heads.items()},'initial_checkpoint_sha256':SHA},TYPE_STATE);REPORT.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n');print(json.dumps({'output':str(REPORT),'baseline':result['baseline_holdout'],'typed':result['typed_holdout'],'gate':result['gate']},indent=2,ensure_ascii=False))
if __name__=='__main__':run()
