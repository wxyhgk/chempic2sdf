#!/usr/bin/env python3
"""Shared type residual + target-free cycle/order repair diagnostic."""
from __future__ import annotations
from collections import Counter
import hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from tools.train_direct_graph_pair_scorer_cpu_r9 import DATA,SHA,_extract,rows,DirectPairResidual
from tools.train_direct_graph_type_residual_cpu_r12_corrected import TypeResidual
from tools.audit_direct_graph_pair_scorer_chemistry_r9 import one,finish
TRIAL=ROOT/"outputs/audits/chemvector-identity-aligned-direct-graph-pair-scorer-cpu-r10-coverage";TS=ROOT/"outputs/audits/chemvector-identity-aligned-direct-graph-type-residual-cpu-r12-corrected/type-state.pt";OUT=ROOT/"outputs/audits/chemvector-identity-aligned-direct-graph-shared-cycle-valence-r12"
SCALE=.5;THRESHOLD=.72
def dig(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def path(adj,a,b,skip):
 seen={a};st=[a]
 while st:
  u=st.pop()
  for v in adj.get(u,()):
   if tuple(sorted((u,v)))==skip:continue
   if v==b:return True
   if v not in seen:seen.add(v);st.append(v)
 return False
def transform(item,presence,typer):
 q=dict(item); pairs=item['pairs'];slots=item['slots'];st={int(x):i for i,x in enumerate(slots.tolist())}; logits=item['chemical_logits']+SCALE*typer(item['features']); scores=(item['base_logits']+presence(item['features'])).sigmoid();chosen=torch.where(scores>=THRESHOLD)[0].tolist();edges=[]
 for i in chosen:
  l,r=(int(v) for v in pairs[i].tolist());
  if l in st and r in st and st[l]!=st[r]:edges.append((st[l],st[r],i))
 adj={}
 for a,b,_ in edges:adj.setdefault(a,set()).add(b);adj.setdefault(b,set()).add(a)
 out=logits.clone()
 for a,b,i in edges:
  if int(out[i].argmax())==3 and not path(adj,a,b,tuple(sorted((a,b)))):out[i,3]=-1e4;out[i,0]=out[i].max()+1
 order={0:1.,1:2.,2:3.,3:1.5}
 for _ in range(32):
  kinds={i:int(out[i].argmax()) for _,_,i in edges};tot={}
  for a,b,i in edges:tot[a]=tot.get(a,0)+order[kinds[i]];tot[b]=tot.get(b,0)+order[kinds[i]]
  bad=[u for u,v in tot.items() if v>4+1e-6]
  if not bad:break
  cs=[(i,kinds[i],float(out[i].max()-out[i].topk(2).values[-1])) for a,b,i in edges if any(u in (a,b) for u in bad) and kinds[i]!=0]
  if not cs:break
  i,_,_=min(cs,key=lambda z:(-order[z[1]],z[2],z[0]));out[i,0]=out[i].max()+1;out[i,1:]-=1e4
 q['chemical_logits']=out;return q
def run():
 if dig(INITIAL)!=SHA:raise RuntimeError('production checkpoint changed')
 ps=torch.load(TRIAL/'scorer-state.pt',map_location='cpu',weights_only=False);presence=DirectPairResidual(262);presence.load_state_dict(ps['state_dict']);presence.eval();ts=torch.load(TS,map_location='cpu',weights_only=False);typer=TypeResidual(262);typer.load_state_dict(ts['state_dict']);typer.eval();model,cfg,_=load_bundle_for_probe(INITIAL,SHA);model.eval().requires_grad_(False);hrs=sorted((r for r in rows() if r.get('split')=='fresh_holdout'),key=lambda x:x['sample_key']);items=[_extract(r,model,cfg,need_labels=True) for r in hrs]
 total=Counter(); profiles={}
 for item in items:total.update(one(transform(item,presence,typer),presence,THRESHOLD))
 for p in sorted({i['row']['profile_id'] for i in items}):
  c=Counter()
  for item in items:
   if item['row']['profile_id']==p:c.update(one(transform(item,presence,typer),presence,THRESHOLD))
  profiles[p]=finish(c)
 after=dig(INITIAL);result={'schema_version':'chemvector-identity-aligned-direct-graph-shared-cycle-valence-r12-0.1','complete':True,'read_only':True,'training_performed':False,'gpu_used':False,'sealed_test_read':False,'production_checkpoint_written':False,'images':len(items),'profile_metadata_used':False,'all':finish(total),'profiles':profiles,'initial_checkpoint_sha256':SHA,'initial_checkpoint_sha256_after':after,'selection':'diagnostic only; shared type head plus target-free cycle/order repair','gate':{'checkpoint_unchanged':after==SHA,'sealed_test_read':False,'production_go':False}}
 OUT.mkdir(parents=True,exist_ok=False);(OUT/'report.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n');print(json.dumps({'output':str(OUT/'report.json'),'all':result['all'],'profiles':profiles},indent=2))
if __name__=='__main__':run()
