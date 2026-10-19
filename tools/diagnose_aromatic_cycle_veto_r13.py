#!/usr/bin/env python3
"""Target-free aromatic cycle veto diagnostic for the r13 type sidecar."""
from __future__ import annotations
from collections import Counter
import hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from chemvector.bond_object import BOND_TYPES
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from tools.train_direct_graph_pair_scorer_cpu_r9 import DATA,SHA,_extract,rows,DirectPairResidual
from tools.train_direct_graph_type_profile_residual_cpu_r13 import TypeResidual
from tools.audit_direct_graph_pair_scorer_chemistry_r9 import one,finish
TRIAL=ROOT/"outputs/audits/chemvector-identity-aligned-direct-graph-pair-scorer-cpu-r10-coverage";TS=ROOT/"outputs/audits/chemvector-identity-aligned-direct-graph-type-profile-residual-cpu-r13/type-state.pt";OUT=ROOT/"outputs/audits/chemvector-identity-aligned-direct-graph-aromatic-cycle-veto-r13-v4"
SCALE=.5;THRESHOLD=.72
def dig(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def has_path(adj,a,b,skip):
 seen={a};stack=[a]
 while stack:
  u=stack.pop()
  for v in adj.get(u,()):
   if tuple(sorted((u,v)))==skip:continue
   if v==b:return True
   if v not in seen:seen.add(v);stack.append(v)
 return False
def type_stats(item, q, presence):
 with torch.inference_mode(): logits=item['base_logits']+presence(item['features'])
 pairs=item['pairs'];slots=item['slots'];mapping=item['mapping'];chosen=torch.where(logits.sigmoid()>=THRESHOLD)[0].tolist();correct=total=true=pred=0;target=item['target_edges']
 seen=set()
 for i in chosen:
  l,r=(int(v) for v in pairs[i].tolist());
  if l not in set(slots.tolist()) or r not in set(slots.tolist()):continue
  li=int((slots==l).nonzero(as_tuple=False)[0]);ri=int((slots==r).nonzero(as_tuple=False)[0]);
  if li not in mapping or ri not in mapping:continue
  edge=tuple(sorted((mapping[li],mapping[ri]))); pred+=1
  if edge in target:
   true+=1
   if edge not in seen:
    seen.add(edge); total+=1; kind=BOND_TYPES[int(q['chemical_logits'][i].argmax())]
    correct+=int(kind==q['target_type_by_edge'].get(edge))
 return {'predicted_edges':pred,'true_edges':true,'target_edges':len(target),'type_correct':correct,'type_rows':total}
def veto_item(item,presence,head,repair_valence=False):
 q=dict(item); logits=item['chemical_logits']+SCALE*head[item['row']['profile_id']](item['features']);pairs=item['pairs'];slots=item['slots']; sel=torch.where((item['base_logits']+presence(item['features'])).sigmoid()>=THRESHOLD)[0].tolist(); st={int(x):i for i,x in enumerate(slots.tolist())}; edges=[]
 for i in sel:
  l,r=(int(v) for v in pairs[i].tolist());
  if l in st and r in st and st[l]!=st[r]:edges.append((st[l],st[r],i))
 adj={}
 for a,b,_ in edges:adj.setdefault(a,set()).add(b);adj.setdefault(b,set()).add(a)
 changed=0; out=logits.clone()
 for a,b,i in edges:
  if int(logits[i].argmax())==3 and not has_path(adj,a,b,tuple(sorted((a,b)))):
   out[i,3]=-1e4;out[i,0]=logits[i].max()+1.0;changed+=1
 if repair_valence:
  # Target-free conservative carbon-like bond-order cap.  This is a
  # diagnostic only: it does not know the element and therefore cannot be
  # promoted without an element-conditioned audit.
  order={0:1.0,1:2.0,2:3.0,3:1.5}
  for _ in range(32):
   kinds={i:int(out[i].argmax()) for _,_,i in edges}; totals={}
   for a,b,i in edges:
    totals[a]=totals.get(a,0.0)+order[kinds[i]];totals[b]=totals.get(b,0.0)+order[kinds[i]]
   bad=[u for u,v in totals.items() if v>4.0+1e-6]
   if not bad:break
   candidates=[(i,kinds[i],float((out[i].max()-out[i].topk(2).values[-1]).detach())) for a,b,i in edges if any(u in (a,b) for u in bad) and kinds[i]!=0]
   if not candidates:break
   i,_,_=min(candidates,key=lambda z:(-order[z[1]],z[2],z[0]))
   out[i,0]=out[i].max()+1.0;out[i,1:]=out[i,1:]-1e4
   changed+=1
 q['chemical_logits']=out;q['_vetoed']=changed;return q,changed
def run():
 if dig(INITIAL)!=SHA:raise RuntimeError('production checkpoint changed')
 ps=torch.load(TRIAL/'scorer-state.pt',map_location='cpu',weights_only=False);presence=DirectPairResidual(262);presence.load_state_dict(ps['state_dict']);presence.eval();ts=torch.load(TS,map_location='cpu',weights_only=False);heads={p:TypeResidual(262) for p in ts['heads']}
 for p,h in heads.items():h.load_state_dict(ts['heads'][p]);h.eval()
 model,cfg,_=load_bundle_for_probe(INITIAL,SHA);model.eval().requires_grad_(False);hrs=sorted((r for r in rows() if r.get('split')=='fresh_holdout'),key=lambda x:x['sample_key']);items=[_extract(r,model,cfg,need_labels=True) for r in hrs]
 base=Counter();veto=Counter();repair=Counter();veto_types=Counter();repair_types=Counter();changed=0;repair_changed=0; by_profile={}; repair_profiles={}
 for item in items:
  q,n=veto_item(item,presence,heads);changed+=n;veto.update(one(q,presence,THRESHOLD));veto_types.update(type_stats(item,q,presence))
  q2,n2=veto_item(item,presence,heads,True);repair_changed+=n2;repair.update(one(q2,presence,THRESHOLD));repair_types.update(type_stats(item,q2,presence))
 # base above is intentionally same q; recompute no-veto separately below.
 plain=Counter()
 for item in items:
  q=dict(item);q['chemical_logits']=item['chemical_logits']+SCALE*heads[item['row']['profile_id']](item['features']);plain.update(one(q,presence,THRESHOLD))
 for p in sorted({i['row']['profile_id'] for i in items}):
  c=Counter(); rc=Counter()
  for item in items:
   if item['row']['profile_id']==p:
    q,_=veto_item(item,presence,heads);c.update(one(q,presence,THRESHOLD))
    q,_=veto_item(item,presence,heads,True);rc.update(one(q,presence,THRESHOLD))
  by_profile[p]=finish(c)
  repair_profiles[p]=finish(rc)
 result={'schema_version':'chemvector-identity-aligned-direct-graph-aromatic-cycle-veto-r13-0.1','complete':True,'read_only':True,'training_performed':False,'gpu_used':False,'sealed_test_read':False,'production_checkpoint_written':False,'images':len(items),'vetoed_edges':changed,'valence_repaired_edges':repair_changed,'plain':finish(plain),'cycle_veto':finish(veto),'cycle_veto_type_stats':dict(veto_types),'cycle_veto_profiles':by_profile,'cycle_plus_valence_repair':finish(repair),'cycle_plus_valence_type_stats':dict(repair_types),'cycle_plus_valence_profiles':repair_profiles,'selection':'diagnostic only; target-free cycle/order rules, no promotion','initial_checkpoint_sha256':SHA,'initial_checkpoint_sha256_after':dig(INITIAL),'gate':{'checkpoint_unchanged':dig(INITIAL)==SHA,'sealed_test_read':False,'production_go':False}}
 OUT.mkdir(parents=True,exist_ok=False);(OUT/'report.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n');print(json.dumps({'output':str(OUT/'report.json'),'vetoed_edges':changed,'plain':result['plain'],'cycle_veto':result['cycle_veto']},indent=2))
if __name__=='__main__':run()
