#!/usr/bin/env python3
"""Evaluate graph-level type correction on the frozen t3f presence shadow."""
from __future__ import annotations
import itertools,json,hashlib,sys,os
from collections import Counter
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import tools.train_r17_pair_incidence_known_clean_cpu as base
from tools.train_r17_pair_incidence_known_clean_cpu import BOND_TYPES
from tools.train_direct_graph_pair_scorer_cpu_r9 import SHA
from tools.audit_direct_graph_pair_scorer_chemistry_r9 import sanitize
OUT=ROOT/os.environ.get("R17_T3F_OUT","outputs/audits/chemvector-r17-t3f-graph-type-correction-cpu")
COHORT=ROOT/"outputs/datasets/chemvector-r17-residual-density-source-20261020-t3f/manifest.json"
SIDE=ROOT/os.environ.get("R17_T3F_PRESENCE_SIDE", "outputs/audits/chemvector-r17-residual-density-pair-incidence-cpu-20261020-t3f/sidecar-state.pt")
ORDER=(1.,2.,3.,1.5); CAP=4
# Keep candidate selection aligned with the t3f threshold when requested.
base.PAIR_THRESHOLD=float(os.environ.get("R17_PAIR_THRESHOLD", str(base.PAIR_THRESHOLD)))
def digest(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def project(item,head,lam):
 z=item['base_logits']+head(item['features']); ids=torch.where(z.sigmoid()>=base.PAIR_THRESHOLD)[0].tolist()
 edges=[]; logits=[]
 for i in ids:
  a,b=map(int,item['pairs'][i].tolist()); edges.append(tuple(sorted((a,b))));logits.append(item['chemical_logits'][i])
 if lam == 0.0:
  # Contract invariant: zero penalty is exactly the frozen chemical route.
  return ids, edges, {i: int(logits[i].argmax()) for i in range(len(edges))}
 if len(edges)>10: # retain lowest-margin six for bounded joint search
  order=sorted(range(len(edges)),key=lambda i:float(logits[i].max()-logits[i].topk(2).values[-1]))
  free=order[:6]; fixed={i:int(logits[i].argmax()) for i in order[6:]}
 else: free=list(range(len(edges)));fixed={}
 best=None
 for ks in itertools.product(range(4),repeat=len(free)):
  a=dict(fixed);a.update({i:k for i,k in zip(free,ks)}); deg=Counter(); score=0.
  for i,(u,v) in enumerate(edges): deg[u]+=ORDER[a[i]];deg[v]+=ORDER[a[i]];score+=float(torch.log_softmax(logits[i],-1)[a[i]])
  score-=lam*sum(max(0.,d-CAP)**2 for d in deg.values()); key=(score,tuple(a[i] for i in range(len(edges))))
  if best is None or key>best[0]:best=(key,a)
 return ids,edges,best[1]
def run():
 if OUT.exists():raise FileExistsError(OUT)
 initial=base.INITIAL
 if digest(initial)!=SHA:raise RuntimeError('production checkpoint changed')
 m=json.loads(COHORT.read_text()); base.SOURCE=COHORT.parent/'samples'; tr,ho=base.choose(m)
 model,cfg,_=base.load_bundle_for_probe(initial,SHA);model.eval().requires_grad_(False)
 st=torch.load(SIDE,map_location='cpu',weights_only=False);head=base.Residual(262);head.load_state_dict(st['state_dict']);head.eval()
 items=[base.extract(r,model,cfg) for r in ho]; rows=[]
 with torch.inference_mode():
  for it in items:
   for lam in (0.,1.,4.):
    ids,edges,a=project(it,head,lam)
    # labels/type labels are evaluation-only; presence selection is target-free.
    true=sum(it['labels'][j]>.5 for j in ids); typed=sum((it['labels'][j]>.5 and BOND_TYPES[int(it['chemical_logits'][j].argmax())]==it['type_labels'][j]) for j in ids)
    corr=0
    for i,j in enumerate(ids):
     if it['labels'][j]>.5 and BOND_TYPES[a[i]]==it['type_labels'][j]:corr+=1
    rows.append({'profile':it['profile'],'lam':lam,'pred':len(ids),'true':int(true),'typed_base':int(typed),'typed_corr':corr,'target':len(it['target_edges'])})
 summary={}
 for lam in (0.,1.,4.):
  q=[r for r in rows if r['lam']==lam]; summary[str(lam)]={}
  for p in ('oa13','oa33'):
   z=[r for r in q if r['profile']==p]; s={k:sum(r[k] for r in z) for k in ('pred','true','typed_base','typed_corr','target')}; summary[str(lam)][p]={'images':len(z),'edge_precision':s['true']/max(s['pred'],1),'edge_recall':s['true']/max(s['target'],1),'typed_precision':s['typed_corr']/max(s['pred'],1),'typed_recall':s['typed_corr']/max(s['target'],1),'baseline_typed_precision':s['typed_base']/max(s['pred'],1),'baseline_typed_recall':s['typed_base']/max(s['target'],1)}
 report={'schema_version':'chemvector-r17-t3f-graph-type-correction-cpu-0.1','complete':True,'training_performed':False,'gpu_used':False,'sealed_test_read':False,'production_checkpoint_written':False,'summary':summary,'policy':{'edge_set_frozen':True,'target_free_type_routing':True,'production_go':False},'initial_checkpoint_sha256':SHA,'initial_checkpoint_sha256_after':digest(initial),'gate':{'checkpoint_unchanged':digest(initial)==SHA,'sealed_test_read':False,'production_go':False}}
 OUT.mkdir(parents=True);(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'output':str(OUT/'report.json'),'summary':summary},indent=2))
if __name__=='__main__':run()
