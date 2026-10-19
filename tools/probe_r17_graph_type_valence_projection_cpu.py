#!/usr/bin/env python3
"""Target-free graph-level bond-type projection shadow.

The frozen candidate edge set is unchanged.  For every image, aromatic edges
from the frozen aromatic sidecar remain fixed; the remaining edge types are
projected jointly by maximizing chemical-head log probability with a soft
valence-overflow penalty.  Ground-truth graph labels are joined only after
prediction for the audit.  This is CPU-only and never writes a checkpoint.
"""
from __future__ import annotations
from collections import Counter
import hashlib, itertools, json, math, os
from pathlib import Path
import torch
import sys
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools.audit_direct_graph_pair_scorer_chemistry_r9 import sanitize
from tools.train_graph_context_aromatic_cpu_r17 import AromaticContextHead
from tools.train_element_context_head_cpu_r17 import ElementContextHead, ELEMENTS
from tools.train_type_residual_cpu_r17 import TypeContextHead
from tools.train_direct_graph_pair_scorer_cpu_r9 import SHA

RUN=ROOT/"outputs/audits/chemvector-graph-context-aromatic-cpu-r17-all-m128-t0p72-c1-panel128"
CACHE=RUN/"feature-cache.pt"; AROM=RUN/"sidecar-state.pt"
ELEM=ROOT/"outputs/audits/chemvector-element-context-head-cpu-r17-all-m128-panel128/head-state.pt"
TYPE=ROOT/"outputs/audits/chemvector-type-residual-cpu-r17-all-m128-panel128/sidecar-state.pt"
SOURCE=ROOT/"outputs/datasets/structure-expansion-batch-12272-20260923/samples"
OUT=ROOT/os.environ.get("R17_GRAPH_PROJ_OUT","outputs/audits/chemvector-r17-graph-type-valence-projection-cpu-m128")
CAP={"C":4,"N":3,"O":2,"S":6,"P":5,"F":1,"Cl":1,"Br":1,"I":1,"B":3}
TYPE_NAMES=("SINGLE","DOUBLE","TRIPLE","AROMATIC")
ORDER=(1.,2.,3.,1.5)

def dig(p):
 h=hashlib.sha256();
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()

def truth(key):
 g=json.loads((SOURCE/f"{key}.graph.json").read_text()); out={}
 for b in g["bonds"]:
  e=tuple(sorted((int(b["begin_atom"]),int(b["end_atom"]))))
  out[e]=str(b.get("chemical_type","SINGLE")).upper()
 return out

def element_features(r,nodes):
 by={n:[] for n in nodes}
 for i,(a,b,_) in enumerate(r["pairs"]):
  by[a].append(r["type_features"][i]); by[b].append(r["type_features"][i])
 out=[]
 for n in nodes:
  z=torch.stack(by[n]); out.append(torch.cat((r["atom_features"][n].float(),z.mean(0),z.max(0).values,torch.tensor([float(len(z))/8.]))))
 return torch.stack(out)

def project(logits, edges, aromatic, elements, lam):
 """Joint MAP over <=3^N assignments; no labels/targets are consulted."""
 free=[i for i,e in enumerate(edges) if e not in aromatic]
 if len(free)>6:
  # Keep the probe bounded while assigning omitted edges to their local MAP.
  omitted=set(free[6:]); free=free[:6]
 else: omitted=set()
 fixed={i:3 for i,e in enumerate(edges) if e in aromatic}
 fixed.update({i:int(logits[i].argmax()) for i in omitted})
 best=None
 for kinds in itertools.product((0,1,2), repeat=len(free)):
  assign=dict(fixed); assign.update({i:k for i,k in zip(free,kinds)})
  deg=Counter(); score=0.
  for i,(a,b) in enumerate(edges):
   k=assign[i]; score+=float(torch.log_softmax(logits[i],-1)[k]); deg[a]+=ORDER[k]; deg[b]+=ORDER[k]
  overflow=sum(max(0.,deg[n]-CAP.get(elements[n],4))**2 for n in deg)
  score-=lam*overflow
  key=(score,tuple(assign[i] for i in range(len(edges))))
  if best is None or key>best[0]:best=(key,assign)
 return best[1]

def run():
 if OUT.exists():raise FileExistsError(OUT)
 initial=ROOT/"outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt"
 if dig(initial)!=SHA:raise RuntimeError("production checkpoint changed")
 c=torch.load(CACHE,map_location="cpu",weights_only=False); a=torch.load(AROM,map_location="cpu",weights_only=False)
 es=torch.load(ELEM,map_location="cpu",weights_only=False); ts=torch.load(TYPE,map_location="cpu",weights_only=False)
 ah=AromaticContextHead(526);ah.load_state_dict(a["state_dict"]);ah.eval()
 eh=ElementContextHead();eh.load_state_dict(es["state_dict"]);eh.eval()
 th=TypeContextHead(262);th.load_state_dict(ts["state_dict"]);th.eval()
 rows=[]
 with torch.inference_mode():
  for split,recs in c["records"].items():
   for r in recs:
    if not len(r["pairs"]):continue
    nodes=sorted({int(x) for p in r["pairs"] for x in p[:2]}); local={x:i for i,x in enumerate(nodes)}
    edges=[tuple(sorted((local[int(a)],local[int(b)]))) for a,b,_ in r["pairs"]]
    ap=ah(r["features"]).sigmoid(); arom={edges[i] for i,p in enumerate(ap.tolist()) if p>=.5}
    el=eh(element_features(r,nodes)); elements=[ELEMENTS[i] for i in el.argmax(-1).tolist()]
    logits=r["edge_chemical_logits"]
    target=truth(r["sample_key"]); mapping=r["mapping"]
    for lam in (0.,1.,4.):
     ass=project(logits,edges,arom,elements,lam)
     mapped={tuple(sorted((int(mapping[nodes[a]]),int(mapping[nodes[b]])))):TYPE_NAMES[k] for (a,b),k in zip(edges,[ass[i] for i in range(len(edges))]) if nodes[a] in mapping and nodes[b] in mapping}
     # sanitize predicted local graph with predicted elements/aromatic flags
     aromatic_nodes={u for e in arom for u in e}
     ok,err=sanitize([(n,elements[i],i in aromatic_nodes) for i,n in enumerate(nodes)],{e:k+1 for e,k in zip(edges,[ass[i] for i in range(len(edges))])})
     inter=set(mapped)&set(target); typed=sum(mapped.get(e)==t for e,t in target.items() if e in mapped)
     rows.append({"split":split,"sample_key":r["sample_key"],"lambda":lam,"sanitize":int(ok),"error":err,"pred":mapped,"target":target,"edge_tp":len(inter),"typed_tp":typed,"pred_edges":len(mapped),"target_edges":len(target)})
 summary={}
 for lam in (0.,1.,4.):
  summary[str(lam)]={}
  for split in ("train","dev","fresh_holdout"):
   q=[r for r in rows if r["lambda"]==lam and r["split"]==split]; tp=sum(x["edge_tp"] for x in q); typ=sum(x["typed_tp"] for x in q); pe=sum(x["pred_edges"] for x in q); te=sum(x["target_edges"] for x in q)
   summary[str(lam)][split]={"images":len(q),"sanitize_rate":sum(x["sanitize"] for x in q)/max(len(q),1),"edge_precision":tp/max(pe,1),"edge_recall":tp/max(te,1),"typed_precision":typ/max(pe,1),"typed_recall":typ/max(te,1),"valence_errors":sum(x["error"]=="valence" for x in q),"aromatic_errors":sum(x["error"]=="aromatic" for x in q)}
 report={"schema_version":"chemvector-r17-graph-type-valence-projection-cpu-0.1","complete":True,"read_only":True,"training_performed":False,"gpu_used":False,"sealed_test_read":False,"production_checkpoint_written":False,"cache":str(CACHE.relative_to(ROOT)),"summary":summary,"policy":{"edge_set_frozen":True,"target_free_projection":True,"labels_joined_posthoc":True,"production_go":False},"initial_checkpoint_sha256":SHA,"initial_checkpoint_sha256_after":dig(initial),"gate":{"checkpoint_unchanged":dig(initial)==SHA,"sealed_test_read":False,"production_go":False}}
 OUT.mkdir(parents=True);(OUT/"report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n"); print(json.dumps({"output":str(OUT/"report.json"),"summary":summary},indent=2,ensure_ascii=False))
if __name__=="__main__":run()
