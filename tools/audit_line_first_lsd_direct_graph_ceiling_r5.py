"""Target-free direct graph ceiling: new-LSD nodes -> graph, no vertex head.

Every new-LSD provisional cluster is an atom-like node.  A cluster endpoint
near a visible glyph box is routed to that glyph node; otherwise it remains an
implicit node.  All retained LSD segments become synthetic SINGLE edges.
This is an isolated CPU ceiling: labels are joined only after proposal
generation for metrics and oracle chemistry sanitization.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
from rdkit import rdBase
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools import audit_line_first_graph_chemistry_roundtrip_r5 as chemistry
from tools import audit_line_first_lsd_all_edge_ownership_r5 as new_lsd
from tools.audit_line_first_graph_traversal_r5 import cluster_endpoints
from chemvector.components.vision.line_segment_rescue import LineSegmentRescueConfig
DATA=chemistry.DATA; OUT=ROOT/'outputs/audits/chemvector-identity-aligned-line-first-lsd-direct-graph-ceiling-r5/report.json'; SHA='09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1'; IMAGE=768.; RADIUS=24.
def boxes(base):
 r=json.loads(Path(str(base)+'.text-records.json').read_text()); g={};
 for x in r.get('glyphs',[]):
  a=int(x['atom_id']); b=x.get('bbox');
  if not b: continue
  if a not in g:g[a]=list(map(float,b))
  else:g[a]=[min(g[a][0],b[0]),min(g[a][1],b[1]),max(g[a][2],b[2]),max(g[a][3],b[3])]
 return g
def finish(c):
 d=dict(c); d['edge_precision']=c['true_edges']/max(c['mapped_edges'],1); d['edge_recall']=c['true_edges']/max(c['gt_bonds'],1); d['oracle_sanitize_rate']=c['sanitize_ok']/max(c['images'],1); d['unresolved_rate']=c['unresolved']/max(c['line_nodes'],1); return d
def evaluate(row):
 key=row['sample_key']; base=DATA/'samples'/key; image=Image.open(Path(str(base)+'.png')).convert('L'); seg=new_lsd._alt_detect(image,config=LineSegmentRescueConfig(maximum_candidates=512)); pts=seg.reshape(-1,2) if len(seg) else np.empty((0,2)); labels,centers=cluster_endpoints(pts,10.); bxs=boxes(base)
 # direct node ids: cluster ids, with text aliases replacing nearby clusters
 node_of={i:i for i in range(len(centers))}; text_of={}
 for aid,b in bxs.items():
  c=np.asarray([(b[0]+b[2])/2,(b[1]+b[3])/2]); hits=[(float(np.linalg.norm(centers[i]-c)),i) for i in range(len(centers)) if np.linalg.norm(centers[i]-c)<=RADIUS]
  if hits:
   tid=-1-aid; text_of[aid]=tid
   for _,i in sorted(hits): node_of[i]=tid
 selected=set()
 for i in range(len(seg)):
  a,b=int(labels[2*i]),int(labels[2*i+1]); na,nb=node_of.get(a,a),node_of.get(b,b)
  if na!=nb:selected.add(tuple(sorted((na,nb))))
 layers=json.loads(Path(str(base)+'.layers.json').read_text()); graph=json.loads(Path(str(base)+'.graph.json').read_text()); targets=np.asarray([a['center'] for a in layers['atoms']],np.float32); gt={tuple(sorted((int(z['begin_atom']),int(z['end_atom'])))) for z in graph['bonds']}
 def tid(n):
  n=int(n); return (-n-1) if n<0 else (int(np.argmin(np.linalg.norm(targets-centers[n],axis=1))) if len(centers) and float(np.min(np.linalg.norm(targets-centers[n],axis=1)))<=RADIUS else None)
 mapped={tuple(sorted((tid(a),tid(b)))) for a,b in selected if tid(a) is not None and tid(b) is not None and tid(a)!=tid(b)}; true=mapped&gt
 oracle_nodes=sorted({x for p in mapped for x in p}); idx={x:i for i,x in enumerate(oracle_nodes)}; edges={(idx[a],idx[b]):1 for a,b in mapped if a!=b}; elems=[str(layers['atoms'][x].get('element') or 'C') for x in oracle_nodes]; arom=[bool(layers['atoms'][x].get('aromatic',False)) for x in oracle_nodes]; ok,_,_=chemistry._sanitize_graph([(x,elems[i],arom[i]) for i,x in enumerate(oracle_nodes)],edges)
 unresolved=sum(1 for i in range(len(centers)) if i not in node_of or node_of[i]==i)
 return Counter(images=1,line_nodes=len(centers),selected_edges=len(selected),mapped_edges=len(mapped),true_edges=len(true),gt_bonds=len(gt),sanitize_ok=int(ok),unresolved=unresolved)
def main():
 rdBase.DisableLog('rdApp.error');
 if chemistry.file_digest(chemistry.INITIAL)!=SHA: raise RuntimeError('checkpoint changed')
 new_lsd._install(); rows=sorted([json.loads(x) for x in (DATA/'samples.jsonl').read_text().splitlines() if x.strip() and json.loads(x).get('render_status')=='accepted'],key=lambda x:x['sample_key']); out={}
 for split in sorted({r['split'] for r in rows}):
  total=Counter(); prof=defaultdict(Counter)
  for r in [z for z in rows if z['split']==split]:
   v=evaluate(r); total.update(v); prof[r['profile_id']].update(v)
  out[split]={'all':finish(total),'by_profile':{k:finish(v) for k,v in sorted(prof.items())}}
 h=out['holdout']['all']; hp=out['holdout']['by_profile']; gate={'checkpoint_unchanged':chemistry.file_digest(chemistry.INITIAL)==SHA,'sealed_test_read':False,'holdout_precision_ge_0.95':h['edge_precision']>=.95,'holdout_recall_ge_0.80':h['edge_recall']>=.80,'all_profiles_recall_ge_0.75':all(v['edge_recall']>=.75 for v in hp.values()),'all_profiles_sanitize_ge_0.90':all(v['oracle_sanitize_rate']>=.90 for v in hp.values()),'production_go':False,'gpu_go':False}
 report={'schema_version':'chemvector-identity-aligned-line-first-lsd-direct-graph-ceiling-r5-0.1','complete':True,'read_only':True,'training_performed':False,'gpu_used':False,'sealed_test_read':False,'production_checkpoint_written':False,'proposal_contract':'new LSD provisional cluster nodes directly -> optional glyph-box alias -> every segment synthetic SINGLE','splits':out,'holdout':h,'holdout_profiles':hp,'gate':gate}; OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n'); print(json.dumps({'output':str(OUT),'holdout':h,'gate':gate},ensure_ascii=False))
if __name__=='__main__':main()
