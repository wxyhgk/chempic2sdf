"""CPU-only chemistry critic probe: demote predicted aromatic edges outside cycles.

Proposals use the selected target-free Hungarian ownership (r24, score lambda 8)
and frozen edge/type sidecar. A second in-memory graph applies a conservative
critic: an AROMATIC edge is retained only if it belongs to a 5/6-cycle; otherwise
it is demoted to SINGLE. GT is used only post-hoc. No training/checkpoint writes.
"""
from __future__ import annotations
from collections import Counter,defaultdict
import json,sys
from pathlib import Path
import numpy as np,torch,networkx as nx
from PIL import Image
from scipy.optimize import linear_sum_assignment
from rdkit import Chem,rdBase
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import audit_line_first_graph_chemistry_roundtrip_r5 as base
from tools.audit_line_first_candidate_alignment_r5 import read_jsonl,IMAGE
from tools.audit_line_first_graph_traversal_r5 import cluster_endpoints
from chemvector.components.vision.line_segment_rescue import LineSegmentRescueConfig,detect_raster_line_segments
from tools.audit_line_first_edge_bond_candidate_integration_r5 import strict_candidate_to_target
DATA=base.DATA; CAND=base.CANDIDATES; R=24.;L=8.
def own(cent,c):
 d=np.linalg.norm(cent[:,None,:]-c[None,:,:2],axis=-1); cost=np.where(d<=R,d-L*c[None,:,2],1e5+d);rr,cc=linear_sum_assignment(cost);return {int(i):int(j) for i,j in zip(rr,cc) if d[i,j]<=R}
def sanitize(nodes,edges):
 m=Chem.RWMol();loc={}
 for i,(_,e,ar) in enumerate(nodes):
  a=Chem.Atom(e or 'C');a.SetIsAromatic(bool(ar));loc[i]=m.AddAtom(a)
 try:
  for (u,v),k in edges.items():
   bt={1:Chem.BondType.SINGLE,2:Chem.BondType.DOUBLE,3:Chem.BondType.TRIPLE,4:Chem.BondType.AROMATIC,5:Chem.BondType.DATIVE}.get(int(k),Chem.BondType.SINGLE);m.AddBond(loc[u],loc[v],bt)
   if bt==Chem.BondType.AROMATIC:m.GetBondBetweenAtoms(loc[u],loc[v]).SetIsAromatic(True)
  x=m.GetMol();Chem.SanitizeMol(x);return True
 except Exception:return False
def finish(c):return {**dict(c),'edge_precision':c['true']/max(c['mapped'],1),'edge_recall':c['true']/max(c['gt'],1),'type_accuracy':c['tc']/max(c['tt'],1),'sanitize_baseline':c['baseok']/max(c['images'],1),'sanitize_critic':c['critok']/max(c['images'],1),'demoted_aromatic_rate':c['demoted']/max(c['aromatic_pred'],1)}
def one(row,cr,model,mean,std,th):
 k=row['sample_key'];b=DATA/'samples'/k;im=Image.open(b.with_suffix('.png')).convert('L');seg=detect_raster_line_segments(im,config=LineSegmentRescueConfig(maximum_candidates=512));pts=seg.reshape(-1,2) if len(seg) else np.empty((0,2),np.float32);lab,cent=cluster_endpoints(pts,16.);c=np.asarray([[v['x']*IMAGE,v['y']*IMAGE,float(v.get('score',0))] for v in cr],np.float64);nm=own(cent,c);f,_,_=base._proposals(k);keys=base._candidate_keys(k)
 with torch.inference_mode():el,tl=model(torch.from_numpy((f-mean)/std))
 sc=torch.sigmoid(el).numpy();kind=(torch.argmax(tl,1)+1).numpy();sel={}
 for i,(u,v) in enumerate(keys):
  if sc[i]<th or u not in nm or v not in nm:continue
  a,b2=nm[u],nm[v]
  if a==b2:continue
  q=tuple(sorted((a,b2))); old=sel.get(q)
  if old is None or sc[i]>old[0]:sel[q]=(float(sc[i]),int(kind[i]))
 basep=json.loads(b.with_suffix('.layers.json').read_text());gr=json.loads(b.with_suffix('.graph.json').read_text());tar=np.asarray([x['center'] for x in basep['atoms']],np.float32);c2t=strict_candidate_to_target(c[:,:2],tar);gt={tuple(sorted((int(x['begin_atom']),int(x['end_atom'])))) for x in gr['bonds']};gty={tuple(sorted((int(x['begin_atom']),int(x['end_atom'])))): (base.TYPE_NAMES.index(str(x.get('chemical_type','OTHER')).upper()) if str(x.get('chemical_type','OTHER')).upper() in base.TYPE_NAMES else 0) for x in gr['bonds']};mapped={};
 for (u,v),(_,kk) in sel.items():
  if u in c2t and v in c2t and c2t[u]!=c2t[v]:mapped[tuple(sorted((c2t[u],c2t[v])))] = kk
 true=set(mapped)&gt;tc=sum(mapped[e]==gty[e] for e in true)
 active=sorted(set(nm.values()));ix={u:i for i,u in enumerate(active)};nodes=[(u,str(basep['atoms'][c2t[u]]['element'] or 'C'),bool(basep['atoms'][c2t[u]].get('aromatic',False))) for u in active if u in c2t];ix={u:i for i,(u,_,_) in enumerate(nodes)};edges={(ix[u],ix[v]):kk for (u,v),(_,kk) in sel.items() if u in ix and v in ix};ok=sanitize(nodes,edges)
 g=nx.Graph();g.add_edges_from((u,v) for (u,v),(_,kk) in sel.items() if kk==4);ring=set()
 for cyc in nx.cycle_basis(g):
  if len(cyc) in (5,6):ring.update(tuple(sorted((cyc[i],cyc[(i+1)%len(cyc)]))) for i in range(len(cyc)))
 dem=sum(1 for q,(_,kk) in sel.items() if kk==4 and q not in ring);sel2={q:(s,1 if kk==4 and q not in ring else kk) for q,(s,kk) in sel.items()};edges2={(ix[u],ix[v]):kk for (u,v),(_,kk) in sel2.items() if u in ix and v in ix};ok2=sanitize(nodes,edges2);ap=sum(kk==4 for _,kk in sel.values());return Counter(images=1,mapped=len(mapped),true=len(true),gt=len(gt),tc=tc,tt=len(true),baseok=int(ok),critok=int(ok2),demoted=dem,aromatic_pred=ap,crit_aromatic_true=sum(mapped.get(e)==4 for e in true if e in mapped))
def main():
 rdBase.DisableLog('rdApp.error');cr={r['sample_key']:r['candidates'] for r in read_jsonl(CAND)};st=torch.load(base.STATE,map_location='cpu',weights_only=False);m=base.EdgeBondMLP(len(st['feature_names']));m.load_state_dict(st['model']);m.eval();mean=np.asarray(st['mean'],np.float32);std=np.asarray(st['std'],np.float32);th=float(st['threshold']);rows=read_jsonl(DATA/'samples.jsonl');by=defaultdict(Counter);prof=defaultdict(Counter)
 for row in rows:
  x=one(row,cr[row['sample_key']],m,mean,std,th);by[row['split']].update(x);prof[row['split']+':'+row['profile_id']].update(x)
 out={'schema_version':'chemvector-identity-aligned-line-first-ownership-aromatic-critic-r5-0.1','read_only':True,'complete':True,'training_performed':False,'gpu_used':False,'sealed_test_read':False,'production_checkpoint_written':False,'proposal_contract':'target-free Hungarian ownership r24/lambda8; cycle critic is post-proposal','splits':{k:finish(v) for k,v in sorted(by.items())},'by_split_profile':{k:finish(v) for k,v in sorted(prof.items())},'gate':{'production_go':False,'gpu_training_admissible':False,'next':'only consider a learned/typed aromatic critic if full profile audit passes'}};p=ROOT/'outputs/audits/chemvector-identity-aligned-line-first-ownership-aromatic-critic-r5/report.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n');print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
