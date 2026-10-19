"""Dense heatmap peak diagnostic for raw-gap targets in the cap panel."""
from __future__ import annotations
from collections import Counter,defaultdict
import hashlib,json,os
from pathlib import Path
import torch
from torch.nn import functional as F
from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.data import CurriculumCollator,load_sample
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from chemvector.training.curriculum.gradient_preflight import state_digest
from chemvector.training.graph.relations.train_direct_atom_pair_bond import match_atom_targets
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'outputs/datasets/chemvector-oa13-oa33-cap-stroke-panel-20261017-r1'; OUT=Path(os.environ.get('CHEMVECTOR_DENSE_PEAK_OUTPUT',str(ROOT/'outputs/audits/chemvector-oa13-cap-stroke-dense-peak-20261017-r1/report.json'))); SHA='09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1'
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def run():
 if OUT.exists(): raise FileExistsError(OUT)
 rows=json.loads((DATA/'samples.json').read_text()); model,cfg,payload=load_bundle_for_probe(INITIAL,SHA); dev=torch.device('cuda')
 if not torch.cuda.is_available() or digest(INITIAL)!=SHA: raise RuntimeError('binding')
 model.to(dev).eval().requires_grad_(False); before=state_digest(model); coll=CurriculumCollator(AtomBondSceneConfig(atom_queries=cfg.maximum_atoms,bond_queries=cfg.maximum_bonds),purpose='diagnostic'); by=defaultdict(Counter)
 for row in rows:
  s=load_sample(DATA,row['sample_key'],purpose='diagnostic'); b=coll([s]); x={k:v.to(dev) for k,v in b.items() if isinstance(v,torch.Tensor)}
  with torch.inference_mode(): f=structure_forward(model,x['pixel_values'],cfg,payload['provenance']['atom_threshold'])
  raw=f[3]; mapping,_=match_atom_targets(raw,x,cfg); n=int(x['atom_mask'][0].sum()); matched={int(v) for v in mapping[0].tolist() if int(v)>=0}; missing=sorted(set(range(n))-matched)
  if not missing: continue
  heat=f[1]['heatmap_logits'].sigmoid()[0,0]; peaks=heat==F.max_pool2d(heat[None,None],3,1,1)[0,0]; py,px=torch.where(peaks); ps=heat[py,px]; order=torch.argsort(ps,descending=True); px,py,ps=px[order],py[order],ps[order]; anchors=f[1]['anchors'][0].permute(1,2,0)[py,px]; atoms={int(a['atom_id']):a for a in s['layers']['atoms']}; profile=row['profile_id']
  for tid in missing:
   d=torch.linalg.vector_norm(anchors-x['atom_anchors'][0,tid][None],dim=-1)*cfg.image_size; j=int(torch.argmin(d)); a=atoms[tid]; role='visible_text' if a['visible_text'] else 'implicit'; c=by[profile]; c['gap_targets']+=1; c[role]+=1; c[role+'_within6']+=int(float(d[j])<=6); c[role+'_top128']+=int(j<128); c[role+'_top256']+=int(j<256); c[role+'_distance_sum']+=float(d[j]); c[role+'_score_sum']+=float(ps[j])
 after=state_digest(model)
 if before!=after: raise RuntimeError('state changed')
 report={'schema_version':'chemvector-oa13-cap-stroke-dense-peak-gap-0.1','complete':True,'read_only':True,'training_performed':False,'gpu_used':True,'sealed_test_read':False,'initial_sha256':SHA,'initial_sha256_after':digest(INITIAL),'state_before':before,'state_after':after,'by_profile':{k:dict(v) for k,v in sorted(by.items())}}
 for c in report['by_profile'].values():
  for role in ('visible_text','implicit'):
   n=c.get(role,0); c[role+'_within6_fraction']=c.get(role+'_within6',0)/max(n,1); c[role+'_top128_fraction']=c.get(role+'_top128',0)/max(n,1); c[role+'_top256_fraction']=c.get(role+'_top256',0)/max(n,1); c[role+'_mean_distance_px']=c.get(role+'_distance_sum',0)/max(n,1); c[role+'_mean_peak_score']=c.get(role+'_score_sum',0)/max(n,1)
 OUT.parent.mkdir(parents=True,exist_ok=False); OUT.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2)); return report
if __name__=='__main__': run()
