"""Read-only pixel/geometry diagnostic for the failed cap/stroke replay.

It measures nearest raw proposal distance to each target atom before any
selector output.  This distinguishes missing/localization candidates from a
later selector drop and reports visible-text versus implicit-carbon strata.
"""
from __future__ import annotations
from collections import Counter,defaultdict
import hashlib,json,os
from pathlib import Path
import torch
from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.data import CurriculumCollator,load_sample
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from chemvector.training.curriculum.gradient_preflight import state_digest
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'outputs/datasets/chemvector-oa13-oa33-cap-stroke-panel-20261017-r1'; OUT=Path(os.environ.get('CHEMVECTOR_CAP_STROKE_PIXEL_OUTPUT',str(ROOT/'outputs/audits/chemvector-oa13-cap-stroke-pixel-geometry-20261017-r1/report.json'))); SHA='09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1'
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def run():
 if OUT.exists(): raise FileExistsError(OUT)
 rows=json.loads((DATA/'samples.json').read_text()); model,cfg,payload=load_bundle_for_probe(INITIAL,SHA)
 if not torch.cuda.is_available() or digest(INITIAL)!=SHA: raise RuntimeError('binding')
 dev=torch.device('cuda'); model.to(dev).eval().requires_grad_(False); before=state_digest(model); coll=CurriculumCollator(AtomBondSceneConfig(atom_queries=cfg.maximum_atoms,bond_queries=cfg.maximum_bonds),purpose='diagnostic'); by=defaultdict(Counter)
 for row in rows:
  s=load_sample(DATA,row['sample_key'],purpose='diagnostic'); b=coll([s]); x={k:v.to(dev) for k,v in b.items() if isinstance(v,torch.Tensor)}
  with torch.inference_mode(): f=structure_forward(model,x['pixel_values'],cfg,payload['provenance']['atom_threshold'])
  raw=f[3]; geom=raw['atom_geometry'][0,:,:2]; mask=raw['atom_mask'][0]; scores=raw['atom_scores'][0]; anchors=x['atom_anchors'][0]; n=int(x['atom_mask'][0].sum()); texts=b['visible_text'][0]; c=by[row['profile_id']]; c['images']+=1
  for tid in range(n):
   radius=float(cfg.atom_match_radius_px); d=((geom-anchors[tid])**2).sum(dim=1).sqrt()*cfg.image_size; valid=d[mask]; nearest=float(valid.min().item()) if valid.numel() else 1e9; count=int((valid<=radius).sum().item()) if valid.numel() else 0; max_score=float(scores[d.argmin()].item()) if scores.numel() else 0.0; vis=bool(texts[tid]); bucket='visible_text' if vis else 'implicit_or_unlabeled'; c['targets']+=1; c[bucket]+=1; c[bucket+'_within_match_radius'] += int(nearest <= radius); c[bucket+'_within_8px'] += int(nearest <= 8.0); c[bucket+'_zero_candidates'] += int(count==0); c[bucket+'_multi_candidates'] += int(count>1); c[bucket+'_nearest_sum_px'] += nearest; c[bucket+'_nearest_max_px'] = max(c[bucket+'_nearest_max_px'],nearest); c[bucket+'_score_sum'] += max_score
 after=state_digest(model)
 if before!=after: raise RuntimeError('state changed')
 report={'schema_version':'chemvector-oa13-cap-stroke-pixel-geometry-0.1','complete':True,'read_only':True,'training_performed':False,'gpu_used':True,'sealed_test_read':False,'initial_sha256':SHA,'initial_sha256_after':digest(INITIAL),'state_before':before,'state_after':after,'match_radius_px':float(cfg.atom_match_radius_px),'by_profile':{k:dict(v) for k,v in sorted(by.items())}}
 for c in report['by_profile'].values():
  for b in ('visible_text','implicit_or_unlabeled'):
   n=c.get(b,0); c[b+'_within_match_radius_fraction']=c.get(b+'_within_match_radius',0)/max(n,1); c[b+'_within_8px_fraction']=c.get(b+'_within_8px',0)/max(n,1); c[b+'_mean_nearest_px']=c.get(b+'_nearest_sum_px',0)/max(n,1); c[b+'_mean_score']=c.get(b+'_score_sum',0)/max(n,1)
 OUT.parent.mkdir(parents=True,exist_ok=False); OUT.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2)); return report
if __name__=='__main__': run()
