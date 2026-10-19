"""CPU-contract audit for dense multi-slot candidate expansion.

No new candidates are fed to the model or decoder.  The audit estimates the
best possible geometric recall from all local heatmap peaks and the resulting
candidate precision before any assignment/training.
"""
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
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'outputs/datasets/chemvector-oa13-oa33-cap-stroke-panel-20261017-r1'; OUT=Path(os.environ.get('CHEMVECTOR_MULTISLOT_OUTPUT',str(ROOT/'outputs/audits/chemvector-oa13-cap-stroke-multislot-admission-20261017-r1/report.json'))); SHA='09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1'
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def run():
 if OUT.exists(): raise FileExistsError(OUT)
 rows=json.loads((DATA/'samples.json').read_text()); model,cfg,payload=load_bundle_for_probe(INITIAL,SHA); dev=torch.device('cuda')
 if not torch.cuda.is_available() or digest(INITIAL)!=SHA: raise RuntimeError('binding')
 model.to(dev).eval().requires_grad_(False); before=state_digest(model); coll=CurriculumCollator(AtomBondSceneConfig(atom_queries=cfg.maximum_atoms,bond_queries=cfg.maximum_bonds),purpose='diagnostic'); by=defaultdict(Counter)
 for row in rows:
  s=load_sample(DATA,row['sample_key'],purpose='diagnostic'); b=coll([s]); x={k:v.to(dev) for k,v in b.items() if isinstance(v,torch.Tensor)}
  with torch.inference_mode(): f=structure_forward(model,x['pixel_values'],cfg,payload['provenance']['atom_threshold'])
  ao=f[1]; heat=ao['heatmap_logits'].sigmoid()[0,0]; peaks=heat==F.max_pool2d(heat[None,None],3,1,1)[0,0]; py,px=torch.where(peaks); anchors=ao['anchors'][0].permute(1,2,0)[py,px]; scores=heat[py,px]; n=int(x['atom_mask'][0].sum()); target=x['atom_anchors'][0,:n]; d=torch.cdist(anchors,target)*cfg.image_size; near=d.min(1).values; positives=near<=float(cfg.atom_match_radius_px); covered=d.min(0).values<=float(cfg.atom_match_radius_px); c=by[row['profile_id']]; c['images']+=1; c['target_atoms']+=n; c['covered_targets']+=int(covered.sum()); c['all_peaks']+=len(anchors); c['positive_peaks']+=int(positives.sum()); c['top128_peaks']+=min(128,len(anchors)); c['top128_positive_peaks']+=int(positives[torch.argsort(scores,descending=True)[:128]].sum())
 after=state_digest(model)
 if before!=after: raise RuntimeError('state changed')
 report={'schema_version':'chemvector-oa13-cap-stroke-multislot-admission-0.1','complete':True,'read_only':True,'training_performed':False,'gpu_used':True,'sealed_test_read':False,'initial_sha256':SHA,'initial_sha256_after':digest(INITIAL),'state_before':before,'state_after':after,'match_radius_px':float(cfg.atom_match_radius_px),'by_profile':{k:dict(v) for k,v in sorted(by.items())},'gate':{'minimum_positive_train':50,'hard_negative_precision_min':0.85,'protected_retention_required':1.0,'training_admissible':False}}
 for c in report['by_profile'].values(): c['all_peak_positive_precision']=c['positive_peaks']/max(c['all_peaks'],1); c['top128_positive_precision']=c['top128_positive_peaks']/max(c['top128_peaks'],1); c['target_coverage']=c['covered_targets']/max(c['target_atoms'],1)
 OUT.parent.mkdir(parents=True,exist_ok=False); OUT.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2)); return report
if __name__=='__main__': run()
