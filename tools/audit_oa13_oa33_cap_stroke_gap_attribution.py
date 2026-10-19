"""Attribute cap/stroke replay raw gaps by visible text and element."""
from __future__ import annotations
from collections import Counter,defaultdict
import json,hashlib,os
from pathlib import Path
import torch
from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.data import CurriculumCollator,load_sample
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from chemvector.training.curriculum.gradient_preflight import state_digest
from chemvector.training.graph.relations.train_direct_atom_pair_bond import match_atom_targets
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'outputs/datasets/chemvector-oa13-oa33-cap-stroke-panel-20261017-r1'; OUT=ROOT/'outputs/audits/chemvector-oa13-oa33-cap-stroke-gap-attribution-20261017-r1/report.json'; SHA='09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1'
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def run():
 if OUT.exists(): raise FileExistsError(OUT)
 rows=json.loads((DATA/'samples.json').read_text()); model,cfg,payload=load_bundle_for_probe(INITIAL,SHA)
 if not torch.cuda.is_available() or digest(INITIAL)!=SHA: raise RuntimeError('binding')
 dev=torch.device('cuda'); model.to(dev).eval().requires_grad_(False); before=state_digest(model); coll=CurriculumCollator(AtomBondSceneConfig(atom_queries=cfg.maximum_atoms,bond_queries=cfg.maximum_bonds),purpose='diagnostic'); by=defaultdict(Counter)
 for row in rows:
  s=load_sample(DATA,row['sample_key'],purpose='diagnostic'); b=coll([s]); x={k:v.to(dev) for k,v in b.items() if isinstance(v,torch.Tensor)}
  with torch.inference_mode():
   forward=structure_forward(model,x['pixel_values'],cfg,payload['provenance']['atom_threshold']); raw=forward[3]; selected=forward[6]
  rm,_=match_atom_targets(raw,x,cfg); n=int(x['atom_mask'][0].sum()); ids=b['atom_ids'][0]; texts=b['visible_text'][0]; atoms={int(a['atom_id']):a for a in s['graph']['atoms']}; c=by[row['profile_id']]; c['images']+=1
  for tid in range(n):
   a=atoms[int(ids[tid])]; vis=bool(texts[tid]); key=f"{a['element']}|{'visible' if vis else 'implicit'}|{'aromatic' if a.get('aromatic') else 'aliphatic'}"; c['target_atoms']+=1; c['matched' if bool(rm[0,tid]>=0) else 'gap']+=1; c[('matched:' if bool(rm[0,tid]>=0) else 'gap:')+key]+=1
 after=state_digest(model)
 if before!=after: raise RuntimeError('state changed')
 report={'schema_version':'chemvector-oa13-oa33-cap-stroke-gap-attribution-0.1','complete':True,'read_only':True,'training_performed':False,'gpu_used':True,'sealed_test_read':False,'initial_sha256':SHA,'initial_sha256_after':digest(INITIAL),'state_before':before,'state_after':after,'by_profile':{k:dict(v) for k,v in sorted(by.items())}}
 OUT.parent.mkdir(parents=True,exist_ok=False); OUT.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2)); return report
if __name__=='__main__': run()
