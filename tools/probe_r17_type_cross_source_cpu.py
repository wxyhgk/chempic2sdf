#!/usr/bin/env python3
"""Cross-source CPU replay of a frozen t3f type sidecar (no fitting)."""
from __future__ import annotations
import hashlib, json, os, sys
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import tools.train_r17_pair_incidence_known_clean_cpu as base
from tools.train_r17_t3f_type_correction_cpu import TypeResidual, type_id
from tools.train_direct_graph_pair_scorer_cpu_r9 import SHA
from chemvector.bond_object import BOND_TYPES
COHORT=ROOT/os.environ.get('R17_CROSS_COHORT','outputs/datasets/chemvector-r17-residual-density-source-20261019-t3e/manifest.json')
SOURCE=ROOT/os.environ.get('R17_CROSS_SOURCE','outputs/datasets/chemvector-r17-residual-density-source-20261019-t3e/samples')
TYPE=ROOT/os.environ.get('R17_CROSS_TYPE','outputs/audits/chemvector-r17-t3f-type-correction-cpu/sidecar-state.pt')
PRESENCE=ROOT/os.environ.get('R17_CROSS_PRESENCE','outputs/audits/chemvector-r17-residual-density-pair-incidence-cpu-20261020-t3f/sidecar-state.pt')
OUT=ROOT/os.environ.get('R17_CROSS_OUT','outputs/audits/chemvector-r17-type-cross-source-t3e-cpu')
THRESH=float(os.environ.get('R17_CROSS_THRESHOLD','0.95'))
def digest(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def run():
 if OUT.exists(): raise FileExistsError(OUT)
 if digest(base.INITIAL)!=SHA: raise RuntimeError('checkpoint changed')
 base.SOURCE=SOURCE; base.PAIR_THRESHOLD=THRESH
 m=json.loads(COHORT.read_text()); _, hold=base.choose(m)
 model,cfg,_=base.load_bundle_for_probe(base.INITIAL,SHA); model.eval().requires_grad_(False)
 ps=torch.load(PRESENCE,map_location='cpu',weights_only=False); ph=base.Residual(262); ph.load_state_dict(ps['state_dict']); ph.eval()
 ts=torch.load(TYPE,map_location='cpu',weights_only=False); th=TypeResidual(266); th.load_state_dict(ts['state_dict']); th.eval()
 items=[base.extract(r,model,cfg) for r in hold]
 summary={}
 with torch.inference_mode():
  for profile in ('oa13','oa33'):
   q=[x for x in items if x['profile']==profile]; vals=[]
   for corrected in (False,True):
    pred=true=typed=target=0
    for it in q:
     z=it['base_logits']+ph(it['features']); ids=torch.where(z.sigmoid()>=THRESH)[0]
     pred+=len(ids); target+=int(it['labels'].sum()); true+=int(it['labels'][ids].sum())
     for j in ids.tolist():
      if it['labels'][j]>.5:
       basek=int(it['chemical_logits'][j].argmax()); k=basek
       if corrected:
        f=torch.cat((it['features'][j],it['chemical_logits'][j])).unsqueeze(0); k=int((it['chemical_logits'][j]+th(f)[0]).argmax())
       typed+=int(BOND_TYPES[k]==it['type_labels'][j])
    vals.append({'predicted_edges':pred,'true_edges':true,'target_edges':target,'edge_precision':true/max(pred,1),'edge_recall':true/max(target,1),'typed_precision':typed/max(pred,1),'typed_recall':typed/max(target,1)})
   summary[profile]={'baseline':vals[0],'corrected':vals[1],'deltas':{k:vals[1][k]-vals[0][k] for k in ('edge_precision','edge_recall','typed_precision','typed_recall')}}
 after=digest(base.INITIAL); report={'schema_version':'chemvector-r17-type-cross-source-cpu-0.1','complete':True,'training_performed':False,'gpu_used':False,'sealed_test_read':False,'production_checkpoint_written':False,'cohort':str(COHORT.relative_to(ROOT)),'source':str(SOURCE.relative_to(ROOT)),'summary':summary,'initial_checkpoint_sha256':SHA,'initial_checkpoint_sha256_after':after,'gate':{'checkpoint_unchanged':after==SHA,'sealed_test_read':False,'identity_source_disjoint':True,'production_go':False}}
 OUT.mkdir(parents=True); (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
if __name__=='__main__': run()
