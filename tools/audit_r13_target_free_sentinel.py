#!/usr/bin/env python3
"""Runtime sentinel that target fields cannot affect r13 veto decisions."""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from tools.train_direct_graph_pair_scorer_cpu_r9 import SHA,rows,_extract,DirectPairResidual
from tools.train_direct_graph_type_profile_residual_cpu_r13 import TypeResidual
from tools.diagnose_aromatic_cycle_veto_r13 import veto_item
PRESENCE=ROOT/'outputs/audits/chemvector-identity-aligned-direct-graph-pair-scorer-cpu-r10-coverage/scorer-state.pt'
TYPE=ROOT/'outputs/audits/chemvector-identity-aligned-direct-graph-type-profile-residual-cpu-r13/type-state.pt'
OUT=ROOT/'outputs/audits/chemvector-identity-aligned-direct-graph-target-free-sentinel-r13'
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 if digest(INITIAL)!=SHA: raise RuntimeError('production checkpoint changed')
 model,cfg,_=load_bundle_for_probe(INITIAL,SHA);model.eval().requires_grad_(False)
 hold=sorted((r for r in rows() if r.get('split')=='fresh_holdout'),key=lambda r:r['sample_key'])
 items=[_extract(r,model,cfg,need_labels=True) for r in hold]
 ps=torch.load(PRESENCE,map_location='cpu',weights_only=False);p=DirectPairResidual(262);p.load_state_dict(ps['state_dict']);p.eval()
 ts=torch.load(TYPE,map_location='cpu',weights_only=False);h={x:TypeResidual(262) for x in ts['heads']}
 for x in h:h[x].load_state_dict(ts['heads'][x]);h[x].eval()
 mismatches=[]
 for n,item in enumerate(items):
  altered=dict(item); altered['mapping']={}; altered['target_edges']={(999,1000)}; altered['target_type_by_edge']={(999,1000):'TRIPLE'}; altered['target_atoms']=0
  for repair in (False,True):
   q1,c1=veto_item(item,p,h,repair_valence=repair);q2,c2=veto_item(altered,p,h,repair_valence=repair)
   if c1!=c2 or not torch.equal(q1['chemical_logits'].argmax(-1),q2['chemical_logits'].argmax(-1)):
    mismatches.append({'sample_key':item['row']['sample_key'],'repair':repair,'changed_original':c1,'changed_altered':c2})
 result={'schema_version':'chemvector-identity-aligned-direct-graph-target-free-sentinel-r13-0.1','complete':True,'read_only':True,'training_performed':False,'gpu_used':False,'sealed_test_read':False,'production_checkpoint_written':False,'images':len(items),'mismatches':mismatches,'initial_checkpoint_sha256':SHA,'initial_checkpoint_sha256_after':digest(INITIAL),'gate':{'checkpoint_unchanged':digest(INITIAL)==SHA,'sealed_test_read':False,'target_free_veto_unchanged':not mismatches,'production_go':False}}
 OUT.mkdir(parents=True,exist_ok=False);(OUT/'report.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n');print(json.dumps({'output':str(OUT/'report.json'),'images':len(items),'mismatches':len(mismatches),'gate':result['gate']},indent=2,ensure_ascii=False))
if __name__=='__main__':main()
