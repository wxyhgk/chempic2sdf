#!/usr/bin/env python3
"""Post-hoc scale sweep for r12 type residual; not a promotion gate."""
from __future__ import annotations
from collections import Counter
import hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from tools.train_direct_graph_pair_scorer_cpu_r9 import DATA,SHA,_extract,rows,DirectPairResidual
from tools.train_direct_graph_type_residual_cpu_r12_corrected import TypeResidual
from tools.audit_direct_graph_pair_scorer_chemistry_r9 import one,finish
TRIAL=ROOT/"outputs/audits/chemvector-identity-aligned-direct-graph-pair-scorer-cpu-r10-coverage"; TS=ROOT/"outputs/audits/chemvector-identity-aligned-direct-graph-type-residual-cpu-r12-corrected/type-state.pt"; OUT=ROOT/"outputs/audits/chemvector-identity-aligned-direct-graph-type-residual-scale-r12"
def dig(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def run():
 if dig(INITIAL)!=SHA:raise RuntimeError('production checkpoint changed')
 ps=torch.load(TRIAL/'scorer-state.pt',map_location='cpu',weights_only=False); presence=DirectPairResidual(262);presence.load_state_dict(ps['state_dict']);presence.eval()
 ts=torch.load(TS,map_location='cpu',weights_only=False); typer=TypeResidual(262);typer.load_state_dict(ts['state_dict']);typer.eval()
 model,cfg,_=load_bundle_for_probe(INITIAL,SHA);model.eval().requires_grad_(False); hrs=sorted((r for r in rows() if r.get('split')=='fresh_holdout'),key=lambda x:x['sample_key']); items=[_extract(r,model,cfg,need_labels=True) for r in hrs]
 result={}
 for scale in (0.0,.05,.1,.2,.3,.4,.5,.7,1.0):
  c=Counter()
  for item in items:
   q=dict(item);q['chemical_logits']=item['chemical_logits']+scale*typer(item['features']);c.update(one(q,presence,.72))
  result[str(scale)]=finish(c)
 out={'schema_version':'chemvector-identity-aligned-direct-graph-type-residual-scale-r12-0.1','complete':True,'read_only':True,'training_performed':False,'gpu_used':False,'sealed_test_read':False,'production_checkpoint_written':False,'split':'fresh_holdout','scales':result,'selection':'post-hoc diagnostic only; no scale selected for promotion','initial_checkpoint_sha256':SHA,'initial_checkpoint_sha256_after':dig(INITIAL),'gate':{'checkpoint_unchanged':dig(INITIAL)==SHA,'sealed_test_read':False,'production_go':False}}
 OUT.mkdir(parents=True,exist_ok=False);(OUT/'report.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n');print(json.dumps({'output':str(OUT/'report.json'),'scales':{k:{'sanitize':v['oracle_sanitize_rate'],'aromatic':v['oracle_aromatic_errors'],'valence':v['oracle_valence_errors']} for k,v in result.items()}},indent=2))
if __name__=='__main__':run()
