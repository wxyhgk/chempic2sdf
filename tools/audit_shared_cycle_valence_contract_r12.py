#!/usr/bin/env python3
"""Contract audit: shared cycle/order transform must be target-independent."""
from __future__ import annotations
from collections import Counter
import copy, hashlib, json, os, sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from tools.train_direct_graph_pair_scorer_cpu_r9 import DATA,SHA,_extract,rows,DirectPairResidual
from tools.train_direct_graph_type_residual_cpu_r12_corrected import TypeResidual
from tools.diagnose_shared_type_cycle_valence_r12 import transform
from tools.diagnose_aromatic_cycle_veto_r13 import type_stats
from tools.audit_direct_graph_pair_scorer_chemistry_r9 import one,finish
TRIAL=ROOT/"outputs/audits/chemvector-identity-aligned-direct-graph-pair-scorer-cpu-r10-coverage";TS=ROOT/"outputs/audits/chemvector-identity-aligned-direct-graph-type-residual-cpu-r12-corrected/type-state.pt";SPLIT=os.environ.get('AUDIT_SPLIT','fresh_holdout');OUT=ROOT/f"outputs/audits/chemvector-identity-aligned-direct-graph-shared-cycle-valence-contract-r12-{SPLIT}"
def digest(x):return hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest()
def run():
 ps=torch.load(TRIAL/'scorer-state.pt',map_location='cpu',weights_only=False);presence=DirectPairResidual(262);presence.load_state_dict(ps['state_dict']);presence.eval();ts=torch.load(TS,map_location='cpu',weights_only=False);typer=TypeResidual(262);typer.load_state_dict(ts['state_dict']);typer.eval();model,cfg,_=load_bundle_for_probe(INITIAL,SHA);model.eval().requires_grad_(False);hrs=sorted((r for r in rows() if r.get('split')==SPLIT),key=lambda x:x['sample_key']);items=[_extract(r,model,cfg,need_labels=True) for r in hrs]
 total=Counter();profiles={}; changed=0; edge_sets_equal=True; type_total=Counter()
 for item in items:
  q=transform(item,presence,typer); total.update(one(q,presence,.72)); type_total.update(type_stats(item,q,presence))
  altered=dict(item); altered['mapping']={};altered['target_edges']=set();altered['target_type_by_edge']={};q2=transform(altered,presence,typer);changed+=int(digest(q['chemical_logits'])!=digest(q2['chemical_logits']))
  with torch.inference_mode():
   a=torch.where((item['base_logits']+presence(item['features'])).sigmoid()>=.72)[0];b=torch.where((altered['base_logits']+presence(altered['features'])).sigmoid()>=.72)[0];edge_sets_equal &= bool(torch.equal(a,b))
 for p in sorted({i['row']['profile_id'] for i in items}):
  c=Counter();s=Counter()
  for item in items:
   if item['row']['profile_id']==p:
    q=transform(item,presence,typer);c.update(one(q,presence,.72));s.update(type_stats(item,q,presence))
  f=finish(c);f['type_accuracy']=s['type_correct']/max(s['type_rows'],1);f['true_edges']=s['true_edges'];f['target_edges']=s['target_edges'];profiles[p]=f
 after=hashlib.sha256(INITIAL.read_bytes()).hexdigest();result={'schema_version':'chemvector-identity-aligned-direct-graph-shared-cycle-valence-contract-r12-0.1','complete':True,'read_only':True,'training_performed':False,'gpu_used':False,'sealed_test_read':False,'production_checkpoint_written':False,'split':SPLIT,'images':len(items),'all':finish(total),'all_type_accuracy':type_total['type_correct']/max(type_total['type_rows'],1),'all_true_edges':type_total['true_edges'],'all_target_edges':type_total['target_edges'],'profiles':profiles,'target_field_perturbation_changed_logits':changed,'edge_set_equal_under_target_perturbation':edge_sets_equal,'initial_checkpoint_sha256':SHA,'initial_checkpoint_sha256_after':after,'gate':{'checkpoint_unchanged':after==SHA,'sealed_test_read':False,'target_independent_transform':changed==0 and edge_sets_equal,'edge_precision_ge_0.99':type_total['true_edges']/max(type_total['predicted_edges'],1)>=.99,'edge_recall_gt_0.95':type_total['true_edges']/max(type_total['target_edges'],1)>.95,'type_accuracy_ge_0.95':type_total['type_correct']/max(type_total['type_rows'],1)>=.95,'oracle_sanitize_ge_0.95_posthoc':finish(total)['oracle_sanitize_rate']>=.95,'all_profile_oracle_sanitize_ge_0.90_posthoc':all(v['oracle_sanitize_rate']>=.90 for v in profiles.values()),'production_go':False}}
 OUT.mkdir(parents=True,exist_ok=False);(OUT/'report.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n');print(json.dumps({'output':str(OUT/'report.json'),'gate':result['gate'],'all':result['all'],'profiles':profiles},indent=2))
if __name__=='__main__':run()
