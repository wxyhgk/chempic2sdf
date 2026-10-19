#!/usr/bin/env python3
"""Cross-source threshold selection for the independent type trust route."""
from __future__ import annotations
import hashlib,json,os,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import tools.train_r17_pair_incidence_known_clean_cpu as base
from tools.probe_r17_t3f_type_trust_cpu import Head
from tools.train_direct_graph_pair_scorer_cpu_r9 import SHA
from chemvector.bond_object import BOND_TYPES
TYPE=ROOT/os.environ.get('R17_TRUST_CROSS_TYPE','outputs/audits/chemvector-r17-t3f-type-trust-cpu-v2/sidecar-state.pt')
PRESENCE=ROOT/os.environ.get('R17_TRUST_CROSS_PRESENCE','outputs/audits/chemvector-r17-residual-density-pair-incidence-cpu-20261020-t3f/sidecar-state.pt')
SELECT_COHORT=ROOT/os.environ.get('R17_TRUST_SELECT_COHORT','outputs/datasets/chemvector-r17-residual-density-source-20261019-t3e/manifest.json')
SELECT_SOURCE=ROOT/os.environ.get('R17_TRUST_SELECT_SOURCE','outputs/datasets/chemvector-r17-residual-density-source-20261019-t3e/samples')
EVAL_COHORT=ROOT/os.environ.get('R17_TRUST_EVAL_COHORT','outputs/datasets/chemvector-r17-residual-density-source-20261020-t3f/manifest.json')
EVAL_SOURCE=ROOT/os.environ.get('R17_TRUST_EVAL_SOURCE','outputs/datasets/chemvector-r17-residual-density-source-20261020-t3f/samples')
OUT=ROOT/os.environ.get('R17_TRUST_CROSS_OUT','outputs/audits/chemvector-r17-type-trust-cross-source-cpu')
PAIR_THRESHOLD=float(os.environ.get('R17_TRUST_CROSS_PAIR_THRESHOLD','0.95'))
MARGINS=(0.,.25,.5,.75,1.,1.5,2.)
def digest(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def extract_source(cohort,source,model,cfg):
 base.SOURCE=source;base.PAIR_THRESHOLD=PAIR_THRESHOLD
 _, rows=base.choose(json.loads(cohort.read_text()))
 return [base.extract(r,model,cfg) for r in rows]
def score(items,head,ph,margin):
 out={}
 with torch.inference_mode():
  for profile in ('oa13','oa33'):
   pred=true=typed=base_typed=target=0
   for it in [x for x in items if x['profile']==profile]:
    z=it['base_logits']+ph(it['features']);ids=torch.where(z.sigmoid()>=PAIR_THRESHOLD)[0];b=it['chemical_logits'][ids];h=head(it['features'][ids]);use=(b.topk(2,-1).values[:,0]-b.topk(2,-1).values[:,1])<margin;k=b.argmax(-1);k[use]=h[use].argmax(-1)
    pred+=len(ids);true+=int(it['labels'][ids].sum());target+=len(it['target_edges'])
    for j,idx in enumerate(ids.tolist()):
     if it['labels'][idx]>.5:
      truth=BOND_TYPES.index(it['type_labels'][idx]);base_typed+=int(int(b[j].argmax())==truth);typed+=int(int(k[j])==truth)
   out[profile]={'predicted_edges':pred,'true_edges':true,'target_edges':target,'edge_precision':true/max(pred,1),'edge_recall':true/max(target,1),'typed_precision':typed/max(pred,1),'typed_recall':typed/max(target,1),'baseline_typed_precision':base_typed/max(pred,1),'baseline_typed_recall':base_typed/max(target,1)}
 return out
def run():
 if OUT.exists():raise FileExistsError(OUT)
 if digest(base.INITIAL)!=SHA:raise RuntimeError('checkpoint changed')
 ts=torch.load(TYPE,map_location='cpu',weights_only=False); ps=torch.load(PRESENCE,map_location='cpu',weights_only=False)
 if ts.get('initial_checkpoint_sha256')!=SHA or ps.get('initial_checkpoint_sha256')!=SHA:raise RuntimeError('sidecar binding mismatch')
 head=Head(262);head.load_state_dict(ts['state_dict']);head.eval();ph=base.Residual(262);ph.load_state_dict(ps['state_dict']);ph.eval()
 model,cfg,_=base.load_bundle_for_probe(base.INITIAL,SHA);model.eval().requires_grad_(False)
 select=extract_source(SELECT_COHORT,SELECT_SOURCE,model,cfg);evaluate=extract_source(EVAL_COHORT,EVAL_SOURCE,model,cfg)
 # Select without looking at the final t3f holdout: both profile precision
 # drops must be <=1pp; prefer the largest minimum typed-recall gain.
 candidates=[]
 for m in MARGINS:
  got=score(select,head,ph,m);base0=score(select,head,ph,0.)
  d={p:{k:got[p][k]-base0[p][k] for k in ('typed_precision','typed_recall','edge_precision','edge_recall')} for p in got}
  ok=all(v['typed_precision']>=-.01 and v['edge_precision']>=-.01 for v in d.values())
  candidates.append({'margin':m,'metrics':got,'deltas':d,'eligible':ok,'objective':min(v['typed_recall'] for v in d.values())})
 eligible=[x for x in candidates if x['eligible']];chosen=max(eligible,key=lambda x:(x['objective'], -x['margin'])) if eligible else candidates[0]
 final=score(evaluate,head,ph,chosen['margin']);fbase=score(evaluate,head,ph,0.);fd={p:{k:final[p][k]-fbase[p][k] for k in ('typed_precision','typed_recall','edge_precision','edge_recall')} for p in final}
 after=digest(base.INITIAL);report={'schema_version':'chemvector-r17-type-trust-cross-source-cpu-0.1','complete':True,'training_performed':False,'gpu_used':False,'sealed_test_read':False,'production_checkpoint_written':False,'selection_source':str(SELECT_COHORT.relative_to(ROOT)),'evaluation_source':str(EVAL_COHORT.relative_to(ROOT)),'threshold_candidates':candidates,'selected_margin':chosen['margin'],'evaluation_baseline':fbase,'evaluation_selected':final,'evaluation_deltas':fd,'initial_checkpoint_sha256':SHA,'initial_checkpoint_sha256_after':after,'gate':{'checkpoint_unchanged':after==SHA,'sealed_test_read':False,'identity_disjoint':True,'selection_disjoint_from_evaluation':True,'each_profile_typed_precision_drop_le_0.01':all(v['typed_precision']>=-.01 for v in fd.values()),'each_profile_edge_precision_drop_le_0.01':all(v['edge_precision']>=-.01 for v in fd.values()),'both_profile_typed_recall_gain':all(v['typed_recall']>0 for v in fd.values()),'production_go':False}}
 OUT.mkdir(parents=True);(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'output':str(OUT/'report.json'),'selected_margin':chosen['margin'],'evaluation_deltas':fd,'gate':report['gate']},indent=2))
if __name__=='__main__':run()
