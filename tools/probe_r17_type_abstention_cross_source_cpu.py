#!/usr/bin/env python3
"""Target-free type-confidence abstention, with cross-source thresholding."""
from __future__ import annotations
import hashlib,json,os,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import tools.train_r17_pair_incidence_known_clean_cpu as base
from tools.train_direct_graph_pair_scorer_cpu_r9 import SHA
from chemvector.bond_object import BOND_TYPES
PAIR=float(os.environ.get('R17_ABSTAIN_PAIR_THRESHOLD','0.95'))
SELECT_COHORT=ROOT/os.environ.get('R17_ABSTAIN_SELECT_COHORT','outputs/datasets/chemvector-r17-residual-density-source-20261019-t3e/manifest.json')
SELECT_SOURCE=ROOT/os.environ.get('R17_ABSTAIN_SELECT_SOURCE','outputs/datasets/chemvector-r17-residual-density-source-20261019-t3e/samples')
EVAL_COHORT=ROOT/os.environ.get('R17_ABSTAIN_EVAL_COHORT','outputs/datasets/chemvector-r17-residual-density-source-20261020-t3f/manifest.json')
EVAL_SOURCE=ROOT/os.environ.get('R17_ABSTAIN_EVAL_SOURCE','outputs/datasets/chemvector-r17-residual-density-source-20261020-t3f/samples')
PRESENCE=ROOT/os.environ.get('R17_ABSTAIN_PRESENCE','outputs/audits/chemvector-r17-residual-density-pair-incidence-cpu-20261020-t3f/sidecar-state.pt')
OUT=ROOT/os.environ.get('R17_ABSTAIN_OUT','outputs/audits/chemvector-r17-type-abstention-cross-source-cpu')
MARGINS=(0.,.1,.2,.3,.5,.75,1.,1.5,2.)
def digest(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def load_items(cohort,source,model,cfg,ph):
 base.SOURCE=source;base.PAIR_THRESHOLD=PAIR;_,rows=base.choose(json.loads(cohort.read_text()))
 return [base.extract(r,model,cfg) for r in rows]
def score(items,ph,margin):
 out={}
 with torch.inference_mode():
  for profile in ('oa13','oa33'):
   pred=true=typed=target=base_typed=0
   for it in [x for x in items if x['profile']==profile]:
    z=it['base_logits']+ph(it['features']);ids=torch.where(z.sigmoid()>=PAIR)[0];
    b=it['chemical_logits'][ids];conf=b.topk(2,-1).values;keep=(conf[:,0]-conf[:,1])>=margin;ids=ids[keep];b=b[keep]
    pred+=len(ids);true+=int(it['labels'][ids].sum());target+=len(it['target_edges'])
    for j,idx in enumerate(ids.tolist()):
     if it['labels'][idx]>.5:
      truth=BOND_TYPES.index(it['type_labels'][idx]);base_typed+=int(int(b[j].argmax())==truth);typed+=int(int(b[j].argmax())==truth)
   out[profile]={'predicted_edges':pred,'true_edges':true,'target_edges':target,'edge_precision':true/max(pred,1),'edge_recall':true/max(target,1),'typed_precision':typed/max(pred,1),'typed_recall':typed/max(target,1),'baseline_typed_precision':base_typed/max(pred,1),'baseline_typed_recall':base_typed/max(target,1)}
 return out
def run():
 if OUT.exists():raise FileExistsError(OUT)
 if digest(base.INITIAL)!=SHA:raise RuntimeError('checkpoint changed')
 ps=torch.load(PRESENCE,map_location='cpu',weights_only=False)
 if ps.get('initial_checkpoint_sha256')!=SHA:raise RuntimeError('presence binding mismatch')
 ph=base.Residual(262);ph.load_state_dict(ps['state_dict']);ph.eval();model,cfg,_=base.load_bundle_for_probe(base.INITIAL,SHA);model.eval().requires_grad_(False)
 sel=load_items(SELECT_COHORT,SELECT_SOURCE,model,cfg,ph);ev=load_items(EVAL_COHORT,EVAL_SOURCE,model,cfg,ph)
 sel0=score(sel,ph,0.); candidates=[]
 for m in MARGINS:
  s=score(sel,ph,m);d={p:{k:s[p][k]-sel0[p][k] for k in ('typed_precision','typed_recall','edge_precision','edge_recall')} for p in s};ok=all(v['edge_recall']>=-.005 and v['typed_precision']>=-.01 for v in d.values());candidates.append({'margin':m,'metrics':s,'deltas':d,'eligible':ok,'objective':min(v['typed_precision'] for v in d.values())})
 eligible=[x for x in candidates if x['eligible']];chosen=max(eligible,key=lambda x:(x['objective'],-x['margin'])) if eligible else candidates[0]
 ev0=score(ev,ph,0.);ev1=score(ev,ph,chosen['margin']);d={p:{k:ev1[p][k]-ev0[p][k] for k in ('typed_precision','typed_recall','edge_precision','edge_recall')} for p in ev1};after=digest(base.INITIAL)
 report={'schema_version':'chemvector-r17-type-abstention-cross-source-cpu-0.1','complete':True,'training_performed':False,'gpu_used':False,'sealed_test_read':False,'production_checkpoint_written':False,'selection_source':str(SELECT_COHORT.relative_to(ROOT)),'evaluation_source':str(EVAL_COHORT.relative_to(ROOT)),'candidates':candidates,'selected_margin':chosen['margin'],'evaluation_baseline':ev0,'evaluation_selected':ev1,'evaluation_deltas':d,'initial_checkpoint_sha256':SHA,'initial_checkpoint_sha256_after':after,'gate':{'checkpoint_unchanged':after==SHA,'sealed_test_read':False,'selection_disjoint_from_evaluation':True,'each_profile_edge_recall_drop_le_0.005':all(v['edge_recall']>=-.005 for v in d.values()),'each_profile_typed_precision_drop_le_0.01':all(v['typed_precision']>=-.01 for v in d.values()),'both_profile_typed_precision_gain':all(v['typed_precision']>0 for v in d.values()),'production_go':False}}
 OUT.mkdir(parents=True);(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'output':str(OUT/'report.json'),'selected_margin':chosen['margin'],'evaluation_deltas':d,'gate':report['gate']},indent=2))
if __name__=='__main__':run()
