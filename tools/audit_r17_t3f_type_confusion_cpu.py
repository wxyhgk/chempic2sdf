#!/usr/bin/env python3
"""CPU/read-only confusion audit for t3f selected true edges."""
from __future__ import annotations
import collections,hashlib,json,os,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import tools.train_r17_pair_incidence_known_clean_cpu as base
from tools.train_direct_graph_pair_scorer_cpu_r9 import SHA
from chemvector.bond_object import BOND_TYPES
COHORT=ROOT/os.environ.get('R17_CONF_COHORT','outputs/datasets/chemvector-r17-residual-density-source-20261020-t3f/manifest.json')
SOURCE=ROOT/os.environ.get('R17_CONF_SOURCE','outputs/datasets/chemvector-r17-residual-density-source-20261020-t3f/samples')
PRESENCE=ROOT/os.environ.get('R17_CONF_PRESENCE','outputs/audits/chemvector-r17-residual-density-pair-incidence-cpu-20261020-t3f/sidecar-state.pt')
OUT=ROOT/os.environ.get('R17_CONF_OUT','outputs/audits/chemvector-r17-t3f-type-confusion-cpu')
THRESH=float(os.environ.get('R17_CONF_THRESHOLD','0.95'))
SPLIT=os.environ.get('R17_CONF_SPLIT','holdout')
def digest(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def run():
 if OUT.exists():raise FileExistsError(OUT)
 if digest(base.INITIAL)!=SHA:raise RuntimeError('checkpoint changed')
 base.SOURCE=SOURCE;base.PAIR_THRESHOLD=THRESH;m=json.loads(COHORT.read_text());train_rows,hold_rows=base.choose(m);rows=train_rows if SPLIT=='train' else hold_rows
 model,cfg,_=base.load_bundle_for_probe(base.INITIAL,SHA);model.eval().requires_grad_(False)
 st=torch.load(PRESENCE,map_location='cpu',weights_only=False)
 if st.get('initial_checkpoint_sha256')!=SHA:raise RuntimeError('sidecar binding mismatch')
 ph=base.Residual(262);ph.load_state_dict(st['state_dict']);ph.eval();cm=collections.Counter(); byprof=collections.defaultdict(collections.Counter); added=collections.Counter(); margins=collections.defaultdict(list)
 with torch.inference_mode():
  for row in rows:
   it=base.extract(row,model,cfg);z=it['base_logits'];zr=z+ph(it['features']);base_ids=set(torch.where(z.sigmoid()>=THRESH)[0].tolist());ids=torch.where(zr.sigmoid()>=THRESH)[0]
   for idx in ids.tolist():
    if it['labels'][idx]<=.5:continue
    truth=str(it['type_labels'][idx]);pred=BOND_TYPES[int(it['chemical_logits'][idx].argmax())];pair=(truth,pred);cm[pair]+=1;byprof[it['profile']][pair]+=1
    if idx not in base_ids:added[pair]+=1
    vals=it['chemical_logits'][idx].topk(2).values;margins[pair].append(float(vals[0]-vals[1]))
 def serial(c):return {f'{a}->{b}':{'count':n,'mean_margin':sum(margins[(a,b)])/max(len(margins[(a,b)]),1),'added_count':added[(a,b)]} for (a,b),n in c.most_common()}
 after=digest(base.INITIAL);report={'schema_version':'chemvector-r17-t3f-type-confusion-cpu-0.1','complete':True,'training_performed':False,'gpu_used':False,'sealed_test_read':False,'production_checkpoint_written':False,'cohort':str(COHORT.relative_to(ROOT)),'split':SPLIT,'threshold':THRESH,'confusion':serial(cm),'confusion_by_profile':{p:serial(c) for p,c in byprof.items()},'selected_true_edges':sum(cm.values()),'new_presence_true_edges':sum(added.values()),'initial_checkpoint_sha256':SHA,'initial_checkpoint_sha256_after':after,'gate':{'checkpoint_unchanged':after==SHA,'sealed_test_read':False,'identity_disjoint':True,'production_go':False}}
 OUT.mkdir(parents=True);(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'output':str(OUT/'report.json'),'confusion':report['confusion'],'new_presence_true_edges':sum(added.values()),'gate':report['gate']},indent=2))
if __name__=='__main__':run()
