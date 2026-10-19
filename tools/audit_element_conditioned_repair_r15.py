#!/usr/bin/env python3
"""Holdout audit for predicted-element-conditioned bond-order repair."""
from __future__ import annotations
from collections import Counter
import hashlib,json,os,sys
from pathlib import Path
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from tools.train_direct_graph_pair_scorer_cpu_r9 import DATA,SHA,_extract,rows,DirectPairResidual,_target,_map_atoms
from tools.train_direct_graph_type_residual_cpu_r12_corrected import TypeResidual
from tools.probe_atom_element_head_cpu_r15 import ELEMENTS,extract
from tools.audit_direct_graph_pair_scorer_chemistry_r9 import one,finish
from tools.diagnose_aromatic_cycle_veto_r13 import type_stats
from tools.diagnose_shared_type_cycle_valence_r12 import transform as cycle_valence_transform
TRIAL=ROOT/"outputs/audits/chemvector-identity-aligned-direct-graph-pair-scorer-cpu-r10-coverage";TS=ROOT/"outputs/audits/chemvector-identity-aligned-direct-graph-type-residual-cpu-r12-corrected/type-state.pt";ES=ROOT/"outputs/audits/chemvector-atom-element-head-cpu-r15/head-state.pt";SPLIT=os.environ.get('AUDIT_SPLIT','fresh_holdout');OUT=ROOT/f"outputs/audits/chemvector-element-conditioned-repair-r15-{SPLIT}-v2"
SCALE=.5;THRESHOLD=.72
def dig(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def cycle_only(item,presence,typer):return cycle_valence_transform(item,presence,typer)
def element_repair(item,presence,typer,element_head,model,cfg):
 q=cycle_only(item,presence,typer);base=DATA/'samples'/item['row']['sample_key'];
 # Predicted element labels are generated from frozen atom features; target
 # mapping is used below only for audit metrics, never for this repair.
 with torch.inference_mode():
  out=__import__('chemvector.training.curriculum.continuation',fromlist=['structure_forward']).structure_forward(model,__import__('chemvector.training.curriculum.pixels',fromlist=['image_tensor']).image_tensor(Image.open(base.with_suffix('.png')).convert('RGB'))[None],cfg,-.25)
 sel=out[6];slots=torch.where(sel['atom_mask'][0])[0];pred_elements=element_head(sel['atom_features'][0,slots].float()).argmax(-1).tolist();pairs=item['pairs'];st={int(x):i for i,x in enumerate(slots.tolist())};logits=q['chemical_logits'].clone();edges=[]
 for i in torch.where((item['base_logits']+presence(item['features'])).sigmoid()>=THRESHOLD)[0].tolist():
  l,r=(int(v) for v in pairs[i].tolist());
  if l in st and r in st and st[l]!=st[r]:edges.append((st[l],st[r],i))
 caps={'C':4.,'N':3.,'O':2.,'S':6.,'P':5.,'F':1.,'Cl':1.,'Br':1.,'I':1.,'B':3.,'Si':4.};order={0:1.,1:2.,2:3.,3:1.5}
 for _ in range(32):
  kinds={i:int(logits[i].argmax()) for _,_,i in edges};tot={}
  for a,b,i in edges:
   z=order[kinds[i]];tot[a]=tot.get(a,0)+z;tot[b]=tot.get(b,0)+z
  bad=[u for u,v in tot.items() if v>caps.get(ELEMENTS[pred_elements[u]],4.)+1e-6]
  if not bad:break
  cs=[(i,kinds[i],float(logits[i].max()-logits[i].topk(2).values[-1])) for a,b,i in edges if any(u in (a,b) for u in bad) and kinds[i]!=0]
  if not cs:break
  i,_,_=min(cs,key=lambda z:(-order[z[1]],z[2],z[0]));logits[i,0]=logits[i].max()+1;logits[i,1:]-=1e4
 q['chemical_logits']=logits;q['_predicted_elements']=pred_elements;return q,pred_elements
def run():
 if dig(INITIAL)!=SHA:raise RuntimeError('production checkpoint changed')
 ps=torch.load(TRIAL/'scorer-state.pt',map_location='cpu',weights_only=False);presence=DirectPairResidual(262);presence.load_state_dict(ps['state_dict']);presence.eval();ts=torch.load(TS,map_location='cpu',weights_only=False);typer=TypeResidual(262);typer.load_state_dict(ts['state_dict']);typer.eval();es=torch.load(ES,map_location='cpu',weights_only=False);element_head=torch.nn.Sequential(torch.nn.LayerNorm(128),torch.nn.Linear(128,len(ELEMENTS)));element_head.load_state_dict(es['state_dict']);element_head.eval();model,cfg,_=load_bundle_for_probe(INITIAL,SHA);model.eval().requires_grad_(False);hrs=sorted((r for r in rows() if r.get('split')==SPLIT),key=lambda x:x['sample_key']);items=[_extract(r,model,cfg,need_labels=True) for r in hrs];plain=Counter();repaired=Counter();type_total=Counter();profile_repaired={};element_correct=element_total=0
 for item in items:
  q=cycle_only(item,presence,typer);plain.update(one(q,presence,THRESHOLD));r,pred=element_repair(item,presence,typer,element_head,model,cfg);repaired.update(one(r,presence,THRESHOLD));type_total.update(type_stats(item,r,presence));
  base=DATA/'samples'/item['row']['sample_key'];layers=json.loads(base.with_suffix('.layers.json').read_text());atoms=sorted(layers['atoms'],key=lambda a:int(a['atom_id']));targets,_,_=_target(item['row']['sample_key']);mapping=_map_atoms(item['points'],targets)
  for c,t in mapping.items():
   if c<len(pred):element_total+=1;element_correct+=int(ELEMENTS[pred[c]]==str(atoms[t].get('element') or 'C'))
 for p in sorted({i['row']['profile_id'] for i in items}):
  c=Counter();s=Counter()
  for item in items:
   if item['row']['profile_id']==p:
    r,_=element_repair(item,presence,typer,element_head,model,cfg);c.update(one(r,presence,THRESHOLD));s.update(type_stats(item,r,presence))
  f=finish(c);f['type_accuracy']=s['type_correct']/max(s['type_rows'],1);f['true_edges']=s['true_edges'];f['target_edges']=s['target_edges'];profile_repaired[p]=f
 after=dig(INITIAL);result={'schema_version':'chemvector-element-conditioned-repair-r15-0.1','complete':True,'read_only':True,'training_performed':False,'gpu_used':False,'sealed_test_read':False,'production_checkpoint_written':False,'split':SPLIT,'images':len(items),'profile_metadata_used':False,'predicted_element_accuracy':element_correct/max(element_total,1),'element_rows':element_total,'plain_cycle_valence':finish(plain),'element_conditioned_repair':finish(repaired),'element_conditioned_type_accuracy':type_total['type_correct']/max(type_total['type_rows'],1),'element_conditioned_profiles':profile_repaired,'initial_checkpoint_sha256':SHA,'initial_checkpoint_sha256_after':after,'gate':{'checkpoint_unchanged':after==SHA,'sealed_test_read':False,'predicted_element_accuracy_ge_0.90':element_correct/max(element_total,1)>=.90,'edge_precision_ge_0.99':type_total['true_edges']/max(type_total['predicted_edges'],1)>=.99,'edge_recall_gt_0.95':type_total['true_edges']/max(type_total['target_edges'],1)>.95,'type_accuracy_ge_0.95':type_total['type_correct']/max(type_total['type_rows'],1)>=.95,'oracle_sanitize_ge_0.95_posthoc':finish(repaired)['oracle_sanitize_rate']>=.95,'all_profile_oracle_sanitize_ge_0.90_posthoc':all(v['oracle_sanitize_rate']>=.90 for v in profile_repaired.values()),'production_go':False}}
 OUT.mkdir(parents=True,exist_ok=False);(OUT/'report.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n');print(json.dumps({'output':str(OUT/'report.json'),'element_accuracy':result['predicted_element_accuracy'],'plain':result['plain_cycle_valence'],'repaired':result['element_conditioned_repair'],'gate':result['gate']},indent=2))
if __name__=='__main__':run()
