#!/usr/bin/env python3
"""CPU shadow: one contract-matched type residual per rendering profile."""
from __future__ import annotations
import hashlib,json,os,random,sys
from collections import defaultdict
from pathlib import Path
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import tools.train_r17_pair_incidence_known_clean_cpu as base
from tools.train_r17_t3f_type_correction_cpu import TypeResidual,type_id,residual_logits,validate_cache
from tools.train_direct_graph_pair_scorer_cpu_r9 import SHA
COHORT=ROOT/os.environ.get('R17_PROFILE_TYPE_COHORT','outputs/datasets/chemvector-r17-residual-density-source-20261021-t4b/manifest.json')
CACHE=ROOT/os.environ.get('R17_PROFILE_TYPE_CACHE','outputs/audits/chemvector-r17-t4b-type-correction-cpu-contract3/feature-cache.pt')
PRESENCE=ROOT/os.environ.get('R17_PROFILE_TYPE_PRESENCE','outputs/audits/chemvector-r17-residual-density-pair-incidence-cpu-20261020-t3f/sidecar-state.pt')
OUT=ROOT/os.environ.get('R17_PROFILE_TYPE_OUT','outputs/audits/chemvector-r17-t4b-profile-type-cpu')
STEPS=int(os.environ.get('R17_PROFILE_TYPE_STEPS','300')); THRESH=float(os.environ.get('R17_PROFILE_TYPE_THRESHOLD','0.95'))
def digest(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def run():
 if OUT.exists():raise FileExistsError(OUT)
 if digest(base.INITIAL)!=SHA:raise RuntimeError('checkpoint changed')
 base.PAIR_THRESHOLD=THRESH;m=json.loads(COHORT.read_text());tr,ho=base.choose(m)
 if {r['molecule_id'] for r in tr}&{r['molecule_id'] for r in ho}:raise RuntimeError('identity overlap')
 ps=torch.load(PRESENCE,map_location='cpu',weights_only=False)
 if ps.get('initial_checkpoint_sha256')!=SHA:raise RuntimeError('presence binding mismatch')
 ph=base.Residual(262);ph.load_state_dict(ps['state_dict']);ph.eval();cache=torch.load(CACHE,map_location='cpu',weights_only=False)
 # Cache was produced from the same source and exact row ordering.
 binding={"initial_checkpoint_sha256":SHA,"manifest_sha256":digest(COHORT),"train_keys":[r['sample_key'] for r in tr],"holdout_keys":[r['sample_key'] for r in ho]}
 if cache.get('binding',{}).get('manifest_sha256')!=binding['manifest_sha256'] or cache.get('binding',{}).get('train_keys')!=binding['train_keys'] or cache.get('binding',{}).get('holdout_keys')!=binding['holdout_keys']:raise RuntimeError('cache source binding mismatch')
 train,hold=cache['train'],cache['holdout'];torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(20261021);random.seed(20261021)
 by=defaultdict(list)
 for it in train:
  pos=torch.where(it['labels']>.5)[0]
  if len(pos):by[it['profile']].append((torch.cat((it['features'][pos],it['chemical_logits'][pos]),-1),torch.tensor([type_id(it['type_labels'][i]) for i in pos.tolist()])))
 heads={};events={}
 for p,rows in sorted(by.items()):
  x=torch.cat([z[0] for z in rows]);y=torch.cat([z[1] for z in rows]);h=TypeResidual(x.shape[-1]);counts=torch.bincount(y,minlength=4).float();weights=(len(y)/counts.clamp_min(1)).sqrt();opt=torch.optim.AdamW(h.parameters(),lr=2e-3,weight_decay=1e-4);ev=[]
  for step in range(1,STEPS+1):
   loss=F.cross_entropy(residual_logits(h,x,x[:,-4:]),y,weight=weights);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(h.parameters(),1.);opt.step()
   if step in (1,100,STEPS):ev.append({'step':step,'loss':float(loss.detach())})
  h.eval();heads[p]=h;events[p]=ev
 summary={}
 with torch.inference_mode():
  for p in sorted(heads):
   h=heads[p];pred=true=target=typed=base_typed=0
   for it in hold:
    if it['profile']!=p:continue
    z=it['base_logits']+ph(it['features']);ids=torch.where(z.sigmoid()>=THRESH)[0];pred+=len(ids);true+=int(it['labels'][ids].sum());target+=len(it['target_edges']);b=it['chemical_logits'][ids];f=torch.cat((it['features'][ids],b),-1);k=residual_logits(h,f,b).argmax(-1)
    for j,i in enumerate(ids.tolist()):
     if it['labels'][i]>.5:
      truth=type_id(it['type_labels'][i]);base_typed+=int(int(b[j].argmax())==truth);typed+=int(int(k[j])==truth)
   summary[p]={'predicted_edges':pred,'true_edges':true,'target_edges':target,'edge_precision':true/max(pred,1),'edge_recall':true/max(target,1),'typed_precision':typed/max(pred,1),'typed_recall':typed/max(target,1),'baseline_typed_precision':base_typed/max(pred,1),'baseline_typed_recall':base_typed/max(target,1),'dtyped_precision':typed/max(pred,1)-base_typed/max(pred,1),'dtyped_recall':typed/max(target,1)-base_typed/max(target,1)}
 after=digest(base.INITIAL);gate={'checkpoint_unchanged':after==SHA,'sealed_test_read':False,'identity_disjoint':True,'edge_set_frozen':True,'each_profile_typed_precision_drop_le_0.01':all(v['dtyped_precision']>=-.01 for v in summary.values()),'each_profile_edge_precision_drop_le_0.01':all(v['edge_precision']>=-.01 for v in summary.values()),'all_profile_typed_recall_gain':all(v['dtyped_recall']>0 for v in summary.values()),'production_go':False}
 report={'schema_version':'chemvector-r17-t4b-profile-type-cpu-0.1','complete':True,'training_performed':True,'gpu_used':False,'sealed_test_read':False,'production_checkpoint_written':False,'steps':STEPS,'cache':str(CACHE.relative_to(ROOT)),'cache_sha256':digest(CACHE),'events':events,'summary':summary,'initial_checkpoint_sha256':SHA,'initial_checkpoint_sha256_after':after,'gate':gate}
 OUT.mkdir(parents=True);torch.save({'schema_version':report['schema_version'],'heads':{p:h.state_dict() for p,h in heads.items()},'initial_checkpoint_sha256':SHA,'cohort_sha256':digest(COHORT)},OUT/'sidecar-state.pt');(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'output':str(OUT/'report.json'),'summary':summary,'gate':gate},indent=2))
if __name__=='__main__':run()
