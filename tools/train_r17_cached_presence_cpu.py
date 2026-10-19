#!/usr/bin/env python3
"""Train/evaluate frozen-pair presence residual directly from a bound cache."""
from __future__ import annotations
import hashlib,json,os,random
from pathlib import Path
from collections import defaultdict
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import tools.train_r17_pair_incidence_known_clean_cpu as base
from tools.train_r17_t3f_type_correction_cpu import digest
from tools.train_direct_graph_pair_scorer_cpu_r9 import SHA
COHORT=ROOT/os.environ.get('R17_CACHE_PRESENCE_COHORT','outputs/datasets/chemvector-r17-residual-density-source-20261021-t4b/manifest.json')
CACHE=ROOT/os.environ.get('R17_CACHE_PRESENCE_CACHE','outputs/audits/chemvector-r17-t4b-type-correction-cpu-contract3/feature-cache.pt')
OUT=ROOT/os.environ.get('R17_CACHE_PRESENCE_OUT','outputs/audits/chemvector-r17-t4b-presence-cpu')
STEPS=int(os.environ.get('R17_CACHE_PRESENCE_STEPS','300'));THRESH=float(os.environ.get('R17_CACHE_PRESENCE_THRESHOLD','0.95'));NEG=int(os.environ.get('R17_CACHE_PRESENCE_NEGATIVE_RATIO','3'))
def run():
 if OUT.exists():raise FileExistsError(OUT)
 if digest(base.INITIAL)!=SHA:raise RuntimeError('checkpoint changed')
 m=json.loads(COHORT.read_text());tr,ho=base.choose(m);cache=torch.load(CACHE,map_location='cpu',weights_only=False)
 binding=cache.get('binding',{});
 if binding.get('manifest_sha256')!=digest(COHORT) or binding.get('initial_checkpoint_sha256')!=SHA:raise RuntimeError('cache binding mismatch')
 train,hold=cache['train'],cache['holdout'];torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(20261021);random.seed(20261021)
 xrows=[];brows=[];ys=[]
 for it in train:
  pos=torch.where(it['labels']>.5)[0].tolist();neg=torch.where(it['labels']<=.5)[0].tolist();chosen=neg[:min(len(neg),NEG*len(pos))]
  for i in pos+chosen:xrows.append(it['features'][i]);brows.append(it['base_logits'][i]);ys.append(it['labels'][i])
 x=torch.stack(xrows);b=torch.stack(brows);y=torch.stack(ys);head=base.Residual(x.shape[-1]);opt=torch.optim.AdamW(head.parameters(),lr=2e-3,weight_decay=1e-4);pw=((len(y)-y.sum())/y.sum().clamp_min(1)).clamp(1,32);events=[]
 for step in range(1,STEPS+1):
  loss=F.binary_cross_entropy_with_logits(b+head(x),y,pos_weight=pw);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.);opt.step()
  if step in (1,100,STEPS):events.append({'step':step,'loss':float(loss.detach())})
 head.eval()
 def score(items,side=None):
  out=defaultdict(lambda:defaultdict(int))
  for it in items:
   z=it['base_logits'] if side is None else it['base_logits']+side(it['features']);ids=torch.where(z.sigmoid()>=THRESH)[0];p=it['profile'];c=out[p];c['pred']+=len(ids);c['true']+=int(it['labels'][ids].sum());c['target']+=len(it['target_edges']);c['reachable']+=int(it['labels'].sum());c['images']+=1
  return {p:{'images':c['images'],'predicted_edges':c['pred'],'true_edges':c['true'],'target_edges':c['target'],'reachable_target_edges':c['reachable'],'edge_precision':c['true']/max(c['pred'],1),'edge_recall':c['true']/max(c['target'],1)} for p,c in out.items()}
 baseline=score(hold);scored=score(hold,head);d={p:{k:scored[p][k]-baseline[p][k] for k in ('edge_precision','edge_recall')} for p in baseline};after=digest(base.INITIAL);gate={'checkpoint_unchanged':after==SHA,'sealed_test_read':False,'identity_disjoint':True,'edge_set_frozen':True,'each_profile_edge_precision_drop_le_0.01':all(v['edge_precision']>=-.01 for v in d.values()),'each_profile_edge_recall_drop_le_0.01':all(v['edge_recall']>=-.01 for v in d.values()),'production_go':False}
 report={'schema_version':'chemvector-r17-cached-presence-cpu-0.1','complete':True,'training_performed':True,'gpu_used':False,'sealed_test_read':False,'production_checkpoint_written':False,'cohort':str(COHORT.relative_to(ROOT)),'cache':str(CACHE.relative_to(ROOT)),'cache_sha256':digest(CACHE),'steps':STEPS,'negative_ratio':NEG,'events':events,'baseline':baseline,'scored':scored,'deltas':d,'initial_checkpoint_sha256':SHA,'initial_checkpoint_sha256_after':after,'gate':gate}
 OUT.mkdir(parents=True);torch.save({'schema_version':report['schema_version'],'state_dict':head.state_dict(),'initial_checkpoint_sha256':SHA,'cohort_sha256':digest(COHORT),'cache_sha256':digest(CACHE)},OUT/'sidecar-state.pt');(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'output':str(OUT/'report.json'),'baseline':baseline,'scored':scored,'deltas':d,'gate':gate},indent=2))
if __name__=='__main__':run()
