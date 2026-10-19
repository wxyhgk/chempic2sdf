#!/usr/bin/env python3
"""Bounded CPU probe for margin-gated independent bond-type correction."""
from __future__ import annotations
import hashlib,json,os,random,sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import tools.train_r17_pair_incidence_known_clean_cpu as base
from tools.train_direct_graph_pair_scorer_cpu_r9 import SHA
from chemvector.bond_object import BOND_TYPES
COHORT=ROOT/os.environ.get('R17_TRUST_COHORT','outputs/datasets/chemvector-r17-residual-density-source-20261020-t3f/manifest.json')
SOURCE=ROOT/os.environ.get('R17_TRUST_SOURCE','outputs/datasets/chemvector-r17-residual-density-source-20261020-t3f/samples')
PRESENCE=ROOT/os.environ.get('R17_TRUST_PRESENCE','outputs/audits/chemvector-r17-residual-density-pair-incidence-cpu-20261020-t3f/sidecar-state.pt')
OUT=ROOT/os.environ.get('R17_TRUST_OUT','outputs/audits/chemvector-r17-t3f-type-trust-cpu')
THRESH=float(os.environ.get('R17_TRUST_THRESHOLD','0.95')); STEPS=int(os.environ.get('R17_TRUST_STEPS','500'))
class Head(nn.Module):
 def __init__(self,d):
  super().__init__(); self.net=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,64),nn.GELU(),nn.Linear(64,4))
 def forward(self,x): return self.net(x)
def digest(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def run():
 if OUT.exists():raise FileExistsError(OUT)
 if digest(base.INITIAL)!=SHA:raise RuntimeError('checkpoint changed')
 base.SOURCE=SOURCE;base.PAIR_THRESHOLD=THRESH
 m=json.loads(COHORT.read_text());tr,ho=base.choose(m)
 if {r['molecule_id'] for r in tr}&{r['molecule_id'] for r in ho}:raise RuntimeError('identity overlap')
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(20261020);random.seed(20261020)
 model,cfg,_=base.load_bundle_for_probe(base.INITIAL,SHA);model.eval().requires_grad_(False)
 ps=torch.load(PRESENCE,map_location='cpu',weights_only=False)
 if ps.get('initial_checkpoint_sha256')!=SHA:raise RuntimeError('presence checkpoint binding mismatch')
 ph=base.Residual(262);ph.load_state_dict(ps['state_dict']);ph.eval()
 train=[base.extract(r,model,cfg) for r in tr];hold=[base.extract(r,model,cfg) for r in ho]
 xs=[];ys=[]
 for it in train:
  pos=torch.where(it['labels']>.5)[0]
  if len(pos):xs.append(it['features'][pos]);ys.append(torch.tensor([BOND_TYPES.index(it['type_labels'][i]) for i in pos.tolist()]))
 x=torch.cat(xs);y=torch.cat(ys);head=Head(x.shape[-1]); counts=torch.bincount(y,minlength=4).float();weights=(len(y)/counts.clamp_min(1)).sqrt();opt=torch.optim.AdamW(head.parameters(),lr=2e-3,weight_decay=1e-4);events=[]
 for step in range(1,STEPS+1):
  loss=F.cross_entropy(head(x),y,weight=weights);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.);opt.step()
  if step in (1,100,300,STEPS):events.append({'step':step,'loss':float(loss.detach())})
 head.eval(); margins=(0.0,0.25,0.5,0.75,1.0,1.5,2.0)
 summary={}
 with torch.inference_mode():
  for profile in ('oa13','oa33'):
   q=[it for it in hold if it['profile']==profile];summary[profile]={}
   for mrg in margins:
    pred=true=target=typed=base_typed=changed=0
    for it in q:
     z=it['base_logits']+ph(it['features']);ids=torch.where(z.sigmoid()>=THRESH)[0];pred+=len(ids);target+=int(it['target_edges'].__len__());true+=int(it['labels'][ids].sum())
     b=it['chemical_logits'][ids]; h=head(it['features'][ids]); top=b.topk(2,-1).values; use=(top[:,0]-top[:,1])<mrg; chosen=b.argmax(-1); chosen[use]=h[use].argmax(-1);changed+=int(use.sum())
     for j,idx in enumerate(ids.tolist()):
      if it['labels'][idx]>.5:
       truth=BOND_TYPES.index(it['type_labels'][idx]);base_typed+=int(int(b[j].argmax())==truth);typed+=int(int(chosen[j])==truth)
    summary[profile][str(mrg)]={'predicted_edges':pred,'true_edges':true,'target_edges':target,'edge_precision':true/max(pred,1),'edge_recall':true/max(target,1),'typed_precision':typed/max(pred,1),'typed_recall':typed/max(target,1),'baseline_typed_precision':base_typed/max(pred,1),'baseline_typed_recall':base_typed/max(target,1),'changed_edges':changed}
 after=digest(base.INITIAL);report={'schema_version':'chemvector-r17-t3f-type-trust-cpu-0.1','complete':True,'training_performed':True,'gpu_used':False,'sealed_test_read':False,'production_checkpoint_written':False,'steps':STEPS,'train_rows':len(y),'events':events,'summary':summary,'initial_checkpoint_sha256':SHA,'initial_checkpoint_sha256_after':after,'gate':{'checkpoint_unchanged':after==SHA,'sealed_test_read':False,'identity_disjoint':True,'production_go':False}}
 OUT.mkdir(parents=True);torch.save({'schema_version':report['schema_version'],'state_dict':head.state_dict(),'initial_checkpoint_sha256':SHA},OUT/'sidecar-state.pt');(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':run()
