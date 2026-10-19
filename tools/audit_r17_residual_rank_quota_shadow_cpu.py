#!/usr/bin/env python3
"""CPU-only residual rank+quota shadow on the frozen t2 pair cache.

The baseline pair set and production checkpoint are immutable.  A balanced
linear probe is fit on train-only pair labels; for each profile, a tiny
train-only quota sweep chooses how many below-threshold residual candidates to
add per image.  Holdout is then evaluated independently.  This is diagnostic
only and never writes a production checkpoint.
"""
from __future__ import annotations
import hashlib, json, math, os
from collections import defaultdict
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/"outputs/audits/chemvector-r17-targeted-source-pair-calibration-cpu-20261018-t2/features-cache.pt"
OUT=ROOT/"outputs/audits/chemvector-r17-residual-rank-quota-shadow-cpu-20261018-t2"
INITIAL=ROOT/"chemvector/training/curriculum/initial.pt"
INITIAL_SHA="09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
THRESH=0.72; LOGIT=math.log(THRESH/(1-THRESH)); QUOTAS=(0,1,2,3,4)

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
class Probe(nn.Module):
    def __init__(self,d):
        super().__init__(); self.out=nn.Linear(d,1); nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
    def forward(self,x): return self.out(x).squeeze(-1)
def fit(train):
    xs=[]; ys=[]
    for it in train:
        y=(it['labels']>.5); pos=torch.where(y)[0]; neg=torch.where(~y)[0]
        if not len(pos) or not len(neg): continue
        n=min(len(pos),len(neg)); hard=neg[torch.argsort(it['base_logits'][neg],descending=True)[:n]]
        xs.append(it['features'][torch.cat((pos,hard))][:,:256]); ys.append(y[torch.cat((pos,hard))].float())
    x=torch.cat(xs); y=torch.cat(ys); mean=x.mean(0); scale=x.std(0,unbiased=False).clamp_min(1e-5)
    p=Probe(x.shape[1]); opt=torch.optim.AdamW(p.parameters(),lr=2e-3,weight_decay=1e-4); pw=((len(y)-y.sum())/y.sum().clamp_min(1)).clamp(1,32)
    for _ in range(300):
        loss=F.binary_cross_entropy_with_logits(p((x-mean)/scale),y,pos_weight=pw); opt.zero_grad(); loss.backward(); opt.step()
    p.eval(); return p,mean,scale
def evaluate(items,probe,mean,scale,quotas):
    total=defaultdict(int); by=defaultdict(lambda:defaultdict(int))
    for it in items:
        score=probe((it['features'][:,:256]-mean)/scale).detach(); base=it['base_logits']; baseline=base>=LOGIT; residual=~baseline
        q=int(quotas.get(it['profile'],0)); ids=torch.where(residual)[0]; chosen=ids[torch.argsort(score[ids],descending=True)[:q]] if q else torch.empty(0,dtype=torch.long); selected=baseline.clone(); selected[chosen]=True
        c=by[it['profile']]; total['images']+=1; c['images']+=1
        for i in torch.where(selected)[0].tolist():
            total['pred']+=1;c['pred']+=1; ok=bool(it['labels'][i]>.5); total['true']+=ok;c['true']+=ok
            if ok and it['type_labels'][i] is not None:
                good=bool(it['chemical_logits'][i].argmax()==({'SINGLE':0,'DOUBLE':1,'TRIPLE':2,'AROMATIC':3}[it['type_labels'][i]])); total['typed']+=good;c['typed']+=good
        target=int(it['labels'].sum()); total['target']+=target;c['target']+=target; total['add']+=len(chosen);c['add']+=len(chosen)
    def fin(c):
        return {'images':int(c['images']),'predicted_edges':int(c['pred']),'true_edges':int(c['true']),'typed_true_edges':int(c['typed']),'target_edges':int(c['target']),'additions':int(c['add']),'edge_precision':c['true']/max(c['pred'],1),'edge_recall':c['true']/max(c['target'],1),'typed_edge_precision':c['typed']/max(c['pred'],1),'typed_edge_recall':c['typed']/max(c['target'],1)}
    return fin(total),{p:fin(c) for p,c in sorted(by.items())}
def run():
    if OUT.exists(): raise FileExistsError(OUT)
    if digest(INITIAL)!=INITIAL_SHA: raise RuntimeError('checkpoint changed')
    torch.set_num_threads(1); torch.manual_seed(20261018); cache=torch.load(CACHE,map_location='cpu'); train,hold=cache['train'],cache['hold']; probe,mean,scale=fit(train)
    base,basep=evaluate(train,probe,mean,scale,{p:0 for p in sorted({x['profile'] for x in train})}); choices={}
    for p in sorted({x['profile'] for x in train}):
        best=None
        for q in QUOTAS:
            m,mp=evaluate([x for x in train if x['profile']==p],probe,mean,scale,{p:q})
            if m['edge_precision']+1e-12<base['edge_precision'] or m['typed_edge_precision']+1e-12<base['typed_edge_precision']: continue
            key=(m['typed_edge_recall'],m['edge_recall'],-q); best=(q,m,mp,key) if best is None or key>best[3] else best
        choices[p]=best[0] if best else 0
    holdm,holdp=evaluate(hold,probe,mean,scale,choices); bhold,bholdp=evaluate(hold,probe,mean,scale,{p:0 for p in choices}); deltas={p:{k:holdp[p][k]-bholdp[p][k] for k in ('edge_precision','edge_recall','typed_edge_precision','typed_edge_recall')} for p in holdp}
    gate={'checkpoint_unchanged':digest(INITIAL)==INITIAL_SHA,'sealed_test_read':False,'gpu_used':False,'production_checkpoint_written':False,'train_precision_floor':True,'each_profile_typed_precision_drop_le_0.01':all(v['typed_edge_precision']>=-.01 for v in deltas.values()),'each_profile_edge_precision_drop_le_0.01':all(v['edge_precision']>=-.01 for v in deltas.values()),'oa13_typed_recall_gain':deltas.get('oa13',{}).get('typed_edge_recall',0)>.005,'oa33_typed_recall_gain':deltas.get('oa33',{}).get('typed_edge_recall',0)>.005,'production_go':False}
    result={'schema_version':'chemvector-r17-residual-rank-quota-shadow-cpu-0.1','complete':True,'training_performed':True,'training_scope':'frozen_visual_residual_rank_quota_cpu','gpu_used':False,'sealed_test_read':False,'production_checkpoint_written':False,'cache_sha256':digest(CACHE),'initial_checkpoint_sha256':INITIAL_SHA,'choices':choices,'baseline_holdout':bhold,'holdout':holdm,'holdout_profiles':holdp,'deltas':deltas,'gate':gate}
    OUT.mkdir(); torch.save({'state_dict':probe.state_dict(),'mean':mean,'scale':scale,'initial_checkpoint_sha256':INITIAL_SHA},OUT/'probe-state.pt'); (OUT/'report.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': run()
