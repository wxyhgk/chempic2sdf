#!/usr/bin/env python3
"""CPU-only shadow: frozen endpoint ranker plus target-free valence budget.

The endpoint candidate set and chemical logits remain frozen.  The sidecar only
reorders candidates and applies a generic incident bond-order budget (4.0 per
endpoint); no target atom identity, sealed data, or checkpoint is consulted.
This is an attribution probe, not a production decoder.
"""
from __future__ import annotations
from collections import defaultdict
import hashlib, json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "outputs/audits/chemvector-graph-context-aromatic-cpu-r17-all-m128-t0p72-c1-panel128/feature-cache.pt"
RANK_STATE = ROOT / "outputs/audits/chemvector-endpoint-conditioned-candidate-rank-cpu-r17-m128-v2/ranker-state.pt"
INITIAL = ROOT / "outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt"
SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
OUT = ROOT / "outputs/audits/chemvector-endpoint-rank-graph-constraint-cpu-r17"

def digest(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

class Ranker(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.net=torch.nn.Sequential(torch.nn.LayerNorm(526),torch.nn.Linear(526,64),torch.nn.GELU(),torch.nn.Linear(64,1))
    def forward(self,x): return self.net(x).squeeze(-1)

def order(logit):
    # BOND_TYPES = SINGLE, DOUBLE, TRIPLE, AROMATIC. Aromatic consumes 1.5.
    return (1.0,2.0,3.0,1.5)[int(logit.argmax())]

def select(rec, head, threshold, constraint):
    with torch.inference_mode(): s=head(rec['features'].float()).sigmoid()
    eligible=[i for i,v in enumerate(s.tolist()) if v>=threshold]
    if not constraint: return eligible
    # Target-free greedy maximum-score independent set under a bond-order budget.
    used=defaultdict(float); chosen=[]
    for i in sorted(eligible,key=lambda j:(-float(s[j]),j)):
        a,b,*_=rec['pairs'][i]; a,b=int(a),int(b)
        w=order(rec['edge_chemical_logits'][i])
        if used[a]+w<=4.000001 and used[b]+w<=4.000001:
            chosen.append(i); used[a]+=w; used[b]+=w
    return chosen

def metric(records,head,t, constraint):
    total=defaultdict(int); by=defaultdict(lambda:defaultdict(int))
    for rec in records:
        chosen=select(rec,head,t,constraint); labels=rec['labels']>.5
        typed=labels & (rec['edge_chemical_logits'].argmax(-1)==rec['type_labels'])
        p=str(rec['sample_key'].rsplit('__',1)[-1]); vals={'images':1,'predicted':len(chosen),'true':sum(bool(labels[i]) for i in chosen),'typed':sum(bool(typed[i]) for i in chosen),'target':int(labels.sum())}
        for k,v in vals.items(): total[k]+=v; by[p][k]+=v
    def done(c):
        pred=max(c['predicted'],1); target=max(c['target'],1)
        return {'images':int(c['images']),'predicted_edges':int(c['predicted']),'true_edges':int(c['true']),'typed_true_edges':int(c['typed']),'target_edges':int(c['target']),'edge_precision':c['true']/pred,'edge_recall':c['true']/target,'typed_edge_precision':c['typed']/pred,'typed_edge_recall':c['typed']/target}
    return done(total),{k:done(v) for k,v in sorted(by.items())}

def main():
    if OUT.exists(): raise FileExistsError(OUT)
    if digest(INITIAL)!=SHA: raise RuntimeError('production checkpoint changed')
    cache=torch.load(CACHE,map_location='cpu',weights_only=False); state=torch.load(RANK_STATE,map_location='cpu',weights_only=False)
    if cache.get('initial_checkpoint_sha256')!=SHA or state.get('initial_checkpoint_sha256')!=SHA: raise RuntimeError('checkpoint binding mismatch')
    head=Ranker(); head.load_state_dict(state['state_dict']); head.eval(); torch.set_num_threads(1)
    records=cache['records']; thresholds=[i/100 for i in range(1,51)]
    baselines={k:metric(v,head,0.05,False) for k,v in records.items()}
    choices={}
    for mode in ('rank_only','valence_greedy'):
        cand=[]
        for t in thresholds:
            m,_=metric(records['train'],head,t,mode=='valence_greedy')
            # retain a modest train precision floor relative to rank-only's
            # chosen-threshold result, then maximize recall.
            floor=baselines['train'][0]['edge_precision']-0.005
            if m['edge_precision']>=floor: cand.append((m['edge_recall'],m['typed_edge_recall'],-t,t,m))
        choices[mode]=max(cand) if cand else None
    scored={}; deltas={}
    for mode in ('rank_only','valence_greedy'):
        t=choices[mode][3] if choices[mode] else .05
        scored[mode]={k:metric(v,head,t,mode=='valence_greedy') for k,v in records.items()}
        deltas[mode]={k:{x:scored[mode][k][0][x]-baselines[k][0][x] for x in ('edge_precision','edge_recall','typed_edge_precision','typed_edge_recall')} for k in records}
    report={'schema_version':'chemvector-endpoint-rank-graph-constraint-cpu-r17-0.1','complete':True,'training_performed':False,'training_scope':'frozen_ranker_target_free_valence_shadow','gpu_used':False,'sealed_test_read':False,'production_checkpoint_written':False,'cache':str(CACHE.relative_to(ROOT)),'rank_state':str(RANK_STATE.relative_to(ROOT)),'initial_checkpoint_sha256':SHA,'initial_checkpoint_sha256_after':digest(INITIAL),'constraint':'greedy score order; per endpoint weighted bond-order budget <=4.0; aromatic=1.5','baseline_rank_threshold':.05,'choices':{k:(None if v is None else {'threshold':v[3],'train':v[4]}) for k,v in choices.items()},'baselines':baselines,'scored':scored,'deltas':deltas,'gate':{'checkpoint_unchanged':digest(INITIAL)==SHA,'candidate_edge_set_frozen':True,'sealed_test_read':False,'gpu_used':False,'production_go':False}}
    OUT.mkdir(parents=False); (OUT/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n'); print(json.dumps({'output':str(OUT/'report.json'),'choices':report['choices'],'deltas':deltas,'gate':report['gate']},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
