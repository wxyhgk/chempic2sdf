#!/usr/bin/env python3
"""Post-hoc safety inventory for the target-free r13 valence repair.

Elements and true bond labels are joined only after the target-free repair has
been applied.  This is an audit of risk, not an element-aware production rule.
"""
from __future__ import annotations
from collections import Counter
import hashlib, json, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from chemvector.bond_object import BOND_TYPES
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from tools.train_direct_graph_pair_scorer_cpu_r9 import DATA, SHA, rows, _extract, DirectPairResidual
from tools.train_direct_graph_type_profile_residual_cpu_r13 import TypeResidual
from tools.diagnose_aromatic_cycle_veto_r13 import has_path

OUT = ROOT / "outputs/audits/chemvector-identity-aligned-direct-graph-valence-repair-safety-r13"
PRESENCE = ROOT / "outputs/audits/chemvector-identity-aligned-direct-graph-pair-scorer-cpu-r10-coverage/scorer-state.pt"
TYPE = ROOT / "outputs/audits/chemvector-identity-aligned-direct-graph-type-profile-residual-cpu-r13/type-state.pt"
THRESHOLD = .72; SCALE = .5

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()

def main():
    if digest(INITIAL) != SHA: raise RuntimeError('production checkpoint changed')
    model,cfg,_=load_bundle_for_probe(INITIAL,SHA); model.eval().requires_grad_(False)
    hold=sorted((r for r in rows() if r.get('split')=='fresh_holdout'),key=lambda r:r['sample_key'])
    items=[_extract(r,model,cfg,need_labels=True) for r in hold]
    ps=torch.load(PRESENCE,map_location='cpu',weights_only=False); presence=DirectPairResidual(262);presence.load_state_dict(ps['state_dict']);presence.eval()
    ts=torch.load(TYPE,map_location='cpu',weights_only=False);heads={p:TypeResidual(262) for p in ts['heads']}
    for p,h in heads.items():h.load_state_dict(ts['heads'][p]);h.eval()
    order={0:1.0,1:2.0,2:3.0,3:1.5}; changes=[]
    for item in items:
        with torch.inference_mode():
            pl=item['base_logits']+presence(item['features']); typ=item['chemical_logits']+SCALE*heads[item['row']['profile_id']](item['features'])
        pairs=item['pairs'];slots=item['slots'];st={int(x):i for i,x in enumerate(slots.tolist())};sel=torch.where(pl.sigmoid()>=THRESHOLD)[0].tolist();edges=[]
        for i in sel:
            a,b=(int(x) for x in pairs[i].tolist())
            if a in st and b in st and st[a]!=st[b]: edges.append((st[a],st[b],i))
        adj={}
        for a,b,_ in edges:adj.setdefault(a,set()).add(b);adj.setdefault(b,set()).add(a)
        out=typ.clone()
        # Match the reviewed cycle veto first.
        for a,b,i in edges:
            if int(typ[i].argmax())==3 and not has_path(adj,a,b,tuple(sorted((a,b)))):
                out[i,3]=-1e4;out[i,0]=typ[i].max()+1; 
        for _ in range(32):
            kinds={i:int(out[i].argmax()) for _,_,i in edges};totals={}
            for a,b,i in edges:
                totals[a]=totals.get(a,0)+order[kinds[i]];totals[b]=totals.get(b,0)+order[kinds[i]]
            bad=[u for u,v in totals.items() if v>4+1e-6]
            if not bad: break
            cand=[(i,kinds[i],float((out[i].max()-out[i].topk(2).values[-1]).detach())) for a,b,i in edges if any(u in (a,b) for u in bad) and kinds[i]!=0]
            if not cand: break
            i,k,margin=min(cand,key=lambda z:(-order[z[1]],z[2],z[0]))
            a,b=next((a,b) for a,b,j in edges if j==i)
            # map compact endpoint IDs to target IDs and true labels post-hoc
            mapping=item['mapping']; ta,tb=mapping.get(a),mapping.get(b); edge=tuple(sorted((ta,tb))) if ta is not None and tb is not None else None
            layer=json.loads((DATA/'samples'/item['row']['sample_key']).with_suffix('.layers.json').read_text())
            atoms={int(x['atom_id']):x for x in layer['atoms']}
            true=item['target_type_by_edge'].get(edge) if edge is not None else None
            changes.append({'sample_key':item['row']['sample_key'],'profile':item['row']['profile_id'],'edge':edge,'elements':([atoms[int(ta)].get('element','C'),atoms[int(tb)].get('element','C')] if edge is not None else []),'before':BOND_TYPES[k],'true':true,'is_true_edge':bool(true),'margin':margin})
            out[i,0]=out[i].max()+1;out[i,1:]=-1e4
    by_kind=Counter((tuple(sorted(x['elements'])),x['before'],x['true'],x['is_true_edge']) for x in changes)
    result={'schema_version':'chemvector-identity-aligned-direct-graph-valence-repair-safety-r13-0.1','complete':True,'read_only':True,'training_performed':False,'gpu_used':False,'sealed_test_read':False,'production_checkpoint_written':False,'repair_count':len(changes),'by_kind':{str(k):v for k,v in by_kind.items()},'changes':changes,'initial_checkpoint_sha256':SHA,'initial_checkpoint_sha256_after':digest(INITIAL),'gate':{'checkpoint_unchanged':digest(INITIAL)==SHA,'sealed_test_read':False,'production_go':False}}
    OUT.mkdir(parents=True,exist_ok=False);(OUT/'report.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n');print(json.dumps({'output':str(OUT/'report.json'),'repair_count':len(changes),'by_kind':result['by_kind'],'gate':result['gate']},indent=2,ensure_ascii=False))
if __name__=='__main__':main()
