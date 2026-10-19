"""Strict candidate-aligned replay of the isolated line-first edge shadow.

This audit removes the optimistic GT-center mapping used during shadow fitting:
line-first nodes are matched to the frozen scored vertex candidate field using
only image/proposal geometry. Ground truth is consulted only for final edge and
bond-type metrics. No production graph or checkpoint is touched.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from chemvector.training.curriculum.checkpoints import file_digest
from tools.shadow_line_first_edge_bond_type_r5 import DATA, INITIAL, INITIAL_SHA, STATE, EdgeBondMLP, _proposals, _candidate_keys
from tools.audit_line_first_candidate_alignment_r5 import align_nodes, read_jsonl, TARGET_RADIUS, AMBIGUITY_RADIUS, IMAGE
from tools.audit_line_first_graph_traversal_r5 import cluster_endpoints
from chemvector.components.vision.line_segment_rescue import LineSegmentRescueConfig, detect_raster_line_segments

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "outputs/audits/chemvector-identity-aligned-vertex-head-shadow-20261017-r5-v5-scored/candidates.jsonl"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-edge-bond-candidate-integration-r5/report.json"

def strict_candidate_to_target(points, targets):
    out={}
    for i,p in enumerate(points):
        d=np.linalg.norm(targets-p,axis=1); close=np.where(d<=TARGET_RADIUS)[0]; near=np.where(d<=AMBIGUITY_RADIUS)[0]
        if len(close)==1 and len(near)==1: out[i]=int(close[0])
    return out

def evaluate(row,cand,model,mean,std,threshold):
    sample_key=row['sample_key']; f,centers,_=_proposals(sample_key); keys=_candidate_keys(sample_key)
    base=DATA/'samples'/sample_key; layers=json.loads(base.with_suffix('.layers.json').read_text()); graph=json.loads(base.with_suffix('.graph.json').read_text())
    points=np.asarray([[x['x']*IMAGE,x['y']*IMAGE] for x in cand],dtype=np.float32)
    node_map=align_nodes(centers.astype(np.float64),points.astype(np.float64),16.0)
    with torch.inference_mode(): edge_logits,type_logits=model(torch.from_numpy((f-mean)/std))
    scores=torch.sigmoid(edge_logits).numpy(); types=type_logits.numpy()
    candidate_types={}; candidate_edges=set()
    for idx,(a,b) in enumerate(keys):
        if scores[idx]<threshold or a not in node_map or b not in node_map: continue
        ca,cb=node_map[a],node_map[b]
        if ca==cb: continue
        pair=tuple(sorted((ca,cb))); candidate_edges.add(pair)
        old=candidate_types.get(pair)
        if old is None or scores[idx]>old[0]: candidate_types[pair]=(float(scores[idx]),int(np.argmax(types[idx])+1))
    targets=np.asarray([x['center'] for x in layers['atoms']],dtype=np.float32)
    c2t=strict_candidate_to_target(points,targets)
    mapped={tuple(sorted((c2t[a],c2t[b]))) for a,b in candidate_edges if a in c2t and b in c2t and c2t[a]!=c2t[b]}
    gt={tuple(sorted((int(x['begin_atom']),int(x['end_atom'])))) for x in graph['bonds']}
    true=mapped&gt
    type_by={tuple(sorted((int(x['begin_atom']),int(x['end_atom'])))):str(x['chemical_type']).upper() for x in graph['bonds']}
    type_names=['SINGLE','DOUBLE','TRIPLE','AROMATIC','DATIVE','OTHER']
    type_correct=0; type_eval=0
    for pair,(score,kind) in candidate_types.items():
        mapped_pair=tuple(sorted((c2t[pair[0]],c2t[pair[1]]))) if pair[0] in c2t and pair[1] in c2t else None
        if mapped_pair in gt:
            type_eval+=1; expected=type_by[mapped_pair]; pred=type_names[kind-1] if 1<=kind<=len(type_names) else 'OTHER'; type_correct+=int(pred==expected)
    return {'sample_key':sample_key,'split':row['split'],'line_edges':len(keys),'aligned_nodes':len(node_map),'selected_candidate_edges':len(candidate_edges),'mapped_edges':len(mapped),'true_edges':len(true),'gt_bonds':len(gt),'edge_precision':len(true)/max(len(mapped),1),'edge_recall':len(true)/max(len(gt),1),'type_correct':type_correct,'type_evaluated':type_eval,'all_bonds_recovered':true==gt}

def main():
    if file_digest(INITIAL)!=INITIAL_SHA: raise RuntimeError('initial checkpoint changed')
    state=torch.load(STATE,map_location='cpu',weights_only=False); model=EdgeBondMLP(17); model.load_state_dict(state['model']); model.eval(); mean=np.asarray(state['mean'],np.float32); std=np.asarray(state['std'],np.float32); threshold=float(state['threshold'])
    cands={x['sample_key']:x['candidates'] for x in read_jsonl(CANDIDATES)}; rows=[json.loads(x) for x in (DATA/'samples.jsonl').read_text().splitlines() if x and json.loads(x).get('render_status')=='accepted']; rows=sorted(rows,key=lambda x:x['sample_key'])
    by={}
    for r in rows: by.setdefault(r['split'],[]).append(evaluate(r,cands[r['sample_key']],model,mean,std,threshold))
    def summ(xs):
        return {'images':len(xs),'line_edges':sum(x['line_edges'] for x in xs),'selected_candidate_edges':sum(x['selected_candidate_edges'] for x in xs),'mapped_edges':sum(x['mapped_edges'] for x in xs),'true_edges':sum(x['true_edges'] for x in xs),'gt_bonds':sum(x['gt_bonds'] for x in xs),'edge_precision':sum(x['true_edges'] for x in xs)/max(sum(x['mapped_edges'] for x in xs),1),'edge_recall':sum(x['true_edges'] for x in xs)/max(sum(x['gt_bonds'] for x in xs),1),'conditional_type_accuracy':sum(x['type_correct'] for x in xs)/max(sum(x['type_evaluated'] for x in xs),1),'images_all_bonds_recovered':sum(x['all_bonds_recovered'] for x in xs)}
    report={'schema_version':'chemvector-identity-aligned-line-first-edge-bond-candidate-integration-r5-0.1','complete':True,'read_only':True,'training_performed':False,'gpu_used':False,'sealed_test_read':False,'deployment_authorized':False,'production_checkpoint_written':False,'initial_sha256':INITIAL_SHA,'initial_checkpoint_sha256_after':file_digest(INITIAL),'sidecar_state_sha256':hashlib.sha256(STATE.read_bytes()).hexdigest(),'candidate_artifact_sha256':hashlib.sha256(CANDIDATES.read_bytes()).hexdigest(),'fixed_threshold':threshold,'proposal_contract':'LSD endpoints -> provisional edges -> target-free scored vertex candidate alignment -> sidecar edge/type score; targets only post-hoc','by_split':{k:summ(v) for k,v in by.items()},'gate':{'checkpoint_unchanged':file_digest(INITIAL)==INITIAL_SHA,'sealed_test_read':False,'production_go':False,'next':'diagnostic only; no production graph merge'}}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))

if __name__=='__main__': main()
