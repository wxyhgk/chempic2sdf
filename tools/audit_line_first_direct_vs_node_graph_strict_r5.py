"""Strict, matched-metric comparison of direct LSD graph and node-graph shadow.

This is diagnostic only.  Both routes are scored on all 474 holdout images and
the six profiles with the same post-hoc identity and edge denominators.  Ground
truth is used only to score predictions; it never creates a prediction.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib, json, sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import audit_line_first_lsd_all_edge_ownership_r5 as lsd
from tools.audit_line_first_graph_traversal_r5 import cluster_endpoints
from tools.audit_line_first_edge_bond_candidate_integration_r5 import strict_candidate_to_target
from tools import shadow_line_first_node_graph_r5 as shadow
from chemvector.components.vision.line_segment_rescue import LineSegmentRescueConfig

DATA = ROOT / "outputs/datasets/chemvector-identity-aligned-visible-text-20261017-r5-render"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-direct-vs-node-graph-strict-r5/report.json"
IMAGE = 768.0
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"

def _boxes(base):
    out = {}
    for x in json.loads(Path(str(base)+'.text-records.json').read_text()).get('glyphs', []):
        aid = int(x['atom_id']); b = x.get('bbox')
        if not b: continue
        b = list(map(float,b))
        if aid not in out: out[aid] = b
        else: out[aid] = [min(out[aid][0],b[0]),min(out[aid][1],b[1]),max(out[aid][2],b[2]),max(out[aid][3],b[3])]
    return out

def _target(base):
    layers = json.loads(Path(str(base)+'.layers.json').read_text())
    graph = json.loads(Path(str(base)+'.graph.json').read_text())
    pts = np.asarray([a['center'] for a in layers['atoms']], np.float32)
    edges = {tuple(sorted((int(b['begin_atom']), int(b['end_atom'])))) for b in graph['bonds']}
    return pts, edges

def _sum(xs):
    keys = ('images','pred_nodes','matched_nodes','target_atoms','selected_edges','mapped_edges','true_edges','gt_bonds','unresolved_edges','self_loops')
    t = {k: sum(int(x.get(k,0)) for x in xs) for k in keys}
    t.update(node_precision=t['matched_nodes']/max(t['pred_nodes'],1), node_recall=t['matched_nodes']/max(t['target_atoms'],1),
             edge_precision=t['true_edges']/max(t['mapped_edges'],1), edge_recall=t['true_edges']/max(t['gt_bonds'],1),
             dangling_rate=t['unresolved_edges']/max(t['selected_edges'],1), self_loop_rate=t['self_loops']/max(t['selected_edges'],1))
    return t

def _direct(row):
    key = row['sample_key']; base = DATA/'samples'/key
    image = Image.open(Path(str(base)+'.png')).convert('L')
    seg = lsd._alt_detect(image, config=LineSegmentRescueConfig(maximum_candidates=512))
    endpoints = seg.reshape(-1,2) if len(seg) else np.empty((0,2), np.float32)
    labels, centers = cluster_endpoints(endpoints, 10.)
    boxes = _boxes(base)
    # Aliases are image-derived glyph boxes, not target labels.
    alias = {i:i for i in range(len(centers))}; pred_pts = [np.asarray(c,float) for c in centers]
    for aid,b in boxes.items():
        c = np.asarray([(b[0]+b[2])/2,(b[1]+b[3])/2])
        hits = [(float(np.linalg.norm(centers[i]-c)),i) for i in range(len(centers)) if np.linalg.norm(centers[i]-c)<=24.]
        if hits:
            tid = -(aid+1); pred_pts.append(c)
            for _,i in sorted(hits): alias[i] = tid
    # Map the negative text aliases and positive clusters to a compact prediction index.
    # alias contains the routed identity for every provisional cluster.  A
    # glyph alias replaces its cluster id; do not retain both (that would
    # double-count one visible node and unfairly depress node precision).
    ids = sorted(set(alias.values()))
    id_to_i = {n:i for i,n in enumerate(ids)}
    pred_pts2 = []
    for n in ids:
        if n < 0:
            aid = -n-1; b = boxes.get(aid); pred_pts2.append(np.asarray([(b[0]+b[2])/2,(b[1]+b[3])/2],np.float32))
        else: pred_pts2.append(np.asarray(centers[n],np.float32))
    pred_pts2 = np.asarray(pred_pts2,np.float32)
    targets, gt = _target(base)
    c2t = strict_candidate_to_target(pred_pts2, targets)
    selected = set()
    for i in range(len(seg)):
        a,b = int(labels[2*i]),int(labels[2*i+1]); na,nb=alias.get(a,a),alias.get(b,b)
        if na != nb: selected.add(tuple(sorted((id_to_i[na],id_to_i[nb]))))
    mapped=set(); unresolved=0; loops=0
    for a,b in selected:
        if a not in c2t or b not in c2t: unresolved += 1; continue
        ta,tb = c2t[a],c2t[b]
        if ta==tb: loops += 1; continue
        mapped.add(tuple(sorted((ta,tb))))
    true = mapped & gt
    return {'images':1,'pred_nodes':len(pred_pts2),'matched_nodes':len(c2t),'target_atoms':len(targets),'selected_edges':len(selected),'mapped_edges':len(mapped),'true_edges':len(true),'gt_bonds':len(gt),'unresolved_edges':unresolved,'self_loops':loops}

def _node_shadow(row, candidates, model, mean, std, threshold):
    # Reuse shadow's target-free node/edge decoder, then score with identical
    # node/edge/dangling denominators.
    vals=candidates[row['sample_key']]; nf, centers, cis, nis = shadow._node_rows(row['sample_key'], vals)
    import torch
    with torch.inference_mode(): scores=torch.sigmoid(model(torch.from_numpy((nf-mean)/std))).numpy()
    keep=scores>=threshold; node_map={int(n):int(ci) for n,ci,k in zip(nis.tolist(),cis.tolist(),keep.tolist()) if k}
    points=np.asarray([[v['x']*IMAGE,v['y']*IMAGE] for v in vals],np.float32); base=DATA/'samples'/row['sample_key']; targets,gt=_target(base)
    c2t=strict_candidate_to_target(points,targets)
    edge_features,_,_=shadow._proposals(row['sample_key']); keys=shadow._candidate_keys(row['sample_key'])
    payload=torch.load(shadow.EDGE_STATE,map_location='cpu',weights_only=False); em=shadow.EdgeBondMLP(len(payload['feature_names'])); em.load_state_dict(payload['model']); em.eval(); emean=np.asarray(payload['mean'],np.float32); estd=np.asarray(payload['std'],np.float32); eth=float(payload['threshold'])
    with torch.inference_mode():
        elogits, _ = em(torch.from_numpy((edge_features-emean)/estd)); es=torch.sigmoid(elogits).numpy()
    selected=set(); unresolved=loops=0
    for s,(a,b) in zip(es.tolist(),keys):
        if s<eth: continue
        if a not in node_map or b not in node_map: unresolved+=1; continue
        ca,cb=node_map[a],node_map[b]
        if ca==cb: loops+=1; continue
        selected.add(tuple(sorted((ca,cb))))
    mapped=set()
    for a,b in selected:
        if a not in c2t or b not in c2t: unresolved+=1; continue
        ta,tb=c2t[a],c2t[b]
        if ta==tb: loops+=1; continue
        mapped.add(tuple(sorted((ta,tb))))
    true=mapped&gt
    return {'images':1,'pred_nodes':len(node_map),'matched_nodes':len({c2t[c] for c in node_map.values() if c in c2t}),'target_atoms':len(targets),'selected_edges':len(selected)+unresolved,'mapped_edges':len(mapped),'true_edges':len(true),'gt_bonds':len(gt),'unresolved_edges':unresolved,'self_loops':loops}

def main():
    if shadow.file_digest(shadow.INITIAL)!=INITIAL_SHA: raise RuntimeError('checkpoint changed')
    rows=[json.loads(x) for x in (DATA/'samples.jsonl').read_text().splitlines() if x.strip() and json.loads(x).get('render_status')=='accepted']
    candidates=shadow._load_candidates(); payload=__import__('torch').load(shadow.STATE,map_location='cpu',weights_only=False); model=shadow.NodeMLP(len(payload['feature_names'])); model.load_state_dict(payload['model']); model.eval(); mean=np.asarray(payload['mean'],np.float32); std=np.asarray(payload['std'],np.float32); threshold=float(payload['threshold'])
    out={}
    for split in ('train','holdout'):
        rr=[r for r in rows if r['split']==split]; d=[_direct(r) for r in rr]; n=[_node_shadow(r,candidates,model,mean,std,threshold) for r in rr]
        out[split]={'direct':_sum(d),'node_shadow':_sum(n),'direct_by_profile':{p:_sum([x for x,r in zip(d,rr) if r['profile_id']==p]) for p in sorted({r['profile_id'] for r in rr})},'node_shadow_by_profile':{p:_sum([x for x,r in zip(n,rr) if r['profile_id']==p]) for p in sorted({r['profile_id'] for r in rr})}}
    rep={'schema_version':'chemvector-identity-aligned-line-first-direct-vs-node-graph-strict-r5-0.1','complete':True,'read_only':True,'training_performed':False,'gpu_used':False,'sealed_test_read':False,'production_checkpoint_written':False,'initial_sha256':INITIAL_SHA,'proposal_contract':'direct new-LSD graph and existing node-graph shadow scored with identical strict target matching; GT post-hoc only','splits':out,'gate':{'checkpoint_unchanged':shadow.file_digest(shadow.INITIAL)==INITIAL_SHA,'sealed_test_read':False,'production_go':False}}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(rep,indent=2)+'\n'); print(json.dumps(rep,indent=2))

if __name__=='__main__': main()
