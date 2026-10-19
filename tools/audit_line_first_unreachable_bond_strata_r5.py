"""Read-only strata audit for GT bonds unreachable by frozen line-node proposals."""
from __future__ import annotations
from collections import Counter, defaultdict
import json, math, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools import shadow_line_first_node_graph_r5 as shadow
from tools.audit_line_first_edge_bond_candidate_integration_r5 import strict_candidate_to_target

OUT=ROOT/'outputs/audits/chemvector-identity-aligned-line-first-unreachable-bond-strata-r5'
INITIAL_SHA='09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1'; IMAGE=768.; ALIGN=16.; TARGET=6.; SEAM=10.

def _boxes(base):
    out={}
    for g in json.loads(Path(str(base)+'.text-records.json').read_text()).get('glyphs',[]):
        aid,b=int(g['atom_id']),g.get('bbox')
        if not b: continue
        b=list(map(float,b)); out[aid]=[min(out.get(aid,b)[0],b[0]),min(out.get(aid,b)[1],b[1]),max(out.get(aid,b)[2],b[2]),max(out.get(aid,b)[3],b[3])] if aid in out else b
    return out

def _box_dist(p,b):
    x0,y0,x1,y1=b; dx=max(x0-p[0],0,p[0]-x1); dy=max(y0-p[1],0,p[1]-y1); return float(np.hypot(dx,dy))

def _target(base):
    layers=json.loads(base.with_suffix('.layers.json').read_text()); graph=json.loads(base.with_suffix('.graph.json').read_text())
    targets=np.asarray([a['center'] for a in layers['atoms']],np.float32)
    gt_edges={tuple(sorted((int(b['begin_atom']),int(b['end_atom'])))) for b in graph['bonds']}
    gt_types={tuple(sorted((int(b['begin_atom']),int(b['end_atom'])))):str(b.get('chemical_type','OTHER')).upper() for b in graph['bonds']}
    return targets,gt_edges,gt_types

def _one(row,candidates):
    key=row['sample_key']; base=shadow.DATA/'samples'/key
    vals=candidates[key]; nf,centers,cis,nis=shadow._node_rows(key,vals)
    points=np.asarray([[v['x']*IMAGE,v['y']*IMAGE] for v in vals],np.float32)
    targets,gt_edges,gt_types=_target(base)
    c2t=strict_candidate_to_target(points,targets)
    # Every node candidate pair is formed before the trained node keep head.
    all_map={int(n):int(ci) for n,ci in zip(nis.tolist(),cis.tolist())}
    ef,_,_=shadow._proposals(key); keys=shadow._candidate_keys(key)
    raw_pairs=defaultdict(list)
    # candidate edge keys use provisional nodes from _proposals; map to chosen candidate indices
    for a,b in keys:
        if a not in all_map or b not in all_map: continue
        ca,cb=all_map[a],all_map[b]
        if ca not in c2t or cb not in c2t or c2t[ca]==c2t[cb]: continue
        raw_pairs[tuple(sorted((c2t[ca],c2t[cb])))].append((a,b))
    boxes=_boxes(base); visible=set(boxes)
    # LSD endpoint clusters are the centers used by node rows; candidate existence is post-hoc only.
    endpoint_has=[]
    for t in targets:
        endpoint_has.append(bool(len(centers) and float(np.min(np.linalg.norm(centers-t[None],axis=1)))<=ALIGN))
    cand_has=[]
    for i,t in enumerate(targets):
        d=np.linalg.norm(points-t[None],axis=1); cand_has.append(bool((d<=TARGET).sum()==1))
    # Any endpoint close to a glyph box boundary counts as a visual seam cue.
    seam_atoms=set()
    try:
        image=Image.open(Path(str(base)+'.png')).convert('L')
        seg=shadow._proposals(key)[2]
    except Exception:
        seg=np.empty((0,2,2),np.float32)
    if len(seg):
        for aid,b in boxes.items():
            if any(_box_dist(p,b)<=SEAM for p in seg.reshape(-1,2)): seam_atoms.add(aid)
    rows=[]
    for edge,typ in gt_types.items():
        if edge in raw_pairs: continue
        a,b=edge; pa,pb=targets[a],targets[b]; delta=pb-pa; length=float(np.linalg.norm(delta)); angle=float(np.degrees(np.arctan2(abs(delta[1]),abs(delta[0]))))
        def endpoint(i):
            return 'text_seam' if i in seam_atoms else ('text' if i in visible else ('implicit_cluster' if endpoint_has[i] else 'no_cluster'))
        rows.append({'profile_id':row['profile_id'],'bond_type':typ,'length_bin':'<30' if length<30 else '30-50' if length<50 else '50-80' if length<80 else '>=80','angle_bin':'0-30' if angle<30 else '30-60' if angle<60 else '60-90','a_source':endpoint(a),'b_source':endpoint(b),'a_vertex_candidate':cand_has[a],'b_vertex_candidate':cand_has[b],'a_cluster':endpoint_has[a],'b_cluster':endpoint_has[b],'both_vertex_candidate':cand_has[a] and cand_has[b],'both_cluster':endpoint_has[a] and endpoint_has[b],'both_text_seam':a in seam_atoms and b in seam_atoms})
    return rows

def _summ(rows):
    total=len(rows); c=Counter()
    for r in rows:
        c['bonds']+=1; c['both_vertex_candidate']+=r['both_vertex_candidate']; c['both_cluster']+=r['both_cluster']; c['both_text_seam']+=r['both_text_seam']; c['a_cluster']+=r['a_cluster']; c['b_cluster']+=r['b_cluster']
    out=dict(c)
    for k in ('both_vertex_candidate','both_cluster','both_text_seam','a_cluster','b_cluster'): out[k+'_rate']=c[k]/max(total,1)
    return out

def main():
    if shadow.file_digest(shadow.INITIAL)!=INITIAL_SHA: raise RuntimeError('production checkpoint changed')
    candidates=shadow._load_candidates(); rows=[json.loads(x) for x in (shadow.DATA/'samples.jsonl').read_text().splitlines() if x.strip() and json.loads(x).get('render_status')=='accepted']; hold=sorted([r for r in rows if r['split']=='holdout'],key=lambda r:r['sample_key'])
    losses=[z for r in hold for z in _one(r,candidates)]
    def group(key): return _summ(losses if key is None else [x for x in losses if key(x)])
    profiles={p:group(lambda x,p=p:x['profile_id']==p) for p in sorted({x['profile_id'] for x in losses})}
    by_type={p:group(lambda x,p=p:x['bond_type']==p) for p in sorted({x['bond_type'] for x in losses})}
    by_len={p:group(lambda x,p=p:x['length_bin']==p) for p in sorted({x['length_bin'] for x in losses})}
    by_angle={p:group(lambda x,p=p:x['angle_bin']==p) for p in sorted({x['angle_bin'] for x in losses})}
    source_pairs=Counter('|'.join(sorted((x['a_source'],x['b_source']))) for x in losses)
    report={'schema_version':'chemvector-identity-aligned-line-first-unreachable-bond-strata-r5-0.1','complete':True,'read_only':True,'training_performed':False,'gpu_used':False,'sealed_test_read':False,'production_checkpoint_written':False,'initial_sha256':INITIAL_SHA,'proposal_contract':'frozen node/vertex candidates + frozen edge proposal; unreachable GT bonds post-hoc strata only','config':{'alignment_radius_px':ALIGN,'vertex_candidate_radius_px':TARGET,'seam_radius_px':SEAM},'all':group(None),'by_profile':profiles,'by_bond_type':by_type,'by_length_bin':by_len,'by_angle_bin':by_angle,'missing_endpoint_source_pairs':dict(source_pairs),'unreachable_bonds':len(losses),'gate':{'checkpoint_unchanged':shadow.file_digest(shadow.INITIAL)==INITIAL_SHA,'sealed_test_read':False,'training_performed':False,'gpu_go':False,'production_go':False}}
    OUT.mkdir(parents=True,exist_ok=True); (OUT/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n'); print(json.dumps({'output':str(OUT/'report.json'),'unreachable_bonds':len(losses),'all':report['all'],'source_pairs':dict(source_pairs)},ensure_ascii=False))
if __name__=='__main__': main()
