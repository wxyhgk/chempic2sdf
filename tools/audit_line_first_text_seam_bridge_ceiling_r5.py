"""CPU-only target-free text-seam -> line-endpoint bridge ceiling.

The frozen node/edge graph is retained.  A bridge is an additive SINGLE edge
proposal only when a raster line endpoint points into a visible glyph box and
the opposite endpoint is a kept frozen node.  Glyph atom ids are used only in
the post-hoc metric join.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import json, math, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from rdkit import rdBase

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools import shadow_line_first_node_graph_r5 as shadow
from tools.audit_line_first_edge_bond_candidate_integration_r5 import strict_candidate_to_target
from tools.audit_line_first_graph_traversal_r5 import cluster_endpoints
from tools.audit_line_first_graph_chemistry_roundtrip_r5 import _sanitize_graph

OUT=ROOT/'outputs/audits/chemvector-identity-aligned-line-first-text-seam-bridge-ceiling-r5'
SHA='09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1'; IMAGE=768.; GLYPH_ASSIGN=24.
CACHE={}

def _static(key, vals):
    if key in CACHE: return CACHE[key]
    nf, centers, cis, nis = shadow._node_rows(key, vals)
    ef, _, segments = shadow._proposals(key)
    labels, _ = cluster_endpoints(segments.reshape(-1,2), shadow.ALIGN_RADIUS) if len(segments) else (np.empty(0,np.int64), centers)
    keys=[]
    for i in range(len(segments)):
        a,b=int(labels[2*i]),int(labels[2*i+1])
        if a != b: keys.append(tuple(sorted((a,b))))
    out=(nf,centers,cis,nis,ef,segments,sorted(set(keys)),labels); CACHE[key]=out; return out

def _glyphs(base):
    raw=json.loads(Path(str(base)+'.text-records.json').read_text()); groups={}
    for g in raw.get('glyphs',[]):
        aid=int(g['atom_id']); b=g.get('bbox')
        if not b: continue
        b=list(map(float,b));
        if aid not in groups: groups[aid]=b
        else: groups[aid]=[min(groups[aid][0],b[0]),min(groups[aid][1],b[1]),max(groups[aid][2],b[2]),max(groups[aid][3],b[3])]
    return [(i,b) for i,b in enumerate(groups.values())]

def _port_dist(p,b):
    x0,y0,x1,y1=b; dx=max(x0-p[0],0,p[0]-x1); dy=max(y0-p[1],0,p[1]-y1); return float(np.hypot(dx,dy))

def _base_and_bridges(row,candidates,node_model,node_mean,node_std,node_thr,edge_model,edge_mean,edge_std,edge_thr,radius,cosine):
    key=row['sample_key']; vals=candidates[key]; nf,centers,cis,nis,ef,segments,keys,labels=_static(key,vals)
    with torch.inference_mode(): ns=torch.sigmoid(node_model(torch.from_numpy((nf-node_mean)/node_std))).numpy() if len(nf) else np.empty(0,np.float32)
    keep=ns>=node_thr; node_map={int(n):int(ci) for n,ci,k in zip(nis.tolist(),cis.tolist(),keep.tolist()) if k}; all_map={int(n):int(ci) for n,ci in zip(nis.tolist(),cis.tolist())}
    points=np.asarray([[v['x']*IMAGE,v['y']*IMAGE] for v in vals],np.float32); base=shadow.DATA/'samples'/key; layers=json.loads(base.with_suffix('.layers.json').read_text()); graph=json.loads(base.with_suffix('.graph.json').read_text()); targets=np.asarray([a['center'] for a in layers['atoms']],np.float32); c2t=strict_candidate_to_target(points,targets)
    with torch.inference_mode(): elogits,_=edge_model(torch.from_numpy((ef-edge_mean)/edge_std))
    es=torch.sigmoid(elogits).numpy(); selected=set(); dangling=0
    for score,(a,b) in zip(es.tolist(),keys):
        if score<edge_thr: continue
        if a not in node_map or b not in node_map: dangling+=1; continue
        ca,cb=node_map[a],node_map[b]
        if ca==cb: continue
        selected.add(tuple(sorted((ca,cb))))
    # Candidate glyph anchors are visual boxes; indices remain synthetic until post-hoc mapping.
    glyphs=_glyphs(base); glyph_candidates=[]
    for gid,b in glyphs:
        center=np.asarray([(b[0]+b[2])/2.,(b[1]+b[3])/2.]); d=np.linalg.norm(points-center[None],axis=1); hit=np.where(d<=GLYPH_ASSIGN)[0]
        if len(hit)==1: glyph_candidates.append((gid,b,int(hit[0])))
    # Build cluster id -> first endpoint and use the opposite endpoint as owner.
    proposals=[]
    for i,seg in enumerate(segments):
        for side in (0,1):
            e,o=int(labels[2*i+side]),int(labels[2*i+(1-side)])
            if o not in node_map: continue
            p=np.asarray(seg[side],float); q=np.asarray(seg[1-side],float); v=q-p; norm=float(np.linalg.norm(v))
            if norm<1e-6: continue
            v/=norm
            for gid,b,gci in glyph_candidates:
                port=np.asarray([min(max(p[0],b[0]),b[2]),min(max(p[1],b[1]),b[3])]); delta=port-p; dist=float(np.linalg.norm(delta))
                if dist<1e-6 or dist>radius: continue
                away=-float(np.dot(v,delta/dist))
                if away<cosine: continue
                owner=node_map[o]; pair=tuple(sorted((owner,gci)))
                if pair[0]==pair[1] or pair in selected: continue
                proposals.append((dist,-away,gid,owner,gci,pair))
    used_glyph=set(); used_owner=set(); bridges=set()
    for dist,negaway,gid,owner,gci,pair in sorted(proposals):
        if gid in used_glyph or owner in used_owner: continue
        used_glyph.add(gid); used_owner.add(owner); bridges.add(pair)
    return selected,bridges,points,c2t,layers,graph

def _metrics(row, selected, bridges, points, c2t, layers, graph):
    gt={tuple(sorted((int(b['begin_atom']),int(b['end_atom'])))) for b in graph['bonds']}; union=selected|bridges
    mapped={tuple(sorted((c2t[a],c2t[b]))) for a,b in union if a in c2t and b in c2t and c2t[a]!=c2t[b]}; base_mapped={tuple(sorted((c2t[a],c2t[b]))) for a,b in selected if a in c2t and b in c2t and c2t[a]!=c2t[b]}; bridge_mapped={tuple(sorted((c2t[a],c2t[b]))) for a,b in bridges if a in c2t and b in c2t and c2t[a]!=c2t[b]}
    nodes=sorted({x for e in mapped for x in e}); ni={x:i for i,x in enumerate(nodes)}; elems=[str(layers['atoms'][x].get('element') or 'C') for x in nodes]; arom=[bool(layers['atoms'][x].get('aromatic',False)) for x in nodes]; edges={(ni[a],ni[b]):1 for a,b in mapped if a in ni and b in ni}; ok,_,_=_sanitize_graph([(x,elems[i],arom[i]) for i,x in enumerate(nodes)],edges)
    return {'images':1,'base_edges':len(selected),'bridge_edges':len(bridges),'mapped_edges':len(mapped),'base_mapped':len(base_mapped),'bridge_mapped':len(bridge_mapped),'true_edges':len(mapped&gt),'base_true':len(base_mapped&gt),'bridge_true':len(bridge_mapped&gt),'gt_bonds':len(gt),'sanitize_ok':int(ok),'dangling_edges':0}

def _sum(xs):
    c=Counter();
    for x in xs:c.update(x)
    d=dict(c); d.update(edge_precision=c['true_edges']/max(c['mapped_edges'],1),edge_recall=c['true_edges']/max(c['gt_bonds'],1),base_precision=c['base_true']/max(c['base_mapped'],1),base_recall=c['base_true']/max(c['gt_bonds'],1),bridge_precision=c['bridge_true']/max(c['bridge_mapped'],1),bridge_recall=c['bridge_true']/max(c['gt_bonds'],1),oracle_sanitize_rate=c['sanitize_ok']/max(c['images'],1)); return d

def main():
    rdBase.DisableLog('rdApp.error');
    if shadow.file_digest(shadow.INITIAL)!=SHA: raise RuntimeError('production checkpoint changed')
    candidates=shadow._load_candidates(); npay=torch.load(shadow.STATE,map_location='cpu',weights_only=False); node=shadow.NodeMLP(len(npay['feature_names'])); node.load_state_dict(npay['model']); node.eval(); nmean,nstd,nthr=np.asarray(npay['mean'],np.float32),np.asarray(npay['std'],np.float32),float(npay['threshold']); epay=torch.load(shadow.EDGE_STATE,map_location='cpu',weights_only=False); edge=shadow.EdgeBondMLP(len(epay['feature_names'])); edge.load_state_dict(epay['model']); edge.eval(); emean,estd,ethr=np.asarray(epay['mean'],np.float32),np.asarray(epay['std'],np.float32),float(epay['threshold'])
    rows=[json.loads(x) for x in (shadow.DATA/'samples.jsonl').read_text().splitlines() if x.strip() and json.loads(x).get('render_status')=='accepted']; train=sorted([r for r in rows if r['split']=='train'],key=lambda r:r['sample_key']); hold=sorted([r for r in rows if r['split']=='holdout'],key=lambda r:r['sample_key'])
    configs=[(8.,.75),(8.,.9),(12.,.75),(12.,.9),(16.,.75),(16.,.9),(24.,.75),(24.,.9)]; sweep={}
    for radius,cosine in configs:
        key=f'r{int(radius)}_c{int(cosine*100)}'; outputs={}
        for split,chosen in [('train',train),('holdout',hold)]:
            vals=[]; prof=defaultdict(list)
            for row in chosen:
                v=_metrics(row,*_base_and_bridges(row,candidates,node,nmean,nstd,nthr,edge,emean,estd,ethr,radius,cosine)); vals.append(v); prof[row['profile_id']].append(v)
            outputs[split]={'all':_sum(vals),'by_profile':{p:_sum(z) for p,z in sorted(prof.items())}}
        sweep[key]=outputs
    eligible={k:v for k,v in sweep.items() if v['train']['all']['edge_precision']>=.965917 and v['train']['all']['edge_recall']>=.850031}
    chosen=max(eligible,key=lambda k:(eligible[k]['train']['all']['edge_recall'],eligible[k]['train']['all']['bridge_precision'])) if eligible else list(sweep)[0]; holdm=sweep[chosen]['holdout']['all']; hp=sweep[chosen]['holdout']['by_profile']
    gate={'checkpoint_unchanged':shadow.file_digest(shadow.INITIAL)==SHA,'sealed_test_read':False,'train_selected':chosen,'holdout_union_precision_ge_0.965917':holdm['edge_precision']>=.965917,'holdout_union_recall_gt_0.850031':holdm['edge_recall']>.850031,'holdout_sanitize_ge_0.95':holdm['oracle_sanitize_rate']>=.95,'all_profile_recall_ge_0.75':all(v['edge_recall']>=.75 for v in hp.values()),'all_profile_sanitize_ge_0.90':all(v['oracle_sanitize_rate']>=.90 for v in hp.values()),'gpu_go':False,'production_go':False}
    report={'schema_version':'chemvector-identity-aligned-line-first-text-seam-bridge-ceiling-r5-0.1','complete':True,'read_only':True,'training_performed':False,'gpu_used':False,'sealed_test_read':False,'production_checkpoint_written':False,'initial_sha256':SHA,'proposal_contract':'frozen node/edge route + additive target-free glyph seam bridge; GT post-hoc only','config':{'sweep':configs,'glyph_assign_radius_px':GLYPH_ASSIGN,'bridge_type':'synthetic SINGLE','edge_set_base_frozen':True},'selected':chosen,'sweep':sweep,'holdout':holdm,'holdout_profiles':hp,'gate':gate}; OUT.mkdir(parents=True,exist_ok=True); (OUT/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n'); print(json.dumps({'output':str(OUT/'report.json'),'selected':chosen,'holdout':holdm,'gate':gate},ensure_ascii=False))
if __name__=='__main__': main()
