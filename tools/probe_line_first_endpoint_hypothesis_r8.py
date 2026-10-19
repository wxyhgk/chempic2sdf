#!/usr/bin/env python3
"""Read-only endpoint-hypothesis ceiling probe (r8).

Unlike the r7 port reducer, this keeps a set of endpoint hypotheses for each
segment port.  Text-seam ports may continue along the incident ray to the
*interior* of a glyph box (up to 64 px); implicit-C ports retain explicit
backward ray hypotheses.  A target-free support vote selects one shared node
per port.  Text labels/atom ids and graph identities are read only after the
proposal is constructed for metrics.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib, json, math, sys
from pathlib import Path
import numpy as np
from PIL import Image
from rdkit import rdBase

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import audit_line_first_lsd_all_edge_ownership_r5 as lsd
from tools.audit_line_first_graph_traversal_r5 import cluster_endpoints
from tools.audit_line_first_edge_bond_candidate_integration_r5 import strict_candidate_to_target
from chemvector.components.vision.line_segment_rescue import LineSegmentRescueConfig

DATA = ROOT / "outputs/datasets/chemvector-identity-aligned-line-first-targeted-diversity-r6"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-endpoint-hypothesis-r8b"
SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
IMAGE = 768.0

def _boxes(base: Path):
    payload = json.loads(Path(str(base) + ".text-records.json").read_text())
    out = []
    for g in payload.get("glyphs", []):
        b = g.get("bbox")
        if b is None or len(b) != 4: continue
        b = np.asarray(b, np.float32)
        if b[2] <= b[0] or b[3] <= b[1]: continue
        if not any(float(np.max(np.abs(b-x))) < 1e-5 for x in out): out.append(b)
    return out

def _unit(x):
    n = float(np.linalg.norm(x)); return x / n if n > 1e-6 else np.zeros(2, np.float32)

def _propose(row, mode):
    base = DATA / "samples" / row["sample_key"]
    image = Image.open(Path(str(base)+".png")).convert("L")
    seg = lsd._alt_detect(image, config=LineSegmentRescueConfig(maximum_candidates=512))
    ep = seg.reshape(-1, 2) if len(seg) else np.empty((0,2), np.float32)
    labels, centers = cluster_endpoints(ep, 10.0)
    boxes = _boxes(base); bcent = [(b[:2]+b[2:])/2.0 for b in boxes]
    # Incident outward directions for each clustered port.
    incident = defaultdict(list)
    for si in range(len(seg)):
        for side in (0,1):
            c = int(labels[2*si+side]); q = seg[si,1-side] - centers[c]
            if np.linalg.norm(q)>1e-5: incident[c].append(_unit(q))
    # Node coordinates: raw clusters plus text centres; text centres are only
    # admitted when a ray actually points into their glyph box.
    node_pts = [np.asarray(c, np.float32) for c in centers]
    node_kind = ["implicit" for _ in node_pts]
    text_nodes = []
    for j,b in enumerate(boxes):
        text_nodes.append(len(node_pts)); node_pts.append(np.asarray(bcent[j], np.float32)); node_kind.append("text")
    # Port -> hypotheses (node id, image-only cost). Text seam gets a much
    # larger radius than r7 and uses directional compatibility, not labels.
    choices = defaultdict(list)
    for c in range(len(centers)): choices[c].append((c, 0.0))
    for c, dirs in incident.items():
        for j,b in enumerate(boxes):
            v = np.asarray(bcent[j]) - centers[c]; dist=float(np.linalg.norm(v))
            if dist > 64.0 or dist < 1e-4: continue
            u = _unit(v); align=max((float(np.dot(d,u)) for d in dirs), default=-1.)
            if align < (0.15 if mode == "ray_anchor" else 0.35): continue
            # box containment / boundary proximity avoids assigning a remote
            # glyph merely because two rays are parallel.
            margin = np.maximum.reduce([b[:2]-centers[c], centers[c]-b[2:]])
            if float(np.max(margin)) > 64.0: continue
            choices[c].append((text_nodes[j], dist - 12.0*align))
        if mode == "extended":
            # Explicit implicit-C hypotheses: move backward from the observed
            # cap along each incident ray.  They are kept as shared nodes only
            # when support from another port agrees spatially.
            for d in dirs:
                for t in (8.0, 16.0, 24.0):
                    p = centers[c] - d*t
                    node_pts.append(p.astype(np.float32)); node_kind.append("implicit_ray")
                    choices[c].append((len(node_pts)-1, t + 3.0))
    # Target-free support vote. A hypothesis earns support from distinct raw
    # ports within 13 px; choose max support, then minimum geometric cost.
    for c in range(len(centers)):
        best = None
        for n,cost in choices[c]:
            p=node_pts[n]; support=0
            for cc in range(len(centers)):
                if any(nn==n for nn,_ in choices[cc]) and float(np.linalg.norm(node_pts[n]-centers[cc])) <= 20.0:
                    support += 1
            key=(-support, float(cost), int(n))
            if best is None or key < best[0]: best=(key,n)
        choices[c]=[(best[1],0.)]
    # Convert every segment port via selected cluster hypothesis.
    selected=set()
    for si in range(len(seg)):
        a=int(labels[2*si]); b=int(labels[2*si+1]); na=choices[a][0][0]; nb=choices[b][0][0]
        if na!=nb: selected.add(tuple(sorted((na,nb))))
    points=np.asarray(node_pts, np.float32)
    layers=json.loads(Path(str(base)+".layers.json").read_text()); graph=json.loads(Path(str(base)+".graph.json").read_text())
    targets=np.asarray([a["center"] for a in layers["atoms"]], np.float32)
    gt={tuple(sorted((int(z["begin_atom"]),int(z["end_atom"])))) for z in graph["bonds"]}
    # Candidate nodes that are aliases of a text box are mapped by their image
    # centre; all ray/raw nodes use nearest target only for post-hoc scoring.
    c2t={}
    for i,p in enumerate(points):
        d=np.linalg.norm(targets-p,axis=1); close=np.where(d<=16.0)[0]; near=np.where(d<=24.0)[0]
        if len(close)==1 and len(near)==1: c2t[i]=int(close[0])
    mapped={tuple(sorted((c2t[a],c2t[b]))) for a,b in selected if a in c2t and b in c2t and c2t[a]!=c2t[b]}
    true=mapped&gt; dangling=sum(1 for a,b in selected if a not in c2t or b not in c2t)
    return Counter(images=1, clusters=len(centers), hypotheses=len(node_pts), selected_edges=len(selected), mapped_edges=len(mapped), true_edges=len(true), gt_bonds=len(gt), dangling_edges=dangling)

def _finish(c):
    d=dict(c); d.update(edge_precision=c['true_edges']/max(c['mapped_edges'],1), edge_recall=c['true_edges']/max(c['gt_bonds'],1), dangling_rate=c['dangling_edges']/max(c['selected_edges'],1)); return d

def main():
    rdBase.DisableLog("rdApp.error")
    # Resolve the registered production initial checkpoint; only its digest is
    # read (the checkpoint payload is never loaded by this probe).
    from chemvector.training.curriculum.chemical_head_probe import INITIAL
    if hashlib.sha256(INITIAL.read_bytes()).hexdigest()!=SHA: raise RuntimeError('production checkpoint changed')
    rows=[json.loads(x) for x in (DATA/'samples.jsonl').read_text().splitlines() if x.strip() and json.loads(x).get('render_status')=='accepted' and json.loads(x).get('split')=='fresh_holdout']
    rows=sorted(rows,key=lambda x:x['sample_key'])[:95]
    results={}
    for mode in ('ray_anchor','extended'):
        total=Counter(); prof=defaultdict(Counter)
        for r in rows:
            v=_propose(r,mode); total.update(v); prof[r['profile_id']].update(v)
        results[mode]={'all':_finish(total),'by_profile':{k:_finish(v) for k,v in sorted(prof.items())}}
    hold=results['extended']['all']
    report={'schema_version':'chemvector-identity-aligned-line-first-endpoint-hypothesis-r8-0.1','complete':True,'read_only':True,'training_performed':False,'gpu_used':False,'sealed_test_read':False,'production_checkpoint_written':False,'dataset':str(DATA),'split':'fresh_holdout','images':len(rows),'proposal_contract':'segment-port endpoint hypothesis set; text-seam ray-to-glyph interior + implicit-C backward ray; target-free support selection; GT post-hoc only','results':results,'initial_sha256':SHA,'initial_checkpoint_sha256_after':hashlib.sha256(INITIAL.read_bytes()).hexdigest(),'gate':{'checkpoint_unchanged':hashlib.sha256(INITIAL.read_bytes()).hexdigest()==SHA,'sealed_test_read':False,'training_admissible':False,'gpu_go':False,'reason':'deterministic ceiling only; scorer training requires all gates'}}
    OUT.mkdir(parents=True,exist_ok=False); (OUT/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n'); print(json.dumps(report,indent=2,ensure_ascii=False))

if __name__=='__main__': main()
