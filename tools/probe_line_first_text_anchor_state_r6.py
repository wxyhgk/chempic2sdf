"""Deterministic text-anchor ownership state-machine probe on r6 holdout."""
from __future__ import annotations
from collections import Counter
import hashlib, json, math
from pathlib import Path
import sys
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from tools import audit_line_first_direct_vs_node_graph_strict_r5 as strict
from tools import audit_line_first_lsd_all_edge_ownership_r5 as lsd
from tools.audit_line_first_graph_traversal_r5 import cluster_endpoints
from tools.audit_line_first_edge_bond_candidate_integration_r5 import strict_candidate_to_target
from chemvector.components.vision.line_segment_rescue import LineSegmentRescueConfig
DATA=ROOT/"outputs/datasets/chemvector-identity-aligned-line-first-targeted-diversity-r6"; OUT=ROOT/"outputs/audits/chemvector-identity-aligned-line-first-text-anchor-state-r6"; INITIAL_SHA="09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"

def boxdist(p,b):
    return math.hypot(max(float(b[0]-p[0]),0.,float(p[0]-b[2])),max(float(b[1]-p[1]),0.,float(p[1]-b[3])))

def one(row,margin,cos_gate):
    base=DATA/"samples"/row["sample_key"]; img=Image.open(str(base)+".png").convert("L"); seg=lsd._alt_detect(img,config=LineSegmentRescueConfig(maximum_candidates=512)); raw=seg.reshape(-1,2) if len(seg) else np.empty((0,2),np.float32); labels,centers=cluster_endpoints(raw,10.); boxes=[]
    for g in json.loads(Path(str(base)+".text-records.json").read_text()).get("glyphs",[]):
        b=g.get("bbox");
        if b: boxes.append(np.asarray(b,np.float32))
    # One state per glyph box; route only when a segment seam is geometrically
    # oriented toward that box.  No text identity or GT coordinates are used.
    glyph_nodes=[np.asarray([(b[0]+b[2])/2.,(b[1]+b[3])/2.],np.float32) for b in boxes]; points=[np.asarray(c,np.float32) for c in centers]; gstart=len(points); points.extend(glyph_nodes); selected=set(); unresolved=0
    for i,(p,q) in enumerate(seg):
        def route(e,other,ci):
            v=np.asarray(other-e,np.float32); nv=float(np.linalg.norm(v)); best=None
            for gi,b in enumerate(boxes):
                c=glyph_nodes[gi]; w=c-np.asarray(e,np.float32); nw=float(np.linalg.norm(w));
                if nw<1e-3 or nv<1e-3: continue
                cosine=float(np.dot(-v,w)/(nv*nw)); d=boxdist(np.asarray(e),b)
                if d<=margin and cosine>=cos_gate: best=(d,-cosine,gi)
            if best is not None: return gstart+best[2]
            return int(ci)
        a=route(p,q,int(labels[2*i])); b=route(q,p,int(labels[2*i+1]));
        if a==b: continue
        selected.add(tuple(sorted((a,b))))
    targets,gt=strict._target(base); c2t=strict_candidate_to_target(np.asarray(points,np.float32),targets); mapped=set(); dang=0
    for a,b in selected:
        if a not in c2t or b not in c2t: dang+=1; continue
        x,y=c2t[a],c2t[b]
        if x!=y: mapped.add(tuple(sorted((x,y))))
    true=mapped&gt
    return {"images":1,"pred_nodes":len(points),"matched_nodes":len(c2t),"target_atoms":len(targets),"selected_edges":len(selected),"mapped_edges":len(mapped),"true_edges":len(true),"gt_bonds":len(gt),"unresolved_edges":dang,"profile_id":row["profile_id"],"split":row["split"]}

def fin(rows):
    f=("images","pred_nodes","matched_nodes","target_atoms","selected_edges","mapped_edges","true_edges","gt_bonds","unresolved_edges"); c=Counter(); [c.update({k:int(r[k]) for k in f}) for r in rows]; return {**dict(c),"node_precision":c["matched_nodes"]/max(c["pred_nodes"],1),"node_recall":c["matched_nodes"]/max(c["target_atoms"],1),"edge_precision":c["true_edges"]/max(c["mapped_edges"],1),"edge_recall":c["true_edges"]/max(c["gt_bonds"],1),"dangling_rate":c["unresolved_edges"]/max(c["selected_edges"],1)}

def main():
    if hashlib.sha256(strict.shadow.INITIAL.read_bytes()).hexdigest()!=INITIAL_SHA: raise RuntimeError("checkpoint changed")
    rows=[json.loads(x) for x in (DATA/"samples.jsonl").read_text().splitlines() if x.strip() and json.loads(x).get("render_status")=="accepted" and json.loads(x).get("split")=="fresh_holdout"]; rows.sort(key=lambda r:r["sample_key"]); variants={}
    for margin in (6.,10.,14.,18.):
        for cosine in (.55,.70,.85):
            vals=[one(r,margin,cosine) for r in rows]; variants[f"m{margin:g}_c{cosine:g}"]={"all":fin(vals),"by_profile":{p:fin([v for v in vals if v["profile_id"]==p]) for p in sorted({v["profile_id"] for v in vals})}}
            print(json.dumps({"variant":f"m{margin:g}_c{cosine:g}","all":variants[f"m{margin:g}_c{cosine:g}"]["all"]}),flush=True)
    report={"schema_version":"chemvector-identity-aligned-line-first-text-anchor-state-r6-0.1","complete":True,"read_only":True,"training_performed":False,"gpu_used":False,"sealed_test_read":False,"production_checkpoint_written":False,"initial_sha256":INITIAL_SHA,"proposal_contract":"target-free glyph bbox seam + segment direction state machine; GT post-hoc only","variants":variants,"gate":{"checkpoint_unchanged":True,"sealed_test_read":False,"gpu_go":False,"production_go":False}}
    OUT.mkdir(parents=True,exist_ok=True); (OUT/"report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n"); print(json.dumps({"output":str(OUT/"report.json"),"variants":len(variants)},indent=2)); return report

if __name__=="__main__": main()
