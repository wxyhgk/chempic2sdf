"""Cheap target-free candidate coverage probe for endpoint extrapolation.

Adds only image-derived glyph centers and outward offsets of LSD segment
endpoints.  It measures a proposal *ceiling* on r6 fresh_holdout; no graph
edges, model, checkpoint, or sealed split is touched.
"""
from __future__ import annotations
from collections import Counter
import hashlib, json
from pathlib import Path
import sys
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from tools import audit_line_first_direct_vs_node_graph_strict_r5 as strict
from tools import audit_line_first_lsd_all_edge_ownership_r5 as lsd
from tools.audit_line_first_graph_traversal_r5 import cluster_endpoints
from chemvector.components.vision.line_segment_rescue import LineSegmentRescueConfig

DATA=ROOT/"outputs/datasets/chemvector-identity-aligned-line-first-targeted-diversity-r6"; OUT=ROOT/"outputs/audits/chemvector-identity-aligned-line-first-targeted-diversity-r6-geometry-offset-ceiling-v2"; INITIAL_SHA="09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"

def candidates(row, offsets):
    base=DATA/"samples"/row["sample_key"]; img=Image.open(str(base)+".png").convert("L"); seg=lsd._alt_detect(img,config=LineSegmentRescueConfig(maximum_candidates=512)); raw=seg.reshape(-1,2) if len(seg) else np.empty((0,2),np.float32); _,centers=cluster_endpoints(raw,10.); pts=[np.asarray(x,np.float32) for x in centers]
    for p,q in seg:
        d=p-q; n=float(np.linalg.norm(d));
        if n<1e-3: continue
        u=d/n
        for off in offsets: pts.append(np.asarray(p+u*float(off),np.float32))
    for g in json.loads(Path(str(base)+".text-records.json").read_text()).get("glyphs",[]):
        b=g.get("bbox");
        if b: pts.append(np.asarray([(float(b[0])+float(b[2]))/2,(float(b[1])+float(b[3]))/2],np.float32))
    # Target-free radius-3 deduplication.
    out=[]
    for p in pts:
        if not out or min(float(np.linalg.norm(p-q)) for q in out)>3.0: out.append(p)
    return np.asarray(out,np.float32), base

def evaluate(row, offsets):
    cp,base=candidates(row,offsets); targets,gt=strict._target(base); d=np.linalg.norm(cp[:,None]-targets[None],axis=-1) if len(cp) and len(targets) else np.empty((len(cp),len(targets))); owned=d.min(axis=0)<=6.0 if len(d) else np.zeros(len(targets),bool); close=d<=6.0 if len(d) else np.zeros((len(cp),len(targets)),bool); unique=np.sum(close,axis=1)==1 if len(close) else np.zeros(len(cp),bool); precision=float(np.sum(unique)/max(len(cp),1)); bond_cov=sum(bool(owned[a] and owned[b]) for a,b in gt); one_cov=sum(bool(owned[a] or owned[b]) for a,b in gt); return {"images":1,"candidates":len(cp),"targets":len(targets),"owned":int(owned.sum()),"candidate_precision":precision,"atom_recall":float(owned.mean()) if len(owned) else 0.,"gt_bonds":len(gt),"both_endpoint_bond_recall":bond_cov/max(len(gt),1),"one_endpoint_bond_recall":one_cov/max(len(gt),1),"profile_id":row["profile_id"],"split":row["split"]}

def finish(rows):
    n=sum(r["images"] for r in rows); targets=sum(r["targets"] for r in rows); owned=sum(r["owned"] for r in rows); bonds=sum(r["gt_bonds"] for r in rows); both=sum(r["both_endpoint_bond_recall"]*r["gt_bonds"] for r in rows); one=sum(r["one_endpoint_bond_recall"]*r["gt_bonds"] for r in rows); candidates=sum(r["candidates"] for r in rows); return {"images":n,"candidates":candidates,"targets":targets,"owned":owned,"atom_recall":owned/max(targets,1),"candidate_precision":sum(r["candidate_precision"]*r["candidates"] for r in rows)/max(candidates,1),"both_endpoint_bond_recall":both/max(bonds,1),"one_endpoint_bond_recall":one/max(bonds,1),"gt_bonds":bonds}

def main():
    if hashlib.sha256(strict.shadow.INITIAL.read_bytes()).hexdigest()!=INITIAL_SHA: raise RuntimeError("checkpoint changed")
    rows=[json.loads(x) for x in (DATA/"samples.jsonl").read_text().splitlines() if x.strip() and json.loads(x).get("render_status")=="accepted"]; rows=[r for r in rows if r["split"]=="fresh_holdout"]; rows.sort(key=lambda r:r["sample_key"])
    variants={"none":(),"short":(4,8,12),"wide":(4,8,12,16,20),"long":(4,8,12,16,20,28,36)}; results={}
    for name,offs in variants.items():
        vals=[evaluate(r,offs) for r in rows]; results[name]={"all":finish(vals),"by_profile":{p:finish([v for v in vals if v["profile_id"]==p]) for p in sorted({v["profile_id"] for v in vals})}}; print(json.dumps({"variant":name,"all":results[name]["all"]}),flush=True)
    report={"schema_version":"chemvector-identity-aligned-line-first-targeted-diversity-r6-geometry-offset-ceiling-0.1","complete":True,"read_only":True,"training_performed":False,"gpu_used":False,"sealed_test_read":False,"production_checkpoint_written":False,"initial_sha256":INITIAL_SHA,"split":"fresh_holdout","rows":len(rows),"variants":results,"gate":{"checkpoint_unchanged":True,"sealed_test_read":False,"proposal_only":True,"gpu_go":False,"production_go":False}}
    OUT.mkdir(parents=True,exist_ok=True); (OUT/"report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n"); return report

if __name__=="__main__": main()
