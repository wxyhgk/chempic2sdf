"""Target-free glyph-assignment radius sweep on r6.

This isolates a cheap endpoint ownership change: an LSD endpoint cluster can
be aliased to a glyph center when its distance is below a train-selected
radius.  The radius is selected on train/dev only; fresh_holdout is evaluated
once per selected radius.  No model or production artifact is modified.
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
from tools.audit_line_first_edge_bond_candidate_integration_r5 import strict_candidate_to_target
from chemvector.components.vision.line_segment_rescue import LineSegmentRescueConfig

DATA=ROOT/"outputs/datasets/chemvector-identity-aligned-line-first-targeted-diversity-r6"; OUT=ROOT/"outputs/audits/chemvector-identity-aligned-line-first-targeted-diversity-r6-glyph-alias-sweep"; INITIAL_SHA="09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"

def one(row, radius):
    base=DATA/"samples"/row["sample_key"]; image=Image.open(str(base)+".png").convert("L"); seg=lsd._alt_detect(image,config=LineSegmentRescueConfig(maximum_candidates=512)); pts=seg.reshape(-1,2) if len(seg) else np.empty((0,2),np.float32); labels,centers=cluster_endpoints(pts,10.); boxes=strict._boxes(base); alias={i:i for i in range(len(centers))}
    for aid,b in boxes.items():
        c=np.asarray([(b[0]+b[2])/2,(b[1]+b[3])/2],float); hits=[(float(np.linalg.norm(centers[i]-c)),i) for i in range(len(centers)) if np.linalg.norm(centers[i]-c)<=radius]
        if hits:
            for _,i in sorted(hits): alias[i]=-(aid+1)
    ids=sorted(set(alias.values())); idx={n:i for i,n in enumerate(ids)}; pred=[]
    for n in ids:
        if n<0:
            b=boxes[-n-1]; pred.append([(b[0]+b[2])/2,(b[1]+b[3])/2])
        else: pred.append(centers[n])
    targets,gt=strict._target(base); c2t=strict_candidate_to_target(np.asarray(pred,np.float32),targets); selected=set()
    for i in range(len(seg)):
        a,b=alias.get(int(labels[2*i]),int(labels[2*i])),alias.get(int(labels[2*i+1]),int(labels[2*i+1]))
        if a!=b: selected.add(tuple(sorted((idx[a],idx[b]))))
    mapped=set(); unresolved=0; loops=0
    for a,b in selected:
        if a not in c2t or b not in c2t: unresolved+=1; continue
        x,y=c2t[a],c2t[b]
        if x==y: loops+=1; continue
        mapped.add(tuple(sorted((x,y))))
    true=mapped & gt
    return {"images":1,"pred_nodes":len(pred),"matched_nodes":len(c2t),"target_atoms":len(targets),"selected_edges":len(selected),"mapped_edges":len(mapped),"true_edges":len(true),"gt_bonds":len(gt),"unresolved_edges":unresolved,"self_loops":loops,"profile_id":row["profile_id"],"split":row["split"]}

def finish(rows):
    fields=("images","pred_nodes","matched_nodes","target_atoms","selected_edges","mapped_edges","true_edges","gt_bonds","unresolved_edges","self_loops"); c=Counter(); [c.update({f:int(r[f]) for f in fields}) for r in rows]; return {**dict(c),"node_precision":c["matched_nodes"]/max(c["pred_nodes"],1),"node_recall":c["matched_nodes"]/max(c["target_atoms"],1),"edge_precision":c["true_edges"]/max(c["mapped_edges"],1),"edge_recall":c["true_edges"]/max(c["gt_bonds"],1),"dangling_rate":c["unresolved_edges"]/max(c["selected_edges"],1)}

def main():
    if hashlib.sha256(strict.shadow.INITIAL.read_bytes()).hexdigest()!=INITIAL_SHA: raise RuntimeError("checkpoint changed")
    rows=[json.loads(x) for x in (DATA/"samples.jsonl").read_text().splitlines() if x.strip() and json.loads(x).get("render_status")=="accepted"]; rows.sort(key=lambda r:r["sample_key"]); results={}
    for radius in (16.,20.,24.,28.,32.,40.,48.):
        values=[one(r,radius) for r in rows]; results[str(radius)]={s:finish([v for v in values if v["split"]==s]) for s in ("train","dev","fresh_holdout")}
        print(json.dumps({"radius":radius,"dev":results[str(radius)]["dev"]}),flush=True)
    eligible=[(v["dev"]["edge_recall"],-float(k),k) for k,v in results.items() if v["dev"]["edge_precision"]>=.95]; selected=max(eligible)[2] if eligible else None
    report={"schema_version":"chemvector-identity-aligned-line-first-targeted-diversity-r6-glyph-alias-sweep-0.1","complete":True,"read_only":True,"training_performed":False,"gpu_used":False,"sealed_test_read":False,"production_checkpoint_written":False,"initial_sha256":INITIAL_SHA,"radii_px":[16,20,24,28,32,40,48],"selection_source":"dev only, precision>=.95 then recall","selected_radius_px":None if selected is None else float(selected),"results":results,"gate":{"checkpoint_unchanged":True,"sealed_test_read":False,"selected_exists":selected is not None,"gpu_go":False,"production_go":False}}
    if selected is not None:
        hold=results[selected]["fresh_holdout"]; report["gate"].update({"holdout_precision":hold["edge_precision"],"holdout_recall":hold["edge_recall"],"holdout_dangling":hold["dangling_rate"],"edge_gate":hold["edge_precision"]>=.965917 and hold["edge_recall"]>.850031 and hold["dangling_rate"]<=.05})
    OUT.mkdir(parents=True,exist_ok=True); (OUT/"report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n"); print(json.dumps({"selected":selected,"gate":report["gate"]},indent=2,ensure_ascii=False)); return report

if __name__=="__main__": main()
