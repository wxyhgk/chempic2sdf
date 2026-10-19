"""CPU deterministic implicit-C junction refinement ceiling on r6."""
from __future__ import annotations
from collections import Counter
import hashlib, json
from pathlib import Path
import sys
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from tools import audit_line_first_lsd_all_edge_ownership_r5 as lsd
from tools.audit_line_first_graph_traversal_r5 import cluster_endpoints
from tools.audit_line_first_edge_bond_candidate_integration_r5 import strict_candidate_to_target
from chemvector.components.vision.line_segment_rescue import LineSegmentRescueConfig
from tools import audit_line_first_direct_vs_node_graph_strict_r5 as strict

DATA=ROOT/"outputs/datasets/chemvector-identity-aligned-line-first-targeted-diversity-r6"; OUT=ROOT/"outputs/audits/chemvector-identity-aligned-line-first-implicit-junction-refinement-r6"; INITIAL_SHA="09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"

def refine(c, segs, labels, beta, guard=30.):
    ids=np.where(labels>=0)[0]; out=np.asarray(c,np.float32).copy()
    for ci in sorted(set(labels.tolist())):
        incident=[]
        for i,(p,q) in enumerate(segs):
            for side in (0,1):
                if int(labels[2*i+side])==ci:
                    e=np.asarray([p,q][side],np.float64); other=np.asarray([q,p][side],np.float64); d=other-e; n=np.linalg.norm(d)
                    if n>1e-3: incident.append((e,d/n))
        if len(incident)<2: continue
        # Require direction diversity to avoid moving a straight-line endpoint.
        if max(float(np.degrees(np.arccos(np.clip(abs(np.dot(a[1],b[1])),-1,1)))) for i,a in enumerate(incident) for b in incident[i+1:])<20.: continue
        A=np.zeros((2,2)); rhs=np.zeros(2); center=np.asarray(c[ci],np.float64)
        for p,u in incident:
            P=np.eye(2)-np.outer(u,u); A+=P; rhs+=P@p
        A+=beta*np.eye(2); rhs+=beta*center
        x=np.linalg.solve(A,rhs)
        if np.linalg.norm(x-center)<=guard: out[ci]=x
    return out

def one(row,beta):
    base=DATA/"samples"/row["sample_key"]; seg=lsd._alt_detect(Image.open(str(base)+".png").convert("L"),config=LineSegmentRescueConfig(maximum_candidates=512)); ep=seg.reshape(-1,2) if len(seg) else np.empty((0,2),np.float32); labels,c=cluster_endpoints(ep,10.); c=refine(c,seg,labels,beta); targets,gt=strict._target(base); mapping=strict_candidate_to_target(c,targets); selected=set();
    for i in range(len(seg)):
        a,b=int(labels[2*i]),int(labels[2*i+1]);
        if a!=b: selected.add(tuple(sorted((a,b))))
    mapped=set(); dang=0
    for a,b in selected:
        if a not in mapping or b not in mapping: dang+=1; continue
        x,y=mapping[a],mapping[b]
        if x!=y: mapped.add(tuple(sorted((x,y))))
    true=mapped&gt
    return {"images":1,"nodes":len(c),"matched":len(mapping),"targets":len(targets),"selected":len(selected),"mapped":len(mapped),"true":len(true),"gt":len(gt),"dangling":dang,"profile":row["profile_id"],"split":row["split"]}

def finish(rows):
    keys=("images","nodes","matched","targets","selected","mapped","true","gt","dangling"); c=Counter(); [c.update({k:int(r[k]) for k in keys}) for r in rows]; return {**dict(c),"node_recall":c["matched"]/max(c["targets"],1),"edge_precision":c["true"]/max(c["mapped"],1),"edge_recall":c["true"]/max(c["gt"],1),"dangling_rate":c["dangling"]/max(c["selected"],1)}

def main():
    if hashlib.sha256(strict.shadow.INITIAL.read_bytes()).hexdigest()!=INITIAL_SHA: raise RuntimeError("checkpoint changed")
    rows=[json.loads(x) for x in (DATA/"samples.jsonl").read_text().splitlines() if x.strip() and json.loads(x).get("render_status")=="accepted" and json.loads(x).get("split")=="fresh_holdout"]; rows.sort(key=lambda r:r["sample_key"]); results={}
    for beta in (.1,.3,1.,3.):
        vals=[one(r,beta) for r in rows]; results[str(beta)]={"all":finish(vals),"by_profile":{p:finish([v for v in vals if v["profile"]==p]) for p in sorted({v["profile"] for v in vals})}}; print(json.dumps({"beta":beta,"all":results[str(beta)]["all"]}),flush=True)
    report={"schema_version":"chemvector-identity-aligned-line-first-implicit-junction-refinement-r6-0.1","complete":True,"read_only":True,"training_performed":False,"gpu_used":False,"sealed_test_read":False,"production_checkpoint_written":False,"initial_sha256":INITIAL_SHA,"proposal_contract":"frozen LSD clusters; target-free multi-incident line least-squares relocation only; GT post-hoc","results":results,"gate":{"checkpoint_unchanged":True,"sealed_test_read":False,"gpu_go":False,"production_go":False}}
    OUT.mkdir(parents=True,exist_ok=True); (OUT/"report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n"); print(json.dumps({"output":str(OUT/"report.json")},indent=2)); return report

if __name__=="__main__": main()
