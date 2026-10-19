"""Target-free endpoint ceiling for the r6 rendered quarantine panel."""
from __future__ import annotations
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib, json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import audit_line_first_direct_vs_node_graph_strict_r5 as strict

DATA = ROOT / "outputs/datasets/chemvector-identity-aligned-line-first-targeted-diversity-r6"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-targeted-diversity-r6-endpoint-ceiling"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"

def job(row):
    strict.DATA = DATA
    result = strict._direct(row)
    result.update(sample_key=row["sample_key"], profile_id=row["profile_id"], split=row["split"], category=row["category"])
    return result

def finish(rows):
    fields=("images","pred_nodes","matched_nodes","target_atoms","selected_edges","mapped_edges","true_edges","gt_bonds","unresolved_edges","self_loops")
    c=Counter(); [c.update({f:int(r.get(f,0)) for f in fields}) for r in rows]
    return {**dict(c),"node_precision":c["matched_nodes"]/max(c["pred_nodes"],1),"node_recall":c["matched_nodes"]/max(c["target_atoms"],1),"edge_precision":c["true_edges"]/max(c["mapped_edges"],1),"edge_recall":c["true_edges"]/max(c["gt_bonds"],1),"dangling_rate":c["unresolved_edges"]/max(c["selected_edges"],1)}

def main(workers=16):
    if hashlib.sha256(strict.shadow.INITIAL.read_bytes()).hexdigest()!=INITIAL_SHA: raise RuntimeError("production checkpoint changed")
    rows=sorted([json.loads(x) for x in (DATA/"samples.jsonl").read_text().splitlines() if x.strip() and json.loads(x).get("render_status")=="accepted"],key=lambda r:r["sample_key"])
    result=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(job,row) for row in rows]
        for i,f in enumerate(as_completed(futures),1):
            result.append(f.result())
            if i%200==0: print(json.dumps({"completed":i,"total":len(rows)}),flush=True)
    result.sort(key=lambda r:r["sample_key"]); splits={}
    for split in sorted({r["split"] for r in result}):
        sr=[r for r in result if r["split"]==split]; profiles={p:finish([r for r in sr if r["profile_id"]==p]) for p in sorted({r["profile_id"] for r in sr})}; cats={c:finish([r for r in sr if r["category"]==c]) for c in sorted({r["category"] for r in sr})}; splits[split]={"all":finish(sr),"by_profile":profiles,"by_category":cats}
    hold=splits.get("fresh_holdout",{}).get("all",{}); hp=splits.get("fresh_holdout",{}).get("by_profile",{})
    gate={"checkpoint_unchanged":hashlib.sha256(strict.shadow.INITIAL.read_bytes()).hexdigest()==INITIAL_SHA,"sealed_test_read":False,"fresh_holdout_images":hold.get("images",0)>0,"proposal_precision_ge_0.965917":hold.get("edge_precision",0)>=.965917,"proposal_recall_gt_0.850031":hold.get("edge_recall",0)>.850031,"all_profile_recall_ge_0.75":all(x["edge_recall"]>=.75 for x in hp.values()),"dangling_le_0.05":hold.get("dangling_rate",1)<=.05,"training_admissible":False,"gpu_go":False}
    report={"schema_version":"chemvector-identity-aligned-line-first-targeted-diversity-r6-endpoint-ceiling-0.1","complete":True,"read_only":True,"training_performed":False,"gpu_used":False,"sealed_test_read":False,"production_checkpoint_written":False,"initial_sha256":INITIAL_SHA,"dataset":str(DATA),"rows":len(rows),"splits":splits,"proposal_contract":"new-LSD endpoint clusters + glyph aliases; GT joined post-hoc; synthetic SINGLE ceiling","gate":gate}
    OUT.mkdir(parents=True,exist_ok=True); (OUT/"report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n"); print(json.dumps({"output":str(OUT/"report.json"),"fresh_holdout":hold,"gate":gate},indent=2,ensure_ascii=False)); return report

if __name__=="__main__": main()
