"""Frozen-initial replay for the cap/stroke isolation panel.

Diagnostic only.  The profile gate is reported per cell; a failing base cell
does not authorize training and is not hidden by aggregate reachability.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib, json, os
from pathlib import Path
import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.data import CurriculumCollator, load_sample
from chemvector.training.curriculum.gradient_preflight import state_digest
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from chemvector.training.graph.relations.train_direct_atom_pair_bond import match_atom_targets

ROOT = Path(__file__).resolve().parents[1]
PLAN_DIR = Path(os.environ.get("CHEMVECTOR_CAP_STROKE_PLAN_DIR", str(ROOT / "outputs/audits/chemvector-oa13-oa33-cap-stroke-panel-plan-20261017-r1")))
DATA = Path(os.environ.get("CHEMVECTOR_CAP_STROKE_DATA", str(ROOT / "outputs/datasets/chemvector-oa13-oa33-cap-stroke-panel-20261017-r1")))
OUT = Path(os.environ.get("CHEMVECTOR_CAP_STROKE_REPLAY_OUTPUT", str(ROOT / "outputs/audits/chemvector-oa13-oa33-cap-stroke-frozen-replay-20261017-r1/report.json")))
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"

def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20), b""): h.update(chunk)
    return h.hexdigest()

def ratios(c: Counter) -> dict:
    out=dict(c); out["raw_reachability"]=c["raw_matched"]/max(c["target_atoms"],1); out["selected_reachability"]=c["selected_matched"]/max(c["target_atoms"],1); out["selector_drop_rate_of_raw"]=c["selector_drop"]/max(c["raw_matched"],1); return out

def run() -> dict:
    if OUT.exists(): raise FileExistsError(OUT)
    plan=json.loads((PLAN_DIR/"manifest.json").read_text()); rows=plan["rows"]
    summary=json.loads((DATA/"summary.json").read_text())
    if len(rows)!=480 or summary.get("accepted_images")!=480 or summary.get("rejected_images")!=0: raise ValueError("quality panel gate failed")
    model,config,payload=load_bundle_for_probe(INITIAL,INITIAL_SHA)
    if digest(INITIAL)!=INITIAL_SHA or not torch.cuda.is_available(): raise RuntimeError("initial/CUDA binding failed")
    dev=torch.device("cuda:0"); model.to(dev).eval().requires_grad_(False); before=state_digest(model)
    collator=CurriculumCollator(AtomBondSceneConfig(atom_queries=config.maximum_atoms,bond_queries=config.maximum_bonds),purpose="diagnostic")
    total=Counter(); by=defaultdict(Counter); split=defaultdict(Counter)
    for row in rows:
        sample=load_sample(DATA,row["sample_key"],purpose="diagnostic"); batch=collator([sample]); tensors={k:v.to(dev) for k,v in batch.items() if isinstance(v,torch.Tensor)}
        with torch.inference_mode(): _,_,_,raw,_,_,selected=structure_forward(model,tensors["pixel_values"],config,payload["provenance"]["atom_threshold"])
        raw_map,_=match_atom_targets(raw,tensors,config); selected_map,_=match_atom_targets(selected,tensors,config)
        n=int(tensors["atom_mask"][0].sum()); atom_ids=batch["atom_ids"][0]; target=set(range(n)); raw_ids={int(v) for v in raw_map[0].tolist() if 0<=int(v)<n}; selected_ids={int(v) for v in selected_map[0].tolist() if 0<=int(v)<n}
        vals={"target_atoms":n,"raw_matched":len(raw_ids&target),"selected_matched":len(selected_ids&target),"images":1,"raw_gap":len(target-raw_ids),"selector_drop":len(raw_ids-selected_ids)}
        for k,v in vals.items(): total[k]+=v; by[row["profile_id"]][k]+=v; split[row["split"]][k]+=v
    after=state_digest(model)
    if before!=after or any(p.grad is not None for p in model.parameters()): raise RuntimeError("frozen state changed")
    by_profile={k:ratios(v) for k,v in sorted(by.items())}; by_split={k:ratios(v) for k,v in sorted(split.items())}; rt=ratios(total)
    gate={"initial_unchanged":before==after,"renderer_quality_pass":True,"each_profile_raw_reachability_ge_098":all(v["raw_reachability"]>=.98 for v in by_profile.values()),"any_control_raw_reachability_ge_098":any(v["raw_reachability"]>=.98 for k,v in by_profile.items() if k!="cov13_base"),"training_admissible":False,"next":"admit only a passing control family; no training until release/admission and sidecar receipt"}
    report={"schema_version":"chemvector-oa13-oa33-cap-stroke-frozen-replay-0.1","complete":True,"read_only":True,"training_performed":False,"optimizer_steps":0,"gpu_used":True,"device":str(dev),"sealed_test_read":False,"deployment_authorized":False,"initial_sha256":INITIAL_SHA,"initial_sha256_after":digest(INITIAL),"panel_manifest_sha256":digest(PLAN_DIR/"manifest.json"),"panel_summary_sha256":digest(DATA/"summary.json"),"state_before":before,"state_after":after,"totals":rt,"by_profile":by_profile,"by_split":by_split,"gate":gate}
    OUT.parent.mkdir(parents=True,exist_ok=False); OUT.write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps({"output":str(OUT),"totals":rt,"by_profile":by_profile,"gate":gate},indent=2)); return report
if __name__=="__main__": run()
