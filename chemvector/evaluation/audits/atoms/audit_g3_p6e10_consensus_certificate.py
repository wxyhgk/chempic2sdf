#!/usr/bin/env python3
"""Calibrate a zero-harmful three-arm consensus certificate for P6 residuals."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from chemvector.components.recovery.structural_peak_residual import StructuralPeakResidual
from chemvector.evaluation.audits.atoms import build_g3_p4e9_recovery_oracle as oracle
from chemvector.training.atoms import train_g3_p6e9_structural_residual_arms as p6e9
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import _atomic_json
from chemvector.g3_structural_data import file_sha256


AUDIT_NAME="chemvector-g3-p6e10-consensus-certificate-calibration"
OUTPUT=Path(f"outputs/audits/{AUDIT_NAME}")
HARD=p6e9.CALIBRATION_HARD; EASY=p6e9.CALIBRATION_EASY
CHECKPOINTS={
    "global":(Path("outputs/checkpoints/chemvector-g3-p6e9-residual-global-hard5171/step-031026.pt"),"e44c7dc26fb2106d4a06ed9fa669e08f3331206c9e86a5b52f4d38a30e8e2c1c"),
    "local17":(Path("outputs/checkpoints/chemvector-g3-p6e9-residual-local17-hard5171/step-031026.pt"),"2bf5c8a84a41658fba01c1f4416ea5ce1d82c8114d49ccab763e69347e3f07da"),
    "multiscale":(Path("outputs/checkpoints/chemvector-g3-p6e9-residual-multiscale-hard5171/step-031026.pt"),"e10967bfff722788106adb2ad40632384ffd2605ca404738bde09a93ec6f37d2"),
}
MINIMUM_RESCUES=100; BATCH_SIZE=64


def calibrate_zero_harmful(records:list[dict])->dict:
    eligible=[row for row in records if row["consensus"] and row["changed"]]
    if not eligible: return {"threshold":math.inf,"executed":0,"rescues":0,"harmful":0,"useless":0,"eligible":0}
    maximum=max(float(row["certificate_score"]) for row in eligible)
    thresholds=[math.nextafter(maximum,math.inf)]+sorted({float(row["certificate_score"]) for row in eligible},reverse=True)
    candidates=[]
    for threshold in thresholds:
        selected=[row for row in eligible if float(row["certificate_score"])>=threshold]
        rescues=sum(row["domain"]=="hard" and row["selected_correct"] for row in selected)
        harmful=sum(row["domain"]=="easy" and not row["selected_correct"] for row in selected)
        candidates.append({"threshold":threshold,"executed":len(selected),"rescues":rescues,"harmful":harmful,"useless":len(selected)-rescues-harmful,"eligible":len(eligible)})
    safe=[row for row in candidates if row["harmful"]==0]
    return max(safe,key=lambda row:(row["rescues"],-row["executed"],row["threshold"]))


def _load_models(device:torch.device)->dict[str,StructuralPeakResidual]:
    models={}
    for arm,(path,expected) in CHECKPOINTS.items():
        if file_sha256(path)!=expected: raise ValueError(f"P6-E10 checkpoint SHA256 differs: {arm}")
        payload=torch.load(path,map_location="cpu",weights_only=False); model=StructuralPeakResidual(arm=arm,cap=2.0).to(device); model.load_state_dict(payload["model"],strict=True); model.eval(); models[arm]=model
    return models


def _batch(cache:dict,rows:torch.Tensor,device:torch.device)->tuple:
    count=len(rows); return (cache["image_crops"][rows].to(device).float(),cache["score_crops"][rows].to(device).float(),cache["candidate_offsets"][rows].to(device).float(),cache["candidate_peak_logits"][rows].to(device).float(),cache["candidate_mask"][rows].to(device),torch.zeros(count,device=device),torch.zeros(count,device=device))


@torch.no_grad()
def records_for_cache(models:dict[str,StructuralPeakResidual],cache:dict,domain:str,device:torch.device)->list[dict]:
    records=[]
    for start in range(0,len(cache["offset_targets"]),BATCH_SIZE):
        rows=torch.arange(start,min(start+BATCH_SIZE,len(cache["offset_targets"]))); batch=_batch(cache,rows,device); outputs={}
        for arm,model in models.items():
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16): outputs[arm]=model.forward_details(*batch)
        selections={arm:value["scores"].argmax(dim=1) for arm,value in outputs.items()}; prior=outputs["global"]["prior"].argmax(dim=1); primary=selections["global"]; index=torch.arange(len(rows),device=device)
        target=cache["offset_targets"][rows].to(device).float(); distance=torch.linalg.vector_norm((batch[2][index,primary]-target)*768,dim=1); score=outputs["global"]["scores"][index,primary]-outputs["global"]["scores"][index,prior]
        consensus=(selections["global"]==selections["local17"])&(selections["global"]==selections["multiscale"])
        for offset,row in enumerate(rows.tolist()):
            records.append({"domain":domain,"sample_key":str(cache["sample_key"][row]),"target_index":int(cache["target_index"][row]),"prior_index":int(prior[offset]),"selected_index":int(primary[offset]),"local17_index":int(selections["local17"][offset]),"multiscale_index":int(selections["multiscale"][offset]),"changed":bool(primary[offset]!=prior[offset]),"consensus":bool(consensus[offset]),"certificate_score":float(score[offset]),"selected_distance_pixels":float(distance[offset]),"selected_correct":bool(distance[offset]<=6.0)})
    return records


def main()->None:
    if OUTPUT.exists(): raise RuntimeError(f"P6-E10 output exists: {OUTPUT}")
    for path,expected in p6e9.SHA256.items():
        if path in (HARD,EASY) and file_sha256(path)!=expected: raise ValueError(f"P6-E10 cache SHA256 differs: {path}")
    if any((path.stat().st_mode&0o777)!=0 for path in p6e9.HOLDOUT_FILES): raise RuntimeError("P6-E10 holdout is not sealed")
    if not torch.cuda.is_available(): raise RuntimeError("P6-E10 requires CUDA")
    device=torch.device("cuda"); models=_load_models(device); hard=torch.load(HARD,map_location="cpu",weights_only=False); easy=torch.load(EASY,map_location="cpu",weights_only=False)
    records=records_for_cache(models,hard,"hard",device)+records_for_cache(models,easy,"easy",device); winner=calibrate_zero_harmful(records)
    checks={"calibration_zero_harmful":winner["harmful"]==0,"calibration_rescues_min100":winner["rescues"]>=MINIMUM_RESCUES,"holdout_mode_000":all((path.stat().st_mode&0o777)==0 for path in p6e9.HOLDOUT_FILES)}
    decision="consensus_certificate_calibration_go" if all(checks.values()) else "consensus_certificate_calibration_no_go"
    OUTPUT.mkdir(parents=True); oracle._atomic_jsonl(OUTPUT/"records.jsonl",records); audit={"schema_version":"g3-p6e10-consensus-certificate-calibration-0.1","decision":decision,"checks":checks,"winner":winner,"capacity":{"hard":len(hard["offset_targets"]),"easy":len(easy["offset_targets"]),"consensus_changed":sum(row["consensus"] and row["changed"] for row in records)},"protocol":{"primary":"global","certificate":"three-arm same non-prior candidate","score":"global final selected score minus prior-winner final score","threshold":"zero harmful, max rescues, min executions, highest threshold","minimum_rescues":MINIMUM_RESCUES,"holdout_read":False},"lineage":{"checkpoints":{arm:sha for arm,(_,sha) in CHECKPOINTS.items()},"hard_cache_sha256":p6e9.SHA256[HARD],"easy_cache_sha256":p6e9.SHA256[EASY]},"holdout_read":False,"dev_confirm_external_sealed_read":False}; _atomic_json(OUTPUT/"audit.json",audit); print(json.dumps(audit,indent=2))


if __name__=="__main__": main()
