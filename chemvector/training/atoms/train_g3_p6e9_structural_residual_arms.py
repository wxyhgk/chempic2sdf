#!/usr/bin/env python3
"""Train and calibration-select three P6 structural residual arms."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from chemvector.components.recovery.structural_peak_residual import ARMS, StructuralPeakResidual, residual_selection_loss
from chemvector.training.atoms import train_g3_p5e9_generic_direction_mixed as p5e9
from chemvector.training.atoms import train_g3_p6e3_structural_peak_selector as p6e3
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import _atomic_json, _atomic_torch_save, module_state_sha256
from chemvector.g3_structural_data import file_sha256


CYCLE_NAME="chemvector-g3-p6e9-structural-residual-arms"
OUTPUT=Path(f"outputs/checkpoints/{CYCLE_NAME}")
SPLIT_ROOT=Path("outputs/audits/chemvector-g3-p6e8-residual-cycle-data/splits")
TRAIN_HARD=SPLIT_ROOT/"hard_train.pt"; TRAIN_EASY=SPLIT_ROOT/"easy_train.pt"
CALIBRATION_HARD=SPLIT_ROOT/"hard_calibration.pt"; CALIBRATION_EASY=SPLIT_ROOT/"easy_calibration.pt"
SHA256={TRAIN_HARD:"19b45252f37406e159c9fa09ddc0898a8e1e725ffbee04e75eae5f6d823e87ae",TRAIN_EASY:"13e5a58725c60ebd773b31b1c2957edb008f0f22fda6a4d374e5d5a6c1e4cb54",CALIBRATION_HARD:"84fb4c205773f5b6a01686cf5de8970fb27aadb4552d67de90179d084f64312c",CALIBRATION_EASY:"6452a042f3dcd4883f0a41de5f7ee89ebf4241761d20b98ef4f0252444ecc7ae"}
HOLDOUT_FILES=(SPLIT_ROOT/"hard_holdout.pt",SPLIT_ROOT/"easy_holdout.pt")
HARD_ACTIONS=5171; CALIBRATION_ACTIONS=643; EXPOSURES=192; HALF_BATCH=32; BATCH_SIZE=64; STEPS=HARD_ACTIONS*EXPOSURES//HALF_BATCH
CAP=2.0; LEARNING_RATE=3e-4; WEIGHT_DECAY=1e-4; MAX_GRAD_NORM=1.0; SEED=20260869; MODEL_SEED=2026086901


def _stratum(cache:dict,index:int)->tuple[str,str,str]: return (str(cache["variant"][index]),str(cache["target_role"][index]),str(cache["degree_bucket"][index]))


def matched_easy_rows(hard:dict,easy:dict,seed:int)->torch.Tensor:
    pools:dict[tuple[str,str,str],list[int]]=defaultdict(list)
    for index in range(len(easy["offset_targets"])): pools[_stratum(easy,index)].append(index)
    output=[]; cursors=defaultdict(int)
    for key,values in pools.items(): values.sort(key=lambda i:hashlib.sha256(f"{seed}:{easy['sample_key'][i]}:{easy['target_index'][i]}".encode()).hexdigest())
    for index in range(len(hard["offset_targets"])):
        key=_stratum(hard,index); values=pools.get(key,[])
        if not values: raise RuntimeError(f"P6-E9 lacks easy stratum: {key}")
        output.append(values[cursors[key]%len(values)]); cursors[key]+=1
    return torch.tensor(output,dtype=torch.long)


def _slice(cache:dict,rows:torch.Tensor)->dict:
    names=("image_crops","score_crops","candidate_offsets","candidate_peak_logits","candidate_mask","offset_targets")
    return {name:cache[name][rows] for name in names}


def _batch(cache:dict,rows:torch.Tensor,device:torch.device)->tuple:
    count=len(rows)
    return (cache["image_crops"][rows].to(device).float(),cache["score_crops"][rows].to(device).float(),cache["candidate_offsets"][rows].to(device).float(),cache["candidate_peak_logits"][rows].to(device).float(),cache["candidate_mask"][rows].to(device),torch.zeros(count,device=device),torch.zeros(count,device=device),cache["offset_targets"][rows].to(device).float())


def _train_step(model,optimizer,tensors:tuple,transforms:torch.Tensor)->dict[str,float]:
    transformed=p6e3._transform(tensors,transforms); images,fields,offsets,peak_logits,mask,source_scores,source_ranks,targets=transformed
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
        details=model.forward_details(images,fields,offsets,peak_logits,mask,source_scores,source_ranks)
        hard_mask=torch.zeros(len(images),dtype=torch.bool,device=images.device); hard_mask[:HALF_BATCH]=True
        losses=residual_selection_loss(details,offsets,mask,targets,hard_mask)
    losses["loss"].backward(); gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),MAX_GRAD_NORM)
    if not bool(torch.isfinite(gradient)): raise FloatingPointError("P6-E9 gradient is not finite")
    optimizer.step(); return {name:float(value.detach()) for name,value in losses.items()}|{"gradient_norm":float(gradient.detach())}


@torch.no_grad()
def evaluate(model,cache:dict,device:torch.device)->dict:
    distances=[]; prior_distances=[]; changed=harmful=rescue=0; model.eval()
    for start in range(0,len(cache["offset_targets"]),BATCH_SIZE):
        rows=torch.arange(start,min(start+BATCH_SIZE,len(cache["offset_targets"]))); batch=_batch(cache,rows,device)
        with torch.autocast(device_type="cuda",dtype=torch.bfloat16): details=model.forward_details(*batch[:-1])
        selected=details["scores"].argmax(dim=1); prior=details["prior"].argmax(dim=1); index=torch.arange(len(rows),device=device)
        current=torch.linalg.vector_norm((batch[2][index,selected]-batch[-1])*768,dim=1); base=torch.linalg.vector_norm((batch[2][index,prior]-batch[-1])*768,dim=1)
        changed+=int((selected!=prior).sum()); harmful+=int(((base<=6)&(current>6)).sum()); rescue+=int(((base>6)&(current<=6)).sum()); distances.extend(current.cpu().tolist()); prior_distances.extend(base.cpu().tolist())
    def metrics(values):
        array=np.asarray(values); return {"actions":len(values),"within_6px":int((array<=6).sum()),"recovery_rate_6px":float((array<=6).mean()),"mae_pixels":float(array.mean()),"median_pixels":float(np.median(array)),"p90_pixels":float(np.percentile(array,90))}
    return {"final":metrics(distances),"prior":metrics(prior_distances),"changed":changed,"rescue":rescue,"harmful":harmful,"prior_preservation_rate":1.0-changed/len(distances)}


def train_arm(arm:str,hard:dict,easy:dict,cal_hard:dict,cal_easy:dict,device:torch.device)->dict:
    model_type=f"chemvector-g3-p6e9-residual-{arm}-hard5171"; model_output=Path(f"outputs/checkpoints/{model_type}")
    if model_output.exists(): raise RuntimeError(f"P6-E9 model exists: {model_output}")
    torch.manual_seed(MODEL_SEED); torch.cuda.manual_seed_all(MODEL_SEED); model=StructuralPeakResidual(arm=arm,cap=CAP).to(device); initial_sha=module_state_sha256(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY); hard_schedule=p5e9._schedule(HARD_ACTIONS,STEPS,HALF_BATCH,SEED); easy_schedule=p5e9._schedule(HARD_ACTIONS,STEPS,HALF_BATCH,SEED+1); generator=torch.Generator().manual_seed(SEED+2)
    started=time.perf_counter(); sums=defaultdict(float); last={}
    for step,(hard_rows,easy_rows) in enumerate(zip(hard_schedule,easy_schedule,strict=True),start=1):
        left=_batch(hard,hard_rows,device); right=_batch(easy,easy_rows,device); tensors=tuple(torch.cat((a,b),dim=0) for a,b in zip(left,right,strict=True)); transforms=torch.randint(0,8,(BATCH_SIZE,),generator=generator).to(device); last=_train_step(model,optimizer,tensors,transforms)
        for name,value in last.items(): sums[name]+=value
        if step%2000==0 or step==STEPS: print(json.dumps({"arm":arm,"step":step,**last}),flush=True)
    calibration={"hard":evaluate(model,cal_hard,device),"easy":evaluate(model,cal_easy,device)}
    model_output.mkdir(parents=True); checkpoint=model_output/f"step-{STEPS:06d}.pt"; _atomic_torch_save(checkpoint,{"checkpoint_schema_version":"g3-p6e9-structural-residual-0.1","model_type":model_type,"model_config":{"arm":arm,"cap":CAP},"model":model.state_dict(),"optimizer":optimizer.state_dict(),"global_step":STEPS,"calibration":calibration})
    result={"model_type":model_type,"arm":arm,"checkpoint":str(checkpoint.resolve()),"checkpoint_sha256":file_sha256(checkpoint),"parameter_count":model.parameter_count,"initial_state_sha256":initial_sha,"final_state_sha256":module_state_sha256(model),"training":{"steps":STEPS,"exposures_per_hard":EXPOSURES,"mean":{name:value/STEPS for name,value in sums.items()},"last":last},"calibration":calibration,"elapsed_seconds":time.perf_counter()-started}; _atomic_json(model_output/"audit.json",{"status":"completed",**result}); return result


def main()->None:
    if OUTPUT.exists(): raise RuntimeError(f"P6-E9 cycle output exists: {OUTPUT}")
    for path,expected in SHA256.items():
        if file_sha256(path)!=expected: raise ValueError(f"P6-E9 cache SHA256 differs: {path}")
    if any((path.stat().st_mode&0o777)!=0 for path in HOLDOUT_FILES): raise RuntimeError("P6-E9 holdout cache is not sealed")
    if not torch.cuda.is_available(): raise RuntimeError("P6-E9 requires CUDA")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8"); random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED); torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark=False; torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; device=torch.device("cuda")
    hard=torch.load(TRAIN_HARD,map_location="cpu",weights_only=False); easy_all=torch.load(TRAIN_EASY,map_location="cpu",weights_only=False); cal_hard=torch.load(CALIBRATION_HARD,map_location="cpu",weights_only=False); cal_easy=torch.load(CALIBRATION_EASY,map_location="cpu",weights_only=False)
    easy=_slice(easy_all,matched_easy_rows(hard,easy_all,SEED+10)); results=[train_arm(arm,hard,easy,cal_hard,cal_easy,device) for arm in ARMS]
    eligible=[row for row in results if row["calibration"]["easy"]["harmful"]==0]
    winner=max(eligible,key=lambda row:(row["calibration"]["hard"]["final"]["recovery_rate_6px"],-row["calibration"]["hard"]["final"]["mae_pixels"],-row["parameter_count"])) if eligible else None
    checks={"three_arms_completed":len(results)==3,"winner_zero_easy_harmful":winner is not None,"holdout_files_mode_000":all((path.stat().st_mode&0o777)==0 for path in HOLDOUT_FILES)}
    decision="residual_calibration_winner_selected" if all(checks.values()) else "residual_calibration_no_winner"
    OUTPUT.mkdir(parents=True); final={"schema_version":"g3-p6e9-structural-residual-arms-final-0.1","decision":decision,"checks":checks,"winner":winner["model_type"] if winner else None,"arms":results,"protocol":{"cap":CAP,"hard_actions":HARD_ACTIONS,"exposures":EXPOSURES,"hard_easy_batch":"32:32","selection":["easy harmful=0","max hard recovery","min hard MAE","min parameters"],"holdout_read":False},"lineage":{"cache_sha256":{str(path):value for path,value in SHA256.items()}},"holdout_read":False,"dev_confirm_external_sealed_read":False}; _atomic_json(OUTPUT/"final_metrics.json",final); _atomic_json(OUTPUT/"audit.json",{"status":"completed",**final}); print(json.dumps(final,indent=2))


if __name__=="__main__": main()
