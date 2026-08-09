#!/usr/bin/env python3
"""Train a scaffold-OOF safety model for P6 residual overrides."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from chemvector.evaluation.audits.atoms import audit_g3_p6e10_consensus_certificate as p6e10
from chemvector.training.atoms import train_g3_p5e9_generic_direction_mixed as p5e9
from chemvector.training.atoms import train_g3_p6e9_structural_residual_arms as p6e9
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import _atomic_json, _atomic_torch_save, module_state_sha256
from chemvector.g3_structural_data import file_sha256


MODEL_TYPE="chemvector-g3-p6e11-oof-residual-safety"
OUTPUT=Path(f"outputs/checkpoints/{MODEL_TYPE}")
CHECKPOINT=OUTPUT/"step-002000.pt"
TRAIN_HARD=p6e9.TRAIN_HARD; TRAIN_EASY=p6e9.TRAIN_EASY; CAL_HARD=p6e9.CALIBRATION_HARD; CAL_EASY=p6e9.CALIBRATION_EASY
FOLDS=5; STEPS=2000; BATCH_SIZE=256; LEARNING_RATE=3e-4; MINIMUM_RESCUES=100; SEED=20260870


class ResidualSafetyMLP(nn.Module):
    def __init__(self,features:int)->None:
        super().__init__(); self.layers=nn.Sequential(nn.Linear(features,32),nn.GELU(),nn.Linear(32,16),nn.GELU(),nn.Linear(16,1))
    def forward(self,values:torch.Tensor)->torch.Tensor:return self.layers(values.float())[:,0]


def scaffold_fold(scaffold:str)->int:return int.from_bytes(hashlib.sha256(f"{SEED}:{scaffold}".encode()).digest()[:8],"big")%FOLDS


def _top_margin(scores:torch.Tensor)->torch.Tensor:
    top=torch.topk(scores.float(),k=2,dim=1).values; return top[:,0]-top[:,1]


def _entropy(scores:torch.Tensor)->torch.Tensor:
    probabilities=torch.softmax(scores.float(),dim=1); return -(probabilities*torch.log(probabilities.clamp_min(1e-8))).sum(dim=1)


@torch.no_grad()
def extract_features(models,cache:dict,domain:str,device:torch.device)->dict:
    features=[]; labels=[]; scaffolds=[]; metadata=[]
    for start in range(0,len(cache["offset_targets"]),64):
        rows=torch.arange(start,min(start+64,len(cache["offset_targets"]))); batch=p6e10._batch(cache,rows,device); outputs={}
        for arm,model in models.items():
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16):outputs[arm]=model.forward_details(*batch)
        selections={arm:value["scores"].argmax(1) for arm,value in outputs.items()}; prior=outputs["global"]["prior"].argmax(1); selected=selections["global"]; changed=selected!=prior; index=torch.arange(len(rows),device=device); target=cache["offset_targets"][rows].to(device).float(); distance=torch.linalg.vector_norm((batch[2][index,selected]-target)*768,dim=1)
        vector=[_top_margin(outputs["global"]["prior"]),_entropy(outputs["global"]["prior"])]
        for arm in ("global","local17","multiscale"):
            details=outputs[arm]; choice=selections[arm]; vector.extend([_top_margin(details["scores"]),_entropy(details["scores"]),details["scores"][index,choice]-details["scores"][index,prior],details["residual"][index,choice],details["residual"][index,prior],(choice==selected).float()])
        chosen_offset=batch[2][index,selected]*768; prior_offset=batch[2][index,prior]*768; vector.extend([chosen_offset[:,0]/16,chosen_offset[:,1]/16,torch.linalg.vector_norm(chosen_offset,dim=1)/16,prior_offset[:,0]/16,prior_offset[:,1]/16,torch.linalg.vector_norm(chosen_offset-prior_offset,dim=1)/16])
        matrix=torch.stack(vector,dim=1)
        for offset,row in enumerate(rows.tolist()):
            if not bool(changed[offset]):continue
            features.append(matrix[offset].cpu()); labels.append(float(distance[offset]<=6)); scaffolds.append(str(cache["scaffold"][row])); metadata.append({"domain":domain,"sample_key":str(cache["sample_key"][row]),"target_index":int(cache["target_index"][row]),"selected_correct":bool(distance[offset]<=6)})
    return {"features":torch.stack(features),"labels":torch.tensor(labels),"scaffolds":scaffolds,"metadata":metadata}


def _combine(left:dict,right:dict)->dict:return {"features":torch.cat((left["features"],right["features"])),"labels":torch.cat((left["labels"],right["labels"])),"scaffolds":left["scaffolds"]+right["scaffolds"],"metadata":left["metadata"]+right["metadata"]}


def train_model(features,labels,rows,device,seed)->ResidualSafetyMLP:
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);model=ResidualSafetyMLP(features.size(1)).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=1e-4); selected=rows.long(); schedule=p5e9._schedule(len(selected),STEPS,BATCH_SIZE,seed+1); positives=float(labels[selected].sum()); negative=len(selected)-positives; pos_weight=torch.tensor(negative/max(positives,1),device=device)
    for indices in schedule:
        batch_rows=selected[indices];optimizer.zero_grad(set_to_none=True);logits=model(features[batch_rows].to(device));loss=F.binary_cross_entropy_with_logits(logits,labels[batch_rows].to(device),pos_weight=pos_weight);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
    return model.eval()


def main()->None:
    if OUTPUT.exists():raise RuntimeError(f"P6-E11 output exists: {OUTPUT}")
    if any((path.stat().st_mode&0o777)!=0 for path in p6e9.HOLDOUT_FILES):raise RuntimeError("P6-E11 holdout is not sealed")
    if not torch.cuda.is_available():raise RuntimeError("P6-E11 requires CUDA")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8");random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;device=torch.device("cuda");started=time.perf_counter();models=p6e10._load_models(device)
    train=_combine(extract_features(models,torch.load(TRAIN_HARD,map_location="cpu",weights_only=False),"hard",device),extract_features(models,torch.load(TRAIN_EASY,map_location="cpu",weights_only=False),"easy",device));cal=_combine(extract_features(models,torch.load(CAL_HARD,map_location="cpu",weights_only=False),"hard",device),extract_features(models,torch.load(CAL_EASY,map_location="cpu",weights_only=False),"easy",device))
    mean=train["features"].mean(0);scale=train["features"].std(0,unbiased=False).clamp_min(1e-4);train_features=(train["features"]-mean)/scale;cal_features=(cal["features"]-mean)/scale;folds=torch.tensor([scaffold_fold(value) for value in train["scaffolds"]]);oof=torch.empty(len(folds))
    for fold in range(FOLDS):
        model=train_model(train_features,train["labels"],torch.where(folds!=fold)[0],device,SEED+fold*10);oof[folds==fold]=torch.sigmoid(model(train_features[folds==fold].to(device))).cpu()
    full=train_model(train_features,train["labels"],torch.arange(len(folds)),device,SEED+100);cal_scores=torch.sigmoid(full(cal_features.to(device))).detach().cpu();records=[]
    for index,meta in enumerate(cal["metadata"]):records.append({**meta,"consensus":True,"changed":True,"certificate_score":float(cal_scores[index])})
    winner=p6e10.calibrate_zero_harmful(records);checks={"five_folds_present":set(folds.tolist())==set(range(FOLDS)),"calibration_zero_harmful":winner["harmful"]==0,"calibration_rescues_min100":winner["rescues"]>=MINIMUM_RESCUES,"holdout_mode_000":all((path.stat().st_mode&0o777)==0 for path in p6e9.HOLDOUT_FILES)};decision="oof_residual_safety_calibration_go" if all(checks.values()) else "oof_residual_safety_calibration_no_go"
    OUTPUT.mkdir(parents=True);_atomic_torch_save(CHECKPOINT,{"checkpoint_schema_version":"g3-p6e11-oof-residual-safety-0.1","model_type":MODEL_TYPE,"feature_count":train_features.size(1),"model":full.state_dict(),"mean":mean,"scale":scale,"threshold":winner["threshold"],"folds":FOLDS});audit={"schema_version":"g3-p6e11-oof-residual-safety-final-0.1","model_type":MODEL_TYPE,"decision":decision,"checks":checks,"checkpoint":str(CHECKPOINT.resolve()),"checkpoint_sha256":file_sha256(CHECKPOINT),"parameter_count":sum(value.numel() for value in full.parameters()),"winner":winner,"capacity":{"train_changed":len(train["labels"]),"train_positive":int(train["labels"].sum()),"calibration_changed":len(cal["labels"]),"calibration_positive":int(cal["labels"].sum()),"features":train_features.size(1)},"oof":{"mean_score":float(oof.mean())},"elapsed_seconds":time.perf_counter()-started,"holdout_read":False,"dev_confirm_external_sealed_read":False};_atomic_json(OUTPUT/"audit.json",audit);print(json.dumps(audit,indent=2))


if __name__=="__main__":main()
