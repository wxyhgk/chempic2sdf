#!/usr/bin/env python3
"""Train the P6 peak-blind selector on a 500 -> 1339 hard-data scale curve."""

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

from chemvector.components.recovery.structural_context_peak import StructuralContextPeakHead, anchor_centered_context_crops
from chemvector.components.recovery.structural_peak_selector import StructuralPeakSelector
from chemvector.evaluation.audits.atoms import audit_g3_p4e6_missing_peak_sources as p4e6
from chemvector.evaluation.audits.atoms import build_g3_p6e6_unselected_hard_pool as p6e6
from chemvector.training.atoms import g3_p4_local_recovery_data as recovery_data
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.atoms import train_g3_p1t_continuous_stride2_center_field as p1t
from chemvector.training.atoms import train_g3_p5e9_generic_direction_mixed as p5e9
from chemvector.training.atoms import train_g3_p6e1_structural_context96_mixed as p6e1
from chemvector.training.atoms import train_g3_p6e3_structural_peak_selector as p6e3
from chemvector.training.atoms import train_g3_p6e5_peak_blind_balanced as p6e5
from chemvector.training.graph.scenes.g3_structural_data import load_g3_sample_manifest
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import _atomic_json, _atomic_torch_save, _autocast_context, module_state_sha256
from chemvector.g3_structural_data import file_sha256


CYCLE_NAME = "chemvector-g3-p6e7-peak-blind-hard-scale"
OUTPUT = Path(f"outputs/checkpoints/{CYCLE_NAME}")
HARD_CACHE = OUTPUT / "hard1339-cache.pt"
HARD_ACTIONS = p6e6.OUTPUT / "hard_actions.jsonl"
HARD_ACTIONS_SHA256 = "1a92dd1f6ed7549d1404dd081d3d412bb65db868353da1157920776bdd5f08ea"
P6E6_AUDIT = p6e6.OUTPUT / "audit.json"
P6E6_AUDIT_SHA256 = "7850931bada748050e1d14945468bf94ad16578bd3dd46f92bb3fed81f8d4ac0"
ARMS = (500, 1339)
EXPOSURES_PER_HARD = 192
HARD_BATCH = 32
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
MINIMUM_SCALE_GAIN = 0.05
BASE_CALIBRATION = 18 / 31
BASE_HOLDOUT = 38 / 67
SEED = 20260867
MODEL_SEED = 2026086701


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


@torch.no_grad()
def _build_hard_cache(peak_model: StructuralContextPeakHead, device: torch.device) -> dict:
    actions = _jsonl(HARD_ACTIONS)
    by_key: dict[str, list[dict]] = defaultdict(list)
    for row in actions: by_key[str(row["sample_key"])].append(row)
    manifest,_=load_g3_sample_manifest(p6e6.p5e8.FIT_MANIFEST,expected_sha256=p6e6.p5e8.FIT_MANIFEST_SHA256,expected_samples=14_000,expected_molecules=3_500,expected_variants=p6e6.p5e8.cycle.VARIANTS)
    selected=[row for row in manifest if row.sample_key in by_key]
    field,atom_head,vision,frozen_block,vision_sha=p4e6._load_model(device); collator=p1a.G3AtomCollator(maximum_atoms=128,image_size=768)
    values: dict[str,list]=defaultdict(list); started=time.perf_counter(); processed=0
    for batch in p1a.iter_manifest_batches(selected,collator,batch_size=8,accepted_split="train"):
        objects,stroke=p1t.model_inputs(atom_head,vision,frozen_block,batch["pixel_values"],device=device,training=False)
        with _autocast_context(device,"bf16"): field_output=field(objects,stroke)
        darkness=recovery_data._grayscale_darkness(batch["pixel_values"].to(device)); center=field_output["center_logits"].detach().float()
        for batch_index,key_value in enumerate(batch["keys"]):
            key=str(key_value); current=by_key[key]
            sources=torch.tensor([row["source_anchor"] for row in current],device=device,dtype=torch.float32)
            images=anchor_centered_context_crops(darkness[batch_index:batch_index+1],sources[None])[0]
            scores=anchor_centered_context_crops(center[batch_index:batch_index+1],sources[None])[0]
            zeros=torch.zeros(len(current),device=device)
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16): logits=peak_model(images,scores,zeros,zeros)
            offsets,peak_logits,mask=p6e3.candidate_tensors(logits)
            targets=torch.tensor([row["offset_target"] for row in current],dtype=torch.float32)
            distances=torch.linalg.vector_norm((offsets.cpu()-targets[:,None])*768,dim=-1).masked_fill(~mask.cpu(),torch.inf)
            if bool((distances[:,0]<=6).any() or (distances.min(dim=1).values>6).any()): raise RuntimeError("P6-E7 hard cache no longer matches P6-E6")
            values["image_crops"].append(images.cpu().half()); values["score_crops"].append(scores.cpu().half())
            values["candidate_offsets"].append(offsets.cpu().half()); values["candidate_peak_logits"].append(torch.zeros_like(peak_logits.cpu()).half())
            values["candidate_mask"].append(mask.cpu()); values["candidate_scores"].append(torch.zeros(len(current)))
            values["candidate_ranks"].append(torch.zeros(len(current))); values["offset_targets"].append(targets)
            values["sample_keys"].extend([key]*len(current)); values["scaffolds"].extend([str(row["scaffold"]) for row in current]); values["variants"].extend([str(row["variant"]) for row in current])
            processed+=len(current)
    cache={name:torch.cat(parts) for name,parts in values.items() if name not in ("sample_keys","scaffolds","variants")}
    cache.update({name:values[name] for name in ("sample_keys","scaffolds","variants")}); cache["hard_actions_sha256"]=HARD_ACTIONS_SHA256; cache["vision_checkpoint_sha256"]=vision_sha; cache["elapsed_seconds"]=time.perf_counter()-started
    if processed!=1339: raise RuntimeError(f"P6-E7 hard cache count differs: {processed}")
    return cache


def _load_or_build_hard_cache(peak_model,device) -> dict:
    if HARD_CACHE.exists():
        cache=torch.load(HARD_CACHE,map_location="cpu",weights_only=False)
        if cache.get("hard_actions_sha256")!=HARD_ACTIONS_SHA256: raise ValueError("P6-E7 hard cache lineage differs")
        return cache
    OUTPUT.mkdir(parents=True,exist_ok=False); cache=_build_hard_cache(peak_model,device); _atomic_torch_save(HARD_CACHE,cache)
    return cache


def _nested_hard_order(cache: dict) -> torch.Tensor:
    ranked=sorted(range(1339),key=lambda i:(hashlib.sha256(f"{SEED}:{cache['sample_keys'][i]}".encode()).hexdigest(),cache["sample_keys"][i]))
    first=[]; later=[]; seen=set()
    for index in ranked:
        scaffold=cache["scaffolds"][index]
        (first if scaffold not in seen else later).append(index); seen.add(scaffold)
    order=torch.tensor(first+later,dtype=torch.long)
    if len(set(cache["scaffolds"][i] for i in order[:500].tolist()))!=500: raise RuntimeError("P6-E7 small arm does not cover 500 scaffolds")
    return order


def _slice(cache: dict, rows: torch.Tensor) -> dict:
    names=("image_crops","score_crops","candidate_offsets","candidate_peak_logits","candidate_mask","candidate_scores","candidate_ranks","offset_targets")
    return {name:cache[name][rows] for name in names}


def _train_arm(size:int,hard_cache:dict,easy_cache:dict,peak_model,device:torch.device) -> dict:
    model_type=f"chemvector-g3-p6e7-peak-blind-hard{size}"
    model_output=Path(f"outputs/checkpoints/{model_type}")
    if model_output.exists(): raise RuntimeError(f"P6-E7 model output exists: {model_output}")
    steps=size*EXPOSURES_PER_HARD//HARD_BATCH
    torch.manual_seed(MODEL_SEED); torch.cuda.manual_seed_all(MODEL_SEED)
    model=StructuralPeakSelector(use_peak_logits=False).to(device); initial_sha=module_state_sha256(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)
    hard_schedule=p5e9._schedule(size,steps,HARD_BATCH,SEED+size); easy_schedule=p5e9._schedule(size,steps,HARD_BATCH,SEED+size+1)
    transforms_generator=torch.Generator().manual_seed(SEED+size+2); started=time.perf_counter(); loss_sum=0.0; last={}
    for step,(hard_rows,easy_rows) in enumerate(zip(hard_schedule,easy_schedule,strict=True),start=1):
        left=p6e5._batch(hard_cache,hard_rows,device); right=p6e5._batch(easy_cache,easy_rows,device); tensors=tuple(torch.cat((a,b),dim=0) for a,b in zip(left,right,strict=True))
        transforms=torch.randint(0,8,(BATCH_SIZE,),generator=transforms_generator).to(device); loss,gradient=p6e3._train_step(model,optimizer,tensors,transforms); loss_sum+=loss; last={"step":step,"loss":loss,"gradient_norm":gradient}
        if step%1000==0 or step==steps: print(json.dumps({"arm":size,**last}),flush=True)
    hard_fit=p6e5._metrics(model,hard_cache,torch.ones(size,dtype=torch.bool),device); easy_fit=p6e5._metrics(model,easy_cache,torch.ones(size,dtype=torch.bool),device)
    domains={}; streams={}
    for split in ("calibration","holdout"): domains[split],streams[split]=p6e3._evaluate_split(model,peak_model,split,device)
    model_output.mkdir(parents=True); checkpoint=model_output/f"step-{steps:06d}.pt"
    _atomic_torch_save(checkpoint,{"checkpoint_schema_version":"g3-p6e7-peak-blind-hard-scale-0.1","model_type":model_type,"model_config":{"use_peak_logits":False},"model":model.state_dict(),"optimizer":optimizer.state_dict(),"global_step":steps,"metrics":{"domains":domains}})
    result={"model_type":model_type,"hard_actions":size,"easy_actions":size,"steps":steps,"exposures_per_hard":EXPOSURES_PER_HARD,"checkpoint":str(checkpoint.resolve()),"checkpoint_sha256":file_sha256(checkpoint),"parameter_count":model.parameter_count,"initial_state_sha256":initial_sha,"final_state_sha256":module_state_sha256(model),"training":{"mean_loss":loss_sum/steps,"last":last},"hard_fit":hard_fit,"easy_fit":easy_fit,"domains":domains,"streams":streams,"elapsed_seconds":time.perf_counter()-started}
    _atomic_json(model_output/"audit.json",{"status":"completed",**result}); return result


def main() -> None:
    if OUTPUT.exists(): raise RuntimeError(f"P6-E7 cycle output already exists: {OUTPUT}")
    for path,expected in ((HARD_ACTIONS,HARD_ACTIONS_SHA256),(P6E6_AUDIT,P6E6_AUDIT_SHA256),(p6e3.P6E1_CHECKPOINT,p6e3.P6E1_CHECKPOINT_SHA256)):
        if file_sha256(path)!=expected: raise ValueError(f"P6-E7 input SHA256 differs: {path}")
    if not torch.cuda.is_available(): raise RuntimeError("P6-E7 requires CUDA BF16")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8"); random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark=False; torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; device=torch.device("cuda")
    payload=torch.load(p6e3.P6E1_CHECKPOINT,map_location="cpu",weights_only=False); peak_model=StructuralContextPeakHead().to(device); peak_model.load_state_dict(payload["model"],strict=True); peak_model.eval()
    hard_all=_load_or_build_hard_cache(peak_model,device); order=_nested_hard_order(hard_all)
    generic,_=p6e1._load_or_build_caches(device); generic_candidates,_=p6e3._load_or_build_candidate_caches(peak_model,generic,p6e1._load_or_build_caches(device)[1],device)
    _,easy_mask=p6e5._difficulty(generic,generic_candidates); easy_rows=p6e5._select_easy(easy_mask,max(ARMS),SEED+100)
    results=[]
    for size in ARMS:
        hard_cache=_slice(hard_all,order[:size]); easy_cache=p6e5._gather(generic,generic_candidates,easy_rows[:size],0,False); results.append(_train_arm(size,hard_cache,easy_cache,peak_model,device))
    small,full=results
    checks={"full_hard_fit_min90pct":full["hard_fit"]["recovery_rate_6px"]>=.90,"calibration_scale_gain_min5pp":full["domains"]["calibration"]["recovery_rate_6px"]>=small["domains"]["calibration"]["recovery_rate_6px"]+MINIMUM_SCALE_GAIN,"holdout_scale_gain_min5pp":full["domains"]["holdout"]["recovery_rate_6px"]>=small["domains"]["holdout"]["recovery_rate_6px"]+MINIMUM_SCALE_GAIN,"calibration_beats_base_min5pp":full["domains"]["calibration"]["recovery_rate_6px"]>=BASE_CALIBRATION+MINIMUM_SCALE_GAIN,"holdout_beats_base_min5pp":full["domains"]["holdout"]["recovery_rate_6px"]>=BASE_HOLDOUT+MINIMUM_SCALE_GAIN,"calibration_mae_no_worse_than_small":full["domains"]["calibration"]["mae_pixels"]<=small["domains"]["calibration"]["mae_pixels"],"holdout_mae_no_worse_than_small":full["domains"]["holdout"]["mae_pixels"]<=small["domains"]["holdout"]["mae_pixels"]}
    decision="peak_blind_hard_scale_go" if all(checks.values()) else "peak_blind_hard_scale_no_go"
    final={"schema_version":"g3-p6e7-peak-blind-hard-scale-final-0.1","decision":decision,"checks":checks,"arms":results,"protocol":{"hard_sizes":ARMS,"exposures_per_hard":EXPOSURES_PER_HARD,"hard_easy_batch":"32:32","peak_logits_used":False,"dev_confirm_external_sealed_read":False},"lineage":{"hard_actions_sha256":HARD_ACTIONS_SHA256,"p6e6_audit_sha256":P6E6_AUDIT_SHA256,"hard_cache_sha256":file_sha256(HARD_CACHE)},"confirm_v2_read":False,"dev_and_external_read":False}
    _atomic_json(OUTPUT/"final_metrics.json",final); _atomic_json(OUTPUT/"audit.json",{"status":"completed",**final}); print(json.dumps(final,indent=2))


if __name__=="__main__": main()
