#!/usr/bin/env python3
"""Train P6 peak selection without peak-rank input on balanced hard/easy actions."""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from chemvector.components.recovery.structural_context_peak import StructuralContextPeakHead
from chemvector.components.recovery.structural_peak_selector import StructuralPeakSelector
from chemvector.training.atoms import train_g3_p5e9_generic_direction_mixed as p5e9
from chemvector.training.atoms import train_g3_p6e1_structural_context96_mixed as p6e1
from chemvector.training.atoms import train_g3_p6e3_structural_peak_selector as p6e3
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import _atomic_json, _atomic_torch_save, module_state_sha256
from chemvector.g3_structural_data import file_sha256


MODEL_TYPE = "chemvector-g3-p6e5-peak-blind-balanced-hard294-easy294"
OUTPUT = Path(f"outputs/checkpoints/{MODEL_TYPE}")
P6E4_AUDIT = Path("outputs/audits/chemvector-g3-p6e4-selector-shortcut/audit.json")
P6E4_AUDIT_SHA256 = "dd13f7661d87fe0005e3a2edc0a32ecec03e880d6847611c516a7a88555327df"
HARD_GENERIC = 169
HARD_REAL = 125
HARD_ACTIONS = 294
EASY_ACTIONS = 294
STEPS = 5_000
BATCH_SIZE = 64
HALF_BATCH = 32
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 1.0
BASE_CALIBRATION = 18 / 31
BASE_HOLDOUT = 38 / 67
MINIMUM_GAIN = 0.05
SEED = 20260865


def _difficulty(base: dict, candidates: dict) -> tuple[torch.Tensor, torch.Tensor]:
    distances = torch.linalg.vector_norm((candidates["candidate_offsets"].float() - base["offset_targets"][:, None].float()) * 768, dim=-1).masked_fill(~candidates["candidate_mask"], torch.inf)
    reachable = distances.min(dim=1).values <= 6.0
    base_correct = distances[:, 0] <= 6.0
    return reachable & ~base_correct, reachable & base_correct


def _select_easy(mask: torch.Tensor, count: int, seed: int) -> torch.Tensor:
    rows = torch.where(mask)[0]
    generator = torch.Generator().manual_seed(seed)
    return rows[torch.randperm(len(rows), generator=generator)[:count]]


def _gather(base: dict, candidates: dict, rows: torch.Tensor, domain: int, hard: bool) -> dict:
    return {
        "image_crops": base["image_crops"][rows], "score_crops": base["score_crops"][rows],
        "candidate_offsets": candidates["candidate_offsets"][rows], "candidate_peak_logits": torch.zeros((len(rows), 8), dtype=torch.float16),
        "candidate_mask": candidates["candidate_mask"][rows], "candidate_scores": base["candidate_scores"][rows],
        "candidate_ranks": base["candidate_ranks"][rows], "offset_targets": base["offset_targets"][rows],
        "hard": torch.full((len(rows),), hard, dtype=torch.bool), "domain": torch.full((len(rows),), domain, dtype=torch.long),
    }


def _combine(parts: list[dict]) -> dict:
    return {name: torch.cat([part[name] for part in parts], dim=0) for name in parts[0]}


def _batch(cache: dict, rows: torch.Tensor, device: torch.device) -> tuple:
    return (
        cache["image_crops"][rows].to(device).float(), cache["score_crops"][rows].to(device).float(),
        cache["candidate_offsets"][rows].to(device).float(), cache["candidate_peak_logits"][rows].to(device).float(),
        cache["candidate_mask"][rows].to(device), cache["candidate_scores"][rows].to(device).float(),
        cache["candidate_ranks"][rows].to(device).float(), cache["offset_targets"][rows].to(device).float(),
    )


@torch.no_grad()
def _metrics(model, cache: dict, mask: torch.Tensor, device: torch.device) -> dict:
    rows_all = torch.where(mask)[0]; distances = []
    model.eval()
    for start in range(0, len(rows_all), BATCH_SIZE):
        rows = rows_all[start:start+BATCH_SIZE]; batch = _batch(cache, rows, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): scores = model(*batch[:-1])
        chosen = scores.argmax(dim=1); predicted = batch[2][torch.arange(len(chosen), device=device), chosen]
        distances.extend(torch.linalg.vector_norm((predicted - batch[-1]) * 768, dim=1).cpu().tolist())
    return p6e1.p5e1._distance_metrics(distances)


def main() -> None:
    if OUTPUT.exists(): raise RuntimeError(f"P6-E5 output already exists: {OUTPUT}")
    if file_sha256(P6E4_AUDIT) != P6E4_AUDIT_SHA256: raise ValueError("P6-E5 P6-E4 audit SHA256 differs")
    if not torch.cuda.is_available(): raise RuntimeError("P6-E5 requires CUDA BF16")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark = False; torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    generic, real = p6e1._load_or_build_caches(device)
    generic_candidates, real_candidates = p6e3._load_or_build_candidate_caches(StructuralContextPeakHead().to(device), generic, real, device)
    hard_g, easy_g = _difficulty(generic, generic_candidates); hard_r, easy_r = _difficulty(real, real_candidates)
    if int(hard_g.sum()) != HARD_GENERIC or int(hard_r.sum()) != HARD_REAL: raise RuntimeError("P6-E5 hard capacity differs")
    parts = [
        _gather(generic, generic_candidates, torch.where(hard_g)[0], 0, True),
        _gather(real, real_candidates, torch.where(hard_r)[0], 1, True),
        _gather(generic, generic_candidates, _select_easy(easy_g, HARD_GENERIC, SEED+1), 0, False),
        _gather(real, real_candidates, _select_easy(easy_r, HARD_REAL, SEED+2), 1, False),
    ]
    cache = _combine(parts)
    model = StructuralPeakSelector(use_peak_logits=False).to(device); initial_sha = module_state_sha256(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    hard_schedule = [torch.where(cache["hard"])[0][rows] for rows in p5e9._schedule(HARD_ACTIONS, STEPS, HALF_BATCH, SEED+10)]
    easy_schedule = [torch.where(~cache["hard"])[0][rows] for rows in p5e9._schedule(EASY_ACTIONS, STEPS, HALF_BATCH, SEED+20)]
    transform_generator = torch.Generator().manual_seed(SEED+30); started=time.perf_counter(); loss_sum=0.0; last={}
    for step,(hard_rows,easy_rows) in enumerate(zip(hard_schedule,easy_schedule,strict=True),start=1):
        rows=torch.cat((hard_rows,easy_rows)); transforms=torch.randint(0,8,(BATCH_SIZE,),generator=transform_generator).to(device)
        loss,gradient=p6e3._train_step(model,optimizer,_batch(cache,rows,device),transforms); loss_sum+=loss
        last={"step":step,"loss":loss,"gradient_norm":gradient}
        if step%1000==0: print(json.dumps(last),flush=True)
    hard_fit=_metrics(model,cache,cache["hard"],device); easy_fit=_metrics(model,cache,~cache["hard"],device)
    peak_payload=torch.load(p6e3.P6E1_CHECKPOINT,map_location="cpu",weights_only=False); peak_model=StructuralContextPeakHead().to(device); peak_model.load_state_dict(peak_payload["model"],strict=True); peak_model.eval()
    domains={}; streams={}
    for split in ("calibration","holdout"): domains[split],streams[split]=p6e3._evaluate_split(model,peak_model,split,device)
    checks={"hard_fit_recovery_min90pct":hard_fit["recovery_rate_6px"]>=.90,"calibration_gain_min5pp":domains["calibration"]["recovery_rate_6px"]>=BASE_CALIBRATION+MINIMUM_GAIN,"holdout_gain_min5pp":domains["holdout"]["recovery_rate_6px"]>=BASE_HOLDOUT+MINIMUM_GAIN,"calibration_mae_no_worse":domains["calibration"]["mae_pixels"]<=6.481761607431596,"holdout_mae_no_worse":domains["holdout"]["mae_pixels"]<=5.7789420062926276}
    decision="peak_blind_balanced_capacity_go" if all(checks.values()) else "peak_blind_balanced_capacity_no_go"
    OUTPUT.mkdir(parents=True); checkpoint=OUTPUT/"step-005000.pt"
    _atomic_torch_save(checkpoint,{"checkpoint_schema_version":"g3-p6e5-peak-blind-balanced-0.1","model_type":MODEL_TYPE,"model_config":{"use_peak_logits":False},"model":model.state_dict(),"optimizer":optimizer.state_dict(),"global_step":STEPS,"metrics":{"decision":decision,"checks":checks,"domains":domains}})
    final={"schema_version":"g3-p6e5-peak-blind-balanced-final-0.1","model_type":MODEL_TYPE,"decision":decision,"checks":checks,"checkpoint":str(checkpoint.resolve()),"checkpoint_sha256":file_sha256(checkpoint),"parameter_count":model.parameter_count,"initial_state_sha256":initial_sha,"final_state_sha256":module_state_sha256(model),"training":{"steps":STEPS,"mean_loss":loss_sum/STEPS,"last":last,"hard_actions":HARD_ACTIONS,"easy_actions":EASY_ACTIONS},"hard_fit":hard_fit,"easy_fit":easy_fit,"domains":domains,"streams":streams,"lineage":{"p6e4_audit_sha256":P6E4_AUDIT_SHA256,"generic_candidate_cache_sha256":file_sha256(p6e3.GENERIC_CANDIDATES),"real_candidate_cache_sha256":file_sha256(p6e3.REAL_CANDIDATES)},"elapsed_seconds":time.perf_counter()-started,"confirm_v2_read":False,"dev_and_external_read":False}
    _atomic_json(OUTPUT/"final_metrics.json",final); _atomic_json(OUTPUT/"audit.json",{"status":"completed",**final}); print(json.dumps(final,indent=2))


if __name__ == "__main__": main()
