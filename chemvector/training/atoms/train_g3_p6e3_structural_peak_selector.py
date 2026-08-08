#!/usr/bin/env python3
"""Train a listwise selector over frozen P6-E1 regional maxima."""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from chemvector.components.atoms.objects.g3_atom_center_field import decode_atom_center_field
from chemvector.components.recovery.structural_context_peak import (
    StructuralContextPeakHead,
    anchor_centered_context_crops,
)
from chemvector.components.recovery.structural_peak_selector import (
    CANDIDATES,
    StructuralPeakSelector,
    compatible_peak_selection_loss,
)
from chemvector.evaluation.audits.atoms.audit_g3_p1n_regional_maxima import (
    extract_regional_maximum_objects,
)
from chemvector.evaluation.audits.atoms import audit_g3_p4e6_missing_peak_sources as p4e6
from chemvector.evaluation.audits.atoms import audit_g3_p6e2_candidate_peak_reachability as p6e2
from chemvector.training.atoms import g3_p4_local_recovery_data as recovery_data
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.atoms import train_g3_p1t_continuous_stride2_center_field as p1t
from chemvector.training.atoms import train_g3_p5e9_generic_direction_mixed as p5e9
from chemvector.training.atoms import train_g3_p6e1_structural_context96_mixed as p6e1
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _atomic_torch_save,
    _autocast_context,
    module_state_sha256,
)
from chemvector.g3_structural_data import file_sha256


MODEL_TYPE = "chemvector-g3-p6e3-structural-peak-selector50k-fit1378"
OUTPUT = Path(f"outputs/checkpoints/{MODEL_TYPE}")
CACHE_ROOT = Path("outputs/checkpoints/chemvector-g3-p6e3-peak-candidates")
GENERIC_CANDIDATES = CACHE_ROOT / "generic50k.pt"
REAL_CANDIDATES = CACHE_ROOT / "real1378.pt"
P6E1_CHECKPOINT = p6e1.OUTPUT / "step-015000.pt"
P6E1_CHECKPOINT_SHA256 = p6e2.CHECKPOINT_SHA256
P6E2_AUDIT = p6e2.OUTPUT / "audit.json"
P6E2_AUDIT_SHA256 = "41a5df076f83f95ee415a8be85339c9dbbb2e8a7f2090df1c7456a606ed8103b"
GENERIC_ACTIONS = 50_000
REAL_ACTIONS = 1_378
PRETRAIN_STEPS = 10_000
MIXED_STEPS = 5_000
BATCH_SIZE = 64
MIXED_HALF_BATCH = 32
PRETRAIN_LR = 3e-4
MIXED_LR = 1e-4
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 1.0
SEED = 20260863


@torch.no_grad()
def candidate_tensors(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    offsets = logits.new_zeros((len(logits), CANDIDATES, 2), dtype=torch.float32)
    peak_logits = logits.new_zeros((len(logits), CANDIDATES), dtype=torch.float32)
    mask = torch.zeros((len(logits), CANDIDATES), device=logits.device, dtype=torch.bool)
    ranked = p6e2.ranked_regional_maximum_offsets(logits, maximum_candidates=CANDIDATES)
    for batch_index, current in enumerate(ranked):
        count = len(current)
        offsets[batch_index, :count] = current
        mask[batch_index, :count] = True
        pixels = current * 768.0
        x = torch.round(pixels[:, 0] + 15.5).long()
        y = torch.round(pixels[:, 1] + 15.5).long()
        peak_logits[batch_index, :count] = logits[batch_index, y, x].float()
    return offsets, peak_logits, mask


@torch.no_grad()
def _build_candidate_cache(
    peak_model: StructuralContextPeakHead, base: dict, device: torch.device
) -> dict:
    values = {"candidate_offsets": [], "candidate_peak_logits": [], "candidate_mask": []}
    for start in range(0, len(base["offset_targets"]), BATCH_SIZE):
        rows = slice(start, min(start + BATCH_SIZE, len(base["offset_targets"])))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = peak_model(
                base["image_crops"][rows].to(device).float(),
                base["score_crops"][rows].to(device).float(),
                base["candidate_scores"][rows].to(device).float(),
                base["candidate_ranks"][rows].to(device).float(),
            )
        offsets, peak_logits, mask = candidate_tensors(logits)
        values["candidate_offsets"].append(offsets.cpu().half())
        values["candidate_peak_logits"].append(peak_logits.cpu().half())
        values["candidate_mask"].append(mask.cpu())
    cache = {name: torch.cat(parts) for name, parts in values.items()}
    distances = torch.linalg.vector_norm(
        (cache["candidate_offsets"].float() - base["offset_targets"][:, None].float()) * 768,
        dim=-1,
    )
    cache["reachable"] = (distances.masked_fill(~cache["candidate_mask"], torch.inf) <= 6.0).any(dim=1)
    cache["p6e1_checkpoint_sha256"] = P6E1_CHECKPOINT_SHA256
    return cache


def _load_or_build_candidate_caches(
    peak_model: StructuralContextPeakHead, generic: dict, real: dict, device: torch.device
) -> tuple[dict, dict]:
    if GENERIC_CANDIDATES.exists() and REAL_CANDIDATES.exists():
        outputs = tuple(
            torch.load(path, map_location="cpu", weights_only=False)
            for path in (GENERIC_CANDIDATES, REAL_CANDIDATES)
        )
        if any(value.get("p6e1_checkpoint_sha256") != P6E1_CHECKPOINT_SHA256 for value in outputs):
            raise ValueError("P6-E3 candidate cache lineage differs")
        return outputs
    if CACHE_ROOT.exists():
        raise RuntimeError(f"P6-E3 incomplete candidate cache exists: {CACHE_ROOT}")
    started = time.perf_counter()
    generic_candidates = _build_candidate_cache(peak_model, generic, device)
    real_candidates = _build_candidate_cache(peak_model, real, device)
    CACHE_ROOT.mkdir(parents=True)
    _atomic_torch_save(GENERIC_CANDIDATES, generic_candidates)
    _atomic_torch_save(REAL_CANDIDATES, real_candidates)
    _atomic_json(
        CACHE_ROOT / "audit.json",
        {
            "schema_version": "g3-p6e3-peak-candidates-0.1",
            "generic_reachable": int(generic_candidates["reachable"].sum()),
            "real_reachable": int(real_candidates["reachable"].sum()),
            "generic_cache_sha256": file_sha256(GENERIC_CANDIDATES),
            "real_cache_sha256": file_sha256(REAL_CANDIDATES),
            "p6e1_checkpoint_sha256": P6E1_CHECKPOINT_SHA256,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    return generic_candidates, real_candidates


def _schedule(reachable: torch.Tensor, steps: int, batch: int, seed: int) -> list[torch.Tensor]:
    pool = torch.where(reachable)[0]
    return [pool[rows] for rows in p5e9._schedule(len(pool), steps, batch, seed)]


def _batch(base: dict, candidates: dict, rows: torch.Tensor, device: torch.device) -> tuple:
    return (
        base["image_crops"][rows].to(device).float(),
        base["score_crops"][rows].to(device).float(),
        candidates["candidate_offsets"][rows].to(device).float(),
        candidates["candidate_peak_logits"][rows].to(device).float(),
        candidates["candidate_mask"][rows].to(device),
        base["candidate_scores"][rows].to(device).float(),
        base["candidate_ranks"][rows].to(device).float(),
        base["offset_targets"][rows].to(device).float(),
    )


def _transform(tensors: tuple, transform_ids: torch.Tensor) -> tuple:
    images, scores, offsets, peak_logits, mask, source_scores, source_ranks, targets = tensors
    output_images, output_scores, output_offsets, output_targets = [], [], [], []
    for index, transform in enumerate(transform_ids.tolist()):
        rotation, flip = transform % 4, transform >= 4
        image = torch.rot90(images[index], rotation, dims=(-2, -1))
        score = torch.rot90(scores[index], rotation, dims=(-2, -1))
        current_offsets = offsets[index].clone()
        target = targets[index].clone()
        for _ in range(rotation):
            current_offsets = torch.stack((current_offsets[:, 1], -current_offsets[:, 0]), dim=1)
            target = torch.stack((target[1], -target[0]))
        if flip:
            image = torch.flip(image, dims=(-1,))
            score = torch.flip(score, dims=(-1,))
            current_offsets[:, 0] = -current_offsets[:, 0]
            target[0] = -target[0]
        output_images.append(image)
        output_scores.append(score)
        output_offsets.append(current_offsets)
        output_targets.append(target)
    return (
        torch.stack(output_images), torch.stack(output_scores), torch.stack(output_offsets),
        peak_logits, mask, source_scores, source_ranks, torch.stack(output_targets),
    )


def _train_step(model, optimizer, tensors: tuple, transforms: torch.Tensor) -> tuple[float, float]:
    images, fields, offsets, peak_logits, mask, source_scores, source_ranks, targets = _transform(tensors, transforms)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        scores = model(images, fields, offsets, peak_logits, mask, source_scores, source_ranks)
        loss = compatible_peak_selection_loss(scores, offsets, mask, targets)
    loss.backward()
    gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
    if not bool(torch.isfinite(gradient)):
        raise FloatingPointError("P6-E3 gradient norm is not finite")
    optimizer.step()
    return float(loss.detach()), float(gradient.detach())


@torch.no_grad()
def _metrics(model, base: dict, candidates: dict, device: torch.device) -> dict:
    distances = []
    model.eval()
    for start in range(0, len(base["offset_targets"]), BATCH_SIZE):
        rows = torch.arange(start, min(start + BATCH_SIZE, len(base["offset_targets"])))
        batch = _batch(base, candidates, rows, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            scores = model(*batch[:-1])
        chosen = scores.argmax(dim=1)
        predicted = batch[2][torch.arange(len(chosen), device=device), chosen]
        distances.extend(torch.linalg.vector_norm((predicted - batch[-1]) * 768, dim=1).cpu().tolist())
    return p6e1.p5e1._distance_metrics(distances)


@torch.no_grad()
def _evaluate_split(model, peak_model, split: str, device: torch.device) -> tuple[dict, dict]:
    manifest, actions_by_key = p6e1.p5e1._action_manifest(split)
    field, atom_head, vision, frozen_block, vision_sha = p4e6._load_model(device)
    collator = p1a.G3AtomCollator(maximum_atoms=128, image_size=768)
    distances, started = [], time.perf_counter()
    for batch in p1a.iter_manifest_batches(manifest, collator, batch_size=8, accepted_split="train"):
        objects, stroke = p1t.model_inputs(atom_head, vision, frozen_block, batch["pixel_values"], device=device, training=False)
        with _autocast_context(device, "bf16"):
            field_output = field(objects, stroke)
        peaks = extract_regional_maximum_objects(decode_atom_center_field(field_output), maximum_objects=128)
        observations = recovery_data.build_local_recovery_observations(batch["pixel_values"].to(device), field_output, peaks)
        darkness = recovery_data._grayscale_darkness(batch["pixel_values"].to(device))
        center = field_output["center_logits"].detach().float()
        for batch_index, key_value in enumerate(batch["keys"]):
            key = str(key_value)
            for action in actions_by_key[key]:
                matches = torch.where(observations["candidate_source_ids"][batch_index] == int(action["base_source_id"]))[0]
                if len(matches) != 1:
                    raise RuntimeError(f"P6-E3 Oracle source replay differs: {key}")
                index = int(matches[0])
                source = observations["candidate_anchors"][batch_index, index]
                image_crop = anchor_centered_context_crops(darkness[batch_index:batch_index+1], source[None, None])[0]
                score_crop = anchor_centered_context_crops(center[batch_index:batch_index+1], source[None, None])[0]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    heatmap = peak_model(image_crop, score_crop, observations["candidate_scores"][batch_index,index:index+1], observations["candidate_ranks"][batch_index,index:index+1])
                offsets, peak_logits, mask = candidate_tensors(heatmap)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    scores = model(image_crop, score_crop, offsets, peak_logits, mask, observations["candidate_scores"][batch_index,index:index+1], observations["candidate_ranks"][batch_index,index:index+1])
                predicted = offsets[0, int(scores[0].argmax())]
                target = source.new_tensor(action["target_anchor"]) - source
                distances.append(float(torch.linalg.vector_norm((predicted - target) * 768)))
    return p6e1.p5e1._distance_metrics(distances), {"images": len(manifest), "actions": len(distances), "vision_checkpoint_sha256": vision_sha, "elapsed_seconds": time.perf_counter() - started}


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"P6-E3 output already exists: {OUTPUT}")
    for path, expected in ((P6E1_CHECKPOINT, P6E1_CHECKPOINT_SHA256), (P6E2_AUDIT, P6E2_AUDIT_SHA256)):
        if file_sha256(path) != expected:
            raise ValueError(f"P6-E3 input SHA256 differs: {path}")
    if not torch.cuda.is_available():
        raise RuntimeError("P6-E3 requires CUDA BF16")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False; torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    generic, real = p6e1._load_or_build_caches(device)
    peak_payload = torch.load(P6E1_CHECKPOINT, map_location="cpu", weights_only=False)
    peak_model = StructuralContextPeakHead().to(device)
    peak_model.load_state_dict(peak_payload["model"], strict=True); peak_model.eval()
    generic_candidates, real_candidates = _load_or_build_candidate_caches(peak_model, generic, real, device)
    model = StructuralPeakSelector().to(device)
    initial_sha = module_state_sha256(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=PRETRAIN_LR, weight_decay=WEIGHT_DECAY)
    transform_generator = torch.Generator().manual_seed(SEED + 1)
    started, pretrain_loss = time.perf_counter(), 0.0
    for step, rows in enumerate(_schedule(generic_candidates["reachable"], PRETRAIN_STEPS, BATCH_SIZE, SEED + 10), start=1):
        transforms = torch.randint(0, 8, (BATCH_SIZE,), generator=transform_generator).to(device)
        loss, gradient = _train_step(model, optimizer, _batch(generic, generic_candidates, rows, device), transforms)
        pretrain_loss += loss
        if step % 1000 == 0:
            print(json.dumps({"stage":"pretrain","step":step,"loss":loss,"gradient_norm":gradient}), flush=True)
    for group in optimizer.param_groups: group["lr"] = MIXED_LR
    generic_schedule = _schedule(generic_candidates["reachable"], MIXED_STEPS, MIXED_HALF_BATCH, SEED + 20)
    real_schedule = _schedule(real_candidates["reachable"], MIXED_STEPS, MIXED_HALF_BATCH, SEED + 30)
    mixed_loss, last = 0.0, {}
    for step, (generic_rows, real_rows) in enumerate(zip(generic_schedule, real_schedule, strict=True), start=1):
        left = _batch(generic, generic_candidates, generic_rows, device); right = _batch(real, real_candidates, real_rows, device)
        tensors = tuple(torch.cat((a, b), dim=0) for a, b in zip(left, right, strict=True))
        transforms = torch.randint(0, 8, (BATCH_SIZE,), generator=transform_generator).to(device)
        loss, gradient = _train_step(model, optimizer, tensors, transforms)
        mixed_loss += loss; last = {"stage":"mixed","step":step,"loss":loss,"gradient_norm":gradient}
        if step % 1000 == 0: print(json.dumps(last), flush=True)
    domains, streams = {}, {}
    for split in ("calibration", "holdout"):
        domains[split], streams[split] = _evaluate_split(model, peak_model, split, device)
    checks = {f"{split}_{metric}": value for split in ("calibration","holdout") for metric, value in (("recovery_min80pct", domains[split]["recovery_rate_6px"] >= .80), ("mae_max3px", domains[split]["mae_pixels"] <= 3.0))}
    decision = "structural_peak_selector_go" if all(checks.values()) else "structural_peak_selector_no_go"
    OUTPUT.mkdir(parents=True)
    checkpoint = OUTPUT / "step-015000.pt"
    _atomic_torch_save(checkpoint, {"checkpoint_schema_version":"g3-p6e3-structural-peak-selector-0.1","model_type":MODEL_TYPE,"model":model.state_dict(),"optimizer":optimizer.state_dict(),"global_step":15000,"metrics":{"decision":decision,"checks":checks,"domains":domains}})
    final = {"schema_version":"g3-p6e3-structural-peak-selector-final-0.1","model_type":MODEL_TYPE,"decision":decision,"checks":checks,"checkpoint":str(checkpoint.resolve()),"checkpoint_sha256":file_sha256(checkpoint),"parameter_count":model.parameter_count,"initial_state_sha256":initial_sha,"final_state_sha256":module_state_sha256(model),"training":{"pretrain_steps":PRETRAIN_STEPS,"pretrain_mean_loss":pretrain_loss/PRETRAIN_STEPS,"mixed_steps":MIXED_STEPS,"mixed_mean_loss":mixed_loss/MIXED_STEPS,"last":last},"candidate_capacity":{"generic_reachable":int(generic_candidates["reachable"].sum()),"real_reachable":int(real_candidates["reachable"].sum())},"final_generic_fit":_metrics(model,generic,generic_candidates,device),"final_real_fit":_metrics(model,real,real_candidates,device),"domains":domains,"streams":streams,"lineage":{"p6e1_checkpoint_sha256":P6E1_CHECKPOINT_SHA256,"p6e2_audit_sha256":P6E2_AUDIT_SHA256,"generic_candidate_cache_sha256":file_sha256(GENERIC_CANDIDATES),"real_candidate_cache_sha256":file_sha256(REAL_CANDIDATES)},"elapsed_seconds":time.perf_counter()-started,"confirm_v2_read":False,"dev_and_external_read":False}
    _atomic_json(OUTPUT / "final_metrics.json", final); _atomic_json(OUTPUT / "audit.json", {"status":"completed", **final})
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
