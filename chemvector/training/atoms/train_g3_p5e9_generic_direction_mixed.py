#!/usr/bin/env python3
"""Pretrain generic directions, then mix them with real P5 refine actions."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from chemvector.components.recovery.spatial_secondary_peak import (
    ContextualSecondaryPeakHead,
    dihedral_transform,
    secondary_peak_loss,
)
from chemvector.evaluation.audits.atoms import (
    audit_g3_p4e6_missing_peak_sources as p4e6,
)
from chemvector.evaluation.audits.atoms import (
    build_g3_p5e8_generic_direction_capacity as p5e8,
)
from chemvector.training.atoms import g3_p4_local_recovery_data as recovery_data
from chemvector.training.atoms import (
    train_g3_p1a_canonical_atom_selector as p1a,
)
from chemvector.training.atoms import (
    train_g3_p1t_continuous_stride2_center_field as p1t,
)
from chemvector.training.atoms import (
    train_g3_p5e1_spatial_secondary_peak_fit264 as p5e1,
)
from chemvector.training.atoms import (
    train_g3_p5e4_contextual_refine_scale_curve as p5e4,
)
from chemvector.training.graph.scenes.g3_structural_data import (
    load_g3_sample_manifest,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _atomic_torch_save,
    _autocast_context,
    module_state_sha256,
)
from chemvector.g3_structural_data import file_sha256


MODEL_TYPE = "chemvector-g3-p5e9-generic-direction-mixed50k-fit1378"
OUTPUT = Path(f"outputs/checkpoints/{MODEL_TYPE}")
CACHE_ROOT = Path("outputs/checkpoints/chemvector-g3-p5e9-generic-direction-cache50k")
CACHE_PATH = CACHE_ROOT / "generic-cache.pt"
PAIRS = p5e8.OUTPUT / "selected_pairs.jsonl"
PAIRS_SHA256 = "d45ad1bc6f3ba88e8a88593a66a2ea6c2ca4a76849bd552ce72cbbf0f28a9faa"
CAPACITY_AUDIT = p5e8.OUTPUT / "audit.json"
CAPACITY_AUDIT_SHA256 = "951f7a8b0e36b1d551727f5a6d11c0bd582bd0c215aac42aa0706308b2e2b209"
REAL_NEW_CACHE = p5e4.CACHE_PATH
REAL_NEW_CACHE_SHA256 = "c03584809d58b9aca95b819a70b8501c82f12752aeee6818eae78f0ab4502912"
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
SEED = 20260858


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _schedule(rows: int, steps: int, batch_size: int, seed: int) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    order = torch.empty(0, dtype=torch.long)
    cursor = 0
    batches = []
    for _ in range(steps):
        values = []
        while len(values) < batch_size:
            if cursor == len(order):
                order = torch.randperm(rows, generator=generator)
                cursor = 0
            amount = min(batch_size - len(values), len(order) - cursor)
            values.extend(order[cursor : cursor + amount].tolist())
            cursor += amount
        batches.append(torch.tensor(values, dtype=torch.long))
    return batches


def _build_generic_cache(device: torch.device) -> dict:
    pairs = _jsonl(PAIRS)
    by_key: dict[str, list[dict]] = defaultdict(list)
    for pair in pairs:
        by_key[str(pair["sample_key"])].append(pair)
    manifest, _ = load_g3_sample_manifest(
        p5e8.FIT_MANIFEST,
        expected_sha256=p5e8.FIT_MANIFEST_SHA256,
        expected_samples=14_000,
        expected_molecules=3_500,
        expected_variants=p5e8.cycle.VARIANTS,
    )
    selected = [record for record in manifest if record.sample_key in by_key]
    if {record.sample_key for record in selected} != set(by_key):
        raise RuntimeError("P5-E9 selected pair manifest is incomplete")
    field, atom_head, vision, frozen_block, vision_sha = p4e6._load_model(device)
    collator = p1a.G3AtomCollator(maximum_atoms=128, image_size=768)
    values: dict[str, list] = defaultdict(list)
    started = time.perf_counter()
    processed = 0
    with torch.no_grad():
        for batch in p1a.iter_manifest_batches(
            selected, collator, batch_size=8, accepted_split="train"
        ):
            objects, stroke = p1t.model_inputs(
                atom_head, vision, frozen_block, batch["pixel_values"], device=device, training=False
            )
            with _autocast_context(device, "bf16"):
                field_output = field(objects, stroke)
            darkness = recovery_data._grayscale_darkness(batch["pixel_values"].to(device))
            score_field = field_output["center_logits"].detach().float()
            for batch_index, key_value in enumerate(batch["keys"]):
                key = str(key_value)
                current = by_key[key]
                sources = torch.tensor(
                    [row["source_anchor"] for row in current],
                    device=device,
                    dtype=torch.float32,
                )[None]
                image_crops = recovery_data.anchor_centered_crops(
                    darkness[batch_index : batch_index + 1], sources
                )[0]
                score_crops = recovery_data.anchor_centered_crops(
                    score_field[batch_index : batch_index + 1], sources
                )[0]
                values["image_crops"].append(image_crops.cpu().half())
                values["score_crops"].append(score_crops.cpu().half())
                values["offset_targets"].append(
                    torch.tensor(
                        [row["offset_target"] for row in current], dtype=torch.float32
                    )
                )
                values["sample_keys"].extend([key] * len(current))
                values["variants"].extend([str(row["variant"]) for row in current])
                values["degree_buckets"].extend(
                    [str(row["degree_bucket"]) for row in current]
                )
                processed += len(current)
            if processed // 5_000 > (processed - sum(len(by_key[str(key)]) for key in batch["keys"])) // 5_000:
                print(
                    json.dumps(
                        {
                            "cache_pairs": processed,
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
    cache = {
        "image_crops": torch.cat(values["image_crops"], dim=0),
        "score_crops": torch.cat(values["score_crops"], dim=0),
        "offset_targets": torch.cat(values["offset_targets"], dim=0),
        "candidate_scores": torch.zeros(processed, dtype=torch.float32),
        "candidate_ranks": torch.zeros(processed, dtype=torch.float32),
        "sample_keys": values["sample_keys"],
        "variants": values["variants"],
        "degree_buckets": values["degree_buckets"],
        "pairs_sha256": PAIRS_SHA256,
        "vision_checkpoint_sha256": vision_sha,
        "elapsed_seconds": time.perf_counter() - started,
    }
    if processed != GENERIC_ACTIONS:
        raise RuntimeError("P5-E9 generic cache count differs")
    return cache


def _load_or_build_generic_cache(device: torch.device) -> dict:
    if CACHE_PATH.exists():
        cache = torch.load(CACHE_PATH, map_location="cpu", weights_only=False)
        if cache.get("pairs_sha256") != PAIRS_SHA256:
            raise ValueError("P5-E9 existing cache lineage differs")
        return cache
    if CACHE_ROOT.exists():
        raise RuntimeError(f"P5-E9 incomplete cache directory exists: {CACHE_ROOT}")
    cache = _build_generic_cache(device)
    CACHE_ROOT.mkdir(parents=True)
    _atomic_torch_save(CACHE_PATH, cache)
    _atomic_json(
        CACHE_ROOT / "audit.json",
        {
            "schema_version": "g3-p5e9-generic-direction-cache-0.1",
            "pairs": GENERIC_ACTIONS,
            "pairs_sha256": PAIRS_SHA256,
            "cache_sha256": file_sha256(CACHE_PATH),
            "vision_checkpoint_sha256": cache["vision_checkpoint_sha256"],
            "elapsed_seconds": cache["elapsed_seconds"],
        },
    )
    return cache


def _real_cache() -> dict:
    if file_sha256(REAL_NEW_CACHE) != REAL_NEW_CACHE_SHA256:
        raise ValueError("P5-E9 real new cache SHA256 differs")
    original = p5e1._load_positive_cache()
    new = torch.load(REAL_NEW_CACHE, map_location="cpu", weights_only=False)
    return p5e4._combine(original, new, torch.arange(len(new["offset_targets"])))


def _batch(cache: dict, rows: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        cache["image_crops"][rows].to(device).float(),
        cache["score_crops"][rows].to(device).float(),
        cache["candidate_scores"][rows].to(device).float(),
        cache["candidate_ranks"][rows].to(device).float(),
        cache["offset_targets"][rows].to(device).float(),
    )


def _train_step(
    model,
    optimizer,
    tensors: tuple[torch.Tensor, ...],
    transform_ids: torch.Tensor,
) -> tuple[float, float]:
    images, scores, candidate_scores, candidate_ranks, offsets = tensors
    images, scores, offsets = dihedral_transform(images, scores, offsets, transform_ids)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(images, scores, candidate_scores, candidate_ranks)
        loss = secondary_peak_loss(logits, offsets)
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
    if not bool(torch.isfinite(gradient_norm)):
        raise FloatingPointError("P5-E9 gradient norm is not finite")
    optimizer.step()
    return float(loss.detach()), float(gradient_norm.detach())


def _evaluate_real(model, device: torch.device) -> tuple[dict, dict]:
    domains = {}
    streams = {}
    for split in ("calibration", "holdout"):
        manifest, actions_by_key = p5e1._action_manifest(split)
        domains[split], streams[split] = p5e1.evaluate_oracle_actions(
            model, manifest, actions_by_key, device=device
        )
    return domains, streams


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"P5-E9 output already exists: {OUTPUT}")
    for path, expected in (
        (PAIRS, PAIRS_SHA256),
        (CAPACITY_AUDIT, CAPACITY_AUDIT_SHA256),
        (p5e8.FIT_MANIFEST, p5e8.FIT_MANIFEST_SHA256),
        (p4e6.P4E3_CHECKPOINT, p4e6.P4E3_CHECKPOINT_SHA256),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"P5-E9 input SHA256 differs: {path}")
    if not torch.cuda.is_available():
        raise RuntimeError("P5-E9 requires CUDA BF16")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")

    generic = _load_or_build_generic_cache(device)
    real = _real_cache()
    if len(generic["offset_targets"]) != GENERIC_ACTIONS or len(real["offset_targets"]) != REAL_ACTIONS:
        raise RuntimeError("P5-E9 cache action count differs")
    model = ContextualSecondaryPeakHead().to(device)
    initial_sha = module_state_sha256(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=PRETRAIN_LR, weight_decay=WEIGHT_DECAY
    )
    transform_generator = torch.Generator().manual_seed(SEED + 1)
    started = time.perf_counter()

    pretrain_schedule = _schedule(GENERIC_ACTIONS, PRETRAIN_STEPS, BATCH_SIZE, SEED + 10)
    pretrain_loss = 0.0
    last = {}
    for step, rows in enumerate(pretrain_schedule, start=1):
        transform_ids = torch.randint(0, 8, (BATCH_SIZE,), generator=transform_generator).to(device)
        loss, gradient = _train_step(model, optimizer, _batch(generic, rows, device), transform_ids)
        pretrain_loss += loss
        last = {"stage": "pretrain", "step": step, "loss": loss, "gradient_norm": gradient}
        if step % 1_000 == 0:
            print(json.dumps(last), flush=True)
    pretrain_domains, pretrain_streams = _evaluate_real(model, device)

    for group in optimizer.param_groups:
        group["lr"] = MIXED_LR
    generic_schedule = _schedule(GENERIC_ACTIONS, MIXED_STEPS, MIXED_HALF_BATCH, SEED + 20)
    real_schedule = _schedule(REAL_ACTIONS, MIXED_STEPS, MIXED_HALF_BATCH, SEED + 30)
    mixed_loss = 0.0
    for step, (generic_rows, real_rows) in enumerate(
        zip(generic_schedule, real_schedule, strict=True), start=1
    ):
        generic_batch = _batch(generic, generic_rows, device)
        real_batch = _batch(real, real_rows, device)
        tensors = tuple(
            torch.cat((left, right), dim=0)
            for left, right in zip(generic_batch, real_batch, strict=True)
        )
        transform_ids = torch.randint(0, 8, (BATCH_SIZE,), generator=transform_generator).to(device)
        loss, gradient = _train_step(model, optimizer, tensors, transform_ids)
        mixed_loss += loss
        last = {"stage": "mixed", "step": step, "loss": loss, "gradient_norm": gradient}
        if step % 1_000 == 0:
            print(json.dumps(last), flush=True)

    generic_fit = p5e1._evaluate_tensors(model, generic, device=device)
    real_fit = p5e1._evaluate_tensors(model, real, device=device)
    domains, streams = _evaluate_real(model, device)
    checks = {
        "calibration_recovery_min80pct": domains["calibration"]["recovery_rate_6px"] >= 0.80,
        "holdout_recovery_min80pct": domains["holdout"]["recovery_rate_6px"] >= 0.80,
        "calibration_mae_max3px": domains["calibration"]["mae_pixels"] <= 3.0,
        "holdout_mae_max3px": domains["holdout"]["mae_pixels"] <= 3.0,
    }
    decision = "generic_direction_mixed_go" if all(checks.values()) else "generic_direction_mixed_no_go"
    OUTPUT.mkdir(parents=True)
    checkpoint = OUTPUT / f"step-{PRETRAIN_STEPS + MIXED_STEPS:06d}.pt"
    _atomic_torch_save(
        checkpoint,
        {
            "checkpoint_schema_version": "g3-p5e9-generic-direction-mixed-0.1",
            "model_type": MODEL_TYPE,
            "config": asdict(model.config),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "global_step": PRETRAIN_STEPS + MIXED_STEPS,
            "metrics": {"decision": decision, "checks": checks, "domains": domains},
        },
    )
    final = {
        "schema_version": "g3-p5e9-generic-direction-mixed-final-0.1",
        "model_type": MODEL_TYPE,
        "decision": decision,
        "checks": checks,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "parameter_count": model.parameter_count,
        "initial_state_sha256": initial_sha,
        "final_state_sha256": module_state_sha256(model),
        "training": {
            "pretrain_steps": PRETRAIN_STEPS,
            "pretrain_mean_loss": pretrain_loss / PRETRAIN_STEPS,
            "mixed_steps": MIXED_STEPS,
            "mixed_mean_loss": mixed_loss / MIXED_STEPS,
            "last": last,
        },
        "pretrain_real_domains": pretrain_domains,
        "pretrain_streams": pretrain_streams,
        "final_generic_fit": generic_fit,
        "final_real_fit": real_fit,
        "domains": domains,
        "streams": streams,
        "lineage": {
            "pairs_sha256": PAIRS_SHA256,
            "capacity_audit_sha256": CAPACITY_AUDIT_SHA256,
            "generic_cache_sha256": file_sha256(CACHE_PATH),
            "real_new_cache_sha256": REAL_NEW_CACHE_SHA256,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "confirm_v2_read": False,
        "dev_and_external_read": False,
    }
    _atomic_json(OUTPUT / "final_metrics.json", final)
    _atomic_json(OUTPUT / "audit.json", {"status": "completed", **final})
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
