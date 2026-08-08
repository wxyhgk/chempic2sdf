#!/usr/bin/env python3
"""Train and evaluate the P5-E1 forced-source spatial secondary-peak decoder."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from chemvector.components.atoms.objects.g3_atom_center_field import (
    decode_atom_center_field,
)
from chemvector.components.recovery.spatial_secondary_peak import (
    SpatialSecondaryPeakHead,
    decode_secondary_peak_offsets,
    dihedral_transform,
    secondary_peak_loss,
)
from chemvector.evaluation.audits.atoms.audit_g3_p1n_regional_maxima import (
    extract_regional_maximum_objects,
)
from chemvector.evaluation.audits.atoms import (
    audit_g3_p4e6_missing_peak_sources as p4e6,
)
from chemvector.evaluation.evaluators.atoms import (
    evaluate_g3_p4e9_local_recovery_pilot20k as p4eval,
)
from chemvector.training.atoms import g3_p4_local_recovery_data as recovery_data
from chemvector.training.atoms import (
    train_g3_p1a_canonical_atom_selector as p1a,
)
from chemvector.training.atoms import (
    train_g3_p1t_continuous_stride2_center_field as p1t,
)
from chemvector.training.atoms import (
    train_g3_p4e9_local_recovery_pilot20k as p4train,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _atomic_torch_save,
    _autocast_context,
    module_state_sha256,
)
from chemvector.g3_structural_data import file_sha256


MODEL_TYPE = "chemvector-g3-p5e1-spatial-secondary-peak-fit264"
OUTPUT = Path(f"outputs/checkpoints/{MODEL_TYPE}")
SOURCE_CACHE = Path(
    "outputs/checkpoints/chemvector-g3-p4e9b-structural-anchor-refine-pilot20k/fit-cache.pt"
)
SOURCE_CACHE_SHA256 = (
    "f1d68d9ba1e8c63ca07c73ec32574de790755f9a46a1ae338dc35fb0c184f0f0"
)
ORACLE_AUDIT_SHA256 = (
    "5dc3189f7901a42edcc0d97b27cd43c8e8bb47b33573db4605d560cce0179698"
)
ORACLE_ACTIONS_SHA256 = (
    "30aff390726d2e50388e45b37bd3710be19fab8b1d08a1e5fa81bfd3a4ce5819"
)
TRAIN_POSITIVES = 264
BATCH_SIZE = 64
STEPS = 3_000
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 1.0
MINIMUM_RECOVERY = 0.80
MAXIMUM_MAE_PIXELS = 3.0
SEED = 20260853


def _load_positive_cache() -> dict:
    if file_sha256(SOURCE_CACHE) != SOURCE_CACHE_SHA256:
        raise ValueError("P5-E1 source fit-cache SHA256 differs")
    cache = torch.load(SOURCE_CACHE, map_location="cpu", weights_only=False)
    if cache.get("route") != "refine":
        raise ValueError("P5-E1 source cache route differs")
    positive = cache["action_targets"].bool()
    if int(positive.sum()) != TRAIN_POSITIVES:
        raise ValueError("P5-E1 source cache positive count differs")
    output = {
        name: cache[name][positive].clone()
        for name in (
            "image_crops",
            "score_crops",
            "candidate_scores",
            "candidate_ranks",
            "offset_targets",
            "source_ids",
        )
    }
    output["sample_keys"] = [
        key for key, keep in zip(cache["sample_keys"], positive.tolist()) if keep
    ]
    return output


def _schedule(rows: int) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(SEED)
    order = torch.empty(0, dtype=torch.long)
    cursor = 0
    batches = []
    for _ in range(STEPS):
        values = []
        while len(values) < BATCH_SIZE:
            if cursor == len(order):
                order = torch.randperm(rows, generator=generator)
                cursor = 0
            amount = min(BATCH_SIZE - len(values), len(order) - cursor)
            values.extend(order[cursor : cursor + amount].tolist())
            cursor += amount
        batches.append(torch.tensor(values, dtype=torch.long))
    return batches


@torch.no_grad()
def _evaluate_tensors(model, cache: dict, *, device: torch.device) -> dict:
    model.eval()
    distances = []
    for start in range(0, len(cache["offset_targets"]), BATCH_SIZE):
        rows = slice(start, min(start + BATCH_SIZE, len(cache["offset_targets"])))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(
                cache["image_crops"][rows].to(device).float(),
                cache["score_crops"][rows].to(device).float(),
                cache["candidate_scores"][rows].to(device).float(),
                cache["candidate_ranks"][rows].to(device).float(),
            )
        predicted = decode_secondary_peak_offsets(logits).cpu()
        expected = cache["offset_targets"][rows].float()
        distances.extend(
            torch.linalg.vector_norm((predicted - expected) * 768.0, dim=-1).tolist()
        )
    return _distance_metrics(distances)


def _distance_metrics(distances: list[float], variants: list[str] | None = None) -> dict:
    if not distances:
        raise ValueError("P5-E1 distance metrics require samples")
    result = {
        "actions": len(distances),
        "within_6px": sum(value <= 6.0 for value in distances),
        "recovery_rate_6px": sum(value <= 6.0 for value in distances) / len(distances),
        "mae_pixels": float(np.mean(distances)),
        "median_pixels": float(np.median(distances)),
        "p90_pixels": float(np.percentile(distances, 90)),
    }
    if variants is not None:
        by_variant = {}
        for variant in sorted(set(variants)):
            values = [
                distance
                for distance, current in zip(distances, variants)
                if current == variant
            ]
            by_variant[variant] = _distance_metrics(values)
        result["by_variant"] = by_variant
    return result


@torch.no_grad()
def evaluate_oracle_actions(
    model,
    manifest,
    actions_by_key: dict[str, list[dict]],
    *,
    device: torch.device,
) -> tuple[dict, dict]:
    field, atom_head, vision, frozen_block, vision_sha = p4e6._load_model(device)
    collator = p1a.G3AtomCollator(maximum_atoms=128, image_size=768)
    distances = []
    variants = []
    digest = hashlib.sha256()
    started = time.perf_counter()
    for batch in p1a.iter_manifest_batches(
        list(manifest), collator, batch_size=8, accepted_split="train"
    ):
        objects, stroke = p1t.model_inputs(
            atom_head,
            vision,
            frozen_block,
            batch["pixel_values"],
            device=device,
            training=False,
        )
        with _autocast_context(device, "bf16"):
            field_output = field(objects, stroke)
        peaks = extract_regional_maximum_objects(
            decode_atom_center_field(field_output), maximum_objects=128
        )
        observations = recovery_data.build_local_recovery_observations(
            batch["pixel_values"].to(device), field_output, peaks
        )
        crop_rows = []
        score_rows = []
        scalar_rows = []
        rank_rows = []
        targets = []
        current_variants = []
        for batch_index, key_value in enumerate(batch["keys"]):
            key = str(key_value)
            for action in actions_by_key[key]:
                matches = torch.where(
                    observations["candidate_source_ids"][batch_index]
                    == int(action["base_source_id"])
                )[0]
                if len(matches) != 1:
                    raise RuntimeError("P5-E1 Oracle source is absent from top128")
                index = int(matches[0])
                crop_rows.append(observations["image_crops"][batch_index, index])
                score_rows.append(observations["score_crops"][batch_index, index])
                scalar_rows.append(observations["candidate_scores"][batch_index, index])
                rank_rows.append(observations["candidate_ranks"][batch_index, index])
                base = observations["candidate_anchors"][batch_index, index].cpu()
                target = torch.tensor(action["target_anchor"], dtype=torch.float32)
                targets.append(target - base)
                current_variants.append(str(batch["variants"][batch_index]))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(
                torch.stack(crop_rows),
                torch.stack(score_rows),
                torch.stack(scalar_rows),
                torch.stack(rank_rows),
            )
        predicted = decode_secondary_peak_offsets(logits).cpu()
        current_targets = torch.stack(targets)
        current_distances = torch.linalg.vector_norm(
            (predicted - current_targets) * 768.0, dim=-1
        ).tolist()
        distances.extend(current_distances)
        variants.extend(current_variants)
        digest.update(np.asarray(current_distances, dtype=np.float32).tobytes())
    return _distance_metrics(distances, variants), {
        "images": len(manifest),
        "actions": len(distances),
        "distances_sha256": digest.hexdigest(),
        "vision_checkpoint_sha256": vision_sha,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _action_manifest(split: str):
    actions = [
        row
        for row in p4eval._jsonl(p4train.ORACLE_ACTIONS)
        if row["route"] == "ADD_REFINE" and row["split"] == split
    ]
    by_key: dict[str, list[dict]] = defaultdict(list)
    for action in actions:
        by_key[str(action["sample_key"])].append(action)
    path = (
        p4eval.ORACLE_CALIBRATION if split == "calibration" else p4eval.ORACLE_HOLDOUT
    )
    full, _ = p4eval._manifest(path)
    selected = [record for record in full if record.sample_key in by_key]
    if {record.sample_key for record in selected} != set(by_key):
        raise RuntimeError(f"P5-E1 {split} action manifest is incomplete")
    return selected, by_key


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P5-E1 output is not empty: {OUTPUT}")
    for path, expected in (
        (SOURCE_CACHE, SOURCE_CACHE_SHA256),
        (p4train.ORACLE_AUDIT, ORACLE_AUDIT_SHA256),
        (p4train.ORACLE_ACTIONS, ORACLE_ACTIONS_SHA256),
        (p4e6.P4E3_CHECKPOINT, p4e6.P4E3_CHECKPOINT_SHA256),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"P5-E1 input SHA256 differs: {path}")
    if not torch.cuda.is_available():
        raise RuntimeError("P5-E1 requires CUDA BF16")
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

    cache = _load_positive_cache()
    schedule = _schedule(len(cache["offset_targets"]))
    model = SpatialSecondaryPeakHead().to(device)
    initial_sha = module_state_sha256(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    OUTPUT.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    generator = torch.Generator().manual_seed(SEED + 1)
    loss_sum = 0.0
    last = {}
    for step, rows in enumerate(schedule, start=1):
        optimizer.zero_grad(set_to_none=True)
        images = cache["image_crops"][rows].to(device).float()
        scores = cache["score_crops"][rows].to(device).float()
        offsets = cache["offset_targets"][rows].to(device).float()
        transform_ids = torch.randint(
            0, 8, (BATCH_SIZE,), generator=generator
        ).to(device)
        images, scores, offsets = dihedral_transform(
            images, scores, offsets, transform_ids
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(
                images,
                scores,
                cache["candidate_scores"][rows].to(device).float(),
                cache["candidate_ranks"][rows].to(device).float(),
            )
            loss = secondary_peak_loss(logits, offsets)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError("P5-E1 gradient norm is not finite")
        optimizer.step()
        loss_sum += float(loss.detach())
        last = {
            "step": step,
            "loss": float(loss.detach()),
            "gradient_norm": float(gradient_norm.detach()),
        }
        if step % 250 == 0 or step == STEPS:
            print(json.dumps(last), flush=True)

    fit_metrics = _evaluate_tensors(model, cache, device=device)
    domains = {}
    streams = {}
    for split in ("calibration", "holdout"):
        manifest, actions_by_key = _action_manifest(split)
        domains[split], streams[split] = evaluate_oracle_actions(
            model, manifest, actions_by_key, device=device
        )
    checks = {
        "calibration_recovery_min80pct": domains["calibration"]["recovery_rate_6px"]
        >= MINIMUM_RECOVERY,
        "holdout_recovery_min80pct": domains["holdout"]["recovery_rate_6px"]
        >= MINIMUM_RECOVERY,
        "calibration_mae_max3px": domains["calibration"]["mae_pixels"]
        <= MAXIMUM_MAE_PIXELS,
        "holdout_mae_max3px": domains["holdout"]["mae_pixels"]
        <= MAXIMUM_MAE_PIXELS,
    }
    decision = (
        "spatial_secondary_peak_go"
        if all(checks.values())
        else "spatial_secondary_peak_no_go"
    )
    payload = {
        "checkpoint_schema_version": "g3-p5e1-spatial-secondary-peak-0.1",
        "model_type": MODEL_TYPE,
        "config": asdict(model.config),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": STEPS,
        "samples_seen": STEPS * BATCH_SIZE,
        "training_args": {
            "batch_size": BATCH_SIZE,
            "steps": STEPS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "max_grad_norm": MAX_GRAD_NORM,
            "seed": SEED,
            "dihedral_augmentation": True,
        },
        "lineage": {
            "source_cache": str(SOURCE_CACHE.resolve()),
            "source_cache_sha256": SOURCE_CACHE_SHA256,
            "oracle_audit_sha256": ORACLE_AUDIT_SHA256,
            "p4e3_checkpoint_sha256": p4e6.P4E3_CHECKPOINT_SHA256,
        },
        "metrics": {"decision": decision, "checks": checks, "domains": domains},
    }
    checkpoint = OUTPUT / f"step-{STEPS:06d}.pt"
    _atomic_torch_save(checkpoint, payload)
    final = {
        "schema_version": "g3-p5e1-spatial-secondary-peak-final-0.1",
        "decision": decision,
        "checks": checks,
        "model_type": MODEL_TYPE,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "parameter_count": model.parameter_count,
        "initial_state_sha256": initial_sha,
        "final_state_sha256": module_state_sha256(model),
        "training": {
            "steps": STEPS,
            "exposure": STEPS * BATCH_SIZE,
            "mean_loss": loss_sum / STEPS,
            "last": last,
        },
        "fit": fit_metrics,
        "domains": domains,
        "streams": streams,
        "schedule_sha256": hashlib.sha256(
            torch.stack(schedule).numpy().tobytes()
        ).hexdigest(),
        "elapsed_seconds": time.perf_counter() - started,
        "confirm_v2_read": False,
        "dev_and_external_read": False,
    }
    _atomic_json(OUTPUT / "final_metrics.json", final)
    _atomic_json(
        OUTPUT / "audit.json",
        {
            "schema_version": "g3-p5e1-spatial-secondary-peak-audit-0.1",
            "status": "completed",
            **final,
            "protocol": {
                "fit_positive_actions": TRAIN_POSITIVES,
                "forced_oracle_source_evaluation": True,
                "global_spatial_pooling": False,
                "dense_secondary_peak": True,
                "confirm_v2_read": False,
            },
        },
    )
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
