#!/usr/bin/env python3
"""Train the P6 dual-resolution 96px structural context recovery head."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from chemvector.components.atoms.objects.g3_atom_center_field import decode_atom_center_field
from chemvector.components.recovery.spatial_secondary_peak import decode_secondary_peak_offsets
from chemvector.components.recovery.structural_context_peak import (
    StructuralContextPeakHead,
    anchor_centered_context_crops,
)
from chemvector.evaluation.audits.atoms.audit_g3_p1n_regional_maxima import (
    extract_regional_maximum_objects,
)
from chemvector.evaluation.audits.atoms import audit_g3_p4e6_missing_peak_sources as p4e6
from chemvector.evaluation.audits.atoms import build_g3_p5e3_refine_capacity_full_train as p5e3
from chemvector.evaluation.audits.atoms import build_g3_p5e8_generic_direction_capacity as p5e8
from chemvector.evaluation.evaluators.atoms import evaluate_g3_p4e9_local_recovery_pilot20k as p4eval
from chemvector.training.atoms import g3_p4_local_recovery_data as recovery_data
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.atoms import train_g3_p1t_continuous_stride2_center_field as p1t
from chemvector.training.atoms import train_g3_p4e9_local_recovery_pilot20k as p4train
from chemvector.training.atoms import train_g3_p5e1_spatial_secondary_peak_fit264 as p5e1
from chemvector.training.atoms import train_g3_p5e9_generic_direction_mixed as p5e9
from chemvector.training.graph.scenes.g3_structural_data import load_g3_sample_manifest
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _atomic_torch_save,
    _autocast_context,
    module_state_sha256,
)
from chemvector.g3_structural_data import file_sha256


MODEL_TYPE = "chemvector-g3-p6e1-structural-context96-mixed50k-fit1378"
OUTPUT = Path(f"outputs/checkpoints/{MODEL_TYPE}")
CACHE_ROOT = Path("outputs/checkpoints/chemvector-g3-p6e1-context96-cache")
GENERIC_CACHE = CACHE_ROOT / "generic50k.pt"
REAL_CACHE = CACHE_ROOT / "real1378.pt"
PAIRS = p5e8.OUTPUT / "selected_pairs.jsonl"
PAIRS_SHA256 = p5e9.PAIRS_SHA256
CAPACITY_AUDIT = p5e8.OUTPUT / "audit.json"
CAPACITY_AUDIT_SHA256 = p5e9.CAPACITY_AUDIT_SHA256
NEW_ACTIONS = p5e3.OUTPUT / "refine_actions.jsonl"
NEW_ACTIONS_SHA256 = "ac65fd980d2e659ad0ddab32f4d4f62ff79e4881e71267d42f793fe3719c0dcd"
ELIGIBLE_MANIFEST = p5e3.OUTPUT / "eligible_train.jsonl"
ELIGIBLE_MANIFEST_SHA256 = "a8e62d8698ed53113d8b4cbc62cd91b0a1cad75eb1defd76d2bd23145ebac8df"
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
SEED = 20260861


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _empty_rows() -> dict[str, list]:
    return defaultdict(list)


def _append_context_rows(
    values: dict[str, list],
    image: torch.Tensor,
    score: torch.Tensor,
    sources: torch.Tensor,
    offsets: torch.Tensor,
    candidate_scores: torch.Tensor,
    candidate_ranks: torch.Tensor,
    keys: list[str],
) -> None:
    crops_image = anchor_centered_context_crops(image, sources[None])[0]
    crops_score = anchor_centered_context_crops(score, sources[None])[0]
    values["image_crops"].append(crops_image.cpu().half())
    values["score_crops"].append(crops_score.cpu().half())
    values["offset_targets"].append(offsets.cpu().float())
    values["candidate_scores"].append(candidate_scores.cpu().float())
    values["candidate_ranks"].append(candidate_ranks.cpu().float())
    values["sample_keys"].extend(keys)


def _finish_cache(values: dict[str, list], expected: int, lineage: dict) -> dict:
    cache = {
        name: torch.cat(values[name], dim=0)
        for name in (
            "image_crops",
            "score_crops",
            "offset_targets",
            "candidate_scores",
            "candidate_ranks",
        )
    }
    cache["sample_keys"] = values["sample_keys"]
    cache.update(lineage)
    if len(cache["offset_targets"]) != expected:
        raise RuntimeError(f"P6 cache count differs: {len(cache['offset_targets'])} != {expected}")
    return cache


def _gather_real_batch(
    values: dict[str, list],
    batch: dict,
    field_output: dict,
    peaks: list[dict],
    actions_by_key: dict[str, list[dict]],
    darkness: torch.Tensor,
) -> None:
    observations = recovery_data.build_local_recovery_observations(
        batch["pixel_values"].to(field_output["center_logits"].device), field_output, peaks
    )
    score_field = field_output["center_logits"].detach().float()
    for batch_index, key_value in enumerate(batch["keys"]):
        key = str(key_value)
        current = actions_by_key.get(key, [])
        if not current:
            continue
        source_rows = []
        offsets = []
        scores = []
        ranks = []
        for action in current:
            matches = torch.where(
                observations["candidate_source_ids"][batch_index]
                == int(action["base_source_id"])
            )[0]
            if len(matches) != 1:
                raise RuntimeError(f"P6 training Oracle source replay differs: {key}")
            index = int(matches[0])
            source = observations["candidate_anchors"][batch_index, index]
            target = source.new_tensor(action["target_anchor"])
            source_rows.append(source)
            offsets.append(target - source)
            scores.append(observations["candidate_scores"][batch_index, index])
            ranks.append(observations["candidate_ranks"][batch_index, index])
        sources = torch.stack(source_rows)
        _append_context_rows(
            values,
            darkness[batch_index : batch_index + 1],
            score_field[batch_index : batch_index + 1],
            sources,
            torch.stack(offsets),
            torch.stack(scores),
            torch.stack(ranks),
            [key] * len(current),
        )


def _build_caches(device: torch.device) -> tuple[dict, dict]:
    pairs = _jsonl(PAIRS)
    pairs_by_key: dict[str, list[dict]] = defaultdict(list)
    for pair in pairs:
        pairs_by_key[str(pair["sample_key"])].append(pair)
    original_actions = [
        row
        for row in p4eval._jsonl(p4train.ORACLE_ACTIONS)
        if row["route"] == "ADD_REFINE" and row["split"] == "fit"
    ]
    original_by_key: dict[str, list[dict]] = defaultdict(list)
    for action in original_actions:
        original_by_key[str(action["sample_key"])].append(action)
    new_actions = _jsonl(NEW_ACTIONS)
    new_by_key: dict[str, list[dict]] = defaultdict(list)
    for action in new_actions:
        new_by_key[str(action["sample_key"])].append(action)

    fit_manifest, _ = load_g3_sample_manifest(
        p5e8.FIT_MANIFEST,
        expected_sha256=p5e8.FIT_MANIFEST_SHA256,
        expected_samples=14_000,
        expected_molecules=3_500,
        expected_variants=p5e8.cycle.VARIANTS,
    )
    eligible_manifest, _ = load_g3_sample_manifest(
        ELIGIBLE_MANIFEST,
        expected_sha256=ELIGIBLE_MANIFEST_SHA256,
        expected_samples=p5e3.EXPECTED_ELIGIBLE_IMAGES,
        expected_molecules=p5e3.EXPECTED_ELIGIBLE_MOLECULES,
        expected_variants=p5e3.cycle.VARIANTS,
    )
    fit_keys = set(pairs_by_key) | set(original_by_key)
    fit_selected = [record for record in fit_manifest if record.sample_key in fit_keys]
    new_selected = [record for record in eligible_manifest if record.sample_key in new_by_key]
    field, atom_head, vision, frozen_block, vision_sha = p4e6._load_model(device)
    collator = p1a.G3AtomCollator(maximum_atoms=128, image_size=768)
    generic_values = _empty_rows()
    real_values = _empty_rows()
    processed_generic = 0
    started = time.perf_counter()

    with torch.no_grad():
        for batch in p1a.iter_manifest_batches(
            fit_selected, collator, batch_size=8, accepted_split="train"
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
                current = pairs_by_key.get(key, [])
                if current:
                    sources = torch.tensor(
                        [row["source_anchor"] for row in current], device=device, dtype=torch.float32
                    )
                    offsets = torch.tensor(
                        [row["offset_target"] for row in current], device=device, dtype=torch.float32
                    )
                    zeros = torch.zeros(len(current), device=device)
                    _append_context_rows(
                        generic_values,
                        darkness[batch_index : batch_index + 1],
                        score_field[batch_index : batch_index + 1],
                        sources,
                        offsets,
                        zeros,
                        zeros,
                        [key] * len(current),
                    )
                    processed_generic += len(current)
            if any(str(key) in original_by_key for key in batch["keys"]):
                peaks = extract_regional_maximum_objects(
                    decode_atom_center_field(field_output), maximum_objects=128
                )
                _gather_real_batch(
                    real_values, batch, field_output, peaks, original_by_key, darkness
                )
            if processed_generic % 5_000 < 32:
                print(
                    json.dumps(
                        {
                            "generic_cache_pairs": processed_generic,
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )

        for batch in p1a.iter_manifest_batches(
            new_selected, collator, batch_size=8, accepted_split="train"
        ):
            objects, stroke = p1t.model_inputs(
                atom_head, vision, frozen_block, batch["pixel_values"], device=device, training=False
            )
            with _autocast_context(device, "bf16"):
                field_output = field(objects, stroke)
            darkness = recovery_data._grayscale_darkness(batch["pixel_values"].to(device))
            peaks = extract_regional_maximum_objects(
                decode_atom_center_field(field_output), maximum_objects=128
            )
            _gather_real_batch(real_values, batch, field_output, peaks, new_by_key, darkness)

    lineage = {
        "vision_checkpoint_sha256": vision_sha,
        "pairs_sha256": PAIRS_SHA256,
    }
    generic = _finish_cache(generic_values, GENERIC_ACTIONS, lineage)
    real = _finish_cache(real_values, REAL_ACTIONS, lineage)
    generic["elapsed_seconds"] = time.perf_counter() - started
    return generic, real


def _load_or_build_caches(device: torch.device) -> tuple[dict, dict]:
    if GENERIC_CACHE.exists() and REAL_CACHE.exists():
        generic = torch.load(GENERIC_CACHE, map_location="cpu", weights_only=False)
        real = torch.load(REAL_CACHE, map_location="cpu", weights_only=False)
        if generic.get("pairs_sha256") != PAIRS_SHA256 or real.get("pairs_sha256") != PAIRS_SHA256:
            raise ValueError("P6 existing cache lineage differs")
        return generic, real
    if CACHE_ROOT.exists():
        raise RuntimeError(f"P6 incomplete cache directory exists: {CACHE_ROOT}")
    generic, real = _build_caches(device)
    CACHE_ROOT.mkdir(parents=True)
    _atomic_torch_save(GENERIC_CACHE, generic)
    _atomic_torch_save(REAL_CACHE, real)
    _atomic_json(
        CACHE_ROOT / "audit.json",
        {
            "schema_version": "g3-p6e1-context96-cache-0.1",
            "generic_pairs": len(generic["offset_targets"]),
            "real_actions": len(real["offset_targets"]),
            "generic_cache_sha256": file_sha256(GENERIC_CACHE),
            "real_cache_sha256": file_sha256(REAL_CACHE),
            "vision_checkpoint_sha256": generic["vision_checkpoint_sha256"],
            "elapsed_seconds": generic["elapsed_seconds"],
        },
    )
    return generic, real


@torch.no_grad()
def _evaluate_cache(model, cache: dict, device: torch.device) -> dict:
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
        distances.extend(
            torch.linalg.vector_norm(
                (predicted - cache["offset_targets"][rows].float()) * 768, dim=-1
            ).tolist()
        )
    return p5e1._distance_metrics(distances)


@torch.no_grad()
def _evaluate_split(model, split: str, device: torch.device) -> tuple[dict, dict]:
    manifest, actions_by_key = p5e1._action_manifest(split)
    field, atom_head, vision, frozen_block, vision_sha = p4e6._load_model(device)
    collator = p1a.G3AtomCollator(maximum_atoms=128, image_size=768)
    distances = []
    started = time.perf_counter()
    for batch in p1a.iter_manifest_batches(
        manifest, collator, batch_size=8, accepted_split="train"
    ):
        objects, stroke = p1t.model_inputs(
            atom_head, vision, frozen_block, batch["pixel_values"], device=device, training=False
        )
        with _autocast_context(device, "bf16"):
            field_output = field(objects, stroke)
        peaks = extract_regional_maximum_objects(
            decode_atom_center_field(field_output), maximum_objects=128
        )
        observations = recovery_data.build_local_recovery_observations(
            batch["pixel_values"].to(device), field_output, peaks
        )
        darkness = recovery_data._grayscale_darkness(batch["pixel_values"].to(device))
        score_field = field_output["center_logits"].detach().float()
        for batch_index, key_value in enumerate(batch["keys"]):
            key = str(key_value)
            for action in actions_by_key[key]:
                matches = torch.where(
                    observations["candidate_source_ids"][batch_index]
                    == int(action["base_source_id"])
                )[0]
                if len(matches) != 1:
                    raise RuntimeError(f"P6 evaluation Oracle source replay differs: {key}")
                index = int(matches[0])
                source = observations["candidate_anchors"][batch_index, index]
                image_crop = anchor_centered_context_crops(
                    darkness[batch_index : batch_index + 1], source[None, None]
                )[0]
                score_crop = anchor_centered_context_crops(
                    score_field[batch_index : batch_index + 1], source[None, None]
                )[0]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(
                        image_crop,
                        score_crop,
                        observations["candidate_scores"][batch_index, index : index + 1],
                        observations["candidate_ranks"][batch_index, index : index + 1],
                    )
                predicted = decode_secondary_peak_offsets(logits)[0]
                target = source.new_tensor(action["target_anchor"])
                distances.append(float(torch.linalg.vector_norm((predicted - (target - source)) * 768)))
    return p5e1._distance_metrics(distances), {
        "images": len(manifest),
        "actions": len(distances),
        "vision_checkpoint_sha256": vision_sha,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"P6-E1 output already exists: {OUTPUT}")
    for path, expected in (
        (PAIRS, PAIRS_SHA256),
        (CAPACITY_AUDIT, CAPACITY_AUDIT_SHA256),
        (NEW_ACTIONS, NEW_ACTIONS_SHA256),
        (ELIGIBLE_MANIFEST, ELIGIBLE_MANIFEST_SHA256),
        (p4e6.P4E3_CHECKPOINT, p4e6.P4E3_CHECKPOINT_SHA256),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"P6-E1 input SHA256 differs: {path}")
    if not torch.cuda.is_available():
        raise RuntimeError("P6-E1 requires CUDA BF16")
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
    generic, real = _load_or_build_caches(device)

    model = StructuralContextPeakHead().to(device)
    initial_sha = module_state_sha256(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=PRETRAIN_LR, weight_decay=WEIGHT_DECAY
    )
    transform_generator = torch.Generator().manual_seed(SEED + 1)
    started = time.perf_counter()
    pretrain_schedule = p5e9._schedule(GENERIC_ACTIONS, PRETRAIN_STEPS, BATCH_SIZE, SEED + 10)
    pretrain_loss = 0.0
    for step, rows in enumerate(pretrain_schedule, start=1):
        transforms = torch.randint(0, 8, (BATCH_SIZE,), generator=transform_generator).to(device)
        loss, gradient = p5e9._train_step(
            model, optimizer, p5e9._batch(generic, rows, device), transforms
        )
        pretrain_loss += loss
        if step % 1_000 == 0:
            print(json.dumps({"stage": "pretrain", "step": step, "loss": loss, "gradient_norm": gradient}), flush=True)
    pretrain_domains = {}
    pretrain_streams = {}
    for split in ("calibration", "holdout"):
        pretrain_domains[split], pretrain_streams[split] = _evaluate_split(model, split, device)

    for group in optimizer.param_groups:
        group["lr"] = MIXED_LR
    generic_schedule = p5e9._schedule(GENERIC_ACTIONS, MIXED_STEPS, MIXED_HALF_BATCH, SEED + 20)
    real_schedule = p5e9._schedule(REAL_ACTIONS, MIXED_STEPS, MIXED_HALF_BATCH, SEED + 30)
    mixed_loss = 0.0
    last = {}
    for step, (generic_rows, real_rows) in enumerate(zip(generic_schedule, real_schedule, strict=True), start=1):
        generic_batch = p5e9._batch(generic, generic_rows, device)
        real_batch = p5e9._batch(real, real_rows, device)
        tensors = tuple(torch.cat((left, right), dim=0) for left, right in zip(generic_batch, real_batch, strict=True))
        transforms = torch.randint(0, 8, (BATCH_SIZE,), generator=transform_generator).to(device)
        loss, gradient = p5e9._train_step(model, optimizer, tensors, transforms)
        mixed_loss += loss
        last = {"stage": "mixed", "step": step, "loss": loss, "gradient_norm": gradient}
        if step % 1_000 == 0:
            print(json.dumps(last), flush=True)

    domains = {}
    streams = {}
    for split in ("calibration", "holdout"):
        domains[split], streams[split] = _evaluate_split(model, split, device)
    checks = {
        "calibration_recovery_min80pct": domains["calibration"]["recovery_rate_6px"] >= 0.80,
        "holdout_recovery_min80pct": domains["holdout"]["recovery_rate_6px"] >= 0.80,
        "calibration_mae_max3px": domains["calibration"]["mae_pixels"] <= 3.0,
        "holdout_mae_max3px": domains["holdout"]["mae_pixels"] <= 3.0,
    }
    decision = "structural_context96_go" if all(checks.values()) else "structural_context96_no_go"
    OUTPUT.mkdir(parents=True)
    checkpoint = OUTPUT / "step-015000.pt"
    _atomic_torch_save(
        checkpoint,
        {
            "checkpoint_schema_version": "g3-p6e1-structural-context96-0.1",
            "model_type": MODEL_TYPE,
            "model_config": {"context_size": 96, "local_size": 32},
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "global_step": PRETRAIN_STEPS + MIXED_STEPS,
            "metrics": {"decision": decision, "checks": checks, "domains": domains},
        },
    )
    final = {
        "schema_version": "g3-p6e1-structural-context96-final-0.1",
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
        "final_generic_fit": _evaluate_cache(model, generic, device),
        "final_real_fit": _evaluate_cache(model, real, device),
        "domains": domains,
        "streams": streams,
        "lineage": {
            "pairs_sha256": PAIRS_SHA256,
            "generic_cache_sha256": file_sha256(GENERIC_CACHE),
            "real_cache_sha256": file_sha256(REAL_CACHE),
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
