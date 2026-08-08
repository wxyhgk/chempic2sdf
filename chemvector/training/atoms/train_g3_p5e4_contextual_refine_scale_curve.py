#!/usr/bin/env python3
"""Build leakage-free refine crops and run the P5-E4/E5 scale curve."""

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

from chemvector.components.atoms.objects.g3_atom_center_field import (
    decode_atom_center_field,
)
from chemvector.components.recovery.spatial_secondary_peak import (
    ContextualSecondaryPeakHead,
    dihedral_transform,
    secondary_peak_loss,
)
from chemvector.evaluation.audits.atoms.audit_g3_p1n_regional_maxima import (
    extract_regional_maximum_objects,
)
from chemvector.evaluation.audits.atoms import (
    audit_g3_p4e6_missing_peak_sources as p4e6,
)
from chemvector.evaluation.audits.atoms import (
    build_g3_p5e3_refine_capacity_full_train as p5e3,
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


CAPACITY_AUDIT = p5e3.OUTPUT / "audit.json"
CAPACITY_AUDIT_SHA256 = (
    "1895a93e8e90a36d33189bcaef3ba56bd31aba86ef6f648d28d7b5c2c0436c74"
)
ELIGIBLE_MANIFEST = p5e3.OUTPUT / "eligible_train.jsonl"
ELIGIBLE_MANIFEST_SHA256 = (
    "a8e62d8698ed53113d8b4cbc62cd91b0a1cad75eb1defd76d2bd23145ebac8df"
)
NEW_ACTIONS = p5e3.OUTPUT / "refine_actions.jsonl"
NEW_ACTIONS_SHA256 = (
    "ac65fd980d2e659ad0ddab32f4d4f62ff79e4881e71267d42f793fe3719c0dcd"
)
CACHE_ROOT = Path("outputs/checkpoints/chemvector-g3-p5e4-refine-scale-cache")
CACHE_PATH = CACHE_ROOT / "new-refine-cache.pt"
MID_MODEL = "chemvector-g3-p5e4-contextual-secondary-peak-fit528"
FULL_MODEL = "chemvector-g3-p5e5-contextual-secondary-peak-fit1378"
MID_NEW_ACTIONS = 264
FULL_NEW_ACTIONS = 1_114
MID_STEPS = 6_000
FULL_STEPS = 15_660
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 1.0
SEED = 20260856


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _build_new_cache(device: torch.device) -> dict:
    actions = _jsonl(NEW_ACTIONS)
    by_key: dict[str, list[dict]] = defaultdict(list)
    for action in actions:
        by_key[str(action["sample_key"])].append(action)
    manifest, _ = load_g3_sample_manifest(
        ELIGIBLE_MANIFEST,
        expected_sha256=ELIGIBLE_MANIFEST_SHA256,
        expected_samples=p5e3.EXPECTED_ELIGIBLE_IMAGES,
        expected_molecules=p5e3.EXPECTED_ELIGIBLE_MOLECULES,
        expected_variants=p5e3.cycle.VARIANTS,
    )
    selected = [row for row in manifest if row.sample_key in by_key]
    if {row.sample_key for row in selected} != set(by_key):
        raise RuntimeError("P5-E4 action manifest is incomplete")
    field, atom_head, vision, frozen_block, vision_sha = p4e6._load_model(device)
    collator = p1a.G3AtomCollator(maximum_atoms=128, image_size=768)
    values: dict[str, list] = defaultdict(list)
    started = time.perf_counter()
    with torch.no_grad():
        for batch in p1a.iter_manifest_batches(
            selected, collator, batch_size=8, accepted_split="train"
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
            for batch_index, key_value in enumerate(batch["keys"]):
                key = str(key_value)
                for action in by_key[key]:
                    matches = torch.where(
                        observations["candidate_source_ids"][batch_index]
                        == int(action["base_source_id"])
                    )[0]
                    if len(matches) != 1:
                        raise RuntimeError("P5-E4 Oracle source is absent from top128")
                    index = int(matches[0])
                    for name in (
                        "image_crops",
                        "score_crops",
                        "candidate_scores",
                        "candidate_ranks",
                    ):
                        values[name].append(observations[name][batch_index, index].cpu())
                    base = observations["candidate_anchors"][batch_index, index].cpu()
                    target = torch.tensor(action["target_anchor"], dtype=torch.float32)
                    values["offset_targets"].append(target - base)
                    values["sample_keys"].append(key)
                    values["scaffolds"].append(str(action["scaffold"]))
                    values["variants"].append(str(action["variant"]))
    cache = {
        name: torch.stack(rows) if name not in {"sample_keys", "scaffolds", "variants"} else rows
        for name, rows in values.items()
    }
    if len(cache["offset_targets"]) != FULL_NEW_ACTIONS:
        raise RuntimeError("P5-E4 new cache action count differs")
    cache["vision_checkpoint_sha256"] = vision_sha
    cache["elapsed_seconds"] = time.perf_counter() - started
    return cache


def _combine(original: dict, new: dict, new_indices: torch.Tensor) -> dict:
    output = {}
    for name in (
        "image_crops",
        "score_crops",
        "candidate_scores",
        "candidate_ranks",
        "offset_targets",
    ):
        output[name] = torch.cat((original[name], new[name][new_indices]), dim=0)
    output["sample_keys"] = original["sample_keys"] + [
        new["sample_keys"][index] for index in new_indices.tolist()
    ]
    return output


def _new_order(cache: dict) -> torch.Tensor:
    rows = sorted(
        range(len(cache["offset_targets"])),
        key=lambda index: (
            hashlib.sha256(
                f"{SEED}:{cache['scaffolds'][index]}:{cache['sample_keys'][index]}".encode()
            ).hexdigest(),
            index,
        ),
    )
    first_by_scaffold = []
    seen = set()
    for index in rows:
        scaffold = cache["scaffolds"][index]
        if scaffold not in seen:
            seen.add(scaffold)
            first_by_scaffold.append(index)
    if len(first_by_scaffold) < MID_NEW_ACTIONS:
        raise RuntimeError("P5-E4 lacks 264 distinct new scaffolds")
    return torch.tensor(first_by_scaffold[:MID_NEW_ACTIONS], dtype=torch.long)


def _schedule(rows: int, steps: int, seed: int) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    order = torch.empty(0, dtype=torch.long)
    cursor = 0
    batches = []
    for _ in range(steps):
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


def _train_one(
    model_type: str,
    cache: dict,
    *,
    steps: int,
    seed: int,
    device: torch.device,
) -> dict:
    output = Path(f"outputs/checkpoints/{model_type}")
    if output.exists():
        raise RuntimeError(f"P5-E4 output already exists: {output}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = ContextualSecondaryPeakHead().to(device)
    initial_sha = module_state_sha256(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    schedule = _schedule(len(cache["offset_targets"]), steps, seed)
    generator = torch.Generator().manual_seed(seed + 1)
    loss_sum = 0.0
    last = {}
    started = time.perf_counter()
    for step, rows in enumerate(schedule, start=1):
        optimizer.zero_grad(set_to_none=True)
        images = cache["image_crops"][rows].to(device).float()
        scores = cache["score_crops"][rows].to(device).float()
        offsets = cache["offset_targets"][rows].to(device).float()
        transform_ids = torch.randint(0, 8, (BATCH_SIZE,), generator=generator).to(device)
        images, scores, offsets = dihedral_transform(images, scores, offsets, transform_ids)
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
            raise FloatingPointError("P5-E4 gradient norm is not finite")
        optimizer.step()
        loss_sum += float(loss.detach())
        last = {"step": step, "loss": float(loss.detach()), "gradient_norm": float(gradient_norm)}
        if step % 1_000 == 0 or step == steps:
            print(json.dumps({"model": model_type, **last}), flush=True)
    fit = p5e1._evaluate_tensors(model, cache, device=device)
    domains = {}
    streams = {}
    for split in ("calibration", "holdout"):
        manifest, actions_by_key = p5e1._action_manifest(split)
        domains[split], streams[split] = p5e1.evaluate_oracle_actions(
            model, manifest, actions_by_key, device=device
        )
    checks = {
        "calibration_recovery_min80pct": domains["calibration"]["recovery_rate_6px"] >= 0.80,
        "holdout_recovery_min80pct": domains["holdout"]["recovery_rate_6px"] >= 0.80,
        "calibration_mae_max3px": domains["calibration"]["mae_pixels"] <= 3.0,
        "holdout_mae_max3px": domains["holdout"]["mae_pixels"] <= 3.0,
    }
    decision = "contextual_scale_go" if all(checks.values()) else "contextual_scale_no_go"
    output.mkdir(parents=True)
    checkpoint = output / f"step-{steps:06d}.pt"
    _atomic_torch_save(
        checkpoint,
        {
            "checkpoint_schema_version": "g3-p5e4-contextual-scale-0.1",
            "model_type": model_type,
            "config": asdict(model.config),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "global_step": steps,
            "samples_seen": steps * BATCH_SIZE,
            "metrics": {"decision": decision, "checks": checks, "domains": domains},
        },
    )
    final = {
        "schema_version": "g3-p5e4-contextual-scale-final-0.1",
        "model_type": model_type,
        "decision": decision,
        "checks": checks,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "parameter_count": model.parameter_count,
        "initial_state_sha256": initial_sha,
        "final_state_sha256": module_state_sha256(model),
        "training": {
            "actions": len(cache["offset_targets"]),
            "steps": steps,
            "exposure": steps * BATCH_SIZE,
            "exposure_per_action": steps * BATCH_SIZE / len(cache["offset_targets"]),
            "mean_loss": loss_sum / steps,
            "last": last,
        },
        "fit": fit,
        "domains": domains,
        "streams": streams,
        "schedule_sha256": hashlib.sha256(torch.stack(schedule).numpy().tobytes()).hexdigest(),
        "elapsed_seconds": time.perf_counter() - started,
        "confirm_v2_read": False,
        "dev_and_external_read": False,
    }
    _atomic_json(output / "final_metrics.json", final)
    _atomic_json(output / "audit.json", {"status": "completed", **final})
    return final


def main() -> None:
    for path, expected in (
        (CAPACITY_AUDIT, CAPACITY_AUDIT_SHA256),
        (ELIGIBLE_MANIFEST, ELIGIBLE_MANIFEST_SHA256),
        (NEW_ACTIONS, NEW_ACTIONS_SHA256),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"P5-E4 input SHA256 differs: {path}")
    if CACHE_ROOT.exists():
        raise RuntimeError(f"P5-E4 cache output already exists: {CACHE_ROOT}")
    if not torch.cuda.is_available():
        raise RuntimeError("P5-E4 requires CUDA BF16")
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

    new = _build_new_cache(device)
    CACHE_ROOT.mkdir(parents=True)
    _atomic_torch_save(CACHE_PATH, new)
    original = p5e1._load_positive_cache()
    mid_indices = _new_order(new)
    mid = _combine(original, new, mid_indices)
    full = _combine(original, new, torch.arange(FULL_NEW_ACTIONS))
    if len(mid["offset_targets"]) != 528 or len(full["offset_targets"]) != 1_378:
        raise RuntimeError("P5-E4 combined scale counts differ")
    results = {
        "mid": _train_one(MID_MODEL, mid, steps=MID_STEPS, seed=SEED + 10, device=device),
        "full": _train_one(FULL_MODEL, full, steps=FULL_STEPS, seed=SEED + 20, device=device),
    }
    _atomic_json(CACHE_ROOT / "scale_curve.json", results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
