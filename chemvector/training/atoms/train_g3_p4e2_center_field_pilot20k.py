#!/usr/bin/env python3
"""Continue the train25k stride-1 center field on leakage-free P4 pilot20k."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from chemvector.components.atoms.objects.g3_atom_center_field import (
    ContinuousStride1AtomCenterField,
)
from chemvector.evaluation import g3_p3e21_cycle as p3_cycle
from chemvector.evaluation import g3_p4_source_union_cycle as cycle
from chemvector.evaluation.audits.atoms import (
    audit_g3_p3e32_train25k_candidate_source_ab as p3e32,
)
from chemvector.evaluation.audits.atoms import (
    build_g3_p3e21_shared_cache as p3e21,
)
from chemvector.training.atoms import (
    train_g3_p1a_canonical_atom_selector as p1a,
)
from chemvector.training.atoms import (
    train_g3_p1d_boundary_band_set_ranker as p1d,
)
from chemvector.training.atoms import (
    train_g3_p1t_continuous_stride2_center_field as p1t,
)
from chemvector.training.graph.scenes.g3_structural_data import (
    load_g3_sample_manifest,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _atomic_torch_save,
    load_frozen_vision_checkpoint,
    module_state_sha256,
)
from chemvector.g3_structural_data import file_sha256


MODEL_TYPE = "chemvector-g3-p4e2-center-field-pilot20k"
OUTPUT = Path(f"outputs/checkpoints/{MODEL_TYPE}")
P4E0_ROOT = Path(
    "outputs/audits/chemvector-g3-p4e0-source-union-cycle-manifests"
)
P4E0_AUDIT_SHA256 = (
    "50d46024cd07d8313590de85903c5795e9919a7f7dfa5cbbdc7beeb5e4f565e3"
)
PILOT_MANIFEST_SHA256 = (
    "22a03e2b2e37b41b7b298a1fec324fbe383e0f09c49e173a39c8d90347636daa"
)
TRAIN_IMAGES = 20_000
BATCH_SIZE = 4
STEPS = TRAIN_IMAGES // BATCH_SIZE
EXPOSURE = TRAIN_IMAGES
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
MIN_RECALL_GAIN = 0.001
MIN_EXACT_GAIN = 5
SEED = 20260842


def capability_gate(step0: dict, final: dict) -> tuple[str, dict[str, bool]]:
    old = step0["top128_reachability"]
    new = final["top128_reachability"]
    checks = {
        "dev_top128_recall_gain_min001": (
            new["atom_recall"] >= old["atom_recall"] + MIN_RECALL_GAIN
        ),
        "dev_top128_exact_gain_min5": new["exact_images"] >= old["exact_images"] + MIN_EXACT_GAIN,
        "known_count_recall_no_regression": (
            final["known_full_count"]["recall_6px"]
            >= step0["known_full_count"]["recall_6px"] - 0.001
        ),
    }
    return (
        "center_field_pilot20k_go"
        if all(checks.values())
        else "center_field_pilot20k_no_go",
        checks,
    )


def _trainable_parameters(field, block):
    return [
        parameter
        for module in (field, block)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]


def _checkpoint_payload(
    field,
    block,
    optimizer,
    *,
    lineage: dict,
    step: int,
    exposure: int,
    losses: dict,
    metrics: dict,
) -> dict:
    return {
        "checkpoint_schema_version": "g3-p4e2-center-field-pilot20k-0.1",
        "model_type": MODEL_TYPE,
        "scale": "stride1",
        "atom_center_field": field.state_dict(),
        "vision_last_stride4_block": block.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": step,
        "samples_seen": exposure,
        "training_images": TRAIN_IMAGES,
        "training_args": {
            "batch_size": BATCH_SIZE,
            "steps": STEPS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "max_grad_norm": MAX_GRAD_NORM,
            "seed": SEED,
        },
        "trainable_parameter_count": sum(
            value.numel() for value in _trainable_parameters(field, block)
        ),
        "center_target": "one_hot",
        "offset_loss_weight": 8.0,
        "peak_margin_weight": 1.0,
        "peak_margin": 1.0,
        "pixel_cluster_radius": None,
        "lineage": lineage,
        "last_losses": losses,
        "metrics": metrics,
    }


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P4-E2 output is not empty: {OUTPUT}")
    for path, expected in (
        (P4E0_ROOT / "audit.json", P4E0_AUDIT_SHA256),
        (p3e32.TRAIN25K_CHECKPOINT, p3e32.TRAIN25K_SHA256),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"P4-E2 input SHA256 differs: {path}")
    if not torch.cuda.is_available():
        raise RuntimeError("P4-E2 requires CUDA BF16")
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

    train, train_audit = load_g3_sample_manifest(
        P4E0_ROOT / "pilot_train.jsonl",
        expected_sha256=PILOT_MANIFEST_SHA256,
        expected_samples=TRAIN_IMAGES,
        expected_molecules=cycle.PILOT_TRAIN_MOLECULES,
        expected_variants=cycle.VARIANTS,
    )
    dev, dev_audit = load_g3_sample_manifest(
        p3e21.MANIFEST_ROOT / "dev.jsonl",
        expected_sha256=p3e21.MANIFESTS["dev"][0],
        expected_samples=p3e21.MANIFESTS["dev"][1],
        expected_molecules=p3e21.MANIFESTS["dev"][2],
        expected_variants=p3_cycle.VARIANTS,
    )
    source = torch.load(
        p3e32.TRAIN25K_CHECKPOINT, map_location="cpu", weights_only=False
    )
    if source.get("model_type") != p3e32.TRAIN25K_MODEL_TYPE:
        raise ValueError("P4-E2 source model type differs")

    atom_head, _, p1a_payload, _ = p1d.load_p1a_boundary_ranker(device)
    atom_head.requires_grad_(False).eval()
    vision, _, _, vision_sha = load_frozen_vision_checkpoint(
        Path(p1a_payload["lineage"]["vision"]["path"]), device
    )
    vision.requires_grad_(False).eval()
    block = p1t.trainable_vision_block(vision)
    frozen_block = copy.deepcopy(block).requires_grad_(False).eval()
    block.load_state_dict(source["vision_last_stride4_block"], strict=True)
    block.requires_grad_(True)
    field = ContinuousStride1AtomCenterField(atom_head).to(device)
    field.load_state_dict(source["atom_center_field"], strict=True)
    field.requires_grad_(True)

    initial = {
        "field": module_state_sha256(field),
        "block": module_state_sha256(block),
        "atom": module_state_sha256(atom_head),
        "frozen_block": module_state_sha256(frozen_block),
        "frozen_vision": p1t.frozen_vision_sha256(vision),
    }
    optimizer = torch.optim.AdamW(
        _trainable_parameters(field, block),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    collator = p1a.G3AtomCollator(maximum_atoms=128, image_size=768)
    lineage = {
        "source_checkpoint": str(p3e32.TRAIN25K_CHECKPOINT.resolve()),
        "source_checkpoint_sha256": p3e32.TRAIN25K_SHA256,
        "source_model_type": p3e32.TRAIN25K_MODEL_TYPE,
        "p4e0_audit_sha256": P4E0_AUDIT_SHA256,
        "train_manifest": train_audit,
        "dev_manifest": dev_audit,
        "vision_checkpoint_sha256": vision_sha,
        "initial_states": initial,
        "optimizer_loaded_from_source": False,
        "confirm_v2_read": False,
        "sealed_test_read": False,
    }

    OUTPUT.mkdir(parents=True, exist_ok=False)
    step0 = p1t.evaluate(
        field, atom_head, vision, frozen_block, dev, collator, device=device
    )
    _atomic_json(OUTPUT / "step0_dev_metrics.json", step0)
    audit = {
        "schema_version": "g3-p4e2-center-field-pilot20k-audit-0.1",
        "model_type": MODEL_TYPE,
        "status": "training",
        "schedule": {
            "epochs": 1,
            "steps": STEPS,
            "exposure": EXPOSURE,
            "batch_size": BATCH_SIZE,
            "fixed_final_only": True,
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "max_grad_norm": MAX_GRAD_NORM,
        },
        "gate": {
            "minimum_dev_top128_recall_gain": MIN_RECALL_GAIN,
            "minimum_dev_top128_exact_gain": MIN_EXACT_GAIN,
            "known_count_maximum_recall_regression": 0.001,
        },
        "lineage": lineage,
    }
    _atomic_json(OUTPUT / "audit.json", audit)

    started = time.perf_counter()
    key_digest = hashlib.sha256()
    step = exposure = 0
    losses = {}
    log_path = OUTPUT / "train.jsonl"
    for batch in p1a.iter_manifest_batches(train, collator, batch_size=BATCH_SIZE):
        for key in batch["keys"]:
            key_digest.update(f"{key}\n".encode())
        field.train()
        vision.eval()
        losses = p1t.training_step(
            field,
            atom_head,
            vision,
            frozen_block,
            optimizer,
            batch,
            device=device,
            max_grad_norm=MAX_GRAD_NORM,
        )
        step += 1
        exposure += len(batch["keys"])
        if step % 100 == 0 or step == STEPS:
            record = {
                "step": step,
                "exposure": exposure,
                "elapsed_seconds": time.perf_counter() - started,
                **losses,
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            print(json.dumps(record, separators=(",", ":")), flush=True)
    expected_key_sha = train_audit["sample_keys_sha256"]
    observed_key_sha = key_digest.hexdigest()
    final_dev = p1t.evaluate(
        field, atom_head, vision, frozen_block, dev, collator, device=device
    )
    repeated_dev = p1t.evaluate(
        field, atom_head, vision, frozen_block, dev, collator, device=device
    )
    decision, checks = capability_gate(step0, final_dev)
    optimizer_ids = {
        id(value) for group in optimizer.param_groups for value in group["params"]
    }
    expected_ids = {id(value) for value in _trainable_parameters(field, block)}
    protocol_checks = {
        "fixed_steps": step == STEPS,
        "fixed_exposure": exposure == EXPOSURE,
        "fixed_train_sequence": observed_key_sha == expected_key_sha,
        "field_changed": module_state_sha256(field) != initial["field"],
        "vision_block_changed": module_state_sha256(block) != initial["block"],
        "atom_unchanged": module_state_sha256(atom_head) == initial["atom"],
        "frozen_block_unchanged": module_state_sha256(frozen_block) == initial["frozen_block"],
        "frozen_vision_unchanged": p1t.frozen_vision_sha256(vision) == initial["frozen_vision"],
        "optimizer_scope_exact": optimizer_ids == expected_ids,
        "repeat_dev_metrics": repeated_dev == final_dev,
        "confirm_v2_not_read": True,
        "sealed_test_not_read": True,
    }
    if not all(protocol_checks.values()):
        decision = "center_field_pilot20k_protocol_failed"
    final = {
        "decision": decision,
        "checks": checks,
        "protocol_checks": protocol_checks,
        "step0_dev": step0,
        "final_dev": final_dev,
        "training_seconds": time.perf_counter() - started,
    }
    payload = _checkpoint_payload(
        field,
        block,
        optimizer,
        lineage=lineage,
        step=step,
        exposure=exposure,
        losses=losses,
        metrics=final,
    )
    checkpoint = OUTPUT / f"step-{STEPS:06d}.pt"
    _atomic_torch_save(checkpoint, payload)
    final["checkpoint"] = str(checkpoint.resolve())
    final["checkpoint_sha256"] = file_sha256(checkpoint)
    audit["status"] = "completed"
    audit["decision"] = decision
    audit["checkpoint_sha256"] = final["checkpoint_sha256"]
    _atomic_json(OUTPUT / "audit.json", audit)
    _atomic_json(OUTPUT / "final_metrics.json", final)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
