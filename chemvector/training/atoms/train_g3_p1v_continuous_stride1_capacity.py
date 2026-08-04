#!/usr/bin/env python3
"""Run the dual-domain continuous stride-1 Atom center-field capacity gate."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from chemvector.components.atoms.objects.g3_atom_center_field import (
    ContinuousStride1AtomCenterField,
)
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.atoms import train_g3_p1d_boundary_band_set_ranker as p1d
from chemvector.training.atoms import train_g3_p1t_continuous_stride2_center_field as p1t
from chemvector.training.atoms import train_g3_p1u_adjacent_atom_capacity as p1u
from chemvector.training.graph.scenes.g3_structural_data import (
    OVERFIT32_MANIFEST,
    load_g3_overfit32_manifest,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _atomic_torch_save,
    _autocast_context,
    load_frozen_vision_checkpoint,
    module_state_sha256,
)
from chemvector.g3_structural_data import file_sha256


MODEL_TYPE = "chemvector-g3-p1v-continuous-stride1-capacity64"
OUTPUT = Path("outputs/smoke/chemvector-g3-p1v-continuous-stride1-capacity64")
FIELD_PARAMETER_COUNT = 17_379
VISION_BLOCK_PARAMETER_COUNT = 79_200
TRAINABLE_PARAMETER_COUNT = FIELD_PARAMETER_COUNT + VISION_BLOCK_PARAMETER_COUNT
STEPS = 4_800
SAVE_EVERY = 800
EXPECTED_EXPOSURE = 19_200
RESUME_CHECKPOINT = OUTPUT / "step-004000.pt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--resume-step4000", action="store_true")
    return parser


def validate_args(args) -> None:
    fixed = {
        "batch_size": 4,
        "learning_rate": 3e-4,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "seed": 20260713,
    }
    for name, expected in fixed.items():
        if getattr(args, name) != expected:
            raise ValueError(f"P1-V requires fixed {name}={expected}")
    if args.output_dir.resolve() != OUTPUT.resolve():
        raise ValueError(f"P1-V requires output directory {OUTPUT}")
    if not torch.cuda.is_available():
        raise RuntimeError("P1-V requires CUDA BF16")


def validate_optimizer(field, atom_head, vision, optimizer) -> None:
    block = p1t.trainable_vision_block(vision)
    expected = {
        id(value)
        for module in (field, block)
        for value in module.parameters()
        if value.requires_grad
    }
    actual = {id(value) for group in optimizer.param_groups for value in group["params"]}
    if expected != actual:
        raise ValueError("P1-V optimizer parameter identity differs")
    if field.trainable_parameter_count() != FIELD_PARAMETER_COUNT:
        raise ValueError("P1-V field must contain exactly 17,379 parameters")
    if sum(value.numel() for value in block.parameters()) != VISION_BLOCK_PARAMETER_COUNT:
        raise ValueError("P1-V vision block must contain exactly 79,200 parameters")
    if sum(value.numel() for value in optimizer.param_groups[0]["params"]) != TRAINABLE_PARAMETER_COUNT:
        raise ValueError("P1-V optimizer must contain exactly 96,579 parameters")
    if any(value.requires_grad for value in atom_head.parameters()):
        raise ValueError("P1-V P1-A Atom head became trainable")
    trainable_prefix = f"stages.0.{len(vision.stages[0]) - 1}."
    for name, value in vision.named_parameters():
        if value.requires_grad != name.startswith(trainable_prefix):
            raise ValueError(f"P1-V unexpected vision trainability: {name}")


@torch.no_grad()
def step0_parity(field, atom_head, object_features, stroke_features) -> dict:
    objects = object_features.detach().float()
    output = field(objects, stroke_features.detach().float())
    expected_center = F.interpolate(
        atom_head.presence(objects)[:, 0:1],
        scale_factor=4.0,
        mode="bilinear",
        align_corners=False,
    )
    center_error = float((output["center_logits"] - expected_center).abs().max())
    offset_error = float(output["offset_logits"].abs().max())
    return {
        "center_max_abs_error": center_error,
        "offset_zero_max_abs_error": offset_error,
        "exact": center_error == 0.0 and offset_error == 0.0,
    }


def checkpoint_payload(
    field, vision_block, optimizer, args, lineage, *, step, exposure, losses, metrics=None
):
    return {
        "checkpoint_schema_version": "g3-p1v-continuous-stride1-capacity-0.1",
        "model_type": MODEL_TYPE,
        "atom_center_field": field.state_dict(),
        "vision_last_stride4_block": vision_block.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": step,
        "samples_seen": exposure,
        "training_args": vars(args),
        "lineage": lineage,
        "trainable_parameter_count": TRAINABLE_PARAMETER_COUNT,
        "center_target": "one_hot",
        "offset_loss_weight": 8.0,
        "peak_margin_weight": 1.0,
        "peak_margin": 1.0,
        "continuous_output_stride": 1,
        "continuous_high_resolution_channels": 32,
        "unfrozen_vision_blocks": 1,
        "pixel_cluster_radius": None,
        "training_domains": ["overfit32", "adjacent32"],
        "alternating_domain_batches": True,
        "last_losses": losses,
        "metrics": metrics,
    }


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    if (
        not args.resume_step4000
        and args.output_dir.exists()
        and any(args.output_dir.iterdir())
    ):
        raise RuntimeError("P1-V output directory is not empty")
    if args.resume_step4000 and not RESUME_CHECKPOINT.is_file():
        raise RuntimeError("P1-V step-4000 resume checkpoint is missing")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")

    atom_head, _, p1a_payload, _ = p1d.load_p1a_boundary_ranker(device)
    atom_head.requires_grad_(False).eval()
    vision, _, _, vision_sha = load_frozen_vision_checkpoint(
        Path(p1a_payload["lineage"]["vision"]["path"]), device
    )
    vision.requires_grad_(False).eval()
    block = p1t.trainable_vision_block(vision)
    frozen_block = copy.deepcopy(block).requires_grad_(False).eval()
    block.requires_grad_(True)
    atom_sha = module_state_sha256(atom_head)
    vision_initial_sha = module_state_sha256(vision)
    block_initial_sha = module_state_sha256(block)
    frozen_block_sha = module_state_sha256(frozen_block)
    frozen_vision_initial_sha = p1t.frozen_vision_sha256(vision)
    field = ContinuousStride1AtomCenterField(atom_head).to(device)
    field_initial_state = {
        name: value.detach().cpu().clone() for name, value in field.named_parameters()
    }
    field_initial_sha = module_state_sha256(field)
    parameters = list(field.parameters()) + list(block.parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    validate_optimizer(field, atom_head, vision, optimizer)
    resume_payload = None

    overfit_manifest, overfit_audit = load_g3_overfit32_manifest(OVERFIT32_MANIFEST)
    adjacent_manifest, adjacent_manifest_audit, adjacent_audit = p1u.load_adjacent_manifest()
    collator = p1a.G3AtomCollator(maximum_atoms=128, image_size=768)
    first_batch = next(p1a.iter_manifest_batches(overfit_manifest, collator, batch_size=4))
    first_objects, first_stroke = p1t.model_inputs(
        atom_head, vision, frozen_block, first_batch["pixel_values"],
        device=device, training=False,
    )
    parity = step0_parity(field, atom_head, first_objects, first_stroke)
    if not parity["exact"]:
        raise RuntimeError("P1-V zero-start P1-A base parity failed")
    p1t_payload = torch.load(p1u.P1T_CHECKPOINT, map_location="cpu", weights_only=False)
    p1t_lineage = p1t_payload["lineage"]
    initialization_checks = {
        "vision_matches_p1t": vision_initial_sha
        == p1t_lineage["vision_initial_state_sha256"],
        "block_matches_p1t": block_initial_sha
        == p1t_lineage["vision_block_initial_sha256"],
        "frozen_base_matches_p1t": frozen_block_sha
        == p1t_lineage["frozen_base_block_sha256"],
    }
    if not all(initialization_checks.values()):
        raise RuntimeError("P1-V shared initialization differs from P1-T")
    if args.resume_step4000:
        resume_payload = torch.load(
            RESUME_CHECKPOINT, map_location=device, weights_only=False
        )
        if (
            resume_payload.get("model_type") != MODEL_TYPE
            or resume_payload.get("global_step") != 4_000
            or resume_payload.get("samples_seen") != 16_000
        ):
            raise RuntimeError("P1-V step-4000 resume checkpoint contract differs")
        field.load_state_dict(resume_payload["atom_center_field"], strict=True)
        block.load_state_dict(
            resume_payload["vision_last_stride4_block"], strict=True
        )
        optimizer.load_state_dict(resume_payload["optimizer"])

    first_epoch = list(
        p1u.alternating_domain_batches(overfit_manifest, adjacent_manifest, collator)
    )
    epoch_domains = [domain for domain, _ in first_epoch]
    epoch_keys = [key for _, batch in first_epoch for key in batch["keys"]]
    epoch_keys_sha = p1u._sequence_sha256(epoch_keys)
    if len(first_epoch) != 16 or epoch_domains != ["overfit32", "adjacent32"] * 8:
        raise RuntimeError("P1-V alternating epoch protocol differs")
    lineage = {
        "p1a_checkpoint": str(p1d.P1A_CHECKPOINT.resolve()),
        "p1a_checkpoint_sha256": file_sha256(p1d.P1A_CHECKPOINT),
        "atom_state_sha256": atom_sha,
        "vision_checkpoint_sha256": vision_sha,
        "vision_initial_state_sha256": vision_initial_sha,
        "vision_block_initial_sha256": block_initial_sha,
        "frozen_base_block_sha256": frozen_block_sha,
        "frozen_vision_initial_sha256": frozen_vision_initial_sha,
        "field_initialization_sha256": field_initial_sha,
        "step0_parity": parity,
        "initialization_checks": initialization_checks,
        "p1u_checkpoint": str((p1u.OUTPUT / "step-004800.pt").resolve()),
        "p1u_checkpoint_sha256": file_sha256(p1u.OUTPUT / "step-004800.pt"),
        "p1u_decision": "dual_capacity_no_go",
        "overfit32_manifest": overfit_audit,
        "adjacent32_manifest": adjacent_manifest_audit,
        "adjacent32_audit_sha256": p1u.ADJACENT_AUDIT_SHA256,
        "training_epoch_domains": epoch_domains,
        "training_epoch_keys_sha256": epoch_keys_sha,
        "resume": {
            "enabled": args.resume_step4000,
            "checkpoint": str(RESUME_CHECKPOINT.resolve())
            if args.resume_step4000
            else None,
            "checkpoint_sha256": file_sha256(RESUME_CHECKPOINT)
            if args.resume_step4000
            else None,
            "global_step": 4_000 if args.resume_step4000 else 0,
            "samples_seen": 16_000 if args.resume_step4000 else 0,
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / (
        "train-resumed-step4000.jsonl" if args.resume_step4000 else "train.jsonl"
    )
    step = 4_000 if args.resume_step4000 else 0
    exposure = 16_000 if args.resume_step4000 else 0
    epoch = 250 if args.resume_step4000 else 0
    last_losses = resume_payload.get("last_losses", {}) if resume_payload else {}
    observed_epoch_sha = None
    started = time.perf_counter()
    while step < STEPS:
        epoch += 1
        current_keys = []
        for domain, batch in p1u.alternating_domain_batches(
            overfit_manifest, adjacent_manifest, collator
        ):
            current_keys.extend(batch["keys"])
            field.train()
            vision.eval()
            last_losses = p1t.training_step(
                field, atom_head, vision, frozen_block, optimizer, batch,
                device=device, max_grad_norm=args.max_grad_norm,
            )
            step += 1
            exposure += len(batch["keys"])
            if step % args.log_every == 0:
                record = {
                    "step": step,
                    "exposure": exposure,
                    "epoch": epoch,
                    "domain": domain,
                    **last_losses,
                }
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                print(json.dumps(record, separators=(",", ":")), flush=True)
            if step % SAVE_EVERY == 0 and step < STEPS:
                payload = checkpoint_payload(
                    field, block, optimizer, args, lineage,
                    step=step, exposure=exposure, losses=last_losses,
                )
                _atomic_torch_save(args.output_dir / f"step-{step:06d}.pt", payload)
                _atomic_torch_save(args.output_dir / "last.pt", payload)
        current_sha = p1u._sequence_sha256(current_keys)
        if current_sha != epoch_keys_sha:
            raise RuntimeError("P1-V training epoch sample order changed")
        observed_epoch_sha = current_sha

    overfit_metrics = p1t.evaluate(
        field, atom_head, vision, frozen_block, overfit_manifest, collator, device=device
    )
    adjacent_metrics = p1t.evaluate(
        field, atom_head, vision, frozen_block, adjacent_manifest, collator, device=device
    )
    collision_metrics = p1u.evaluate_collision_pairs(
        field, atom_head, vision, frozen_block, adjacent_manifest, collator,
        adjacent_audit, device=device,
    )
    repeated = {
        "overfit32": p1t.evaluate(
            field, atom_head, vision, frozen_block, overfit_manifest, collator,
            device=device,
        ),
        "adjacent32": p1t.evaluate(
            field, atom_head, vision, frozen_block, adjacent_manifest, collator,
            device=device,
        ),
        "collision_pairs": p1u.evaluate_collision_pairs(
            field, atom_head, vision, frozen_block, adjacent_manifest, collator,
            adjacent_audit, device=device,
        ),
    }
    decision, checks = p1u.capacity_gate(
        overfit_metrics, adjacent_metrics, collision_metrics
    )
    decision = "dual_stride1_capacity_go" if decision == "dual_capacity_go" else "dual_stride1_capacity_no_go"
    metrics = {
        "overfit32": overfit_metrics,
        "adjacent32": adjacent_metrics,
        "collision_pairs": collision_metrics,
    }
    changed = {
        name: not torch.equal(field_initial_state[name], value.detach().cpu())
        for name, value in field.named_parameters()
    }
    protocol = {
        "shared_initialization_matches_p1t": all(initialization_checks.values()),
        "parameter_count_96579": sum(value.numel() for value in parameters)
        == TRAINABLE_PARAMETER_COUNT,
        "optimizer_field_and_last_block_only": validate_optimizer(
            field, atom_head, vision, optimizer
        ) is None,
        "all_field_tensors_changed": all(changed.values()),
        "atom_state_unchanged": module_state_sha256(atom_head) == atom_sha,
        "vision_state_changed": module_state_sha256(vision) != vision_initial_sha,
        "last_block_changed": module_state_sha256(block) != block_initial_sha,
        "frozen_base_block_unchanged": module_state_sha256(frozen_block)
        == frozen_block_sha,
        "frozen_vision_unchanged": p1t.frozen_vision_sha256(vision)
        == frozen_vision_initial_sha,
        "step0_base_exact": parity["exact"],
        "fixed_steps": step == STEPS,
        "fixed_exposure": exposure == EXPECTED_EXPOSURE,
        "fixed_epochs": epoch == 300,
        "alternating_epoch_order": observed_epoch_sha == epoch_keys_sha,
        "resume_contract": (
            not args.resume_step4000
            or (
                resume_payload is not None
                and resume_payload["global_step"] == 4_000
                and resume_payload["samples_seen"] == 16_000
            )
        ),
        "repeat_metrics": repeated == metrics,
    }
    if not all(protocol.values()):
        decision = "protocol_failed"
    final = {
        "decision": decision,
        "checks": checks,
        "protocol_checks": protocol,
        "changed_field_parameter_tensors": changed,
        "metrics": metrics,
        "elapsed_seconds": time.perf_counter() - started,
    }
    payload = checkpoint_payload(
        field, block, optimizer, args, lineage,
        step=step, exposure=exposure, losses=last_losses, metrics=final,
    )
    final_path = args.output_dir / "step-004800.pt"
    _atomic_torch_save(final_path, payload)
    _atomic_torch_save(args.output_dir / "last.pt", payload)
    final["checkpoint"] = str(final_path.resolve())
    final["checkpoint_sha256"] = file_sha256(final_path)
    _atomic_json(args.output_dir / "final_metrics.json", final)
    _atomic_json(args.output_dir / "status.json", {
        "status": "completed",
        "decision": decision,
        "checkpoint": str(final_path.resolve()),
        "checkpoint_sha256": final["checkpoint_sha256"],
    })
    print(json.dumps({"decision": decision, "checkpoint": str(final_path)}, indent=2))


if __name__ == "__main__":
    main()
