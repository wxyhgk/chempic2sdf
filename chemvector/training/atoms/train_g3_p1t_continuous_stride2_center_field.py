#!/usr/bin/env python3
"""Capacity-test a continuous stride-2 field with one unfrozen vision block."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from chemvector.components.atoms.objects.g3_atom_center_field import (
    ContinuousStride2AtomCenterField,
    atom_center_field_loss,
)
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.atoms import train_g3_p1d_boundary_band_set_ranker as p1d
from chemvector.training.atoms import train_g3_p1r_virtual_stride2_center_field as p1r
from chemvector.training.atoms import train_g3_p1s_phase_residual_center_field as p1s
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


MODEL_TYPE = "chemvector-g3-p1t-continuous-stride2-center-field-overfit32"
OUTPUT = Path("outputs/smoke/chemvector-g3-p1t-continuous-stride2-center-field-overfit32")
FIELD_PARAMETER_COUNT = 12_835
VISION_BLOCK_PARAMETER_COUNT = 79_200
TRAINABLE_PARAMETER_COUNT = FIELD_PARAMETER_COUNT + VISION_BLOCK_PARAMETER_COUNT
STEPS = 2_400
SAVE_EVERY = 400


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--log-every", type=int, default=50)
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
            raise ValueError(f"P1-T requires fixed {name}={expected}")
    if args.output_dir.resolve() != OUTPUT.resolve():
        raise ValueError(f"P1-T requires output directory {OUTPUT}")
    if not torch.cuda.is_available():
        raise RuntimeError("P1-T requires CUDA BF16")


def trainable_vision_block(vision):
    if not len(vision.stages[0]):
        raise ValueError("P1-T vision has no stride-4 ConvNeXt block")
    return vision.stages[0][-1]


def frozen_vision_sha256(vision) -> str:
    trainable_prefix = f"stages.0.{len(vision.stages[0]) - 1}."
    digest = hashlib.sha256()
    for name, value in sorted(vision.state_dict().items()):
        if name.startswith(trainable_prefix):
            continue
        digest.update(name.encode())
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode())
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def validate_optimizer(field, atom_head, vision, optimizer) -> None:
    block = trainable_vision_block(vision)
    expected = {
        id(value)
        for module in (field, block)
        for value in module.parameters()
        if value.requires_grad
    }
    actual = {id(value) for group in optimizer.param_groups for value in group["params"]}
    trainable = sum(
        value.numel()
        for module in (field, block)
        for value in module.parameters()
        if value.requires_grad
    )
    if expected != actual or trainable != TRAINABLE_PARAMETER_COUNT:
        raise ValueError("P1-T optimizer must contain exactly 92,035 parameters")
    if field.trainable_parameter_count() != FIELD_PARAMETER_COUNT:
        raise ValueError("P1-T field must contain exactly 12,835 parameters")
    if sum(value.numel() for value in block.parameters()) != VISION_BLOCK_PARAMETER_COUNT:
        raise ValueError("P1-T vision block must contain exactly 79,200 parameters")
    if any(value.requires_grad for value in atom_head.parameters()):
        raise ValueError("P1-T P1-A Atom head became trainable")
    for name, value in vision.named_parameters():
        expected_trainable = name.startswith(
            f"stages.0.{len(vision.stages[0]) - 1}."
        )
        if value.requires_grad != expected_trainable:
            raise ValueError(f"P1-T unexpected vision trainability: {name}")


def model_inputs(atom_head, vision, frozen_block, pixels, *, device, training: bool):
    with torch.no_grad(), _autocast_context(device, "bf16"):
        hidden = vision.stem(pixels.to(device))
        for block in vision.stages[0][:-1]:
            hidden = block(hidden)
        base_stroke = frozen_block(hidden.detach())
    context = torch.enable_grad() if training else torch.no_grad()
    with context, _autocast_context(device, "bf16"):
        stroke = trainable_vision_block(vision)(hidden.detach())
    with torch.no_grad(), _autocast_context(device, "bf16"):
        dense = atom_head(base_stroke.detach())
    return dense["object_features"].detach(), stroke


def training_step(
    field, atom_head, vision, frozen_block, optimizer, batch, *, device, max_grad_norm
):
    optimizer.zero_grad(set_to_none=True)
    objects, stroke = model_inputs(
        atom_head, vision, frozen_block, batch["pixel_values"],
        device=device, training=True,
    )
    with _autocast_context(device, "bf16"):
        output = field(objects, stroke)
        losses = atom_center_field_loss(
            output,
            batch["atom_anchors"].to(device),
            batch["atom_mask"].to(device),
            adaptive_gaussian=False,
            peak_margin_weight=1.0,
            peak_margin=1.0,
        )
    losses["loss"].backward()
    parameters = list(field.parameters()) + list(trainable_vision_block(vision).parameters())
    gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
    if not bool(torch.isfinite(gradient_norm)):
        raise FloatingPointError("P1-T gradient norm is not finite")
    optimizer.step()
    if any(value.grad is not None for value in atom_head.parameters()):
        raise RuntimeError("P1-T produced P1-A gradients")
    for name, value in vision.named_parameters():
        if not value.requires_grad and value.grad is not None:
            raise RuntimeError(f"P1-T produced frozen vision gradient: {name}")
    return {
        **{name: float(value.detach()) for name, value in losses.items()},
        "gradient_norm": float(gradient_norm.detach()),
    }


@torch.no_grad()
def evaluate(field, atom_head, vision, frozen_block, manifest, collator, *, device) -> dict:
    field.eval()
    vision.eval()

    def infer_batch(batch):
        objects, stroke = model_inputs(
            atom_head, vision, frozen_block, batch["pixel_values"],
            device=device, training=False,
        )
        with _autocast_context(device, "bf16"):
            return field(objects, stroke)

    return p1r.evaluate_inference(manifest, collator, infer_batch)


def capacity_gate(metrics: dict) -> tuple[str, dict[str, bool]]:
    decision, checks = p1r.capacity_gate(metrics)
    return (
        decision if decision == "overfit_capacity_go" else "continuous_stride2_capacity_no_go",
        checks,
    )


def checkpoint_payload(field, vision_block, optimizer, args, lineage, *, step, exposure, losses, metrics=None):
    return {
        "checkpoint_schema_version": "g3-p1t-continuous-stride2-center-field-0.1",
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
        "continuous_high_resolution_channels": 32,
        "unfrozen_vision_blocks": 1,
        "pixel_cluster_radius": None,
        "last_losses": losses,
        "metrics": metrics,
    }


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError("P1-T output directory is not empty")
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
    block = trainable_vision_block(vision)
    frozen_block = copy.deepcopy(block).requires_grad_(False).eval()
    block.requires_grad_(True)
    atom_sha = module_state_sha256(atom_head)
    vision_initial_sha = module_state_sha256(vision)
    block_initial_sha = module_state_sha256(block)
    frozen_block_sha = module_state_sha256(frozen_block)
    frozen_initial_sha = frozen_vision_sha256(vision)
    field = ContinuousStride2AtomCenterField(atom_head).to(device)
    field_initial_state = {
        name: value.detach().cpu().clone() for name, value in field.named_parameters()
    }
    field_initial_sha = module_state_sha256(field)
    parameters = list(field.parameters()) + list(block.parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    validate_optimizer(field, atom_head, vision, optimizer)

    manifest, manifest_audit = load_g3_overfit32_manifest(OVERFIT32_MANIFEST)
    collator = p1a.G3AtomCollator(maximum_atoms=128, image_size=768)
    first_batch = next(p1a.iter_manifest_batches(manifest, collator, batch_size=4))
    first_objects, first_stroke = model_inputs(
        atom_head, vision, frozen_block, first_batch["pixel_values"],
        device=device, training=False,
    )
    parity = p1s.step0_parity(field, atom_head, first_objects, first_stroke)
    if not parity["exact"]:
        raise RuntimeError("P1-T zero-start P1-A base parity failed")
    lineage = {
        "p1a_checkpoint": str(p1d.P1A_CHECKPOINT.resolve()),
        "p1a_checkpoint_sha256": file_sha256(p1d.P1A_CHECKPOINT),
        "atom_state_sha256": atom_sha,
        "vision_checkpoint_sha256": vision_sha,
        "vision_initial_state_sha256": vision_initial_sha,
        "vision_block_initial_sha256": block_initial_sha,
        "frozen_base_block_sha256": frozen_block_sha,
        "frozen_vision_initial_sha256": frozen_initial_sha,
        "manifest": manifest_audit,
        "field_initialization_sha256": field_initial_sha,
        "step0_parity": parity,
        "p1s_checkpoint": str((p1s.OUTPUT / "step-002400.pt").resolve()),
        "p1s_checkpoint_sha256": file_sha256(p1s.OUTPUT / "step-002400.pt"),
        "p1s_decision": "phase_residual_capacity_no_go",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "train.jsonl"
    step = exposure = 0
    last_losses = {}
    started = time.perf_counter()
    while step < STEPS:
        for batch in p1a.iter_manifest_batches(manifest, collator, batch_size=4):
            field.train()
            vision.eval()
            last_losses = training_step(
                field, atom_head, vision, frozen_block, optimizer, batch,
                device=device, max_grad_norm=args.max_grad_norm,
            )
            step += 1
            exposure += len(batch["keys"])
            if step % args.log_every == 0:
                record = {"step": step, "exposure": exposure, **last_losses}
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
            if step >= STEPS:
                break

    metrics = evaluate(
        field, atom_head, vision, frozen_block, manifest, collator, device=device
    )
    repeated = evaluate(
        field, atom_head, vision, frozen_block, manifest, collator, device=device
    )
    decision, checks = capacity_gate(metrics)
    changed = {
        name: not torch.equal(field_initial_state[name], value.detach().cpu())
        for name, value in field.named_parameters()
    }
    protocol = {
        "parameter_count_92035": sum(value.numel() for value in parameters) == TRAINABLE_PARAMETER_COUNT,
        "optimizer_field_and_last_block_only": validate_optimizer(field, atom_head, vision, optimizer) is None,
        "all_field_tensors_changed": all(changed.values()),
        "atom_state_unchanged": module_state_sha256(atom_head) == atom_sha,
        "vision_state_changed": module_state_sha256(vision) != vision_initial_sha,
        "last_block_changed": module_state_sha256(block) != block_initial_sha,
        "frozen_base_block_unchanged": module_state_sha256(frozen_block) == frozen_block_sha,
        "frozen_vision_unchanged": frozen_vision_sha256(vision) == frozen_initial_sha,
        "step0_base_exact": parity["exact"],
        "fixed_steps": step == STEPS,
        "fixed_exposure": exposure == STEPS * args.batch_size,
        "repeat_peak_digest": metrics["peak_source_ids_sha256"] == repeated["peak_source_ids_sha256"],
        "repeat_metrics": metrics == repeated,
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
    final_path = args.output_dir / "step-002400.pt"
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
