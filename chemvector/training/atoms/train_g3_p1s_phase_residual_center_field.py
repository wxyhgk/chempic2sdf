#!/usr/bin/env python3
"""Capacity-test phase-aware residuals over the frozen P1-A center field."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from chemvector.components.atoms.objects.g3_atom_center_field import (
    VirtualStride2AtomCenterResidual,
    atom_center_field_loss,
)
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.atoms import train_g3_p1d_boundary_band_set_ranker as p1d
from chemvector.training.atoms import train_g3_p1r_virtual_stride2_center_field as p1r
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


MODEL_TYPE = "chemvector-g3-p1s-phase-residual-center-field-overfit32"
OUTPUT = Path("outputs/smoke/chemvector-g3-p1s-phase-residual-center-field-overfit32")
PARAMETER_COUNT = 7_628
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
            raise ValueError(f"P1-S requires fixed {name}={expected}")
    if args.output_dir.resolve() != OUTPUT.resolve():
        raise ValueError(f"P1-S requires output directory {OUTPUT}")
    if not torch.cuda.is_available():
        raise RuntimeError("P1-S requires CUDA BF16")


def validate_optimizer(field, atom_head, vision, optimizer) -> None:
    expected = {id(value) for value in field.parameters() if value.requires_grad}
    actual = {id(value) for group in optimizer.param_groups for value in group["params"]}
    if expected != actual or field.trainable_parameter_count() != PARAMETER_COUNT:
        raise ValueError("P1-S optimizer must contain exactly 7,628 residual parameters")
    if any(value.requires_grad for value in atom_head.parameters()):
        raise ValueError("P1-S P1-A Atom head became trainable")
    if any(value.requires_grad for value in vision.parameters()):
        raise ValueError("P1-S vision became trainable")


@torch.no_grad()
def frozen_inputs(atom_head, vision, pixels, *, device):
    with _autocast_context(device, "bf16"):
        stroke = vision.forward_stroke_resolution(pixels.to(device))
        dense = atom_head(stroke.detach())
    return dense["object_features"].detach(), stroke.detach()


@torch.no_grad()
def step0_parity(field, atom_head, object_features, stroke_features) -> dict:
    objects = object_features.detach().float()
    output = field(objects, stroke_features.detach().float())
    expected_center = F.interpolate(
        atom_head.presence(objects)[:, 0:1],
        scale_factor=2.0,
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


def training_step(field, atom_head, vision, optimizer, batch, *, device, max_grad_norm):
    optimizer.zero_grad(set_to_none=True)
    objects, stroke = frozen_inputs(
        atom_head, vision, batch["pixel_values"], device=device
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
    gradient_norm = torch.nn.utils.clip_grad_norm_(field.parameters(), max_grad_norm)
    if not bool(torch.isfinite(gradient_norm)):
        raise FloatingPointError("P1-S gradient norm is not finite")
    optimizer.step()
    if any(value.grad is not None for value in atom_head.parameters()):
        raise RuntimeError("P1-S produced P1-A gradients")
    if any(value.grad is not None for value in vision.parameters()):
        raise RuntimeError("P1-S produced vision gradients")
    return {
        **{name: float(value.detach()) for name, value in losses.items()},
        "gradient_norm": float(gradient_norm.detach()),
    }


@torch.no_grad()
def evaluate(field, atom_head, vision, manifest, collator, *, device) -> dict:
    field.eval()

    def infer_batch(batch):
        objects, stroke = frozen_inputs(
            atom_head, vision, batch["pixel_values"], device=device
        )
        with _autocast_context(device, "bf16"):
            return field(objects, stroke)

    return p1r.evaluate_inference(manifest, collator, infer_batch)


def capacity_gate(metrics: dict) -> tuple[str, dict[str, bool]]:
    decision, checks = p1r.capacity_gate(metrics)
    return (
        decision if decision == "overfit_capacity_go" else "phase_residual_capacity_no_go",
        checks,
    )


def checkpoint_payload(field, optimizer, args, lineage, *, step, exposure, losses, metrics=None):
    return {
        "checkpoint_schema_version": "g3-p1s-phase-residual-center-field-0.1",
        "model_type": MODEL_TYPE,
        "atom_center_field": field.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": step,
        "samples_seen": exposure,
        "training_args": vars(args),
        "lineage": lineage,
        "trainable_parameter_count": PARAMETER_COUNT,
        "center_target": "one_hot",
        "offset_loss_weight": 8.0,
        "peak_margin_weight": 1.0,
        "peak_margin": 1.0,
        "virtual_stride": 2,
        "frozen_p1a_base": True,
        "pixel_cluster_radius": None,
        "last_losses": losses,
        "metrics": metrics,
    }


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError("P1-S output directory is not empty")
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
    atom_sha = module_state_sha256(atom_head)
    vision_state_sha = module_state_sha256(vision)
    field = VirtualStride2AtomCenterResidual(atom_head).to(device)
    initial_state = {
        name: value.detach().cpu().clone() for name, value in field.named_parameters()
    }
    initial_sha = module_state_sha256(field)
    optimizer = torch.optim.AdamW(
        field.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    validate_optimizer(field, atom_head, vision, optimizer)

    manifest, manifest_audit = load_g3_overfit32_manifest(OVERFIT32_MANIFEST)
    collator = p1a.G3AtomCollator(maximum_atoms=128, image_size=768)
    first_batch = next(p1a.iter_manifest_batches(manifest, collator, batch_size=4))
    first_objects, first_stroke = frozen_inputs(
        atom_head, vision, first_batch["pixel_values"], device=device
    )
    parity = step0_parity(field, atom_head, first_objects, first_stroke)
    if not parity["exact"]:
        raise RuntimeError("P1-S zero-start P1-A base parity failed")
    lineage = {
        "p1a_checkpoint": str(p1d.P1A_CHECKPOINT.resolve()),
        "p1a_checkpoint_sha256": file_sha256(p1d.P1A_CHECKPOINT),
        "atom_state_sha256": atom_sha,
        "vision_checkpoint_sha256": vision_sha,
        "vision_state_sha256": vision_state_sha,
        "manifest": manifest_audit,
        "field_initialization_sha256": initial_sha,
        "step0_parity": parity,
        "p1r_checkpoint": str(
            (p1r.OUTPUT / "step-002400.pt").resolve()
        ),
        "p1r_checkpoint_sha256": file_sha256(p1r.OUTPUT / "step-002400.pt"),
        "p1r_decision": "virtual_stride2_capacity_no_go",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "train.jsonl"
    step = exposure = 0
    last_losses = {}
    started = time.perf_counter()
    while step < STEPS:
        for batch in p1a.iter_manifest_batches(manifest, collator, batch_size=4):
            field.train()
            last_losses = training_step(
                field, atom_head, vision, optimizer, batch,
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
                    field, optimizer, args, lineage, step=step, exposure=exposure,
                    losses=last_losses,
                )
                _atomic_torch_save(args.output_dir / f"step-{step:06d}.pt", payload)
                _atomic_torch_save(args.output_dir / "last.pt", payload)
            if step >= STEPS:
                break

    metrics = evaluate(field, atom_head, vision, manifest, collator, device=device)
    repeated = evaluate(field, atom_head, vision, manifest, collator, device=device)
    decision, checks = capacity_gate(metrics)
    changed = {
        name: not torch.equal(initial_state[name], value.detach().cpu())
        for name, value in field.named_parameters()
    }
    protocol = {
        "parameter_count_7628": field.trainable_parameter_count() == PARAMETER_COUNT,
        "optimizer_residual_only": validate_optimizer(field, atom_head, vision, optimizer) is None,
        "all_residual_tensors_changed": all(changed.values()),
        "atom_state_unchanged": module_state_sha256(atom_head) == atom_sha,
        "vision_state_unchanged": module_state_sha256(vision) == vision_state_sha,
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
        "changed_parameter_tensors": changed,
        "metrics": metrics,
        "elapsed_seconds": time.perf_counter() - started,
    }
    payload = checkpoint_payload(
        field, optimizer, args, lineage, step=step, exposure=exposure,
        losses=last_losses, metrics=final,
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
