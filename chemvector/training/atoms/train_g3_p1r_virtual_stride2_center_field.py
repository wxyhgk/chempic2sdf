#!/usr/bin/env python3
"""Capacity-test a virtual stride-2 Atom center field on frozen stroke features."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from chemvector.components.atoms.objects.g3_atom_center_field import (
    VirtualStride2AtomCenterField,
    atom_center_field_loss,
    decode_atom_center_field,
)
from chemvector.evaluation.audits.atoms.audit_g3_p1n_regional_maxima import (
    extract_regional_maximum_objects,
)
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.atoms import train_g3_p1d_boundary_band_set_ranker as p1d
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


MODEL_TYPE = "chemvector-g3-p1r-virtual-stride2-center-field-overfit32"
OUTPUT = Path("outputs/smoke/chemvector-g3-p1r-virtual-stride2-center-field-overfit32")
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
            raise ValueError(f"P1-R requires fixed {name}={expected}")
    if args.output_dir.resolve() != OUTPUT.resolve():
        raise ValueError(f"P1-R requires output directory {OUTPUT}")
    if not torch.cuda.is_available():
        raise RuntimeError("P1-R requires CUDA BF16")


def validate_optimizer(field, atom_head, vision, optimizer) -> None:
    expected = {id(value) for value in field.parameters() if value.requires_grad}
    actual = {id(value) for group in optimizer.param_groups for value in group["params"]}
    if expected != actual or field.trainable_parameter_count() != PARAMETER_COUNT:
        raise ValueError("P1-R optimizer must contain exactly 7,628 field parameters")
    if any(value.requires_grad for value in atom_head.parameters()):
        raise ValueError("P1-R P1-A Atom head became trainable")
    if any(value.requires_grad for value in vision.parameters()):
        raise ValueError("P1-R vision became trainable")


@torch.no_grad()
def frozen_stroke(vision, pixels, *, device) -> torch.Tensor:
    with _autocast_context(device, "bf16"):
        return vision.forward_stroke_resolution(pixels.to(device)).detach()


def training_step(field, atom_head, vision, optimizer, batch, *, device, max_grad_norm):
    optimizer.zero_grad(set_to_none=True)
    stroke = frozen_stroke(vision, batch["pixel_values"], device=device)
    with _autocast_context(device, "bf16"):
        output = field(stroke)
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
        raise FloatingPointError("P1-R gradient norm is not finite")
    optimizer.step()
    if any(value.grad is not None for value in atom_head.parameters()):
        raise RuntimeError("P1-R produced P1-A gradients")
    if any(value.grad is not None for value in vision.parameters()):
        raise RuntimeError("P1-R produced vision gradients")
    return {
        **{name: float(value.detach()) for name, value in losses.items()},
        "gradient_norm": float(gradient_norm.detach()),
    }


@torch.no_grad()
def evaluate_inference(manifest, collator, infer_batch) -> dict:
    """Evaluate an Atom center-field inference callback under the P1-R gate."""

    known = p1a._empty_atom_totals()
    reach_targets = reach_matched = reach_exact = 0
    maxima_counts: list[int] = []
    shortfall = 0
    peak_digest = hashlib.sha256()
    sample_digest = hashlib.sha256()
    variants: dict[str, dict] = {}
    for batch in p1a.iter_manifest_batches(manifest, collator, batch_size=4):
        for key in batch["keys"]:
            sample_digest.update(f"{key}\n".encode())
        output = infer_batch(batch)
        peaks = extract_regional_maximum_objects(decode_atom_center_field(output))
        for index, peak in enumerate(peaks):
            expected = batch["atom_anchors"][index, batch["atom_mask"][index]].float()
            count = len(expected)
            current = p1a.atom_sample_statistics(peak["anchors"][:count], expected)
            p1a._merge_atom_totals(known, current)
            top = p1a.atom_sample_statistics(peak["anchors"], expected)
            reach_targets += count
            reach_matched += int(top["matched_atoms"])
            reach_exact += int(top["matched_atoms"] == count)
            maxima_count = int(peak["regional_maxima_count"])
            maxima_counts.append(maxima_count)
            shortfall += int(maxima_count < count)
            variant = str(batch["variants"][index])
            variants.setdefault(variant, p1a._empty_atom_totals())
            p1a._merge_atom_totals(variants[variant], current)
            peak_digest.update(peak["source_ids"].numpy().tobytes())

    def finalize(value: dict) -> dict:
        result = p1a._finalize_atom_totals(value)
        result["exact_rate_6px"] = result["whole_set_exact_images"] / max(result["images"], 1)
        return result

    return {
        "known_full_count": finalize(known),
        "top128_reachability": {
            "targets": reach_targets,
            "matched_targets": reach_matched,
            "atom_recall": reach_matched / max(reach_targets, 1),
            "exact_images": reach_exact,
            "exact_rate": reach_exact / max(len(manifest), 1),
        },
        "regional_maxima": {
            "mean_per_image": float(np.mean(maxima_counts)),
            "p50_per_image": float(np.percentile(maxima_counts, 50)),
            "p90_per_image": float(np.percentile(maxima_counts, 90)),
            "maximum_per_image": int(max(maxima_counts)),
            "peak_shortfall_images": shortfall,
        },
        "variants": {name: finalize(value) for name, value in sorted(variants.items())},
        "sample_keys_sha256": sample_digest.hexdigest(),
        "peak_source_ids_sha256": peak_digest.hexdigest(),
    }


@torch.no_grad()
def evaluate(field, vision, manifest, collator, *, device) -> dict:
    field.eval()

    def infer_batch(batch):
        stroke = frozen_stroke(vision, batch["pixel_values"], device=device)
        with _autocast_context(device, "bf16"):
            return field(stroke)

    return evaluate_inference(manifest, collator, infer_batch)


def capacity_gate(metrics: dict) -> tuple[str, dict[str, bool]]:
    known = metrics["known_full_count"]
    reach = metrics["top128_reachability"]
    checks = {
        "known_all_atoms": known["matched_atoms"] == known["target_atoms"] == 1_424,
        "known_exact_32": known["whole_set_exact_images"] == 32,
        "reach_all_atoms": reach["matched_targets"] == reach["targets"] == 1_424,
        "reach_exact_32": reach["exact_images"] == 32,
        "no_peak_shortfall": metrics["regional_maxima"]["peak_shortfall_images"] == 0,
    }
    return (
        "overfit_capacity_go" if all(checks.values()) else "virtual_stride2_capacity_no_go",
        checks,
    )


def checkpoint_payload(field, optimizer, args, lineage, *, step, exposure, losses, metrics=None):
    return {
        "checkpoint_schema_version": "g3-p1r-virtual-stride2-center-field-0.1",
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
        "pixel_cluster_radius": None,
        "last_losses": losses,
        "metrics": metrics,
    }


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError("P1-R output directory is not empty")
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
    field = VirtualStride2AtomCenterField().to(device)
    initial_state = {name: value.detach().cpu().clone() for name, value in field.named_parameters()}
    initial_sha = module_state_sha256(field)
    optimizer = torch.optim.AdamW(
        field.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    validate_optimizer(field, atom_head, vision, optimizer)

    manifest, manifest_audit = load_g3_overfit32_manifest(OVERFIT32_MANIFEST)
    collator = p1a.G3AtomCollator(maximum_atoms=128, image_size=768)
    lineage = {
        "p1a_checkpoint": str(p1d.P1A_CHECKPOINT.resolve()),
        "p1a_checkpoint_sha256": file_sha256(p1d.P1A_CHECKPOINT),
        "atom_state_sha256": atom_sha,
        "vision_checkpoint_sha256": vision_sha,
        "vision_state_sha256": vision_state_sha,
        "manifest": manifest_audit,
        "field_initialization_sha256": initial_sha,
        "p1q_capacity_checkpoint": str(
            Path("outputs/smoke/chemvector-g3-p1q-high-resolution-center-field-overfit32/step-002400.pt").resolve()
        ),
        "p1q_capacity_decision": "high_resolution_capacity_no_go",
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

    metrics = evaluate(field, vision, manifest, collator, device=device)
    repeated = evaluate(field, vision, manifest, collator, device=device)
    decision, checks = capacity_gate(metrics)
    changed = {
        name: not torch.equal(initial_state[name], value.detach().cpu())
        for name, value in field.named_parameters()
    }
    protocol = {
        "parameter_count_7628": field.trainable_parameter_count() == PARAMETER_COUNT,
        "optimizer_field_only": validate_optimizer(field, atom_head, vision, optimizer) is None,
        "all_field_tensors_changed": all(changed.values()),
        "atom_state_unchanged": module_state_sha256(atom_head) == atom_sha,
        "vision_state_unchanged": module_state_sha256(vision) == vision_state_sha,
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
