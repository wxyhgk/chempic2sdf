#!/usr/bin/env python3
"""Train the 771-parameter P1-O Atom center field on frozen P1-A features."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from chemvector.components.atoms.objects.g3_atom_center_field import (
    G3AtomCenterField,
    atom_center_field_loss,
    decode_atom_center_field,
    warmstart_atom_center_field_from_p1a,
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


MODEL_TYPE = "chemvector-g3-p1o-atom-center-field-train5k"
PARAMETER_COUNT = 771
OVERFIT_OUTPUT = Path("outputs/smoke/chemvector-g3-p1o-atom-center-field-overfit32")
TRAIN_OUTPUT = Path("outputs/checkpoints/chemvector-g3-p1o-atom-center-field-train5k")
PHASE_CONFIG = {
    "overfit32": {"steps": 2_400, "save_every": 400, "images": 32},
    "train5k": {"steps": 6_250, "save_every": 1_250, "images": 5_000},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=tuple(PHASE_CONFIG), required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--log-every", type=int, default=50)
    return parser


def validate_args(args) -> Path:
    fixed = {
        "batch_size": 4,
        "learning_rate": 3e-4,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "seed": 20260713,
    }
    for name, expected in fixed.items():
        if getattr(args, name) != expected:
            raise ValueError(f"P1-O requires fixed {name}={expected}")
    if args.device != "cuda" or args.precision != "bf16":
        raise ValueError("P1-O requires fixed CUDA BF16")
    if not torch.cuda.is_available():
        raise RuntimeError("P1-O requires CUDA")
    expected = OVERFIT_OUTPUT if args.phase == "overfit32" else TRAIN_OUTPUT
    output = expected if args.output_dir is None else args.output_dir
    if output.resolve() != expected.resolve():
        raise ValueError(f"P1-O {args.phase} requires output directory {expected}")
    return output


def validate_optimizer(field, atom_head, vision, optimizer) -> None:
    expected = {id(value) for value in field.parameters()}
    actual = {
        id(value) for group in optimizer.param_groups for value in group["params"]
    }
    if expected != actual or sum(value.numel() for value in field.parameters()) != PARAMETER_COUNT:
        raise ValueError("P1-O optimizer must contain exactly 771 field parameters")
    if any(value.requires_grad for value in atom_head.parameters()):
        raise ValueError("P1-O P1-A Atom head became trainable")
    if any(value.requires_grad for value in vision.parameters()):
        raise ValueError("P1-O vision became trainable")


@torch.no_grad()
def frozen_hidden(atom_head, vision, pixels, *, device, precision):
    with _autocast_context(device, precision):
        features = vision.forward_stroke_resolution(pixels.to(device))
        dense = atom_head(features.detach())
    return dense["object_features"].detach()


@torch.no_grad()
def warmstart_parity(field, atom_head, hidden) -> dict[str, float | bool]:
    hidden = hidden.detach().float()
    field_output = field(hidden)
    expected_center = atom_head.presence(hidden)[:, 0:1]
    expected_offset = atom_head.anchor_offset(hidden).reshape(
        len(hidden), 3, 2, hidden.size(2), hidden.size(3)
    )[:, 0]
    center_error = float((field_output["center_logits"] - expected_center).abs().max())
    offset_error = float((field_output["offset_logits"] - expected_offset).abs().max())
    return {
        "center_max_abs_error": center_error,
        "offset_max_abs_error": offset_error,
        "exact": center_error == 0.0 and offset_error == 0.0,
    }


def _empty_metrics() -> dict:
    return p1a._empty_atom_totals()


def _finalize_metrics(value: dict) -> dict:
    result = p1a._finalize_atom_totals(value)
    result["exact_rate_6px"] = result["whole_set_exact_images"] / max(result["images"], 1)
    return result


@torch.no_grad()
def evaluate_center_field(field, atom_head, vision, manifest, collator, *, device):
    field.eval()
    known = _empty_metrics()
    reach_targets = reach_matched = reach_exact = 0
    maxima_counts = []
    shortfall = 0
    digest = hashlib.sha256()
    sample_digest = hashlib.sha256()
    variants: dict[str, dict] = {}
    for batch in p1a.iter_manifest_batches(manifest, collator, batch_size=4):
        for key in batch["keys"]:
            sample_digest.update(f"{key}\n".encode())
        hidden = frozen_hidden(
            atom_head,
            vision,
            batch["pixel_values"],
            device=device,
            precision="bf16",
        )
        with _autocast_context(device, "bf16"):
            output = field(hidden)
        peaks = extract_regional_maximum_objects(decode_atom_center_field(output))
        for index, peak in enumerate(peaks):
            expected = batch["atom_anchors"][index, batch["atom_mask"][index]].float()
            count = len(expected)
            predicted = peak["anchors"][:count]
            current = p1a.atom_sample_statistics(predicted, expected)
            p1a._merge_atom_totals(known, current)
            top = p1a.atom_sample_statistics(peak["anchors"], expected)
            reach_targets += count
            reach_matched += int(top["matched_atoms"])
            reach_exact += int(top["matched_atoms"] == count)
            maxima_count = int(peak["regional_maxima_count"])
            maxima_counts.append(maxima_count)
            shortfall += int(maxima_count < count)
            variant = str(batch["variants"][index])
            variants.setdefault(variant, _empty_metrics())
            p1a._merge_atom_totals(variants[variant], current)
            digest.update(peak["source_ids"].numpy().tobytes())
    return {
        "known_full_count": _finalize_metrics(known),
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
        "variants": {
            name: _finalize_metrics(value) for name, value in sorted(variants.items())
        },
        "sample_keys_sha256": sample_digest.hexdigest(),
        "peak_source_ids_sha256": digest.hexdigest(),
    }


def training_step(field, atom_head, vision, optimizer, batch, *, device, max_grad_norm):
    optimizer.zero_grad(set_to_none=True)
    hidden = frozen_hidden(
        atom_head, vision, batch["pixel_values"], device=device, precision="bf16"
    )
    with _autocast_context(device, "bf16"):
        output = field(hidden)
        losses = atom_center_field_loss(
            output,
            batch["atom_anchors"].to(device),
            batch["atom_mask"].to(device),
            adaptive_gaussian=False,
        )
    losses["loss"].backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(field.parameters(), max_grad_norm)
    if not bool(torch.isfinite(gradient_norm)):
        raise FloatingPointError("P1-O gradient norm is not finite")
    optimizer.step()
    if any(value.grad is not None for value in atom_head.parameters()):
        raise RuntimeError("P1-O produced P1-A gradients")
    if any(value.grad is not None for value in vision.parameters()):
        raise RuntimeError("P1-O produced vision gradients")
    return {
        **{name: float(value.detach()) for name, value in losses.items()},
        "gradient_norm": float(gradient_norm.detach()),
    }


def _checkpoint_payload(field, optimizer, args, lineage, *, step, exposure, last_losses, metrics=None):
    return {
        "checkpoint_schema_version": "g3-p1o-atom-center-field-0.1",
        "model_type": MODEL_TYPE,
        "phase": args.phase,
        "atom_center_field": field.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": step,
        "samples_seen": exposure,
        "training_args": vars(args),
        "lineage": lineage,
        "parameter_count": PARAMETER_COUNT,
        "adaptive_gaussian": False,
        "pixel_cluster_radius": None,
        "last_losses": last_losses,
        "metrics": metrics,
    }


def _phase_gate(phase: str, metrics: dict) -> tuple[str, dict[str, bool]]:
    known = metrics["known_full_count"]
    reach = metrics["top128_reachability"]
    if phase == "overfit32":
        checks = {
            "known_all_atoms": known["matched_atoms"] == known["target_atoms"] == 1_424,
            "known_exact_32": known["whole_set_exact_images"] == 32,
            "reach_all_atoms": reach["matched_targets"] == reach["targets"] == 1_424,
            "reach_exact_32": reach["exact_images"] == 32,
            "no_peak_shortfall": metrics["regional_maxima"]["peak_shortfall_images"] == 0,
        }
        return ("capacity_go" if all(checks.values()) else "capacity_no_go", checks)
    checks = {
        "known_recall_99_6": known["recall_6px"] >= 0.996,
        "known_exact_92": known["exact_rate_6px"] >= 0.92,
        "reach_recall_99_95": reach["atom_recall"] >= 0.9995,
        "reach_exact_99": reach["exact_rate"] >= 0.99,
        "no_peak_shortfall": metrics["regional_maxima"]["peak_shortfall_images"] == 0,
    }
    return (
        "mechanism_go_requires_new_fresh_confirmation"
        if all(checks.values())
        else "capability_no_go",
        checks,
    )


def main() -> None:
    args = build_parser().parse_args()
    output = validate_args(args)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("P1-O output directory is not empty")
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
    field = G3AtomCenterField().to(device)
    warmstart_atom_center_field_from_p1a(field, atom_head)
    initialization_sha = module_state_sha256(field)
    optimizer = torch.optim.AdamW(
        field.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    validate_optimizer(field, atom_head, vision, optimizer)

    if args.phase == "overfit32":
        manifest, manifest_audit = load_g3_overfit32_manifest(OVERFIT32_MANIFEST)
    else:
        manifest, manifest_audit = p1a.load_train_manifest()
    collator = p1a.G3AtomCollator(maximum_atoms=128, image_size=768)
    first_batch = next(p1a.iter_manifest_batches(manifest, collator, batch_size=4))
    first_hidden = frozen_hidden(
        atom_head, vision, first_batch["pixel_values"], device=device, precision="bf16"
    )
    parity = warmstart_parity(field, atom_head, first_hidden)
    if not parity["exact"]:
        raise RuntimeError("P1-O slot-0 warmstart parity failed")

    config = PHASE_CONFIG[args.phase]
    lineage = {
        "p1a_checkpoint": str(p1d.P1A_CHECKPOINT.resolve()),
        "p1a_checkpoint_sha256": file_sha256(p1d.P1A_CHECKPOINT),
        "atom_state_sha256": atom_sha,
        "vision_checkpoint_sha256": vision_sha,
        "vision_state_sha256": vision_state_sha,
        "manifest": manifest_audit,
        "field_initialization_sha256": initialization_sha,
        "warmstart_parity": parity,
    }
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "train.jsonl"
    step = exposure = 0
    last_losses = {}
    started = time.perf_counter()
    epoch = 0
    while step < config["steps"]:
        epoch += 1
        for batch in p1a.iter_manifest_batches(manifest, collator, batch_size=4):
            last_losses = training_step(
                field,
                atom_head,
                vision,
                optimizer,
                batch,
                device=device,
                max_grad_norm=args.max_grad_norm,
            )
            step += 1
            exposure += len(batch["keys"])
            if step % args.log_every == 0:
                record = {"step": step, "exposure": exposure, **last_losses}
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                print(json.dumps(record, separators=(",", ":")), flush=True)
            if step % config["save_every"] == 0 and step < config["steps"]:
                payload = _checkpoint_payload(
                    field, optimizer, args, lineage, step=step, exposure=exposure,
                    last_losses=last_losses,
                )
                _atomic_torch_save(output / f"step-{step:06d}.pt", payload)
                _atomic_torch_save(output / "last.pt", payload)
            if step >= config["steps"]:
                break

    metrics = evaluate_center_field(
        field, atom_head, vision, manifest, collator, device=device
    )
    decision, checks = _phase_gate(args.phase, metrics)
    frozen_checks = {
        "parameter_count_771": field.parameter_count() == PARAMETER_COUNT,
        "optimizer_field_only": validate_optimizer(field, atom_head, vision, optimizer) is None,
        "atom_state_unchanged": module_state_sha256(atom_head) == atom_sha,
        "vision_state_unchanged": module_state_sha256(vision) == vision_state_sha,
        "warmstart_exact": parity["exact"],
        "fixed_steps": step == config["steps"],
        "fixed_exposure": exposure == config["steps"] * args.batch_size,
    }
    if not all(frozen_checks.values()):
        decision = "protocol_failed"
    final = {
        "decision": decision,
        "checks": checks,
        "protocol_checks": frozen_checks,
        "metrics": metrics,
        "elapsed_seconds": time.perf_counter() - started,
    }
    payload = _checkpoint_payload(
        field, optimizer, args, lineage, step=step, exposure=exposure,
        last_losses=last_losses, metrics=final,
    )
    final_path = output / f"step-{step:06d}.pt"
    _atomic_torch_save(final_path, payload)
    _atomic_torch_save(output / "last.pt", payload)
    final["checkpoint"] = str(final_path.resolve())
    final["checkpoint_sha256"] = file_sha256(final_path)
    _atomic_json(output / "final_metrics.json", final)
    _atomic_json(output / "status.json", {
        "status": "completed",
        "phase": args.phase,
        "decision": decision,
        "checkpoint": str(final_path.resolve()),
        "checkpoint_sha256": final["checkpoint_sha256"],
    })
    print(json.dumps({"decision": decision, "checkpoint": str(final_path)}, indent=2))


if __name__ == "__main__":
    main()
