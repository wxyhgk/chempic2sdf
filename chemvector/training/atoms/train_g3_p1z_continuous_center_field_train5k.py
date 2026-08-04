#!/usr/bin/env python3
"""Fine-tune one continuous Atom center field from capacity64 to train5k."""

from __future__ import annotations

import argparse
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
    ContinuousStride2AtomCenterField,
)
from chemvector.evaluation.audits.atoms import audit_g3_p1w_multiscale_peak_rank as p1w
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.atoms import train_g3_p1d_boundary_band_set_ranker as p1d
from chemvector.training.atoms import train_g3_p1t_continuous_stride2_center_field as p1t
from chemvector.training.atoms import train_g3_p1u_adjacent_atom_capacity as p1u
from chemvector.training.atoms import train_g3_p1v_continuous_stride1_capacity as p1v
from chemvector.evaluation.audits.atoms import build_g3_p2f_train25k_manifest as p2f
from chemvector.training.graph.scenes.g3_structural_data import load_g3_sample_manifest
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _atomic_torch_save,
    load_frozen_vision_checkpoint,
    module_state_sha256,
)
from chemvector.g3_structural_data import file_sha256


SCALE_PROTOCOLS = {
    "stride2": {
        "model_type": "chemvector-g3-p1z-stride2-center-field-train5k",
        "output": Path("outputs/checkpoints/chemvector-g3-p1z-stride2-center-field-train5k"),
        "checkpoint": p1w.P1U_CHECKPOINT,
        "checkpoint_sha256": p1w.P1U_CHECKPOINT_SHA256,
        "source_model_type": p1u.MODEL_TYPE,
        "field_factory": ContinuousStride2AtomCenterField,
        "field_parameters": 12_835,
        "trainable_parameters": 92_035,
        "step0_matched": 2_989,
        "step0_exact": 49,
        "epochs": 5,
        "gate_mode": "relative",
    },
    "stride1": {
        "model_type": "chemvector-g3-p2a-stride1-center-field-train5k",
        "output": Path("outputs/checkpoints/chemvector-g3-p2a-stride1-center-field-train5k"),
        "checkpoint": p1w.P1V_CHECKPOINT,
        "checkpoint_sha256": p1w.P1V_CHECKPOINT_SHA256,
        "source_model_type": p1v.MODEL_TYPE,
        "field_factory": ContinuousStride1AtomCenterField,
        "field_parameters": 17_379,
        "trainable_parameters": 96_579,
        "step0_matched": 2_912,
        "step0_exact": 26,
        "epochs": 5,
        "gate_mode": "relative",
    },
}
STEPS = 6_250
SAVE_EVERY = 1_250
EXPECTED_EXPOSURE = 25_000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=tuple(SCALE_PROTOCOLS), required=True)
    parser.add_argument(
        "--training-set", choices=("train5k", "train25k"), default="train5k"
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--resume-step2500", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> dict:
    fixed = {
        "batch_size": 4,
        "learning_rate": 3e-4,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "seed": 20260713,
    }
    for name, expected in fixed.items():
        if getattr(args, name) != expected:
            raise ValueError(f"P1-Z requires fixed {name}={expected}")
    if args.log_every < 1:
        raise ValueError("P1-Z log interval must be positive")
    if args.resume_step2500 and args.training_set != "train25k":
        raise ValueError("P2-F step2500 resume is only valid for train25k")
    if not torch.cuda.is_available():
        raise RuntimeError("P1-Z requires CUDA BF16")
    if args.training_set == "train25k":
        if args.scale != "stride1":
            raise ValueError("P2-F train25k supports only stride1")
        return {
            **SCALE_PROTOCOLS["stride1"],
            "model_type": "chemvector-g3-p2f-stride1-center-field-train25k",
            "output": Path("outputs/checkpoints/chemvector-g3-p2f-stride1-center-field-train25k"),
            "checkpoint": SCALE_PROTOCOLS["stride1"]["output"] / "step-006250.pt",
            "checkpoint_sha256": "2869451245f50f98f5e3f5259179ca338e82ad81a8fd59de2d2037a10056404b",
            "source_model_type": SCALE_PROTOCOLS["stride1"]["model_type"],
            "step0_matched": 3_073,
            "step0_exact": 98,
            "epochs": 1,
            "gate_mode": "absolute",
        }
    return SCALE_PROTOCOLS[args.scale]


def validate_optimizer(field, atom_head, vision, optimizer, protocol) -> None:
    block = p1t.trainable_vision_block(vision)
    expected = {
        id(value)
        for module in (field, block)
        for value in module.parameters()
        if value.requires_grad
    }
    actual = {
        id(value) for group in optimizer.param_groups for value in group["params"]
    }
    if expected != actual:
        raise ValueError("P1-Z optimizer parameter identity differs")
    if field.trainable_parameter_count() != protocol["field_parameters"]:
        raise ValueError("P1-Z field parameter count differs")
    if sum(value.numel() for value in expected_parameters(field, block)) != protocol[
        "trainable_parameters"
    ]:
        raise ValueError("P1-Z trainable parameter count differs")
    if any(value.requires_grad for value in atom_head.parameters()):
        raise ValueError("P1-Z P1-A Atom head became trainable")


def expected_parameters(field, block):
    return [
        value
        for module in (field, block)
        for value in module.parameters()
        if value.requires_grad
    ]


def gate(protocol: dict, step0: dict, final: dict) -> tuple[str, dict[str, bool]]:
    step0_reach = step0["top128_reachability"]
    final_reach = final["top128_reachability"]
    checks = {
        "step0_matches_independent_audit": step0_reach["matched_targets"]
        == protocol["step0_matched"],
        "step0_exact_matches_independent_audit": step0_reach["exact_images"]
        == protocol["step0_exact"],
        "route_capability": (
            final_reach["matched_targets"] >= 3_075
            and final_reach["exact_images"] >= 99
            if protocol["gate_mode"] == "absolute"
            else final_reach["atom_recall"] >= step0_reach["atom_recall"] + 0.005
            and final_reach["exact_images"] >= step0_reach["exact_images"] + 10
        ),
    }
    return (
        "route_causal_go" if all(checks.values()) else "route_causal_no_go",
        checks,
    )


def checkpoint_payload(
    field, vision_block, optimizer, args, protocol, lineage,
    *, step, exposure, losses, metrics,
):
    return {
        "checkpoint_schema_version": "g3-continuous-center-field-train5k-0.1",
        "model_type": protocol["model_type"],
        "scale": args.scale,
        "atom_center_field": field.state_dict(),
        "vision_last_stride4_block": vision_block.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": step,
        "samples_seen": exposure,
        "training_args": vars(args),
        "trainable_parameter_count": protocol["trainable_parameters"],
        "center_target": "one_hot",
        "offset_loss_weight": 8.0,
        "peak_margin_weight": 1.0,
        "peak_margin": 1.0,
        "pixel_cluster_radius": None,
        "initialization": "capacity64-checkpoint-reset-optimizer",
        "lineage": lineage,
        "last_losses": losses,
        "metrics": metrics,
    }


def main() -> None:
    args = build_parser().parse_args()
    protocol = validate_args(args)
    output_dir = protocol["output"]
    if not args.resume_step2500 and output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"P1-Z output directory is not empty: {output_dir}")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")

    source_checkpoint = protocol["checkpoint"]
    if file_sha256(source_checkpoint) != protocol["checkpoint_sha256"]:
        raise ValueError("P1-Z capacity checkpoint SHA256 differs")
    source_payload = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    if source_payload.get("model_type") != protocol["source_model_type"]:
        raise ValueError("P1-Z capacity checkpoint type differs")
    atom_head, _, p1a_payload, _ = p1d.load_p1a_boundary_ranker(device)
    atom_head.requires_grad_(False).eval()
    vision, _, _, vision_sha = load_frozen_vision_checkpoint(
        Path(p1a_payload["lineage"]["vision"]["path"]), device
    )
    vision.requires_grad_(False).eval()
    block = p1t.trainable_vision_block(vision)
    frozen_block = copy.deepcopy(block).requires_grad_(False).eval()
    block.load_state_dict(source_payload["vision_last_stride4_block"], strict=True)
    block.requires_grad_(True)
    field = protocol["field_factory"](atom_head).to(device)
    field.load_state_dict(source_payload["atom_center_field"], strict=True)
    field.requires_grad_(True)
    initial_field_sha = module_state_sha256(field)
    initial_block_sha = module_state_sha256(block)
    frozen_block_sha = module_state_sha256(frozen_block)
    atom_sha = module_state_sha256(atom_head)
    frozen_vision_sha = p1t.frozen_vision_sha256(vision)
    optimizer = torch.optim.AdamW(
        expected_parameters(field, block),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    validate_optimizer(field, atom_head, vision, optimizer, protocol)
    resume_payload = None
    if args.resume_step2500:
        resume_path = output_dir / "step-002500.pt"
        resume_payload = torch.load(resume_path, map_location=device, weights_only=False)
        if (
            resume_payload.get("model_type") != protocol["model_type"]
            or resume_payload.get("global_step") != 2_500
            or resume_payload.get("samples_seen") != 10_000
        ):
            raise ValueError("P2-F step2500 resume checkpoint contract differs")
        field.load_state_dict(resume_payload["atom_center_field"], strict=True)
        block.load_state_dict(resume_payload["vision_last_stride4_block"], strict=True)
        optimizer.load_state_dict(resume_payload["optimizer"])

    if args.training_set == "train25k":
        train, train_audit = load_g3_sample_manifest(
            p2f.MANIFEST,
            expected_sha256="b0f10eb0922599636895e19294e55b95ba9c051492b7bd69ae42a639c2846161",
            expected_samples=25_000,
            expected_molecules=6_250,
            expected_variants=p1a.EXPECTED_VARIANTS,
        )
    else:
        train, train_audit = p1a.load_train_manifest()
    route, route_audit = p1a.load_route_manifest()
    collator = p1a.G3AtomCollator(maximum_atoms=128, image_size=768)
    lineage = {
        "capacity_checkpoint": str(source_checkpoint.resolve()),
        "capacity_checkpoint_sha256": protocol["checkpoint_sha256"],
        "capacity_model_type": protocol["source_model_type"],
        "p1a_checkpoint": str(p1d.P1A_CHECKPOINT.resolve()),
        "p1a_checkpoint_sha256": file_sha256(p1d.P1A_CHECKPOINT),
        "vision_checkpoint_sha256": vision_sha,
        "atom_state_sha256": atom_sha,
        "field_initial_state_sha256": initial_field_sha,
        "vision_block_initial_state_sha256": initial_block_sha,
        "frozen_base_block_sha256": frozen_block_sha,
        "frozen_vision_initial_sha256": frozen_vision_sha,
        "train5k_manifest": train_audit,
        "route100_manifest": route_audit,
        "optimizer_state_loaded_from_capacity": False,
    }
    output_dir.mkdir(parents=True, exist_ok=args.resume_step2500)
    _atomic_json(output_dir / "training_args.json", vars(args))
    if args.resume_step2500:
        step0 = json.loads(
            (output_dir / "step0_route_metrics.json").read_text(encoding="utf-8")
        )
    else:
        step0 = p1t.evaluate(
            field, atom_head, vision, frozen_block, route, collator, device=device
        )
        _atomic_json(output_dir / "step0_route_metrics.json", step0)

    step = 2_500 if args.resume_step2500 else 0
    exposure = 10_000 if args.resume_step2500 else 0
    losses = resume_payload.get("last_losses", {}) if resume_payload else {}
    started = time.perf_counter()
    log_path = output_dir / (
        "train-resumed-step2500.jsonl" if args.resume_step2500 else "train.jsonl"
    )
    epoch_key_sha = None
    training_manifest = train[10_000:] if args.resume_step2500 else train
    expected_training_sha = hashlib.sha256(
        "".join(f"{record.sample_key}\n" for record in training_manifest).encode()
    ).hexdigest()
    for epoch in range(1, protocol["epochs"] + 1):
        key_digest = hashlib.sha256()
        for batch in p1a.iter_manifest_batches(training_manifest, collator, batch_size=4):
            for key in batch["keys"]:
                key_digest.update(f"{key}\n".encode())
            field.train()
            vision.eval()
            losses = p1t.training_step(
                field, atom_head, vision, frozen_block, optimizer, batch,
                device=device, max_grad_norm=args.max_grad_norm,
            )
            step += 1
            exposure += len(batch["keys"])
            if step % args.log_every == 0:
                record = {
                    "step": step,
                    "epoch": epoch,
                    "exposure": exposure,
                    "elapsed_seconds": time.perf_counter() - started,
                    **losses,
                }
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                print(json.dumps(record, separators=(",", ":")), flush=True)
            if step % SAVE_EVERY == 0 and step < STEPS:
                payload = checkpoint_payload(
                    field, block, optimizer, args, protocol, lineage,
                    step=step, exposure=exposure, losses=losses, metrics=None,
                )
                _atomic_torch_save(output_dir / f"step-{step:06d}.pt", payload)
                _atomic_torch_save(output_dir / "last.pt", payload)
        current_sha = key_digest.hexdigest()
        if epoch_key_sha is None:
            epoch_key_sha = current_sha
        elif current_sha != epoch_key_sha:
            raise RuntimeError("P1-Z train5k epoch order changed")

    final_route = p1t.evaluate(
        field, atom_head, vision, frozen_block, route, collator, device=device
    )
    repeated_route = p1t.evaluate(
        field, atom_head, vision, frozen_block, route, collator, device=device
    )
    decision, checks = gate(protocol, step0, final_route)
    protocol_checks = {
        "fixed_steps": step == STEPS,
        "fixed_exposure": exposure == EXPECTED_EXPOSURE,
        "fixed_epochs": epoch == protocol["epochs"],
        "fixed_train_sequence": epoch_key_sha == expected_training_sha,
        "field_changed": module_state_sha256(field) != initial_field_sha,
        "vision_block_changed": module_state_sha256(block) != initial_block_sha,
        "frozen_base_block_unchanged": module_state_sha256(frozen_block)
        == frozen_block_sha,
        "atom_unchanged": module_state_sha256(atom_head) == atom_sha,
        "frozen_vision_unchanged": p1t.frozen_vision_sha256(vision)
        == frozen_vision_sha,
        "optimizer_scope": validate_optimizer(
            field, atom_head, vision, optimizer, protocol
        ) is None,
        "repeat_route_metrics": repeated_route == final_route,
    }
    if not all(protocol_checks.values()):
        decision = "protocol_failed"
    final = {
        "decision": decision,
        "checks": checks,
        "protocol_checks": protocol_checks,
        "step0_route_metrics": step0,
        "final_route_metrics": final_route,
        "elapsed_seconds": time.perf_counter() - started,
    }
    payload = checkpoint_payload(
        field, block, optimizer, args, protocol, lineage,
        step=step, exposure=exposure, losses=losses, metrics=final,
    )
    checkpoint = output_dir / "step-006250.pt"
    _atomic_torch_save(checkpoint, payload)
    _atomic_torch_save(output_dir / "last.pt", payload)
    final["checkpoint"] = str(checkpoint.resolve())
    final["checkpoint_sha256"] = file_sha256(checkpoint)
    _atomic_json(output_dir / "final_metrics.json", final)
    _atomic_json(output_dir / "status.json", {
        "status": "completed",
        "decision": decision,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": final["checkpoint_sha256"],
    })
    print(json.dumps({
        "model_type": protocol["model_type"],
        "decision": decision,
        "step0": step0["top128_reachability"],
        "final": final_route["top128_reachability"],
    }, indent=2))


if __name__ == "__main__":
    main()
