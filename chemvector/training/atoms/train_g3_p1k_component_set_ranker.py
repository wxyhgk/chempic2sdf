#!/usr/bin/env python3
"""Train the P1-K ranker on target-free 3px Atom components."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from chemvector.components.atoms.g3_component_set_ranker import (
    G3ComponentSetRanker,
    build_dynamic_component_targets,
    build_target_free_components,
    component_boundary_band_loss,
    select_ranked_components,
)
from chemvector.components.atoms.g3_top128_set_ranker import (
    build_hybrid_top128_targets,
    extract_frozen_top128,
)
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.atoms import train_g3_p1b_frozen_presence_ranker as p1b
from chemvector.training.atoms import train_g3_p1d_boundary_band_set_ranker as p1d
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _atomic_torch_save,
    _autocast_context,
    load_frozen_vision_checkpoint,
    module_state_sha256,
)
from chemvector.g3_structural_data import file_sha256


DEFAULT_OUTPUT = Path(
    "outputs/checkpoints/chemvector-g3-p1k-component-set-ranker-train5k"
)
MODEL_TYPE = "chemvector-g3-p1k-component-set-ranker-train5k"
CHECKPOINT_STEPS = (1_250, 2_500, 3_750, 5_000, 6_250)
PARAMETER_COUNT = 60_609
STEP0_TRAIN_RECALL = 0.9817572670759339
STEP0_TRAIN_EXACT = 0.6422
STEP0_ROUTE_RECALL = 0.974609375
STEP0_ROUTE_EXACT = 0.53


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", default=str(p1a.TRAIN_MANIFEST))
    parser.add_argument("--route-manifest", default=str(p1a.ROUTE_SELECT_MANIFEST))
    parser.add_argument("--p1a-checkpoint", default=str(p1d.P1A_CHECKPOINT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=6_250)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=1_250)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260713)
    return parser


def validate_args(args):
    fixed = {
        "epochs": 5,
        "max_steps": 6_250,
        "batch_size": 4,
        "learning_rate": 3e-4,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "save_every": 1_250,
        "num_workers": 0,
        "seed": 20260713,
    }
    for name, expected in fixed.items():
        if getattr(args, name) != expected:
            raise ValueError(f"P1-K requires fixed {name}={expected}")
    if args.device != "cuda" or args.precision != "bf16":
        raise ValueError("P1-K requires fixed CUDA BF16")
    paths = {
        "train_manifest": p1a.TRAIN_MANIFEST,
        "route_manifest": p1a.ROUTE_SELECT_MANIFEST,
        "p1a_checkpoint": p1d.P1A_CHECKPOINT,
        "output_dir": DEFAULT_OUTPUT,
    }
    for name, expected in paths.items():
        if Path(getattr(args, name)).resolve() != expected.resolve():
            raise ValueError(f"P1-K requires fixed {name}: {expected}")


def validate_optimizer(atom_head, ranker, optimizer):
    expected = {id(value) for value in ranker.parameters()}
    actual = {
        id(value) for group in optimizer.param_groups for value in group["params"]
    }
    if expected != actual:
        raise ValueError("P1-K optimizer must contain only component ranker")
    if any(value.requires_grad for value in atom_head.parameters()):
        raise ValueError("P1-K Atom head became trainable")


def p1k_training_step(
    atom_head, ranker, vision, optimizer, batch, *, device, precision, max_grad_norm
):
    validate_optimizer(atom_head, ranker, optimizer)
    optimizer.zero_grad(set_to_none=True)
    pixels = batch["pixel_values"].to(device)
    targets = {
        "atom_anchors": batch["atom_anchors"].to(device),
        "atom_mask": batch["atom_mask"].to(device),
    }
    with torch.no_grad(), _autocast_context(device, precision):
        features = vision.forward_stroke_resolution(pixels)
        dense = atom_head(features.detach())
        candidates = extract_frozen_top128(dense)
        components = build_target_free_components(candidates)
    with _autocast_context(device, precision):
        ranked = ranker(components)
    fixed = build_hybrid_top128_targets(candidates, targets)
    dynamic = build_dynamic_component_targets(
        ranked, fixed, targets["atom_anchors"]
    )
    losses = component_boundary_band_loss(ranked, dynamic)
    losses["loss"].backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(ranker.parameters(), max_grad_norm)
    if not bool(torch.isfinite(gradient_norm)):
        raise FloatingPointError("P1-K gradient norm is not finite")
    optimizer.step()
    if any(value.grad is not None for value in atom_head.parameters()):
        raise RuntimeError("P1-K produced Atom gradients")
    if any(value.grad is not None for value in vision.parameters()):
        raise RuntimeError("P1-K produced vision gradients")
    metrics = {
        **{name: float(value.detach()) for name, value in losses.items()},
        "gradient_norm": float(gradient_norm.detach()),
        "components": float(components["component_counts"].sum()),
        "representable_targets": float(dynamic["representable_counts"].sum()),
        "batch_images": float(len(batch["keys"])),
    }
    return metrics, {"candidates": candidates, "components": components, "fixed": fixed}


def _empty_metrics():
    return {
        "images": 0,
        "predicted_atoms": 0,
        "target_atoms": 0,
        "matched_atoms": 0,
        "count_exact_images": 0,
        "whole_set_exact_images": 0,
        "count_absolute_error": 0,
    }


def _add_sample(total, predicted, expected):
    current = p1a.atom_sample_statistics(predicted.float().cpu(), expected.float().cpu())
    for name in total:
        total[name] += int(current[name])


def _finalize_metrics(total):
    precision = total["matched_atoms"] / total["predicted_atoms"]
    recall = total["matched_atoms"] / total["target_atoms"]
    return {
        **total,
        "precision_6px": precision,
        "recall_6px": recall,
        "f1_6px": 2 * precision * recall / (precision + recall),
        "count_mae": total["count_absolute_error"] / total["images"],
        "exact_rate_6px": total["whole_set_exact_images"] / total["images"],
    }


@torch.no_grad()
def evaluate_component_ranker(
    atom_head, ranker, vision, manifest, collator, *, device, require_zero_parity
):
    atom_head.eval()
    ranker.eval()
    vision.eval()
    known = _empty_metrics()
    deploy = _empty_metrics()
    sample_digest = hashlib.sha256()
    candidate_digest = hashlib.sha256()
    component_digest = hashlib.sha256()
    count_digest = hashlib.sha256()
    deploy_counts_all = []
    zero_parity = target_free_repeat = True
    for batch in p1a.iter_manifest_batches(manifest, collator, batch_size=4):
        for key in batch["keys"]:
            sample_digest.update(f"{key}\n".encode())
        pixels = batch["pixel_values"].to(device)
        targets = {
            "atom_anchors": batch["atom_anchors"].to(device),
            "atom_mask": batch["atom_mask"].to(device),
        }
        with _autocast_context(device, "bf16"):
            features = vision.forward_stroke_resolution(pixels)
            dense = atom_head(features.detach())
            candidates = extract_frozen_top128(dense)
            components = build_target_free_components(candidates)
            ranked = ranker(components)
            repeated = ranker(components)
        fixed = build_hybrid_top128_targets(candidates, targets)
        known_counts = fixed["representable_counts"].long()
        deploy_counts = dense["count_logits"].argmax(-1).long()
        deploy_counts = torch.minimum(deploy_counts, components["component_counts"])
        known_selected = select_ranked_components(ranked, known_counts)
        deploy_selected = select_ranked_components(ranked, deploy_counts)
        zero_parity &= torch.equal(
            ranked["ranking_logits"], components["base_logits"].float()
        )
        target_free_repeat &= torch.equal(
            ranked["ranking_logits"], repeated["ranking_logits"]
        )
        candidate_digest.update(candidates["candidate_ids"].cpu().numpy().tobytes())
        component_digest.update(
            components["source_component_labels"].cpu().numpy().tobytes()
        )
        count_digest.update(dense["count_logits"].float().cpu().numpy().tobytes())
        deploy_counts_all.extend(int(value) for value in deploy_counts.cpu().tolist())
        for index in range(len(batch["keys"])):
            fixed_mask = fixed["positive_mask"][index]
            target_indices = fixed["matched_target_indices"][index, fixed_mask]
            known_expected = targets["atom_anchors"][index, target_indices]
            known_mask = known_selected["atom_mask"][index]
            deploy_mask = deploy_selected["atom_mask"][index]
            all_expected = targets["atom_anchors"][index, targets["atom_mask"][index]]
            _add_sample(
                known,
                known_selected["atom_anchors"][index, known_mask],
                known_expected,
            )
            _add_sample(
                deploy,
                deploy_selected["atom_anchors"][index, deploy_mask],
                all_expected,
            )
    count_bytes = json.dumps(deploy_counts_all, separators=(",", ":")).encode()
    return {
        "images": len(manifest),
        "known_representable_count": _finalize_metrics(known),
        "deploy_count": _finalize_metrics(deploy),
        "sample_keys_sha256": sample_digest.hexdigest(),
        "candidate_ids_sha256": candidate_digest.hexdigest(),
        "component_labels_sha256": component_digest.hexdigest(),
        "count_logits_sha256": count_digest.hexdigest(),
        "deploy_count_vector_sha256": hashlib.sha256(count_bytes).hexdigest(),
        "zero_score_parity": zero_parity if require_zero_parity else None,
        "target_free_repeat": target_free_repeat,
    }


def _gate(protocol, step0_train, step0_route, final_train, final_route):
    train0 = step0_train["known_representable_count"]
    route0 = step0_route["known_representable_count"]
    train = final_train["known_representable_count"]
    route = final_route["known_representable_count"]
    causal = {
        "train_exact_plus_5pp": train["exact_rate_6px"] - train0["exact_rate_6px"] >= 0.05,
        "route_exact_plus_3pp": route["exact_rate_6px"] - route0["exact_rate_6px"] >= 0.03,
        "train_recall_no_regression": train["recall_6px"] >= train0["recall_6px"],
        "route_recall_no_regression": route["recall_6px"] >= route0["recall_6px"],
    }
    capability = {
        "train_exact_75": train["exact_rate_6px"] >= 0.75,
        "train_recall_99": train["recall_6px"] >= 0.99,
        "route_exact_60": route["exact_rate_6px"] >= 0.60,
        "route_recall_98": route["recall_6px"] >= 0.98,
    }
    protocol_passed = all(protocol.values())
    causal_passed = protocol_passed and all(causal.values())
    capability_passed = protocol_passed and all(capability.values())
    return {
        "protocol_checks": protocol,
        "protocol_passed": protocol_passed,
        "causal_checks": causal,
        "causal_passed": causal_passed,
        "capability_checks": capability,
        "capability_passed": capability_passed,
        "train_exact_gain_pp": 100 * (train["exact_rate_6px"] - train0["exact_rate_6px"]),
        "route_exact_gain_pp": 100 * (route["exact_rate_6px"] - route0["exact_rate_6px"]),
        "decision": (
            "protocol_failed" if not protocol_passed
            else "causal_go_capability_go" if causal_passed and capability_passed
            else "causal_go_capability_no_go" if causal_passed
            else "causal_no_go"
        ),
    }


def _payload(ranker, optimizer, args, lineage, audit, step, epoch, exposure, losses,
             step0_train, step0_route, final_train, final_route, gate, epoch_audits):
    return {
        "checkpoint_schema_version": "g3-p1k-component-ranker-0.1",
        "model_type": MODEL_TYPE,
        "component_ranker": ranker.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": step,
        "epoch": epoch,
        "samples_seen": exposure,
        "training_args": vars(args),
        "lineage": lineage,
        "audit": audit,
        "trainer_sha256": file_sha256(__file__),
        "last_losses": losses,
        "step0_train_metrics": step0_train,
        "step0_route_metrics": step0_route,
        "final_train_metrics": final_train,
        "final_route_metrics": final_route,
        "p1k_gate": gate,
        "completed_epoch_audits": epoch_audits,
    }


def main():
    args = build_parser().parse_args()
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("P1-K requires CUDA")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("P1-K output directory is not empty")
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

    train, train_audit = p1a.load_train_manifest(args.train_manifest)
    route, route_audit = p1a.load_route_manifest(args.route_manifest)
    collator = p1a.G3AtomCollator(maximum_atoms=128, image_size=768)
    train_preflight = p1a.stream_preflight(train, collator)
    route_preflight = p1a.stream_preflight(route, collator)
    overlap = p1a.identity_overlap_audit(train, route, train_preflight, route_preflight)
    if not overlap["all_three_identity_zero"]:
        raise RuntimeError("P1-K train/route identities overlap")
    atom_head, _, p1a_payload, config = p1d.load_p1a_boundary_ranker(device)
    vision_path = Path(p1a_payload["lineage"]["vision"]["path"])
    vision, _, _, vision_sha = load_frozen_vision_checkpoint(vision_path, device)
    if vision_sha != p1b.DEFAULT_VISION_CHECKPOINT_SHA256:
        raise RuntimeError("P1-K vision checkpoint differs")
    vision.requires_grad_(False).eval()
    atom_sha = module_state_sha256(atom_head)
    vision_state_sha = module_state_sha256(vision)
    ranker = G3ComponentSetRanker().to(device)
    if ranker.parameter_count() != PARAMETER_COUNT:
        raise RuntimeError("P1-K parameter count differs")
    optimizer = torch.optim.AdamW(ranker.parameters(), lr=3e-4, weight_decay=0.01)
    validate_optimizer(atom_head, ranker, optimizer)
    lineage = {
        "p1a": {"path": str(p1d.P1A_CHECKPOINT.resolve()), "sha256": p1d.P1A_CHECKPOINT_SHA256,
                 "atom_state_sha256": atom_sha},
        "vision": {"path": str(vision_path.resolve()), "sha256": vision_sha,
                   "state_sha256": vision_state_sha},
        "p1g_checkpoint_loaded": False,
    }
    audit = {
        "schema_version": "g3-p1k-training-audit-0.1",
        "scope": "target-free-3px-component-presence-ranker-train5k",
        "manifests": {"train": train_audit, "route": route_audit, "overlap": overlap},
        "preflight": {"train": train_preflight, "route": route_preflight},
        "lineage": lineage,
        "parameter_count": PARAMETER_COUNT,
        "component_radius_pixels": 3.0,
        "representative": "base_softmax",
        "loss": "valid-component-boundary-band-8x8",
        "forbidden_data_read": False,
        "implementation": {
            "trainer_sha256": file_sha256(__file__),
            "component_sha256": file_sha256(
                Path(__file__).parents[2] / "components/atoms/g3_component_set_ranker.py"
            ),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "audit.json", audit)
    _atomic_json(output / "training_args.json", vars(args))
    _atomic_json(output / "config.json", {"g3_config": asdict(config)})

    step0_train = evaluate_component_ranker(
        atom_head, ranker, vision, train, collator, device=device, require_zero_parity=True
    )
    step0_route = evaluate_component_ranker(
        atom_head, ranker, vision, route, collator, device=device, require_zero_parity=True
    )
    train0 = step0_train["known_representable_count"]
    route0 = step0_route["known_representable_count"]
    if (
        abs(train0["recall_6px"] - STEP0_TRAIN_RECALL) > 1e-12
        or abs(train0["exact_rate_6px"] - STEP0_TRAIN_EXACT) > 1e-12
        or abs(route0["recall_6px"] - STEP0_ROUTE_RECALL) > 1e-12
        or abs(route0["exact_rate_6px"] - STEP0_ROUTE_EXACT) > 1e-12
        or not step0_train["zero_score_parity"]
        or not step0_route["zero_score_parity"]
    ):
        raise RuntimeError("P1-K step0 does not reproduce P1-J")
    _atomic_json(output / "step0_train_metrics.json", step0_train)
    _atomic_json(output / "step0_route_metrics.json", step0_route)
    ranker.train()

    step = exposure = 0
    saved_steps = []
    epoch_audits = []
    losses = {}
    running = {}
    started = time.perf_counter()
    log_path = output / "training_log.jsonl"
    reference = None
    for epoch in range(1, 6):
        key_digest = hashlib.sha256()
        component_digest = hashlib.sha256()
        epoch_images = epoch_steps = 0
        for batch in p1a.iter_manifest_batches(train, collator, batch_size=4):
            for key in batch["keys"]:
                key_digest.update(f"{key}\n".encode())
            losses, evidence = p1k_training_step(
                atom_head, ranker, vision, optimizer, batch,
                device=device, precision="bf16", max_grad_norm=1.0
            )
            component_digest.update(
                evidence["components"]["source_component_labels"].cpu().numpy().tobytes()
            )
            step += 1
            epoch_steps += 1
            exposure += 4
            epoch_images += 4
            for name, value in losses.items():
                running[name] = running.get(name, 0.0) + value
            if step % args.log_every == 0:
                record = {"step": step, "epoch": epoch, "samples_seen": exposure,
                          "elapsed_seconds": time.perf_counter() - started,
                          **{name: value / args.log_every for name, value in running.items()}}
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                print(json.dumps(record, separators=(",", ":")), flush=True)
                running.clear()
        current = component_digest.hexdigest()
        if reference is None:
            reference = current
        elif current != reference:
            raise RuntimeError("P1-K target-free components changed across epochs")
        epoch_audits.append({"epoch": epoch, "images": epoch_images, "steps": epoch_steps,
                             "sample_keys_sha256": key_digest.hexdigest(),
                             "component_labels_sha256": current})
        if epoch_images != 5_000 or epoch_steps != 1_250:
            raise RuntimeError("P1-K epoch size differs")
        if step in CHECKPOINT_STEPS[:-1]:
            payload = _payload(ranker, optimizer, args, lineage, audit, step, epoch, exposure,
                               losses, step0_train, step0_route, None, None, None, epoch_audits)
            _atomic_torch_save(output / f"step-{step:06d}.pt", payload)
            _atomic_torch_save(output / "last.pt", payload)
            saved_steps.append(step)
    final_train = evaluate_component_ranker(
        atom_head, ranker, vision, train, collator, device=device, require_zero_parity=False
    )
    final_route = evaluate_component_ranker(
        atom_head, ranker, vision, route, collator, device=device, require_zero_parity=False
    )
    protocol = {
        "train_manifest": train_audit["sha256"] == p1a.TRAIN_MANIFEST_SHA256,
        "train_sequence": all(x["sample_keys_sha256"] == p1a.TRAIN_SEQUENCE_SHA256 for x in epoch_audits),
        "component_identity": all(x["component_labels_sha256"] == reference for x in epoch_audits),
        "route_sequence": step0_route["sample_keys_sha256"] == final_route["sample_keys_sha256"],
        "candidate_identity": step0_train["candidate_ids_sha256"] == final_train["candidate_ids_sha256"],
        "component_eval_identity": step0_train["component_labels_sha256"] == final_train["component_labels_sha256"],
        "count_logits_unchanged": step0_route["count_logits_sha256"] == final_route["count_logits_sha256"],
        "atom_state_unchanged": module_state_sha256(atom_head) == atom_sha,
        "vision_state_unchanged": module_state_sha256(vision) == vision_state_sha,
        "ranker_only_optimizer": validate_optimizer(atom_head, ranker, optimizer) is None,
        "fixed_exposure": step == 6_250 and exposure == 25_000,
        "checkpoint_schedule": saved_steps + [step] == list(CHECKPOINT_STEPS),
        "target_free_repeat": final_train["target_free_repeat"] and final_route["target_free_repeat"],
        "forbidden_data_absent": not audit["forbidden_data_read"],
    }
    gate = _gate(protocol, step0_train, step0_route, final_train, final_route)
    payload = _payload(ranker, optimizer, args, lineage, audit, step, 5, exposure, losses,
                       step0_train, step0_route, final_train, final_route, gate, epoch_audits)
    final_path = output / "step-006250.pt"
    _atomic_torch_save(final_path, payload)
    _atomic_torch_save(output / "last.pt", payload)
    _atomic_json(output / "epoch_audit.json", epoch_audits)
    _atomic_json(output / "final_train_metrics.json", final_train)
    _atomic_json(output / "final_route_metrics.json", final_route)
    _atomic_json(output / "p1k_gate.json", gate)
    _atomic_json(output / "status.json", {
        "status": "completed", "decision": gate["decision"], "global_step": step,
        "samples_seen": exposure, "checkpoint": str(final_path.resolve()),
        "checkpoint_sha256": file_sha256(final_path),
        "elapsed_seconds": time.perf_counter() - started,
    })
    print(json.dumps(gate, indent=2), flush=True)


if __name__ == "__main__":
    main()
