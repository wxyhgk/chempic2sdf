#!/usr/bin/env python3
"""Train P1-G with dynamic one-to-one equivalent-candidate supervision."""

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

from chemvector.components.atoms.g3_boundary_band_loss import (
    G3_BOUNDARY_BAND_SIZE,
    boundary_band_ranking_loss,
)
from chemvector.components.atoms.g3_equivalent_candidate_matching import (
    G3_EQUIVALENT_RADIUS_PIXELS,
    build_score_matched_equivalent_targets,
)
from chemvector.components.atoms.g3_top128_set_ranker import (
    G3_TOP128_SET_RANKER_PARAMETERS,
    extract_frozen_top128,
    select_ranked_top128,
)
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.atoms import train_g3_p1b_frozen_presence_ranker as p1b
from chemvector.training.atoms import train_g3_p1c_top128_set_ranker as p1c
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
    "outputs/checkpoints/chemvector-g3-p1g-equivalent-set-matching-train5k"
)
MODEL_TYPE = "chemvector-g3-p1g-equivalent-set-matching-train5k"
CHECKPOINT_STEPS = (1_250, 2_500, 3_750, 5_000, 6_250)
P1D_TRAIN_GEOMETRY_EXACT = 0.1868
P1D_TRAIN_GEOMETRY_RECALL = 0.9425605670352654
P1D_ROUTE_GEOMETRY_EXACT = 0.17
P1D_ROUTE_KNOWN_F1 = 0.9331


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", default=str(p1a.TRAIN_MANIFEST))
    parser.add_argument("--route-manifest", default=str(p1a.ROUTE_SELECT_MANIFEST))
    parser.add_argument("--p1a-checkpoint", default=str(p1d.P1A_CHECKPOINT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
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


def validate_args(args: argparse.Namespace) -> None:
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
            raise ValueError(f"P1-G requires fixed {name}={expected}")
    if args.device != "cuda" or args.precision != "bf16":
        raise ValueError("P1-G requires fixed CUDA BF16 training")
    if args.log_every < 1:
        raise ValueError("P1-G log interval must be positive")
    paths = {
        "train_manifest": p1a.TRAIN_MANIFEST,
        "route_manifest": p1a.ROUTE_SELECT_MANIFEST,
        "p1a_checkpoint": p1d.P1A_CHECKPOINT,
        "output_dir": DEFAULT_OUTPUT,
    }
    for name, expected in paths.items():
        if Path(getattr(args, name)).resolve() != expected.resolve():
            raise ValueError(f"P1-G requires fixed {name}: {expected}")


def validate_optimizer(atom_head, ranker, optimizer) -> None:
    expected = {id(value) for value in ranker.parameters()}
    actual = {
        id(value)
        for group in optimizer.param_groups
        for value in group["params"]
    }
    if expected != actual:
        raise ValueError("P1-G optimizer must contain only ranker parameters")
    if any(parameter.requires_grad for parameter in atom_head.parameters()):
        raise ValueError("P1-G frozen Atom head became trainable")


def p1g_training_step(
    atom_head,
    ranker,
    vision,
    optimizer,
    batch,
    *,
    device,
    precision,
    max_grad_norm,
):
    validate_optimizer(atom_head, ranker, optimizer)
    if any(parameter.requires_grad for parameter in vision.parameters()):
        raise ValueError("P1-G vision must remain frozen")
    optimizer.zero_grad(set_to_none=True)
    pixels = batch["pixel_values"].to(device, non_blocking=True)
    targets = {
        "atom_anchors": batch["atom_anchors"].to(device, non_blocking=True),
        "atom_mask": batch["atom_mask"].to(device, non_blocking=True),
    }
    with torch.no_grad(), _autocast_context(device, precision):
        features = vision.forward_stroke_resolution(pixels)
        dense = atom_head(features.detach())
        candidates = extract_frozen_top128(dense)
    with _autocast_context(device, precision):
        ranked = ranker(candidates)
    fixed = p1d.build_hybrid_top128_targets(candidates, targets)
    dynamic = build_score_matched_equivalent_targets(
        ranked, fixed, targets["atom_anchors"]
    )
    losses = boundary_band_ranking_loss(
        ranked, dynamic, boundary_size=G3_BOUNDARY_BAND_SIZE
    )
    losses["loss"].backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        ranker.parameters(), max_grad_norm
    )
    if not bool(torch.isfinite(gradient_norm)):
        raise FloatingPointError("P1-G ranker gradient norm is not finite")
    optimizer.step()
    if any(parameter.grad is not None for parameter in atom_head.parameters()):
        raise RuntimeError("P1-G produced gradients for frozen Atom head")
    if any(parameter.grad is not None for parameter in vision.parameters()):
        raise RuntimeError("P1-G produced gradients for frozen vision")
    metrics = {
        **{name: float(value.detach()) for name, value in losses.items()},
        "gradient_norm": float(gradient_norm.detach()),
        "identity_changes": float(dynamic["identity_changes"].sum()),
        "alternative_candidates": float(dynamic["alternative_candidates"].sum()),
        "batch_images": float(len(batch["keys"])),
    }
    return metrics, {"candidates": candidates, "fixed": fixed, "dynamic": dynamic}


def _empty_geometry_totals() -> dict[str, int]:
    return {
        "images": 0,
        "representable_atoms": 0,
        "selected_atoms": 0,
        "matched_atoms": 0,
        "exact_images": 0,
    }


def _geometry_sample(ranked, fixed, target_anchors, index) -> dict[str, int]:
    fixed_mask = fixed["positive_mask"][index].bool()
    count = int(fixed_mask.sum())
    selected = select_ranked_top128(
        {
            key: (
                value[index : index + 1]
                if isinstance(value, torch.Tensor) and value.ndim
                else value
            )
            for key, value in ranked.items()
        },
        torch.tensor([count], device=fixed_mask.device),
    )
    selected_mask = selected["atom_mask"][0]
    predicted = selected["atom_anchors"][0, selected_mask].float().cpu()
    target_indices = fixed["matched_target_indices"][index, fixed_mask]
    expected = target_anchors[index, target_indices].float().cpu()
    matched = p1a.maximum_cardinality_radius_match_count(
        predicted, expected, radius_pixels=6, image_size=768
    )
    return {
        "images": 1,
        "representable_atoms": count,
        "selected_atoms": len(predicted),
        "matched_atoms": matched,
        "exact_images": int(matched == count),
    }


def _finalize_geometry(totals: dict[str, int]) -> dict:
    atoms = totals["representable_atoms"]
    images = totals["images"]
    return {
        **totals,
        "atom_recall_6px": totals["matched_atoms"] / atoms,
        "exact_rate_6px": totals["exact_images"] / images,
    }


@torch.no_grad()
def evaluate_geometry(atom_head, ranker, vision, manifest, collator, *, device):
    atom_head.eval()
    ranker.eval()
    vision.eval()
    totals = _empty_geometry_totals()
    sample_digest = hashlib.sha256()
    fixed_audit = p1c.FrozenSetAuditAccumulator()
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
            ranked = ranker(candidates)
        fixed = p1d.build_hybrid_top128_targets(candidates, targets)
        fixed_audit.add(batch["keys"], candidates, fixed)
        for index in range(len(batch["keys"])):
            current = _geometry_sample(
                ranked, fixed, targets["atom_anchors"], index
            )
            for name, value in current.items():
                totals[name] += value
    return {
        "scope": "representable-gt-count-6px-geometry",
        **_finalize_geometry(totals),
        "sample_keys_sha256": sample_digest.hexdigest(),
        "frozen_set_audit": fixed_audit.finalize(),
    }


def _checkpoint_payload(
    ranker,
    optimizer,
    args,
    config,
    lineage,
    audit,
    step,
    epoch,
    exposure,
    losses,
    step0_route,
    final_route,
    train_geometry,
    route_geometry,
    gate,
    epoch_audits,
):
    return {
        "checkpoint_schema_version": "g3-p1g-equivalent-set-checkpoint-0.1",
        "model_type": MODEL_TYPE,
        "set_ranker": ranker.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": step,
        "epoch": epoch,
        "samples_seen": exposure,
        "training_args": vars(args),
        "g3_config": asdict(config),
        "lineage": lineage,
        "audit": audit,
        "trainer_sha256": file_sha256(__file__),
        "last_losses": losses,
        "step0_route_metrics": step0_route,
        "final_route_metrics": final_route,
        "train_geometry_metrics": train_geometry,
        "route_geometry_metrics": route_geometry,
        "p1g_gate": gate,
        "completed_epoch_audits": epoch_audits,
    }


def _gate(protocol_checks, train_geometry, route_geometry, final_route):
    protocol_passed = all(protocol_checks.values())
    train_gain = (
        train_geometry["exact_rate_6px"] - P1D_TRAIN_GEOMETRY_EXACT
    )
    route_f1 = final_route["known_count"]["f1_6px"]
    causal_checks = {
        "train_geometry_exact_plus_10pp": train_gain >= 0.10,
        "route_geometry_exact_no_regression": route_geometry["exact_rate_6px"]
        >= P1D_ROUTE_GEOMETRY_EXACT,
        "route_known_f1_within_0_5pp": route_f1 >= P1D_ROUTE_KNOWN_F1 - 0.005,
    }
    capability_checks = {
        "train_geometry_exact_50": train_geometry["exact_rate_6px"] >= 0.50,
        "train_geometry_recall_98": train_geometry["atom_recall_6px"] >= 0.98,
    }
    causal_passed = protocol_passed and all(causal_checks.values())
    capability_passed = protocol_passed and all(capability_checks.values())
    return {
        "protocol_checks": protocol_checks,
        "protocol_passed": protocol_passed,
        "causal_checks": causal_checks,
        "causal_passed": causal_passed,
        "capability_checks": capability_checks,
        "capability_passed": capability_passed,
        "train_geometry_exact_gain_pp": 100.0 * train_gain,
        "decision": (
            "protocol_failed"
            if not protocol_passed
            else "causal_go_capability_go"
            if causal_passed and capability_passed
            else "causal_go_capability_no_go"
            if causal_passed
            else "causal_no_go"
        ),
    }


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("P1-G CUDA training requested but CUDA is unavailable")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"P1-G output directory is not empty: {output_dir}")
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
    p1a.validate_occupancy(train_preflight, name="P1-G train5k")
    p1a.validate_occupancy(route_preflight, name="P1-G route100")
    overlap = p1a.identity_overlap_audit(
        train, route, train_preflight, route_preflight
    )
    if not overlap["all_three_identity_zero"]:
        raise ValueError("P1-G train and route identities overlap")
    if (
        train_audit["sha256"] != p1a.TRAIN_MANIFEST_SHA256
        or train_preflight["sample_keys_sha256"] != p1a.TRAIN_SEQUENCE_SHA256
        or route_preflight["sample_keys_sha256"] != p1d.EXPECTED_ROUTE_SEQUENCE_SHA256
    ):
        raise ValueError("P1-G manifest identity/order differs from protocol")

    atom_head, ranker, p1a_payload, config = p1d.load_p1a_boundary_ranker(device)
    vision_path = Path(p1a_payload["lineage"]["vision"]["path"])
    vision, _, _, vision_sha = load_frozen_vision_checkpoint(vision_path, device)
    if vision_sha != p1b.DEFAULT_VISION_CHECKPOINT_SHA256:
        raise ValueError("P1-G vision checkpoint SHA differs from protocol")
    vision.requires_grad_(False).eval()
    initial_atom_sha = module_state_sha256(atom_head)
    initial_vision_sha = module_state_sha256(vision)
    optimizer = torch.optim.AdamW(
        ranker.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    validate_optimizer(atom_head, ranker, optimizer)
    lineage = {
        "p1a": {
            "path": str(p1d.P1A_CHECKPOINT.resolve()),
            "sha256": p1d.P1A_CHECKPOINT_SHA256,
            "atom_head_state_sha256": initial_atom_sha,
        },
        "vision": {
            "path": str(vision_path.resolve()),
            "sha256": vision_sha,
            "state_sha256": initial_vision_sha,
        },
        "p1d_checkpoint_loaded": False,
    }
    audit = {
        "schema_version": "g3-p1g-training-audit-0.1",
        "scope": "equivalent-candidate-dynamic-one-to-one-train5k",
        "manifests": {
            "train5k": train_audit,
            "route100": route_audit,
            "identity_overlap": overlap,
        },
        "preflight": {"train5k": train_preflight, "route100": route_preflight},
        "lineage": lineage,
        "optimizer": {
            "name": "AdamW",
            "parameter_count": G3_TOP128_SET_RANKER_PARAMETERS,
            "ranker_only": True,
        },
        "matching": {
            "name": "maximum-cardinality-score-ranked-bipartite",
            "radius_pixels": G3_EQUIVALENT_RADIUS_PIXELS,
            "one_candidate_per_gt": True,
            "one_gt_per_candidate": True,
        },
        "loss": {
            "name": "boundary-band-pairwise-softplus",
            "boundary_size": G3_BOUNDARY_BAND_SIZE,
        },
        "forbidden_data_read": False,
        "forbidden_components_loaded": {
            "bond": False,
            "ocr": False,
            "chemscene": False,
            "sdf": False,
        },
        "implementation": {
            "trainer_sha256": file_sha256(__file__),
            "matcher_sha256": file_sha256(
                Path(__file__).parents[2]
                / "components/atoms/g3_equivalent_candidate_matching.py"
            ),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "audit.json", audit)
    _atomic_json(output_dir / "training_args.json", vars(args))
    _atomic_json(output_dir / "config.json", {"g3_config": asdict(config)})

    step0 = p1c.evaluate_route(
        atom_head,
        ranker,
        vision,
        route,
        collator,
        device=device,
        precision="bf16",
        batch_size=4,
        require_zero_parity=True,
    )
    step0["step"] = 0
    _atomic_json(output_dir / "step0_route_metrics.json", step0)
    ranker.train()

    step = exposure = 0
    saved_steps = []
    epoch_audits = []
    losses = {}
    running = {}
    started = time.perf_counter()
    log_path = output_dir / "training_log.jsonl"
    reference_frozen_hashes = None
    for epoch in range(1, 6):
        key_digest = hashlib.sha256()
        fixed_accumulator = p1c.FrozenSetAuditAccumulator()
        epoch_images = epoch_steps = identity_changes = alternatives = 0
        for batch in p1a.iter_manifest_batches(train, collator, batch_size=4):
            if len(batch["keys"]) != 4:
                raise RuntimeError("P1-G encountered a residual training batch")
            for key in batch["keys"]:
                key_digest.update(f"{key}\n".encode())
            losses, evidence = p1g_training_step(
                atom_head,
                ranker,
                vision,
                optimizer,
                batch,
                device=device,
                precision="bf16",
                max_grad_norm=1.0,
            )
            fixed_accumulator.add(
                batch["keys"], evidence["candidates"], evidence["fixed"]
            )
            step += 1
            epoch_steps += 1
            exposure += 4
            epoch_images += 4
            identity_changes += int(losses["identity_changes"])
            alternatives += int(losses["alternative_candidates"])
            for name, value in losses.items():
                running[name] = running.get(name, 0.0) + value
            if step % args.log_every == 0:
                record = {
                    "step": step,
                    "epoch": epoch,
                    "samples_seen": exposure,
                    "elapsed_seconds": time.perf_counter() - started,
                    **{
                        name: value / args.log_every
                        for name, value in running.items()
                    },
                }
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                print(json.dumps(record, separators=(",", ":")), flush=True)
                running.clear()
        frozen = fixed_accumulator.finalize()
        frozen_hashes = {
            key: value for key, value in frozen.items() if key.endswith("sha256")
        }
        if reference_frozen_hashes is None:
            reference_frozen_hashes = frozen_hashes
        elif frozen_hashes != reference_frozen_hashes:
            raise RuntimeError("P1-G frozen candidates or GT set changed")
        epoch_audits.append(
            {
                "epoch": epoch,
                "images": epoch_images,
                "steps": epoch_steps,
                "sample_keys_sha256": key_digest.hexdigest(),
                "identity_changes": identity_changes,
                "alternative_candidates": alternatives,
                "fixed_set": frozen,
            }
        )
        if (
            epoch_images != 5_000
            or epoch_steps != 1_250
            or key_digest.hexdigest() != p1a.TRAIN_SEQUENCE_SHA256
        ):
            raise RuntimeError("P1-G epoch stream differs from protocol")
        if step in CHECKPOINT_STEPS[:-1]:
            payload = _checkpoint_payload(
                ranker,
                optimizer,
                args,
                config,
                lineage,
                audit,
                step,
                epoch,
                exposure,
                losses,
                step0,
                None,
                None,
                None,
                None,
                epoch_audits,
            )
            _atomic_torch_save(output_dir / f"step-{step:06d}.pt", payload)
            _atomic_torch_save(output_dir / "last.pt", payload)
            saved_steps.append(step)
    if step != 6_250 or exposure != 25_000:
        raise RuntimeError("P1-G final step/exposure differs from protocol")

    train_geometry = evaluate_geometry(
        atom_head, ranker, vision, train, collator, device=device
    )
    route_geometry = evaluate_geometry(
        atom_head, ranker, vision, route, collator, device=device
    )
    final_route = p1c.evaluate_route(
        atom_head,
        ranker,
        vision,
        route,
        collator,
        device=device,
        precision="bf16",
        batch_size=4,
        require_zero_parity=False,
    )
    final_route["step"] = step
    final_atom_sha = module_state_sha256(atom_head)
    final_vision_sha = module_state_sha256(vision)
    protocol_checks = {
        "train_manifest": train_audit["sha256"] == p1a.TRAIN_MANIFEST_SHA256,
        "train_sequence": all(
            value["sample_keys_sha256"] == p1a.TRAIN_SEQUENCE_SHA256
            for value in epoch_audits
        ),
        "frozen_input_set": all(
            {
                key: value
                for key, value in epoch["fixed_set"].items()
                if key.endswith("sha256")
            }
            == reference_frozen_hashes
            for epoch in epoch_audits
        ),
        "route_sequence": step0["sample_keys_sha256"]
        == final_route["sample_keys_sha256"]
        == p1d.EXPECTED_ROUTE_SEQUENCE_SHA256,
        "identity_disjoint": overlap["all_three_identity_zero"],
        "p1a_checkpoint": file_sha256(p1d.P1A_CHECKPOINT)
        == p1d.P1A_CHECKPOINT_SHA256,
        "atom_state_unchanged": initial_atom_sha
        == final_atom_sha
        == p1d.P1A_ATOM_STATE_SHA256,
        "vision_state_unchanged": initial_vision_sha == final_vision_sha,
        "ranker_only_optimizer": validate_optimizer(
            atom_head, ranker, optimizer
        )
        is None,
        "fixed_exposure": step == 6_250 and exposure == 25_000,
        "checkpoint_schedule": saved_steps + [step] == list(CHECKPOINT_STEPS),
        "count_vector_unchanged": step0["deploy_count_vector"]
        == final_route["deploy_count_vector"],
        "forbidden_data_absent": not audit["forbidden_data_read"],
        "forbidden_components_absent": not any(
            audit["forbidden_components_loaded"].values()
        ),
    }
    gate = _gate(
        protocol_checks, train_geometry, route_geometry, final_route
    )
    final_payload = _checkpoint_payload(
        ranker,
        optimizer,
        args,
        config,
        lineage,
        audit,
        step,
        5,
        exposure,
        losses,
        step0,
        final_route,
        train_geometry,
        route_geometry,
        gate,
        epoch_audits,
    )
    final_path = output_dir / "step-006250.pt"
    _atomic_torch_save(final_path, final_payload)
    _atomic_torch_save(output_dir / "last.pt", final_payload)
    saved_steps.append(step)
    _atomic_json(output_dir / "epoch_audit.json", epoch_audits)
    _atomic_json(output_dir / "train_geometry_metrics.json", train_geometry)
    _atomic_json(output_dir / "route_geometry_metrics.json", route_geometry)
    _atomic_json(output_dir / "final_route_metrics.json", final_route)
    _atomic_json(output_dir / "p1g_gate.json", gate)
    _atomic_json(
        output_dir / "status.json",
        {
            "status": "completed",
            "decision": gate["decision"],
            "global_step": step,
            "samples_seen": exposure,
            "checkpoint": str(final_path.resolve()),
            "checkpoint_sha256": file_sha256(final_path),
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    print(json.dumps(gate, indent=2), flush=True)


if __name__ == "__main__":
    main()
