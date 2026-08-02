#!/usr/bin/env python3
"""Align the frozen P0-G pair head to P0-I predicted Atom anchors."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from chemvector.audit_g3_p0j_frozen_composition import (
    P0G_CHECKPOINT,
    P0G_CHECKPOINT_SHA256,
    P0I_CHECKPOINT,
    P0I_CHECKPOINT_SHA256,
    evaluate_composition_mode,
    gather_slot_features,
    load_frozen_p0_models,
    maximum_radius_atom_mapping,
)
from chemvector.direct_atom_pair_bond import (
    direct_atom_pair_loss,
    enumerate_unordered_atom_pairs,
)
from chemvector.g3_hierarchical_graph import build_g3_bond_targets
from chemvector.g3_structural_data import (
    G3StructuralCollator,
    OVERFIT32_MANIFEST,
    file_sha256,
    g3_targets_to_device,
    load_g3_manifest_samples,
    load_g3_overfit32_manifest,
)
from chemvector.train_g3_hierarchical_graph import (
    DEFAULT_VISION_CHECKPOINT_SHA256,
    _atomic_json,
    _atomic_torch_save,
    _autocast_context,
    cache_frozen_vision_features,
    cyclic_batch_indices,
    evaluate_g3_count_modes,
    load_frozen_vision_checkpoint,
    module_state_sha256,
)


DEFAULT_OUTPUT = Path(
    "outputs/smoke/chemvector-g3-p0k-predicted-anchor-relation-alignment"
)
MODEL_TYPE = "g3-predicted-anchor-relation-alignment-p0k-v0.1"
EXPECTED_IMAGES = 32
EXPECTED_ATOMS = 1424
EXPECTED_BONDS = 1508
EXPECTED_PAIRS = 37380


@torch.no_grad()
def cache_predicted_atom_inputs(
    p0i,
    p0g,
    features: torch.Tensor,
    *,
    batch_size: int,
    precision: str,
) -> dict[str, torch.Tensor]:
    cached: dict[str, list[torch.Tensor]] = {
        "atom_mask": [],
        "atom_anchors": [],
        "atom_scores": [],
        "atom_features": [],
        "atom_slot_indices": [],
    }
    for start in range(0, len(features), batch_size):
        current = features[start : start + batch_size]
        with _autocast_context(features.device, precision):
            atoms = p0i(current)
            dense = p0g.atom_head(current)
            atom_features = gather_slot_features(
                dense, atoms["atom_slot_indices"], atoms["atom_mask"]
            )
        values = {
            "atom_mask": atoms["atom_mask"],
            "atom_anchors": atoms["atom_anchors"],
            "atom_scores": atoms["atom_scores"],
            "atom_features": atom_features,
            "atom_slot_indices": atoms["atom_slot_indices"],
        }
        for name, value in values.items():
            cached[name].append(value.detach())
    return {name: torch.cat(values, dim=0) for name, values in cached.items()}


def build_predicted_pair_targets(
    atoms: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    pair_indices: torch.Tensor,
    sample_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    """Build labels only after predicted-pair forward identities are fixed."""

    normalized = build_g3_bond_targets(targets)
    presence = torch.zeros(
        len(pair_indices), device=pair_indices.device, dtype=torch.float32
    )
    chemical = torch.zeros(
        len(pair_indices), device=pair_indices.device, dtype=torch.long
    )
    mapped_atoms = covered_bonds = target_bonds = 0
    for batch_index in range(atoms["atom_mask"].size(0)):
        predicted = torch.where(atoms["atom_mask"][batch_index])[0]
        expected = torch.where(targets["atom_mask"][batch_index].bool())[0]
        mapping = maximum_radius_atom_mapping(
            atoms["atom_anchors"][batch_index],
            predicted,
            targets["atom_anchors"][batch_index],
            expected,
        )
        if len(mapping) != len(predicted) or len(mapping) != len(expected):
            raise ValueError(
                "P0-K requires every predicted Atom to have one 6px GT match"
            )
        mapped_atoms += len(mapping)
        count = int(normalized["mask"][batch_index].sum())
        expected_graph = {}
        for index in range(count):
            begin, end = normalized["atom_refs"][batch_index, index].tolist()
            expected_graph[tuple(sorted((int(begin), int(end))))] = int(
                normalized["chemical"][batch_index, index]
            )
        target_bonds += len(expected_graph)
        covered_bonds += sum(
            begin in set(mapping.values()) and end in set(mapping.values())
            for begin, end in expected_graph
        )
        rows = torch.where(sample_indices == batch_index)[0]
        for row in rows.tolist():
            begin, end = pair_indices[row].tolist()
            pair = tuple(sorted((mapping[int(begin)], mapping[int(end)])))
            if pair in expected_graph:
                presence[row] = 1.0
                chemical[row] = expected_graph[pair]
    return presence, chemical, {
        "mapped_atoms": mapped_atoms,
        "target_bonds": target_bonds,
        "covered_bonds": covered_bonds,
        "pair_universe": len(pair_indices),
    }


def validate_pair_optimizer(p0g, optimizer: torch.optim.Optimizer) -> None:
    expected = list(p0g.pair_bond.parameters())
    actual = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    if len(expected) != len(actual) or {id(value) for value in expected} != {
        id(value) for value in actual
    }:
        raise ValueError("P0-K optimizer must contain only P0-G pair-head parameters")


def p0k_training_step(
    p0g,
    optimizer: torch.optim.Optimizer,
    features: torch.Tensor,
    atoms: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    precision: str,
    max_grad_norm: float,
) -> dict[str, float]:
    validate_pair_optimizer(p0g, optimizer)
    pair_indices, sample_indices = enumerate_unordered_atom_pairs(atoms["atom_mask"])
    optimizer.zero_grad(set_to_none=True)
    with _autocast_context(features.device, precision):
        output = p0g.pair_bond(
            features,
            atoms["atom_features"],
            atoms["atom_anchors"],
            atoms["atom_scores"],
            pair_indices,
            sample_indices,
        )
        presence, chemical, coverage = build_predicted_pair_targets(
            atoms, targets, pair_indices, sample_indices
        )
        losses = direct_atom_pair_loss(output, presence, chemical)
    losses["loss"].backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        p0g.pair_bond.parameters(), max_grad_norm
    )
    if not bool(torch.isfinite(gradient_norm)):
        raise FloatingPointError("P0-K pair gradient norm is not finite")
    optimizer.step()
    return {
        **{name: float(value.detach()) for name, value in losses.items()},
        "gradient_norm": float(gradient_norm.detach()),
        **{name: float(value) for name, value in coverage.items()},
    }


@torch.no_grad()
def evaluate_p0k(
    p0i,
    p0g,
    features: torch.Tensor,
    targets: dict[str, torch.Tensor],
    sample_keys: list[str],
    *,
    batch_size: int,
    precision: str,
) -> dict:
    was_training = p0g.pair_bond.training
    p0g.pair_bond.eval()
    predicted = evaluate_composition_mode(
        p0i,
        p0g,
        features,
        targets,
        sample_keys,
        feature_mode="p0g_slot_replay",
        batch_size=batch_size,
        precision=precision,
    )
    oracle = evaluate_g3_count_modes(
        p0g,
        features,
        targets,
        batch_size=batch_size,
        precision=precision,
        gt_atom_slot_oracle=True,
        direct_gt_atom_pair_bonds=True,
    )
    if was_training:
        p0g.pair_bond.train()
    checks = {
        "atom_mapping": (
            predicted["atom_matched"]
            == predicted["atom_targets"]
            == EXPECTED_ATOMS
            and predicted["atom_set_exact_images"] == EXPECTED_IMAGES
        ),
        "bond_coverage": (
            predicted["representable_bonds"]
            == predicted["target_bonds"]
            == EXPECTED_BONDS
        ),
        "pair_universe": predicted["pair_universe"] == EXPECTED_PAIRS,
        "predicted_gt_skeleton": (
            predicted["gt_bond_count"]["skeleton_exact_images"] >= 30
        ),
        "predicted_gt_typed_graph": (
            predicted["gt_bond_count"]["typed_graph_exact_images"] >= 30
        ),
        "predicted_deploy_count": (
            predicted["deploy_bond_count"]["count_exact_images"] >= 30
        ),
        "predicted_deploy_skeleton": (
            predicted["deploy_bond_count"]["skeleton_exact_images"] >= 30
        ),
        "predicted_deploy_typed_graph": (
            predicted["deploy_bond_count"]["typed_graph_exact_images"] >= 30
        ),
        "oracle_gt_graph": oracle["gt_count"]["structural_graph_exact_images"] >= 30,
        "oracle_deploy_graph": (
            oracle["deploy_count"]["structural_graph_exact_images"] >= 30
        ),
    }
    return {
        "scope": "train-only-overfit32-p0k-predicted-anchor-relation-alignment",
        "predicted_atom_composition": predicted,
        "gt_atom_oracle_replay": oracle,
        "p0k_gate": {
            "status": "passed" if all(checks.values()) else "failed",
            "checks": checks,
            "fixed_final_step_only": True,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-manifest", default=str(OVERFIT32_MANIFEST))
    parser.add_argument("--p0i-checkpoint", default=str(P0I_CHECKPOINT))
    parser.add_argument("--p0g-checkpoint", default=str(P0G_CHECKPOINT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vision-batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--evaluate-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260713)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.max_steps,
        args.batch_size,
        args.vision_batch_size,
        args.learning_rate,
        args.max_grad_norm,
        args.log_every,
        args.evaluate_every,
        args.save_every,
    )
    if min(positive) <= 0 or args.weight_decay < 0.0:
        raise ValueError("P0-K training dimensions are invalid")
    if args.device == "cpu" and args.precision != "fp32":
        raise ValueError("P0-K CPU training requires fp32")
    if Path(args.output_dir).resolve() != DEFAULT_OUTPUT.resolve():
        raise ValueError(f"P0-K requires fixed output directory: {DEFAULT_OUTPUT}")
    if Path(args.sample_manifest).resolve() != Path(OVERFIT32_MANIFEST).resolve():
        raise ValueError("P0-K requires the committed overfit32 manifest")


def checkpoint_payload(
    *,
    p0g,
    optimizer,
    args,
    audit,
    step,
    losses,
    metrics,
) -> dict:
    return {
        "checkpoint_schema_version": "g3-p0k-relation-alignment-checkpoint-0.1",
        "model_type": MODEL_TYPE,
        "pair_bond": p0g.pair_bond.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": step,
        "samples_seen": step * args.batch_size,
        "training_args": vars(args),
        "last_losses": losses,
        "train_only_metrics": metrics,
        "source_p0i_checkpoint": audit["checkpoints"]["p0i"],
        "source_p0g_checkpoint": audit["checkpoints"]["p0g"],
        "source_vision_checkpoint": audit["vision"],
        "sample_manifest": audit["sample_manifest"],
        "training_scope": {
            "predicted_atom_anchors": True,
            "gt_anchor_in_pair_forward": False,
            "p0g_slot_replay_features": True,
            "train_pair_bond_only": True,
        },
    }


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("P0-K CUDA training requested but CUDA is unavailable")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"P0-K output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)

    p0i, p0g, p0i_payload, p0g_payload, config = load_frozen_p0_models(
        Path(args.p0i_checkpoint), Path(args.p0g_checkpoint), device
    )
    initial_pair_sha = module_state_sha256(p0g.pair_bond)
    p0g.pair_bond.requires_grad_(True)
    p0g.pair_bond.train()
    optimizer = torch.optim.AdamW(
        p0g.pair_bond.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    validate_pair_optimizer(p0g, optimizer)

    manifest, manifest_audit = load_g3_overfit32_manifest(args.sample_manifest)
    records = load_g3_manifest_samples(manifest)
    vision, _, _, vision_sha = load_frozen_vision_checkpoint(
        p0i_payload["source_vision_checkpoint"], device
    )
    if vision_sha != DEFAULT_VISION_CHECKPOINT_SHA256:
        raise ValueError("P0-K vision SHA differs from the protocol")
    collator = G3StructuralCollator(
        maximum_atoms=config.maximum_atoms,
        maximum_bonds=config.maximum_bonds,
        image_size=768,
    )
    batch = collator(records)
    features = cache_frozen_vision_features(
        vision,
        batch["pixel_values"],
        device=device,
        batch_size=args.vision_batch_size,
        precision=args.precision,
    )
    targets = g3_targets_to_device(batch, device)
    sample_keys = list(batch["keys"])
    atoms = cache_predicted_atom_inputs(
        p0i,
        p0g,
        features,
        batch_size=args.batch_size,
        precision=args.precision,
    )
    del vision, batch

    trainable_names = [name for name, _ in p0g.pair_bond.named_parameters()]
    audit = {
        "schema_version": "g3-p0k-training-audit-0.1",
        "sample_manifest": manifest_audit,
        "vision": {
            "path": str(Path(p0i_payload["source_vision_checkpoint"]).resolve()),
            "sha256": vision_sha,
        },
        "checkpoints": {
            "p0i": {
                "path": str(Path(args.p0i_checkpoint).resolve()),
                "sha256": file_sha256(args.p0i_checkpoint),
            },
            "p0g": {
                "path": str(Path(args.p0g_checkpoint).resolve()),
                "sha256": file_sha256(args.p0g_checkpoint),
            },
        },
        "initial_pair_bond_sha256": initial_pair_sha,
        "parameter_count": sum(
            parameter.numel() for parameter in p0g.pair_bond.parameters()
        ),
        "trainable_parameter_names": trainable_names,
        "implementation": {
            "trainer": str(Path(__file__).resolve()),
            "trainer_sha256": file_sha256(__file__),
        },
        "feature_cache_shape": list(features.shape),
        "atom_cache": {name: list(value.shape) for name, value in atoms.items()},
        "protocol": {
            "name": "g3-p0k-predicted-anchor-relation-alignment",
            "gt_anchor_in_pair_forward": False,
            "p0g_slot_replay_features": True,
            "train_pair_bond_only": True,
        },
    }
    if audit["checkpoints"]["p0i"]["sha256"] != P0I_CHECKPOINT_SHA256:
        raise ValueError("P0-K P0-I checkpoint changed during load")
    if audit["checkpoints"]["p0g"]["sha256"] != P0G_CHECKPOINT_SHA256:
        raise ValueError("P0-K P0-G checkpoint changed during load")
    _atomic_json(output_dir / "audit.json", audit)
    _atomic_json(output_dir / "training_args.json", vars(args))
    _atomic_json(output_dir / "config.json", {"g3_config": asdict(config)})

    metrics = evaluate_p0k(
        p0i,
        p0g,
        features,
        targets,
        sample_keys,
        batch_size=args.batch_size,
        precision=args.precision,
    )
    metrics["step"] = 0
    _atomic_json(output_dir / "initial_metrics.json", metrics)
    _atomic_json(output_dir / "metrics.json", metrics)
    print(json.dumps({"train_only_evaluation": metrics}), flush=True)

    started = time.perf_counter()
    running: dict[str, float] = {}
    log_path = output_dir / "training_log.jsonl"
    for step in range(1, args.max_steps + 1):
        indices = cyclic_batch_indices(
            step,
            sample_count=len(records),
            batch_size=args.batch_size,
            device=device,
        )
        current_atoms = {name: value[indices] for name, value in atoms.items()}
        current_targets = {name: value[indices] for name, value in targets.items()}
        losses = p0k_training_step(
            p0g,
            optimizer,
            features[indices],
            current_atoms,
            current_targets,
            precision=args.precision,
            max_grad_norm=args.max_grad_norm,
        )
        for name, value in losses.items():
            running[name] = running.get(name, 0.0) + value
        if step % args.log_every == 0:
            record = {
                "step": step,
                "samples_seen": step * args.batch_size,
                "elapsed_seconds": time.perf_counter() - started,
                **{name: value / args.log_every for name, value in running.items()},
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            print(json.dumps(record, separators=(",", ":")), flush=True)
            running.clear()
        if step % args.evaluate_every == 0:
            metrics = evaluate_p0k(
                p0i,
                p0g,
                features,
                targets,
                sample_keys,
                batch_size=args.batch_size,
                precision=args.precision,
            )
            metrics["step"] = step
            metrics["elapsed_seconds"] = time.perf_counter() - started
            _atomic_json(output_dir / "metrics.json", metrics)
            print(json.dumps({"train_only_evaluation": metrics}), flush=True)
        if step % args.save_every == 0:
            checkpoint = checkpoint_payload(
                p0g=p0g,
                optimizer=optimizer,
                args=args,
                audit=audit,
                step=step,
                losses=losses,
                metrics=metrics,
            )
            step_path = output_dir / f"step-{step:06d}.pt"
            _atomic_torch_save(step_path, checkpoint)
            _atomic_torch_save(output_dir / "last.pt", checkpoint)
            _atomic_json(
                output_dir / "status.json",
                {
                    "status": "completed" if step == args.max_steps else "running",
                    "global_step": step,
                    "samples_seen": step * args.batch_size,
                    "elapsed_seconds": time.perf_counter() - started,
                    "checkpoint": step_path.name,
                    "last_losses": losses,
                    "train_only_metrics": metrics,
                },
            )


if __name__ == "__main__":
    main()
