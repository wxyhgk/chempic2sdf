#!/usr/bin/env python3
"""Distill P0-G same-slot Atom features into a residual P0-I adapter."""

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
from torch.nn import functional as F

from chemvector.atom_feature_adapter import (
    ATOM_FEATURE_ADAPTER_PARAMETERS,
    ATOM_FEATURE_DIM,
    AtomFeatureAdapter,
)
from chemvector.audit_g3_p0j_frozen_composition import (
    P0G_CHECKPOINT,
    P0G_CHECKPOINT_SHA256,
    P0I_CHECKPOINT,
    P0I_CHECKPOINT_SHA256,
    composition_batch_statistics,
    compose_pair_output,
    empty_mode_totals,
    evaluate_composition_mode,
    finalize_mode_totals,
    gather_slot_features,
    load_frozen_p0_models,
    merge_totals,
    pair_identity_sha256,
)
from chemvector.g3_hierarchical_graph import build_g3_bond_targets
from chemvector.g3_structural_data import (
    G3StructuralCollator,
    OVERFIT32_MANIFEST,
    OVERFIT32_MANIFEST_SHA256,
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


P0K_CHECKPOINT = Path(
    "outputs/smoke/chemvector-g3-p0k-predicted-anchor-relation-alignment/last.pt"
)
P0K_CHECKPOINT_SHA256 = (
    "a91d5bb1052063c819b3ba71e26987c45d5e7e819281ff26fa7b3bd80f90eac5"
)
DEFAULT_OUTPUT = Path(
    "outputs/smoke/chemvector-g3-p0l-native-feature-adapter-overfit32"
)
MODEL_TYPE = "g3-native-feature-adapter-p0l-v0.1"
EXPECTED_IMAGES = 32
EXPECTED_ATOMS = 1424
EXPECTED_BONDS = 1508
EXPECTED_PAIRS = 37380


def load_p0k_pair_overlay(p0g, checkpoint_path: Path, device: torch.device) -> dict:
    """Strictly overlay the frozen P0-K pair head onto the P0-G model."""

    if file_sha256(checkpoint_path) != P0K_CHECKPOINT_SHA256:
        raise ValueError("P0-L P0-K checkpoint SHA differs from the protocol")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("model_type") != "g3-predicted-anchor-relation-alignment-p0k-v0.1":
        raise ValueError("P0-L P0-K checkpoint has an unexpected model type")
    if payload.get("global_step") != 400:
        raise ValueError("P0-L requires the final P0-K step-400 checkpoint")
    if payload.get("source_p0i_checkpoint", {}).get("sha256") != P0I_CHECKPOINT_SHA256:
        raise ValueError("P0-L P0-K lineage has an unexpected P0-I checkpoint")
    if payload.get("source_p0g_checkpoint", {}).get("sha256") != P0G_CHECKPOINT_SHA256:
        raise ValueError("P0-L P0-K lineage has an unexpected P0-G checkpoint")
    pair_state = payload.get("pair_bond")
    if not isinstance(pair_state, dict) or not pair_state:
        raise ValueError("P0-L P0-K checkpoint contains no pair_bond state")
    p0g.pair_bond.load_state_dict(pair_state, strict=True)
    p0g.to(device).requires_grad_(False).eval()
    return payload


@torch.no_grad()
def cache_adapter_inputs(
    p0i,
    p0g,
    features: torch.Tensor,
    *,
    batch_size: int,
    precision: str,
) -> dict[str, torch.Tensor]:
    """Cache native inputs and detached same-slot teacher features."""

    cached: dict[str, list[torch.Tensor]] = {
        "native_features": [],
        "teacher_features": [],
        "atom_mask": [],
        "atom_slot_indices": [],
    }
    for start in range(0, len(features), batch_size):
        current = features[start : start + batch_size]
        with _autocast_context(features.device, precision):
            atoms = p0i(current)
            dense = p0g.atom_head(current)
            teacher = gather_slot_features(
                dense, atoms["atom_slot_indices"], atoms["atom_mask"]
            )
        values = {
            "native_features": atoms["atom_features"],
            "teacher_features": teacher,
            "atom_mask": atoms["atom_mask"],
            "atom_slot_indices": atoms["atom_slot_indices"],
        }
        for name, value in values.items():
            cached[name].append(value.detach())
    return {name: torch.cat(values, dim=0) for name, values in cached.items()}


def masked_feature_distillation_loss(
    adapter: AtomFeatureAdapter,
    native_features: torch.Tensor,
    teacher_features: torch.Tensor,
    atom_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute SmoothL1 only for valid Atoms; teacher never receives gradients."""

    if native_features.shape != teacher_features.shape:
        raise ValueError("P0-L native and teacher feature shapes differ")
    if atom_mask.shape != native_features.shape[:2] or atom_mask.dtype != torch.bool:
        raise ValueError("P0-L Atom mask is incompatible with feature tensors")
    if not bool(atom_mask.any()):
        raise ValueError("P0-L feature batch contains no valid Atom")
    adapted = adapter(native_features)
    return F.smooth_l1_loss(
        adapted[atom_mask], teacher_features.detach()[atom_mask], reduction="mean"
    )


def validate_adapter_optimizer(
    adapter: AtomFeatureAdapter, optimizer: torch.optim.Optimizer
) -> None:
    expected = list(adapter.parameters())
    actual = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    if len(expected) != len(actual) or {id(value) for value in expected} != {
        id(value) for value in actual
    }:
        raise ValueError("P0-L optimizer must contain only adapter parameters")


def p0l_training_step(
    adapter: AtomFeatureAdapter,
    optimizer: torch.optim.Optimizer,
    native_features: torch.Tensor,
    teacher_features: torch.Tensor,
    atom_mask: torch.Tensor,
    *,
    precision: str,
) -> dict[str, float]:
    validate_adapter_optimizer(adapter, optimizer)
    optimizer.zero_grad(set_to_none=True)
    with _autocast_context(native_features.device, precision):
        loss = masked_feature_distillation_loss(
            adapter, native_features, teacher_features, atom_mask
        )
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
    if not bool(torch.isfinite(gradient_norm)):
        raise FloatingPointError("P0-L adapter gradient norm is not finite")
    optimizer.step()
    return {
        "loss": float(loss.detach()),
        "gradient_norm": float(gradient_norm.detach()),
        "valid_atoms": float(atom_mask.sum()),
    }


@torch.no_grad()
def evaluate_adapted_composition(
    adapter: AtomFeatureAdapter,
    p0i,
    p0g,
    features: torch.Tensor,
    targets: dict[str, torch.Tensor],
    sample_keys: list[str],
    *,
    batch_size: int,
    precision: str,
) -> dict:
    """Evaluate adapted native features through both frozen P0-K consumers."""

    was_training = adapter.training
    adapter.eval()
    totals = empty_mode_totals()
    identity_masks: list[torch.Tensor] = []
    for start in range(0, len(features), batch_size):
        current_features = features[start : start + batch_size]
        current_targets = {
            name: value[start : start + batch_size] for name, value in targets.items()
        }
        with _autocast_context(features.device, precision):
            atoms = p0i(current_features)
            adapted_features = adapter(atoms["atom_features"])
            pair_output = compose_pair_output(
                p0g, current_features, atoms, adapted_features
            )
        current = composition_batch_statistics(
            pair_output,
            atoms,
            current_targets,
            build_g3_bond_targets(current_targets),
        )
        merge_totals(totals, current)
        identity_masks.extend(mask.detach().cpu() for mask in atoms["atom_mask"])
    result = finalize_mode_totals(totals)
    result["feature_mode"] = "p0i_native_plus_adapter"
    result["pair_identity_sha256"] = pair_identity_sha256(
        sample_keys, identity_masks
    )
    if was_training:
        adapter.train()
    return result


@torch.no_grad()
def feature_alignment_metrics(
    adapter: AtomFeatureAdapter,
    cache: dict[str, torch.Tensor],
    *,
    batch_size: int,
    precision: str,
) -> dict[str, float | int]:
    was_training = adapter.training
    adapter.eval()
    total_loss = 0.0
    total_squared_error = 0.0
    total_teacher_energy = 0.0
    total_cosine = 0.0
    total_elements = 0
    valid_atoms = 0
    for start in range(0, len(cache["native_features"]), batch_size):
        native = cache["native_features"][start : start + batch_size]
        teacher = cache["teacher_features"][start : start + batch_size]
        mask = cache["atom_mask"][start : start + batch_size]
        with _autocast_context(native.device, precision):
            adapted = adapter(native)
            valid_adapted = adapted[mask].float()
            valid_teacher = teacher[mask].float()
            per_element = F.smooth_l1_loss(
                valid_adapted, valid_teacher, reduction="sum"
            )
            squared_error = F.mse_loss(
                valid_adapted, valid_teacher, reduction="sum"
            )
            teacher_energy = valid_teacher.square().sum()
            cosine = F.cosine_similarity(
                valid_adapted, valid_teacher, dim=-1, eps=1e-8
            ).sum()
        elements = int(mask.sum()) * native.size(-1)
        total_loss += float(per_element)
        total_squared_error += float(squared_error)
        total_teacher_energy += float(teacher_energy)
        total_cosine += float(cosine)
        total_elements += elements
        valid_atoms += int(mask.sum())
    if was_training:
        adapter.train()
    return {
        "smooth_l1": total_loss / max(total_elements, 1),
        "cosine_similarity": total_cosine / max(valid_atoms, 1),
        "nrmse": (
            total_squared_error / max(total_teacher_energy, 1e-12)
        )
        ** 0.5,
        "valid_atoms": valid_atoms,
        "feature_elements": total_elements,
    }


@torch.no_grad()
def evaluate_p0l(
    adapter: AtomFeatureAdapter,
    p0i,
    p0g,
    features: torch.Tensor,
    targets: dict[str, torch.Tensor],
    sample_keys: list[str],
    cache: dict[str, torch.Tensor],
    *,
    batch_size: int,
    precision: str,
    step0_smooth_l1: float | None = None,
) -> dict:
    adapted = evaluate_adapted_composition(
        adapter,
        p0i,
        p0g,
        features,
        targets,
        sample_keys,
        batch_size=batch_size,
        precision=precision,
    )
    control = evaluate_composition_mode(
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
    alignment = feature_alignment_metrics(
        adapter, cache, batch_size=batch_size, precision=precision
    )
    baseline = alignment["smooth_l1"] if step0_smooth_l1 is None else step0_smooth_l1
    if baseline <= 0.0:
        raise ValueError("P0-L step-0 SmoothL1 baseline must be positive")
    alignment["step0_smooth_l1"] = baseline
    alignment["smooth_l1_ratio_to_step0"] = alignment["smooth_l1"] / baseline
    checks = {
        "fixed_coverage": (
            adapted["images"] == EXPECTED_IMAGES
            and adapted["atom_matched"] == adapted["atom_targets"] == EXPECTED_ATOMS
            and adapted["representable_bonds"]
            == adapted["target_bonds"]
            == EXPECTED_BONDS
            and adapted["pair_universe"] == EXPECTED_PAIRS
        ),
        "pair_identity_unchanged": (
            adapted["pair_identity_sha256"] == control["pair_identity_sha256"]
        ),
        "feature_smooth_l1_reduced_90_percent": (
            alignment["smooth_l1_ratio_to_step0"] <= 0.1
        ),
        "adapted_gt_graph": (
            adapted["gt_bond_count"]["typed_graph_exact_images"] >= 30
        ),
        "adapted_deploy_graph": (
            adapted["deploy_bond_count"]["typed_graph_exact_images"]
            >= 30
        ),
        "same_slot_p0k_gt_graph": (
            control["gt_bond_count"]["typed_graph_exact_images"] == EXPECTED_IMAGES
        ),
        "same_slot_p0k_deploy_graph": (
            control["deploy_bond_count"]["typed_graph_exact_images"]
            == EXPECTED_IMAGES
        ),
        "gt_atom_oracle_gt_graph": (
            oracle["gt_count"]["structural_graph_exact_images"] == EXPECTED_IMAGES
        ),
        "gt_atom_oracle_deploy_graph": (
            oracle["deploy_count"]["structural_graph_exact_images"]
            == EXPECTED_IMAGES
        ),
    }
    return {
        "scope": "train-only-overfit32-p0l-native-feature-adapter",
        "feature_alignment": alignment,
        "adapted_native_feature": adapted,
        "same_slot_p0k_control": control,
        "gt_atom_oracle_control": oracle,
        "p0l_gate": {
            "status": "passed" if all(checks.values()) else "failed",
            "checks": checks,
            "fixed_final_step_only": True,
        },
    }


def validate_frozen_scope(p0i, p0g, adapter: AtomFeatureAdapter) -> None:
    if any(parameter.requires_grad for parameter in p0i.parameters()):
        raise ValueError("P0-L P0-I must remain frozen")
    if any(parameter.requires_grad for parameter in p0g.parameters()):
        raise ValueError("P0-L P0-G and P0-K pair overlay must remain frozen")
    if not all(parameter.requires_grad for parameter in adapter.parameters()):
        raise ValueError("P0-L adapter parameters must all be trainable")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-manifest", default=str(OVERFIT32_MANIFEST))
    parser.add_argument("--p0i-checkpoint", default=str(P0I_CHECKPOINT))
    parser.add_argument("--p0g-checkpoint", default=str(P0G_CHECKPOINT))
    parser.add_argument("--p0k-checkpoint", default=str(P0K_CHECKPOINT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vision-batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--evaluate-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260713)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    fixed = {
        "max_steps": 400,
        "batch_size": 4,
        "vision_batch_size": 4,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "log_every": 20,
        "evaluate_every": 100,
        "save_every": 100,
        "seed": 20260713,
    }
    for name, expected in fixed.items():
        if getattr(args, name) != expected:
            raise ValueError(f"P0-L requires fixed {name}={expected}")
    if args.device != "cuda" or args.precision != "bf16":
        raise ValueError("P0-L requires fixed CUDA BF16 training")
    fixed_paths = {
        "sample_manifest": Path(OVERFIT32_MANIFEST),
        "p0i_checkpoint": P0I_CHECKPOINT,
        "p0g_checkpoint": P0G_CHECKPOINT,
        "p0k_checkpoint": P0K_CHECKPOINT,
        "output_dir": DEFAULT_OUTPUT,
    }
    for name, expected in fixed_paths.items():
        if Path(getattr(args, name)).resolve() != expected.resolve():
            raise ValueError(f"P0-L requires fixed {name}: {expected}")


def checkpoint_payload(
    *,
    adapter: AtomFeatureAdapter,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    lineage: dict,
    step: int,
    losses: dict[str, float],
    metrics: dict,
    step0_feature_alignment: dict,
) -> dict:
    return {
        "checkpoint_schema_version": "g3-p0l-native-feature-adapter-checkpoint-0.1",
        "model_type": MODEL_TYPE,
        "adapter": adapter.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": step,
        "samples_seen": step * args.batch_size,
        "training_args": vars(args),
        "last_losses": losses,
        "train_only_metrics": metrics,
        "step0_feature_alignment": step0_feature_alignment,
        "lineage": lineage,
    }


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("P0-L CUDA training requested but CUDA is unavailable")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"P0-L output directory is not empty: {output_dir}")

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

    p0i, p0g, p0i_payload, _, config = load_frozen_p0_models(
        Path(args.p0i_checkpoint), Path(args.p0g_checkpoint), device
    )
    if config.d_model != ATOM_FEATURE_DIM:
        raise ValueError("P0-L source feature dimension differs from 384")
    p0k_payload = load_p0k_pair_overlay(p0g, Path(args.p0k_checkpoint), device)
    pair_sha = module_state_sha256(p0g.pair_bond)
    p0i_sha = module_state_sha256(p0i)
    p0g_sha = module_state_sha256(p0g)
    adapter = AtomFeatureAdapter().to(device)
    adapter.train()
    validate_frozen_scope(p0i, p0g, adapter)
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    validate_adapter_optimizer(adapter, optimizer)

    manifest, manifest_audit = load_g3_overfit32_manifest(args.sample_manifest)
    if manifest_audit["sha256"] != OVERFIT32_MANIFEST_SHA256:
        raise ValueError("P0-L overfit32 manifest SHA differs from the protocol")
    records = load_g3_manifest_samples(manifest)
    vision, _, _, vision_sha = load_frozen_vision_checkpoint(
        p0i_payload["source_vision_checkpoint"], device
    )
    if vision_sha != DEFAULT_VISION_CHECKPOINT_SHA256:
        raise ValueError("P0-L vision SHA differs from the protocol")
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
    cache = cache_adapter_inputs(
        p0i,
        p0g,
        features,
        batch_size=args.batch_size,
        precision=args.precision,
    )
    del vision, batch

    lineage = {
        "sample_manifest": manifest_audit,
        "vision": {
            "path": str(Path(p0i_payload["source_vision_checkpoint"]).resolve()),
            "sha256": vision_sha,
        },
        "p0i": {
            "path": str(Path(args.p0i_checkpoint).resolve()),
            "sha256": file_sha256(args.p0i_checkpoint),
        },
        "p0g": {
            "path": str(Path(args.p0g_checkpoint).resolve()),
            "sha256": file_sha256(args.p0g_checkpoint),
        },
        "p0k": {
            "path": str(Path(args.p0k_checkpoint).resolve()),
            "sha256": file_sha256(args.p0k_checkpoint),
            "global_step": p0k_payload["global_step"],
        },
    }
    audit = {
        "schema_version": "g3-p0l-training-audit-0.1",
        "lineage": lineage,
        "implementation": {
            "trainer": str(Path(__file__).resolve()),
            "trainer_sha256": file_sha256(__file__),
        },
        "adapter": {
            "architecture": "LayerNorm(384, affine=False) -> Linear(384,384) residual",
            "parameter_count": sum(parameter.numel() for parameter in adapter.parameters()),
            "trainable_parameter_names": [
                name for name, _ in adapter.named_parameters()
            ],
            "initial_state_sha256": module_state_sha256(adapter),
        },
        "frozen_pair_bond_sha256": pair_sha,
        "frozen_p0i_state_sha256": p0i_sha,
        "frozen_p0g_state_sha256": p0g_sha,
        "feature_cache": {name: list(value.shape) for name, value in cache.items()},
        "protocol": {
            "teacher": "p0g_same_slot_detached",
            "student": "p0i_native_atom_features",
            "loss": "valid_atom_smooth_l1_only",
            "gt_atom_or_bond_labels_in_training": False,
            "train_adapter_only": True,
        },
    }
    if audit["adapter"]["parameter_count"] != ATOM_FEATURE_ADAPTER_PARAMETERS:
        raise ValueError("P0-L adapter parameter count differs from the protocol")
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "audit.json", audit)
    _atomic_json(output_dir / "training_args.json", vars(args))
    _atomic_json(output_dir / "config.json", {"g3_config": asdict(config)})

    metrics = evaluate_p0l(
        adapter,
        p0i,
        p0g,
        features,
        targets,
        sample_keys,
        cache,
        batch_size=args.batch_size,
        precision=args.precision,
    )
    step0_feature_alignment = dict(metrics["feature_alignment"])
    metrics["step"] = 0
    _atomic_json(output_dir / "initial_metrics.json", metrics)
    _atomic_json(output_dir / "metrics.json", metrics)
    print(json.dumps({"train_only_evaluation": metrics}), flush=True)

    started = time.perf_counter()
    running: dict[str, float] = {}
    log_path = output_dir / "training_log.jsonl"
    losses: dict[str, float] = {}
    for step in range(1, args.max_steps + 1):
        indices = cyclic_batch_indices(
            step,
            sample_count=len(records),
            batch_size=args.batch_size,
            device=device,
        )
        losses = p0l_training_step(
            adapter,
            optimizer,
            cache["native_features"][indices],
            cache["teacher_features"][indices],
            cache["atom_mask"][indices],
            precision=args.precision,
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
            metrics = evaluate_p0l(
                adapter,
                p0i,
                p0g,
                features,
                targets,
                sample_keys,
                cache,
                batch_size=args.batch_size,
                precision=args.precision,
                step0_smooth_l1=step0_feature_alignment["smooth_l1"],
            )
            metrics["step"] = step
            metrics["elapsed_seconds"] = time.perf_counter() - started
            _atomic_json(output_dir / "metrics.json", metrics)
            print(json.dumps({"train_only_evaluation": metrics}), flush=True)
        if step % args.save_every == 0:
            if module_state_sha256(p0i) != p0i_sha:
                raise RuntimeError("P0-L changed the frozen P0-I model")
            if module_state_sha256(p0g) != p0g_sha:
                raise RuntimeError("P0-L changed the frozen P0-G/P0-K model")
            if module_state_sha256(p0g.pair_bond) != pair_sha:
                raise RuntimeError("P0-L changed the frozen P0-K pair head")
            checkpoint = checkpoint_payload(
                adapter=adapter,
                optimizer=optimizer,
                args=args,
                lineage=lineage,
                step=step,
                losses=losses,
                metrics=metrics,
                step0_feature_alignment=step0_feature_alignment,
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
