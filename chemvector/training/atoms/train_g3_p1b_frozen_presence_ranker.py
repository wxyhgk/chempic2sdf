#!/usr/bin/env python3
"""Train the frozen P1-A Atom presence ranker on the committed train5k stream."""

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

from chemvector.components.atoms.g3_presence_ranker import (
    G3FrozenAtomPresenceRanker,
    canonical_presence_ranking_loss,
)
from chemvector.components.graph.joint.g3_hierarchical_graph import (
    G3AtomOnlyPredictor,
    G3HierarchicalGraphConfig,
    build_canonical_atom_slot_targets,
    select_count_conditioned_atom_slots,
)
from chemvector.training.atoms.train_g3_p1a_canonical_atom_selector import (
    G3AtomCollator,
    TRAIN_MANIFEST,
    TRAIN_MANIFEST_SHA256,
    TRAIN_SEQUENCE_SHA256,
    atom_sample_statistics,
    identity_overlap_audit,
    iter_manifest_batches,
    load_route_manifest,
    load_train_manifest,
    stream_preflight,
    validate_occupancy,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    DEFAULT_VISION_CHECKPOINT_SHA256,
    _atomic_json,
    _atomic_torch_save,
    _autocast_context,
    load_frozen_vision_checkpoint,
    module_state_sha256,
)
from chemvector.g3_structural_data import file_sha256


P1A_CHECKPOINT = Path(
    "outputs/checkpoints/chemvector-g3-p1a-canonical-atom-selector-train5k/last.pt"
)
P1A_CHECKPOINT_SHA256 = (
    "bf0d35f31dafc467fafe3fd1bcbb758d47685a4febc7d9ad6ff96d1032d12ec2"
)
P1A_ATOM_STATE_SHA256 = (
    "05386591037f0c9ae9624ef6befe25c7a82315faf720ab915a01532592e44b8a"
)
DEFAULT_OUTPUT = Path(
    "outputs/checkpoints/chemvector-g3-p1b-frozen-presence-ranker-train5k"
)
MODEL_TYPE = "chemvector-g3-p1b-frozen-presence-ranker-train5k"
CHECKPOINT_STEPS = (1_250, 2_500, 3_750, 5_000, 6_250)
EXPECTED_RESIDUAL_PARAMETERS = 16_643
EXPECTED_ROUTE_SEQUENCE_SHA256 = (
    "2beb4f442ec60ba792c0f18449ecfeeeff15328c6b3d977ecd76b1a07c89fd7d"
)
IMAGE_SIZE = 768.0
MATCH_RADIUS_PIXELS = 6.0


def load_p1a_ranker(device: torch.device):
    if file_sha256(P1A_CHECKPOINT) != P1A_CHECKPOINT_SHA256:
        raise ValueError("P1-B P1-A checkpoint SHA differs from the protocol")
    payload = torch.load(P1A_CHECKPOINT, map_location="cpu", weights_only=False)
    if (
        payload.get("checkpoint_schema_version")
        != "g3-p1a-atom-selector-checkpoint-0.1"
        or payload.get("global_step") != 6_250
        or payload.get("p1a_gate", {}).get("protocol_status") != "passed"
        or payload.get("p1a_gate", {}).get("decision") != "NO-GO"
    ):
        raise ValueError("P1-B requires the protocol-valid P1-A fixed-final state")
    config = G3HierarchicalGraphConfig(**payload["g3_config"])
    base = G3AtomOnlyPredictor(config)
    base.atom_head.load_state_dict(payload["atom_head"], strict=True)
    if module_state_sha256(base.atom_head) != P1A_ATOM_STATE_SHA256:
        raise ValueError("P1-B P1-A Atom state SHA differs from the protocol")
    ranker = G3FrozenAtomPresenceRanker(base.atom_head, config).to(device)
    ranker.train()
    if sum(parameter.numel() for parameter in ranker.presence_ranker.parameters()) != (
        EXPECTED_RESIDUAL_PARAMETERS
    ):
        raise ValueError("P1-B residual parameter count differs from the protocol")
    return ranker, payload, config


def validate_optimizer(
    ranker: G3FrozenAtomPresenceRanker, optimizer: torch.optim.Optimizer
) -> None:
    expected = list(ranker.presence_ranker.parameters())
    actual = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    if len(expected) != len(actual) or {id(value) for value in expected} != {
        id(value) for value in actual
    }:
        raise ValueError("P1-B optimizer must contain only residual parameters")
    if any(parameter.requires_grad for parameter in ranker.atom_head.parameters()):
        raise ValueError("P1-B frozen Atom head became trainable")


def _empty_metrics() -> dict[str, float]:
    return {
        "images": 0.0,
        "predicted_atoms": 0.0,
        "target_atoms": 0.0,
        "matched_atoms": 0.0,
        "count_exact_images": 0.0,
        "whole_set_exact_images": 0.0,
        "count_absolute_error": 0.0,
    }


def _merge_metrics(total: dict[str, float], current: dict) -> None:
    for name in total:
        total[name] += float(current[name])


def _finalize_metrics(total: dict[str, float]) -> dict:
    predicted = int(total["predicted_atoms"])
    target = int(total["target_atoms"])
    matched = int(total["matched_atoms"])
    precision = matched / predicted if predicted else 0.0
    recall = matched / target if target else 0.0
    return {
        "images": int(total["images"]),
        "predicted_atoms": predicted,
        "target_atoms": target,
        "matched_atoms": matched,
        "precision_6px": precision,
        "recall_6px": recall,
        "f1_6px": (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        ),
        "count_exact_images": int(total["count_exact_images"]),
        "count_mae": (
            total["count_absolute_error"] / total["images"]
            if total["images"]
            else 0.0
        ),
        "whole_set_exact_images": int(total["whole_set_exact_images"]),
    }


def _slot_flat_indices(
    selected: dict[str, torch.Tensor], index: int, height: int, width: int
):
    mask = selected["atom_mask"][index]
    slots = selected["atom_slot_indices"][index, mask]
    return slots[:, 0] * height * width + slots[:, 1] * width + slots[:, 2]


@torch.no_grad()
def evaluate_route(
    ranker: G3FrozenAtomPresenceRanker,
    vision: torch.nn.Module,
    manifest,
    collator: G3AtomCollator,
    *,
    device: torch.device,
    precision: str,
    batch_size: int,
    require_zero_parity: bool,
) -> dict:
    ranker.eval()
    vision.eval()
    known = _empty_metrics()
    deploy = _empty_metrics()
    identity_matched = identity_whole = 0
    top128_matched = 0
    fixed_localized = fixed_total = 0
    zero_logit_parity = zero_selection_parity = True
    residual_abs_max = 0.0
    predicted_counts: list[int] = []
    sample_digest = hashlib.sha256()
    count_logits_digest = hashlib.sha256()
    count_invariant_dense_scores = True
    for batch in iter_manifest_batches(manifest, collator, batch_size=batch_size):
        for key in batch["keys"]:
            sample_digest.update(f"{key}\n".encode("utf-8"))
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        targets = {
            "atom_anchors": batch["atom_anchors"].to(device, non_blocking=True),
            "atom_mask": batch["atom_mask"].to(device, non_blocking=True),
        }
        gt_counts = targets["atom_mask"].sum(dim=1).long()
        with _autocast_context(device, precision):
            features = vision.forward_stroke_resolution(pixels)
            dense = ranker.forward_dense(features.detach())
        known_selected = select_count_conditioned_atom_slots(
            dense, gt_counts, maximum_atoms=ranker.config.maximum_atoms
        )
        deploy_counts = dense["count_logits"].argmax(dim=-1)
        count_logits_digest.update(
            dense["count_logits"]
            .detach()
            .float()
            .cpu()
            .contiguous()
            .numpy()
            .tobytes()
        )
        deploy_selected = select_count_conditioned_atom_slots(
            dense, deploy_counts, maximum_atoms=ranker.config.maximum_atoms
        )
        predicted_counts.extend(int(value) for value in deploy_counts.cpu().tolist())
        canonical = build_canonical_atom_slot_targets(dense, targets)
        positive = canonical["positive"].flatten(1)
        ranked = dense["presence_logits"].flatten(1)
        base = dense["base_presence_logits"].flatten(1)
        residual_abs_max = max(
            residual_abs_max,
            float(dense["presence_residual"].detach().float().abs().max().cpu()),
        )
        if require_zero_parity:
            zero_logit_parity &= torch.equal(ranked, base)
            base_dense = dict(dense)
            base_dense["presence_logits"] = dense["base_presence_logits"]
            base_selected = select_count_conditioned_atom_slots(
                base_dense, gt_counts, maximum_atoms=ranker.config.maximum_atoms
            )
            zero_selection_parity &= torch.equal(
                base_selected["atom_slot_indices"],
                known_selected["atom_slot_indices"],
            )
            score_snapshot = dense["presence_logits"].detach().clone()
            for probe_counts in (
                torch.ones_like(gt_counts),
                gt_counts,
                torch.full_like(gt_counts, ranker.config.maximum_atoms),
            ):
                select_count_conditioned_atom_slots(
                    dense,
                    probe_counts,
                    maximum_atoms=ranker.config.maximum_atoms,
                )
                count_invariant_dense_scores &= torch.equal(
                    score_snapshot, dense["presence_logits"]
                )
        top128 = ranked.topk(ranker.config.maximum_atoms, dim=1).indices
        batch_size_now, _, height, width = dense["presence_logits"].shape
        dense_anchors = dense["anchors"].permute(0, 1, 3, 4, 2)
        expected_anchors = canonical["anchors"].permute(0, 1, 3, 4, 2)
        for index in range(batch_size_now):
            target_mask = targets["atom_mask"][index]
            target_anchors = targets["atom_anchors"][index, target_mask].float().cpu()
            known_indices = torch.where(known_selected["atom_mask"][index])[0]
            deploy_indices = torch.where(deploy_selected["atom_mask"][index])[0]
            _merge_metrics(
                known,
                atom_sample_statistics(
                    known_selected["atom_anchors"][index, known_indices]
                    .detach()
                    .float()
                    .cpu(),
                    target_anchors,
                ),
            )
            _merge_metrics(
                deploy,
                atom_sample_statistics(
                    deploy_selected["atom_anchors"][index, deploy_indices]
                    .detach()
                    .float()
                    .cpu(),
                    target_anchors,
                ),
            )
            positive_indices = torch.where(positive[index])[0]
            selected_indices = _slot_flat_indices(
                known_selected, index, height, width
            )
            selected_set = set(selected_indices.tolist())
            positive_set = set(positive_indices.tolist())
            identity_matched += len(selected_set & positive_set)
            identity_whole += int(selected_set == positive_set)
            flat_dense_anchors = dense_anchors[index].reshape(-1, 2)
            top128_stats = atom_sample_statistics(
                flat_dense_anchors[top128[index]].detach().float().cpu(),
                target_anchors,
            )
            top128_matched += int(top128_stats["matched_atoms"])
            local_distances = torch.linalg.vector_norm(
                (
                    dense_anchors[index][canonical["positive"][index]]
                    - expected_anchors[index][canonical["positive"][index]]
                ).float(),
                dim=-1,
            )
            fixed_localized += int(
                (local_distances * IMAGE_SIZE <= MATCH_RADIUS_PIXELS).sum()
            )
            fixed_total += len(local_distances)
        del pixels, features, dense, canonical, known_selected, deploy_selected
    known_metrics = _finalize_metrics(known)
    deploy_metrics = _finalize_metrics(deploy)
    predicted_count_bytes = json.dumps(
        predicted_counts, separators=(",", ":")
    ).encode("utf-8")
    return {
        "scope": "route-select100-frozen-presence-ranker",
        "known_count": known_metrics,
        "deploy_count": deploy_metrics,
        "canonical_identity": {
            "matched": identity_matched,
            "total": int(known["target_atoms"]),
            "recall": identity_matched / known["target_atoms"],
            "whole_set_exact_images": identity_whole,
        },
        "top128_coverage": {
            "matched": top128_matched,
            "total": int(known["target_atoms"]),
            "recall": top128_matched / known["target_atoms"],
        },
        "fixed_positive_localization": {
            "matched": fixed_localized,
            "total": fixed_total,
            "recall": fixed_localized / fixed_total,
        },
        "deploy_count_vector": predicted_counts,
        "deploy_count_vector_sha256": hashlib.sha256(predicted_count_bytes).hexdigest(),
        "count_logits_sha256": count_logits_digest.hexdigest(),
        "sample_keys_sha256": sample_digest.hexdigest(),
        "residual_abs_max": residual_abs_max,
        "zero_logit_parity": zero_logit_parity,
        "zero_selection_parity": zero_selection_parity,
        "count_invariant_dense_scores": count_invariant_dense_scores,
    }


def p1b_training_step(
    ranker: G3FrozenAtomPresenceRanker,
    vision: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: dict,
    *,
    device: torch.device,
    precision: str,
    max_grad_norm: float,
) -> dict[str, float]:
    validate_optimizer(ranker, optimizer)
    if any(parameter.requires_grad for parameter in vision.parameters()):
        raise ValueError("P1-B vision must remain frozen")
    optimizer.zero_grad(set_to_none=True)
    pixels = batch["pixel_values"].to(device, non_blocking=True)
    targets = {
        "atom_anchors": batch["atom_anchors"].to(device, non_blocking=True),
        "atom_mask": batch["atom_mask"].to(device, non_blocking=True),
    }
    with torch.no_grad():
        with _autocast_context(device, precision):
            features = vision.forward_stroke_resolution(pixels)
    with _autocast_context(device, precision):
        dense = ranker.forward_dense(features.detach())
        losses = canonical_presence_ranking_loss(
            dense, targets, negative_multiplier=4
        )
    losses["loss"].backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        ranker.presence_ranker.parameters(), max_grad_norm
    )
    if not bool(torch.isfinite(gradient_norm)):
        raise FloatingPointError("P1-B residual gradient norm is not finite")
    optimizer.step()
    if any(parameter.grad is not None for parameter in ranker.atom_head.parameters()):
        raise RuntimeError("P1-B produced gradients for the frozen Atom head")
    if any(parameter.grad is not None for parameter in vision.parameters()):
        raise RuntimeError("P1-B produced gradients for frozen vision")
    return {
        **{name: float(value.detach()) for name, value in losses.items()},
        "gradient_norm": float(gradient_norm.detach()),
        "batch_images": float(len(batch["keys"])),
    }


def selector_gate(protocol_checks: dict[str, bool], metrics: dict) -> dict:
    known = metrics["known_count"]
    capability_checks = {
        "known_precision_97": known["precision_6px"] >= 0.97,
        "known_recall_97": known["recall_6px"] >= 0.97,
        "known_f1_97": known["f1_6px"] >= 0.97,
        "known_whole_50": known["whole_set_exact_images"] >= 50,
        "canonical_identity_95": metrics["canonical_identity"]["recall"] >= 0.95,
        "top128_coverage_guard": metrics["top128_coverage"]["matched"] >= 3_072,
        "fixed_positive_localization_guard": (
            metrics["fixed_positive_localization"]["matched"] == 3_080
            and metrics["fixed_positive_localization"]["total"] == 3_080
        ),
    }
    protocol_passed = all(protocol_checks.values())
    capability_passed = all(capability_checks.values())
    return {
        "status": "passed" if protocol_passed and capability_passed else "failed",
        "decision": "GO" if protocol_passed and capability_passed else "NO-GO",
        "protocol_status": "passed" if protocol_passed else "failed",
        "protocol_checks": protocol_checks,
        "capability_status": "passed" if capability_passed else "failed",
        "capability_checks": capability_checks,
        "fixed_final_step": 6_250,
    }


def _checkpoint_payload(
    *,
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
    step0_metrics,
    final_metrics,
    gate,
    epoch_audits,
) -> dict:
    return {
        "checkpoint_schema_version": "g3-p1b-presence-ranker-checkpoint-0.1",
        "model_type": MODEL_TYPE,
        "residual": ranker.presence_ranker.state_dict(),
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
        "step0_route_metrics": step0_metrics,
        "final_route_metrics": final_metrics,
        "p1b_gate": gate,
        "completed_epoch_audits": [dict(item) for item in epoch_audits],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", default=str(TRAIN_MANIFEST))
    parser.add_argument(
        "--route-manifest",
        default="Data/sdf/chemtraverse_v0_1_capacity/dev_route_select_100.jsonl",
    )
    parser.add_argument("--p1a-checkpoint", default=str(P1A_CHECKPOINT))
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
            raise ValueError(f"P1-B requires fixed {name}={expected}")
    if args.device != "cuda" or args.precision != "bf16":
        raise ValueError("P1-B requires fixed CUDA BF16 training")
    if args.log_every < 1:
        raise ValueError("P1-B log interval must be positive")
    fixed_paths = {
        "train_manifest": TRAIN_MANIFEST,
        "route_manifest": Path(
            "Data/sdf/chemtraverse_v0_1_capacity/dev_route_select_100.jsonl"
        ),
        "p1a_checkpoint": P1A_CHECKPOINT,
        "output_dir": DEFAULT_OUTPUT,
    }
    for name, expected in fixed_paths.items():
        if Path(getattr(args, name)).resolve() != expected.resolve():
            raise ValueError(f"P1-B requires fixed {name}: {expected}")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("P1-B CUDA training requested but CUDA is unavailable")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"P1-B output directory is not empty: {output_dir}")

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

    train_manifest, train_audit = load_train_manifest(args.train_manifest)
    route_manifest, route_audit = load_route_manifest(args.route_manifest)
    collator = G3AtomCollator(maximum_atoms=128, image_size=768)
    train_preflight = stream_preflight(train_manifest, collator)
    route_preflight = stream_preflight(route_manifest, collator)
    validate_occupancy(train_preflight, name="P1-B train5k")
    validate_occupancy(route_preflight, name="P1-B route-select100")
    overlap = identity_overlap_audit(
        train_manifest, route_manifest, train_preflight, route_preflight
    )
    if not overlap["all_three_identity_zero"]:
        raise ValueError("P1-B train and route identities overlap")
    if (
        train_audit["sha256"] != TRAIN_MANIFEST_SHA256
        or train_preflight["sample_keys_sha256"] != TRAIN_SEQUENCE_SHA256
        or route_preflight["sample_keys_sha256"] != EXPECTED_ROUTE_SEQUENCE_SHA256
    ):
        raise ValueError("P1-B manifest identity/order differs from the protocol")

    ranker, p1a_payload, config = load_p1a_ranker(device)
    vision_path = Path(p1a_payload["lineage"]["vision"]["path"])
    vision, _, _, vision_sha = load_frozen_vision_checkpoint(vision_path, device)
    if vision_sha != DEFAULT_VISION_CHECKPOINT_SHA256:
        raise ValueError("P1-B vision checkpoint SHA differs from the protocol")
    vision.requires_grad_(False).eval()
    ranker.train()
    initial_atom_sha = module_state_sha256(ranker.atom_head)
    initial_vision_sha = module_state_sha256(vision)
    optimizer = torch.optim.AdamW(
        ranker.presence_ranker.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    validate_optimizer(ranker, optimizer)
    lineage = {
        "p1a": {
            "path": str(P1A_CHECKPOINT.resolve()),
            "sha256": P1A_CHECKPOINT_SHA256,
            "global_step": 6_250,
            "atom_head_state_sha256": initial_atom_sha,
        },
        "vision": {
            "path": str(vision_path.resolve()),
            "sha256": vision_sha,
            "state_sha256": initial_vision_sha,
        },
    }
    audit = {
        "schema_version": "g3-p1b-training-audit-0.1",
        "scope": "train-distribution-frozen-presence-ranker-train5k",
        "manifests": {
            "train5k": train_audit,
            "route_select100": route_audit,
            "identity_overlap": overlap,
        },
        "preflight": {"train5k": train_preflight, "route_select100": route_preflight},
        "lineage": lineage,
        "implementation": {
            "trainer": str(Path(__file__).resolve()),
            "trainer_sha256": file_sha256(__file__),
            "ranker_sha256": file_sha256(
                Path(__file__).parents[2] / "components/atoms/g3_presence_ranker.py"
            ),
        },
        "optimizer": {
            "name": "AdamW",
            "parameter_count": EXPECTED_RESIDUAL_PARAMETERS,
            "residual_only": True,
            "restored_state": False,
        },
        "streaming": {
            "shuffle": False,
            "num_workers": 0,
            "drop_last": False,
            "whole_dataset_cache": False,
        },
        "forbidden_components_loaded": {
            "bond": False,
            "atom_pair": False,
            "ocr": False,
            "chemscene": False,
        },
        "constructed_modules": {
            "vision": type(vision).__name__,
            "atom_head": type(ranker.atom_head).__name__,
            "presence_ranker": type(ranker.presence_ranker).__name__,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "audit.json", audit)
    _atomic_json(output_dir / "training_args.json", vars(args))
    _atomic_json(output_dir / "config.json", {"g3_config": asdict(config)})

    route_steps = [0]
    step0_metrics = evaluate_route(
        ranker,
        vision,
        route_manifest,
        collator,
        device=device,
        precision=args.precision,
        batch_size=args.batch_size,
        require_zero_parity=True,
    )
    step0_metrics["step"] = 0
    expected_step0 = {
        "known_matched": step0_metrics["known_count"]["matched_atoms"] == 2_870,
        "known_whole": step0_metrics["known_count"]["whole_set_exact_images"] == 14,
        "identity_matched": step0_metrics["canonical_identity"]["matched"] == 2_581,
        "top128": step0_metrics["top128_coverage"]["matched"] == 3_072,
        "fixed_localized": (
            step0_metrics["fixed_positive_localization"]["matched"] == 3_080
        ),
        "deploy_matched": step0_metrics["deploy_count"]["matched_atoms"] == 2_807,
        "deploy_whole": step0_metrics["deploy_count"]["whole_set_exact_images"] == 2,
        "count_exact": step0_metrics["deploy_count"]["count_exact_images"] == 13,
        "count_mae": step0_metrics["deploy_count"]["count_mae"] == 3.83,
        "zero_logit_parity": step0_metrics["zero_logit_parity"],
        "zero_selection_parity": step0_metrics["zero_selection_parity"],
        "count_invariant_dense_scores": step0_metrics[
            "count_invariant_dense_scores"
        ],
    }
    if not all(expected_step0.values()):
        raise RuntimeError(f"P1-B step0 failed fixed baseline: {expected_step0}")
    _atomic_json(output_dir / "step0_route_metrics.json", step0_metrics)
    ranker.train()

    step = exposure = 0
    saved_steps: list[int] = []
    epoch_audits: list[dict] = []
    losses: dict[str, float] = {}
    running: dict[str, float] = {}
    started = time.perf_counter()
    log_path = output_dir / "training_log.jsonl"
    for epoch in range(1, args.epochs + 1):
        digest = hashlib.sha256()
        epoch_images = epoch_steps = 0
        for batch in iter_manifest_batches(
            train_manifest, collator, batch_size=args.batch_size
        ):
            if len(batch["keys"]) != args.batch_size:
                raise RuntimeError("P1-B encountered a residual training batch")
            for key in batch["keys"]:
                digest.update(f"{key}\n".encode("utf-8"))
            losses = p1b_training_step(
                ranker,
                vision,
                optimizer,
                batch,
                device=device,
                precision=args.precision,
                max_grad_norm=args.max_grad_norm,
            )
            step += 1
            epoch_steps += 1
            exposure += len(batch["keys"])
            epoch_images += len(batch["keys"])
            for name, value in losses.items():
                running[name] = running.get(name, 0.0) + value
            if step % args.log_every == 0:
                record = {
                    "step": step,
                    "epoch": epoch,
                    "samples_seen": exposure,
                    "elapsed_seconds": time.perf_counter() - started,
                    **{name: value / args.log_every for name, value in running.items()},
                }
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                print(json.dumps(record, separators=(",", ":")), flush=True)
                running.clear()
        epoch_audit = {
            "epoch": epoch,
            "images": epoch_images,
            "steps": epoch_steps,
            "sample_keys_sha256": digest.hexdigest(),
        }
        if (
            epoch_images != 5_000
            or epoch_steps != 1_250
            or epoch_audit["sample_keys_sha256"] != TRAIN_SEQUENCE_SHA256
        ):
            raise RuntimeError("P1-B epoch stream differs from the fixed protocol")
        epoch_audits.append(epoch_audit)
        if step in CHECKPOINT_STEPS[:-1]:
            payload = _checkpoint_payload(
                ranker=ranker,
                optimizer=optimizer,
                args=args,
                config=config,
                lineage=lineage,
                audit=audit,
                step=step,
                epoch=epoch,
                exposure=exposure,
                losses=losses,
                step0_metrics=step0_metrics,
                final_metrics=None,
                gate=None,
                epoch_audits=epoch_audits,
            )
            _atomic_torch_save(output_dir / f"step-{step:06d}.pt", payload)
            _atomic_torch_save(output_dir / "last.pt", payload)
            saved_steps.append(step)
    if step != 6_250 or exposure != 25_000:
        raise RuntimeError("P1-B final step/exposure differs from the protocol")

    final_metrics = evaluate_route(
        ranker,
        vision,
        route_manifest,
        collator,
        device=device,
        precision=args.precision,
        batch_size=args.batch_size,
        require_zero_parity=False,
    )
    final_metrics["step"] = step
    route_steps.append(step)
    final_atom_sha = module_state_sha256(ranker.atom_head)
    final_vision_sha = module_state_sha256(vision)
    protocol_checks = {
        "train_manifest": train_audit["sha256"] == TRAIN_MANIFEST_SHA256,
        "train_sequence": all(
            item["sample_keys_sha256"] == TRAIN_SEQUENCE_SHA256
            for item in epoch_audits
        ),
        "route_sequence": (
            step0_metrics["sample_keys_sha256"]
            == final_metrics["sample_keys_sha256"]
            == EXPECTED_ROUTE_SEQUENCE_SHA256
        ),
        "identity_disjoint": overlap["all_three_identity_zero"],
        "p1a_checkpoint": file_sha256(P1A_CHECKPOINT) == P1A_CHECKPOINT_SHA256,
        "atom_state_unchanged": (
            initial_atom_sha == final_atom_sha == P1A_ATOM_STATE_SHA256
        ),
        "vision_state_unchanged": initial_vision_sha == final_vision_sha,
        "residual_only_optimizer": (
            validate_optimizer(ranker, optimizer) is None
        ),
        "fixed_exposure": step == 6_250 and exposure == 25_000,
        "five_epochs": len(epoch_audits) == 5,
        "checkpoint_schedule": saved_steps + [step] == list(CHECKPOINT_STEPS),
        "route_steps": route_steps == [0, 6_250],
        "deploy_count_vector_unchanged": (
            step0_metrics["deploy_count_vector"]
            == final_metrics["deploy_count_vector"]
        ),
        "count_logits_unchanged": (
            step0_metrics["count_logits_sha256"]
            == final_metrics["count_logits_sha256"]
        ),
        "deploy_count_metrics_unchanged": (
            final_metrics["deploy_count"]["count_exact_images"] == 13
            and final_metrics["deploy_count"]["count_mae"] == 3.83
        ),
        "forbidden_components_absent": (
            not any(audit["forbidden_components_loaded"].values())
            and set(audit["constructed_modules"])
            == {"vision", "atom_head", "presence_ranker"}
        ),
        "count_free_dense_forward": (
            tuple(
                __import__("inspect").signature(ranker.forward_dense).parameters
            )
            == ("features",)
        ),
    }
    gate = selector_gate(protocol_checks, final_metrics)
    final_payload = _checkpoint_payload(
        ranker=ranker,
        optimizer=optimizer,
        args=args,
        config=config,
        lineage=lineage,
        audit=audit,
        step=step,
        epoch=5,
        exposure=exposure,
        losses=losses,
        step0_metrics=step0_metrics,
        final_metrics=final_metrics,
        gate=gate,
        epoch_audits=epoch_audits,
    )
    final_path = output_dir / "step-006250.pt"
    _atomic_torch_save(final_path, final_payload)
    _atomic_torch_save(output_dir / "last.pt", final_payload)
    saved_steps.append(step)
    actual_checkpoints = sorted(path.name for path in output_dir.glob("step-*.pt"))
    expected_checkpoints = [f"step-{value:06d}.pt" for value in CHECKPOINT_STEPS]
    if actual_checkpoints != expected_checkpoints:
        raise RuntimeError("P1-B checkpoint file set differs from the protocol")
    _atomic_json(output_dir / "epoch_audit.json", epoch_audits)
    _atomic_json(output_dir / "final_route_metrics.json", final_metrics)
    _atomic_json(output_dir / "p1b_gate.json", gate)
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
