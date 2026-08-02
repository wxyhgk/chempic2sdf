#!/usr/bin/env python3
"""Train the frozen G3 top-128 global Atom set ranker on train5k."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from chemvector.components.atoms.g3_top128_set_ranker import (
    G3_TOP128_SET_RANKER_PARAMETERS,
    G3Top128SetRanker,
    build_hybrid_top128_targets,
    extract_frozen_top128,
    select_ranked_top128,
    top128_set_ranking_loss,
)
from chemvector.components.graph.joint.g3_hierarchical_graph import (
    G3AtomOnlyPredictor,
    G3HierarchicalGraphConfig,
    build_canonical_atom_slot_targets,
    select_count_conditioned_atom_slots,
)
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.atoms import train_g3_p1b_frozen_presence_ranker as p1b
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _atomic_torch_save,
    _autocast_context,
    load_frozen_vision_checkpoint,
    module_state_sha256,
)
from chemvector.g3_structural_data import file_sha256


P1A_CHECKPOINT = p1b.P1A_CHECKPOINT
P1A_CHECKPOINT_SHA256 = p1b.P1A_CHECKPOINT_SHA256
P1A_ATOM_STATE_SHA256 = p1b.P1A_ATOM_STATE_SHA256
DEFAULT_OUTPUT = Path(
    "outputs/checkpoints/chemvector-g3-p1c-top128-global-set-ranker-train5k"
)
MODEL_TYPE = "chemvector-g3-p1c-top128-global-set-ranker-train5k"
CHECKPOINT_STEPS = (1_250, 2_500, 3_750, 5_000, 6_250)
EXPECTED_ROUTE_SEQUENCE_SHA256 = p1b.EXPECTED_ROUTE_SEQUENCE_SHA256
IMAGE_SIZE = 768.0
MATCH_RADIUS_PIXELS = 6.0


def _hash_tensor(digest: "hashlib._Hash", value: torch.Tensor) -> None:
    tensor = value.detach().cpu().contiguous()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    digest.update(tensor.view(torch.uint8).numpy().tobytes())


class FrozenSetAuditAccumulator:
    """Hash target-free candidates and fixed hybrid labels in stream order."""

    def __init__(self) -> None:
        self.images = 0
        self.candidates = 0
        self.representable = 0
        self.canonical = 0
        self.targets = 0
        self.ids = hashlib.sha256()
        self.anchors = hashlib.sha256()
        self.features = hashlib.sha256()
        self.positive_ids = hashlib.sha256()
        self.canonical_ids = hashlib.sha256()

    def add(self, keys: list[str], candidates: dict, hybrid: dict) -> None:
        if len(keys) != len(candidates["candidate_ids"]):
            raise ValueError("P1-C audit keys and candidates differ")
        ids_cpu = candidates["candidate_ids"].detach().cpu()
        anchors_cpu = candidates["candidate_anchors"].detach().cpu()
        features_cpu = candidates["candidate_features"].detach().cpu()
        positive_cpu = hybrid["positive_mask"].detach().cpu()
        canonical_cpu = hybrid["canonical_positive_mask"].detach().cpu()
        for index, key in enumerate(keys):
            prefix = f"{key}\n".encode("utf-8")
            for digest in (
                self.ids,
                self.anchors,
                self.features,
                self.positive_ids,
                self.canonical_ids,
            ):
                digest.update(prefix)
            ids = ids_cpu[index]
            positive = positive_cpu[index]
            canonical = canonical_cpu[index]
            _hash_tensor(self.ids, ids)
            _hash_tensor(self.anchors, anchors_cpu[index])
            _hash_tensor(self.features, features_cpu[index])
            _hash_tensor(self.positive_ids, ids[positive])
            _hash_tensor(self.canonical_ids, ids[canonical])
        self.images += len(keys)
        self.candidates += int(candidates["candidate_ids"].numel())
        self.representable += int(hybrid["representable_counts"].sum())
        self.canonical += int(hybrid["canonical_counts"].sum())
        self.targets += int(hybrid["target_counts"].sum())

    def finalize(self) -> dict:
        return {
            "images": self.images,
            "candidate_count": self.candidates,
            "representable_positive_count": self.representable,
            "canonical_positive_count": self.canonical,
            "geometry_rescue_positive_count": self.representable - self.canonical,
            "target_count": self.targets,
            "candidate_flat_ids_sha256": self.ids.hexdigest(),
            "candidate_anchors_sha256": self.anchors.hexdigest(),
            "candidate_features_sha256": self.features.hexdigest(),
            "positive_flat_ids_sha256": self.positive_ids.hexdigest(),
            "canonical_positive_flat_ids_sha256": self.canonical_ids.hexdigest(),
        }


def load_p1a_set_ranker(device: torch.device):
    if file_sha256(P1A_CHECKPOINT) != P1A_CHECKPOINT_SHA256:
        raise ValueError("P1-C P1-A checkpoint SHA differs from the protocol")
    payload = torch.load(P1A_CHECKPOINT, map_location="cpu", weights_only=False)
    if (
        payload.get("checkpoint_schema_version")
        != "g3-p1a-atom-selector-checkpoint-0.1"
        or payload.get("global_step") != 6_250
        or payload.get("p1a_gate", {}).get("protocol_status") != "passed"
    ):
        raise ValueError("P1-C requires the protocol-valid P1-A fixed-final state")
    config = G3HierarchicalGraphConfig(**payload["g3_config"])
    base = G3AtomOnlyPredictor(config)
    base.atom_head.load_state_dict(payload["atom_head"], strict=True)
    atom_head = base.atom_head.requires_grad_(False).eval().to(device)
    if module_state_sha256(atom_head) != P1A_ATOM_STATE_SHA256:
        raise ValueError("P1-C P1-A Atom state SHA differs from the protocol")
    ranker = G3Top128SetRanker(config).to(device).train()
    if ranker.parameter_count() != G3_TOP128_SET_RANKER_PARAMETERS:
        raise ValueError("P1-C set-ranker parameter count differs from the protocol")
    return atom_head, ranker, payload, config


def validate_optimizer(
    atom_head: torch.nn.Module,
    ranker: G3Top128SetRanker,
    optimizer: torch.optim.Optimizer,
) -> None:
    expected = list(ranker.parameters())
    actual = [value for group in optimizer.param_groups for value in group["params"]]
    if len(expected) != len(actual) or {id(value) for value in expected} != {
        id(value) for value in actual
    }:
        raise ValueError("P1-C optimizer must contain only set-ranker parameters")
    if any(value.requires_grad for value in atom_head.parameters()):
        raise ValueError("P1-C frozen Atom head became trainable")


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
        "f1_6px": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
        "count_exact_images": int(total["count_exact_images"]),
        "count_mae": total["count_absolute_error"] / total["images"]
        if total["images"]
        else 0.0,
        "whole_set_exact_images": int(total["whole_set_exact_images"]),
    }


def _flat_dense_ids(selected: dict, index: int, height: int, width: int):
    mask = selected["atom_mask"][index]
    slots = selected["atom_slot_indices"][index, mask]
    return slots[:, 0] * height * width + slots[:, 1] * width + slots[:, 2]


@torch.no_grad()
def evaluate_route(
    atom_head,
    ranker,
    vision,
    manifest,
    collator,
    *,
    device: torch.device,
    precision: str,
    batch_size: int,
    require_zero_parity: bool,
) -> dict:
    atom_head.eval()
    ranker.eval()
    vision.eval()
    known = _empty_metrics()
    deploy = _empty_metrics()
    identity_matched = identity_whole = 0
    top128_matched = fixed_localized = fixed_total = 0
    sample_digest = hashlib.sha256()
    count_digest = hashlib.sha256()
    predicted_counts = []
    zero_score_parity = zero_selection_parity = True
    count_invariant_scores = target_free_forward = True
    residual_abs_max = 0.0
    set_audit = FrozenSetAuditAccumulator()
    for batch in p1a.iter_manifest_batches(manifest, collator, batch_size=batch_size):
        for key in batch["keys"]:
            sample_digest.update(f"{key}\n".encode("utf-8"))
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        targets = {
            "atom_anchors": batch["atom_anchors"].to(device, non_blocking=True),
            "atom_mask": batch["atom_mask"].to(device, non_blocking=True),
        }
        with _autocast_context(device, precision):
            features = vision.forward_stroke_resolution(pixels)
            dense = atom_head(features.detach())
            candidates = extract_frozen_top128(dense)
            ranked = ranker(candidates)
        hybrid = build_hybrid_top128_targets(candidates, targets)
        set_audit.add(batch["keys"], candidates, hybrid)
        gt_counts = targets["atom_mask"].sum(1).long()
        deploy_counts = dense["count_logits"].argmax(-1)
        known_selected = select_ranked_top128(ranked, gt_counts)
        deploy_selected = select_ranked_top128(ranked, deploy_counts)
        predicted_counts.extend(int(value) for value in deploy_counts.cpu().tolist())
        _hash_tensor(count_digest, dense["count_logits"].float())
        snapshot = ranked["ranking_logits"].detach().clone()
        for probe in (
            torch.ones_like(gt_counts),
            gt_counts,
            torch.full_like(gt_counts, 128),
        ):
            select_ranked_top128(ranked, probe)
            count_invariant_scores &= torch.equal(snapshot, ranked["ranking_logits"])
        repeated = ranker(candidates)
        target_free_forward &= torch.equal(
            ranked["ranking_logits"], repeated["ranking_logits"]
        )
        residual_abs_max = max(
            residual_abs_max,
            float(ranked["ranking_residual"].float().abs().max().cpu()),
        )
        if require_zero_parity:
            zero_score_parity &= torch.equal(
                ranked["ranking_logits"], ranked["base_logits"].float()
            )
        canonical = build_canonical_atom_slot_targets(dense, targets)
        batch_now, _, height, width = dense["presence_logits"].shape
        dense_anchors = dense["anchors"].permute(0, 1, 3, 4, 2)
        expected_anchors = canonical["anchors"].permute(0, 1, 3, 4, 2)
        if require_zero_parity:
            base_selected = select_count_conditioned_atom_slots(
                dense, gt_counts, maximum_atoms=128
            )
        for index in range(batch_now):
            target = targets["atom_anchors"][index, targets["atom_mask"][index]].float().cpu()
            known_mask = known_selected["atom_mask"][index]
            deploy_mask = deploy_selected["atom_mask"][index]
            _merge_metrics(
                known,
                p1a.atom_sample_statistics(
                    known_selected["atom_anchors"][index, known_mask].float().cpu(),
                    target,
                ),
            )
            _merge_metrics(
                deploy,
                p1a.atom_sample_statistics(
                    deploy_selected["atom_anchors"][index, deploy_mask].float().cpu(),
                    target,
                ),
            )
            selected_ids = set(
                known_selected["atom_flat_ids"][index, known_mask].cpu().tolist()
            )
            canonical_ids = set(
                candidates["candidate_ids"][
                    index, hybrid["canonical_positive_mask"][index]
                ]
                .cpu()
                .tolist()
            )
            identity_matched += len(selected_ids & canonical_ids)
            identity_whole += int(
                len(canonical_ids) == len(target) and selected_ids == canonical_ids
            )
            top128_matched += int(
                p1a.atom_sample_statistics(
                    candidates["candidate_anchors"][index].float().cpu(), target
                )["matched_atoms"]
            )
            distances = torch.linalg.vector_norm(
                (
                    dense_anchors[index][canonical["positive"][index]]
                    - expected_anchors[index][canonical["positive"][index]]
                ).float(),
                dim=-1,
            )
            fixed_localized += int(
                (distances * IMAGE_SIZE <= MATCH_RADIUS_PIXELS).sum()
            )
            fixed_total += len(distances)
            if require_zero_parity:
                base_ids = set(
                    _flat_dense_ids(base_selected, index, height, width).cpu().tolist()
                )
                zero_selection_parity &= selected_ids == base_ids
        del pixels, features, dense, candidates, ranked, hybrid
    known_metrics = _finalize_metrics(known)
    count_bytes = json.dumps(predicted_counts, separators=(",", ":")).encode()
    return {
        "scope": "route-select100-top128-global-set-ranker",
        "known_count": known_metrics,
        "deploy_count": _finalize_metrics(deploy),
        "canonical_identity": {
            "matched": identity_matched,
            "total": known_metrics["target_atoms"],
            "recall": identity_matched / known_metrics["target_atoms"],
            "whole_set_exact_images": identity_whole,
        },
        "top128_coverage": {
            "matched": top128_matched,
            "total": known_metrics["target_atoms"],
            "recall": top128_matched / known_metrics["target_atoms"],
        },
        "fixed_positive_localization": {
            "matched": fixed_localized,
            "total": fixed_total,
            "recall": fixed_localized / fixed_total,
        },
        "frozen_set_audit": set_audit.finalize(),
        "deploy_count_vector": predicted_counts,
        "deploy_count_vector_sha256": hashlib.sha256(count_bytes).hexdigest(),
        "count_logits_sha256": count_digest.hexdigest(),
        "sample_keys_sha256": sample_digest.hexdigest(),
        "residual_abs_max": residual_abs_max,
        "zero_score_parity": zero_score_parity,
        "legacy_topk_selection_parity": zero_selection_parity,
        "count_invariant_ranked_scores": count_invariant_scores,
        "target_free_forward_bytes": target_free_forward,
    }


def p1c_training_step(
    atom_head,
    ranker,
    vision,
    optimizer,
    batch: dict,
    *,
    device: torch.device,
    precision: str,
    max_grad_norm: float,
) -> tuple[dict[str, float], dict]:
    validate_optimizer(atom_head, ranker, optimizer)
    if any(value.requires_grad for value in vision.parameters()):
        raise ValueError("P1-C vision must remain frozen")
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
    hybrid = build_hybrid_top128_targets(candidates, targets)
    losses = top128_set_ranking_loss(ranked, hybrid, negative_multiplier=4)
    losses["loss"].backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(ranker.parameters(), max_grad_norm)
    if not bool(torch.isfinite(gradient_norm)):
        raise FloatingPointError("P1-C set-ranker gradient norm is not finite")
    optimizer.step()
    if any(value.grad is not None for value in atom_head.parameters()):
        raise RuntimeError("P1-C produced gradients for the frozen Atom head")
    if any(value.grad is not None for value in vision.parameters()):
        raise RuntimeError("P1-C produced gradients for frozen vision")
    metrics = {
        **{name: float(value.detach()) for name, value in losses.items()},
        "gradient_norm": float(gradient_norm.detach()),
        "batch_images": float(len(batch["keys"])),
    }
    return metrics, {"candidates": candidates, "hybrid": hybrid}


def selector_gate(protocol_checks: dict[str, bool], metrics: dict) -> dict:
    known = metrics["known_count"]
    capability = {
        "known_precision_97": known["precision_6px"] >= 0.97,
        "known_recall_97": known["recall_6px"] >= 0.97,
        "known_f1_97": known["f1_6px"] >= 0.97,
        "known_matched_2988": known["matched_atoms"] >= 2_988,
        "known_whole_50": known["whole_set_exact_images"] >= 50,
        "canonical_identity_95": metrics["canonical_identity"]["recall"] >= 0.95,
        "top128_coverage_guard": metrics["top128_coverage"]["matched"] >= 3_072,
        "fixed_positive_localization_guard": (
            metrics["fixed_positive_localization"]["matched"] == 3_080
            and metrics["fixed_positive_localization"]["total"] == 3_080
        ),
    }
    protocol_passed = all(protocol_checks.values())
    capability_passed = all(capability.values())
    return {
        "status": "passed" if protocol_passed and capability_passed else "failed",
        "decision": "GO" if protocol_passed and capability_passed else "NO-GO",
        "protocol_status": "passed" if protocol_passed else "failed",
        "protocol_checks": protocol_checks,
        "capability_status": "passed" if capability_passed else "failed",
        "capability_checks": capability,
        "fixed_final_step": 6_250,
    }


def _checkpoint_payload(
    *, ranker, optimizer, args, config, lineage, audit, step, epoch, exposure,
    losses, step0_metrics, final_metrics, gate, epoch_audits,
) -> dict:
    return {
        "checkpoint_schema_version": "g3-p1c-top128-set-ranker-checkpoint-0.1",
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
        "step0_route_metrics": step0_metrics,
        "final_route_metrics": final_metrics,
        "p1c_gate": gate,
        "completed_epoch_audits": [dict(value) for value in epoch_audits],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", default=str(p1a.TRAIN_MANIFEST))
    parser.add_argument("--route-manifest", default=str(p1a.ROUTE_SELECT_MANIFEST))
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
        "epochs": 5, "max_steps": 6_250, "batch_size": 4,
        "learning_rate": 3e-4, "weight_decay": 0.01, "max_grad_norm": 1.0,
        "save_every": 1_250, "num_workers": 0, "seed": 20260713,
    }
    for name, expected in fixed.items():
        if getattr(args, name) != expected:
            raise ValueError(f"P1-C requires fixed {name}={expected}")
    if args.device != "cuda" or args.precision != "bf16":
        raise ValueError("P1-C requires fixed CUDA BF16 training")
    if args.log_every < 1:
        raise ValueError("P1-C log interval must be positive")
    fixed_paths = {
        "train_manifest": p1a.TRAIN_MANIFEST,
        "route_manifest": p1a.ROUTE_SELECT_MANIFEST,
        "p1a_checkpoint": P1A_CHECKPOINT,
        "output_dir": DEFAULT_OUTPUT,
    }
    for name, expected in fixed_paths.items():
        if Path(getattr(args, name)).resolve() != expected.resolve():
            raise ValueError(f"P1-C requires fixed {name}: {expected}")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("P1-C CUDA training requested but CUDA is unavailable")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"P1-C output directory is not empty: {output_dir}")
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

    train_manifest, train_manifest_audit = p1a.load_train_manifest(args.train_manifest)
    route_manifest, route_manifest_audit = p1a.load_route_manifest(args.route_manifest)
    collator = p1a.G3AtomCollator(maximum_atoms=128, image_size=768)
    train_preflight = p1a.stream_preflight(train_manifest, collator)
    route_preflight = p1a.stream_preflight(route_manifest, collator)
    p1a.validate_occupancy(train_preflight, name="P1-C train5k")
    p1a.validate_occupancy(route_preflight, name="P1-C route-select100")
    overlap = p1a.identity_overlap_audit(
        train_manifest, route_manifest, train_preflight, route_preflight
    )
    if not overlap["all_three_identity_zero"]:
        raise ValueError("P1-C train and route identities overlap")
    if (
        train_manifest_audit["sha256"] != p1a.TRAIN_MANIFEST_SHA256
        or train_preflight["sample_keys_sha256"] != p1a.TRAIN_SEQUENCE_SHA256
        or route_preflight["sample_keys_sha256"] != EXPECTED_ROUTE_SEQUENCE_SHA256
    ):
        raise ValueError("P1-C manifest identity/order differs from the protocol")

    atom_head, ranker, p1a_payload, config = load_p1a_set_ranker(device)
    vision_path = Path(p1a_payload["lineage"]["vision"]["path"])
    vision, _, _, vision_sha = load_frozen_vision_checkpoint(vision_path, device)
    if vision_sha != p1b.DEFAULT_VISION_CHECKPOINT_SHA256:
        raise ValueError("P1-C vision checkpoint SHA differs from the protocol")
    vision.requires_grad_(False).eval()
    initial_atom_sha = module_state_sha256(atom_head)
    initial_vision_sha = module_state_sha256(vision)
    optimizer = torch.optim.AdamW(
        ranker.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    validate_optimizer(atom_head, ranker, optimizer)
    lineage = {
        "p1a": {
            "path": str(P1A_CHECKPOINT.resolve()), "sha256": P1A_CHECKPOINT_SHA256,
            "global_step": 6_250, "atom_head_state_sha256": initial_atom_sha,
        },
        "vision": {
            "path": str(vision_path.resolve()), "sha256": vision_sha,
            "state_sha256": initial_vision_sha,
        },
    }
    audit = {
        "schema_version": "g3-p1c-training-audit-0.1",
        "scope": "train-distribution-top128-global-set-ranker-train5k",
        "manifests": {
            "train5k": train_manifest_audit,
            "route_select100": route_manifest_audit,
            "identity_overlap": overlap,
        },
        "preflight": {"train5k": train_preflight, "route_select100": route_preflight},
        "lineage": lineage,
        "implementation": {
            "trainer": str(Path(__file__).resolve()),
            "trainer_sha256": file_sha256(__file__),
            "ranker_sha256": file_sha256(
                Path(__file__).parents[2] / "components/atoms/g3_top128_set_ranker.py"
            ),
        },
        "optimizer": {
            "name": "AdamW", "parameter_count": G3_TOP128_SET_RANKER_PARAMETERS,
            "set_ranker_only": True, "restored_state": False,
        },
        "streaming": {
            "shuffle": False, "num_workers": 0, "drop_last": False,
            "whole_dataset_cache": False,
        },
        "constructed_modules": {
            "vision": type(vision).__name__, "atom_head": type(atom_head).__name__,
            "set_ranker": type(ranker).__name__,
        },
        "forbidden_components_loaded": {
            "p1b_ranker": False, "bond": False, "atom_pair": False,
            "ocr": False, "chemscene": False,
        },
        "target_free_forward": tuple(inspect.signature(ranker.forward).parameters)
        == ("candidates",),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "audit.json", audit)
    _atomic_json(output_dir / "training_args.json", vars(args))
    _atomic_json(output_dir / "config.json", {"g3_config": asdict(config)})

    route_steps = [0]
    step0 = evaluate_route(
        atom_head, ranker, vision, route_manifest, collator, device=device,
        precision=args.precision, batch_size=args.batch_size, require_zero_parity=True,
    )
    step0["step"] = 0
    expected_step0 = {
        "fixed": step0["fixed_positive_localization"]["matched"] == 3_080,
        "count_exact": step0["deploy_count"]["count_exact_images"] == 13,
        "count_mae": step0["deploy_count"]["count_mae"] == 3.83,
        "zero_score": step0["zero_score_parity"],
        "count_invariant": step0["count_invariant_ranked_scores"],
        "target_free": step0["target_free_forward_bytes"],
    }
    if not all(expected_step0.values()):
        raise RuntimeError(f"P1-C step0 failed fixed baseline: {expected_step0}")
    _atomic_json(output_dir / "step0_route_metrics.json", step0)
    ranker.train()

    step = exposure = 0
    saved_steps = []
    epoch_audits = []
    reference_set_hashes = None
    losses = {}
    running = {}
    started = time.perf_counter()
    log_path = output_dir / "training_log.jsonl"
    for epoch in range(1, 6):
        key_digest = hashlib.sha256()
        set_accumulator = FrozenSetAuditAccumulator()
        epoch_images = epoch_steps = 0
        for batch in p1a.iter_manifest_batches(train_manifest, collator, batch_size=4):
            if len(batch["keys"]) != 4:
                raise RuntimeError("P1-C encountered a residual training batch")
            for key in batch["keys"]:
                key_digest.update(f"{key}\n".encode())
            losses, evidence = p1c_training_step(
                atom_head, ranker, vision, optimizer, batch, device=device,
                precision=args.precision, max_grad_norm=args.max_grad_norm,
            )
            set_accumulator.add(
                batch["keys"], evidence["candidates"], evidence["hybrid"]
            )
            step += 1
            epoch_steps += 1
            exposure += 4
            epoch_images += 4
            for name, value in losses.items():
                running[name] = running.get(name, 0.0) + value
            if step % args.log_every == 0:
                record = {
                    "step": step, "epoch": epoch, "samples_seen": exposure,
                    "elapsed_seconds": time.perf_counter() - started,
                    **{name: value / args.log_every for name, value in running.items()},
                }
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                print(json.dumps(record, separators=(",", ":")), flush=True)
                running.clear()
        frozen_set = set_accumulator.finalize()
        epoch_audit = {
            "epoch": epoch, "images": epoch_images, "steps": epoch_steps,
            "sample_keys_sha256": key_digest.hexdigest(), "frozen_set": frozen_set,
        }
        if (
            epoch_images != 5_000 or epoch_steps != 1_250
            or key_digest.hexdigest() != p1a.TRAIN_SEQUENCE_SHA256
        ):
            raise RuntimeError("P1-C epoch stream differs from fixed protocol")
        current_hashes = {
            key: value for key, value in frozen_set.items() if key.endswith("sha256")
        }
        if reference_set_hashes is None:
            reference_set_hashes = current_hashes
        elif current_hashes != reference_set_hashes:
            raise RuntimeError("P1-C frozen candidates or hybrid labels changed by epoch")
        epoch_audits.append(epoch_audit)
        if step in CHECKPOINT_STEPS[:-1]:
            payload = _checkpoint_payload(
                ranker=ranker, optimizer=optimizer, args=args, config=config,
                lineage=lineage, audit=audit, step=step, epoch=epoch,
                exposure=exposure, losses=losses, step0_metrics=step0,
                final_metrics=None, gate=None, epoch_audits=epoch_audits,
            )
            _atomic_torch_save(output_dir / f"step-{step:06d}.pt", payload)
            _atomic_torch_save(output_dir / "last.pt", payload)
            saved_steps.append(step)
    if step != 6_250 or exposure != 25_000:
        raise RuntimeError("P1-C final step/exposure differs from protocol")

    final = evaluate_route(
        atom_head, ranker, vision, route_manifest, collator, device=device,
        precision=args.precision, batch_size=4, require_zero_parity=False,
    )
    final["step"] = step
    route_steps.append(step)
    final_atom_sha = module_state_sha256(atom_head)
    final_vision_sha = module_state_sha256(vision)
    final_set = final["frozen_set_audit"]
    step0_set = step0["frozen_set_audit"]
    protocol_checks = {
        "train_manifest": train_manifest_audit["sha256"] == p1a.TRAIN_MANIFEST_SHA256,
        "train_sequence": len(epoch_audits) == 5 and all(
            value["sample_keys_sha256"] == p1a.TRAIN_SEQUENCE_SHA256
            for value in epoch_audits
        ),
        "frozen_train_set_identity": all(
            {k: v for k, v in value["frozen_set"].items() if k.endswith("sha256")}
            == reference_set_hashes for value in epoch_audits
        ),
        "route_sequence": step0["sample_keys_sha256"]
        == final["sample_keys_sha256"] == EXPECTED_ROUTE_SEQUENCE_SHA256,
        "route_candidate_identity": all(
            step0_set[key] == final_set[key]
            for key in step0_set if key.endswith("sha256")
        ),
        "identity_disjoint": overlap["all_three_identity_zero"],
        "p1a_checkpoint": file_sha256(P1A_CHECKPOINT) == P1A_CHECKPOINT_SHA256,
        "atom_state_unchanged": initial_atom_sha == final_atom_sha == P1A_ATOM_STATE_SHA256,
        "vision_state_unchanged": initial_vision_sha == final_vision_sha,
        "set_ranker_only_optimizer": validate_optimizer(atom_head, ranker, optimizer) is None,
        "fixed_exposure": step == 6_250 and exposure == 25_000,
        "checkpoint_schedule": saved_steps + [step] == list(CHECKPOINT_STEPS),
        "route_steps": route_steps == [0, 6_250],
        "count_vector_unchanged": step0["deploy_count_vector"] == final["deploy_count_vector"],
        "count_logits_unchanged": step0["count_logits_sha256"] == final["count_logits_sha256"],
        "deploy_count_metrics_unchanged": final["deploy_count"]["count_exact_images"] == 13
        and final["deploy_count"]["count_mae"] == 3.83,
        "count_invariant_scores": step0["count_invariant_ranked_scores"]
        and final["count_invariant_ranked_scores"],
        "target_free_forward": audit["target_free_forward"]
        and step0["target_free_forward_bytes"] and final["target_free_forward_bytes"],
        "forbidden_components_absent": not any(
            audit["forbidden_components_loaded"].values()
        ),
    }
    gate = selector_gate(protocol_checks, final)
    final_payload = _checkpoint_payload(
        ranker=ranker, optimizer=optimizer, args=args, config=config,
        lineage=lineage, audit=audit, step=step, epoch=5, exposure=exposure,
        losses=losses, step0_metrics=step0, final_metrics=final, gate=gate,
        epoch_audits=epoch_audits,
    )
    final_path = output_dir / "step-006250.pt"
    _atomic_torch_save(final_path, final_payload)
    _atomic_torch_save(output_dir / "last.pt", final_payload)
    saved_steps.append(step)
    expected_names = [f"step-{value:06d}.pt" for value in CHECKPOINT_STEPS]
    if sorted(path.name for path in output_dir.glob("step-*.pt")) != expected_names:
        raise RuntimeError("P1-C checkpoint file set differs from protocol")
    _atomic_json(output_dir / "epoch_audit.json", epoch_audits)
    _atomic_json(output_dir / "final_route_metrics.json", final)
    _atomic_json(output_dir / "p1c_gate.json", gate)
    _atomic_json(
        output_dir / "status.json",
        {
            "status": "completed", "decision": gate["decision"],
            "global_step": step, "samples_seen": exposure,
            "route_evaluation_steps": route_steps,
            "checkpoint": str(final_path.resolve()),
            "checkpoint_sha256": file_sha256(final_path),
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    print(json.dumps(gate, indent=2), flush=True)


if __name__ == "__main__":
    main()
