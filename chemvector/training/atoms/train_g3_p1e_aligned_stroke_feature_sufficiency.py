#!/usr/bin/env python3
"""Train the fixed P1-E CAP/SHAM/ALIGN strong-ranker experiment."""

from __future__ import annotations

import argparse
import copy
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

from chemvector.components.atoms.g3_boundary_band_loss import (
    G3_BOUNDARY_BAND_SIZE,
    boundary_band_ranking_loss,
)
from chemvector.components.atoms.g3_strong_top128_set_ranker import (
    G3_STRONG_SET_RANKER_PARAMETERS,
    G3StrongTop128SetRanker,
    align_candidate_patches,
    cap_zero_candidate_patches,
    extract_predicted_anchor_stroke_patches,
    sham_candidate_patches,
)
from chemvector.components.atoms.g3_top128_set_ranker import (
    build_hybrid_top128_targets,
    extract_frozen_top128,
    select_ranked_top128,
)
from chemvector.components.graph.joint.g3_hierarchical_graph import (
    G3AtomOnlyPredictor,
    G3HierarchicalGraphConfig,
    build_canonical_atom_slot_targets,
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


ARMS = ("cap", "sham", "align")
P1A_CHECKPOINT = p1c.P1A_CHECKPOINT
P1A_CHECKPOINT_SHA256 = p1c.P1A_CHECKPOINT_SHA256
P1A_ATOM_STATE_SHA256 = p1c.P1A_ATOM_STATE_SHA256
EXPECTED_ROUTE_SEQUENCE_SHA256 = p1c.EXPECTED_ROUTE_SEQUENCE_SHA256
DEFAULT_OUTPUT = Path(
    "outputs/checkpoints/chemvector-g3-p1e-aligned-stroke-feature-sufficiency-train5k"
)
MODEL_TYPE = "chemvector-g3-p1e-aligned-stroke-feature-sufficiency-train5k"
CHECKPOINT_STEPS = (1_250, 2_500, 3_750, 5_000, 6_250)
SHAM_EVAL_SEEDS = (31_001, 31_013, 31_019, 31_033, 31_039)
IMAGE_SIZE = 768.0
MATCH_RADIUS_PIXELS = 6.0
FROZEN_INPUT_SHA_KEYS = (
    "candidate_flat_ids_sha256",
    "candidate_anchors_sha256",
    "candidate_features_sha256",
    "positive_flat_ids_sha256",
    "canonical_positive_flat_ids_sha256",
    "raw_aligned_patch_sha256",
)
EXPECTED_DEPENDENCY_SHA256 = {
    "ranker": "b4b60d1f6d2a1709df45e765dbb4b978ea77a6f3c715a8cf4a7de211cacf1374",
    "loss": "7ea06a351b43d55987dda510974d27b85620d168c0d2df9ba46c0d623c9338c0",
    "p1c_helper": "845eabdd90bd386f77ca40744a27aa3b2f08f39978786a4d0f4f18efbeda478f",
    "p1d_helper": "f13cad9a1c4daeeeb8ae130a578ba2c54a6fb2a105d71408690f5d3b98461e87",
}


def validate_dependency_sources() -> dict[str, str]:
    paths = {
        "ranker": Path(__file__).parents[2]
        / "components/atoms/g3_strong_top128_set_ranker.py",
        "loss": Path(__file__).parents[2]
        / "components/atoms/g3_boundary_band_loss.py",
        "p1c_helper": Path(p1c.__file__),
        "p1d_helper": Path(p1d.__file__),
    }
    actual = {name: file_sha256(path) for name, path in paths.items()}
    if actual != EXPECTED_DEPENDENCY_SHA256:
        raise RuntimeError(f"P1-E dependency source SHA differs: {actual}")
    return actual


def load_p1a_frozen_atom_head(device: torch.device):
    if file_sha256(P1A_CHECKPOINT) != P1A_CHECKPOINT_SHA256:
        raise ValueError("P1-E P1-A checkpoint SHA differs from protocol")
    payload = torch.load(P1A_CHECKPOINT, map_location="cpu", weights_only=False)
    if (
        payload.get("checkpoint_schema_version")
        != "g3-p1a-atom-selector-checkpoint-0.1"
        or payload.get("global_step") != 6_250
        or payload.get("p1a_gate", {}).get("protocol_status") != "passed"
    ):
        raise ValueError("P1-E requires protocol-valid P1-A fixed-final")
    config = G3HierarchicalGraphConfig(**payload["g3_config"])
    base = G3AtomOnlyPredictor(config)
    base.atom_head.load_state_dict(payload["atom_head"], strict=True)
    atom_head = base.atom_head.requires_grad_(False).eval().to(device)
    if module_state_sha256(atom_head) != P1A_ATOM_STATE_SHA256:
        raise ValueError("P1-E P1-A Atom state SHA differs from protocol")
    return atom_head, payload, config


def build_identical_arms(config, device: torch.device):
    template = G3StrongTop128SetRanker(config).to(device)
    arms = {name: copy.deepcopy(template).train() for name in ARMS}
    state_hashes = {name: module_state_sha256(model) for name, model in arms.items()}
    if len(set(state_hashes.values())) != 1:
        raise RuntimeError("P1-E initial arm states differ")
    tensor_ids = [
        {id(tensor) for tensor in (*model.parameters(), *model.buffers())}
        for model in arms.values()
    ]
    if any(tensor_ids[left] & tensor_ids[right] for left in range(3) for right in range(left)):
        raise RuntimeError("P1-E arm parameters or buffers are shared")
    storage_pointers = [
        {
            tensor.data_ptr()
            for tensor in (*model.parameters(), *model.buffers())
            if tensor.numel()
        }
        for model in arms.values()
    ]
    if any(storage_pointers[left] & storage_pointers[right]
           for left in range(3) for right in range(left)):
        raise RuntimeError("P1-E arm parameter storage is shared")
    if any(model.num_parameters() != G3_STRONG_SET_RANKER_PARAMETERS for model in arms.values()):
        raise RuntimeError("P1-E strong-ranker parameter total differs")
    return arms, state_hashes


def validate_optimizers(atom_head, arms: dict, optimizers: dict) -> None:
    if tuple(arms) != ARMS or tuple(optimizers) != ARMS:
        raise ValueError("P1-E requires exactly CAP/SHAM/ALIGN arms")
    if any(parameter.requires_grad for parameter in atom_head.parameters()):
        raise ValueError("P1-E frozen Atom head became trainable")
    optimizer_ids = []
    for name in ARMS:
        expected = {id(parameter) for parameter in arms[name].parameters()}
        actual_parameters = [
            parameter
            for group in optimizers[name].param_groups
            for parameter in group["params"]
        ]
        actual = {id(parameter) for parameter in actual_parameters}
        if len(actual) != len(actual_parameters):
            raise ValueError(f"P1-E {name} optimizer contains duplicate parameters")
        if expected != actual:
            raise ValueError(f"P1-E {name} optimizer parameters differ")
        optimizer_ids.append(actual)
    if any(optimizer_ids[left] & optimizer_ids[right] for left in range(3) for right in range(left)):
        raise ValueError("P1-E optimizer parameters overlap across arms")


def arm_parameter_storage_is_disjoint(arms: dict) -> bool:
    pointers = [
        {
            tensor.data_ptr()
            for tensor in (*arms[name].parameters(), *arms[name].buffers())
            if tensor.numel()
        }
        for name in ARMS
    ]
    return not any(
        pointers[left] & pointers[right]
        for left in range(3)
        for right in range(left)
    )


def _validate_sham_assignment(
    aligned: torch.Tensor,
    sham_patches: torch.Tensor,
    permutation: torch.Tensor,
) -> None:
    identity = torch.arange(128, device=aligned.device)[None].expand(
        aligned.size(0), -1
    )
    if permutation.shape != identity.shape:
        raise RuntimeError("P1-E SHAM permutation shape differs")
    if bool((permutation == identity).any()):
        raise RuntimeError("P1-E SHAM permutation contains a fixed point")
    if any(len(row.unique()) != 128 for row in permutation):
        raise RuntimeError("P1-E SHAM permutation is not a bijection")
    inverse = permutation.argsort(dim=1)
    index = inverse[..., None, None, None].expand_as(aligned)
    if not torch.equal(sham_patches.gather(1, index), aligned):
        raise RuntimeError("P1-E ALIGN and SHAM patch multisets differ")


def build_arm_patches(aligned: torch.Tensor, keys: list[str], *, seed: int):
    sham = sham_candidate_patches(aligned, keys, seed=seed)
    identity = torch.arange(128, device=aligned.device)[None].expand(len(keys), -1)
    permutations = {
        "cap": identity,
        "sham": sham["permutation"],
        "align": identity,
    }
    patches = {
        "cap": cap_zero_candidate_patches(aligned),
        "sham": sham["candidate_patches"],
        "align": align_candidate_patches(aligned),
    }
    _validate_sham_assignment(
        aligned, patches["sham"], permutations["sham"]
    )
    return patches, permutations


class P1ESharedInputAuditAccumulator:
    def __init__(self) -> None:
        self.frozen = p1c.FrozenSetAuditAccumulator()
        self.raw_aligned_patch = hashlib.sha256()
        self.images = 0

    def add(self, keys, candidates, hybrid, aligned) -> None:
        self.frozen.add(keys, candidates, hybrid)
        aligned_cpu = aligned.detach().cpu()
        for index, key in enumerate(keys):
            prefix = f"{key}\n".encode("utf-8")
            self.raw_aligned_patch.update(prefix)
            p1c._hash_tensor(self.raw_aligned_patch, aligned_cpu[index])
        self.images += len(keys)

    def finalize(self) -> dict:
        return {
            **self.frozen.finalize(),
            "raw_aligned_patch_sha256": self.raw_aligned_patch.hexdigest(),
            "patch_images": self.images,
        }


class P1EArmAuditAccumulator:
    def __init__(self, arm: str) -> None:
        if arm not in (*ARMS, "align_on_sham"):
            raise ValueError("P1-E audit arm is invalid")
        self.arm = arm
        self.permutation = hashlib.sha256()
        self.images = 0
        self.fixed_points = 0

    def add(self, keys, permutation) -> None:
        permutation_cpu = permutation.detach().cpu()
        identity = torch.arange(128)
        for index, key in enumerate(keys):
            self.permutation.update(f"{key}\n".encode("utf-8"))
            p1c._hash_tensor(self.permutation, permutation_cpu[index])
            self.fixed_points += int((permutation_cpu[index] == identity).sum())
        self.images += len(keys)

    def finalize(self, shared: dict) -> dict:
        permutation_sha = self.permutation.hexdigest()
        certificate = hashlib.sha256(
            (
                f"{self.arm}\n{shared['raw_aligned_patch_sha256']}\n"
                f"{permutation_sha}\n"
            ).encode("ascii")
        ).hexdigest()
        return {
            **shared,
            "arm": self.arm,
            "permutation_sha256": permutation_sha,
            "patch_assignment_certificate_sha256": certificate,
            "permutation_images": self.images,
            "permutation_fixed_points": self.fixed_points,
            "patch_multiset_sha256": (
                shared["raw_aligned_patch_sha256"]
                if self.arm in ("sham", "align", "align_on_sham")
                else None
            ),
            "sham_multiset_verified_by_inverse_gather": self.arm in (
                "sham", "align_on_sham"
            ),
        }


def p1e_training_step(
    atom_head,
    arms,
    vision,
    optimizers,
    batch,
    *,
    device: torch.device,
    precision: str,
    max_grad_norm: float,
    seed: int,
):
    validate_optimizers(atom_head, arms, optimizers)
    if any(parameter.requires_grad for parameter in vision.parameters()):
        raise ValueError("P1-E vision must remain frozen")
    pixels = batch["pixel_values"].to(device, non_blocking=True)
    targets = {
        "atom_anchors": batch["atom_anchors"].to(device, non_blocking=True),
        "atom_mask": batch["atom_mask"].to(device, non_blocking=True),
    }
    with torch.no_grad(), _autocast_context(device, precision):
        stroke_features = vision.forward_stroke_resolution(pixels)
        dense = atom_head(stroke_features.detach())
        candidates = extract_frozen_top128(dense)
        aligned = extract_predicted_anchor_stroke_patches(stroke_features, candidates)
        patches, permutations = build_arm_patches(aligned, batch["keys"], seed=seed)
    hybrid = build_hybrid_top128_targets(candidates, targets)
    arm_metrics = {}
    for name in ARMS:
        optimizers[name].zero_grad(set_to_none=True)
        with _autocast_context(device, precision):
            ranked = arms[name](candidates, patches[name])
            losses = boundary_band_ranking_loss(
                ranked, hybrid, boundary_size=G3_BOUNDARY_BAND_SIZE
            )
        losses["loss"].backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            arms[name].parameters(), max_grad_norm
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError(f"P1-E {name} gradient norm is not finite")
        optimizers[name].step()
        arm_metrics[name] = {
            **{key: float(value.detach()) for key, value in losses.items()},
            "gradient_norm": float(gradient_norm.detach()),
        }
    if any(parameter.grad is not None for parameter in atom_head.parameters()):
        raise RuntimeError("P1-E produced gradients for frozen Atom head")
    if any(parameter.grad is not None for parameter in vision.parameters()):
        raise RuntimeError("P1-E produced gradients for frozen vision")
    return arm_metrics, {
        "candidates": candidates,
        "hybrid": hybrid,
        "permutations": permutations,
        "aligned_patches": aligned,
    }


def _empty_route_totals():
    return {name: 0.0 for name in (
        "images", "predicted_atoms", "target_atoms", "matched_atoms",
        "count_exact_images", "whole_set_exact_images", "count_absolute_error",
    )}


def _merge_route(total, current):
    for name in total:
        total[name] += float(current[name])


def _finalize_route(total):
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
        "f1_6px": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "count_exact_images": int(total["count_exact_images"]),
        "count_mae": total["count_absolute_error"] / total["images"] if total["images"] else 0.0,
        "whole_set_exact_images": int(total["whole_set_exact_images"]),
    }


def _new_route_arm():
    return {
        "known": _empty_route_totals(),
        "deploy": _empty_route_totals(),
        "identity_matched": 0,
        "identity_whole": 0,
        "residual_abs_max": 0.0,
    }


def _update_route_arm(state, ranked, hybrid, targets, deploy_counts):
    gt_counts = targets["atom_mask"].sum(1).long()
    known = select_ranked_top128(ranked, gt_counts)
    deploy = select_ranked_top128(ranked, deploy_counts)
    for index in range(len(gt_counts)):
        target = targets["atom_anchors"][index, targets["atom_mask"][index]].float().cpu()
        known_mask = known["atom_mask"][index]
        deploy_mask = deploy["atom_mask"][index]
        _merge_route(
            state["known"],
            p1a.atom_sample_statistics(known["atom_anchors"][index, known_mask].float().cpu(), target),
        )
        _merge_route(
            state["deploy"],
            p1a.atom_sample_statistics(deploy["atom_anchors"][index, deploy_mask].float().cpu(), target),
        )
        selected_ids = set(known["atom_flat_ids"][index, known_mask].cpu().tolist())
        canonical_ids = set(
            ranked["candidate_ids"][index, hybrid["canonical_positive_mask"][index]].cpu().tolist()
        )
        state["identity_matched"] += len(selected_ids & canonical_ids)
        state["identity_whole"] += int(
            len(canonical_ids) == len(target) and selected_ids == canonical_ids
        )
    state["residual_abs_max"] = max(
        state["residual_abs_max"],
        float(ranked["ranking_residual"].float().abs().max().cpu()),
    )


def _metric_at_path(metrics: dict, path: tuple[str, ...]) -> float:
    value = metrics
    for part in path:
        value = value[part]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"P1-E seed-bank metric is not numeric: {path}")
    return float(value)


def summarize_seed_bank(
    per_seed: dict,
    *,
    metric_paths: dict[str, tuple[str, ...]],
    selection_path: tuple[str, ...],
    selected_seed: int | None = None,
) -> dict:
    """Summarize a fixed seed bank with deterministic conservative selection."""

    normalized = {str(int(seed)): value for seed, value in per_seed.items()}
    if not normalized:
        raise ValueError("P1-E seed bank must not be empty")
    available = sorted(int(seed) for seed in normalized)
    if len(available) != len(normalized):
        raise ValueError("P1-E seed bank contains duplicate normalized seeds")
    selection_locked = selected_seed is not None
    if selected_seed is None:
        selected_seed = min(
            available,
            key=lambda seed: (
                -_metric_at_path(normalized[str(seed)], selection_path),
                seed,
            ),
        )
    if selected_seed not in available:
        raise ValueError("P1-E selected seed is absent from seed bank")
    best = normalized[str(selected_seed)]
    summary = {}
    for name, path in metric_paths.items():
        values = [_metric_at_path(normalized[str(seed)], path) for seed in available]
        summary[name] = {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "best": _metric_at_path(best, path),
        }
    return {
        "seeds": available,
        "per_seed": normalized,
        "best_seed": selected_seed,
        "best": best,
        "selection_path": ".".join(selection_path),
        "selection_policy": (
            "locked_from_train_seed_bank"
            if selection_locked
            else "maximum_then_smallest_seed_tie_break"
        ),
        "summary": summary,
    }


TRAIN_BANK_METRICS = {
    "boundary_exact_rate": ("boundary_exact_rate",),
    "hybrid_topm_whole_set_rate": ("hybrid_topm_whole_set_rate",),
    "hybrid_topm_recall": ("hybrid_topm_recall",),
}
ROUTE_BANK_METRICS = {
    "known_precision_6px": ("known_count", "precision_6px"),
    "known_recall_6px": ("known_count", "recall_6px"),
    "known_f1_6px": ("known_count", "f1_6px"),
    "known_whole_set_exact_images": ("known_count", "whole_set_exact_images"),
    "canonical_identity_recall": ("canonical_identity", "recall"),
}


@torch.no_grad()
def evaluate_route_all_arms(
    atom_head,
    arms,
    vision,
    manifest,
    collator,
    *,
    device,
    precision,
    batch_size,
    sham_seeds,
    include_counterfactual: bool,
    selected_sham_seed: int | None = None,
    selected_counterfactual_seed: int | None = None,
):
    for model in arms.values():
        model.eval()
    atom_head.eval()
    vision.eval()
    sham_seeds = tuple(int(seed) for seed in sham_seeds)
    if not sham_seeds or len(set(sham_seeds)) != len(sham_seeds):
        raise ValueError("P1-E route SHAM seeds must be non-empty and unique")
    if len(sham_seeds) > 1 and selected_sham_seed is None:
        raise ValueError("P1-E final route requires train-selected SHAM seed")
    if include_counterfactual and len(sham_seeds) > 1 and selected_counterfactual_seed is None:
        raise ValueError("P1-E final route requires train-selected counterfactual seed")
    sham_names = [f"sham:{seed}" for seed in sham_seeds]
    counterfactual_names = (
        [f"align_on_sham:{seed}" for seed in sham_seeds]
        if include_counterfactual
        else []
    )
    names = ["cap", "align", *sham_names, *counterfactual_names]
    states = {name: _new_route_arm() for name in names}
    shared_accumulator = P1ESharedInputAuditAccumulator()
    audits = {
        name: P1EArmAuditAccumulator(
            "align_on_sham" if name.startswith("align_on_sham:")
            else "sham" if name.startswith("sham:") else name
        )
        for name in names
    }
    sample_digest = hashlib.sha256()
    count_digest = hashlib.sha256()
    predicted_counts = []
    top128_matched = fixed_localized = fixed_total = 0
    score_parity = True
    for batch in p1a.iter_manifest_batches(manifest, collator, batch_size=batch_size):
        for key in batch["keys"]:
            sample_digest.update(f"{key}\n".encode("utf-8"))
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        targets = {
            "atom_anchors": batch["atom_anchors"].to(device, non_blocking=True),
            "atom_mask": batch["atom_mask"].to(device, non_blocking=True),
        }
        with _autocast_context(device, precision):
            stroke = vision.forward_stroke_resolution(pixels)
            dense = atom_head(stroke.detach())
            candidates = extract_frozen_top128(dense)
            aligned = extract_predicted_anchor_stroke_patches(stroke, candidates)
            cap_patches = cap_zero_candidate_patches(aligned)
            align_patches = align_candidate_patches(aligned)
            identity = torch.arange(128, device=aligned.device)[None].expand(
                len(batch["keys"]), -1
            )
            permutations = {"cap": identity, "align": identity}
            ranked = {
                "cap": arms["cap"](candidates, cap_patches),
                "align": arms["align"](candidates, align_patches),
            }
            for current_seed in sham_seeds:
                sham = sham_candidate_patches(
                    aligned, batch["keys"], seed=current_seed
                )
                _validate_sham_assignment(
                    aligned, sham["candidate_patches"], sham["permutation"]
                )
                sham_name = f"sham:{current_seed}"
                permutations[sham_name] = sham["permutation"]
                ranked[sham_name] = arms["sham"](
                    candidates, sham["candidate_patches"]
                )
                if include_counterfactual:
                    counterfactual_name = f"align_on_sham:{current_seed}"
                    permutations[counterfactual_name] = sham["permutation"]
                    ranked[counterfactual_name] = arms["align"](
                        candidates, sham["candidate_patches"]
                    )
        hybrid = build_hybrid_top128_targets(candidates, targets)
        deploy_counts = dense["count_logits"].argmax(-1)
        predicted_counts.extend(int(value) for value in deploy_counts.cpu().tolist())
        p1c._hash_tensor(count_digest, dense["count_logits"].float())
        score_parity &= all(
            torch.equal(value["ranking_logits"], ranked["cap"]["ranking_logits"])
            for value in ranked.values()
        )
        shared_accumulator.add(batch["keys"], candidates, hybrid, aligned)
        for name in names:
            _update_route_arm(states[name], ranked[name], hybrid, targets, deploy_counts)
            audits[name].add(batch["keys"], permutations[name])
        canonical = build_canonical_atom_slot_targets(dense, targets)
        dense_anchors = dense["anchors"].permute(0, 1, 3, 4, 2)
        expected = canonical["anchors"].permute(0, 1, 3, 4, 2)
        for index in range(len(targets["atom_mask"])):
            target = targets["atom_anchors"][index, targets["atom_mask"][index]].float().cpu()
            top128_matched += int(
                p1a.atom_sample_statistics(candidates["candidate_anchors"][index].float().cpu(), target)["matched_atoms"]
            )
            distances = torch.linalg.vector_norm(
                (dense_anchors[index][canonical["positive"][index]] - expected[index][canonical["positive"][index]]).float(),
                dim=-1,
            )
            fixed_localized += int((distances * IMAGE_SIZE <= MATCH_RADIUS_PIXELS).sum())
            fixed_total += len(distances)
    target_total = int(next(iter(states.values()))["known"]["target_atoms"])
    shared_input = shared_accumulator.finalize()
    all_metrics = {}
    for name, state in states.items():
        all_metrics[name] = {
            "known_count": _finalize_route(state["known"]),
            "deploy_count": _finalize_route(state["deploy"]),
            "canonical_identity": {
                "matched": state["identity_matched"],
                "total": target_total,
                "recall": state["identity_matched"] / target_total,
                "whole_set_exact_images": state["identity_whole"],
            },
            "residual_abs_max": state["residual_abs_max"],
            "input_audit": audits[name].finalize(shared_input),
        }
    sham_bank = summarize_seed_bank(
        {seed: all_metrics[f"sham:{seed}"] for seed in sham_seeds},
        metric_paths=ROUTE_BANK_METRICS,
        selection_path=("known_count", "f1_6px"),
        selected_seed=selected_sham_seed,
    )
    arm_metrics = {
        "cap": all_metrics["cap"],
        "sham": sham_bank["best"],
        "align": all_metrics["align"],
    }
    counterfactual_bank = None
    if include_counterfactual:
        counterfactual_bank = summarize_seed_bank(
            {
                seed: all_metrics[f"align_on_sham:{seed}"]
                for seed in sham_seeds
            },
            metric_paths=ROUTE_BANK_METRICS,
            selection_path=("known_count", "f1_6px"),
            selected_seed=selected_counterfactual_seed,
        )
        arm_metrics["align_on_sham"] = counterfactual_bank["best"]
    count_bytes = json.dumps(predicted_counts, separators=(",", ":")).encode("utf-8")
    return {
        "scope": "route-select100-p1e-three-arm",
        "arms": arm_metrics,
        "sham_eval_bank": sham_bank,
        "align_on_sham_eval_bank": counterfactual_bank,
        "top128_coverage": {"matched": top128_matched, "total": target_total},
        "fixed_positive_localization": {"matched": fixed_localized, "total": fixed_total},
        "deploy_count_vector": predicted_counts,
        "deploy_count_vector_sha256": hashlib.sha256(count_bytes).hexdigest(),
        "count_logits_sha256": count_digest.hexdigest(),
        "sample_keys_sha256": sample_digest.hexdigest(),
        "step0_arm_score_parity": score_parity,
    }


@torch.no_grad()
def evaluate_train_all_arms(
    atom_head, arms, vision, manifest, collator, *, device, precision, batch_size,
    sham_seeds=SHAM_EVAL_SEEDS,
):
    for model in arms.values():
        model.eval()
    atom_head.eval()
    vision.eval()
    sham_names = [f"sham:{seed}" for seed in sham_seeds]
    counterfactual_names = [f"align_on_sham:{seed}" for seed in sham_seeds]
    names = ["cap", "align", *sham_names, *counterfactual_names]
    totals = {
        name: {key: 0 for key in (
            "images", "positive_count", "matched_positive_count",
            "whole_set_exact_images", "boundary_exact_images",
        )}
        for name in names
    }
    shared_accumulator = P1ESharedInputAuditAccumulator()
    audits = {
        name: P1EArmAuditAccumulator(
            "align_on_sham" if name.startswith("align_on_sham:")
            else "sham" if name.startswith("sham:") else name
        )
        for name in names
    }
    sample_digest = hashlib.sha256()
    for batch in p1a.iter_manifest_batches(manifest, collator, batch_size=batch_size):
        for key in batch["keys"]:
            sample_digest.update(f"{key}\n".encode("utf-8"))
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        targets = {
            "atom_anchors": batch["atom_anchors"].to(device, non_blocking=True),
            "atom_mask": batch["atom_mask"].to(device, non_blocking=True),
        }
        with _autocast_context(device, precision):
            stroke = vision.forward_stroke_resolution(pixels)
            dense = atom_head(stroke.detach())
            candidates = extract_frozen_top128(dense)
            aligned = extract_predicted_anchor_stroke_patches(stroke, candidates)
            cap_patches = cap_zero_candidate_patches(aligned)
            align_patches = align_candidate_patches(aligned)
            identity = torch.arange(128, device=aligned.device)[None].expand(len(batch["keys"]), -1)
            permutations = {"cap": identity, "align": identity}
            ranked = {
                "cap": arms["cap"](candidates, cap_patches),
                "align": arms["align"](candidates, align_patches),
            }
            for current_seed in sham_seeds:
                sham = sham_candidate_patches(
                    aligned, batch["keys"], seed=current_seed
                )
                sham_name = f"sham:{current_seed}"
                counterfactual_name = f"align_on_sham:{current_seed}"
                _validate_sham_assignment(
                    aligned, sham["candidate_patches"], sham["permutation"]
                )
                permutations[sham_name] = sham["permutation"]
                permutations[counterfactual_name] = sham["permutation"]
                ranked[sham_name] = arms["sham"](
                    candidates, sham["candidate_patches"]
                )
                ranked[counterfactual_name] = arms["align"](
                    candidates, sham["candidate_patches"]
                )
        hybrid = build_hybrid_top128_targets(candidates, targets)
        shared_accumulator.add(batch["keys"], candidates, hybrid, aligned)
        for name in names:
            current = p1d.hybrid_topm_statistics(ranked[name], hybrid)
            for key in totals[name]:
                totals[name][key] += current[key]
            audits[name].add(batch["keys"], permutations[name])
    shared_input = shared_accumulator.finalize()
    output = {}
    for name in names:
        value = totals[name]
        output[name] = {
            **value,
            "hybrid_topm_recall": value["matched_positive_count"] / value["positive_count"],
            "hybrid_topm_whole_set_rate": value["whole_set_exact_images"] / value["images"],
            "boundary_exact_rate": value["boundary_exact_images"] / value["images"],
            "input_audit": audits[name].finalize(shared_input),
        }
    sham_bank = summarize_seed_bank(
        {seed: output[f"sham:{seed}"] for seed in sham_seeds},
        metric_paths=TRAIN_BANK_METRICS,
        selection_path=("boundary_exact_rate",),
    )
    counterfactual_bank = summarize_seed_bank(
        {seed: output[f"align_on_sham:{seed}"] for seed in sham_seeds},
        metric_paths=TRAIN_BANK_METRICS,
        selection_path=("boundary_exact_rate",),
    )
    return {
        "scope": "train5k-p1e-three-arm-fixed-final",
        "arms": {
            "cap": output["cap"],
            "sham": sham_bank["best"],
            "align": output["align"],
            "align_on_sham": counterfactual_bank["best"],
        },
        "sham_eval_bank": sham_bank,
        "align_on_sham_eval_bank": counterfactual_bank,
        "sample_keys_sha256": sample_digest.hexdigest(),
    }


def arm_gates(train_metrics, route_metrics):
    gates = {}
    for name in ARMS:
        train = train_metrics["arms"][name]
        route = route_metrics["arms"][name]
        known = route["known_count"]
        train_checks = {
            "boundary_exact_90": train["boundary_exact_rate"] >= 0.90,
            "whole_set_90": train["hybrid_topm_whole_set_rate"] >= 0.90,
            "recall_995": train["hybrid_topm_recall"] >= 0.995,
        }
        route_checks = {
            "known_precision_97": known["precision_6px"] >= 0.97,
            "known_recall_97": known["recall_6px"] >= 0.97,
            "known_f1_97": known["f1_6px"] >= 0.97,
            "known_matched_2988": known["matched_atoms"] >= 2_988,
            "known_whole_50": known["whole_set_exact_images"] >= 50,
            "canonical_identity_95": route["canonical_identity"]["recall"] >= 0.95,
            "top128_guard": route_metrics["top128_coverage"]["matched"] >= 3_072,
            "fixed_guard": route_metrics["fixed_positive_localization"] == {
                "matched": 3_080, "total": 3_080,
            },
        }
        gates[name] = {
            "train_fit_status": "passed" if all(train_checks.values()) else "failed",
            "train_fit_checks": train_checks,
            "route_status": "passed" if all(route_checks.values()) else "failed",
            "route_checks": route_checks,
        }
    align = train_metrics["arms"]["align"]["boundary_exact_rate"]
    sham = train_metrics["arms"]["sham"]["boundary_exact_rate"]
    counterfactual = train_metrics["arms"]["align_on_sham"]["boundary_exact_rate"]
    train_passed = {
        name: gates[name]["train_fit_status"] == "passed" for name in ARMS
    }
    align_minus_sham = align - sham >= 0.15
    counterfactual_drop = align - counterfactual >= 0.15
    if train_passed["cap"]:
        causal_decision = "current_token_capacity_sufficient"
    elif train_passed["sham"]:
        causal_decision = "image_level_patch_multiset_sufficient"
    elif train_passed["align"] and align_minus_sham and counterfactual_drop:
        causal_decision = "candidate_aligned_stroke_information_necessary"
    else:
        causal_decision = "no_go_indeterminate"
    return {
        "arms": gates,
        "causal_checks": {
            "align_minus_sham_15pp": align_minus_sham,
            "align_counterfactual_drop_15pp": counterfactual_drop,
        },
        "causal_decision": causal_decision,
    }


def checkpoint_payload(*, arm, model, optimizer, args, config, lineage, audit, step,
                       epoch, exposure, losses, step0, train_final, route_final,
                       gate, epoch_audits):
    return {
        "checkpoint_schema_version": "g3-p1e-three-arm-checkpoint-0.1",
        "model_type": MODEL_TYPE,
        "arm": arm,
        "strong_ranker": model.state_dict(),
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
        "step0_route_metrics": step0,
        "train_fixed_final": train_final,
        "route_fixed_final": route_final,
        "p1e_gate": gate,
        "completed_epoch_audits": [dict(value) for value in epoch_audits],
    }


def build_parser():
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


def validate_args(args):
    fixed = {
        "epochs": 5, "max_steps": 6_250, "batch_size": 4,
        "learning_rate": 3e-4, "weight_decay": 0.01,
        "max_grad_norm": 1.0, "save_every": 1_250,
        "num_workers": 0, "seed": 20260713,
    }
    for name, expected in fixed.items():
        if getattr(args, name) != expected:
            raise ValueError(f"P1-E requires fixed {name}={expected}")
    if args.device != "cuda" or args.precision != "bf16":
        raise ValueError("P1-E requires fixed CUDA BF16 training")
    if args.log_every < 1:
        raise ValueError("P1-E log interval must be positive")
    paths = {
        "train_manifest": p1a.TRAIN_MANIFEST,
        "route_manifest": p1a.ROUTE_SELECT_MANIFEST,
        "p1a_checkpoint": P1A_CHECKPOINT,
        "output_dir": DEFAULT_OUTPUT,
    }
    for name, expected in paths.items():
        if Path(getattr(args, name)).resolve() != expected.resolve():
            raise ValueError(f"P1-E requires fixed {name}: {expected}")


def main():
    args = build_parser().parse_args()
    validate_args(args)
    dependency_sha256 = validate_dependency_sources()
    if not torch.cuda.is_available():
        raise RuntimeError("P1-E CUDA training requested but unavailable")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"P1-E output directory is not empty: {output_dir}")
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
    p1a.validate_occupancy(train_preflight, name="P1-E train5k")
    p1a.validate_occupancy(route_preflight, name="P1-E route100")
    overlap = p1a.identity_overlap_audit(
        train_manifest, route_manifest, train_preflight, route_preflight
    )
    if not overlap["all_three_identity_zero"]:
        raise ValueError("P1-E train and route identities overlap")
    if (
        train_manifest_audit["sha256"] != p1a.TRAIN_MANIFEST_SHA256
        or train_preflight["sample_keys_sha256"] != p1a.TRAIN_SEQUENCE_SHA256
        or route_preflight["sample_keys_sha256"] != EXPECTED_ROUTE_SEQUENCE_SHA256
    ):
        raise ValueError("P1-E manifest identity/order differs")

    atom_head, p1a_payload, config = load_p1a_frozen_atom_head(device)
    vision_path = Path(p1a_payload["lineage"]["vision"]["path"])
    vision, _, _, vision_sha = load_frozen_vision_checkpoint(vision_path, device)
    if vision_sha != p1b.DEFAULT_VISION_CHECKPOINT_SHA256:
        raise ValueError("P1-E vision checkpoint SHA differs")
    vision.requires_grad_(False).eval()
    arms, initial_arm_hashes = build_identical_arms(config, device)
    optimizers = {
        name: torch.optim.AdamW(
            arms[name].parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        for name in ARMS
    }
    validate_optimizers(atom_head, arms, optimizers)
    initial_atom_sha = module_state_sha256(atom_head)
    initial_vision_sha = module_state_sha256(vision)
    lineage = {
        "p1a": {"path": str(P1A_CHECKPOINT.resolve()), "sha256": P1A_CHECKPOINT_SHA256},
        "vision": {"path": str(vision_path.resolve()), "sha256": vision_sha},
        "p1c_checkpoint_loaded": False,
        "p1d_checkpoint_loaded": False,
        "initial_arm_state_sha256": initial_arm_hashes,
    }
    audit = {
        "schema_version": "g3-p1e-training-audit-0.1",
        "manifests": {"train5k": train_manifest_audit, "route100": route_manifest_audit,
                      "identity_overlap": overlap},
        "preflight": {"train5k": train_preflight, "route100": route_preflight},
        "lineage": lineage,
        "implementation": {
            "trainer_sha256": file_sha256(__file__),
            **{f"{name}_sha256": value for name, value in dependency_sha256.items()},
        },
        "arms": {name: {"parameters": G3_STRONG_SET_RANKER_PARAMETERS,
                         "optimizer": "AdamW", "restored": False} for name in ARMS},
        "loss": {"boundary_size": 8, "margin": 0.0, "temperature": 1.0,
                 "only_loss": True},
        "streaming": {"batch": 4, "workers": 0, "shuffle": False,
                      "whole_dataset_cache": False},
        "forbidden_components_loaded": {
            "p1c_checkpoint": False, "p1d_checkpoint": False,
            "bond": False, "ocr": False, "chemscene": False,
        },
        "ranker_forward_signature": tuple(inspect.signature(
            next(iter(arms.values())).forward
        ).parameters),
    }
    output_dir.mkdir(parents=True)
    for name in ARMS:
        (output_dir / name).mkdir()
    _atomic_json(output_dir / "audit.json", audit)
    _atomic_json(output_dir / "training_args.json", vars(args))

    route_steps = [0]
    step0 = evaluate_route_all_arms(
        atom_head, arms, vision, route_manifest, collator, device=device,
        precision=args.precision, batch_size=4, sham_seeds=(args.seed,),
        include_counterfactual=False,
    )
    step0["step"] = 0
    if not all((
        step0["step0_arm_score_parity"],
        step0["fixed_positive_localization"] == {"matched": 3_080, "total": 3_080},
        all(value["known_count"] == step0["arms"]["cap"]["known_count"]
            for value in step0["arms"].values()),
    )):
        raise RuntimeError("P1-E step0 arm parity or P1-A baseline failed")
    _atomic_json(output_dir / "step0_route_metrics.json", step0)
    for model in arms.values():
        model.train()

    step = exposure = 0
    saved_steps = {name: [] for name in ARMS}
    epoch_audits = {name: [] for name in ARMS}
    references = {name: None for name in ARMS}
    last_losses = {name: {} for name in ARMS}
    running = {name: {} for name in ARMS}
    started = time.perf_counter()
    for epoch in range(1, 6):
        key_digest = hashlib.sha256()
        shared_accumulator = P1ESharedInputAuditAccumulator()
        accumulators = {name: P1EArmAuditAccumulator(name) for name in ARMS}
        images = steps = 0
        for batch in p1a.iter_manifest_batches(train_manifest, collator, batch_size=4):
            if len(batch["keys"]) != 4:
                raise RuntimeError("P1-E encountered residual batch")
            for key in batch["keys"]:
                key_digest.update(f"{key}\n".encode("utf-8"))
            metrics, evidence = p1e_training_step(
                atom_head, arms, vision, optimizers, batch, device=device,
                precision=args.precision, max_grad_norm=args.max_grad_norm,
                seed=args.seed + epoch,
            )
            shared_accumulator.add(
                batch["keys"], evidence["candidates"], evidence["hybrid"],
                evidence["aligned_patches"],
            )
            for name in ARMS:
                accumulators[name].add(
                    batch["keys"], evidence["permutations"][name]
                )
                last_losses[name] = metrics[name]
                for key, value in metrics[name].items():
                    running[name][key] = running[name].get(key, 0.0) + value
            step += 1
            steps += 1
            exposure += 4
            images += 4
            if step % args.log_every == 0:
                record = {"step": step, "epoch": epoch, "samples_seen": exposure,
                          "elapsed_seconds": time.perf_counter() - started}
                for name in ARMS:
                    record[name] = {
                        key: value / args.log_every for key, value in running[name].items()
                    }
                    running[name].clear()
                with (output_dir / "training_log.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                print(json.dumps(record, separators=(",", ":")), flush=True)
        if images != 5_000 or steps != 1_250 or key_digest.hexdigest() != p1a.TRAIN_SEQUENCE_SHA256:
            raise RuntimeError("P1-E epoch stream differs")
        shared_input = shared_accumulator.finalize()
        for name in ARMS:
            arm_input = accumulators[name].finalize(shared_input)
            hashes = {key: arm_input[key] for key in FROZEN_INPUT_SHA_KEYS}
            if references[name] is None:
                references[name] = hashes
            elif references[name] != hashes:
                raise RuntimeError(f"P1-E {name} frozen inputs changed")
            epoch_audits[name].append({
                "epoch": epoch, "images": images, "steps": steps,
                "sample_keys_sha256": key_digest.hexdigest(),
                "sham_seed": args.seed + epoch,
                "input_audit": arm_input,
            })
            if step in CHECKPOINT_STEPS[:-1]:
                payload = checkpoint_payload(
                    arm=name, model=arms[name], optimizer=optimizers[name], args=args,
                    config=config, lineage=lineage, audit=audit, step=step,
                    epoch=epoch, exposure=exposure, losses=last_losses[name],
                    step0=step0, train_final=None, route_final=None, gate=None,
                    epoch_audits=epoch_audits[name],
                )
                _atomic_torch_save(output_dir / name / f"step-{step:06d}.pt", payload)
                _atomic_torch_save(output_dir / name / "last.pt", payload)
                saved_steps[name].append(step)
    if step != 6_250 or exposure != 25_000:
        raise RuntimeError("P1-E final step/exposure differs")

    train_final = evaluate_train_all_arms(
        atom_head, arms, vision, train_manifest, collator, device=device,
        precision=args.precision, batch_size=4, sham_seeds=SHAM_EVAL_SEEDS,
    )
    selected_sham_seed = train_final["sham_eval_bank"]["best_seed"]
    selected_counterfactual_seed = train_final[
        "align_on_sham_eval_bank"
    ]["best_seed"]
    route_final = evaluate_route_all_arms(
        atom_head, arms, vision, route_manifest, collator, device=device,
        precision=args.precision, batch_size=4, sham_seeds=SHAM_EVAL_SEEDS,
        include_counterfactual=True,
        selected_sham_seed=selected_sham_seed,
        selected_counterfactual_seed=selected_counterfactual_seed,
    )
    route_final["step"] = step
    route_steps.append(step)
    protocol_checks = {
        "train_sequence": all(
            len(epoch_audits[name]) == 5 and all(
                value["sample_keys_sha256"] == p1a.TRAIN_SEQUENCE_SHA256
                for value in epoch_audits[name]
            ) for name in ARMS
        ),
        "frozen_inputs": all(
            all({key: epoch["input_audit"][key] for key in FROZEN_INPUT_SHA_KEYS}
                == references[name]
                for epoch in epoch_audits[name]) for name in ARMS
        ),
        "sham_epoch_seeds": [value["sham_seed"] for value in epoch_audits["sham"]]
        == [20_260_714, 20_260_715, 20_260_716, 20_260_717, 20_260_718],
        "cross_arm_frozen_inputs": all(
            references[name] == references["cap"] for name in ARMS
        ),
        "sham_zero_fixed_points": all(
            value["input_audit"]["permutation_fixed_points"] == 0
            for value in epoch_audits["sham"]
        ),
        "sham_epoch_permutations_unique": len({
            value["input_audit"]["permutation_sha256"]
            for value in epoch_audits["sham"]
        }) == 5,
        "train_sham_align_multiset_equal": all(
            epoch_audits["sham"][index]["input_audit"]["patch_multiset_sha256"]
            == epoch_audits["align"][index]["input_audit"]["patch_multiset_sha256"]
            == epoch_audits["align"][index]["input_audit"]["raw_aligned_patch_sha256"]
            for index in range(5)
        ),
        "identity_arm_permutations": all(
            value["input_audit"]["permutation_fixed_points"]
            == value["images"] * 128
            for name in ("cap", "align")
            for value in epoch_audits[name]
        ),
        "train_final_sequence": train_final["sample_keys_sha256"] == p1a.TRAIN_SEQUENCE_SHA256,
        "route_sequence": step0["sample_keys_sha256"]
        == route_final["sample_keys_sha256"] == EXPECTED_ROUTE_SEQUENCE_SHA256,
        "p1a_only": not lineage["p1c_checkpoint_loaded"] and not lineage["p1d_checkpoint_loaded"],
        "frozen_states": module_state_sha256(atom_head) == initial_atom_sha == P1A_ATOM_STATE_SHA256
        and module_state_sha256(vision) == initial_vision_sha,
        "independent_optimizers": validate_optimizers(atom_head, arms, optimizers) is None,
        "initial_arm_state_equal": len(set(initial_arm_hashes.values())) == 1,
        "arm_parameter_storage_disjoint": arm_parameter_storage_is_disjoint(arms),
        "fixed_budget": step == 6_250 and exposure == 25_000,
        "checkpoint_schedule": all(saved_steps[name] + [step] == list(CHECKPOINT_STEPS)
                                   for name in ARMS),
        "route_steps": route_steps == [0, 6_250],
        "eval_seed_banks": (
            train_final["sham_eval_bank"]["seeds"]
            == train_final["align_on_sham_eval_bank"]["seeds"]
            == route_final["sham_eval_bank"]["seeds"]
            == route_final["align_on_sham_eval_bank"]["seeds"]
            == list(SHAM_EVAL_SEEDS)
            and route_final["sham_eval_bank"]["best_seed"]
            == selected_sham_seed
            and route_final["align_on_sham_eval_bank"]["best_seed"]
            == selected_counterfactual_seed
        ),
        "eval_sham_zero_fixed_points": all(
            value["input_audit"]["permutation_fixed_points"] == 0
            and value["input_audit"]["sham_multiset_verified_by_inverse_gather"]
            for bank_name in ("sham_eval_bank", "align_on_sham_eval_bank")
            for metrics in (
                train_final[bank_name], route_final[bank_name]
            )
            for value in metrics["per_seed"].values()
        ),
        "eval_sham_align_multiset_equal": all(
            value["input_audit"]["patch_multiset_sha256"]
            == evaluation["arms"]["align"]["input_audit"][
                "patch_multiset_sha256"
            ]
            for evaluation in (train_final, route_final)
            for bank_name in ("sham_eval_bank", "align_on_sham_eval_bank")
            for value in evaluation[bank_name]["per_seed"].values()
        ),
        "count_unchanged": step0["deploy_count_vector"] == route_final["deploy_count_vector"]
        and step0["count_logits_sha256"] == route_final["count_logits_sha256"],
        "forbidden_absent": not any(audit["forbidden_components_loaded"].values()),
        "target_free_forward": audit["ranker_forward_signature"]
        == ("candidates", "candidate_patches"),
    }
    gates = arm_gates(train_final, route_final)
    protocol_passed = all(protocol_checks.values())
    capability_passed = gates["causal_decision"] != "no_go_indeterminate"
    gate = {
        "status": "passed" if protocol_passed and capability_passed else "failed",
        "protocol_status": "passed" if protocol_passed else "failed",
        "capability_status": "passed" if capability_passed else "failed",
        "decision": "GO" if protocol_passed and capability_passed else "NO-GO",
        "protocol_checks": protocol_checks,
        **gates,
    }
    for name in ARMS:
        final_payload = checkpoint_payload(
            arm=name, model=arms[name], optimizer=optimizers[name], args=args,
            config=config, lineage=lineage, audit=audit, step=step, epoch=5,
            exposure=exposure, losses=last_losses[name], step0=step0,
            train_final=train_final["arms"][name],
            route_final=route_final["arms"][name], gate=gate,
            epoch_audits=epoch_audits[name],
        )
        path = output_dir / name / "step-006250.pt"
        _atomic_torch_save(path, final_payload)
        _atomic_torch_save(output_dir / name / "last.pt", final_payload)
        saved_steps[name].append(step)
        expected = sorted(f"step-{value:06d}.pt" for value in CHECKPOINT_STEPS)
        if sorted(value.name for value in (output_dir / name).glob("step-*.pt")) != expected:
            raise RuntimeError(f"P1-E {name} checkpoint files differ")
    _atomic_json(output_dir / "epoch_audit.json", epoch_audits)
    _atomic_json(output_dir / "train_fixed_final.json", train_final)
    _atomic_json(output_dir / "route_fixed_final.json", route_final)
    _atomic_json(output_dir / "p1e_gate.json", gate)
    _atomic_json(output_dir / "status.json", {
        "status": "completed", "protocol_status": gate["protocol_status"],
        "capability_status": gate["capability_status"],
        "decision": gate["decision"],
        "causal_decision": gate["causal_decision"],
        "global_step_per_arm": step, "samples_seen_per_arm": exposure,
        "route_evaluation_steps": route_steps,
        "train_diagnostic_steps": [step],
        "elapsed_seconds": time.perf_counter() - started,
        "arm_checkpoints": {
            name: {
                "path": str((output_dir / name / "step-006250.pt").resolve()),
                "sha256": file_sha256(output_dir / name / "step-006250.pt"),
            } for name in ARMS
        },
    })
    print(json.dumps(gate, indent=2), flush=True)


if __name__ == "__main__":
    main()
