#!/usr/bin/env python3
"""Run the preregistered P3-E2 fixed-vs-coverage margin pilot arms."""

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

from chemvector.components.atoms.objects.g3_latent_matching import (
    g3_best_invalid_assignment,
    g3_candidate_set_gt_coverage,
    g3_gt_candidate_compatibility,
    g3_latent_positive_assignment,
)
from chemvector.training.atoms import audit_g3_p3e0_best_invalid_solver as p3e0b
from chemvector.training.atoms import audit_g3_p3e0_latent_solver as p3e0
from chemvector.training.atoms import (
    train_g3_p2e_stride1_top128_selector_train5k as p2e,
)
from chemvector.training.atoms import train_g3_p2h_atom_query_set_capacity64 as p2h
from chemvector.training.atoms import train_g3_p3e1_latent_structured_pilot256 as p3e1
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _atomic_torch_save,
    _autocast_context,
    module_state_sha256,
)
from chemvector.g3_structural_data import file_sha256


ARM_CONFIGS = {
    "capacity-fixed": {
        "model_type": "chemvector-g3-p3e2-capacity-fixed-margin-pilot64",
        "training_scope": "capacity64",
        "margin_mode": "fixed_whole_image",
        "training_images": 64,
        "steps_per_epoch": 16,
        "expected_steps": 480,
        "expected_exposure": 1_920,
        "update_mode": "microbatch4",
    },
    "capacity-coverage": {
        "model_type": "chemvector-g3-p3e2-capacity-coverage-margin-pilot64",
        "training_scope": "capacity64",
        "margin_mode": "gt_coverage_deficiency",
        "training_images": 64,
        "steps_per_epoch": 16,
        "expected_steps": 480,
        "expected_exposure": 1_920,
        "update_mode": "microbatch4",
    },
    "capacity-coverage-fullset": {
        "model_type": "chemvector-g3-p3e3-capacity-coverage-fullset-pilot64",
        "training_scope": "capacity64",
        "margin_mode": "gt_coverage_deficiency",
        "training_images": 64,
        "steps_per_epoch": 16,
        "expected_steps": 30,
        "expected_exposure": 1_920,
        "update_mode": "fullset_gradient_accumulation",
    },
    "mixed-coverage": {
        "model_type": "chemvector-g3-p3e2-mixed-coverage-margin-pilot256",
        "training_scope": "mixed256",
        "margin_mode": "gt_coverage_deficiency",
        "training_images": 256,
        "steps_per_epoch": 64,
        "expected_steps": 1_920,
        "expected_exposure": 7_680,
        "update_mode": "microbatch4",
    },
    "mixed-coverage-fullset": {
        "model_type": "chemvector-g3-p3e4-mixed-coverage-fullset-pilot256",
        "training_scope": "mixed256",
        "margin_mode": "gt_coverage_deficiency",
        "training_images": 256,
        "steps_per_epoch": 64,
        "expected_steps": 30,
        "expected_exposure": 7_680,
        "update_mode": "fullset_gradient_accumulation",
    },
}
EPOCHS = 30
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
SEED = 20260713
TRAINABLE_PARAMETERS = 292_693


def structured_margin(mode: str, count: int, deficiency: int) -> float:
    if count < 1 or not 1 <= deficiency <= count:
        raise ValueError("P3-E2 structured margin counts are invalid")
    if mode == "fixed_whole_image":
        return 1.0
    if mode == "gt_coverage_deficiency":
        return deficiency / count
    raise ValueError(f"Unknown P3-E2 margin mode: {mode}")


def latent_structured_loss(
    output: dict[str, torch.Tensor],
    candidate_anchors: torch.Tensor,
    target_anchors: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    margin_mode: str,
) -> dict[str, torch.Tensor]:
    scores = p3e1.legacy_edge_scores(output)
    losses = []
    selected_satisfied = fixed_satisfied = coverage_satisfied = 0
    valid_scores = []
    invalid_scores = []
    margins = []
    deficiencies = []
    for index in range(len(scores)):
        targets = target_anchors[index, target_mask[index]].float()
        compatibility = g3_gt_candidate_compatibility(
            targets, candidate_anchors[index]
        )
        positive = g3_latent_positive_assignment(
            scores[index], compatibility, max_ownerships=p3e0.MAX_OWNERSHIPS
        )
        negative = g3_best_invalid_assignment(
            scores[index],
            compatibility,
            max_component_gt=p3e0b.MAX_COMPONENT_GT,
            max_witnesses=p3e0b.MAX_WITNESSES,
        )
        positive_score = scores[
            index, positive["query_indices"], positive["candidate_indices"]
        ].mean()
        if negative is None:
            losses.append(positive_score * 0.0)
            valid_scores.append(positive_score)
            invalid_scores.append(positive_score.detach() - 1.0)
            margins.append(0.0)
            deficiencies.append(0)
            selected_satisfied += 1
            fixed_satisfied += 1
            coverage_satisfied += 1
            continue
        negative_score = scores[
            index, negative["query_indices"], negative["candidate_indices"]
        ].mean()
        coverage = g3_candidate_set_gt_coverage(
            negative["candidate_indices"], compatibility
        )
        count = len(targets)
        deficiency = count - coverage
        margin = structured_margin(margin_mode, count, deficiency)
        selected = (margin + negative_score - positive_score).clamp_min(0.0)
        fixed = (1.0 + negative_score - positive_score).clamp_min(0.0)
        coverage_loss = (
            deficiency / count + negative_score - positive_score
        ).clamp_min(0.0)
        losses.append(selected)
        valid_scores.append(positive_score)
        invalid_scores.append(negative_score)
        margins.append(margin)
        deficiencies.append(deficiency)
        selected_satisfied += int(float(selected.detach()) == 0.0)
        fixed_satisfied += int(float(fixed.detach()) == 0.0)
        coverage_satisfied += int(float(coverage_loss.detach()) == 0.0)
    loss = torch.stack(losses).mean()
    return {
        "loss": loss,
        "latent_structured_loss": loss,
        "selected_margin_satisfied_images": loss.new_tensor(
            float(selected_satisfied)
        ),
        "fixed_margin_satisfied_images": loss.new_tensor(float(fixed_satisfied)),
        "coverage_margin_satisfied_images": loss.new_tensor(
            float(coverage_satisfied)
        ),
        "valid_score_mean": torch.stack(valid_scores).mean(),
        "invalid_score_mean": torch.stack(invalid_scores).mean(),
        "margin_mean": loss.new_tensor(float(np.mean(margins))),
        "deficiency_mean": loss.new_tensor(float(np.mean(deficiencies))),
    }


def training_step(model, optimizer, records, *, device, margin_mode: str) -> dict:
    batch = p2h.collate(records, device)
    optimizer.zero_grad(set_to_none=True)
    model.train()
    with _autocast_context(device, "bf16"):
        output = model(p2h.model_candidates(batch))
        losses = latent_structured_loss(
            output,
            batch["candidate_anchors"],
            batch["atom_anchors"],
            batch["full_atom_mask"],
            margin_mode=margin_mode,
        )
    losses["loss"].backward()
    trainable = [value for value in model.parameters() if value.requires_grad]
    gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, MAX_GRAD_NORM)
    if not bool(torch.isfinite(gradient_norm)):
        raise FloatingPointError("P3-E2 gradient norm is not finite")
    optimizer.step()
    if any(value.grad is not None for value in model.base_model.count_head.parameters()):
        raise RuntimeError("P3-E2 produced Count-head gradients")
    gradient_norm_float = float(gradient_norm.detach())
    return {
        **{name: float(value.detach()) for name, value in losses.items()},
        "gradient_norm": gradient_norm_float,
        "clip_scale": min(1.0, MAX_GRAD_NORM / max(gradient_norm_float, 1e-12)),
    }


def fullset_training_step(
    model, optimizer, batches, *, device, margin_mode: str
) -> dict:
    """Accumulate one exact epoch-wide sample mean before a single update."""

    optimizer.zero_grad(set_to_none=True)
    model.train()
    sums: dict[str, float] = {}
    image_count = sum(len(records) for records in batches)
    if image_count < 1:
        raise ValueError("P3-E3 full-set update cannot be empty")
    for records in batches:
        batch = p2h.collate(records, device)
        with _autocast_context(device, "bf16"):
            output = model(p2h.model_candidates(batch))
            losses = latent_structured_loss(
                output,
                batch["candidate_anchors"],
                batch["atom_anchors"],
                batch["full_atom_mask"],
                margin_mode=margin_mode,
            )
            scaled_loss = losses["loss"] * len(records) / image_count
        scaled_loss.backward()
        for name, value in losses.items():
            number = float(value.detach())
            if name.endswith("satisfied_images"):
                sums[name] = sums.get(name, 0.0) + number
            else:
                sums[name] = sums.get(name, 0.0) + number * len(records)
    trainable = [value for value in model.parameters() if value.requires_grad]
    gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, MAX_GRAD_NORM)
    if not bool(torch.isfinite(gradient_norm)):
        raise FloatingPointError("P3-E3 full-set gradient norm is not finite")
    optimizer.step()
    if any(value.grad is not None for value in model.base_model.count_head.parameters()):
        raise RuntimeError("P3-E3 produced Count-head gradients")
    gradient_norm_float = float(gradient_norm.detach())
    return {
        **{
            name: (value if name.endswith("satisfied_images") else value / image_count)
            for name, value in sums.items()
        },
        "gradient_norm": gradient_norm_float,
        "clip_scale": min(1.0, MAX_GRAD_NORM / max(gradient_norm_float, 1e-12)),
    }


@torch.no_grad()
def evaluate_margins(model, records, *, device, margin_mode: str) -> dict:
    model.eval()
    sums: dict[str, float] = {}
    digest = hashlib.sha256()
    for start in range(0, len(records), 4):
        current = records[start : start + 4]
        batch = p2h.collate(current, device)
        with _autocast_context(device, "bf16"):
            output = model(p2h.model_candidates(batch))
        values = latent_structured_loss(
            output,
            batch["candidate_anchors"],
            batch["atom_anchors"],
            batch["full_atom_mask"],
            margin_mode=margin_mode,
        )
        for name, value in values.items():
            number = float(value.detach())
            if name.endswith("satisfied_images"):
                sums[name] = sums.get(name, 0.0) + number
            else:
                sums[name] = sums.get(name, 0.0) + number * len(current)
        digest.update(
            f"{start}:{float(values['loss']):.12f}:"
            f"{int(values['selected_margin_satisfied_images'])}\n".encode()
        )
    return {
        "images": len(records),
        **{
            name: (value if name.endswith("satisfied_images") else value / len(records))
            for name, value in sums.items()
        },
        "digest": digest.hexdigest(),
    }


def select_records(cache: dict) -> tuple[dict[str, list[dict]], dict]:
    eligible = [
        record for record in cache["train5k"]
        if torch.equal(record["atom_mask"], record["full_atom_mask"])
    ]
    domains, selection_audit = p2h.select_capacity_domains(cache["train5k"])
    capacity = domains["easy32"] + domains["hard32"]
    capacity_keys = {str(record["sample_key"]) for record in capacity}
    general = sorted(
        (
            record for record in eligible
            if str(record["sample_key"]) not in capacity_keys
        ),
        key=lambda record: str(record["sample_key"]),
    )[:p3e0.GENERAL_IMAGES]
    capacity_molecules = {
        p2h._molecule_key(str(record["sample_key"])) for record in capacity
    }
    general_molecules = [
        p2h._molecule_key(str(record["sample_key"])) for record in general
    ]
    overlap_molecules = capacity_molecules & set(general_molecules)
    overlap_images = sum(value in capacity_molecules for value in general_molecules)
    audit = {
        "capacity_selection": selection_audit,
        "general_images": len(general),
        "general_unique_molecules": len(set(general_molecules)),
        "capacity_general_overlap_molecules": len(overlap_molecules),
        "capacity_general_overlap_images": overlap_images,
        "general_sample_keys": [str(record["sample_key"]) for record in general],
    }
    return {
        "easy32": domains["easy32"],
        "hard32": domains["hard32"],
        "capacity64": capacity,
        "general192": general,
    }, audit


def epoch_batches(config: dict, records: dict, generator: torch.Generator):
    if config["training_scope"] == "capacity64":
        easy_order = torch.randperm(32, generator=generator).tolist()
        hard_order = torch.randperm(32, generator=generator).tolist()
        for index in range(16):
            yield [
                records["easy32"][easy_order[index * 2]],
                records["easy32"][easy_order[index * 2 + 1]],
                records["hard32"][hard_order[index * 2]],
                records["hard32"][hard_order[index * 2 + 1]],
            ]
        return
    general_order = torch.randperm(192, generator=generator).tolist()
    capacity_order = torch.randperm(64, generator=generator).tolist()
    for index in range(64):
        yield [
            records["general192"][value]
            for value in general_order[index * 3 : (index + 1) * 3]
        ] + [records["capacity64"][capacity_order[index]]]


def evaluate_all(model, count_model, records, *, device, margin_mode: str) -> dict:
    return {
        "capacity_easy": p3e1.evaluate_structure(
            model, count_model, records["easy32"], device=device
        ),
        "capacity_hard": p3e1.evaluate_structure(
            model, count_model, records["hard32"], device=device
        ),
        "general192": p3e1.evaluate_structure(
            model, count_model, records["general192"], device=device
        ),
        "capacity_margin": evaluate_margins(
            model, records["capacity64"], device=device, margin_mode=margin_mode
        ),
        "general_margin": evaluate_margins(
            model, records["general192"], device=device, margin_mode=margin_mode
        ),
    }


def gate(config: dict, metrics: dict) -> tuple[str, dict[str, bool]]:
    easy = metrics["capacity_easy"]
    hard = metrics["capacity_hard"]
    general = metrics["general192"]["known_count"]
    checks = {
        "capacity_easy_known_exact32": (
            easy["known_count"]["whole_set_exact_images"] == 32
        ),
        "capacity_hard_known_exact32": (
            hard["known_count"]["whole_set_exact_images"] == 32
        ),
        "capacity_easy_frozen_count_exact32": (
            easy["frozen_p2w_count"]["whole_set_exact_images"] == 32
        ),
        "capacity_hard_frozen_count_exact32": (
            hard["frozen_p2w_count"]["whole_set_exact_images"] == 32
        ),
        "capacity_selected_margin_satisfied64": (
            metrics["capacity_margin"]["selected_margin_satisfied_images"] == 64
        ),
    }
    if config["training_scope"] == "mixed256":
        checks.update({
            "general_known_exact_min190": general["whole_set_exact_images"] >= 190,
            "general_known_recall_min999": (
                general["matched_atoms"] / general["target_atoms"] >= 0.999
            ),
            "general_selected_margin_satisfied_min183": (
                metrics["general_margin"]["selected_margin_satisfied_images"] >= 183
            ),
        })
    decision = "go" if all(checks.values()) else "no_go"
    return decision, checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=sorted(ARM_CONFIGS), required=True)
    args = parser.parse_args()
    config = ARM_CONFIGS[args.arm]
    output_dir = Path(f"outputs/smoke/{config['model_type']}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"P3-E2 output is not empty: {output_dir}")
    if file_sha256(p3e1.SOURCE_CHECKPOINT) != p3e1.SOURCE_CHECKPOINT_SHA256:
        raise ValueError("P3-E2 P2-W checkpoint SHA256 differs")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    model = p3e0.load_model(p3e1.SOURCE_CHECKPOINT, device)
    count_model = p3e0.load_model(p3e1.SOURCE_CHECKPOINT, device)
    count_model.requires_grad_(False)
    model.requires_grad_(True)
    model.base_model.count_head.requires_grad_(False)
    trainable = [value for value in model.parameters() if value.requires_grad]
    if sum(value.numel() for value in trainable) != TRAINABLE_PARAMETERS:
        raise RuntimeError("P3-E2 trainable parameter count changed")
    optimizer = torch.optim.AdamW(
        trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    source_state_sha256 = module_state_sha256(model)
    count_state_sha256 = module_state_sha256(count_model)

    cache, cache_audit = p2e.load_cache()
    records, selection_audit = select_records(cache)
    if len(records["general192"]) != 192:
        raise RuntimeError("P3-E2 general sample count changed")
    output_dir.mkdir(parents=True, exist_ok=False)
    step0 = evaluate_all(
        model,
        count_model,
        records,
        device=device,
        margin_mode=config["margin_mode"],
    )
    protocol = {
        "schema_version": "g3-p3e2-margin-ab-audit-0.1",
        "arm": args.arm,
        **config,
        "epochs": EPOCHS,
        "exposures_per_training_image": 30,
        "count_training": False,
        "coordinate_loss": False,
        "factorized_loss": False,
        "fixed_query_identity": False,
        "route100_read": False,
        "checkpoint_selection": "fixed_final_only",
        "fixed_gate": (
            "capacity double-domain exact + selected margin"
            if config["training_scope"] == "capacity64" else
            "P3-E1 strict structure gate with coverage-calibrated margin"
        ),
        "lineage": {
            "source_checkpoint": str(p3e1.SOURCE_CHECKPOINT.resolve()),
            "source_checkpoint_sha256": p3e1.SOURCE_CHECKPOINT_SHA256,
            "source_model_state_sha256": source_state_sha256,
            "candidate_cache_sha256": p2e.CACHE_SHA256,
            "candidate_cache_tensor_sha256": cache_audit["candidate_protocol"][
                "candidate_cache_sha256"
            ],
            "selection": selection_audit,
        },
    }
    _atomic_json(output_dir / "audit.json", protocol)
    _atomic_json(output_dir / "step0_metrics.json", step0)

    generator = torch.Generator(device="cpu").manual_seed(SEED)
    step = exposure = 0
    started = time.perf_counter()
    last = {}
    log_path = output_dir / "train.jsonl"
    for epoch in range(1, EPOCHS + 1):
        sums: dict[str, float] = {}
        batches = list(epoch_batches(config, records, generator))
        if len(batches) != config["steps_per_epoch"]:
            raise RuntimeError("P3-E2 epoch batch count changed")
        if config["update_mode"] == "fullset_gradient_accumulation":
            last = fullset_training_step(
                model,
                optimizer,
                batches,
                device=device,
                margin_mode=config["margin_mode"],
            )
            step += 1
            exposure += sum(len(current) for current in batches)
            sums = dict(last)
            log_divisor = 1
        else:
            for current in batches:
                last = training_step(
                    model,
                    optimizer,
                    current,
                    device=device,
                    margin_mode=config["margin_mode"],
                )
                step += 1
                exposure += len(current)
                for name, value in last.items():
                    sums[name] = sums.get(name, 0.0) + value
            log_divisor = config["steps_per_epoch"]
        log = {
            "epoch": epoch,
            "step": step,
            "exposure": exposure,
            "elapsed_seconds": time.perf_counter() - started,
            **{
                name: value / log_divisor
                for name, value in sums.items()
            },
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(log, separators=(",", ":")) + "\n")
        print(json.dumps(log, separators=(",", ":")), flush=True)

    metrics = evaluate_all(
        model,
        count_model,
        records,
        device=device,
        margin_mode=config["margin_mode"],
    )
    repeated = evaluate_all(
        model,
        count_model,
        records,
        device=device,
        margin_mode=config["margin_mode"],
    )
    arm_decision, checks = gate(config, metrics)
    protocol_checks = {
        "fixed_steps": step == config["expected_steps"],
        "fixed_exposure": exposure == config["expected_exposure"],
        "repeat_metrics": repeated == metrics,
        "frozen_count_unchanged": module_state_sha256(count_model) == count_state_sha256,
        "count_head_no_grad": all(
            not value.requires_grad for value in model.base_model.count_head.parameters()
        ),
    }
    decision = (
        f"{config['model_type']}_go"
        if arm_decision == "go" and all(protocol_checks.values())
        else f"{config['model_type']}_no_go"
    )
    final = {
        "decision": decision,
        "checks": checks,
        "protocol_checks": protocol_checks,
        "step0": step0,
        "metrics": metrics,
        "elapsed_seconds": time.perf_counter() - started,
    }
    payload = {
        "checkpoint_schema_version": "g3-p3e2-margin-ab-0.1",
        "model_type": config["model_type"],
        "arm": args.arm,
        "atom_query_set": model.state_dict(),
        "model_config": asdict(model.config),
        "optimizer": optimizer.state_dict(),
        "global_step": step,
        "samples_seen": exposure,
        "training_images": config["training_images"],
        "trainable_parameter_count": TRAINABLE_PARAMETERS,
        "margin_mode": config["margin_mode"],
        "last_losses": last,
        "metrics": final,
        "lineage": protocol["lineage"],
    }
    checkpoint = output_dir / f"step-{step:06d}.pt"
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
        "decision": decision,
        "checks": checks,
        "metrics": metrics,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": final["checkpoint_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
