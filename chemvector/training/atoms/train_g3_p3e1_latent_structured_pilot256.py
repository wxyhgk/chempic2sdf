#!/usr/bin/env python3
"""Pilot exact latent valid-vs-invalid Atom matching on capacity64 + general192."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from chemvector.components.atoms.objects.g3_atom_query_set import (
    G3AtomQuerySet,
    G3AtomQuerySetConfig,
    G3PointerUtilityExistenceModel,
    decode_g3_atom_query_set_joint,
)
from chemvector.components.atoms.objects.g3_latent_matching import (
    g3_best_invalid_assignment,
    g3_gt_candidate_compatibility,
    g3_latent_positive_assignment,
)
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.atoms import (
    train_g3_p2e_stride1_top128_selector_train5k as p2e,
)
from chemvector.training.atoms import train_g3_p2h_atom_query_set_capacity64 as p2h
from chemvector.training.atoms import (
    train_g3_p2w_fixed_identity_pointer_utility_step4800 as p2w,
)
from chemvector.training.atoms import audit_g3_p3e0_latent_solver as p3e0
from chemvector.training.atoms import (
    audit_g3_p3e0_best_invalid_solver as p3e0b,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _atomic_torch_save,
    _autocast_context,
    module_state_sha256,
)
from chemvector.g3_structural_data import file_sha256


MODEL_TYPE = "chemvector-g3-p3e1-latent-structured-pilot256"
OUTPUT = Path(f"outputs/smoke/{MODEL_TYPE}")
SOURCE_CHECKPOINT = p2w.OUTPUT / "step-004800.pt"
SOURCE_CHECKPOINT_SHA256 = (
    "229fd1299a68aa544c0bf3ee43acabcdc1386a9df3f57dfb153113c24328b6b9"
)
EPOCHS = 30
GENERAL_PER_BATCH = 3
CAPACITY_PER_BATCH = 1
STEPS_PER_EPOCH = 64
EXPECTED_STEPS = EPOCHS * STEPS_PER_EPOCH
EXPECTED_EXPOSURE = EXPECTED_STEPS * 4
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
STRUCTURED_MARGIN = 1.0
SEED = 20260713
TRAINABLE_PARAMETERS = 292_693


def legacy_edge_scores(output: dict[str, torch.Tensor]) -> torch.Tensor:
    return (
        F.logsigmoid(output["existence_logits"]).unsqueeze(-1)
        + output["pointer_logits"].log_softmax(dim=-1)
    )


def latent_structured_loss(
    output: dict[str, torch.Tensor],
    candidate_anchors: torch.Tensor,
    target_anchors: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    margin: float = STRUCTURED_MARGIN,
) -> dict[str, torch.Tensor]:
    if margin < 0.0:
        raise ValueError("P3-E1 structured margin must be nonnegative")
    scores = legacy_edge_scores(output)
    losses = []
    satisfied = 0
    valid_scores = []
    invalid_scores = []
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
        valid_scores.append(positive_score)
        if negative is None:
            losses.append(positive_score * 0.0)
            invalid_scores.append(positive_score.detach() - margin)
            satisfied += 1
            continue
        negative_score = scores[
            index, negative["query_indices"], negative["candidate_indices"]
        ].mean()
        invalid_scores.append(negative_score)
        current = (margin + negative_score - positive_score).clamp_min(0.0)
        losses.append(current)
        satisfied += int(float(current.detach()) == 0.0)
    loss = torch.stack(losses).mean()
    return {
        "loss": loss,
        "latent_structured_loss": loss,
        "margin_satisfied_images": loss.new_tensor(float(satisfied)),
        "valid_score_mean": torch.stack(valid_scores).mean(),
        "invalid_score_mean": torch.stack(invalid_scores).mean(),
    }


def training_step(model, optimizer, records, *, device) -> dict:
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
        )
    losses["loss"].backward()
    trainable = [value for value in model.parameters() if value.requires_grad]
    gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, MAX_GRAD_NORM)
    if not bool(torch.isfinite(gradient_norm)):
        raise FloatingPointError("P3-E1 gradient norm is not finite")
    optimizer.step()
    if any(value.grad is not None for value in model.base_model.count_head.parameters()):
        raise RuntimeError("P3-E1 produced Count-head gradients")
    return {
        **{name: float(value.detach()) for name, value in losses.items()},
        "gradient_norm": float(gradient_norm.detach()),
        "clip_scale": min(1.0, MAX_GRAD_NORM / max(float(gradient_norm), 1e-12)),
    }


@torch.no_grad()
def evaluate_structure(model, count_model, records, *, device) -> dict:
    model.eval()
    count_model.eval()
    known_totals = p1a._empty_atom_totals()
    predicted_totals = p1a._empty_atom_totals()
    digest = hashlib.sha256()
    failures = []
    for start in range(0, len(records), 4):
        current = records[start : start + 4]
        batch = p2h.collate(current, device)
        with _autocast_context(device, "bf16"):
            output = model(p2h.model_candidates(batch))
            count_output = count_model(p2h.model_candidates(batch))
        known_counts = batch["full_atom_mask"].sum(dim=1).long()
        predicted_counts = count_output["count_logits"].argmax(dim=-1).long()
        known = decode_g3_atom_query_set_joint(
            output, batch["candidate_anchors"], counts=known_counts
        )
        predicted = decode_g3_atom_query_set_joint(
            output, batch["candidate_anchors"], counts=predicted_counts
        )
        digest.update(predicted_counts.cpu().numpy().tobytes())
        for index, record in enumerate(current):
            expected = batch["atom_anchors"][
                index, batch["full_atom_mask"][index]
            ].float().cpu()
            known_stats = p1a.atom_sample_statistics(
                known[index]["anchors"].float().cpu(), expected
            )
            predicted_stats = p1a.atom_sample_statistics(
                predicted[index]["anchors"].float().cpu(), expected
            )
            p1a._merge_atom_totals(known_totals, known_stats)
            p1a._merge_atom_totals(predicted_totals, predicted_stats)
            if not known_stats["whole_set_exact_images"]:
                failures.append(str(record["sample_key"]))
    return {
        "known_count": p1a._finalize_atom_totals(known_totals),
        "frozen_p2w_count": p1a._finalize_atom_totals(predicted_totals),
        "frozen_count_predictions_sha256": digest.hexdigest(),
        "known_count_failure_sample_keys": failures,
    }


@torch.no_grad()
def evaluate_margins(model, records, *, device) -> dict:
    model.eval()
    satisfied = 0
    losses = []
    digest = hashlib.sha256()
    for start in range(0, len(records), 4):
        current = records[start : start + 4]
        batch = p2h.collate(current, device)
        with _autocast_context(device, "bf16"):
            output = model(p2h.model_candidates(batch))
        current_losses = latent_structured_loss(
            output,
            batch["candidate_anchors"],
            batch["atom_anchors"],
            batch["full_atom_mask"],
        )
        satisfied += int(current_losses["margin_satisfied_images"])
        losses.append(float(current_losses["loss"])*len(current))
        digest.update(
            f"{start}:{float(current_losses['loss']):.12f}:"
            f"{int(current_losses['margin_satisfied_images'])}\n".encode()
        )
    return {
        "images": len(records),
        "margin_satisfied_images": satisfied,
        "latent_structured_loss": sum(losses) / len(records),
        "digest": digest.hexdigest(),
    }


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P3-E1 output is not empty: {OUTPUT}")
    if file_sha256(SOURCE_CHECKPOINT) != SOURCE_CHECKPOINT_SHA256:
        raise ValueError("P3-E1 P2-W checkpoint SHA256 differs")
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
    model = p3e0.load_model(SOURCE_CHECKPOINT, device)
    count_model = p3e0.load_model(SOURCE_CHECKPOINT, device)
    count_model.requires_grad_(False)
    model.requires_grad_(True)
    model.base_model.count_head.requires_grad_(False)
    trainable = [value for value in model.parameters() if value.requires_grad]
    trainable_count = sum(value.numel() for value in trainable)
    if trainable_count != TRAINABLE_PARAMETERS:
        raise RuntimeError(f"P3-E1 trainable parameter count changed: {trainable_count}")
    optimizer = torch.optim.AdamW(
        trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    source_state_sha256 = module_state_sha256(model)
    frozen_count_state_sha256 = module_state_sha256(count_model)

    cache, cache_audit = p2e.load_cache()
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
    if len(general) != 192 or len(capacity) != 64:
        raise RuntimeError("P3-E1 pilot sample count changed")

    OUTPUT.mkdir(parents=True, exist_ok=False)
    step0 = {
        "capacity_easy": evaluate_structure(
            model, count_model, domains["easy32"], device=device
        ),
        "capacity_hard": evaluate_structure(
            model, count_model, domains["hard32"], device=device
        ),
        "general192": evaluate_structure(model, count_model, general, device=device),
        "capacity_margin": evaluate_margins(model, capacity, device=device),
        "general_margin": evaluate_margins(model, general, device=device),
    }
    _atomic_json(OUTPUT / "step0_metrics.json", step0)
    _atomic_json(OUTPUT / "audit.json", {
        "schema_version": "g3-p3e1-latent-structured-pilot256-audit-0.1",
        "scope": "exact latent valid-vs-best-invalid capacity pilot",
        "training_updates": EXPECTED_STEPS,
        "training_images": 256,
        "general_images": 192,
        "capacity_images": 64,
        "exposures_per_image": 30,
        "count_training": False,
        "coordinate_loss": False,
        "factorized_object_pointer_loss": False,
        "fixed_query_identity": False,
        "route100_read": False,
        "edge_score": "legacy logsigmoid(existence)+logsoftmax(pointer)",
        "structured_margin": STRUCTURED_MARGIN,
        "fixed_gate": {
            "capacity_known_and_frozen_count": "32/32 both domains",
            "general_known_exact_min": 190,
            "general_known_recall_min": 0.999,
            "capacity_margin_satisfied": 64,
            "general_margin_satisfied_min": 183,
        },
        "lineage": {
            "source_checkpoint": str(SOURCE_CHECKPOINT.resolve()),
            "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
            "source_model_state_sha256": source_state_sha256,
            "frozen_count_state_sha256": frozen_count_state_sha256,
            "candidate_cache_sha256": p2e.CACHE_SHA256,
            "candidate_cache_tensor_sha256": cache_audit["candidate_protocol"][
                "candidate_cache_sha256"
            ],
            "capacity_selection": selection_audit,
            "general_sample_keys": [str(record["sample_key"]) for record in general],
        },
    })

    generator = torch.Generator(device="cpu").manual_seed(SEED)
    step = exposure = 0
    started = time.perf_counter()
    log_path = OUTPUT / "train.jsonl"
    last = {}
    for epoch in range(1, EPOCHS + 1):
        general_order = torch.randperm(192, generator=generator).tolist()
        capacity_order = torch.randperm(64, generator=generator).tolist()
        sums = {}
        for batch_index in range(STEPS_PER_EPOCH):
            records = [
                general[index]
                for index in general_order[
                    batch_index * GENERAL_PER_BATCH : (batch_index + 1) * GENERAL_PER_BATCH
                ]
            ]
            records.append(capacity[capacity_order[batch_index]])
            last = training_step(model, optimizer, records, device=device)
            step += 1
            exposure += 4
            for name, value in last.items():
                sums[name] = sums.get(name, 0.0) + value
        record = {
            "epoch": epoch,
            "step": step,
            "exposure": exposure,
            "elapsed_seconds": time.perf_counter() - started,
            **{name: value / STEPS_PER_EPOCH for name, value in sums.items()},
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        print(json.dumps(record, separators=(",", ":")), flush=True)

    metrics = {
        "capacity_easy": evaluate_structure(
            model, count_model, domains["easy32"], device=device
        ),
        "capacity_hard": evaluate_structure(
            model, count_model, domains["hard32"], device=device
        ),
        "general192": evaluate_structure(model, count_model, general, device=device),
        "capacity_margin": evaluate_margins(model, capacity, device=device),
        "general_margin": evaluate_margins(model, general, device=device),
    }
    repeated = {
        "capacity_easy": evaluate_structure(
            model, count_model, domains["easy32"], device=device
        ),
        "capacity_hard": evaluate_structure(
            model, count_model, domains["hard32"], device=device
        ),
        "general192": evaluate_structure(model, count_model, general, device=device),
        "capacity_margin": evaluate_margins(model, capacity, device=device),
        "general_margin": evaluate_margins(model, general, device=device),
    }
    general_known = metrics["general192"]["known_count"]
    checks = {
        "capacity_easy_known_exact32": (
            metrics["capacity_easy"]["known_count"]["whole_set_exact_images"] == 32
        ),
        "capacity_hard_known_exact32": (
            metrics["capacity_hard"]["known_count"]["whole_set_exact_images"] == 32
        ),
        "capacity_easy_frozen_count_exact32": (
            metrics["capacity_easy"]["frozen_p2w_count"]["whole_set_exact_images"] == 32
        ),
        "capacity_hard_frozen_count_exact32": (
            metrics["capacity_hard"]["frozen_p2w_count"]["whole_set_exact_images"] == 32
        ),
        "general_known_exact_min190": general_known["whole_set_exact_images"] >= 190,
        "general_known_recall_min999": (
            general_known["matched_atoms"] / general_known["target_atoms"] >= 0.999
        ),
        "capacity_margin_satisfied64": (
            metrics["capacity_margin"]["margin_satisfied_images"] == 64
        ),
        "general_margin_satisfied_min183": (
            metrics["general_margin"]["margin_satisfied_images"] >= 183
        ),
    }
    protocol_checks = {
        "fixed_steps": step == EXPECTED_STEPS,
        "fixed_exposure": exposure == EXPECTED_EXPOSURE,
        "repeat_metrics": repeated == metrics,
        "frozen_count_unchanged": module_state_sha256(count_model) == frozen_count_state_sha256,
        "count_head_no_grad": all(
            not value.requires_grad for value in model.base_model.count_head.parameters()
        ),
    }
    decision = (
        "latent_structured_pilot256_go"
        if all(checks.values()) and all(protocol_checks.values())
        else "latent_structured_pilot256_no_go"
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
        "checkpoint_schema_version": "g3-p3e1-latent-structured-pilot256-0.1",
        "model_type": MODEL_TYPE,
        "atom_query_set": model.state_dict(),
        "model_config": asdict(model.config),
        "optimizer": optimizer.state_dict(),
        "global_step": step,
        "samples_seen": exposure,
        "training_images": 256,
        "trainable_parameter_count": TRAINABLE_PARAMETERS,
        "structured_margin": STRUCTURED_MARGIN,
        "last_losses": last,
        "metrics": final,
        "lineage": {
            "source_checkpoint": str(SOURCE_CHECKPOINT.resolve()),
            "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
            "source_model_state_sha256": source_state_sha256,
            "frozen_count_state_sha256": frozen_count_state_sha256,
        },
    }
    checkpoint = OUTPUT / f"step-{step:06d}.pt"
    _atomic_torch_save(checkpoint, payload)
    _atomic_torch_save(OUTPUT / "last.pt", payload)
    final["checkpoint"] = str(checkpoint.resolve())
    final["checkpoint_sha256"] = file_sha256(checkpoint)
    _atomic_json(OUTPUT / "final_metrics.json", final)
    _atomic_json(OUTPUT / "status.json", {
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
