#!/usr/bin/env python3
"""Fine-tune 128 Atom queries over true-global192 candidates on Micro160."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from chemvector.components.atoms.objects.g3_atom_query_set import (
    decode_g3_atom_query_set_joint,
    g3_atom_query_set_loss,
    hungarian_pointer_query_targets,
)
from chemvector.evaluation.audits.atoms import (
    audit_g3_p3e23_top192_cycle_recovery as p3e23,
)
from chemvector.training.atoms import audit_g3_p3e7_base_expert_router as p3e7
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.atoms import (
    train_g3_p3e25_hard_latent_occupancy_micro160 as p3e25,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _atomic_torch_save,
    _autocast_context,
    module_state_sha256,
)
from chemvector.g3_structural_data import file_sha256


MODEL_TYPE = "chemvector-g3-p3e27-query-slot-occupancy-micro160"
OUTPUT = Path(f"outputs/checkpoints/{MODEL_TYPE}")
P3E26_AUDIT = Path(
    "outputs/checkpoints/chemvector-g3-p3e26-soft-coverage-occupancy-micro160/audit.json"
)
P3E26_AUDIT_SHA256 = "4668453ca81a5c9e4f75ef0ad44d8bcb0d8e58ccccfe7194e43b92d017e5dad8"
SAMPLE_KEYS_SHA256 = "06f44d30a88013643be6041fe55741fd75f6d58bcf1d875a22224972a7944501"
INPUT_TENSORS_SHA256 = "274222ec409a6f90999976cbb7898af6ca4d8dca9c5597fcea28c515353d8a9a"
EPOCHS = 100
BATCH_SIZE = 4
STEPS_PER_EPOCH = p3e25.TRAIN_IMAGES // BATCH_SIZE
STEPS = EPOCHS * STEPS_PER_EPOCH
EXPOSURE = EPOCHS * p3e25.TRAIN_IMAGES
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
SEED = 20260827


def training_batches() -> tuple[list[list[int]], dict]:
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    batches = []
    epoch_digests = []
    for _ in range(EPOCHS):
        order = torch.randperm(p3e25.TRAIN_IMAGES, generator=generator).tolist()
        epoch_digests.append(
            hashlib.sha256(torch.tensor(order).numpy().tobytes()).hexdigest()
        )
        batches.extend(
            order[start : start + BATCH_SIZE]
            for start in range(0, p3e25.TRAIN_IMAGES, BATCH_SIZE)
        )
    if len(batches) != STEPS or sum(map(len, batches)) != EXPOSURE:
        raise RuntimeError("P3-E27 training schedule differs")
    return batches, {
        "epochs": EPOCHS,
        "steps": len(batches),
        "exposure": sum(map(len, batches)),
        "epoch_order_sha256": epoch_digests,
    }


def collate(records: list[dict], device: torch.device) -> dict:
    batch = p3e23.collate(records, device)
    batch["atom_anchors"] = torch.stack(
        [record["atom_anchors"] for record in records]
    ).to(device)
    batch["full_atom_mask"] = torch.stack(
        [record["full_atom_mask"] for record in records]
    ).to(device)
    return batch


def atom_stats(record: dict, anchors: torch.Tensor) -> dict:
    expected = record["atom_anchors"][record["full_atom_mask"]].float()
    return p1a.atom_sample_statistics(anchors.float().cpu(), expected)


@torch.no_grad()
def evaluate(model, records: list[dict], old_by_key: dict, device: torch.device) -> dict:
    model.eval()
    predicted_totals = p1a._empty_atom_totals()
    known_totals = p1a._empty_atom_totals()
    count_exact = count_absolute_error = 0
    boundary_exact = 0
    newly_reachable = newly_reachable_exact = 0
    unique_candidates = True
    by_bucket = defaultdict(lambda: {"images": 0, "count_exact": 0, "atom_exact": 0})
    by_variant = defaultdict(lambda: {"images": 0, "count_exact": 0, "atom_exact": 0})
    digest = hashlib.sha256()
    for start in range(0, len(records), BATCH_SIZE):
        current = records[start : start + BATCH_SIZE]
        batch = collate(current, device)
        with _autocast_context(device, "bf16"):
            output = model(batch)
        predicted_counts = output["count_logits"].argmax(dim=-1)
        known_counts = batch["full_atom_mask"].sum(dim=1).long()
        predicted = decode_g3_atom_query_set_joint(
            output, batch["candidate_anchors"], counts=predicted_counts
        )
        known = decode_g3_atom_query_set_joint(
            output, batch["candidate_anchors"], counts=known_counts
        )
        assignments = hungarian_pointer_query_targets(
            output,
            batch["candidate_anchors"],
            batch["atom_anchors"],
            batch["full_atom_mask"],
        )
        digest.update(predicted_counts.cpu().numpy().tobytes())
        for index, record in enumerate(current):
            label = int(known_counts[index])
            predicted_count = int(predicted_counts[index])
            count_exact += int(predicted_count == label)
            count_absolute_error += abs(predicted_count - label)
            predicted_stats = atom_stats(record, predicted[index]["anchors"])
            known_stats = atom_stats(record, known[index]["anchors"])
            p1a._merge_atom_totals(predicted_totals, predicted_stats)
            p1a._merge_atom_totals(known_totals, known_stats)
            candidates = predicted[index]["candidate_indices"]
            unique_candidates &= len(candidates.unique()) == len(candidates)
            digest.update(candidates.cpu().numpy().tobytes())
            positive_queries = assignments[index][0]
            positive_mask = torch.zeros(128, dtype=torch.bool, device=device)
            positive_mask[positive_queries] = True
            existence = output["existence_logits"][index].float()
            boundary_exact += int(
                bool((existence[positive_mask].min() > existence[~positive_mask].max()).cpu())
            )
            key = str(record["sample_key"])
            old = old_by_key[key]
            requires_tail = not torch.equal(old["atom_mask"], old["full_atom_mask"])
            newly_reachable += int(requires_tail)
            newly_reachable_exact += int(
                requires_tail and predicted_stats["whole_set_exact_images"]
            )
            bucket = p3e25.count_bucket(label)
            variant = key.rsplit("_", 1)[-1]
            for group in (by_bucket[bucket], by_variant[variant]):
                group["images"] += 1
                group["count_exact"] += int(predicted_count == label)
                group["atom_exact"] += int(predicted_stats["whole_set_exact_images"])
    predicted_metrics = p1a._finalize_atom_totals(predicted_totals)
    known_metrics = p1a._finalize_atom_totals(known_totals)
    return {
        "images": len(records),
        "count_exact": count_exact,
        "count_accuracy": count_exact / len(records),
        "count_mae": count_absolute_error / len(records),
        "predicted": predicted_metrics,
        "known_count": known_metrics,
        "existence_boundary_exact": boundary_exact,
        "newly_reachable_images": newly_reachable,
        "newly_reachable_exact": newly_reachable_exact,
        "newly_reachable_exact_rate": (
            newly_reachable_exact / newly_reachable if newly_reachable else 0.0
        ),
        "unique_candidate_indices": unique_candidates,
        "by_count": dict(by_bucket),
        "by_variant": dict(by_variant),
        "prediction_digest": digest.hexdigest(),
    }


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P3-E27 output is not empty: {OUTPUT}")
    for path, expected in (
        (P3E26_AUDIT, P3E26_AUDIT_SHA256),
        (p3e23.TOP256_CACHE, p3e23.TOP256_CACHE_SHA256),
        (p3e23.OLD_CACHE_ROOT / "train.pt", p3e23.OLD_CACHE_SHA256["train"]),
        (p3e7.EXPERT_CHECKPOINT, p3e7.EXPERT_SHA256),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"P3-E27 input SHA256 differs: {path}")
    if not torch.cuda.is_available():
        raise RuntimeError("P3-E27 requires CUDA BF16")

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

    old = torch.load(
        p3e23.OLD_CACHE_ROOT / "train.pt", map_location="cpu", weights_only=False
    )["records"]
    old_by_key = {str(record["sample_key"]): record for record in old}
    top256 = torch.load(p3e23.TOP256_CACHE, map_location="cpu", weights_only=False)
    records, input_audit = p3e25.true_global192_records(top256, old)
    micro, selection_audit = p3e25.select_micro160(records, old)
    if selection_audit["sample_keys_sha256"] != SAMPLE_KEYS_SHA256:
        raise ValueError("P3-E27 Micro160 selection differs")
    if input_audit["candidate_tensors_sha256"] != INPUT_TENSORS_SHA256:
        raise ValueError("P3-E27 true-global192 input differs")

    model = p3e23.load_dynamic_model(
        p3e7.EXPERT_CHECKPOINT, p3e7.EXPERT_SHA256, device
    )
    model.requires_grad_(True)
    source_state_sha256 = module_state_sha256(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    batches, schedule = training_batches()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    audit = {
        "schema_version": "g3-p3e27-query-slot-occupancy-micro160-audit-0.1",
        "model_type": MODEL_TYPE,
        "status": "training",
        "input": input_audit,
        "selection": selection_audit,
        "schedule": schedule,
        "model_config": {
            "candidates": model.config.candidates,
            "queries": model.config.queries,
            "feature_dim": model.config.feature_dim,
            "d_model": model.config.d_model,
            "heads": model.config.heads,
            "ffn_dim": model.config.ffn_dim,
            "encoder_layers": model.config.encoder_layers,
            "decoder_layers": model.config.decoder_layers,
            "dropout": model.config.dropout,
        },
        "trainable_parameters": sum(value.numel() for value in model.parameters()),
        "loss": {
            "query_matching": "pointer-compatible Hungarian",
            "unique_pointer_training_target": False,
            "coordinate_weight": 5.0,
            "existence_boundary_weight": 1.0,
            "existence_boundary_margin": 1.0,
        },
        "lineage": {
            "p3e26_audit_sha256": P3E26_AUDIT_SHA256,
            "expert_checkpoint_sha256": p3e7.EXPERT_SHA256,
            "source_state_sha256": source_state_sha256,
            "sample_keys_sha256": SAMPLE_KEYS_SHA256,
            "input_tensors_sha256": INPUT_TENSORS_SHA256,
            "dev_read": False,
            "old_confirm_read": False,
            "fresh_dev_read": False,
            "sealed_test_read": False,
        },
    }
    _atomic_json(OUTPUT / "audit.json", audit)

    started = time.perf_counter()
    sums = defaultdict(float)
    log_path = OUTPUT / "train.jsonl"
    for step, indices in enumerate(batches, start=1):
        current = [micro[index] for index in indices]
        batch = collate(current, device)
        optimizer.zero_grad(set_to_none=True)
        model.train()
        with _autocast_context(device, "bf16"):
            output = model(batch)
            losses = g3_atom_query_set_loss(
                output,
                batch["candidate_anchors"],
                batch["atom_anchors"],
                batch["full_atom_mask"],
                query_matching="pointer",
                unique_pointers=False,
                existence_boundary_weight=1.0,
                existence_boundary_margin=1.0,
            )
        losses["loss"].backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError("P3-E27 gradient norm is not finite")
        optimizer.step()
        for name, value in losses.items():
            sums[name] += float(value.detach())
        sums["gradient_norm"] += float(gradient_norm)
        if step % 200 == 0 or step == STEPS:
            record = {
                "step": step,
                "epoch": math.ceil(step / STEPS_PER_EPOCH),
                "exposure": min(step * BATCH_SIZE, EXPOSURE),
                **{f"mean_{name}": value / step for name, value in sums.items()},
                "elapsed_seconds": time.perf_counter() - started,
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            print(json.dumps(record, separators=(",", ":")), flush=True)

    metrics = evaluate(model, micro, old_by_key, device)
    repeated = evaluate(model, micro, old_by_key, device)
    predicted = metrics["predicted"]
    checks = {
        "count_exact_min152": metrics["count_exact"] >= 152,
        "count_mae_max0_10": metrics["count_mae"] <= 0.10,
        "predicted_atom_exact_min152": predicted["whole_set_exact_images"] >= 152,
        "known_atom_exact_min152": metrics["known_count"]["whole_set_exact_images"] >= 152,
        "predicted_atom_recall_min999": (
            predicted["matched_atoms"] / predicted["target_atoms"] >= 0.999
        ),
        "existence_boundary_exact_min152": metrics["existence_boundary_exact"] >= 152,
        "newly_reachable_exact_min50pct": metrics["newly_reachable_exact_rate"] >= 0.5,
        "all_count_buckets_min90pct": all(
            value["count_exact"] / value["images"] >= 0.9
            and value["atom_exact"] / value["images"] >= 0.9
            for value in metrics["by_count"].values()
        ),
        "all_variants_min90pct": all(
            value["count_exact"] / value["images"] >= 0.9
            and value["atom_exact"] / value["images"] >= 0.9
            for value in metrics["by_variant"].values()
        ),
        "unique_candidate_indices": metrics["unique_candidate_indices"],
        "repeat_metrics": repeated == metrics,
        "fixed_steps": len(batches) == STEPS,
        "fixed_exposure": schedule["exposure"] == EXPOSURE,
    }
    decision = (
        "query_slot_occupancy_micro160_go"
        if all(checks.values())
        else "query_slot_occupancy_micro160_no_go"
    )
    final = {
        "decision": decision,
        "checks": checks,
        "metrics": metrics,
        "training_seconds": time.perf_counter() - started,
    }
    payload = {
        "checkpoint_schema_version": "g3-p3e27-query-slot-occupancy-micro160-0.1",
        "model_type": MODEL_TYPE,
        "model_config": audit["model_config"],
        "query_slot_occupancy": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": STEPS,
        "samples_seen": EXPOSURE,
        "training_images": p3e25.TRAIN_IMAGES,
        "trainable_parameter_count": audit["trainable_parameters"],
        "metrics": final,
        "lineage": audit["lineage"],
        "input": input_audit,
        "selection": selection_audit,
    }
    checkpoint = OUTPUT / f"step-{STEPS:06d}.pt"
    _atomic_torch_save(checkpoint, payload)
    final["checkpoint"] = str(checkpoint.resolve())
    final["checkpoint_sha256"] = file_sha256(checkpoint)
    audit["status"] = "completed"
    audit["decision"] = decision
    audit["checkpoint_sha256"] = final["checkpoint_sha256"]
    _atomic_json(OUTPUT / "audit.json", audit)
    _atomic_json(OUTPUT / "final_metrics.json", final)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
