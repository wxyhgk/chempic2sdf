#!/usr/bin/env python3
"""Train hard latent candidate occupancy on a balanced top192 capacity set."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch.nn import functional as F

from chemvector.components.atoms.objects.g3_deepsets_count import (
    g3_deepsets_count_features,
)
from chemvector.components.atoms.objects.g3_latent_matching import (
    g3_gt_candidate_compatibility,
)
from chemvector.components.atoms.objects.g3_poisson_binomial_count import (
    G3PoissonBinomialCount,
    G3PoissonBinomialCountConfig,
)
from chemvector.evaluation.audits.atoms import (
    audit_g3_p3e23_top192_cycle_recovery as p3e23,
)
from chemvector.evaluation.audits.atoms import (
    audit_g3_p3e24_top192_input_protocol_ab as p3e24,
)
from chemvector.training.atoms import audit_g3_p3e7_base_expert_router as p3e7
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.atoms import train_g3_p1x_multiscale_peak_selector_capacity as p1x
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _atomic_torch_save,
    _autocast_context,
    module_state_sha256,
)
from chemvector.g3_structural_data import file_sha256


MODEL_TYPE = "chemvector-g3-p3e25-hard-latent-occupancy-micro160"
OUTPUT = Path(f"outputs/checkpoints/{MODEL_TYPE}")
P3E24_AUDIT = p3e24.OUTPUT / "audit.json"
P3E24_AUDIT_SHA256 = "f738cb949b78da99b49dc9791cdf5adbe538aaae967f78e61504ce8473c45e19"
MOLECULES_PER_BUCKET = 8
VARIANTS_PER_MOLECULE = 4
BUCKETS = ((1, 20), (21, 40), (41, 60), (61, 80), (81, 100))
TRAIN_IMAGES = len(BUCKETS) * MOLECULES_PER_BUCKET * VARIANTS_PER_MOLECULE
EPOCHS = 100
BATCH_SIZE = 16
STEPS_PER_EPOCH = math.ceil(TRAIN_IMAGES / BATCH_SIZE)
STEPS = EPOCHS * STEPS_PER_EPOCH
EXPOSURE = EPOCHS * TRAIN_IMAGES
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.0
MAX_GRAD_NORM = 5.0
COUNT_CLASSES = 129
SEED = 20260825


def count_bucket(count: int) -> str:
    for lower, upper in BUCKETS:
        if lower <= count <= upper:
            return f"{lower}-{upper}"
    raise ValueError(f"P3-E25 Count is outside fixed buckets: {count}")


def true_global192_records(top256: dict, old: list[dict]) -> tuple[list[dict], dict]:
    old_by_key = {str(record["sample_key"]): record for record in old}
    records = []
    digest = hashlib.sha256()
    prefix_raw_equal = True
    for key in sorted(top256["records"]["train"]):
        record = p3e23.format_top192_record(key, top256["records"]["train"][key])
        record["normalized_scores"] = p1x._scale_score_features(record["raw_scores"])
        record["scale_ranks"] = (
            torch.arange(p3e23.CANDIDATES).float() / (p3e23.CANDIDATES - 1)
        )
        prefix_raw_equal &= all(
            torch.equal(record[name][:128], old_by_key[key][name])
            for name in (
                "candidate_ids",
                "candidate_features",
                "candidate_anchors",
                "raw_scores",
            )
        )
        for name in (
            "candidate_ids",
            "candidate_features",
            "candidate_anchors",
            "raw_scores",
            "normalized_scores",
            "scale_ranks",
        ):
            digest.update(record[name].numpy().tobytes())
        records.append(record)
    return records, {
        "protocol": "true_global192",
        "normalization": "mean/std of logits(raw_scores[:192])",
        "scale_rank": "rank/191",
        "prefix128_raw_geometry_equal": prefix_raw_equal,
        "candidate_tensors_sha256": digest.hexdigest(),
    }


def select_micro160(
    records: list[dict], old: list[dict]
) -> tuple[list[dict], dict]:
    old_by_key = {str(record["sample_key"]): record for record in old}
    groups = defaultdict(list)
    for record in records:
        molecule = str(record["sample_key"]).rsplit("_", 1)[0]
        groups[molecule].append(record)
    eligible = defaultdict(list)
    for molecule, values in groups.items():
        values = sorted(values, key=lambda record: str(record["sample_key"]))
        if len(values) != VARIANTS_PER_MOLECULE or not all(
            torch.equal(record["atom_mask"], record["full_atom_mask"])
            for record in values
        ):
            continue
        counts = {int(record["full_atom_mask"].sum()) for record in values}
        if len(counts) != 1:
            raise ValueError(f"P3-E25 molecule Count differs: {molecule}")
        bucket = count_bucket(next(iter(counts)))
        tail_required = any(
            not torch.equal(
                old_by_key[str(record["sample_key"])]["atom_mask"],
                old_by_key[str(record["sample_key"])]["full_atom_mask"],
            )
            for record in values
        )
        rank = hashlib.sha256(f"{SEED}:{molecule}".encode()).hexdigest()
        eligible[bucket].append((not tail_required, rank, molecule, values))

    selected = []
    selection = {}
    digest = hashlib.sha256()
    for lower, upper in BUCKETS:
        bucket = f"{lower}-{upper}"
        ordered = sorted(eligible[bucket])
        current = ordered[:MOLECULES_PER_BUCKET]
        if len(current) != MOLECULES_PER_BUCKET:
            raise RuntimeError(f"P3-E25 lacks Micro molecules in {bucket}")
        values = [record for _, _, _, records_group in current for record in records_group]
        selected.extend(values)
        selection[bucket] = {
            "molecules": [molecule for _, _, molecule, _ in current],
            "tail_required_molecules": sum(not ordinary for ordinary, *_ in current),
            "images": len(values),
        }
        for record in values:
            digest.update(f"{bucket}:{record['sample_key']}\n".encode())
    if len(selected) != TRAIN_IMAGES:
        raise RuntimeError("P3-E25 Micro image count differs")
    return selected, {
        "selection": "five Count buckets; tail-required first; SHA-ranked; four variants",
        "buckets": selection,
        "images": len(selected),
        "molecules": len(selected) // VARIANTS_PER_MOLECULE,
        "sample_keys_sha256": digest.hexdigest(),
        "sample_keys": [str(record["sample_key"]) for record in selected],
    }


def training_batches() -> tuple[list[list[int]], dict]:
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    batches = []
    epoch_digests = []
    for _ in range(EPOCHS):
        order = torch.randperm(TRAIN_IMAGES, generator=generator).tolist()
        epoch_digests.append(
            hashlib.sha256(torch.tensor(order).numpy().tobytes()).hexdigest()
        )
        batches.extend(
            order[start : start + BATCH_SIZE]
            for start in range(0, TRAIN_IMAGES, BATCH_SIZE)
        )
    if len(batches) != STEPS or sum(map(len, batches)) != EXPOSURE:
        raise RuntimeError("P3-E25 training schedule differs")
    return batches, {
        "epochs": EPOCHS,
        "steps": len(batches),
        "exposure": sum(map(len, batches)),
        "epoch_order_sha256": epoch_digests,
    }


def batch_evidence(records: list[dict], indices: list[int], device: torch.device):
    current = [records[index] for index in indices]
    return current, (
        torch.stack([record["query_features"] for record in current]).to(device),
        torch.stack([record["count_candidate_features"] for record in current]).to(device),
        torch.stack([record["global_features"] for record in current]).to(device),
    )


def bounded_count_log_probs(output: dict[str, torch.Tensor]) -> torch.Tensor:
    log_probs = output["count_log_probs"][:, :COUNT_CLASSES].float()
    return log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)


def hard_presence_targets(
    presence_logits: torch.Tensor, records: list[dict]
) -> torch.Tensor:
    targets = torch.zeros_like(presence_logits, dtype=torch.bool)
    for index, record in enumerate(records):
        expected = record["atom_anchors"][record["full_atom_mask"]].float()
        candidates = record["candidate_anchors"].float()
        compatibility = g3_gt_candidate_compatibility(expected, candidates).cpu()
        score = presence_logits[index].detach().double().cpu()
        cost = -score.unsqueeze(0).expand(len(expected), -1).clone()
        cost[~compatibility] = 1_000_000_000.0
        rows, columns = linear_sum_assignment(cost.numpy())
        if len(rows) != len(expected) or not bool(
            compatibility[rows, columns].all()
        ):
            raise RuntimeError("P3-E25 hard latent assignment is infeasible")
        targets[index, torch.as_tensor(columns, device=targets.device)] = True
    return targets


def hard_occupancy_loss(
    output: dict[str, torch.Tensor], records: list[dict]
) -> dict[str, torch.Tensor]:
    logits = output["presence_logits"].float()
    targets = hard_presence_targets(logits, records)
    positive = []
    negative = []
    for index in range(len(records)):
        positive.append(-F.logsigmoid(logits[index, targets[index]]).mean())
        negative.append(-F.logsigmoid(-logits[index, ~targets[index]]).mean())
    positive_loss = torch.stack(positive).mean()
    negative_loss = torch.stack(negative).mean()
    balanced_bce = 0.5 * (positive_loss + negative_loss)
    labels = torch.tensor(
        [int(record["full_atom_mask"].sum()) for record in records],
        device=logits.device,
    )
    count_nll = F.nll_loss(bounded_count_log_probs(output), labels)
    loss = balanced_bce + 0.1 * count_nll
    return {
        "loss": loss,
        "balanced_bce": balanced_bce,
        "positive_bce": positive_loss,
        "negative_bce": negative_loss,
        "count_nll": count_nll,
    }


def stable_topk_indices(scores: torch.Tensor, count: int) -> torch.Tensor:
    if scores.ndim != 1 or not 0 <= count <= 128:
        raise ValueError("P3-E25 stable top-K Count is outside 0--128")
    return torch.argsort(scores, descending=True, stable=True)[:count]


def atom_stats_from_indices(record: dict, indices: torch.Tensor) -> dict:
    predicted = record["candidate_anchors"][indices].float()
    expected = record["atom_anchors"][record["full_atom_mask"]].float()
    return p1a.atom_sample_statistics(predicted, expected)


@torch.no_grad()
def evaluate(model, records: list[dict], old_by_key: dict, device: torch.device) -> dict:
    model.eval()
    predicted_totals = p1a._empty_atom_totals()
    known_totals = p1a._empty_atom_totals()
    count_exact = 0
    count_absolute_error = 0
    gaps = []
    boundary_gaps = []
    newly_reachable = newly_reachable_exact = 0
    correct_tail_hits = 0
    by_bucket = defaultdict(lambda: {"images": 0, "count_exact": 0, "atom_exact": 0})
    by_variant = defaultdict(lambda: {"images": 0, "count_exact": 0, "atom_exact": 0})
    probability_checks = []
    digest = hashlib.sha256()
    for start in range(0, len(records), BATCH_SIZE):
        indices = list(range(start, min(start + BATCH_SIZE, len(records))))
        current, evidence = batch_evidence(records, indices, device)
        with _autocast_context(device, "bf16"):
            output = model(*evidence)
        count_log_probs = bounded_count_log_probs(output)
        probability_checks.append(
            bool(torch.isfinite(count_log_probs).all())
            and bool(torch.allclose(
                count_log_probs.logsumexp(dim=-1),
                torch.zeros(len(current), device=device),
                atol=1e-5,
            ))
        )
        predicted_counts = count_log_probs.argmax(dim=-1)
        probabilities = output["presence_logits"].float().sigmoid()
        latent = hard_presence_targets(output["presence_logits"], current)
        digest.update(predicted_counts.cpu().numpy().tobytes())
        digest.update(probabilities.cpu().numpy().tobytes())
        for index, record in enumerate(current):
            label = int(record["full_atom_mask"].sum())
            predicted_count = int(predicted_counts[index])
            count_exact += int(predicted_count == label)
            count_absolute_error += abs(predicted_count - label)
            predicted_indices = stable_topk_indices(
                probabilities[index], predicted_count
            ).cpu()
            known_indices = stable_topk_indices(probabilities[index], label).cpu()
            predicted_stats = atom_stats_from_indices(record, predicted_indices)
            known_stats = atom_stats_from_indices(record, known_indices)
            p1a._merge_atom_totals(predicted_totals, predicted_stats)
            p1a._merge_atom_totals(known_totals, known_stats)
            positive_min = float(probabilities[index, latent[index]].min())
            negative_max = float(probabilities[index, ~latent[index]].max())
            gaps.append(positive_min - negative_max)
            ordered = probabilities[index].sort(descending=True, stable=True).values
            boundary_gaps.append(float(ordered[label - 1] - ordered[label]))
            key = str(record["sample_key"])
            old = old_by_key[key]
            requires_tail = not torch.equal(old["atom_mask"], old["full_atom_mask"])
            newly_reachable += int(requires_tail)
            newly_reachable_exact += int(
                requires_tail and predicted_stats["whole_set_exact_images"]
            )
            tail = predicted_indices[predicted_indices >= 128]
            if len(tail):
                expected = record["atom_anchors"][record["full_atom_mask"]].float()
                anchors = record["candidate_anchors"][tail].float()
                correct_tail_hits += int(
                    (torch.cdist(anchors, expected) <= 6.0 / 768.0).any(dim=1).sum()
                )
            bucket = count_bucket(label)
            variant = str(record["sample_key"]).rsplit("_", 1)[-1]
            for group in (by_bucket[bucket], by_variant[variant]):
                group["images"] += 1
                group["count_exact"] += int(predicted_count == label)
                group["atom_exact"] += int(predicted_stats["whole_set_exact_images"])
    predicted = p1a._finalize_atom_totals(predicted_totals)
    known = p1a._finalize_atom_totals(known_totals)
    return {
        "images": len(records),
        "count_exact": count_exact,
        "count_accuracy": count_exact / len(records),
        "count_mae": count_absolute_error / len(records),
        "predicted": predicted,
        "known_count": known,
        "presence_gap_min": min(gaps),
        "presence_gap_mean": sum(gaps) / len(gaps),
        "presence_gap_ge0_8_images": sum(value >= 0.8 for value in gaps),
        "known_topk_boundary_gap_min": min(boundary_gaps),
        "known_topk_boundary_gap_mean": sum(boundary_gaps) / len(boundary_gaps),
        "newly_reachable_images": newly_reachable,
        "newly_reachable_exact": newly_reachable_exact,
        "newly_reachable_exact_rate": (
            newly_reachable_exact / newly_reachable if newly_reachable else 0.0
        ),
        "correct_tail_hits": correct_tail_hits,
        "count_distribution_finite_normalized": all(probability_checks),
        "by_count": dict(by_bucket),
        "by_variant": dict(by_variant),
        "prediction_digest": digest.hexdigest(),
    }


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P3-E25 output is not empty: {OUTPUT}")
    for path, expected in (
        (P3E24_AUDIT, P3E24_AUDIT_SHA256),
        (p3e23.TOP256_CACHE, p3e23.TOP256_CACHE_SHA256),
        (p3e23.OLD_CACHE_ROOT / "train.pt", p3e23.OLD_CACHE_SHA256["train"]),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"P3-E25 input SHA256 differs: {path}")
    if not torch.cuda.is_available():
        raise RuntimeError("P3-E25 requires CUDA BF16")

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
    records, input_audit = true_global192_records(top256, old)
    micro, selection_audit = select_micro160(records, old)

    base_model = p3e23.load_dynamic_model(p3e7.BASE_CHECKPOINT, p3e7.BASE_SHA256, device)
    expert_model = p3e23.load_dynamic_model(p3e7.EXPERT_CHECKPOINT, p3e7.EXPERT_SHA256, device)
    evidence_by_split, compatibility = p3e24.build_query_evidence(
        {"micro160": micro}, base_model, expert_model, device
    )
    micro = evidence_by_split["micro160"]
    frozen_base_sha = module_state_sha256(base_model)
    frozen_expert_sha = module_state_sha256(expert_model)
    del base_model, expert_model

    config = G3PoissonBinomialCountConfig(hidden_dim=64)
    model = G3PoissonBinomialCount(config).to(device)
    with torch.no_grad():
        model.occupancy[-1].weight.zero_()
        model.occupancy[-1].bias.fill_(-1.0)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    batches, schedule_audit = training_batches()
    trainable_parameters = sum(value.numel() for value in model.parameters())

    OUTPUT.mkdir(parents=True, exist_ok=False)
    evidence_digest = hashlib.sha256()
    for record in micro:
        evidence_digest.update(str(record["sample_key"]).encode())
        for name in ("query_features", "count_candidate_features", "global_features"):
            evidence_digest.update(record[name].numpy().tobytes())
    audit = {
        "schema_version": "g3-p3e25-hard-latent-occupancy-micro160-audit-0.1",
        "model_type": MODEL_TYPE,
        "status": "training",
        "input": input_audit,
        "selection": selection_audit,
        "schedule": schedule_audit,
        "model_config": asdict(config),
        "trainable_parameters": trainable_parameters,
        "loss": {
            "balanced_bce_weight": 1.0,
            "count_nll_weight": 0.1,
            "dynamic_one_to_one": True,
        },
        "lineage": {
            "p3e24_audit_sha256": P3E24_AUDIT_SHA256,
            "top256_cache_sha256": p3e23.TOP256_CACHE_SHA256,
            "base_checkpoint_sha256": p3e7.BASE_SHA256,
            "expert_checkpoint_sha256": p3e7.EXPERT_SHA256,
            "frozen_base_state_sha256": frozen_base_sha,
            "frozen_expert_state_sha256": frozen_expert_sha,
            "evidence_sha256": evidence_digest.hexdigest(),
            "compatibility": compatibility,
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
        current, evidence = batch_evidence(micro, indices, device)
        optimizer.zero_grad(set_to_none=True)
        model.train()
        with _autocast_context(device, "bf16"):
            output = model(*evidence)
            losses = hard_occupancy_loss(output, current)
        losses["loss"].backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError("P3-E25 gradient norm is not finite")
        optimizer.step()
        for name, value in losses.items():
            sums[name] += float(value.detach())
        sums["gradient_norm"] += float(gradient_norm)
        if step % 50 == 0 or step == STEPS:
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
        "presence_gap_ge0_8_min152": metrics["presence_gap_ge0_8_images"] >= 152,
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
        "count_distribution_finite_normalized": (
            metrics["count_distribution_finite_normalized"]
        ),
        "repeat_metrics": repeated == metrics,
        "fixed_steps": len(batches) == STEPS,
        "fixed_exposure": schedule_audit["exposure"] == EXPOSURE,
    }
    decision = (
        "hard_latent_occupancy_micro160_go"
        if all(checks.values())
        else "hard_latent_occupancy_micro160_no_go"
    )
    final = {
        "decision": decision,
        "checks": checks,
        "metrics": metrics,
        "training_seconds": time.perf_counter() - started,
    }
    payload = {
        "checkpoint_schema_version": "g3-p3e25-hard-latent-occupancy-micro160-0.1",
        "model_type": MODEL_TYPE,
        "model_config": asdict(config),
        "hard_latent_occupancy": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": STEPS,
        "samples_seen": EXPOSURE,
        "training_images": TRAIN_IMAGES,
        "trainable_parameter_count": trainable_parameters,
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
