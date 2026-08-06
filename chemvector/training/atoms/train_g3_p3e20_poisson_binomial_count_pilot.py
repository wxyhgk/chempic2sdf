#!/usr/bin/env python3
"""Pilot additive Poisson-binomial Count on balanced molecule splits."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from chemvector.components.atoms.objects.g3_poisson_binomial_count import (
    G3PoissonBinomialCount,
    G3PoissonBinomialCountConfig,
    g3_poisson_binomial_count_loss,
)
from chemvector.training.atoms import train_g3_p2h_atom_query_set_capacity64 as p2h
from chemvector.training.atoms import train_g3_p3e9_deepsets_count as p3e9
from chemvector.training.atoms import train_g3_p3e14_style_count_scale as p3e14
from chemvector.training.atoms import train_g3_p3e16_style_count_epoch6 as p3e16
from chemvector.training.atoms import train_g3_p3e19_bucket_balanced_count as p3e19
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _atomic_torch_save,
    _autocast_context,
)
from chemvector.g3_structural_data import file_sha256


MODEL_TYPE = "chemvector-g3-p3e20-poisson-binomial-count-pilot2560-h64"
OUTPUT = Path(f"outputs/smoke/{MODEL_TYPE}")
BASELINE_CHECKPOINT = p3e19.SOURCE_CHECKPOINT
BASELINE_SHA256 = p3e19.SOURCE_SHA256
BUCKETS = tuple(p3e19.EXPECTED_BUCKET_COUNTS)
TRAIN_MOLECULES_PER_BUCKET = 128
DEV_MOLECULES_PER_BUCKET = 64
VARIANTS_PER_MOLECULE = 4
TRAIN_IMAGES = len(BUCKETS) * TRAIN_MOLECULES_PER_BUCKET * VARIANTS_PER_MOLECULE
DEV_IMAGES = len(BUCKETS) * DEV_MOLECULES_PER_BUCKET * VARIANTS_PER_MOLECULE
EPOCHS = 30
BATCH_SIZE = 64
STEPS_PER_EPOCH = math.ceil(TRAIN_IMAGES / BATCH_SIZE)
STEPS = EPOCHS * STEPS_PER_EPOCH
EXPOSURE = EPOCHS * TRAIN_IMAGES
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
EXPECTED_COUNT_WEIGHT = 0.1
SEED = 20260820


def balanced_molecule_splits(records: list[dict]) -> tuple[dict[str, list[dict]], dict]:
    groups = defaultdict(list)
    for record in records:
        groups[p2h._molecule_key(str(record["sample_key"]))].append(record)
    by_bucket = defaultdict(list)
    for molecule, values in groups.items():
        values = sorted(values, key=lambda value: str(value["sample_key"]))
        labels = {int(value["label"]) for value in values}
        variants = {p2h._variant(str(value["sample_key"])) for value in values}
        if len(values) != VARIANTS_PER_MOLECULE or len(labels) != 1 or len(variants) != 4:
            raise ValueError(f"P3-E20 molecule group is incomplete: {molecule}")
        bucket = p3e19.count_bucket(next(iter(labels)))
        rank = hashlib.sha256(f"{SEED}:{molecule}".encode()).hexdigest()
        by_bucket[bucket].append((rank, molecule, values))
    required = TRAIN_MOLECULES_PER_BUCKET + DEV_MOLECULES_PER_BUCKET
    selected = {}
    for bucket in BUCKETS:
        ordered = sorted(by_bucket[bucket])
        if len(ordered) < required:
            raise RuntimeError(f"P3-E20 lacks molecules in bucket {bucket}")
        selected[bucket] = ordered[:required]
    train = [
        record
        for bucket in BUCKETS
        for _, _, values in selected[bucket][:TRAIN_MOLECULES_PER_BUCKET]
        for record in values
    ]
    dev = [
        record
        for bucket in BUCKETS
        for _, _, values in selected[bucket][TRAIN_MOLECULES_PER_BUCKET:]
        for record in values
    ]
    train_molecules = {
        p2h._molecule_key(str(record["sample_key"])) for record in train
    }
    dev_molecules = {
        p2h._molecule_key(str(record["sample_key"])) for record in dev
    }
    if train_molecules & dev_molecules:
        raise RuntimeError("P3-E20 train/dev molecule overlap")
    digest = hashlib.sha256()
    for split, values in (("train", train), ("dev", dev)):
        for record in values:
            digest.update(f"{split}:{record['sample_key']}\n".encode())
    return {"train": train, "dev": dev}, {
        "selection": "hash-ranked, molecule-disjoint, equal molecules per Count bucket",
        "train_images": len(train),
        "dev_images": len(dev),
        "train_molecules": len(train_molecules),
        "dev_molecules": len(dev_molecules),
        "molecule_overlap": len(train_molecules & dev_molecules),
        "train_bucket_images": dict(
            Counter(p3e19.count_bucket(int(record["label"])) for record in train)
        ),
        "dev_bucket_images": dict(
            Counter(p3e19.count_bucket(int(record["label"])) for record in dev)
        ),
        "sample_keys_sha256": digest.hexdigest(),
    }


def training_batches(record_count: int) -> tuple[list[list[int]], dict]:
    if record_count != TRAIN_IMAGES:
        raise ValueError("P3-E20 training image count differs")
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    batches = []
    epoch_digests = []
    for _ in range(EPOCHS):
        order = torch.randperm(record_count, generator=generator).tolist()
        epoch_digests.append(
            hashlib.sha256(torch.tensor(order, dtype=torch.int64).numpy().tobytes()).hexdigest()
        )
        batches.extend(
            order[start : start + BATCH_SIZE]
            for start in range(0, record_count, BATCH_SIZE)
        )
    if len(batches) != STEPS or sum(map(len, batches)) != EXPOSURE:
        raise RuntimeError("P3-E20 training schedule differs")
    return batches, {
        "epochs": EPOCHS,
        "unique_images_per_epoch": record_count,
        "steps": len(batches),
        "exposure": sum(map(len, batches)),
        "epoch_order_sha256": epoch_digests,
    }


@torch.no_grad()
def evaluate(model, records: list[dict], *, device: torch.device) -> dict:
    model.eval()
    predicted = []
    labels = []
    baselines = []
    variants = []
    expected_errors = []
    digest = hashlib.sha256()
    for start in range(0, len(records), p3e19.EVALUATION_BATCH_SIZE):
        indices = list(
            range(start, min(start + p3e19.EVALUATION_BATCH_SIZE, len(records)))
        )
        query, candidate, global_features, target = p3e9.batch(records, indices, device)
        with _autocast_context(device, "bf16"):
            output = model(query, candidate, global_features)
        current = output["count_log_probs"].argmax(dim=-1).long().cpu()
        predicted.extend(current.tolist())
        target_cpu = target.cpu()
        labels.extend(target_cpu.tolist())
        expected_errors.extend(
            (output["expected_count"].cpu() - target_cpu.float()).abs().tolist()
        )
        baselines.extend(int(records[index]["baseline_count"]) for index in indices)
        variants.extend(p2h._variant(str(records[index]["sample_key"])) for index in indices)
        digest.update(current.numpy().tobytes())
    return {
        **p3e16.bucketed_count_metrics(predicted, labels, baselines, variants),
        "expected_count_mae": sum(expected_errors) / len(expected_errors),
        "prediction_digest": digest.hexdigest(),
    }


def capability_gate(train: dict, dev: dict, baseline: dict) -> tuple[str, dict[str, bool]]:
    checks = {
        "train_accuracy_min50": train["count_accuracy"] >= 0.50,
        "dev_accuracy_gain5pp": dev["count_accuracy"] >= baseline["count_accuracy"] + 0.05,
        "dev_mae_gain0_25": dev["count_mae"] <= baseline["count_mae"] - 0.25,
        "dev_61_80_gain5pp": (
            dev["by_count"]["61-80"]["count_accuracy"]
            >= baseline["by_count"]["61-80"]["count_accuracy"] + 0.05
        ),
        "dev_81_100_gain5pp": (
            dev["by_count"]["81-100"]["count_accuracy"]
            >= baseline["by_count"]["81-100"]["count_accuracy"] + 0.05
        ),
        "dev_1_20_regression_max5pp": (
            dev["by_count"]["1-20"]["count_accuracy"]
            >= baseline["by_count"]["1-20"]["count_accuracy"] - 0.05
        ),
        "dev_21_40_regression_max5pp": (
            dev["by_count"]["21-40"]["count_accuracy"]
            >= baseline["by_count"]["21-40"]["count_accuracy"] - 0.05
        ),
    }
    return (
        "poisson_binomial_count_pilot_go"
        if all(checks.values())
        else "poisson_binomial_count_pilot_no_go",
        checks,
    )


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P3-E20 output is not empty: {OUTPUT}")
    if not torch.cuda.is_available():
        raise RuntimeError("P3-E20 requires CUDA BF16")
    if file_sha256(BASELINE_CHECKPOINT) != BASELINE_SHA256:
        raise ValueError("P3-E20 baseline checkpoint SHA256 differs")
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

    style_records, evidence_audit = p3e14.load_style_evidence()
    splits, split_audit = balanced_molecule_splits(style_records)
    batches, schedule_audit = training_batches(len(splits["train"]))
    baseline_model = p3e14.load_count_model(BASELINE_CHECKPOINT, device)
    baseline_dev = p3e19.evaluate_bucketed(
        baseline_model, splits["dev"], device=device
    )
    del baseline_model
    config = G3PoissonBinomialCountConfig(hidden_dim=64)
    model = G3PoissonBinomialCount(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    audit = {
        "schema_version": "g3-p3e20-poisson-binomial-count-pilot-audit-0.1",
        "model_type": MODEL_TYPE,
        "fixed_protocol": {
            "representation": "candidate occupancy -> Poisson-binomial Count distribution",
            "supervision": "whole-image Count only; no fixed candidate identity labels",
            "checkpoint_selection": "fixed_final_only",
            "fresh_dev_read": False,
            "sealed_test_read": False,
        },
        "model_config": asdict(config),
        "trainable_parameters": sum(value.numel() for value in model.parameters()),
        "split": split_audit,
        "schedule": schedule_audit,
        "baseline_dev": baseline_dev,
        "lineage": {
            "evidence": evidence_audit,
            "baseline_checkpoint": str(BASELINE_CHECKPOINT.resolve()),
            "baseline_sha256": BASELINE_SHA256,
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    _atomic_json(OUTPUT / "audit.json", audit)

    started = time.perf_counter()
    loss_sums = defaultdict(float)
    exposure = 0
    model.train()
    log_path = OUTPUT / "train.jsonl"
    for step, indices in enumerate(batches, start=1):
        query, candidate, global_features, target = p3e9.batch(
            splits["train"], indices, device
        )
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, "bf16"):
            output = model(query, candidate, global_features)
            losses = g3_poisson_binomial_count_loss(
                output, target, expected_count_weight=EXPECTED_COUNT_WEIGHT
            )
        losses["loss"].backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        for name, value in losses.items():
            loss_sums[name] += float(value.detach())
        loss_sums["gradient_norm"] += float(gradient_norm)
        exposure += len(indices)
        if step % 200 == 0 or step == STEPS:
            current = {
                "step": step,
                "epoch": math.ceil(step / STEPS_PER_EPOCH),
                "exposure": exposure,
                **{f"mean_{name}": value / step for name, value in loss_sums.items()},
                "elapsed_seconds": time.perf_counter() - started,
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(current, separators=(",", ":")) + "\n")
            print(json.dumps(current, separators=(",", ":")), flush=True)

    metrics = {
        name: evaluate(model, records, device=device) for name, records in splits.items()
    }
    decision, checks = capability_gate(metrics["train"], metrics["dev"], baseline_dev)
    protocol_checks = {
        "fixed_steps": len(batches) == STEPS,
        "fixed_exposure": exposure == EXPOSURE,
        "molecule_disjoint": split_audit["molecule_overlap"] == 0,
    }
    if not all(protocol_checks.values()):
        decision = "poisson_binomial_count_pilot_protocol_failed"
    final = {
        "decision": decision,
        "checks": checks,
        "protocol_checks": protocol_checks,
        "baseline_dev": baseline_dev,
        "metrics": metrics,
        "training_seconds": time.perf_counter() - started,
    }
    payload = {
        "checkpoint_schema_version": "g3-p3e20-poisson-binomial-count-pilot-0.1",
        "model_type": MODEL_TYPE,
        "model_config": asdict(config),
        "poisson_binomial_count": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": STEPS,
        "samples_seen": exposure,
        "training_images": TRAIN_IMAGES,
        "trainable_parameter_count": audit["trainable_parameters"],
        "metrics": final,
        "lineage": audit["lineage"],
    }
    checkpoint = OUTPUT / f"step-{STEPS:06d}.pt"
    _atomic_torch_save(checkpoint, payload)
    _atomic_torch_save(OUTPUT / "last.pt", payload)
    final["checkpoint"] = str(checkpoint.resolve())
    final["checkpoint_sha256"] = file_sha256(checkpoint)
    _atomic_json(OUTPUT / "final_metrics.json", final)
    _atomic_json(
        OUTPUT / "status.json",
        {
            "status": "completed",
            "decision": decision,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": final["checkpoint_sha256"],
        },
    )
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
