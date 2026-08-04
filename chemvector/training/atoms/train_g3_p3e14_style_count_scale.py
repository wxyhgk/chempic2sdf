#!/usr/bin/env python3
"""Train matched-exposure P3-E14 DeepSets Count on style90,932."""

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

from chemvector.components.atoms.objects.g3_deepsets_count import (
    G3DeepSetsCount,
    G3DeepSetsCountConfig,
)
from chemvector.training.atoms import audit_g3_p3e0_latent_solver as p3e0
from chemvector.training.atoms import audit_g3_p3e7_base_expert_router as p3e7
from chemvector.training.atoms import (
    build_g3_p3e14_style_count_evidence as evidence_protocol,
)
from chemvector.training.atoms import (
    train_g3_p2e_stride1_top128_selector_train5k as p2e,
)
from chemvector.training.atoms import train_g3_p3e9_deepsets_count as p3e9
from chemvector.training.atoms import train_g3_p3e10_count_scale_ab as p3e10
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _atomic_torch_save,
    _autocast_context,
)
from chemvector.g3_structural_data import file_sha256


MODEL_TYPE = "chemvector-g3-p3e14-deepsets-count-style90932-h64"
OUTPUT = Path(f"outputs/smoke/{MODEL_TYPE}")
EVIDENCE_INDEX = evidence_protocol.OUTPUT / "index.json"
EVIDENCE_INDEX_SHA256 = "14444a819fedf62130452de0aa85a6d62b061f644deb0dda4f46978a1c68ebb9"
COARSE_CHECKPOINT = Path(
    "outputs/smoke/chemvector-g3-p3e10-deepsets-count-expanded3316-h64/"
    "step-001560.pt"
)
COARSE_SHA256 = "9f734bf72d1f0c432dd2d7da14212bc94b013caf12849f30c752801a1a5bb3b7"
TRAIN_IMAGES = 90_932
EXPECTED_EXPOSURE = p3e10.EXPECTED_EXPOSURE
EXPECTED_STEPS = p3e10.EXPECTED_STEPS
FULL_BATCHES = 1_200
SHORT_BATCHES = 360
FULL_BATCH_SIZE = 64
SHORT_BATCH_SIZE = 63
REPEATED_EXPOSURE = EXPECTED_EXPOSURE - TRAIN_IMAGES
LEARNING_RATE = p3e10.LEARNING_RATE
WEIGHT_DECAY = p3e10.WEIGHT_DECAY
MAX_GRAD_NORM = p3e10.MAX_GRAD_NORM
SEED = p3e10.SEED


def load_style_evidence() -> tuple[list[dict], dict]:
    if file_sha256(EVIDENCE_INDEX) != EVIDENCE_INDEX_SHA256:
        raise ValueError("P3-E14 evidence index SHA256 differs")
    index = json.loads(EVIDENCE_INDEX.read_text(encoding="utf-8"))
    if index.get("status") != "passed" or index.get("samples") != TRAIN_IMAGES:
        raise ValueError("P3-E14 evidence index did not pass")
    records = []
    file_digest = hashlib.sha256()
    for shard in index["shards"]:
        path = Path(shard["path"])
        actual_sha = file_sha256(path)
        if actual_sha != shard["sha256"]:
            raise ValueError(f"P3-E14 evidence shard SHA differs: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        current = payload.get("records")
        if (
            payload.get("schema_version") != "g3-p3e14-count-evidence-shard-0.1"
            or payload.get("candidate_batch_size") != 24
            or payload.get("manifest_sha256") != evidence_protocol.MANIFEST_SHA256
            or not isinstance(current, list)
            or len(current) != shard["samples"]
            or evidence_protocol.sample_keys_sha256(current)
            != shard["sample_keys_sha256"]
        ):
            raise ValueError(f"P3-E14 evidence shard contract differs: {path}")
        records.extend(current)
        file_digest.update(f"{actual_sha}\n".encode())
    keys = [str(record["sample_key"]) for record in records]
    if len(records) != TRAIN_IMAGES or len(set(keys)) != TRAIN_IMAGES:
        raise RuntimeError("P3-E14 evidence sample identity differs")
    for record in records:
        if (
            record["query"].shape != (128, 8)
            or record["candidate"].shape != (128, 41)
            or record["global"].shape != (256,)
            or not 0 <= int(record["label"]) <= 128
        ):
            raise ValueError("P3-E14 evidence tensor contract differs")
    return records, {
        "index": str(EVIDENCE_INDEX.resolve()),
        "index_sha256": EVIDENCE_INDEX_SHA256,
        "shard_sha256_sequence": file_digest.hexdigest(),
        "sample_keys_sha256": evidence_protocol.sample_keys_sha256(records),
        "samples": len(records),
        "molecules": 22_733,
        "candidate_batch_size": 24,
    }


def matched_exposure_batches(record_count: int) -> tuple[list[list[int]], dict]:
    if record_count != TRAIN_IMAGES:
        raise ValueError("P3-E14 matched exposure record count differs")
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    unique_order = torch.randperm(record_count, generator=generator).tolist()
    repeat_order = torch.randperm(record_count, generator=generator).tolist()
    order = unique_order + repeat_order[:REPEATED_EXPOSURE]
    sizes = [FULL_BATCH_SIZE] * FULL_BATCHES + [SHORT_BATCH_SIZE] * SHORT_BATCHES
    if len(sizes) != EXPECTED_STEPS or sum(sizes) != EXPECTED_EXPOSURE:
        raise RuntimeError("P3-E14 matched batch schedule differs")
    batches = []
    start = 0
    for size in sizes:
        batches.append(order[start : start + size])
        start += size
    if start != len(order) or len(set(unique_order)) != TRAIN_IMAGES:
        raise RuntimeError("P3-E14 matched exposure order differs")
    digest = hashlib.sha256(torch.tensor(order, dtype=torch.int64).numpy().tobytes())
    return batches, {
        "unique_images_seen": len(set(order)),
        "repeated_exposure": REPEATED_EXPOSURE,
        "exposure": len(order),
        "steps": len(batches),
        "full_batches": FULL_BATCHES,
        "short_batches": SHORT_BATCHES,
        "order_sha256": digest.hexdigest(),
    }


def load_count_model(path: Path, device: torch.device) -> G3DeepSetsCount:
    source = torch.load(path, map_location=device, weights_only=False)
    model = G3DeepSetsCount(G3DeepSetsCountConfig(**source["model_config"])).to(device)
    model.load_state_dict(source["deepsets_count"], strict=True)
    model.eval()
    return model


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P3-E14 output is not empty: {OUTPUT}")
    for path, expected in (
        (p3e7.BASE_CHECKPOINT, p3e7.BASE_SHA256),
        (p3e7.EXPERT_CHECKPOINT, p3e7.EXPERT_SHA256),
        (COARSE_CHECKPOINT, COARSE_SHA256),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"P3-E14 checkpoint SHA differs: {path}")
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

    style_records, evidence_audit = load_style_evidence()
    cache, cache_audit = p2e.load_cache()
    fixed_splits, selection_audit = p3e10.expanded_splits(cache)
    fixed_splits = {
        name: fixed_splits[name]
        for name in ("dev256", "confirm256", "historical_route100")
    }
    base_model = p3e0.load_model(p3e7.BASE_CHECKPOINT, device)
    expert_model = p3e0.load_model(p3e7.EXPERT_CHECKPOINT, device)
    eval_evidence = p3e9.collect_evidence(
        (base_model, expert_model), fixed_splits, device=device
    )
    del base_model, expert_model

    config = G3DeepSetsCountConfig(hidden_dim=64)
    model = G3DeepSetsCount(config).to(device)
    trainable_parameters = sum(value.numel() for value in model.parameters())
    baseline = load_count_model(COARSE_CHECKPOINT, device)
    baseline_metrics = {
        "style90932": p3e9.evaluate(baseline, style_records, device=device),
        **{
            name: p3e9.evaluate(baseline, records, device=device)
            for name, records in eval_evidence.items()
        },
    }
    del baseline
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    batches, exposure_audit = matched_exposure_batches(len(style_records))
    audit = {
        "schema_version": "g3-p3e14-style-count-scale-audit-0.1",
        "model_type": MODEL_TYPE,
        "fixed_protocol": {
            "architecture": "P3-E10 DeepSets h64 unchanged",
            "checkpoint_selection": "fixed_final_only",
            "expected_steps": EXPECTED_STEPS,
            "expected_exposure": EXPECTED_EXPOSURE,
            "unique_style_images": TRAIN_IMAGES,
            "route100_training_or_selection": False,
            "gate": {
                "dev_count_accuracy_min": 0.5,
                "confirm_count_accuracy_min": 0.5,
                "confirm_count_mae_max": 1.0,
                "route100_count_accuracy_min": 0.4,
            },
        },
        "model_config": asdict(config),
        "trainable_parameters": trainable_parameters,
        "exposure": exposure_audit,
        "baseline_metrics": baseline_metrics,
        "lineage": {
            "evidence": evidence_audit,
            "base_checkpoint": str(p3e7.BASE_CHECKPOINT.resolve()),
            "base_sha256": p3e7.BASE_SHA256,
            "expert_checkpoint": str(p3e7.EXPERT_CHECKPOINT.resolve()),
            "expert_sha256": p3e7.EXPERT_SHA256,
            "coarse_checkpoint": str(COARSE_CHECKPOINT.resolve()),
            "coarse_sha256": COARSE_SHA256,
            "candidate_cache_sha256": p2e.CACHE_SHA256,
            "candidate_cache_tensor_sha256": cache_audit["candidate_protocol"][
                "candidate_cache_sha256"
            ],
            "fixed_eval_selection": selection_audit,
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    _atomic_json(OUTPUT / "audit.json", audit)

    log_path = OUTPUT / "train.jsonl"
    started = time.perf_counter()
    exposure = 0
    loss_sum = gradient_sum = clip_sum = 0.0
    model.train()
    for step, indices in enumerate(batches, start=1):
        query, candidate, global_features, target = p3e9.batch(
            style_records, indices, device
        )
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, "bf16"):
            loss = F.cross_entropy(model(query, candidate, global_features), target)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        gradient_float = float(gradient_norm)
        loss_sum += float(loss.detach())
        gradient_sum += gradient_float
        clip_sum += min(1.0, MAX_GRAD_NORM / max(gradient_float, 1e-12))
        exposure += len(indices)
        if step % 100 == 0 or step == EXPECTED_STEPS:
            log = {
                "step": step,
                "exposure": exposure,
                "mean_loss": loss_sum / step,
                "mean_gradient_norm": gradient_sum / step,
                "mean_clip_scale": clip_sum / step,
                "elapsed_seconds": time.perf_counter() - started,
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(log, separators=(",", ":")) + "\n")
            print(json.dumps(log, separators=(",", ":")), flush=True)

    evaluation_sets = {"style90932": style_records, **eval_evidence}
    metrics = {
        name: p3e9.evaluate(model, records, device=device)
        for name, records in evaluation_sets.items()
    }
    repeated = {
        name: p3e9.evaluate(model, records, device=device)
        for name, records in evaluation_sets.items()
    }
    checks = {
        "dev_count_accuracy_min50": metrics["dev256"]["count_accuracy"] >= 0.5,
        "confirm_count_accuracy_min50": metrics["confirm256"]["count_accuracy"] >= 0.5,
        "confirm_count_mae_max1": metrics["confirm256"]["count_mae"] <= 1.0,
        "route100_count_accuracy_min40": (
            metrics["historical_route100"]["count_accuracy"] >= 0.4
        ),
    }
    protocol_checks = {
        "fixed_steps": len(batches) == EXPECTED_STEPS,
        "fixed_exposure": exposure == EXPECTED_EXPOSURE,
        "all_unique_images_seen": exposure_audit["unique_images_seen"] == TRAIN_IMAGES,
        "repeat_metrics": repeated == metrics,
    }
    decision = (
        "style_count_scale_go"
        if all(checks.values()) and all(protocol_checks.values())
        else "style_count_scale_no_go"
    )
    final = {
        "decision": decision,
        "checks": checks,
        "protocol_checks": protocol_checks,
        "baseline_metrics": baseline_metrics,
        "metrics": metrics,
        "training_seconds": time.perf_counter() - started,
    }
    payload = {
        "checkpoint_schema_version": "g3-p3e14-style-count-scale-0.1",
        "model_type": MODEL_TYPE,
        "model_config": asdict(config),
        "deepsets_count": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": len(batches),
        "samples_seen": exposure,
        "training_images": TRAIN_IMAGES,
        "trainable_parameter_count": trainable_parameters,
        "metrics": final,
        "lineage": audit["lineage"],
    }
    checkpoint = OUTPUT / f"step-{len(batches):06d}.pt"
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
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
