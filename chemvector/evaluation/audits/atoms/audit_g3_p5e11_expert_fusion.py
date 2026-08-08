#!/usr/bin/env python3
"""Fuse the P5-E5 real expert and P5-E9 generic-regularized expert."""

from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from chemvector.components.atoms.objects.g3_atom_center_field import decode_atom_center_field
from chemvector.components.recovery.spatial_secondary_peak import (
    ContextualSecondaryPeakHead,
    spatial_offset_grid,
    spatial_support_mask,
)
from chemvector.evaluation.audits.atoms.audit_g3_p1n_regional_maxima import (
    extract_regional_maximum_objects,
)
from chemvector.evaluation.audits.atoms import (
    audit_g3_p4e6_missing_peak_sources as p4e6,
)
from chemvector.evaluation.audits.atoms import (
    build_g3_p4e9_recovery_oracle as oracle,
)
from chemvector.evaluation.evaluators.atoms import (
    evaluate_g3_p4e9_local_recovery_pilot20k as p4eval,
)
from chemvector.training.atoms import g3_p4_local_recovery_data as recovery_data
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.atoms import train_g3_p1t_continuous_stride2_center_field as p1t
from chemvector.training.atoms import train_g3_p4e9_local_recovery_pilot20k as p4train
from chemvector.training.atoms import train_g3_p5e1_spatial_secondary_peak_fit264 as p5e1
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _autocast_context,
)
from chemvector.g3_structural_data import file_sha256


AUDIT_NAME = "chemvector-g3-p5e11-expert-fusion"
OUTPUT = Path(f"outputs/audits/{AUDIT_NAME}")
REAL_CHECKPOINT = Path(
    "outputs/checkpoints/chemvector-g3-p5e5-contextual-secondary-peak-fit1378/step-015660.pt"
)
REAL_CHECKPOINT_SHA256 = "7c259f50d9a6846a7d748d7de3b02c7b4cb55b1d3fcd7a2e422f0f63db624bcd"
GENERIC_CHECKPOINT = Path(
    "outputs/checkpoints/chemvector-g3-p5e9-generic-direction-mixed50k-fit1378/step-015000.pt"
)
GENERIC_CHECKPOINT_SHA256 = "b852bdc369639c8e1ceeed9e8c8b36ae4e785dae25fa24cb3666d009edd1e42e"
REAL_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
IMAGE_SIZE = 768
SEED = 20260860


def select_weight(metrics: dict[float, dict]) -> float:
    passed = [
        weight
        for weight, values in metrics.items()
        if values["recovery_rate_6px"] >= 0.80 and values["mae_pixels"] <= 3.0
    ]
    candidates = passed if passed else list(metrics)
    return min(
        candidates,
        key=lambda weight: (
            -metrics[weight]["recovery_rate_6px"],
            metrics[weight]["mae_pixels"],
            weight,
        ),
    )


def _metrics(distances: list[float]) -> dict:
    return {
        "actions": len(distances),
        "within_6px": sum(value <= 6.0 for value in distances),
        "recovery_rate_6px": sum(value <= 6.0 for value in distances) / len(distances),
        "mae_pixels": float(np.mean(distances)),
        "median_pixels": float(np.median(distances)),
    }


def _load_model(path: Path, device: torch.device) -> ContextualSecondaryPeakHead:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = ContextualSecondaryPeakHead().to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"P5-E11 output already exists: {OUTPUT}")
    for path, expected in (
        (REAL_CHECKPOINT, REAL_CHECKPOINT_SHA256),
        (GENERIC_CHECKPOINT, GENERIC_CHECKPOINT_SHA256),
        (p4train.ORACLE_ACTIONS, p5e1.ORACLE_ACTIONS_SHA256),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"P5-E11 input SHA256 differs: {path}")
    if not torch.cuda.is_available():
        raise RuntimeError("P5-E11 requires CUDA BF16")
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
    real_model = _load_model(REAL_CHECKPOINT, device)
    generic_model = _load_model(GENERIC_CHECKPOINT, device)
    field, atom_head, vision, frozen_block, vision_sha = p4e6._load_model(device)
    collator = p1a.G3AtomCollator(maximum_atoms=128, image_size=IMAGE_SIZE)
    support = spatial_support_mask(device=device).flatten()
    grid = spatial_offset_grid(device=device).reshape(-1, 2)
    records: dict[str, list[dict]] = {"calibration": [], "holdout": []}
    started = time.perf_counter()

    with torch.no_grad():
        for split in records:
            actions = [
                row
                for row in p4eval._jsonl(p4train.ORACLE_ACTIONS)
                if row["route"] == "ADD_REFINE" and row["split"] == split
            ]
            by_key: dict[str, list[dict]] = defaultdict(list)
            for action in actions:
                by_key[str(action["sample_key"])].append(action)
            path = p4eval.ORACLE_CALIBRATION if split == "calibration" else p4eval.ORACLE_HOLDOUT
            full, _ = p4eval._manifest(path)
            manifest = [row for row in full if row.sample_key in by_key]
            for batch in p1a.iter_manifest_batches(
                manifest, collator, batch_size=8, accepted_split="train"
            ):
                objects, stroke = p1t.model_inputs(
                    atom_head, vision, frozen_block, batch["pixel_values"], device=device, training=False
                )
                with _autocast_context(device, "bf16"):
                    field_output = field(objects, stroke)
                peaks = extract_regional_maximum_objects(
                    decode_atom_center_field(field_output), maximum_objects=128
                )
                observations = recovery_data.build_local_recovery_observations(
                    batch["pixel_values"].to(device), field_output, peaks
                )
                for batch_index, key_value in enumerate(batch["keys"]):
                    key = str(key_value)
                    for action in by_key[key]:
                        matches = torch.where(
                            observations["candidate_source_ids"][batch_index]
                            == int(action["base_source_id"])
                        )[0]
                        if len(matches) != 1:
                            raise RuntimeError(f"P5-E11 Oracle source replay differs: {key}")
                        index = int(matches[0])
                        inputs = (
                            observations["image_crops"][batch_index, index : index + 1],
                            observations["score_crops"][batch_index, index : index + 1],
                            observations["candidate_scores"][batch_index, index : index + 1],
                            observations["candidate_ranks"][batch_index, index : index + 1],
                        )
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            real_logits = real_model(*inputs)[0].float().flatten()
                            generic_logits = generic_model(*inputs)[0].float().flatten()
                        real_probability = torch.softmax(real_logits.masked_fill(~support, -torch.inf), dim=0)
                        generic_probability = torch.softmax(
                            generic_logits.masked_fill(~support, -torch.inf), dim=0
                        )
                        source = observations["candidate_anchors"][batch_index, index]
                        target = source.new_tensor(action["target_anchor"])
                        expected_offset = target - source
                        distances = {}
                        for weight in REAL_WEIGHTS:
                            probability = weight * real_probability + (1.0 - weight) * generic_probability
                            predicted_offset = grid[int(probability.argmax())] / IMAGE_SIZE
                            distances[weight] = float(
                                torch.linalg.vector_norm(
                                    (predicted_offset - expected_offset) * IMAGE_SIZE
                                )
                            )
                        records[split].append(
                            {"sample_key": key, "variant": action["variant"], "distances": distances}
                        )

    weight_metrics = {
        split: {
            weight: _metrics([row["distances"][weight] for row in rows])
            for weight in REAL_WEIGHTS
        }
        for split, rows in records.items()
    }
    selected_weight = select_weight(weight_metrics["calibration"])
    calibration = weight_metrics["calibration"][selected_weight]
    holdout = weight_metrics["holdout"][selected_weight]
    oracle_union = {}
    for split, rows in records.items():
        oracle_union[split] = _metrics(
            [min(row["distances"][0.0], row["distances"][1.0]) for row in rows]
        )
    checks = {
        "calibration_recovery_min80pct": calibration["recovery_rate_6px"] >= 0.80,
        "holdout_recovery_min80pct": holdout["recovery_rate_6px"] >= 0.80,
        "calibration_mae_max3px": calibration["mae_pixels"] <= 3.0,
        "holdout_mae_max3px": holdout["mae_pixels"] <= 3.0,
    }
    decision = "expert_probability_fusion_go" if all(checks.values()) else "expert_probability_fusion_no_go"
    OUTPUT.mkdir(parents=True)
    audit = {
        "schema_version": "g3-p5e11-expert-fusion-0.1",
        "decision": decision,
        "checks": checks,
        "selected_real_weight": selected_weight,
        "weight_metrics": weight_metrics,
        "selected": {"calibration": calibration, "holdout": holdout},
        "oracle_union": oracle_union,
        "protocol": {
            "training_performed": False,
            "fusion": "convex mixture of annulus-normalized probabilities",
            "real_weights": REAL_WEIGHTS,
            "dev_confirm_external_sealed_read": False,
        },
        "lineage": {
            "real_checkpoint_sha256": REAL_CHECKPOINT_SHA256,
            "generic_checkpoint_sha256": GENERIC_CHECKPOINT_SHA256,
            "vision_checkpoint_sha256": vision_sha,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(OUTPUT / "audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
