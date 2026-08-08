#!/usr/bin/env python3
"""Audit whether P5-E5 wrong directions point toward another true Atom."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from chemvector.components.atoms.objects.g3_atom_center_field import (
    decode_atom_center_field,
)
from chemvector.components.recovery.spatial_secondary_peak import (
    ContextualSecondaryPeakHead,
    decode_secondary_peak_offsets,
)
from chemvector.evaluation.audits.atoms.audit_g3_p1n_regional_maxima import (
    extract_regional_maximum_objects,
)
from chemvector.evaluation.audits.atoms import (
    audit_g3_p4e6_missing_peak_sources as p4e6,
)
from chemvector.evaluation.audits.atoms import (
    audit_g3_p5e6_refine_error_decomposition as p5e6,
)
from chemvector.evaluation.audits.atoms import (
    build_g3_p4e9_recovery_oracle as oracle,
)
from chemvector.evaluation.evaluators.atoms import (
    evaluate_g3_p4e9_local_recovery_pilot20k as p4eval,
)
from chemvector.training.atoms import g3_p4_local_recovery_data as recovery_data
from chemvector.training.atoms import (
    train_g3_p1a_canonical_atom_selector as p1a,
)
from chemvector.training.atoms import (
    train_g3_p1t_continuous_stride2_center_field as p1t,
)
from chemvector.training.atoms import (
    train_g3_p4e9_local_recovery_pilot20k as p4train,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _autocast_context,
)
from chemvector.g3_structural_data import file_sha256


AUDIT_NAME = "chemvector-g3-p5e7-wrong-destination"
OUTPUT = Path(f"outputs/audits/{AUDIT_NAME}")
P5E6_AUDIT = p5e6.OUTPUT / "audit.json"
P5E6_AUDIT_SHA256 = (
    "c47f29083c455a90efa99b09431e319b5a7cd33b4f332b3fcfe9534bb069a4d0"
)
RADIUS_PIXELS = 6.0
MIN_OTHER_CLOSER_RATE = 0.40


def destination_class(target_distance: float, other_distance: float) -> str:
    if target_distance <= RADIUS_PIXELS:
        return "TARGET_HIT"
    if other_distance <= RADIUS_PIXELS:
        return "OTHER_GT_HIT"
    if other_distance < target_distance:
        return "OTHER_GT_CLOSER"
    return "TARGET_STILL_NEAREST"


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"P5-E7 output already exists: {OUTPUT}")
    for path, expected in (
        (p5e6.CHECKPOINT, p5e6.CHECKPOINT_SHA256),
        (P5E6_AUDIT, P5E6_AUDIT_SHA256),
        (p4train.ORACLE_ACTIONS, p5e6.p5e1.ORACLE_ACTIONS_SHA256),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"P5-E7 input SHA256 differs: {path}")
    if not torch.cuda.is_available():
        raise RuntimeError("P5-E7 requires CUDA BF16")
    p5e6.os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    p5e6.random.seed(p5e6.SEED)
    np.random.seed(p5e6.SEED)
    torch.manual_seed(p5e6.SEED)
    torch.cuda.manual_seed_all(p5e6.SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    payload = torch.load(p5e6.CHECKPOINT, map_location="cpu", weights_only=False)
    model = ContextualSecondaryPeakHead().to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    field, atom_head, vision, frozen_block, vision_sha = p4e6._load_model(device)
    collator = p1a.G3AtomCollator(maximum_atoms=128, image_size=p5e6.IMAGE_SIZE)
    records = []
    started = time.perf_counter()

    with torch.no_grad():
        for split in ("calibration", "holdout"):
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
                    gt = batch["atom_anchors"][batch_index][batch["atom_mask"][batch_index]].float()
                    for action in by_key[key]:
                        matches = torch.where(
                            observations["candidate_source_ids"][batch_index]
                            == int(action["base_source_id"])
                        )[0]
                        if len(matches) != 1:
                            raise RuntimeError(f"P5-E7 Oracle source replay differs: {key}")
                        index = int(matches[0])
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            logits = model(
                                observations["image_crops"][batch_index, index : index + 1],
                                observations["score_crops"][batch_index, index : index + 1],
                                observations["candidate_scores"][batch_index, index : index + 1],
                                observations["candidate_ranks"][batch_index, index : index + 1],
                            )
                        offset = decode_secondary_peak_offsets(logits)[0].cpu()
                        source = observations["candidate_anchors"][batch_index, index].cpu()
                        predicted = source + offset
                        target_index = int(action["target_index"])
                        distances = torch.linalg.vector_norm((gt - predicted) * p5e6.IMAGE_SIZE, dim=1)
                        target_distance = float(distances[target_index])
                        other = torch.cat((distances[:target_index], distances[target_index + 1 :]))
                        other_distance = float(other.min()) if len(other) else float("inf")
                        kind = destination_class(target_distance, other_distance)
                        records.append(
                            {
                                "split": split,
                                "sample_key": key,
                                "variant": action["variant"],
                                "target_index": target_index,
                                "target_distance_pixels": target_distance,
                                "nearest_other_gt_pixels": other_distance,
                                "destination": kind,
                            }
                        )

    summaries = {}
    checks = {}
    for split, expected in (("calibration", 31), ("holdout", 67)):
        current = [record for record in records if record["split"] == split]
        if len(current) != expected:
            raise RuntimeError(f"P5-E7 {split} action count differs")
        counts = Counter(record["destination"] for record in current)
        failures = expected - counts["TARGET_HIT"]
        other_closer = counts["OTHER_GT_HIT"] + counts["OTHER_GT_CLOSER"]
        rate = other_closer / failures if failures else 0.0
        summaries[split] = {
            "actions": expected,
            "failures": failures,
            "destinations": dict(sorted(counts.items())),
            "failed_prediction_other_gt_closer": other_closer,
            "failed_prediction_other_gt_closer_rate": rate,
        }
        checks[f"{split}_other_gt_closer_min40pct"] = rate >= MIN_OTHER_CLOSER_RATE
    decision = (
        "wrong_destination_competition_go"
        if all(checks.values())
        else "wrong_destination_competition_no_go"
    )
    OUTPUT.mkdir(parents=True)
    oracle._atomic_jsonl(OUTPUT / "records.jsonl", records)
    audit = {
        "schema_version": "g3-p5e7-wrong-destination-0.1",
        "decision": decision,
        "checks": checks,
        "summaries": summaries,
        "protocol": {
            "training_performed": False,
            "target_hit_radius_pixels": RADIUS_PIXELS,
            "other_gt_closer_gate": MIN_OTHER_CLOSER_RATE,
            "dev_confirm_external_sealed_read": False,
        },
        "lineage": {
            "p5e5_checkpoint_sha256": p5e6.CHECKPOINT_SHA256,
            "p5e6_audit_sha256": P5E6_AUDIT_SHA256,
            "vision_checkpoint_sha256": vision_sha,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(OUTPUT / "audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
