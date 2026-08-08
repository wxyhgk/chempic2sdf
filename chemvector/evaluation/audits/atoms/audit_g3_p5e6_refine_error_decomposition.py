#!/usr/bin/env python3
"""Decompose P5-E5 forced-source localization errors by frozen evidence buckets."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from Data.sdf.chemscene_v0_2 import build_chemscene_v0_2
from chemvector.components.atoms.objects.g3_atom_center_field import (
    decode_atom_center_field,
)
from chemvector.components.recovery.spatial_secondary_peak import (
    ContextualSecondaryPeakHead,
    decode_secondary_peak_offsets,
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
from chemvector.training.atoms import (
    train_g3_p1a_canonical_atom_selector as p1a,
)
from chemvector.training.atoms import (
    train_g3_p1t_continuous_stride2_center_field as p1t,
)
from chemvector.training.atoms import (
    train_g3_p4e9_local_recovery_pilot20k as p4train,
)
from chemvector.training.atoms import (
    train_g3_p5e1_spatial_secondary_peak_fit264 as p5e1,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _autocast_context,
)
from chemvector.g3_structural_data import file_sha256


AUDIT_NAME = "chemvector-g3-p5e6-refine-error-decomposition"
OUTPUT = Path(f"outputs/audits/{AUDIT_NAME}")
CHECKPOINT = Path(
    "outputs/checkpoints/chemvector-g3-p5e5-contextual-secondary-peak-fit1378/step-015660.pt"
)
CHECKPOINT_SHA256 = (
    "7c259f50d9a6846a7d748d7de3b02c7b4cb55b1d3fcd7a2e422f0f63db624bcd"
)
IMAGE_SIZE = 768
CROP_RADIUS_PIXELS = 16.0
SUCCESS_RADIUS_PIXELS = 6.0
PEAK_REACH_RADIUS_PIXELS = 3.0
MIN_INTERPRETABLE_BUCKET = 5
SEED = 20260856


class DiagnosticCollator(p1a.G3AtomCollator):
    def __call__(self, samples: list[dict]) -> dict:
        batch = super().__call__(samples)
        batch["scenes"] = [
            build_chemscene_v0_2(
                sample["chemmol"], sample["render"], sample.get("layers")
            )
            for sample in samples
        ]
        return batch


def distance_bucket(value: float) -> str:
    if not 6.0 < value <= 12.0:
        raise ValueError("P5-E6 source distance is outside the refine annulus")
    if value <= 8.0:
        return "6-8"
    if value <= 10.0:
        return "8-10"
    return "10-12"


def nearest_neighbor_bucket(value: float) -> str:
    if value < 16.0:
        return "lt16"
    if value < 32.0:
        return "16-32"
    return "ge32"


def rank_bucket(rank: int) -> str:
    if rank <= 16:
        return "top16"
    if rank <= 64:
        return "17-64"
    return "gt64"


def _scene_atom(scene: dict, target: torch.Tensor) -> dict:
    atoms = scene["atom_objects"]
    distances = [
        math.hypot(atom["anchor"][0] / 1023.0 - float(target[0]), atom["anchor"][1] / 1023.0 - float(target[1]))
        * IMAGE_SIZE
        for atom in atoms
    ]
    index = int(np.argmin(distances))
    if distances[index] > 1.0:
        raise RuntimeError("P5-E6 target does not map to a ChemScene AtomObject")
    return atoms[index]


def _bbox_intersects_crop(bbox: list[int], source: torch.Tensor) -> bool:
    source_pixels = source.float() * IMAGE_SIZE
    x0, y0, x1, y1 = (float(value) / 1023.0 * IMAGE_SIZE for value in bbox)
    left = float(source_pixels[0]) - CROP_RADIUS_PIXELS
    right = float(source_pixels[0]) + CROP_RADIUS_PIXELS
    top = float(source_pixels[1]) - CROP_RADIUS_PIXELS
    bottom = float(source_pixels[1]) + CROP_RADIUS_PIXELS
    return x1 >= left and x0 <= right and y1 >= top and y0 <= bottom


def _bucket_summary(records: list[dict], field: str) -> dict:
    output = {}
    for value in sorted({str(record[field]) for record in records}):
        current = [record for record in records if str(record[field]) == value]
        errors = [record["error_pixels"] for record in current]
        successes = sum(record["success_6px"] for record in current)
        output[value] = {
            "actions": len(current),
            "successes": successes,
            "recovery_rate_6px": successes / len(current),
            "mae_pixels": float(np.mean(errors)),
            "median_pixels": float(np.median(errors)),
            "interpretable_n_ge5": len(current) >= MIN_INTERPRETABLE_BUCKET,
        }
    return output


def _summaries(records: list[dict]) -> dict:
    fields = (
        "variant",
        "source_distance_bucket",
        "target_role",
        "target_element",
        "target_degree",
        "crop_has_text",
        "source_gt_count_16px",
        "nearest_neighbor_bucket",
        "target_score_rank_bucket",
        "score_peak_within_3px",
        "angular_error_bucket",
    )
    return {field: _bucket_summary(records, field) for field in fields}


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"P5-E6 output already exists: {OUTPUT}")
    for path, expected in (
        (CHECKPOINT, CHECKPOINT_SHA256),
        (p4train.ORACLE_AUDIT, p5e1.ORACLE_AUDIT_SHA256),
        (p4train.ORACLE_ACTIONS, p5e1.ORACLE_ACTIONS_SHA256),
        (p4e6.P4E3_CHECKPOINT, p4e6.P4E3_CHECKPOINT_SHA256),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"P5-E6 input SHA256 differs: {path}")
    if not torch.cuda.is_available():
        raise RuntimeError("P5-E6 requires CUDA BF16")
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
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = ContextualSecondaryPeakHead().to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    field, atom_head, vision, frozen_block, vision_sha = p4e6._load_model(device)
    collator = DiagnosticCollator(maximum_atoms=128, image_size=IMAGE_SIZE)
    grid = spatial_offset_grid()
    support = spatial_support_mask()
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
                    atom_head,
                    vision,
                    frozen_block,
                    batch["pixel_values"],
                    device=device,
                    training=False,
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
                    gt_anchors = batch["atom_anchors"][batch_index][batch["atom_mask"][batch_index]].float()
                    scene = batch["scenes"][batch_index]
                    for action in by_key[key]:
                        matches = torch.where(
                            observations["candidate_source_ids"][batch_index]
                            == int(action["base_source_id"])
                        )[0]
                        if len(matches) != 1:
                            raise RuntimeError(
                                "P5-E6 Oracle source replay differs: "
                                f"key={key} source={action['base_source_id']} "
                                f"matches={len(matches)}"
                            )
                        index = int(matches[0])
                        image_crop = observations["image_crops"][batch_index, index : index + 1]
                        score_crop = observations["score_crops"][batch_index, index : index + 1]
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            logits = model(
                                image_crop,
                                score_crop,
                                observations["candidate_scores"][batch_index, index : index + 1],
                                observations["candidate_ranks"][batch_index, index : index + 1],
                            )
                        predicted_offset = decode_secondary_peak_offsets(logits)[0].cpu()
                        source = observations["candidate_anchors"][batch_index, index].cpu()
                        target = torch.tensor(action["target_anchor"], dtype=torch.float32)
                        target_offset = target - source
                        error = float(torch.linalg.vector_norm((predicted_offset - target_offset) * IMAGE_SIZE))
                        source_distance = float(torch.linalg.vector_norm(target_offset * IMAGE_SIZE))
                        target_pixels = target_offset * IMAGE_SIZE
                        score = score_crop[0, 0].float().cpu()
                        target_index = int(((grid - target_pixels).square().sum(dim=-1)).argmin())
                        flat_score = score.masked_fill(~support, -torch.inf).flatten()
                        target_value = flat_score[target_index]
                        target_rank = int((flat_score > target_value).sum()) + 1
                        local_peaks = (score == F.max_pool2d(score[None, None], 3, 1, 1)[0, 0]) & support
                        peak_distances = torch.linalg.vector_norm(grid - target_pixels, dim=-1).masked_fill(~local_peaks, torch.inf)
                        peak_within = bool(peak_distances.min() <= PEAK_REACH_RADIUS_PIXELS)
                        support_device = support.to(logits.device)
                        masked_logits = logits[0].float().masked_fill(
                            ~support_device, -torch.inf
                        ).flatten()
                        probabilities = torch.softmax(masked_logits, dim=0)
                        top2 = masked_logits.topk(2).values
                        entropy = float(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())
                        margin = float(top2[0] - top2[1])

                        target_atom = _scene_atom(scene, target)
                        text_boxes = [atom["text"]["bbox"] for atom in scene["atom_objects"] if atom["text"] is not None]
                        crop_has_text = any(_bbox_intersects_crop(box, source) for box in text_boxes)
                        target_distances = torch.linalg.vector_norm((gt_anchors - target) * IMAGE_SIZE, dim=1)
                        nonzero = target_distances[target_distances > 1.0]
                        nearest = float(nonzero.min()) if len(nonzero) else math.inf
                        source_gt_count = int((torch.linalg.vector_norm((gt_anchors - source) * IMAGE_SIZE, dim=1) <= CROP_RADIUS_PIXELS).sum())
                        target_vector = target_offset * IMAGE_SIZE
                        predicted_vector = predicted_offset * IMAGE_SIZE
                        cosine = float(
                            torch.dot(target_vector, predicted_vector)
                            / (torch.linalg.vector_norm(target_vector) * torch.linalg.vector_norm(predicted_vector)).clamp_min(1e-6)
                        )
                        angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
                        if angle <= 30.0:
                            angle_bucket = "le30"
                        elif angle <= 90.0:
                            angle_bucket = "31-90"
                        else:
                            angle_bucket = "gt90"
                        records.append(
                            {
                                "split": split,
                                "sample_key": key,
                                "molecule_id": action["molecule_id"],
                                "scaffold": action["scaffold"],
                                "variant": action["variant"],
                                "base_source_id": action["base_source_id"],
                                "source_distance_pixels": source_distance,
                                "source_distance_bucket": distance_bucket(source_distance),
                                "target_role": target_atom["role"],
                                "target_element": target_atom["chemical"]["element"],
                                "target_degree": len(target_atom["incident_ports"]),
                                "crop_has_text": crop_has_text,
                                "source_gt_count_16px": min(source_gt_count, 3),
                                "nearest_neighbor_pixels": nearest,
                                "nearest_neighbor_bucket": nearest_neighbor_bucket(nearest),
                                "target_score_rank": target_rank,
                                "target_score_rank_bucket": rank_bucket(target_rank),
                                "score_peak_within_3px": peak_within,
                                "prediction_radius_pixels": float(torch.linalg.vector_norm(predicted_vector)),
                                "angular_error_degrees": angle,
                                "angular_error_bucket": angle_bucket,
                                "heatmap_margin": margin,
                                "heatmap_entropy": entropy,
                                "error_pixels": error,
                                "success_6px": error <= SUCCESS_RADIUS_PIXELS,
                            }
                        )

    expected = {"calibration": 31, "holdout": 67}
    counts = Counter(record["split"] for record in records)
    if dict(counts) != expected:
        raise RuntimeError(f"P5-E6 action counts differ: {counts}")
    summaries = {
        split: _summaries([record for record in records if record["split"] == split])
        for split in expected
    }
    summaries["combined"] = _summaries(records)
    primary = {}
    for split in expected:
        current = [record for record in records if record["split"] == split]
        success = [record for record in current if record["success_6px"]]
        failure = [record for record in current if not record["success_6px"]]
        primary[split] = {
            "actions": len(current),
            "successes": len(success),
            "recovery_rate_6px": len(success) / len(current),
            "failure_angular_gt90": sum(record["angular_error_degrees"] > 90.0 for record in failure),
            "failure_score_peak_absent": sum(not record["score_peak_within_3px"] for record in failure),
            "failure_text_target": sum(record["target_role"] == "TEXT" for record in failure),
            "failure_crowded_source": sum(record["source_gt_count_16px"] >= 2 for record in failure),
        }
    decision = "error_decomposition_completed"
    OUTPUT.mkdir(parents=True)
    oracle._atomic_jsonl(OUTPUT / "records.jsonl", records)
    audit = {
        "schema_version": "g3-p5e6-refine-error-decomposition-0.1",
        "decision": decision,
        "protocol": {
            "training_performed": False,
            "forced_oracle_source": True,
            "success_radius_pixels": SUCCESS_RADIUS_PIXELS,
            "minimum_interpretable_bucket": MIN_INTERPRETABLE_BUCKET,
            "dev_confirm_external_sealed_read": False,
        },
        "primary": primary,
        "summaries": summaries,
        "lineage": {
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "vision_checkpoint_sha256": vision_sha,
            "oracle_actions_sha256": p5e1.ORACLE_ACTIONS_SHA256,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(OUTPUT / "audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
