#!/usr/bin/env python3
"""Audit target-free top-K peak reachability of the frozen P6-E1 heatmap."""

from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from chemvector.components.atoms.objects.g3_atom_center_field import (
    decode_atom_center_field,
)
from chemvector.components.recovery.spatial_secondary_peak import (
    spatial_offset_grid,
    spatial_support_mask,
)
from chemvector.components.recovery.structural_context_peak import (
    StructuralContextPeakHead,
    anchor_centered_context_crops,
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
from chemvector.training.atoms import g3_p4_local_recovery_data as recovery_data
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.atoms import train_g3_p1t_continuous_stride2_center_field as p1t
from chemvector.training.atoms import train_g3_p6e1_structural_context96_mixed as p6e1
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _autocast_context,
)
from chemvector.g3_structural_data import file_sha256


AUDIT_NAME = "chemvector-g3-p6e2-candidate-peak-reachability"
OUTPUT = Path(f"outputs/audits/{AUDIT_NAME}")
CHECKPOINT = p6e1.OUTPUT / "step-015000.pt"
CHECKPOINT_SHA256 = (
    "2af36baf611b9a4c899f807742345e48bf905a30705901ca69c29efac9ca987d"
)
P6E1_AUDIT = p6e1.OUTPUT / "audit.json"
P6E1_AUDIT_SHA256 = (
    "a70a5bf3b7920736b14c7d67dc1e25bce5bf44491aef4c7fe3395e63acc2ca3d"
)
BUDGETS = (1, 2, 4, 8)
IMAGE_SIZE = 768
TARGET_RADIUS_PIXELS = 6.0
MINIMUM_RECOVERY_RATE = 0.80
MAXIMUM_ORACLE_MAE_PIXELS = 3.0
SEED = p6e1.SEED


@torch.no_grad()
def ranked_regional_maximum_offsets(
    logits: torch.Tensor, *, maximum_candidates: int = max(BUDGETS)
) -> list[torch.Tensor]:
    """Return score-ranked 3x3 regional maxima without using target coordinates."""

    if logits.ndim != 3 or logits.shape[1:] != (32, 32):
        raise ValueError("P6-E2 logits must have shape [batch,32,32]")
    if not logits.is_floating_point() or not bool(torch.isfinite(logits).all()):
        raise ValueError("P6-E2 logits must be finite floating point")
    if isinstance(maximum_candidates, bool) or maximum_candidates < 1:
        raise ValueError("P6-E2 candidate budget must be positive")
    support = spatial_support_mask(device=logits.device)
    masked = logits.float().masked_fill(~support[None], -torch.inf)
    pooled = F.max_pool2d(masked[:, None], kernel_size=3, stride=1, padding=1)[:, 0]
    maxima = support[None] & (masked == pooled)
    grid = spatial_offset_grid(device=logits.device).reshape(-1, 2)
    outputs = []
    for batch_index in range(len(logits)):
        indices = torch.where(maxima[batch_index].flatten())[0]
        scores = masked[batch_index].flatten()[indices]
        # Stable score sorting preserves flat-index order for exact ties, matching argmax.
        order = scores.argsort(descending=True, stable=True)
        selected = indices[order[:maximum_candidates]]
        if not len(selected):
            raise RuntimeError("P6-E2 support unexpectedly has no regional maximum")
        outputs.append(grid[selected] / IMAGE_SIZE)
    return outputs


def _summary(records: list[dict], split: str) -> dict:
    current = [record for record in records if record["split"] == split]
    result = {
        "actions": len(current),
        "mean_regional_maxima": float(
            np.mean([record["regional_maxima"] for record in current])
        ),
        "budgets": {},
    }
    for budget in BUDGETS:
        distances = [record["oracle_distance_pixels"][str(budget)] for record in current]
        result["budgets"][str(budget)] = {
            "within_6px": sum(value <= TARGET_RADIUS_PIXELS for value in distances),
            "recovery_rate_6px": sum(
                value <= TARGET_RADIUS_PIXELS for value in distances
            )
            / len(distances),
            "oracle_mae_pixels": float(np.mean(distances)),
            "oracle_median_pixels": float(np.median(distances)),
            "oracle_p90_pixels": float(np.percentile(distances, 90)),
        }
    return result


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"P6-E2 output already exists: {OUTPUT}")
    for path, expected in (
        (CHECKPOINT, CHECKPOINT_SHA256),
        (P6E1_AUDIT, P6E1_AUDIT_SHA256),
        (p4e6.P4E3_CHECKPOINT, p4e6.P4E3_CHECKPOINT_SHA256),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"P6-E2 input SHA256 differs: {path}")
    if not torch.cuda.is_available():
        raise RuntimeError("P6-E2 requires CUDA BF16")
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
    model = StructuralContextPeakHead().to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    field, atom_head, vision, frozen_block, vision_sha = p4e6._load_model(device)
    collator = p1a.G3AtomCollator(maximum_atoms=128, image_size=768)
    records = []
    streams = {}
    started = time.perf_counter()

    with torch.no_grad():
        for split in ("calibration", "holdout"):
            manifest, actions_by_key = p6e1.p5e1._action_manifest(split)
            split_started = time.perf_counter()
            split_actions = 0
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
                darkness = recovery_data._grayscale_darkness(
                    batch["pixel_values"].to(device)
                )
                score_field = field_output["center_logits"].detach().float()
                for batch_index, key_value in enumerate(batch["keys"]):
                    key = str(key_value)
                    for action in actions_by_key[key]:
                        matches = torch.where(
                            observations["candidate_source_ids"][batch_index]
                            == int(action["base_source_id"])
                        )[0]
                        if len(matches) != 1:
                            raise RuntimeError(
                                f"P6-E2 Oracle source replay differs: {key}"
                            )
                        index = int(matches[0])
                        source = observations["candidate_anchors"][batch_index, index]
                        image_crop = anchor_centered_context_crops(
                            darkness[batch_index : batch_index + 1], source[None, None]
                        )[0]
                        score_crop = anchor_centered_context_crops(
                            score_field[batch_index : batch_index + 1], source[None, None]
                        )[0]
                        with torch.autocast(
                            device_type="cuda", dtype=torch.bfloat16
                        ):
                            logits = model(
                                image_crop,
                                score_crop,
                                observations["candidate_scores"][
                                    batch_index, index : index + 1
                                ],
                                observations["candidate_ranks"][
                                    batch_index, index : index + 1
                                ],
                            )
                        all_candidates = ranked_regional_maximum_offsets(
                            logits, maximum_candidates=32 * 32
                        )[0]
                        candidates = all_candidates[: max(BUDGETS)]
                        target = source.new_tensor(action["target_anchor"]) - source
                        distances = torch.linalg.vector_norm(
                            (candidates - target[None]) * 768, dim=1
                        ).cpu()
                        oracle_distances = {
                            str(budget): float(distances[:budget].min())
                            for budget in BUDGETS
                        }
                        records.append(
                            {
                                "split": split,
                                "sample_key": key,
                                "base_source_id": int(action["base_source_id"]),
                                "variant": str(action["variant"]),
                                "regional_maxima": len(all_candidates),
                                "oracle_distance_pixels": oracle_distances,
                            }
                        )
                        split_actions += 1
            streams[split] = {
                "images": len(manifest),
                "actions": split_actions,
                "elapsed_seconds": time.perf_counter() - split_started,
            }

    summaries = {
        split: _summary(records, split) for split in ("calibration", "holdout")
    }
    if summaries["calibration"]["actions"] != 31:
        raise RuntimeError("P6-E2 calibration action count differs")
    if summaries["holdout"]["actions"] != 67:
        raise RuntimeError("P6-E2 holdout action count differs")
    checks = {}
    for split in ("calibration", "holdout"):
        top8 = summaries[split]["budgets"]["8"]
        checks[f"{split}_top8_recovery_min80pct"] = (
            top8["recovery_rate_6px"] >= MINIMUM_RECOVERY_RATE
        )
        checks[f"{split}_top8_oracle_mae_max3px"] = (
            top8["oracle_mae_pixels"] <= MAXIMUM_ORACLE_MAE_PIXELS
        )
    decision = (
        "structured_peak_selection_capacity_go"
        if all(checks.values())
        else "structured_peak_selection_capacity_no_go"
    )
    OUTPUT.mkdir(parents=True)
    oracle._atomic_jsonl(OUTPUT / "records.jsonl", records)
    audit = {
        "schema_version": "g3-p6e2-candidate-peak-reachability-0.1",
        "decision": decision,
        "checks": checks,
        "summaries": summaries,
        "streams": streams,
        "protocol": {
            "training_performed": False,
            "candidate_extraction": "target-free 3x3 regional maxima of frozen P6-E1 logits",
            "candidate_budgets": BUDGETS,
            "target_hit_radius_pixels": TARGET_RADIUS_PIXELS,
            "top8_minimum_recovery_rate": MINIMUM_RECOVERY_RATE,
            "top8_maximum_oracle_mae_pixels": MAXIMUM_ORACLE_MAE_PIXELS,
            "dev_confirm_external_sealed_read": False,
        },
        "lineage": {
            "p6e1_checkpoint_sha256": CHECKPOINT_SHA256,
            "p6e1_audit_sha256": P6E1_AUDIT_SHA256,
            "vision_checkpoint_sha256": vision_sha,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(OUTPUT / "audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
