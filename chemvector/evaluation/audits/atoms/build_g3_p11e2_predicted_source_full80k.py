#!/usr/bin/env python3
"""Build full-scale real predicted-source KEEP/MOVE caches from 80k images."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from chemvector.components.atoms.objects.g3_atom_center_field import decode_atom_center_field
from chemvector.components.recovery.structural_context_peak import (
    StructuralContextPeakHead,
    anchor_centered_context_crops,
)
from chemvector.evaluation.audits.atoms import audit_g3_p1n_regional_maxima as p1n
from chemvector.evaluation.audits.atoms import audit_g3_p4e6_missing_peak_sources as p4e6
from chemvector.evaluation.audits.atoms import build_g3_p4e9_recovery_oracle as oracle
from chemvector.evaluation.audits.atoms import build_g3_p6e8_residual_cycle_data as p6e8
from chemvector.evaluation.audits.atoms import build_g3_p11e0_predicted_source_pilot as p11e0
from chemvector.g3_structural_data import file_sha256
from chemvector.training.atoms import g3_p4_local_recovery_data as recovery_data
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.atoms import train_g3_p1t_continuous_stride2_center_field as p1t
from chemvector.training.atoms import train_g3_p6e3_structural_peak_selector as p6e3
from chemvector.training.atoms import train_g3_p6e9_structural_residual_arms as p6e9
from chemvector.training.atoms import train_g3_p7_pairwise_counterfactual as p7
from chemvector.training.graph.scenes.g3_structural_data import load_g3_sample_manifest
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import _autocast_context


AUDIT_NAME = "chemvector-g3-p11e2-predicted-source-full80k"
OUTPUT = Path("outputs/audits") / AUDIT_NAME
FULL_MANIFEST = Path("outputs/audits/chemvector-g3-p4e0-source-union-cycle-manifests/full_train.jsonl")
FULL_MANIFEST_SHA256 = "c1c787a646b5827ae72accd62dbe791ba2beb0a866d88741be3d79d26d114795"
SPLITS = ("train", "calibration", "holdout")
IMAGE_SIZE = 768.0
TOP_K = 128
BATCH_SIZE = 8
KEEP_PER_IMAGE = 8
MINIMUM_MOVE = 1_000
MINIMUM_REACHABLE_MOVE = 800
MINIMUM_TRAIN_KEEP = 6_000
MINIMUM_CALIBRATION_MOVE = 100
MINIMUM_HOLDOUT_MOVE = 100
SEED = 20260896


def scaffold_split(scaffold: str) -> str:
    value = int.from_bytes(hashlib.sha256(f"p11e2-split:{scaffold}".encode()).digest()[:8], "big") % 10
    return "train" if value < 8 else "calibration" if value == 8 else "holdout"


def _empty_cache() -> dict[str, list]:
    return defaultdict(list)


def _append(
    cache: dict[str, list],
    image_crops: torch.Tensor,
    score_crops: torch.Tensor,
    offsets: torch.Tensor,
    logits: torch.Tensor,
    masks: torch.Tensor,
    target_offsets: torch.Tensor,
    metadata: list[dict],
) -> None:
    cache["image_crops"].append(image_crops.cpu().half())
    cache["score_crops"].append(score_crops.cpu().half())
    cache["candidate_offsets"].append(offsets.cpu().half())
    cache["candidate_peak_logits"].append(logits.cpu().half())
    cache["candidate_mask"].append(masks.cpu())
    cache["offset_targets"].append(target_offsets.cpu().float())
    for name in (
        "sample_key",
        "molecule_id",
        "scaffold",
        "variant",
        "action",
        "target_role",
        "degree_bucket",
        "target_index",
        "base_source_id",
    ):
        cache[name].extend([row[name] for row in metadata])


def _finish(cache: dict[str, list], split: str, vision_sha: str) -> dict:
    tensors = (
        "image_crops",
        "score_crops",
        "candidate_offsets",
        "candidate_peak_logits",
        "candidate_mask",
        "offset_targets",
    )
    output = {name: torch.cat(cache[name]) for name in tensors}
    output.update({name: value for name, value in cache.items() if name not in tensors})
    output.update(
        {
            "schema_version": "g3-p11e2-predicted-source-full80k-cache-0.1",
            "split": split,
            "p6e1_checkpoint_sha256": p6e8.P6E1_CHECKPOINT_SHA256,
            "vision_checkpoint_sha256": vision_sha,
            "full_manifest_sha256": FULL_MANIFEST_SHA256,
        }
    )
    return output


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"P11-E2 output already exists: {OUTPUT}")
    if file_sha256(FULL_MANIFEST) != FULL_MANIFEST_SHA256:
        raise ValueError("P11-E2 full manifest SHA256 differs")
    if file_sha256(p6e8.P6E1_CHECKPOINT) != p6e8.P6E1_CHECKPOINT_SHA256:
        raise ValueError("P11-E2 P6-E1 checkpoint SHA256 differs")
    if any((path.stat().st_mode & 0o777) != 0 for path in p6e9.HOLDOUT_FILES):
        raise RuntimeError("P11-E2 historical holdout must remain sealed")
    if not torch.cuda.is_available():
        raise RuntimeError("P11-E2 requires CUDA")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest, manifest_audit = load_g3_sample_manifest(
        FULL_MANIFEST,
        expected_sha256=FULL_MANIFEST_SHA256,
        expected_samples=80_000,
        expected_molecules=20_000,
        expected_variants=frozenset({"coordgen", "original", "rdkit", "stress"}),
    )
    raw_meta = [json.loads(line) for line in FULL_MANIFEST.read_text().splitlines() if line]
    metadata_by_key = {str(row["sample_key"]): row for row in raw_meta}
    device = torch.device("cuda")
    payload = torch.load(p6e8.P6E1_CHECKPOINT, map_location="cpu", weights_only=False)
    peak_model = StructuralContextPeakHead().to(device)
    peak_model.load_state_dict(payload["model"], strict=True)
    peak_model.eval().requires_grad_(False)
    field, atom_head, vision, frozen_block, vision_sha = p4e6._load_model(device)
    collator = p6e8.ResidualCycleCollator(maximum_atoms=128, image_size=768)
    caches = {split: _empty_cache() for split in SPLITS}
    counts = Counter()
    variants = Counter()
    split_scaffolds = {split: set() for split in SPLITS}
    processed = 0
    started = time.perf_counter()
    with torch.no_grad():
        for batch in p1a.iter_manifest_batches(manifest, collator, batch_size=BATCH_SIZE, accepted_split="train"):
            objects, stroke = p1t.model_inputs(
                atom_head, vision, frozen_block, batch["pixel_values"], device=device, training=False
            )
            with _autocast_context(device, "bf16"):
                field_output = field(objects, stroke)
            peaks = p1n.extract_regional_maximum_objects(
                decode_atom_center_field(field_output), maximum_objects=TOP_K
            )
            darkness = recovery_data._grayscale_darkness(batch["pixel_values"].to(device))
            center = field_output["center_logits"].detach().float()
            for batch_index, key_value in enumerate(batch["keys"]):
                key = str(key_value)
                meta = metadata_by_key[key]
                split = scaffold_split(str(meta["scaffold"]))
                raw = peaks[batch_index]["anchors"].float().to(device)
                source_ids = peaks[batch_index]["source_ids"].long().to(device)
                expected = batch["atom_anchors"][batch_index][batch["atom_mask"][batch_index]].float().to(device)
                actions = [
                    row
                    for row in oracle.recovery_actions(raw.cpu(), source_ids.cpu(), expected.cpu())
                    if row["route"] == "ADD_REFINE"
                ]
                if not actions:
                    continue
                move_sources = {int(row["candidate_index"]) for row in actions}
                specs = [
                    (
                        "MOVE",
                        raw[int(row["candidate_index"])],
                        int(row["target_index"]),
                        int(row["base_source_id"]),
                    )
                    for row in actions
                ]
                assignments = p4e6.maximum_cardinality_assignments(raw.cpu(), expected.cpu())
                for target_index, source_index in p11e0.select_keep_assignments(
                    key, assignments, move_sources, maximum=KEEP_PER_IMAGE
                ):
                    specs.append(("KEEP", raw[source_index], target_index, int(source_ids[source_index])))
                sources = torch.stack([source for _, source, _, _ in specs])
                targets = torch.stack([expected[target_index] for _, _, target_index, _ in specs])
                image_crops = anchor_centered_context_crops(darkness[batch_index : batch_index + 1], sources[None])[0]
                score_crops = anchor_centered_context_crops(center[batch_index : batch_index + 1], sources[None])[0]
                zeros = torch.zeros(len(sources), device=device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    heatmaps = peak_model(image_crops, score_crops, zeros, zeros)
                offsets, logits, masks = p6e3.candidate_tensors(heatmaps)
                target_offsets = targets - sources
                distances = torch.linalg.vector_norm(
                    (offsets - target_offsets[:, None]) * IMAGE_SIZE, dim=-1
                ).masked_fill(~masks, torch.inf)
                item_meta = []
                for row, (action, _source, target_index, base_source_id) in enumerate(specs):
                    reachable = bool((distances[row] <= 6.0).any())
                    harmful = int(((distances[row] > 6.0) & masks[row]).sum())
                    equivalent = int(((distances[row] <= 6.0) & masks[row]).sum())
                    counts[(split, action, "sources")] += 1
                    counts[(split, action, "reachable")] += int(reachable)
                    counts[(split, action, "harmful_pairs")] += harmful
                    counts[(split, action, "equivalent_pairs")] += equivalent
                    variants[(split, action, str(meta["variant"]))] += 1
                    item_meta.append(
                        {
                            "sample_key": key,
                            "molecule_id": str(meta["molecule_id"]),
                            "scaffold": str(meta["scaffold"]),
                            "variant": str(meta["variant"]),
                            "action": action,
                            "target_role": batch["atom_roles"][batch_index][target_index],
                            "degree_bucket": p6e8.degree_bucket(batch["atom_degrees"][batch_index][target_index]),
                            "target_index": target_index,
                            "base_source_id": base_source_id,
                        }
                    )
                    split_scaffolds[split].add(str(meta["scaffold"]))
                _append(caches[split], image_crops, score_crops, offsets, logits, masks, target_offsets, item_meta)
            processed += len(batch["keys"])
            if processed % 4_000 == 0:
                print(
                    json.dumps(
                        {
                            "processed": processed,
                            "move": sum(value for (split, action, kind), value in counts.items() if action == "MOVE" and kind == "sources"),
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
    total_move = sum(counts[(split, "MOVE", "sources")] for split in SPLITS)
    total_reachable = sum(counts[(split, "MOVE", "reachable")] for split in SPLITS)
    overlap = bool(
        (split_scaffolds["train"] & split_scaffolds["calibration"])
        or (split_scaffolds["train"] & split_scaffolds["holdout"])
        or (split_scaffolds["calibration"] & split_scaffolds["holdout"])
    )
    checks = {
        "images_processed_80000": processed == 80_000,
        "move_sources_min1000": total_move >= MINIMUM_MOVE,
        "move_reachable_min800": total_reachable >= MINIMUM_REACHABLE_MOVE,
        "train_keep_min6000": counts[("train", "KEEP", "sources")] >= MINIMUM_TRAIN_KEEP,
        "calibration_move_min100": counts[("calibration", "MOVE", "sources")] >= MINIMUM_CALIBRATION_MOVE,
        "holdout_move_min100": counts[("holdout", "MOVE", "sources")] >= MINIMUM_HOLDOUT_MOVE,
        "scaffold_overlap_zero": not overlap,
        "all_split_variants": all(
            variants[(split, "MOVE", variant)] > 0
            for split in SPLITS
            for variant in ("coordgen", "original", "rdkit", "stress")
        ),
        "historical_holdout_mode_000": all((path.stat().st_mode & 0o777) == 0 for path in p6e9.HOLDOUT_FILES),
    }
    OUTPUT.mkdir(parents=True)
    split_outputs = {}
    for split in SPLITS:
        path = OUTPUT / f"{split}.pt"
        p7._atomic_torch_save(path, _finish(caches[split], split, vision_sha))
        split_outputs[split] = {
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
            "rows": counts[(split, "MOVE", "sources")] + counts[(split, "KEEP", "sources")],
        }
    os.chmod(OUTPUT / "holdout.pt", 0)
    checks["new_holdout_mode_000"] = (OUTPUT / "holdout.pt").stat().st_mode & 0o777 == 0
    decision = "predicted_source_full80k_go" if all(checks.values()) else "predicted_source_full80k_no_go"
    audit = {
        "schema_version": "g3-p11e2-predicted-source-full80k-audit-0.1",
        "decision": decision,
        "checks": checks,
        "processed_images": processed,
        "counts": {str(key): value for key, value in sorted(counts.items())},
        "variants": {str(key): value for key, value in sorted(variants.items())},
        "scaffolds": {split: len(value) for split, value in split_scaffolds.items()},
        "total_move": total_move,
        "total_reachable_move": total_reachable,
        "move_reachability": total_reachable / total_move,
        "splits": split_outputs,
        "protocol": {
            "source": "real frozen P4-E3 top128 prediction",
            "keep_candidate": "explicit zero offset",
            "move_candidates": "P6-E1 target-free top8",
            "keep_sampling": "up to 8 matched sources only on images containing MOVE",
            "split": "scaffold hash 80/10/10",
            "training_performed": False,
            "holdout_mode_000": True,
            "confirm_external_sealed_read": False,
        },
        "manifest_audit": manifest_audit,
        "elapsed_seconds": time.perf_counter() - started,
        "historical_p7_holdout_read": False,
        "confirm_external_sealed_read": False,
    }
    p7._atomic_json(OUTPUT / "audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
