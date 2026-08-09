#!/usr/bin/env python3
"""Build a KEEP-vs-MOVE pilot from real predicted top128 Atom sources."""

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
from chemvector.evaluation.audits.atoms import build_g3_p6e8_residual_cycle_data as p6e8
from chemvector.g3_structural_data import file_sha256
from chemvector.training.atoms import g3_p4_local_recovery_data as recovery_data
from chemvector.training.atoms import train_g3_p1t_continuous_stride2_center_field as p1t
from chemvector.training.atoms import train_g3_p6e3_structural_peak_selector as p6e3
from chemvector.training.atoms import train_g3_p6e9_structural_residual_arms as p6e9
from chemvector.training.atoms import train_g3_p7_pairwise_counterfactual as p7
from chemvector.training.atoms import train_g3_p9_stratified_pairwise_anchor as p9
from chemvector.training.atoms import train_g3_p4e9_local_recovery_pilot20k as p4e9
from chemvector.training.graph.scenes.g3_structural_data import load_g3_sample_manifest
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import _autocast_context


AUDIT_NAME = "chemvector-g3-p11e0-predicted-source-pilot"
OUTPUT = Path("outputs/audits") / AUDIT_NAME
CACHE = OUTPUT / "fit.pt"
PILOT_MANIFEST = Path("outputs/audits/chemvector-g3-p4e0-source-union-cycle-manifests/pilot_train.jsonl")
PILOT_MANIFEST_SHA256 = "22a03e2b2e37b41b7b298a1fec324fbe383e0f09c49e173a39c8d90347636daa"
ORACLE_ACTIONS = p4e9.ORACLE_ACTIONS
ORACLE_ACTIONS_SHA256 = "30aff390726d2e50388e45b37bd3710be19fab8b1d08a1e5fa81bfd3a4ce5819"
ORACLE_IMAGES = p4e9.ORACLE_IMAGES
ORACLE_IMAGES_SHA256 = "b1f6db4da1735e20113a76439c761d0cad3f024e0935aef9747832afde138a9a"
ORACLE_AUDIT = p4e9.ORACLE_AUDIT
ORACLE_AUDIT_SHA256 = "5dc3189f7901a42edcc0d97b27cd43c8e8bb47b33573db4605d560cce0179698"
IMAGE_SIZE = 768.0
TOP_K = 128
KEEP_PER_IMAGE = 8
BATCH_SIZE = 4
EXPECTED_MOVE = 264
MINIMUM_REACHABLE_MOVE = 180
MINIMUM_KEEP = 1_500
MINIMUM_KEEP_HARMFUL_PAIRS = 720
SEED = 20260894


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def select_keep_assignments(
    sample_key: str,
    assignments: list[int | None],
    excluded_sources: set[int],
    maximum: int = KEEP_PER_IMAGE,
) -> list[tuple[int, int]]:
    rows = [
        (gt_index, int(source_index))
        for gt_index, source_index in enumerate(assignments)
        if source_index is not None and int(source_index) not in excluded_sources
    ]
    rows.sort(
        key=lambda value: hashlib.sha256(
            f"p11:keep:{sample_key}:{value[0]}:{value[1]}".encode()
        ).hexdigest()
    )
    return rows[:maximum]


def _empty_cache() -> dict[str, list]:
    return defaultdict(list)


def _append(
    cache: dict[str, list],
    images: torch.Tensor,
    scores: torch.Tensor,
    offsets: torch.Tensor,
    logits: torch.Tensor,
    masks: torch.Tensor,
    target_offsets: list[torch.Tensor],
    metadata: list[dict],
) -> None:
    cache["image_crops"].append(images.cpu().half())
    cache["score_crops"].append(scores.cpu().half())
    cache["candidate_offsets"].append(offsets.cpu().half())
    cache["candidate_peak_logits"].append(logits.cpu().half())
    cache["candidate_mask"].append(masks.cpu())
    cache["offset_targets"].append(torch.stack(target_offsets).cpu().float())
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


def _finish(cache: dict[str, list]) -> dict:
    tensor_names = (
        "image_crops",
        "score_crops",
        "candidate_offsets",
        "candidate_peak_logits",
        "candidate_mask",
        "offset_targets",
    )
    output = {name: torch.cat(cache[name]) for name in tensor_names}
    output.update({name: value for name, value in cache.items() if name not in tensor_names})
    output.update(
        {
            "schema_version": "g3-p11e0-predicted-source-pilot-cache-0.1",
            "p6e1_checkpoint_sha256": p6e8.P6E1_CHECKPOINT_SHA256,
            "oracle_actions_sha256": ORACLE_ACTIONS_SHA256,
        }
    )
    return output


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"P11-E0 output already exists: {OUTPUT}")
    for path, expected in (
        (PILOT_MANIFEST, PILOT_MANIFEST_SHA256),
        (ORACLE_ACTIONS, ORACLE_ACTIONS_SHA256),
        (ORACLE_IMAGES, ORACLE_IMAGES_SHA256),
        (ORACLE_AUDIT, ORACLE_AUDIT_SHA256),
        (p6e8.P6E1_CHECKPOINT, p6e8.P6E1_CHECKPOINT_SHA256),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"P11-E0 input SHA256 differs: {path}")
    if any((path.stat().st_mode & 0o777) != 0 for path in p6e9.HOLDOUT_FILES):
        raise RuntimeError("P11-E0 historical holdout must remain sealed")
    if not torch.cuda.is_available():
        raise RuntimeError("P11-E0 requires CUDA")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda")
    actions = [
        row
        for row in _jsonl(ORACLE_ACTIONS)
        if row["route"] == "ADD_REFINE" and row["split"] == "fit"
    ]
    if len(actions) != EXPECTED_MOVE:
        raise RuntimeError("P11-E0 fit refine action count differs")
    actions_by_key: dict[str, list[dict]] = defaultdict(list)
    for row in actions:
        actions_by_key[str(row["sample_key"])].append(row)
    image_meta = {str(row["sample_key"]): row for row in _jsonl(ORACLE_IMAGES)}
    full_manifest, manifest_audit = load_g3_sample_manifest(
        PILOT_MANIFEST,
        expected_sha256=PILOT_MANIFEST_SHA256,
        expected_samples=20_000,
        expected_molecules=5_000,
        expected_variants=frozenset({"coordgen", "original", "rdkit", "stress"}),
    )
    manifest = [row for row in full_manifest if row.sample_key in actions_by_key]
    if len(manifest) != len(actions_by_key):
        raise RuntimeError("P11-E0 action image manifest is incomplete")
    payload = torch.load(p6e8.P6E1_CHECKPOINT, map_location="cpu", weights_only=False)
    peak_model = StructuralContextPeakHead().to(device)
    peak_model.load_state_dict(payload["model"], strict=True)
    peak_model.eval().requires_grad_(False)
    field, atom_head, vision, frozen_block, vision_sha = p4e6._load_model(device)
    collator = p6e8.ResidualCycleCollator(maximum_atoms=128, image_size=768)
    cache = _empty_cache()
    counts = Counter()
    variants = Counter()
    started = time.perf_counter()
    with torch.no_grad():
        for batch in p6e8.p1a.iter_manifest_batches(
            manifest, collator, batch_size=BATCH_SIZE, accepted_split="train"
        ):
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
                raw = peaks[batch_index]["anchors"].float().to(device)
                source_ids = peaks[batch_index]["source_ids"].long().to(device)
                expected = batch["atom_anchors"][batch_index][batch["atom_mask"][batch_index]].float().to(device)
                move_rows = actions_by_key[key]
                source_specs = []
                excluded = set()
                for action in move_rows:
                    candidate_index = int(action["candidate_index"])
                    excluded.add(candidate_index)
                    if int(source_ids[candidate_index]) != int(action["base_source_id"]):
                        raise RuntimeError("P11-E0 replay source identity differs")
                    target_index = int(action["target_index"])
                    source_specs.append(("MOVE", candidate_index, target_index))
                assignments = p4e6.maximum_cardinality_assignments(raw.cpu(), expected.cpu())
                for target_index, source_index in select_keep_assignments(key, assignments, excluded):
                    source_specs.append(("KEEP", source_index, target_index))
                sources = torch.stack([raw[source_index] for _, source_index, _ in source_specs])
                targets = torch.stack([expected[target_index] for _, _, target_index in source_specs])
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
                metadata = []
                meta = image_meta[key]
                for row, (action, source_index, target_index) in enumerate(source_specs):
                    reachable = bool((distances[row] <= 6.0).any())
                    harmful = int(((distances[row] > 6.0) & masks[row]).sum())
                    equivalent = int(((distances[row] <= 6.0) & masks[row]).sum())
                    counts[f"{action.lower()}_sources"] += 1
                    counts[f"{action.lower()}_reachable"] += int(reachable)
                    counts[f"{action.lower()}_harmful_pairs"] += harmful
                    counts[f"{action.lower()}_equivalent_pairs"] += equivalent
                    variants[(action, str(meta["variant"]))] += 1
                    metadata.append(
                        {
                            "sample_key": key,
                            "molecule_id": str(meta["molecule_id"]),
                            "scaffold": str(meta["scaffold"]),
                            "variant": str(meta["variant"]),
                            "action": action,
                            "target_role": batch["atom_roles"][batch_index][target_index],
                            "degree_bucket": p6e8.degree_bucket(batch["atom_degrees"][batch_index][target_index]),
                            "target_index": target_index,
                            "base_source_id": int(source_ids[source_index]),
                        }
                    )
                _append(cache, image_crops, score_crops, offsets, logits, masks, list(target_offsets), metadata)
            if sum(counts[name] for name in ("move_sources", "keep_sources")) % 500 < 32:
                print(json.dumps({"counts": dict(counts), "elapsed_seconds": time.perf_counter() - started}), flush=True)
    result = _finish(cache)
    checks = {
        "move_sources_264": counts["move_sources"] == EXPECTED_MOVE,
        "move_reachable_min180": counts["move_reachable"] >= MINIMUM_REACHABLE_MOVE,
        "keep_sources_min1500": counts["keep_sources"] >= MINIMUM_KEEP,
        "keep_harmful_pairs_min720": counts["keep_harmful_pairs"] >= MINIMUM_KEEP_HARMFUL_PAIRS,
        "four_move_variants": all(variants[("MOVE", name)] > 0 for name in ("coordgen", "original", "rdkit", "stress")),
        "explicit_keep_zero_offset": True,
        "historical_holdout_mode_000": all((path.stat().st_mode & 0o777) == 0 for path in p6e9.HOLDOUT_FILES),
    }
    decision = "predicted_source_pilot_go" if all(checks.values()) else "predicted_source_pilot_no_go"
    OUTPUT.mkdir(parents=True)
    p7._atomic_torch_save(CACHE, result)
    audit = {
        "schema_version": "g3-p11e0-predicted-source-pilot-audit-0.1",
        "decision": decision,
        "checks": checks,
        "counts": dict(sorted(counts.items())),
        "variants": {str(key): value for key, value in sorted(variants.items())},
        "cache": str(CACHE.resolve()),
        "cache_sha256": file_sha256(CACHE),
        "lineage": {
            "p6e1_checkpoint_sha256": p6e8.P6E1_CHECKPOINT_SHA256,
            "vision_checkpoint_sha256": vision_sha,
            "oracle_actions_sha256": ORACLE_ACTIONS_SHA256,
            "pilot_manifest_sha256": PILOT_MANIFEST_SHA256,
        },
        "protocol": {
            "source": "real frozen P4-E3 top128 prediction",
            "keep_candidate": "explicit zero offset",
            "move_candidates": "P6-E1 target-free top8",
            "fit_refine_actions_only": True,
            "training_performed": False,
            "historical_holdout_read": False,
            "confirm_external_sealed_read": False,
        },
        "manifest_audit": manifest_audit,
        "elapsed_seconds": time.perf_counter() - started,
        "historical_holdout_read": False,
        "confirm_external_sealed_read": False,
    }
    p7._atomic_json(OUTPUT / "audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
