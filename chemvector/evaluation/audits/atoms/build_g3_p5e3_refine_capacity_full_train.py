#!/usr/bin/env python3
"""Audit leakage-free refine-action capacity in the unused full-train remainder."""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from chemvector.components.atoms.objects.g3_atom_center_field import (
    decode_atom_center_field,
)
from chemvector.evaluation import g3_p4_source_union_cycle as cycle
from chemvector.evaluation.audits.atoms.audit_g3_p1n_regional_maxima import (
    extract_regional_maximum_objects,
)
from chemvector.evaluation.audits.atoms import (
    audit_g3_p4e6_missing_peak_sources as p4e6,
)
from chemvector.evaluation.audits.atoms import (
    build_g3_p4e9_recovery_oracle as oracle,
)
from chemvector.training.atoms import (
    train_g3_p1a_canonical_atom_selector as p1a,
)
from chemvector.training.atoms import (
    train_g3_p1t_continuous_stride2_center_field as p1t,
)
from chemvector.training.graph.scenes.g3_structural_data import (
    load_g3_sample_manifest,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _autocast_context,
)
from chemvector.g3_structural_data import file_sha256


AUDIT_NAME = "chemvector-g3-p5e3-refine-capacity-full-train"
OUTPUT = Path(f"outputs/audits/{AUDIT_NAME}")
FULL_MANIFEST = oracle.P4E0_ROOT / "full_train.jsonl"
FULL_MANIFEST_SHA256 = (
    "c1c787a646b5827ae72accd62dbe791ba2beb0a866d88741be3d79d26d114795"
)
PILOT_MANIFEST = oracle.PILOT_MANIFEST
PILOT_MANIFEST_SHA256 = oracle.PILOT_MANIFEST_SHA256
ORACLE_AUDIT = oracle.OUTPUT / "audit.json"
ORACLE_AUDIT_SHA256 = (
    "5dc3189f7901a42edcc0d97b27cd43c8e8bb47b33573db4605d560cce0179698"
)
CALIBRATION_MANIFEST = oracle.OUTPUT / "calibration.jsonl"
HOLDOUT_MANIFEST = oracle.OUTPUT / "holdout.jsonl"
EXPECTED_ELIGIBLE_IMAGES = 58_376
EXPECTED_ELIGIBLE_MOLECULES = 14_594
MINIMUM_REFINE_ACTIONS = 800
MINIMUM_REFINE_MOLECULES = 500
BATCH_SIZE = 8
SEED = 20260855


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"P5-E3 output already exists: {OUTPUT}")
    for path, expected in (
        (FULL_MANIFEST, FULL_MANIFEST_SHA256),
        (PILOT_MANIFEST, PILOT_MANIFEST_SHA256),
        (ORACLE_AUDIT, ORACLE_AUDIT_SHA256),
        (p4e6.P4E3_CHECKPOINT, p4e6.P4E3_CHECKPOINT_SHA256),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"P5-E3 input SHA256 differs: {path}")
    if not torch.cuda.is_available():
        raise RuntimeError("P5-E3 requires CUDA BF16")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    full_rows = _rows(FULL_MANIFEST)
    pilot_molecules = {row["molecule_id"] for row in _rows(PILOT_MANIFEST)}
    evaluation_rows = _rows(CALIBRATION_MANIFEST) + _rows(HOLDOUT_MANIFEST)
    blocked_scaffolds = {row["scaffold"] for row in evaluation_rows}
    eligible_rows = [
        row
        for row in full_rows
        if row["molecule_id"] not in pilot_molecules
        and row["scaffold"] not in blocked_scaffolds
    ]
    eligible_keys = {row["sample_key"] for row in eligible_rows}
    metadata = {row["sample_key"]: row for row in eligible_rows}
    if len(eligible_rows) != EXPECTED_ELIGIBLE_IMAGES:
        raise RuntimeError("P5-E3 eligible image count differs")
    if len({row["molecule_id"] for row in eligible_rows}) != EXPECTED_ELIGIBLE_MOLECULES:
        raise RuntimeError("P5-E3 eligible molecule count differs")
    if {row["scaffold"] for row in eligible_rows} & blocked_scaffolds:
        raise RuntimeError("P5-E3 evaluation scaffold leaked into capacity pool")

    full_manifest, full_audit = load_g3_sample_manifest(
        FULL_MANIFEST,
        expected_sha256=FULL_MANIFEST_SHA256,
        expected_samples=80_000,
        expected_molecules=cycle.FULL_TRAIN_MOLECULES,
        expected_variants=cycle.VARIANTS,
    )
    manifest = [row for row in full_manifest if row.sample_key in eligible_keys]
    if len(manifest) != EXPECTED_ELIGIBLE_IMAGES:
        raise RuntimeError("P5-E3 loaded manifest differs from eligible keys")

    device = torch.device("cuda")
    field, atom_head, vision, frozen_block, vision_sha = p4e6._load_model(device)
    collator = p1a.G3AtomCollator(maximum_atoms=128, image_size=oracle.IMAGE_SIZE)
    action_rows = []
    image_action_counts = []
    started = time.perf_counter()
    processed = 0
    with torch.no_grad():
        for batch in p1a.iter_manifest_batches(
            manifest, collator, batch_size=BATCH_SIZE, accepted_split="train"
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
                output = field(objects, stroke)
            peaks = extract_regional_maximum_objects(
                decode_atom_center_field(output), maximum_objects=oracle.TOP_K
            )
            for index, key_value in enumerate(batch["keys"]):
                sample_key = str(key_value)
                expected = batch["atom_anchors"][index][batch["atom_mask"][index]].cpu()
                actions = oracle.recovery_actions(
                    peaks[index]["anchors"].cpu(),
                    peaks[index]["source_ids"].cpu(),
                    expected,
                )
                refine = [row for row in actions if row["route"] == "ADD_REFINE"]
                meta = metadata[sample_key]
                for action in refine:
                    action.update(
                        {
                            "sample_key": sample_key,
                            "molecule_id": meta["molecule_id"],
                            "scaffold": meta["scaffold"],
                            "variant": meta["variant"],
                            "split": "capacity_train",
                        }
                    )
                    action_rows.append(action)
                image_action_counts.append(len(refine))
            processed += len(batch["keys"])
            if processed % 4_000 == 0:
                print(
                    json.dumps(
                        {
                            "processed": processed,
                            "refine_actions": len(action_rows),
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )

    variants = Counter(row["variant"] for row in action_rows)
    action_molecules = {row["molecule_id"] for row in action_rows}
    action_scaffolds = {row["scaffold"] for row in action_rows}
    checks = {
        "eligible_images_exact": processed == EXPECTED_ELIGIBLE_IMAGES,
        "eligible_molecules_exact": len(
            {row["molecule_id"] for row in eligible_rows}
        )
        == EXPECTED_ELIGIBLE_MOLECULES,
        "evaluation_scaffold_overlap_zero": not (
            action_scaffolds & blocked_scaffolds
        ),
        "refine_actions_min800": len(action_rows) >= MINIMUM_REFINE_ACTIONS,
        "refine_molecules_min500": len(action_molecules)
        >= MINIMUM_REFINE_MOLECULES,
        "all_four_variants": set(variants) == set(cycle.VARIANTS),
        "confirm_v2_not_read": True,
        "external_and_sealed_not_read": True,
    }
    decision = "scaled_refine_capacity_go" if all(checks.values()) else "scaled_refine_capacity_no_go"
    OUTPUT.mkdir(parents=True)
    oracle._atomic_jsonl(OUTPUT / "eligible_train.jsonl", eligible_rows)
    oracle._atomic_jsonl(OUTPUT / "refine_actions.jsonl", action_rows)
    audit = {
        "schema_version": "g3-p5e3-refine-capacity-full-train-0.1",
        "decision": decision,
        "checks": checks,
        "protocol": {
            "training_performed": False,
            "top_k": oracle.TOP_K,
            "match_radius_pixels": oracle.MATCH_RADIUS,
            "refine_radius_pixels": oracle.REFINE_RADIUS,
            "pilot_molecules_excluded": True,
            "pilot_calibration_holdout_scaffolds_excluded": True,
            "confirm_v2_read": False,
        },
        "eligible": {
            "images": processed,
            "molecules": len({row["molecule_id"] for row in eligible_rows}),
            "scaffolds": len({row["scaffold"] for row in eligible_rows}),
            "blocked_evaluation_scaffolds": len(blocked_scaffolds),
        },
        "refine": {
            "actions": len(action_rows),
            "images": sum(value > 0 for value in image_action_counts),
            "molecules": len(action_molecules),
            "scaffolds": len(action_scaffolds),
            "variants": dict(sorted(variants.items())),
            "maximum_actions_per_image": max(image_action_counts, default=0),
        },
        "lineage": {
            "full_manifest_sha256": FULL_MANIFEST_SHA256,
            "pilot_manifest_sha256": PILOT_MANIFEST_SHA256,
            "oracle_audit_sha256": ORACLE_AUDIT_SHA256,
            "p4e3_checkpoint_sha256": p4e6.P4E3_CHECKPOINT_SHA256,
            "vision_checkpoint_sha256": vision_sha,
            "full_manifest_audit": full_audit,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(OUTPUT / "audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
