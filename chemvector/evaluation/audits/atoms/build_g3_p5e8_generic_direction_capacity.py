#!/usr/bin/env python3
"""Build a leakage-free capacity manifest for generic source-to-Atom directions."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import torch

from chemvector.evaluation import g3_p4_source_union_cycle as cycle
from chemvector.evaluation.audits.atoms import (
    build_g3_p4e9_recovery_oracle as oracle,
)
from chemvector.training.atoms import (
    train_g3_p1a_canonical_atom_selector as p1a,
)
from chemvector.training.graph.scenes.g3_structural_data import (
    load_g3_sample_manifest,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
)
from chemvector.g3_structural_data import file_sha256
from chemvector.data import structure_labels


AUDIT_NAME = "chemvector-g3-p5e8-generic-direction-capacity"
OUTPUT = Path(f"outputs/audits/{AUDIT_NAME}")
FIT_MANIFEST = oracle.OUTPUT / "fit.jsonl"
FIT_MANIFEST_SHA256 = (
    "69d7d5c7555041039e83d975f10680b8b1c94e8cc1df65a8d7bfa446b7a4c32f"
)
ORACLE_AUDIT = oracle.OUTPUT / "audit.json"
ORACLE_AUDIT_SHA256 = (
    "5dc3189f7901a42edcc0d97b27cd43c8e8bb47b33573db4605d560cce0179698"
)
IMAGE_SIZE = 768
CROP_RADIUS_PIXELS = 16.0
MINIMUM_RADIUS_PIXELS = 6.0
MAXIMUM_RADIUS_PIXELS = 12.0
UNIQUE_NEAREST_MARGIN_PIXELS = 1.0
SELECTED_PAIRS = 50_000
PAIRS_PER_VARIANT = 12_500
MINIMUM_DEGREE2_PAIRS = 15_000
MINIMUM_TEXT_PAIRS = 5_000
SEED = 20260857


def deterministic_offset(sample_key: str, target_index: int) -> torch.Tensor:
    digest = hashlib.sha256(
        f"{SEED}:{sample_key}:{target_index}".encode("utf-8")
    ).digest()
    angle_value = int.from_bytes(digest[:8], "big") / float(2**64)
    radius_value = int.from_bytes(digest[8:16], "big") / float(2**64)
    angle = angle_value * 2.0 * math.pi
    radius = MINIMUM_RADIUS_PIXELS + (MAXIMUM_RADIUS_PIXELS - MINIMUM_RADIUS_PIXELS) * radius_value
    radius = max(radius, math.nextafter(MINIMUM_RADIUS_PIXELS, math.inf))
    return torch.tensor(
        [math.cos(angle) * radius / IMAGE_SIZE, math.sin(angle) * radius / IMAGE_SIZE],
        dtype=torch.float32,
    )


def degree_bucket(value: int) -> str:
    if value <= 1:
        return "degree1"
    if value == 2:
        return "degree2"
    return "degree3plus"


def _selection_hash(row: dict) -> str:
    return hashlib.sha256(
        f"{SEED}:select:{row['sample_key']}:{row['target_index']}".encode()
    ).hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"P5-E8 output already exists: {OUTPUT}")
    for path, expected in (
        (FIT_MANIFEST, FIT_MANIFEST_SHA256),
        (ORACLE_AUDIT, ORACLE_AUDIT_SHA256),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"P5-E8 input SHA256 differs: {path}")
    manifest, manifest_audit = load_g3_sample_manifest(
        FIT_MANIFEST,
        expected_sha256=FIT_MANIFEST_SHA256,
        expected_samples=14_000,
        expected_molecules=3_500,
        expected_variants=cycle.VARIANTS,
    )
    pools: dict[str, list[dict]] = {variant: [] for variant in cycle.VARIANTS}
    rejected = Counter()
    total_atoms = 0
    for sample in p1a.iter_manifest_samples(manifest, accepted_split="train"):
        labels = structure_labels(sample)
        atoms = sorted(labels["atoms"], key=lambda item: int(item["atom_id"]))
        anchors = torch.tensor(p1a.G3AtomCollator.atom_anchors(sample), dtype=torch.float32)
        atom_id_to_index = {int(atom["atom_id"]): index for index, atom in enumerate(atoms)}
        degrees = [0] * len(atoms)
        for bond in labels["bonds"]:
            for name in ("begin_atom", "end_atom"):
                atom_id = int(bond[name])
                if atom_id in atom_id_to_index:
                    degrees[atom_id_to_index[atom_id]] += 1
        sample_key = str(sample["key"])
        variant = str(sample["render"]["variant"])
        scaffold = str(sample["meta"]["scaffold"])
        total_atoms += len(atoms)
        for target_index, (atom, target) in enumerate(zip(atoms, anchors, strict=True)):
            target_offset = deterministic_offset(sample_key, target_index)
            source = target - target_offset
            source_pixels = source * IMAGE_SIZE
            if bool(
                (source_pixels < CROP_RADIUS_PIXELS).any()
                or (source_pixels > IMAGE_SIZE - CROP_RADIUS_PIXELS).any()
            ):
                rejected["crop_outside_canvas"] += 1
                continue
            distances = torch.linalg.vector_norm((anchors - source) * IMAGE_SIZE, dim=1)
            order = distances.argsort(stable=True)
            if int(order[0]) != target_index:
                rejected["another_gt_nearest"] += 1
                continue
            if len(order) > 1 and float(distances[order[1]] - distances[order[0]]) < UNIQUE_NEAREST_MARGIN_PIXELS:
                rejected["nearest_margin_lt1px"] += 1
                continue
            row = {
                "sample_key": sample_key,
                "molecule_id": str(sample["chemmol"]["molecule_id"]),
                "scaffold": scaffold,
                "variant": variant,
                "target_index": target_index,
                "target_atom_id": int(atom["atom_id"]),
                "target_anchor": [float(value) for value in target],
                "source_anchor": [float(value) for value in source],
                "offset_target": [float(value) for value in target_offset],
                "radius_pixels": float(torch.linalg.vector_norm(target_offset) * IMAGE_SIZE),
                "target_role": "TEXT" if atom["visible_text"] else "IMPLICIT",
                "target_element": str(atom["element"]),
                "target_degree": degrees[target_index],
                "degree_bucket": degree_bucket(degrees[target_index]),
            }
            pools[variant].append(row)

    selected = []
    pool_summary = {}
    for variant in sorted(pools):
        rows = sorted(pools[variant], key=lambda row: (_selection_hash(row), row["sample_key"], row["target_index"]))
        pool_summary[variant] = len(rows)
        selected.extend(rows[:PAIRS_PER_VARIANT])
    selected.sort(key=lambda row: (row["sample_key"], row["target_index"]))
    variants = Counter(row["variant"] for row in selected)
    degrees = Counter(row["degree_bucket"] for row in selected)
    roles = Counter(row["target_role"] for row in selected)
    checks = {
        "selected_pairs_50000": len(selected) == SELECTED_PAIRS,
        "variants_exact_12500_each": all(
            variants[variant] == PAIRS_PER_VARIANT for variant in cycle.VARIANTS
        ),
        "degree2_pairs_min15000": degrees["degree2"] >= MINIMUM_DEGREE2_PAIRS,
        "text_pairs_min5000": roles["TEXT"] >= MINIMUM_TEXT_PAIRS,
        "all_offsets_in_6_12px": all(
            MINIMUM_RADIUS_PIXELS < row["radius_pixels"] <= MAXIMUM_RADIUS_PIXELS
            for row in selected
        ),
        "fit_manifest_scaffold_isolated_by_p4e9": True,
        "dev_confirm_external_sealed_not_read": True,
    }
    decision = "generic_direction_capacity_go" if all(checks.values()) else "generic_direction_capacity_no_go"
    OUTPUT.mkdir(parents=True)
    oracle._atomic_jsonl(OUTPUT / "selected_pairs.jsonl", selected)
    audit = {
        "schema_version": "g3-p5e8-generic-direction-capacity-0.1",
        "decision": decision,
        "checks": checks,
        "protocol": {
            "training_performed": False,
            "source_generation": "sha256 continuous angle and radius in (6,12]px",
            "crop_radius_pixels": CROP_RADIUS_PIXELS,
            "unique_nearest_margin_pixels": UNIQUE_NEAREST_MARGIN_PIXELS,
            "selection": "sha256 first 12500 per variant",
            "dev_confirm_external_sealed_read": False,
        },
        "capacity": {
            "images": len(manifest),
            "molecules": len({row.molecule_id for row in manifest}),
            "total_atoms": total_atoms,
            "valid_pairs": sum(pool_summary.values()),
            "pool_by_variant": pool_summary,
            "rejected": dict(sorted(rejected.items())),
        },
        "selected": {
            "pairs": len(selected),
            "molecules": len({row["molecule_id"] for row in selected}),
            "scaffolds": len({row["scaffold"] for row in selected}),
            "variants": dict(sorted(variants.items())),
            "degree_buckets": dict(sorted(degrees.items())),
            "roles": dict(sorted(roles.items())),
            "elements": dict(sorted(Counter(row["target_element"] for row in selected).items())),
        },
        "lineage": {
            "fit_manifest_sha256": FIT_MANIFEST_SHA256,
            "oracle_audit_sha256": ORACLE_AUDIT_SHA256,
            "manifest_audit": manifest_audit,
        },
    }
    _atomic_json(OUTPUT / "audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
