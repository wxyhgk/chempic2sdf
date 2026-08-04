#!/usr/bin/env python3
"""Build the fixed 32-image adjacent-Atom capacity manifest from train5k."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.graph.scenes.g3_structural_data import (
    OVERFIT32_MANIFEST,
    file_sha256,
    load_g3_overfit32_manifest,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import _atomic_json


DEFAULT_OUTPUT = Path("outputs/audits/chemvector-g3-adjacent-atom-capacity32")
EXPECTED_IMAGES = 32
EXPECTED_COLLISION_IMAGES = 7
GRID_SIZE = 192
IMAGE_SIZE = 768.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def sample_geometry(anchors: torch.Tensor) -> dict:
    anchors = anchors.detach().float().cpu()
    if anchors.ndim != 2 or anchors.size(1) != 2 or len(anchors) < 2:
        raise ValueError("Adjacent-Atom selection requires at least two Atom anchors")
    distances = torch.cdist(anchors, anchors) * IMAGE_SIZE
    distances.fill_diagonal_(float("inf"))
    flat = int(distances.argmin())
    first, second = divmod(flat, len(anchors))
    cells = (anchors * GRID_SIZE).clamp(0, GRID_SIZE - 1e-4).long()
    cell_ids = cells[:, 1] * GRID_SIZE + cells[:, 0]
    collision_cells = int(len(cell_ids) - len(cell_ids.unique()))
    return {
        "atom_count": len(anchors),
        "minimum_distance_pixels": float(distances[first, second]),
        "nearest_atom_indices": [min(first, second), max(first, second)],
        "same_stride4_cell": collision_cells > 0,
        "collision_excess_atoms": collision_cells,
    }


def select_capacity_records(records: list[dict], *, count: int = EXPECTED_IMAGES) -> list[dict]:
    if count < 1 or len(records) < count:
        raise ValueError("Adjacent-Atom selection has insufficient records")
    ordered = sorted(
        records,
        key=lambda item: (item["minimum_distance_pixels"], item["sample_key"]),
    )
    collisions = [item for item in ordered if item["same_stride4_cell"]]
    selected = collisions + [item for item in ordered if not item["same_stride4_cell"]]
    return selected[:count]


def _sequence_sha256(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def main() -> None:
    args = build_parser().parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError("Adjacent-Atom capacity output directory is not empty")
    train_manifest, train_audit = p1a.load_train_manifest()
    overfit_manifest, overfit_audit = load_g3_overfit32_manifest(OVERFIT32_MANIFEST)
    by_key = {record.sample_key: record for record in train_manifest}
    collator = p1a.G3AtomCollator(maximum_atoms=128, image_size=768)
    candidates = []
    for batch in p1a.iter_manifest_batches(train_manifest, collator, batch_size=4):
        for index, sample_key in enumerate(batch["keys"]):
            expected = batch["atom_anchors"][index, batch["atom_mask"][index]]
            geometry = sample_geometry(expected)
            source = by_key[str(sample_key)]
            candidates.append(
                {
                    "sample_key": source.sample_key,
                    "molecule_id": source.molecule_id,
                    "inchikey": source.inchikey,
                    "shard": source.shard,
                    "variant": source.variant,
                    "source_index": len(candidates),
                    **geometry,
                }
            )
    collision_count = sum(item["same_stride4_cell"] for item in candidates)
    if collision_count != EXPECTED_COLLISION_IMAGES:
        raise RuntimeError(
            "Train5k same-stride4-cell collision count differs: "
            f"{collision_count} != {EXPECTED_COLLISION_IMAGES}"
        )
    selected = select_capacity_records(candidates)
    if len(selected) != EXPECTED_IMAGES:
        raise RuntimeError("Adjacent-Atom capacity selection is incomplete")
    selected_collision_count = sum(item["same_stride4_cell"] for item in selected)
    if selected_collision_count != EXPECTED_COLLISION_IMAGES:
        raise RuntimeError("Adjacent-Atom capacity selection omitted a collision image")
    for selection_rank, item in enumerate(selected):
        item["selection_rank"] = selection_rank
    selected.sort(key=lambda item: item["source_index"])

    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = args.output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for item in selected:
            handle.write(
                json.dumps(
                    {
                        key: item[key]
                        for key in (
                            "sample_key", "molecule_id", "inchikey", "shard", "variant"
                        )
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    overfit_keys = {record.sample_key for record in overfit_manifest}
    selected_keys = [item["sample_key"] for item in selected]
    audit = {
        "schema_version": "g3-adjacent-atom-capacity32-0.1",
        "status": "passed",
        "selection": "all_stride4_collisions_then_minimum_distance",
        "manifest_order": "source_train5k_order",
        "image_count": len(selected),
        "molecule_count": len({item["molecule_id"] for item in selected}),
        "collision_images": selected_collision_count,
        "minimum_distance_pixels": min(item["minimum_distance_pixels"] for item in selected),
        "maximum_selected_distance_pixels": max(
            item["minimum_distance_pixels"] for item in selected
        ),
        "overfit32_overlap_count": len(set(selected_keys) & overfit_keys),
        "overfit32_overlap_keys": sorted(set(selected_keys) & overfit_keys),
        "sample_keys_sha256": _sequence_sha256(selected_keys),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": file_sha256(manifest_path),
        "source_train5k": train_audit,
        "source_overfit32": overfit_audit,
        "records": selected,
    }
    _atomic_json(args.output_dir / "audit.json", audit)
    print(json.dumps({key: audit[key] for key in (
        "status", "image_count", "molecule_count", "collision_images",
        "minimum_distance_pixels", "maximum_selected_distance_pixels",
        "overfit32_overlap_count", "manifest_sha256",
    )}, indent=2))


if __name__ == "__main__":
    main()
