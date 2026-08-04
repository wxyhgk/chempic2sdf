#!/usr/bin/env python3
"""Audit whether one-center-per-cell targets can express adjacent Atom samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.atoms import train_g3_p1u_adjacent_atom_capacity as p1u
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import _atomic_json


DEFAULT_OUTPUT = Path("outputs/audits/chemvector-g3-p1u-target-resolution")
EXPECTED = {
    192: {"collision_images": 7, "lost_atoms": 7},
    384: {"collision_images": 3, "lost_atoms": 3},
    768: {"collision_images": 0, "lost_atoms": 0},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def occupancy_collisions(anchors: torch.Tensor, *, grid_size: int) -> dict:
    if anchors.ndim != 2 or anchors.size(1) != 2 or grid_size < 1:
        raise ValueError("Target-resolution occupancy inputs differ")
    cells = (anchors.detach().float() * grid_size).clamp(0, grid_size - 1e-4).long()
    identities = cells[:, 1] * grid_size + cells[:, 0]
    _, counts = identities.unique(return_counts=True)
    collisions = counts[counts > 1]
    return {
        "collision_cells": len(collisions),
        "lost_atoms": int((collisions - 1).sum()),
        "maximum_occupancy": int(counts.max()) if len(counts) else 0,
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError("P1-U target-resolution output directory is not empty")
    manifest, manifest_audit, source_audit = p1u.load_adjacent_manifest()
    collator = p1a.G3AtomCollator(maximum_atoms=128, image_size=768)
    grids = {
        grid: {"collision_images": 0, "lost_atoms": 0, "records": []}
        for grid in EXPECTED
    }
    for batch in p1a.iter_manifest_batches(manifest, collator, batch_size=4):
        for index, sample_key in enumerate(batch["keys"]):
            anchors = batch["atom_anchors"][index, batch["atom_mask"][index]]
            for grid, totals in grids.items():
                current = occupancy_collisions(anchors, grid_size=grid)
                if current["lost_atoms"]:
                    totals["collision_images"] += 1
                    totals["lost_atoms"] += current["lost_atoms"]
                    totals["records"].append(
                        {"sample_key": str(sample_key), **current}
                    )
    checks = {
        f"grid_{grid}_matches_expected": {
            key: grids[grid][key] for key in ("collision_images", "lost_atoms")
        }
        == expected
        for grid, expected in EXPECTED.items()
    }
    decision = (
        "stride1_required_for_lossless_one_center_targets"
        if all(checks.values())
        else "protocol_failed"
    )
    audit = {
        "schema_version": "g3-p1u-target-resolution-0.1",
        "decision": decision,
        "checks": checks,
        "source_manifest": manifest_audit,
        "source_audit_sha256": p1u.ADJACENT_AUDIT_SHA256,
        "grids": {str(grid): value for grid, value in grids.items()},
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    _atomic_json(args.output_dir / "audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
