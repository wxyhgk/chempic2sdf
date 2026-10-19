#!/usr/bin/env python3
"""Read-only complexity/strata audit for the t4b train/holdout split."""
from __future__ import annotations

import hashlib
import json
import os
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / os.environ.get("R17_COMPLEXITY_MANIFEST", "outputs/datasets/chemvector-r17-residual-density-source-20261021-t4b/manifest.json")
OUT = ROOT / os.environ.get("R17_COMPLEXITY_OUT", "outputs/audits/chemvector-r17-t4b-split-complexity-audit-v2/report.json")
SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    manifest = json.loads(MANIFEST.read_text())
    rows = manifest.get("rows", manifest)
    source = MANIFEST.parent / "samples"
    molecules = {}
    for row in rows:
        mid = row["molecule_id"]
        if mid in molecules:
            continue
        graph = json.loads((source / f"{row['sample_key']}.graph.json").read_text())
        bonds = graph.get("bonds", [])
        molecules[mid] = {
            "split": row["split"],
            "molecule_id": mid,
            "atoms": len(graph.get("atoms", [])),
            "bonds": len(bonds),
            "aromatic_bonds": sum(str(b.get("chemical_type", "")).upper() == "AROMATIC" for b in bonds),
            "has_aromatic": any(str(b.get("chemical_type", "")).upper() == "AROMATIC" for b in bonds),
            "ring": not str(row.get("legacy_scaffold", "")).startswith("ACYCLIC:"),
        }
    stats = {}
    for split in ("train", "holdout"):
        vals = [m for m in molecules.values() if m["split"] == split]
        stats[split] = {
            "molecules": len(vals),
            "atoms_mean": statistics.mean(m["atoms"] for m in vals),
            "atoms_median": statistics.median(m["atoms"] for m in vals),
            "bonds_mean": statistics.mean(m["bonds"] for m in vals),
            "bonds_median": statistics.median(m["bonds"] for m in vals),
            "aromatic_molecules": sum(m["has_aromatic"] for m in vals),
            "aromatic_fraction": sum(m["has_aromatic"] for m in vals) / max(len(vals), 1),
            "complexity_histogram": {"%d/%d/%d" % key: value for key, value in Counter((m["atoms"], m["bonds"], m["aromatic_bonds"]) for m in vals).items()},
            "molecule_ids": sorted(m["molecule_id"] for m in vals),
        }
    train_ids, hold_ids = set(stats["train"]["molecule_ids"]), set(stats["holdout"]["molecule_ids"])
    report = {
        "schema_version": "chemvector-r17-t4b-split-complexity-audit-0.1",
        "complete": True, "read_only": True, "training_performed": False,
        "gpu_used": False, "sealed_test_read": False, "production_checkpoint_written": False,
        "manifest": str(MANIFEST.relative_to(ROOT)), "manifest_sha256": digest(MANIFEST),
        "checkpoint_binding": SHA, "stats": stats,
        "gate": {
            "identity_disjoint": not (train_ids & hold_ids),
            "train_has_aromatic": stats["train"]["aromatic_molecules"] > 0,
            "holdout_has_aromatic": stats["holdout"]["aromatic_molecules"] > 0,
            "aromatic_fraction_gap_le_0.20": abs(stats["train"]["aromatic_fraction"] - stats["holdout"]["aromatic_fraction"]) <= 0.20,
            "mean_atom_gap_le_1.0": abs(stats["train"]["atoms_mean"] - stats["holdout"]["atoms_mean"]) <= 1.0,
            "complexity_matched_for_training": (
                not (train_ids & hold_ids)
                and stats["train"]["aromatic_molecules"] > 0
                and stats["holdout"]["aromatic_molecules"] > 0
                and abs(stats["train"]["aromatic_fraction"] - stats["holdout"]["aromatic_fraction"]) <= 0.20
                and abs(stats["train"]["atoms_mean"] - stats["holdout"]["atoms_mean"]) <= 1.0
            ),
            "production_go": False,
        },
        "next": "rebuild an identity-disjoint holdout stratified by aromatic/ring and atom/bond complexity before another type training run",
    }
    OUT.parent.mkdir(parents=True, exist_ok=False)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(OUT), "stats": stats, "gate": report["gate"]}, indent=2))


if __name__ == "__main__":
    run()
