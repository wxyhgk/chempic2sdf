#!/usr/bin/env python3
"""Build an identity-disjoint aromatic attribute probe manifest.

The source r6 images are reused read-only.  Only molecule identities from its
original train partition are eligible, and the probe split is made anew from
those identities: balanced aromatic/non-aromatic train and holdout groups,
with all render profiles kept together.  This is a sidecar diagnostic, not a
production data release.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/datasets/chemvector-identity-aligned-line-first-targeted-diversity-r6"
OUT = ROOT / "outputs/audits/chemvector-aromatic-probe-split-r16"
TRAIN_AROMATIC = 48
TRAIN_CONTROL = 48
HOLDOUT_AROMATIC = 16
HOLDOUT_CONTROL = 16


def key(row: dict) -> str:
    return hashlib.sha256((row["molecule_id"] + "|" + row["canonical_smiles"]).encode()).hexdigest()


def molecule_kind(row: dict) -> str:
    graph = json.loads((SOURCE / "samples" / (row["sample_key"] + ".graph.json")).read_text())
    return "aromatic" if any(bool(b.get("aromatic")) or str(b.get("chemical_type")) == "AROMATIC" for b in graph["bonds"]) else "control"


def run() -> dict:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    rows = [json.loads(line) for line in (SOURCE / "samples.jsonl").read_text().splitlines() if line.strip()]
    train_rows = [row for row in rows if row.get("split") == "train"]
    by_id = {}
    for row in train_rows:
        by_id.setdefault(row["molecule_id"], row)
    groups = {"aromatic": [], "control": []}
    for row in by_id.values():
        groups[molecule_kind(row)].append(row)
    for group in groups.values():
        group.sort(key=key)
    need = {"aromatic": TRAIN_AROMATIC + HOLDOUT_AROMATIC, "control": TRAIN_CONTROL + HOLDOUT_CONTROL}
    if any(len(groups[name]) < count for name, count in need.items()):
        raise RuntimeError({name: (len(groups[name]), need[name]) for name in groups})
    selected = {}
    for name, group in groups.items():
        selected[name] = {
            "train": group[: (TRAIN_AROMATIC if name == "aromatic" else TRAIN_CONTROL)],
            "holdout": group[(TRAIN_AROMATIC if name == "aromatic" else TRAIN_CONTROL): (TRAIN_AROMATIC if name == "aromatic" else TRAIN_CONTROL) + (HOLDOUT_AROMATIC if name == "aromatic" else HOLDOUT_CONTROL)],
        }
    planned = []
    source_by_id = {row["molecule_id"]: row for row in train_rows}
    for split in ("train", "holdout"):
        for kind in ("aromatic", "control"):
            for molecule in selected[kind][split]:
                molecule_rows = [row for row in train_rows if row["molecule_id"] == molecule["molecule_id"]]
                for row in molecule_rows:
                    planned.append({
                        "sample_key": row["sample_key"],
                        "molecule_id": row["molecule_id"],
                        "canonical_smiles": row["canonical_smiles"],
                        "profile_id": row["profile_id"],
                        "split": split,
                        "kind": kind,
                        "source_dataset": str(SOURCE.relative_to(ROOT)),
                        "training_authorized": False,
                    })
    ids = {split: {row["molecule_id"] for row in planned if row["split"] == split} for split in ("train", "holdout")}
    result = {
        "schema_version": "chemvector-aromatic-probe-split-r16-0.1",
        "complete": True, "plan_only": True, "image_bytes_read": False,
        "model_forwards": 0, "training_performed": False, "gpu_used": False,
        "sealed_test_read": False, "production_checkpoint_written": False,
        "source_dataset": str(SOURCE.relative_to(ROOT)),
        "molecule_counts": {split: {kind: len(selected[kind][split]) for kind in ("aromatic", "control")} for split in ("train", "holdout")},
        "image_counts": dict(Counter(row["split"] for row in planned)),
        "identity_overlap": len(ids["train"] & ids["holdout"]),
        "profile_counts": {
            f"{split}:{profile}": count
            for (split, profile), count in Counter((row["split"], row["profile_id"]) for row in planned).items()
        },
        "rows": planned,
        "gate": {"identity_disjoint": not (ids["train"] & ids["holdout"]), "production_go": False},
    }
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "manifest.json").write_text(json.dumps({"rows": planned}, indent=2, ensure_ascii=False) + "\n")
    (OUT / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT), "molecule_counts": result["molecule_counts"], "image_counts": result["image_counts"], "gate": result["gate"]}, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    run()
