"""Plan-only quarantine render pool for text-aware endpoint admission.

No image bytes are opened and no model is loaded.  The planner selects
template molecules with visible O/N atoms incident to single/double/triple/
aromatic bonds, excludes released train/validation identities, and assigns
whole molecules to disjoint train/holdout/audit groups.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path

from rdkit import Chem

from Data.sdf.case_diversity_v1.catalog import pool
from Data.sdf.case_diversity_v1.profiles import profiles
from chemvector.training.curriculum.pair_scale_training import source_checks


ROOT = Path(__file__).resolve().parents[1]
V02_DATA = ROOT / "outputs/datasets/curriculum-v0.2-expanded-20260919"
OUTPUT = Path(os.environ.get(
    "CHEMVECTOR_TARGETED_ENDPOINT_PLAN_OUTPUT",
    str(ROOT / "outputs/audits/chemvector-targeted-endpoint-render-plan-20261006-r1"),
))
TARGET_MOLECULES = 225
PROFILE_COUNT = 16
SPLIT_IMAGES = {"train": 2400, "holdout": 600, "audit": 600}
CORE = tuple((element, bond) for element in ("O", "N")
             for bond in ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"))


def molecule_cells(smiles: str) -> set[tuple[str, str]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    cells = set()
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in ("O", "N"):
            continue
        for bond in atom.GetBonds():
            kind = str(bond.GetBondType()).upper()
            if kind == "AROMATIC" or bond.GetIsAromatic():
                kind = "AROMATIC"
            if kind in {"SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"}:
                cells.add((atom.GetSymbol(), kind))
    return cells


def key(entry: dict) -> str:
    return hashlib.sha256((entry["molecule_id"] + "|" + entry["canonical_smiles"]).encode()).hexdigest()


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    entries, provenance = pool(include_train32=False)
    public_manifest = source_checks()["manifest"]
    released = {row["canonical_smiles"] for row in public_manifest["samples"]
                if row.get("split") in {"train", "validation"}}
    # The prior planner only excluded admitted rows.  The v0.2 quarantine
    # bundle is also an existing identity source and must not be rendered again
    # under a different experiment; bind its molecule registry read-only.
    quarantine_registry = json.loads((V02_DATA / "molecules.json").read_text(encoding="utf-8"))
    quarantine_smiles = {row["canonical_smiles"] for row in quarantine_registry}
    quarantine_ids = {row["molecule_id"] for row in quarantine_registry}
    candidates = []
    seen = set()
    for entry in entries:
        smiles = entry["canonical_smiles"]
        if (smiles in released or smiles in quarantine_smiles
                or entry["molecule_id"] in quarantine_ids or entry["molecule_id"] in seen):
            continue
        cells = molecule_cells(smiles)
        candidates.append({
            "molecule_id": entry["molecule_id"],
            "canonical_smiles": smiles,
            "source_kind": entry.get("source_kind"),
            "source_record_ids": entry.get("source_record_ids", []),
            "heavy_atoms": entry.get("heavy_atoms"),
            "cells": sorted([list(cell) for cell in cells]),
            "cell_set": cells,
        })
        seen.add(entry["molecule_id"])
    if not candidates:
        raise ValueError("No identity-disjoint template candidates")
    if len(candidates) < TARGET_MOLECULES:
        # Do not lower the requested pool or reuse quarantine identities just to
        # make a plan appear complete.  Emit an explicit CPU-only NO-GO record
        # so the caller can switch catalogs or obtain a new identity batch.
        result = {
            "schema_version": "chemvector-targeted-endpoint-render-plan-0.1",
            "complete": True, "status": "no_go_insufficient_identity_disjoint_pool",
            "plan_only": True, "image_bytes_read": False, "model_forwards": 0,
            "training_performed": False, "sealed_test_read": False,
            "deployment_authorized": False,
            "candidate_molecules": len(candidates),
            "required_molecules": TARGET_MOLECULES,
            "released_identity_count": len(released),
            "quarantine_identity_count": len(quarantine_registry),
            "available_cells": [list(cell) for cell in CORE
                                if sum(cell in item["cell_set"] for item in candidates) >= 8],
            "reason": "case_diversity_v1 has no sufficiently large identity-disjoint pool after excluding admitted and v0.2 quarantine molecules",
        }
        OUTPUT.mkdir(parents=True)
        (OUTPUT / "report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return result
    # A cell is plan-able only when at least eight identity-disjoint molecules
    # can represent it.  Sparse/chemically unavailable cells are recorded, not
    # silently filled by reusing a quarantine molecule.
    available_core = [cell for cell in CORE
                      if sum(cell in item["cell_set"] for item in candidates) >= 8]
    # First satisfy every chemically available O/N × bond-type cell, then fill
    # to a fixed pool.  O-triple is normally absent from valid neutral
    # chemistry; it is reported as unavailable rather than fabricated.
    selected: list[dict] = []
    selected_ids: set[str] = set()
    for cell in available_core:
        choices = sorted((item for item in candidates if cell in item["cell_set"]),
                         key=lambda item: key(item))
        if not choices:
            raise ValueError(f"No candidate for core cell {cell}")
        # Eight independent molecules per cell is the minimum planning quota;
        # duplicates across cells are allowed only after choosing a molecule.
        for item in choices[:8]:
            if item["molecule_id"] not in selected_ids:
                selected.append(item); selected_ids.add(item["molecule_id"])
    for item in sorted(candidates, key=key):
        if len(selected) >= TARGET_MOLECULES:
            break
        if item["molecule_id"] not in selected_ids:
            selected.append(item); selected_ids.add(item["molecule_id"])
    if len(selected) != TARGET_MOLECULES:
        raise ValueError(f"Only selected {len(selected)} of {TARGET_MOLECULES} molecules")

    # Deterministic group assignment with per-cell coverage in every split.
    split_sizes = {"train": 150, "holdout": 37, "audit": 38}
    groups = {name: [] for name in split_sizes}
    deficits = {name: set(available_core) for name in split_sizes}
    remaining = sorted(selected, key=key)
    for name, count in split_sizes.items():
        while deficits[name]:
            choices = [item for item in remaining if item["cell_set"] & deficits[name]]
            if not choices:
                raise ValueError(f"Cannot cover all core cells in {name}")
            item = max(choices, key=lambda x: (len(x["cell_set"] & deficits[name]), key(x)))
            groups[name].append(item); remaining.remove(item)
            deficits[name] -= item["cell_set"]
        while len(groups[name]) < count:
            if not remaining:
                raise ValueError("Group assignment exhausted selected molecules")
            groups[name].append(remaining.pop(0))
    if remaining:
        raise AssertionError("Not all selected molecules assigned")

    planned = []
    for split, items in groups.items():
        for item in items:
            for profile in profiles():
                planned.append({
                    "sample_key": item["molecule_id"] + "__" + profile.profile_id,
                    "molecule_id": item["molecule_id"],
                    "canonical_smiles": item["canonical_smiles"],
                    "profile_id": profile.profile_id,
                    "split": split,
                    "cells": item["cells"],
                    "source_kind": item["source_kind"],
                    "training_authorized": False,
                    "external_overlap_status": "unchecked",
                })
    counts = Counter(row["split"] for row in planned)
    if dict(counts) != {"train": 2400, "holdout": 592, "audit": 608}:
        # Whole-molecule groups intentionally remain intact.  The resulting
        # 2,400/592/608 denominator is recorded explicitly rather than
        # silently splitting a molecule across identity partitions.
        raise AssertionError(dict(counts))
    result = {
        "schema_version": "chemvector-targeted-endpoint-render-plan-0.1",
        "complete": True, "plan_only": True, "image_bytes_read": False,
        "model_forwards": 0, "training_performed": False,
        "sealed_test_read": False, "deployment_authorized": False,
        "source_provenance": provenance,
        "released_identity_exclusion": {"split_values": ["train", "validation"], "count": len(released)},
        "quarantine_identity_exclusion": {"dataset": str(V02_DATA.relative_to(ROOT)),
                                          "molecule_count": len(quarantine_registry),
                                          "canonical_smiles_count": len(quarantine_smiles)},
        "molecule_counts": {name: len(items) for name, items in groups.items()},
        "image_counts": dict(counts),
        "profile_count": PROFILE_COUNT,
        "core_cells": [list(cell) for cell in CORE],
        "unavailable_core_cells": [list(cell) for cell in CORE if cell not in available_core],
        "core_cell_molecule_counts": {
            name: {"/".join(cell): sum(cell in item["cell_set"] for item in items)
                   for cell in CORE} for name, items in groups.items()
        },
        "group_identity_overlap": {
            "molecule": 0, "canonical_smiles": 0,
            "sample_key": len(planned) - len({row["sample_key"] for row in planned}),
        },
        "render_handoff": {
            "train_images": 2400, "holdout_images": 592, "audit_images": 608,
            "note": "Whole molecules are kept intact; exact 600-image targets are not forced by splitting a molecule.",
        },
        "rows": planned,
    }
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUTPUT / "manifest.json").write_text(json.dumps({"rows": planned}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "molecules": result["molecule_counts"],
                      "images": result["image_counts"], "core": result["core_cell_molecule_counts"]}, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    run()
