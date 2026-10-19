"""Prepare a fixed 48/16 molecule targeted pilot from the v0.2 proposal.

This is CPU-only and diagnostic: it reads only the already reviewed quarantine
bundles, checks exact identity/style payloads, and writes a deterministic pilot
plan.  It does not promote data, train, or expose sealed evaluation content.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from rdkit import Chem

from chemvector.training.curriculum.data import load_sample

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs/datasets/curriculum-v0.2-expanded-20260919"
PREP = ROOT / "outputs/audits/curriculum-v02-release-preparation-20260919"
ROWS = PREP / "sample-split-proposal.json"
OUTPUT = Path(__import__("os").environ.get(
    "CHEMVECTOR_V02_TARGETED_PILOT_OUTPUT",
    str(ROOT / "outputs/audits/chemvector-v02-targeted-pilot-preflight-20261006-r1")))
ROWS_SHA = "f033e48b2194be5c41f3a0050509edd4684a285e05b39000f5a5513802d5cb28"
PROFILES = ("font_large", "serif")
CELLS = (("N", "SINGLE"), ("N", "TRIPLE"), ("O", "DOUBLE"), ("O", "AROMATIC"))


def digest(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cells(smiles: str) -> set[tuple[str, str]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    found = set()
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in {"N", "O"}:
            continue
        for bond in atom.GetBonds():
            kind = str(bond.GetBondType()).upper()
            if bond.GetIsAromatic() or kind == "AROMATIC":
                kind = "AROMATIC"
            if kind in {"SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"}:
                found.add((atom.GetSymbol(), kind))
    return found


def run(output: Path = OUTPUT) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if digest(ROWS) != ROWS_SHA:
        raise ValueError("Fixed v0.2 split proposal changed")
    rows = json.loads(ROWS.read_text(encoding="utf-8"))
    by_molecule: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("proposed_split") in {"train", "validation"}:
            by_molecule[row["molecule_id"]].append(row)
    candidates = {}
    for molecule_id, group in by_molecule.items():
        if {r["proposed_split"] for r in group} != {"train"} and {r["proposed_split"] for r in group} != {"validation"}:
            raise ValueError("Molecule crosses proposed train/validation split")
        profile_rows = {r["profile_id"]: r for r in group}
        if not set(PROFILES) <= set(profile_rows):
            continue
        smiles = group[0]["canonical_smiles"]
        candidates[molecule_id] = {
            "molecule_id": molecule_id, "canonical_smiles": smiles,
            "split": group[0]["proposed_split"], "cells": cells(smiles),
            "rows": {profile: profile_rows[profile] for profile in PROFILES},
        }
    selected: dict[str, list[dict]] = {"train": [], "validation": []}
    for split, count in (("train", 48), ("validation", 16)):
        pool = [r for r in candidates.values() if r["split"] == split]
        # First reserve a balanced quota for every requested chemistry cell,
        # then fill deterministically without reusing identities.
        chosen: dict[str, dict] = {}
        for cell in CELLS:
            options = sorted((r for r in pool if cell in r["cells"] and r["molecule_id"] not in chosen),
                             key=lambda r: hashlib.sha256(f"{split}:{cell}:{r['molecule_id']}".encode()).hexdigest())
            quota = 8 if split == "train" else 4
            for row in options[:quota]:
                chosen[row["molecule_id"]] = row
        if len(chosen) < count:
            for row in sorted(pool, key=lambda r: hashlib.sha256(f"{split}:{r['molecule_id']}".encode()).hexdigest()):
                if row["molecule_id"] not in chosen:
                    chosen[row["molecule_id"]] = row
                if len(chosen) >= count:
                    break
        if len(chosen) != count:
            raise ValueError(f"Only {len(chosen)} {split} molecules available")
        selected[split] = list(chosen.values())

    selected_rows = []
    quality = Counter()
    for split, molecules in selected.items():
        for molecule in molecules:
            for profile, row in molecule["rows"].items():
                sample = load_sample(DATA, row["sample_key"], purpose="diagnostic")
                meta = sample["meta"]
                if (meta["split"], meta["training_authorized"], meta["external_overlap_status"]) != ("quarantine", False, "unchecked"):
                    raise ValueError("Pilot source is not quarantine-only")
                if digest(DATA / "samples" / f"{row['sample_key']}.meta.json") != row["meta_sha256"]:
                    raise ValueError("Pilot metadata hash mismatch")
                quality[(split, profile)] += 1
                selected_rows.append({"sample_key": row["sample_key"], "molecule_id": row["molecule_id"],
                                     "canonical_smiles": row["canonical_smiles"], "profile_id": profile,
                                     "split": split, "cells": [list(c) for c in sorted(molecule["cells"])]})
    train_ids = {r["molecule_id"] for r in selected["train"]}
    val_ids = {r["molecule_id"] for r in selected["validation"]}
    report = {
        "schema_version": "chemvector-v02-targeted-pilot-preflight-0.1",
        "complete": True, "status": "passed_read_only_pilot_preflight",
        "training_authorized": False, "training_performed": False,
        "model_forwards": 0, "optimizer_steps": 0, "gpu_used": False,
        "sealed_test_read": False, "identity_overlap": {"molecule": len(train_ids & val_ids),
            "canonical_smiles": len({r['canonical_smiles'] for r in selected['train']} &
                                     {r['canonical_smiles'] for r in selected['validation']})},
        "molecule_counts": {"train": len(train_ids), "validation": len(val_ids)},
        "image_counts": {"train": len(train_ids) * 2, "validation": len(val_ids) * 2},
        "profile_counts": {f"{split}:{profile}": count for (split, profile), count in quality.items()},
        "cell_counts": {split: {"/".join(cell): sum(cell in r["cells"] for r in selected[split])
                                for cell in CELLS} for split in selected},
        "rows_sha256": ROWS_SHA, "source_dataset": str(DATA.relative_to(ROOT)),
        "rows": selected_rows,
        "gate": {"identity_disjoint": not (train_ids & val_ids),
                 "all_profiles_present": all(v == (48 if split == "train" else 16)
                                               for (split, _), v in quality.items()),
                 "training_admissible": False,
                 "next": "frozen initial replay before any publication or training"},
    }
    output.mkdir(parents=True)
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "molecule_counts", "image_counts", "cell_counts", "gate")}, indent=2))
    return report


if __name__ == "__main__":
    run()
