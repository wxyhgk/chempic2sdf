"""Plan a small identity-disjoint targeted replay slice from structure expansion.

This is a metadata-only planner.  It deliberately does not open PNG/SDF/layer
payloads, load a model, publish a release, or authorize training.  The source
batch is quarantine-only; the resulting plan is therefore only an input to a
later release/identity admission and frozen-initial replay.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path

from rdkit import Chem


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs/datasets/structure-expansion-batch-12272-20260923"
REPLAY = ROOT / "outputs/audits/chemvector-structure-expansion-targeted-replay-20261006-r2/report.json"
RELEASE = ROOT / "outputs/datasets/chemvector-expanded-release-20260915/manifest.json"
V02 = ROOT / "outputs/datasets/curriculum-v0.2-expanded-20260919/molecules.json"
OUTPUT = Path(os.environ.get(
    "CHEMVECTOR_TARGETED_SPLIT_OUTPUT",
    str(ROOT / "outputs/audits/chemvector-structure-expansion-targeted-split-plan-20261006-r1"),
))
PROFILES = ("oa11", "oa13")
TARGET_CELLS = (
    ("O", "AROMATIC"),
    ("O", "DOUBLE"),
    ("O", "SINGLE"),
    ("N", "AROMATIC"),
    ("N", "SINGLE"),
)
TRAIN_MOLECULES = 49
HOLDOUT_MOLECULES = 16


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def identity_cells(smiles: str) -> set[tuple[str, str]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid canonical SMILES: {smiles}")
    cells: set[tuple[str, str]] = set()
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in {"O", "N"}:
            continue
        for bond in atom.GetBonds():
            kind = "AROMATIC" if bond.GetIsAromatic() else str(bond.GetBondType()).upper()
            if kind in {"SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"}:
                cells.add((atom.GetSymbol(), kind))
    return cells


def connection_identity(smiles: str) -> str:
    """Return the batch's stereo/isotope-independent connectivity key."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid canonical SMILES: {smiles}")
    for atom in mol.GetAtoms():
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        atom.SetIsotope(0)
    for bond in mol.GetBonds():
        bond.SetStereo(Chem.BondStereo.STEREONONE)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def load() -> tuple[list[dict], list[dict], dict]:
    molecules = json.loads((DATA / "molecules.json").read_text(encoding="utf-8"))
    samples = json.loads((DATA / "samples.json").read_text(encoding="utf-8"))
    replay = json.loads(REPLAY.read_text(encoding="utf-8"))
    if replay.get("training_performed") or replay.get("sealed_test_read"):
        raise ValueError("Replay receipt is not diagnostic-only")
    if any(row.get("split") != "quarantine" or row.get("training_authorized") is not False
           for row in molecules):
        raise ValueError("Structure expansion source is not quarantine-only")
    # The release and v0.2 manifests are metadata-only identity references.
    released = json.loads(RELEASE.read_text(encoding="utf-8"))
    v02 = json.loads(V02.read_text(encoding="utf-8"))
    released_smiles = {row["canonical_smiles"] for row in released.get("samples", [])}
    v02_smiles = {row["canonical_smiles"] for row in v02}
    blocked = released_smiles | v02_smiles
    if any(row["canonical_smiles"] in blocked for row in molecules):
        raise ValueError("Structure expansion candidate overlaps released/v0.2 canonical identity")
    blocked_groups = {connection_identity(smiles) for smiles in blocked}
    return molecules, samples, {
        "replay": replay,
        "blocked_count": len(blocked),
        "blocked_groups": blocked_groups,
    }


def choose(entries: list[dict], n: int, *, forbidden_groups: set[str], minimum: int) -> list[dict]:
    """Greedily satisfy per-cell minima, then fill deterministically."""
    chosen: list[dict] = []
    chosen_ids: set[str] = set()
    available = [e for e in entries if e["group"] not in forbidden_groups]

    def count(cell: tuple[str, str]) -> int:
        return sum(cell in row["cells"] for row in chosen)

    # One stable representative per connection identity is enforced by the
    # caller.  Select enough rows to cover every chemically available target
    # cell, with a non-trivial holdout quota per cell.
    for cell in TARGET_CELLS:
        options = sorted((e for e in available if cell in e["cells"]),
                         key=lambda e: stable_key(e["molecule_id"]))
        while count(cell) < minimum:
            candidate = next((e for e in options if e["molecule_id"] not in chosen_ids), None)
            if candidate is None:
                raise ValueError(f"Unable to satisfy target cell {cell}")
            chosen.append(candidate)
            chosen_ids.add(candidate["molecule_id"])
    while len(chosen) < n:
        options = [e for e in available if e["molecule_id"] not in chosen_ids]
        if not options:
            raise ValueError("Insufficient identity-disjoint candidates")
        candidate = max(options, key=lambda e: (
            len(set(TARGET_CELLS) & e["cells"]),
            stable_key(e["molecule_id"]),
        ))
        chosen.append(candidate)
        chosen_ids.add(candidate["molecule_id"])
    return chosen


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    molecules, samples, context = load()
    by_id = {row["molecule_id"]: row for row in molecules}
    replay_ids = {row["molecule_id"] for row in context["replay"].get("records", [])}
    replay_groups = {
        by_id[molecule_id]["connection_identity_without_isotope_stereo"]
        for molecule_id in replay_ids if molecule_id in by_id
    }
    by_sample = {(row["molecule_id"], row["profile_id"]): row for row in samples
                 if row.get("profile_id") in PROFILES}
    candidates = []
    for molecule_id, molecule in by_id.items():
        if (molecule_id in replay_ids
                or molecule["connection_identity_without_isotope_stereo"] in replay_groups
                or molecule["connection_identity_without_isotope_stereo"] in context["blocked_groups"]):
            continue
        rows = [by_sample.get((molecule_id, profile)) for profile in PROFILES]
        if any(row is None for row in rows):
            continue
        # Require an explicit visible O/N label in both selected styles.  This
        # is metadata from the renderer, not an image inspection.
        labels = [set(row.get("visible", {}).get("labels", {})) for row in rows]
        cells = identity_cells(molecule["canonical_smiles"])
        cells = {cell for cell in cells if all(cell[0] in present for present in labels)}
        cells &= set(TARGET_CELLS)
        if not cells:
            continue
        candidates.append({
            "molecule_id": molecule_id,
            "canonical_smiles": molecule["canonical_smiles"],
            "group": molecule["connection_identity_without_isotope_stereo"],
            "cells": cells,
            "sample_rows": rows,
        })

    # Collapse stereo/isotope variants by connection identity before splitting;
    # this prevents a near-duplicate group crossing train and holdout.
    representatives: dict[str, dict] = {}
    for candidate in candidates:
        group = candidate["group"]
        old = representatives.get(group)
        if old is None or stable_key(candidate["molecule_id"]) < stable_key(old["molecule_id"]):
            representatives[group] = candidate
    entries = sorted(representatives.values(), key=lambda e: stable_key(e["molecule_id"]))
    if len(entries) < TRAIN_MOLECULES + HOLDOUT_MOLECULES:
        raise ValueError("Insufficient identity-disjoint two-profile candidates")

    # Holdout first, so all train groups are disjoint by construction.  Four
    # per cell in holdout and eight per cell in train are fixed gates.
    holdout = choose(entries, HOLDOUT_MOLECULES, forbidden_groups=set(), minimum=4)
    holdout_ids = {row["molecule_id"] for row in holdout}
    holdout_groups = {row["group"] for row in holdout}
    train = choose(entries, TRAIN_MOLECULES, forbidden_groups=holdout_groups, minimum=8)
    train_ids = {row["molecule_id"] for row in train}

    if train_ids & holdout_ids or {row["group"] for row in train} & holdout_groups:
        raise AssertionError("Train/holdout identity overlap")
    counts = {
        "train": {"molecules": len(train), "groups": len({row["group"] for row in train}),
                   "cells": {"/".join(c): sum(c in row["cells"] for row in train)
                             for c in TARGET_CELLS}},
        "holdout": {"molecules": len(holdout), "groups": len(holdout_groups),
                    "cells": {"/".join(c): sum(c in row["cells"] for row in holdout)
                              for c in TARGET_CELLS}},
    }
    if any(value < 8 for value in counts["train"]["cells"].values()):
        raise AssertionError("Train cell quota failed")
    if any(value < 4 for value in counts["holdout"]["cells"].values()):
        raise AssertionError("Holdout cell quota failed")

    planned = []
    for split, rows in (("train", train), ("holdout", holdout)):
        for row in rows:
            for sample in row["sample_rows"]:
                planned.append({
                    "sample_key": sample["sample_key"],
                    "molecule_id": row["molecule_id"],
                    "canonical_smiles": row["canonical_smiles"],
                    "connection_identity_without_isotope_stereo": row["group"],
                    "profile_id": sample["profile_id"],
                    "split": split,
                    "cells": [list(cell) for cell in sorted(row["cells"])],
                    "png_sha256": sample["png_sha256"],
                    "rgb_sha256": sample["rgb_sha256"],
                })
    planned.sort(key=lambda row: row["sample_key"])
    source_hashes = {str(path.relative_to(ROOT)): digest(path) for path in (
        DATA / "molecules.json", DATA / "samples.json", DATA / "summary.json", REPLAY,
        RELEASE, V02,
    )}
    report = {
        "schema_version": "chemvector-structure-expansion-targeted-split-plan-0.1",
        "complete": True,
        "status": "plan_only_quarantine",
        "read_only": True,
        "image_bytes_read": False,
        "model_forwards": 0,
        "optimizer_steps": 0,
        "training_performed": False,
        "training_authorized": False,
        "sealed_test_read": False,
        "deployment_authorized": False,
        "source": {
            "dataset": str(DATA.relative_to(ROOT)),
            "replay_report": str(REPLAY.relative_to(ROOT)),
            "source_hashes": source_hashes,
            "candidate_molecules_after_replay_exclusion": len(candidates),
            "connection_identity_representatives": len(entries),
            "replay_excluded_molecules": len(replay_ids),
            "replay_excluded_connection_groups": len(replay_groups),
            "released_v02_blocked_connection_groups": len(context["blocked_groups"]),
            "released_v02_blocked_canonical_count": context["blocked_count"],
        },
        "selection": {
            "profiles": list(PROFILES),
            "target_cells": [list(c) for c in TARGET_CELLS],
            "molecules": TRAIN_MOLECULES + HOLDOUT_MOLECULES,
            "images": len(planned),
            "train_images": TRAIN_MOLECULES * len(PROFILES),
            "holdout_images": HOLDOUT_MOLECULES * len(PROFILES),
            "previous_replay_molecules_excluded": True,
        },
        "counts": counts,
        "identity_split": {
            "train_holdout_molecule_overlap": 0,
            "train_holdout_connection_identity_overlap": 0,
            "released_or_v02_canonical_overlap": 0,
            "sealed_access": False,
        },
        "gate": {
            "metadata_plan_pass": True,
            "next_required": [
                "exact-byte release/admission audit (still quarantine)",
                "frozen-initial replay with raw/selected endpoint strata",
                "strict selector-drop positives >= 50 train / >= 14 holdout",
                "baseline prefix retention == 100%",
                "strict residual precision >= 0.85",
                "only then one bounded shadow training trial",
            ],
            "training_admissible": False,
            "reason": "planning artifact only; no release or model replay performed",
        },
        "rows": planned,
    }
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "manifest.json").write_text(json.dumps({"rows": planned}, indent=2, ensure_ascii=False) + "\n")
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUTPUT), "counts": counts, "images": len(planned)}, indent=2))
    return report


if __name__ == "__main__":
    run()
