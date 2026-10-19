#!/usr/bin/env python3
"""Plan a structure-disjoint aromatic curriculum from v0.2 quarantine images.

The planner reads molecule metadata and existing rendered labels only.  It does
not read sealed data, run model inference, or authorize training.  Molecules
are grouped by Murcko scaffold (ring systems) or non-isomeric connectivity
(acyclic controls) before train/dev/fresh assignment, preventing near-duplicate
scaffolds from leaking across splits.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/datasets/curriculum-v0.2-expanded-20260919"
R6 = ROOT / "outputs/datasets/chemvector-identity-aligned-line-first-targeted-diversity-r6"
OUT = ROOT / "outputs/audits/chemvector-aromatic-curriculum-r17-plan"
PROFILES = ("reference", "font_large", "bond_short", "serif")
TARGET_GROUPS = {"train": 256, "dev": 64, "fresh": 128}
MINIMA = {
    "train": {"AROMATIC_CARBON": 64, "HETEROAROMATIC_NSP": 64, "AROMATIC_S": 32, "FUSED_AROMATIC": 64,
               "AROMATIC_MIXED_MULTIPLE": 64, "SINGLE_CONTROL": 48, "DOUBLE_CONTROL": 48, "TRIPLE_CONTROL": 32},
    "dev": {"AROMATIC_CARBON": 16, "HETEROAROMATIC_NSP": 16, "AROMATIC_S": 8, "FUSED_AROMATIC": 16,
            "AROMATIC_MIXED_MULTIPLE": 16, "SINGLE_CONTROL": 12, "DOUBLE_CONTROL": 12, "TRIPLE_CONTROL": 8},
    "fresh": {"AROMATIC_CARBON": 32, "HETEROAROMATIC_NSP": 32, "AROMATIC_S": 16, "FUSED_AROMATIC": 32,
              "AROMATIC_MIXED_MULTIPLE": 32, "SINGLE_CONTROL": 24, "DOUBLE_CONTROL": 24, "TRIPLE_CONTROL": 16},
}


def stable(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def connectivity(mol) -> str:
    clone = Chem.Mol(mol)
    Chem.RemoveStereochemistry(clone)
    return Chem.MolToSmiles(clone, canonical=True, isomericSmiles=False)


def fused_aromatic(mol) -> bool:
    rings = [set(r) for r in Chem.GetSymmSSSR(mol)]
    aromatic = [ring for ring in rings if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)]
    return any(len(a & b) >= 2 for i, a in enumerate(aromatic) for b in aromatic[i + 1:])


def descriptors(row: dict) -> set[str]:
    mol = Chem.MolFromSmiles(row["canonical_smiles"])
    if mol is None:
        return set()
    aromatic_atoms = [a for a in mol.GetAtoms() if a.GetIsAromatic()]
    aromatic_bonds = [b for b in mol.GetBonds() if b.GetIsAromatic() or str(b.GetBondType()).upper() == "AROMATIC"]
    nonaromatic = [str(b.GetBondType()).upper() for b in mol.GetBonds() if not b.GetIsAromatic()]
    labels: set[str] = set()
    if aromatic_atoms:
        labels.add("AROMATIC_CARBON")
        if any(a.GetSymbol() in {"N", "S", "P"} for a in aromatic_atoms):
            labels.add("HETEROAROMATIC_NSP")
        if any(a.GetSymbol() == "S" for a in aromatic_atoms):
            labels.add("AROMATIC_S")
        if fused_aromatic(mol):
            labels.add("FUSED_AROMATIC")
        if any(kind in {"DOUBLE", "TRIPLE"} for kind in nonaromatic):
            labels.add("AROMATIC_MIXED_MULTIPLE")
    elif nonaromatic:
        if "SINGLE" in nonaromatic:
            labels.add("SINGLE_CONTROL")
        if "DOUBLE" in nonaromatic:
            labels.add("DOUBLE_CONTROL")
        if "TRIPLE" in nonaromatic:
            labels.add("TRIPLE_CONTROL")
    return labels


def load_candidates() -> list[dict]:
    molecules = json.loads((SOURCE / "molecules.json").read_text())
    r6_rows = [json.loads(line) for line in (R6 / "samples.jsonl").read_text().splitlines() if line.strip()]
    blocked_ids = {row["molecule_id"] for row in r6_rows}
    blocked_conn = {row["canonical_smiles"] for row in r6_rows}
    unique: dict[str, dict] = {}
    for row in molecules:
        smiles = row["canonical_smiles"]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        conn = connectivity(mol)
        if row["molecule_id"] in blocked_ids or conn in blocked_conn:
            continue
        labels = descriptors(row)
        if not labels:
            continue
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        group = scaffold if scaffold else conn
        # one representative per structural group; deterministic and complete
        candidate = {
            "molecule_id": row["molecule_id"], "canonical_smiles": smiles,
            "connection_identity": conn, "group_key": group,
            "group_digest": stable(group), "labels": sorted(labels),
            "heavy_atoms": row.get("heavy_atoms"),
        }
        old = unique.get(group)
        if old is None or stable(row["molecule_id"]) < stable(old["molecule_id"]):
            unique[group] = candidate
    return sorted(unique.values(), key=lambda item: item["group_digest"])


def choose(candidates: list[dict]) -> dict[str, list[dict]]:
    available = {item["group_key"]: item for item in candidates}
    chosen: dict[str, list[dict]] = {split: [] for split in TARGET_GROUPS}
    used: set[str] = set()
    for split in ("fresh", "dev", "train"):
        for label, quota in MINIMA[split].items():
            while sum(label in item["labels"] for item in chosen[split]) < quota:
                options = [item for item in available.values() if item["group_key"] not in used and label in item["labels"]]
                if not options:
                    raise RuntimeError(f"insufficient groups for {split}/{label}")
                # Prefer candidates that satisfy other unmet cells too.
                item = max(options, key=lambda x: (sum(any(xlabel in x["labels"] for xlabel in (other,)) and sum(other in y["labels"] for y in chosen[split]) < MINIMA[split][other] for other in MINIMA[split]), x["group_digest"]))
                chosen[split].append(item); used.add(item["group_key"])
        while len(chosen[split]) < TARGET_GROUPS[split]:
            options = [item for item in available.values() if item["group_key"] not in used]
            if not options:
                raise RuntimeError(f"insufficient groups for target {split}")
            item = options[0]
            chosen[split].append(item); used.add(item["group_key"])
    return chosen


def run() -> dict:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    candidates = load_candidates()
    chosen = choose(candidates)
    molecule_rows = json.loads((SOURCE / "molecules.json").read_text())
    rows_by_id = {row["molecule_id"]: row for row in molecule_rows}
    planned = []
    missing_profiles = []
    for split, items in chosen.items():
        for item in items:
            source = rows_by_id[item["molecule_id"]]
            for profile in PROFILES:
                sample_key = f"{item['molecule_id']}__{profile}"
                if not (SOURCE / "samples" / f"{sample_key}.layers.json").exists():
                    missing_profiles.append(sample_key)
                planned.append({
                    "sample_key": sample_key, "molecule_id": item["molecule_id"],
                    "canonical_smiles": item["canonical_smiles"], "connection_identity": item["connection_identity"],
                    "group_key": item["group_key"], "labels": item["labels"], "profile_id": profile,
                    "split": split, "source_root": str(SOURCE.relative_to(ROOT)), "training_authorized": False,
                })
    group_sets = {split: {item["group_key"] for item in values} for split, values in chosen.items()}
    id_sets = {split: {item["molecule_id"] for item in values} for split, values in chosen.items()}
    overlaps = {}
    for a, b in (("train", "dev"), ("train", "fresh"), ("dev", "fresh")):
        overlaps[f"{a}:{b}"] = {
            "group": len(group_sets[a] & group_sets[b]),
            "molecule": len(id_sets[a] & id_sets[b]),
            "connection": len({x["connection_identity"] for x in chosen[a]} & {x["connection_identity"] for x in chosen[b]}),
        }
    result = {
        "schema_version": "chemvector-aromatic-curriculum-r17-plan-0.1",
        "complete": True, "plan_only": True, "image_bytes_read": False, "model_forwards": 0,
        "training_performed": False, "gpu_used": False, "sealed_test_read": False,
        "production_checkpoint_written": False, "source_root": str(SOURCE.relative_to(ROOT)),
        "candidate_groups": len(candidates), "selected_group_counts": {split: len(items) for split, items in chosen.items()},
        "image_counts": dict(Counter(row["split"] for row in planned)), "profiles": list(PROFILES),
        "missing_profiles": missing_profiles, "overlaps": overlaps,
        "coverage": {split: {label: sum(label in item["labels"] for item in items) for label in MINIMA[split]} for split, items in chosen.items()},
        "rows": planned,
        "gate": {
            "all_profile_files_present": not missing_profiles,
            "identity_disjoint": all(all(value == 0 for value in overlap.values()) for overlap in overlaps.values()),
            "fresh_aromatic_coverage_nonzero": all(result_count > 0 for label, result_count in {label: sum(label in item["labels"] for item in chosen["fresh"]) for label in MINIMA["fresh"] if "AROMATIC" in label}.items()),
            "production_go": False,
        },
    }
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "manifest.json").write_text(json.dumps({"rows": planned}, indent=2, ensure_ascii=False) + "\n")
    (OUT / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT), "candidate_groups": len(candidates), "selected": result["selected_group_counts"], "images": result["image_counts"], "coverage": result["coverage"], "gate": result["gate"]}, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    run()
