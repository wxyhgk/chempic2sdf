"""Prepare a metadata-only r6 endpoint diversity manifest.

The pool is globally disjoint from the fixed r5 source and the accepted 6k
targeted panel.  This script does not render, load a model, read sealed data,
or authorize training.  Labels are used only to stratify a quarantine plan;
candidate artifacts must remain inference-only after rendering.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Data.sdf.chemvector_curriculum_v0_2.catalog import entries
from tools.prepare_line_first_valence_targeted_manifest_r5 import connection_identity

SOURCE = ROOT / "outputs/datasets/chemvector-identity-aligned-visible-text-20261017-r5-render"
PANEL6K = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-targeted-6k-manifest-r5/manifest.json"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-targeted-diversity-r6"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"

STYLES = (
    {"profile_id": "r6_s24_r30_w16_sans", "font_family": "DejaVu Sans", "font_size": 24, "bond_length_cap": 24.0, "bond_line_width": 1.6, "rotation_degrees": 30.0},
    {"profile_id": "r6_s32_r60_w32_serif", "font_family": "DejaVu Serif", "font_size": 32, "bond_length_cap": 32.0, "bond_line_width": 3.2, "rotation_degrees": 60.0},
    {"profile_id": "r6_s24_r60_w32_sans", "font_family": "DejaVu Sans", "font_size": 24, "bond_length_cap": 32.0, "bond_line_width": 3.2, "rotation_degrees": 60.0},
    {"profile_id": "r6_s32_r30_w16_serif", "font_family": "DejaVu Serif", "font_size": 32, "bond_length_cap": 24.0, "bond_line_width": 1.6, "rotation_degrees": 30.0},
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bond_flags(smiles: str) -> dict[str, bool]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid smiles: {smiles}")
    bonds = list(mol.GetBonds())
    aromatic = any(b.GetIsAromatic() or b.GetBondType() == Chem.BondType.AROMATIC for b in bonds)
    double = any(b.GetBondType() == Chem.BondType.DOUBLE for b in bonds)
    triple = any(b.GetBondType() == Chem.BondType.TRIPLE for b in bonds)
    hetero_multiple = any(
        a.GetSymbol() in {"N", "O"} and any(
            b.GetBondType() in {Chem.BondType.DOUBLE, Chem.BondType.TRIPLE}
            for b in a.GetBonds()
        ) for a in mol.GetAtoms()
    )
    return {"aromatic": aromatic, "double": double, "triple": triple,
            "hetero_multiple": hetero_multiple}


def category(flags: dict[str, bool]) -> str | None:
    if flags["double"] and not flags["triple"] and not flags["aromatic"]:
        return "DOUBLE-only"
    if flags["triple"] and not flags["double"] and not flags["aromatic"]:
        return "TRIPLE-only"
    if flags["aromatic"] and not (flags["double"] or flags["triple"]):
        return "AROMATIC"
    if flags["aromatic"] and (flags["double"] or flags["triple"]):
        return "mixed_multiple_bond"
    if flags["hetero_multiple"]:
        return "hetero_N_O_multiple"
    return None


def stable(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def run() -> dict:
    if OUT.exists():
        raise FileExistsError(OUT)
    source_rows = [json.loads(x) for x in (SOURCE / "samples.jsonl").read_text().splitlines() if x.strip()]
    excluded = {str(row["connection_identity"]) for row in source_rows if row.get("connection_identity")}
    panel = json.loads(PANEL6K.read_text())
    excluded |= {str(row["connection_identity"]) for row in panel["items"]}
    pool = {}
    for entry in entries():
        ci = connection_identity(entry["canonical_smiles"])
        if ci in excluded or ci in pool:
            continue
        flags = bond_flags(entry["canonical_smiles"])
        tag = category(flags)
        if tag is not None:
            pool[ci] = {"molecule_id": entry["molecule_id"], "canonical_smiles": entry["canonical_smiles"],
                        "connection_identity": ci, "category": tag, "flags": flags}
    quotas = {"DOUBLE-only": 60, "TRIPLE-only": 40, "AROMATIC": 60,
              "mixed_multiple_bond": 40, "hetero_N_O_multiple": 40}
    selected = []
    used = set()
    for tag, quota in quotas.items():
        choices = sorted((x for x in pool.values() if x["category"] == tag), key=lambda x: stable(x["connection_identity"]))
        for item in choices:
            if item["connection_identity"] in used:
                continue
            selected.append({**item, "selection_stratum": tag})
            used.add(item["connection_identity"])
            if sum(x["selection_stratum"] == tag for x in selected) >= quota:
                break
    missing = {tag: quota - sum(x["selection_stratum"] == tag for x in selected) for tag, quota in quotas.items()}
    if any(value > 0 for value in missing.values()):
        result = {"schema_version": "chemvector-identity-aligned-line-first-targeted-diversity-r6-0.1",
                  "complete": True, "metadata_only": True, "status": "no_go_quota_shortfall",
                  "training_performed": False, "images_rendered": False, "sealed_test_read": False,
                  "production_checkpoint_written": False, "initial_sha256": INITIAL_SHA,
                  "excluded_identity_count": len(excluded), "candidate_pool": len(pool),
                  "missing_quotas": missing, "gate": {"identity_disjoint": True, "quota_complete": False,
                  "training_authorized": False}}
    else:
        selected.sort(key=lambda x: stable(x["connection_identity"]))
        for i, item in enumerate(selected):
            item["split"] = "train" if i < 192 else ("dev" if i < 216 else "fresh_holdout")
            item["target_profiles"] = [style["profile_id"] for style in STYLES]
        rows = [{"sample_key": item["molecule_id"] + "__" + style["profile_id"],
                 "molecule_id": item["molecule_id"], "canonical_smiles": item["canonical_smiles"],
                 "connection_identity": item["connection_identity"], "category": item["category"],
                 "selection_stratum": item["selection_stratum"], "split": item["split"],
                 **style, "training_authorized": False} for item in selected for style in STYLES]
        result = {"schema_version": "chemvector-identity-aligned-line-first-targeted-diversity-r6-0.1",
                  "complete": True, "metadata_only": True, "status": "quarantine_plan_only",
                  "training_performed": False, "images_rendered": False, "sealed_test_read": False,
                  "production_checkpoint_written": False, "initial_sha256": INITIAL_SHA,
                  "excluded_identity_count": len(excluded), "candidate_pool": len(pool),
                  "selected_molecules": len(selected), "planned_images": len(rows),
                  "split_counts": dict(Counter(x["split"] for x in rows)),
                  "category_counts": dict(Counter(x["selection_stratum"] for x in selected)),
                  "styles": list(STYLES), "rows": rows,
                  "gate": {"identity_disjoint": True, "quota_complete": True,
                           "style_complete": len(STYLES) == 4, "metadata_only": True,
                           "sealed_test_read": False, "training_authorized": False,
                           "next": "render quarantine, validate hashes/graph-SDF identity, then endpoint ceiling before CPU fit"}}
    OUT.mkdir(parents=True)
    (OUT / "manifest.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    (OUT / "rows.jsonl").write_text("".join(json.dumps(x, sort_keys=True, ensure_ascii=False) + "\n" for x in result.get("rows", [])))
    print(json.dumps({k: result.get(k) for k in ("status", "candidate_pool", "selected_molecules", "planned_images", "split_counts", "category_counts", "missing_quotas", "gate")}, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    run()
