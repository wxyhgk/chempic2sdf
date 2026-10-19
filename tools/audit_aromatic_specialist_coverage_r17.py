#!/usr/bin/env python3
"""Compute target-side aromatic coverage for the admitted r17 slice."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "outputs/audits/chemvector-aromatic-specialist-plan-r17-v2/rows.jsonl"
SOURCE = ROOT / "outputs/datasets/structure-expansion-batch-12272-20260923"
ADMISSION = ROOT / "outputs/audits/chemvector-aromatic-specialist-admission-r17/report.json"
OUT = ROOT / "outputs/audits/chemvector-aromatic-specialist-coverage-r17"


def run() -> dict:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    admission = json.loads(ADMISSION.read_text())
    if not all(admission["gate"].get(key) for key in ("private_identity_admitted", "public_identity_admitted", "source_membership", "render_hash_contract", "graph_sdf_contract")):
        raise RuntimeError("r17 admission gate incomplete")
    rows = [json.loads(line) for line in PLAN.read_text().splitlines() if line.strip()]
    totals = {}
    per_profile = {}
    for split in ("train", "dev", "fresh_holdout"):
        counter = Counter()
        for row in rows:
            if row["split"] != split:
                continue
            layers = json.loads((SOURCE / "samples" / f"{row['sample_key']}.layers.json").read_text())
            graph = json.loads((SOURCE / "samples" / f"{row['sample_key']}.graph.json").read_text())
            atoms = layers["atoms"]
            bonds = graph["bonds"]
            counter["images"] += 1
            counter["aromatic_atoms"] += sum(bool(atom.get("aromatic")) for atom in atoms)
            counter["aromatic_C"] += sum(bool(atom.get("aromatic")) and atom.get("element") == "C" for atom in atoms)
            counter["aromatic_N"] += sum(bool(atom.get("aromatic")) and atom.get("element") == "N" for atom in atoms)
            counter["aromatic_S"] += sum(bool(atom.get("aromatic")) and atom.get("element") == "S" for atom in atoms)
            counter["aromatic_P"] += sum(bool(atom.get("aromatic")) and atom.get("element") == "P" for atom in atoms)
            counter["aromatic_bonds"] += sum(str(bond.get("chemical_type")).upper() == "AROMATIC" or bool(bond.get("aromatic")) for bond in bonds)
            counter["double_bonds"] += sum(str(bond.get("chemical_type")).upper() == "DOUBLE" for bond in bonds)
            counter["triple_bonds"] += sum(str(bond.get("chemical_type")).upper() == "TRIPLE" for bond in bonds)
        totals[split] = dict(counter)
    for profile in sorted({row["profile_id"] for row in rows}):
        counter = Counter()
        for row in rows:
            if row["split"] != "fresh_holdout" or row["profile_id"] != profile:
                continue
            layers = json.loads((SOURCE / "samples" / f"{row['sample_key']}.layers.json").read_text())
            counter["images"] += 1
            counter["aromatic_atoms"] += sum(bool(atom.get("aromatic")) for atom in layers["atoms"])
        per_profile[profile] = dict(counter)
    fresh = totals["fresh_holdout"]
    report = {
        "schema_version": "chemvector-aromatic-specialist-coverage-r17-0.1",
        "complete": True, "read_only": True, "training_performed": False, "gpu_used": False,
        "sealed_test_read": False, "production_checkpoint_written": False,
        "totals": totals, "fresh_by_profile": per_profile,
        "gate": {
            "admission_inherited": True,
            "fresh_aromatic_atoms_nonzero": fresh["aromatic_atoms"] > 0,
            "fresh_aromatic_bonds_nonzero": fresh["aromatic_bonds"] > 0,
            "fresh_aromatic_C_nonzero": fresh["aromatic_C"] > 0,
            "fresh_aromatic_N_nonzero": fresh["aromatic_N"] > 0,
            "fresh_aromatic_S_nonzero": fresh["aromatic_S"] > 0,
            "fresh_double_control_nonzero": fresh["double_bonds"] > 0,
            "fresh_triple_control_nonzero": fresh["triple_bonds"] > 0,
            "all_profiles_have_aromatic_atoms": all(value["aromatic_atoms"] > 0 for value in per_profile.values()),
            "training_admissible": False,
            "production_go": False,
        },
    }
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT / "report.json"), "totals": totals, "gate": report["gate"]}, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    run()
