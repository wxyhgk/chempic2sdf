"""Read-only targeted replay on audited structure-expansion quarantine data."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.training.curriculum.checkpoints import file_digest
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.data import CurriculumCollator, load_sample
from chemvector.training.curriculum.gradient_preflight import state_digest
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from chemvector.training.graph.relations.train_direct_atom_pair_bond import match_atom_targets

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs/datasets/structure-expansion-batch-12272-20260923"
V02 = ROOT / "outputs/datasets/curriculum-v0.2-expanded-20260919/molecules.json"
MOLECULES = DATA / "molecules.json"
SAMPLES = DATA / "samples.json"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
OUTPUT = ROOT / "outputs/audits/chemvector-structure-expansion-targeted-replay-20261006-r1"
PLAN = Path(__import__("os").environ.get("CHEMVECTOR_TARGETED_SPLIT_PLAN", ""))


def _canonical_bonds(smiles: str) -> tuple[set[str], set[str]]:
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid canonical SMILES")
    elements, bonds = set(), set()
    for atom in mol.GetAtoms():
        if atom.GetSymbol() in {"O", "N", "I"}:
            elements.add(atom.GetSymbol())
            for bond in atom.GetBonds():
                kind = str(bond.GetBondType()).upper()
                bonds.add("AROMATIC" if bond.GetIsAromatic() or kind == "AROMATIC" else kind)
    return elements, bonds


def _select_rows() -> list[dict]:
    if PLAN:
        plan = json.loads((PLAN if PLAN.is_absolute() else ROOT / PLAN).read_text(encoding="utf-8"))
        rows = plan.get("rows", [])
        if len(rows) not in {128, 130} or len({r["sample_key"] for r in rows}) != len(rows):
            raise ValueError("Targeted split plan must contain 128 or 130 unique rows")
        return rows
    rows = json.loads(SAMPLES.read_text(encoding="utf-8"))
    blocked = {r["canonical_smiles"] for r in json.loads(V02.read_text(encoding="utf-8"))}
    smiles_by_id = {r["molecule_id"]: r["canonical_smiles"] for r in json.loads(MOLECULES.read_text(encoding="utf-8"))}
    by_molecule = {}
    for row in rows:
        smiles = smiles_by_id[row["molecule_id"]]
        if row.get("profile_id") not in {"oa11", "oa13"} or smiles in blocked:
            continue
        elements, bonds = _canonical_bonds(smiles)
        if not elements & {"O", "N"} or not bonds & {"DOUBLE", "TRIPLE", "AROMATIC"}:
            continue
        by_molecule.setdefault(row["molecule_id"], []).append(row)
    if not by_molecule:
        raise ValueError("No targeted identity-disjoint molecules")
    selected = []
    ordered = sorted(by_molecule.items(), key=lambda kv: hashlib.sha256(kv[0].encode()).hexdigest())[:64]
    for index, (molecule_id, candidates) in enumerate(ordered):
        wanted = "oa11" if index % 2 == 0 else "oa13"
        selected.append(next((r for r in candidates if r["profile_id"] == wanted),
                             min(candidates, key=lambda r: r["sample_key"])))
    return sorted(selected, key=lambda r: r["sample_key"])


def run(output: Path = OUTPUT) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if not torch.cuda.is_available() or file_digest(INITIAL) != INITIAL_SHA:
        raise RuntimeError("CUDA or initial checkpoint contract unavailable")
    rows = _select_rows()
    model, config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    model.cuda().eval().requires_grad_(False)
    before = state_digest(model)
    collator = CurriculumCollator(AtomBondSceneConfig(atom_queries=config.maximum_atoms,
                                                       bond_queries=config.maximum_bonds), purpose="diagnostic")
    totals = defaultdict(int)
    strata = defaultdict(lambda: defaultdict(int))
    by_split = defaultdict(lambda: defaultdict(int))
    records = []
    with torch.inference_mode():
        for row in rows:
            sample = load_sample(DATA, row["sample_key"], purpose="diagnostic")
            if sample["meta"].get("split") != "quarantine" or sample["meta"].get("training_authorized") is not False:
                raise ValueError("Targeted source is not quarantine-only")
            batch = collator([sample])
            tensors = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}
            out = structure_forward(model, tensors["pixel_values"], config, payload["provenance"]["atom_threshold"])
            raw, selected = out[3], out[6]
            raw_map, _ = match_atom_targets(raw, tensors, config)
            selected_map, _ = match_atom_targets(selected, tensors, config)
            target_ids = [int(v) for v in batch["atom_ids"][0]]
            raw_ids = {target_ids[int(v)] for v in raw_map[0].tolist() if 0 <= int(v) < len(target_ids)}
            selected_ids = {target_ids[int(v)] for v in selected_map[0].tolist() if 0 <= int(v) < len(target_ids)}
            atoms = {int(a["atom_id"]): a for a in sample["layers"]["atoms"]}
            for atom_id, atom in atoms.items():
                role = "visible" if atom.get("visible_text") else "implicit"
                key = f"{sample['meta']['profile_id']}:{role}:{atom['element']}"
                block = strata[key]
                block["targets"] += 1
                block["raw_matched"] += int(atom_id in raw_ids)
                block["selected_matched"] += int(atom_id in selected_ids)
                block["selector_drop"] += int(atom_id in raw_ids - selected_ids)
                totals["target_atoms"] += 1
                totals["raw_matched"] += int(atom_id in raw_ids)
                totals["selected_matched"] += int(atom_id in selected_ids)
            totals["images"] += 1
            totals["raw_gap"] += len(set(atoms) - raw_ids)
            totals["selector_drop"] += len(raw_ids - selected_ids)
            split = str(row.get("split", "diagnostic"))
            by_split[split]["images"] += 1
            by_split[split]["target_atoms"] += len(atoms)
            by_split[split]["raw_matched"] += len(raw_ids)
            by_split[split]["selected_matched"] += len(selected_ids)
            by_split[split]["raw_gap"] += len(set(atoms) - raw_ids)
            by_split[split]["selector_drop"] += len(raw_ids - selected_ids)
            records.append({"sample_key": row["sample_key"], "molecule_id": row["molecule_id"],
                            "profile_id": row["profile_id"], "split": split, "raw_matched": len(raw_ids),
                            "selected_matched": len(selected_ids), "target_atoms": len(atoms)})
    after = state_digest(model)
    if before != after or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("Frozen model state changed")
    result = {"schema_version": "chemvector-structure-expansion-targeted-replay-0.1",
              "complete": True, "read_only": True, "training_performed": False,
              "optimizer_steps": 0, "gpu_used": True, "sealed_test_read": False,
              "deployment_authorized": False, "initial_sha256": INITIAL_SHA,
              "source": {"dataset": str(DATA.relative_to(ROOT)), "samples_sha256": file_digest(SAMPLES),
                         "blocked_v02_molecules_sha256": file_digest(V02)},
              "selection": {"images": len(rows), "profiles": ["oa11", "oa13"],
                            "identity_disjoint_from_v02": True,
                            "requires_visible_on_and_non_single_bond": True},
              "totals": dict(totals), "by_split": {k: dict(v) for k, v in sorted(by_split.items())},
              "strata": {k: dict(v) for k, v in sorted(strata.items())},
              "records": records, "state_before": before, "state_after": after,
              "gate": {"training_admissible": False, "minimum_64_molecules": len(rows) >= 64,
                       "reason": "targeted reachability diagnostic only"}}
    output.mkdir(parents=True)
    (output / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "totals": dict(totals)}, indent=2))
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    run()
