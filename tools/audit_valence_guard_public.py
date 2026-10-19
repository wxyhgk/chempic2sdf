"""Bounded public-validation replay of the explicit-H valence guard.

This is a read-only audit.  It evaluates the fixed public validation cohort
with the initial model and the rejected joint-edge head delta, then applies
the guard only to the in-memory scene before compilation.  No sealed data,
training, checkpoint writes, or deployment changes are permitted.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from rdkit import Chem

from Data.sdf.chemscene_v0_2 import mol_from_chemscene_v0_2
from chemvector.training.curriculum.checkpoints import file_digest
from chemvector.training.curriculum.evaluation import FrozenOCRPredictor
from chemvector.training.curriculum.pixels import image_tensor
from chemvector.training.curriculum.release import RecordLoader
from chemvector.training.curriculum.pair_scale_training import source_checks
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from chemvector.training.curriculum.validation_truth import canonical_molblock, expected_identity
from chemvector.placeholder_scene import compile_carbon_placeholder


ROOT = Path(__file__).resolve().parents[1]
INITIAL = ROOT / "outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
DELTA = ROOT / "outputs/experiments/chemvector-joint-edge-probe-20261005-400/joint-000100.pt"
MANIFEST = ROOT / "outputs/datasets/chemvector-expanded-release-20260915/manifest.json"
OUTPUT = Path(os.environ.get(
    "CHEMVECTOR_VALENCE_GUARD_OUTPUT",
    str(ROOT / "outputs/audits/chemvector-valence-guard-public-20261006-r2"),
))


def filter_explicit_hydrogen_overbonded_edges(scene: dict) -> tuple[dict, dict]:
    """Audit-local, parameter-free replay guard; not production inference."""
    from copy import deepcopy
    result = deepcopy(scene)
    atoms = result.get("atom_objects", [])
    count = {int(a["id"]): 0 for a in atoms}
    kept, removed = [], []
    for bond in result.get("bond_objects", []):
        refs = bond.get("atom_refs", [])
        unsafe = []
        if bond.get("chemical_type") == "AROMATIC":
            for ref in refs:
                if 0 <= ref < len(atoms):
                    c = atoms[ref].get("chemical", {})
                    if (c.get("element") == "O" and c.get("formal_charge", 0) == 0
                            and c.get("explicit_h_count", 0) > 0
                            and c.get("radical_electrons", 0) == 0):
                        unsafe.append(ref)
        if unsafe and any(count[ref] >= 1 for ref in unsafe):
            removed.append({"bond_id": int(bond.get("id", len(kept))),
                            "atom_refs": [int(r) for r in refs],
                            "chemical_type": bond.get("chemical_type"),
                            "reason": "explicit_hydrogen_atom_second_heavy_edge"})
            continue
        kept.append(bond)
        for ref in refs:
            if ref in count:
                count[ref] += 1
    id_map = {}
    for new_id, bond in enumerate(kept):
        id_map[int(bond["id"])] = new_id
        bond["id"] = new_id
    result["bond_objects"] = kept
    for atom in atoms:
        atom["incident_ports"] = [{**p, "bond_id": id_map[int(p["bond_id"])]}
            for p in atom.get("incident_ports", []) if int(p.get("bond_id", -1)) in id_map]
    return result, {"removed": removed, "removed_count": len(removed)}


def _compile_guarded(trace):
    guarded, report = filter_explicit_hydrogen_overbonded_edges(trace["scene_before_fallback"])
    # Keep the compiler path exactly aligned with FrozenOCRPredictor.  The
    # trace is captured before its deterministic carbon/aromatic placeholder
    # fallback, so an audit-only guard must run that same fallback after its
    # in-memory edit. Otherwise valid baseline scenes are spuriously counted
    # as new fallbacks (the old r2 audit did this).
    try:
        mol = mol_from_chemscene_v0_2(guarded)
        return canonical_molblock(Chem.MolToMolBlock(mol)), report, None
    except (ValueError, RuntimeError) as error:
        try:
            _, molecule, _ = compile_carbon_placeholder(guarded)
            return canonical_molblock(Chem.MolToMolBlock(molecule)), report, None
        except (ValueError, RuntimeError) as fallback_error:
            return None, report, f"{error}; placeholder fallback: {fallback_error}"


def _evaluate_arm(model, config, payload, manifest, loader, device, label):
    predictor = FrozenOCRPredictor(payload["provenance"], device)
    records = []
    model.eval()
    with torch.no_grad():
        for row in manifest["samples"]:
            if row["split"] != "validation":
                continue
            sample = loader(manifest, row)
            pixels = image_tensor(sample["image"]).unsqueeze(0).to(device)
            trace = {}
            structure = predictor.predict_structure(model, config, pixels)
            prediction = predictor.from_structure(
                structure, model.role, model.bbox, pixels, sample["image"], trace=trace
            )
            guarded_smiles, report, guard_error = _compile_guarded(trace)
            expected = expected_identity(sample)
            original_smiles = canonical_molblock(prediction["molblock"])
            records.append({
                "key": row["source"] + ":" + row["sample_key"],
                "arm": label,
                "original_valid": original_smiles is not None,
                "original_exact": original_smiles == expected,
                "guarded_valid": guarded_smiles is not None,
                "guarded_exact": guarded_smiles == expected,
                "removed_count": report["removed_count"],
                "removed": report["removed"],
                "guard_error": guard_error,
            })
            del pixels
    return records


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if file_digest(INITIAL) != INITIAL_SHA:
        raise ValueError("initial checkpoint changed")
    if not DELTA.exists() or not MANIFEST.exists():
        raise FileNotFoundError("fixed public audit inputs missing")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for bounded parity replay")
    device = torch.device("cuda:0")
    sources = source_checks()
    manifest = sources["manifest"]
    loader = RecordLoader(manifest)
    model, config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    model.to(device).eval().requires_grad_(False)
    initial = _evaluate_arm(model, config, payload, manifest, loader, device, "initial")
    state = torch.load(DELTA, map_location="cpu", weights_only=True)
    if state.get("initial_sha256") != INITIAL_SHA or state.get("deployment_authorized") is not False:
        raise ValueError("delta binding/deployment contract changed")
    model.pair.load_state_dict(state["heads"], strict=False)
    joint = _evaluate_arm(model, config, payload, manifest, loader, device, "joint-100")
    def summary(rows):
        return {
            "images": len(rows),
            "original_exact": sum(r["original_exact"] for r in rows),
            "original_valid": sum(r["original_valid"] for r in rows),
            "guarded_exact": sum(r["guarded_exact"] for r in rows),
            "guarded_valid": sum(r["guarded_valid"] for r in rows),
            "removed_edges": sum(r["removed_count"] for r in rows),
            "changed": sum(r["removed_count"] > 0 for r in rows),
            "guarded_new_fallback": sum(r["original_valid"] and not r["guarded_valid"] for r in rows),
            "guarded_broken_exact": sum(r["original_exact"] and not r["guarded_exact"] for r in rows),
            "guarded_rescued_exact": sum(not r["original_exact"] and r["guarded_exact"] for r in rows),
        }
    result = {
        "schema_version": "chemvector-valence-guard-public-audit-0.1",
        "sealed_test_read": False,
        "training_performed": False,
        "deployment_authorized": False,
        "initial_sha256": INITIAL_SHA,
        "delta_sha256": file_digest(DELTA),
        "summary": {"initial": summary(initial), "joint-100": summary(joint)},
        "joint_001966": [r for r in joint if "001966_coordgen" in r["key"]],
        "records": initial + joint,
    }
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    return result


if __name__ == "__main__":
    run()
