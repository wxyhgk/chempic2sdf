"""Pure scene interventions with explicit node correspondence and no hidden rescue."""

from copy import deepcopy

import numpy as np
from rdkit import Chem
from scipy.optimize import linear_sum_assignment

from Data.sdf.chemscene_v0_2 import QUANTIZATION_MAX, mol_from_chemscene_v0_2
from chemvector.direct_scene_inference import apply_atom_ocr
from chemvector.placeholder_scene import compile_carbon_placeholder
from chemvector.shadow_chemscene_v02_oracle import apply_aromatic_consistency


def correspondence(scene, truth, radius=6.0):
    if not scene["atom_objects"] or not truth["atom_objects"]:
        return {}
    predicted = np.array([a["anchor"] for a in scene["atom_objects"]], dtype=float)
    expected = np.array([a["anchor"] for a in truth["atom_objects"]], dtype=float)
    distances = np.linalg.norm(predicted[:, None] - expected[None], axis=-1) * 768 / QUANTIZATION_MAX
    penalty = (min(distances.shape) + 1) * (radius + 1)
    rows, cols = linear_sum_assignment(np.where(distances <= radius, distances, penalty))
    return {int(r): int(c) for r, c in zip(rows, cols) if distances[r, c] <= radius}


def intervene(scene, truth, *, attributes=False, bonds=False):
    result = deepcopy(scene)
    mapping = correspondence(result, truth)
    if attributes:
        for i, target in mapping.items():
            result["atom_objects"][i]["chemical"] = deepcopy(truth["atom_objects"][target]["chemical"])
    unavailable_bonds = 0
    if bonds:
        inverse = {target: i for i, target in mapping.items()}
        replaced = []
        for bond in truth["bond_objects"]:
            if not all(i in inverse for i in bond["atom_refs"]):
                unavailable_bonds += 1
                continue
            item = deepcopy(bond)
            item["id"] = len(replaced)
            item["atom_refs"] = [inverse[i] for i in bond["atom_refs"]]
            references = item.get("stereo_atom_refs")
            if references is not None:
                if all(i in inverse for i in references):
                    item["stereo_atom_refs"] = [inverse[i] for i in references]
                else:
                    item.pop("stereo_atom_refs")
                    item["stereo"] = "STEREONONE"
            replaced.append(item)
        result["bond_objects"] = replaced
        # Rebuild only ports/references. An empty OCR mapping never changes text/chemistry.
        apply_atom_ocr(result, list(range(len(result["atom_objects"]))), {})
    return result, {"matched_atoms": len(mapping), "unavailable_truth_bonds": unavailable_bonds}


def identity(molecule, stereo=False):
    mol = Chem.RemoveHs(Chem.Mol(molecule))
    if not stereo:
        Chem.RemoveStereochemistry(mol)
    # Keep isotopes/charges even when stereo is excluded.
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def compile_and_score(scene, target):
    result, aromatic = apply_aromatic_consistency(scene, "demote-noncycle-clear-atoms")
    errors, fallback, molecule = [], False, None
    if not result["atom_objects"]:
        errors.append("empty atom set")
    else:
        try:
            molecule = mol_from_chemscene_v0_2(result)
        except Exception as error:
            errors.append(str(error))
            try:
                result, molecule, _ = compile_carbon_placeholder(result)
                fallback = True
            except Exception as error:
                errors.append(str(error))
    block, reread = None, None
    if molecule is not None:
        try:
            block = Chem.MolToMolBlock(molecule)
            reread = Chem.MolFromMolBlock(block, sanitize=True, removeHs=True)
            if reread is None:
                errors.append("SDF roundtrip failed")
        except Exception as error:
            errors.append(str(error))
    metrics = {"sdf_valid": reread is not None, "carbon_fallback": fallback,
               "core_graph_exact": reread is not None and identity(reread) == identity(target),
               "stereo_graph_exact": reread is not None and identity(reread, True) == identity(target, True),
               "predicted_smiles": identity(reread) if reread is not None else None,
               "target_smiles": identity(target), "atoms": len(result["atom_objects"]),
               "bonds": len(result["bond_objects"]), "errors": errors, "aromatic": aromatic}
    return metrics, result, block
