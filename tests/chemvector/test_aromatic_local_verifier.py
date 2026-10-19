from copy import deepcopy

from rdkit import Chem

from Data.sdf.chemscene_v0_2 import mol_from_chemscene_v0_2
from chemvector.components.atoms.chemistry.aromatic_local_verifier import verify_aromatic_scene


def scene():
    atoms = []
    for index in range(6):
        atoms.append({
            "id": index,
            "anchor": [80 + index * 90, 300],
            "role": "IMPLICIT",
            "text": None,
            "chemical": {
                "element": "C", "formal_charge": 0, "isotope": 0,
                "aromatic": True, "explicit_h_count": 0, "no_implicit": False,
                "radical_electrons": 0, "canonical_chiral_tag": "CHI_UNSPECIFIED",
            },
            "incident_ports": [],
        })
    bonds = []
    for index in range(6):
        left, right = index, (index + 1) % 6
        bond = {
            "id": index, "atom_refs": [left, right],
            "ports": [list(atoms[left]["anchor"]), list(atoms[right]["anchor"])],
            "chemical_type": "AROMATIC", "display_type": "AROMATIC",
            "direction": "NONE", "stereo": "STEREONONE", "bbox": None,
        }
        bonds.append(bond)
    return {
        "schema_version": "chemscene-0.2", "molecule_id": "verifier-test",
        "units": "q1023", "canvas": {"width": 768, "height": 768},
        "atom_objects": atoms, "bond_objects": bonds,
    }


def test_valid_baseline_is_returned_without_intervention():
    raw = scene()
    output, molecule, journal = verify_aromatic_scene(
        raw, {index: 0.1 for index in range(6)}, compiler=lambda _: object()
    )
    assert output == raw and molecule is not None
    assert journal["status"] == "baseline_valid"
    assert journal["compiler_calls"] == 1


def test_failed_baseline_accepts_one_bounded_local_demotion():
    raw = scene()
    # A synthetic compiler fails until bond 2 is demoted.
    def compiler(candidate):
        if candidate["bond_objects"][2]["chemical_type"] == "AROMATIC":
            raise ValueError("synthetic kekulization failure")
        return object()

    before = deepcopy(raw)
    output, molecule, journal = verify_aromatic_scene(
        raw, {0: 0.9, 1: 0.8, 2: 0.2, 3: 0.7}, compiler=compiler,
    )
    assert molecule is not None and journal["status"] == "local_repair_accepted"
    assert journal["repaired_bond_id"] == 2
    assert raw == before and output["atom_objects"] == before["atom_objects"]
    assert output["bond_objects"][2]["chemical_type"] == "SINGLE"


def test_failed_repair_returns_original_scene_and_no_molecule():
    raw = scene()
    before = deepcopy(raw)
    output, molecule, journal = verify_aromatic_scene(
        raw, {index: 0.1 for index in range(6)}, compiler=lambda _: (_ for _ in ()).throw(ValueError("bad")),
    )
    assert output == before and molecule is None
    assert journal["status"] == "local_repair_failed"
    assert not journal["intervention"]
