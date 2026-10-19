from collections import Counter

import pytest
from rdkit import Chem

from chemvector.training.curriculum.ctab_inventory import (
    UnsupportedInventory, deficiency_witness, literal_inventory, molecular_inventory,
)


def block(smiles="CCO", properties=()):
    result = Chem.MolToMolBlock(Chem.MolFromSmiles(smiles))
    return result.replace("M  END", "\n".join([*properties, "M  END"]))


def test_literal_inventory_is_not_an_identity_or_formula_fingerprint():
    left = literal_inventory(block("CCO"))
    right = literal_inventory(block("COC"))
    assert left["guaranteed_non_hydrogen_inventory"] == right["guaranteed_non_hydrogen_inventory"] == {"C": 2, "O": 1}
    assert deficiency_witness({"C": 2, "O": 1}, left) is None
    assert deficiency_witness({"C": 2}, left) == {"element": "O", "candidate_count": 0, "public_minimum": 1}
    assert deficiency_witness({"C": 100, "O": 10}, left) is None


def test_sup_members_are_all_masked_and_multiline_label_stays_opaque():
    payload = block(properties=("M  STY  1   1 SUP", "M  SAL   1  2   2   3", "M  SMT   1 O", "CH3"))
    inventory = literal_inventory(payload)
    assert inventory["literal_non_hydrogen_inventory"] == {"C": 2, "O": 1}
    assert inventory["guaranteed_non_hydrogen_inventory"] == {"C": 1}
    assert inventory["masked_atom_count"] == 2
    assert inventory["opaque_sup_groups"] == 1
    assert "CH3" in payload
    assert deficiency_witness({"C": 1}, inventory) is None  # Do not invent O from the label.


def test_atom_alias_is_masked_without_expansion():
    inventory = literal_inventory(block(properties=("A    1", "Ph")))
    assert inventory["guaranteed_non_hydrogen_inventory"] == {"C": 1, "O": 1}
    assert inventory["masked_atom_count"] == 1


def test_entirely_opaque_inventory_never_clears_any_candidate():
    inventory = literal_inventory(block("C", ("M  STY  1   1 SUP", "M  SAL   1  1   1", "M  SMT   1 R")))
    assert inventory["guaranteed_non_hydrogen_inventory"] == {}
    assert deficiency_witness({}, inventory) is None
    assert deficiency_witness({"C": 100}, None) is None


def test_all_fragments_retained_and_isotopes_hydrogens_not_used_to_disprove():
    molecule = Chem.MolFromSmiles("[13CH3]CO.[Na+].[Cl-]")
    assert molecular_inventory(molecule) == {"C": 2, "Cl": 1, "Na": 1, "O": 1}
    inventory = literal_inventory(Chem.MolToMolBlock(Chem.AddHs(molecule)))
    assert inventory["guaranteed_non_hydrogen_inventory"] == molecular_inventory(molecule)


@pytest.mark.parametrize("properties", [
    ("M  STY  1   1 SRU", "M  SAL   1  1   1"),
    ("M  ALS   1  2 F C N",), ("M  RGP  1   1   1",), ("M  XYZ  1   1   1",),
    ("M  STY  1   1 SUP",), ("M  SAL   1  1   1",),
    ("M  STY  1   1 SUP", "M  SAL   1  1  10"),
    ("M  STY  1   1 SUP", "M  SAL   1  2   1"),
    ("M  STY  1   1 SUP", "M  SAL   1  1   1", "M  SMT   2 Me"),
    ("unexpected free text",), ("A   99", "Me"),
])
def test_unknown_or_incomplete_ctab_yields_no_negative_evidence(properties):
    with pytest.raises(UnsupportedInventory):
        literal_inventory(block(properties=properties))


@pytest.mark.parametrize("mutation", ["version", "atom_list", "stext", "atom", "bond", "query_bond", "truncated", "end", "extra", "nan"])
def test_malformed_structural_table_is_rejected(mutation):
    lines = block().splitlines()
    if mutation == "version":
        lines[3] = lines[3].replace("V2000", "V3000")
    elif mutation in {"atom_list", "stext"}:
        offset = 6 if mutation == "atom_list" else 15
        lines[3] = lines[3][:offset] + "  1" + lines[3][offset + 3:]
    elif mutation == "atom":
        lines[4] = lines[4][:31] + "*  " + lines[4][34:]
    elif mutation in {"bond", "query_bond"}:
        offset = 0 if mutation == "bond" else 6
        lines[7] = lines[7][:offset] + (" 99" if mutation == "bond" else "  8") + lines[7][offset + 3:]
    elif mutation == "truncated":
        lines = lines[:5]
    elif mutation == "end":
        lines = lines[:-1]
    elif mutation == "extra":
        lines.append("unknown properties")
    else:
        lines[4] = "       nan" + lines[4][10:]
    with pytest.raises(UnsupportedInventory):
        literal_inventory("\n".join(lines))


def test_many_realistic_molecules_preserve_the_necessary_atom_lower_bound():
    smiles = ["CCO", "c1ccccc1", "B(O)O", "C[Si](C)(C)C", "[NH4+].[Cl-]", "C[C@H](O)Cl", "[2H]C([2H])([2H])O"]
    for smi in smiles:
        molecule = Chem.MolFromSmiles(smi)
        for mol in (molecule, Chem.AddHs(molecule)):
            table = literal_inventory(Chem.MolToMolBlock(mol))
            assert Counter(table["guaranteed_non_hydrogen_inventory"]) == Counter(molecular_inventory(molecule))
            assert deficiency_witness(molecular_inventory(molecule), table) is None
