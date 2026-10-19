from collections import Counter

import pytest
from rdkit import Chem

from Data.sdf.chemvector_curriculum_v0_1.functional_groups import (
    CATALOG_STATUS,
    annotate_groups,
    records,
)


def _names(smiles):
    return {group["name"] for group in annotate_groups(Chem.MolFromSmiles(smiles))}


def test_draft_catalogue_is_balanced_unique_and_returns_independent_records():
    examples = records()
    assert CATALOG_STATUS == "draft/quarantine"
    assert len(examples) == 64
    assert len({row["record_id"] for row in examples}) == len(examples)
    assert all(row["stage"] == "functional_groups" for row in examples)
    counts = Counter(concept for row in examples for concept in row["concepts"])
    assert len(counts) == 16
    assert set(counts.values()) == {4}
    canonical = [Chem.MolToSmiles(Chem.MolFromSmiles(row["smiles"])) for row in examples]
    assert len(set(canonical)) == len(examples)
    examples[0]["concepts"].append("not_a_real_concept")
    examples[0]["smiles"] = "invalid"
    assert records()[0]["concepts"] == ["alcohol"]
    assert records()[0]["smiles"] == "CCO"


@pytest.mark.parametrize("record", records(), ids=lambda row: row["record_id"])
def test_examples_are_complete_valid_and_declared_concepts_match(record):
    mol = Chem.MolFromSmiles(record["smiles"])
    assert mol is not None
    assert Chem.SanitizeMol(mol, catchErrors=True) == Chem.SanitizeFlags.SANITIZE_NONE
    assert len(Chem.GetMolFrags(mol)) == 1
    assert 3 <= mol.GetNumHeavyAtoms() <= 12
    assert all(atom.GetAtomicNum() > 0 for atom in mol.GetAtoms())
    assert all(atom.GetNumRadicalElectrons() == 0 for atom in mol.GetAtoms())
    assert sum(atom.GetFormalCharge() for atom in mol.GetAtoms()) == 0
    before = Chem.MolToMolBlock(mol)
    annotations = annotate_groups(mol)
    assert {item["name"] for item in annotations} == set(record["concepts"])
    assert annotations == annotate_groups(mol)
    assert Chem.MolToMolBlock(mol) == before
    for item in annotations:
        assert item["atom_ids"] == sorted(set(item["atom_ids"]))
        assert item["bond_ids"] == sorted(set(item["bond_ids"]))
        assert all(0 <= idx < mol.GetNumAtoms() for idx in item["atom_ids"])
        assert all(0 <= idx < mol.GetNumBonds() for idx in item["bond_ids"])
        for idx in item["bond_ids"]:
            bond = mol.GetBondWithIdx(idx)
            assert bond.GetBeginAtomIdx() in item["atom_ids"]
            assert bond.GetEndAtomIdx() in item["atom_ids"]


@pytest.mark.parametrize(
    ("left", "right", "left_group", "right_group"),
    [
        ("CCO", "CC=O", "alcohol", "aldehyde"),
        ("CC(O)C", "CC(=O)C", "alcohol", "ketone"),
        ("COC", "CC(=O)OC", "ether", "ester"),
        ("CC(=O)O", "CC(=O)OC", "carboxylic_acid", "ester"),
        ("CNC", "CC(=O)NC", "amine", "amide"),
        ("CCN", "CC#N", "amine", "nitrile"),
        ("CCS", "CCSC", "thiol", "thioether"),
        ("CSC", "CS(C)=O", "thioether", "sulfoxide"),
        ("CS(C)=O", "CS(C)(=O)=O", "sulfoxide", "sulfone"),
        ("CS(C)(=O)=O", "CS(N)(=O)=O", "sulfone", "sulfonamide"),
        ("CCO", "Oc1ccccc1", "alcohol", "phenol"),
    ],
)
def test_nearby_bond_order_or_neighbor_changes_distinguish_groups(
    left, right, left_group, right_group
):
    assert _names(left) == {left_group}
    assert _names(right) == {right_group}


def test_shared_context_is_not_forced_into_a_single_label():
    # Two adjacent phenols share the aromatic C-C neighborhood; glycolic acid
    # contains both an alcohol and an acid without treating acid OH as alcohol.
    catechol = annotate_groups(Chem.MolFromSmiles("Oc1ccccc1O"))
    assert len(catechol) == 2
    assert {group["name"] for group in catechol} == {"phenol"}
    assert _names("OCC(=O)O") == {"alcohol", "carboxylic_acid"}
    # A real overlap at the carbonyl atom is allowed (carbamate).
    carbamate = annotate_groups(Chem.MolFromSmiles("COC(N)=O"))
    assert {group["name"] for group in carbamate} == {"ester", "amide"}
    assert set(carbamate[0]["atom_ids"]) & set(carbamate[1]["atom_ids"])


@pytest.mark.parametrize("smiles", ["CCN", "CNC", "CN(C)C", "N1CCCC1"])
def test_each_amine_center_has_one_annotation(smiles):
    annotations = annotate_groups(Chem.MolFromSmiles(smiles))
    assert len(annotations) == 1
    assert annotations[0]["name"] == "amine"


def test_indices_follow_supplied_molecule_including_reordered_atoms():
    mol = Chem.MolFromSmiles("CC(=O)OCCN")
    order = list(reversed(range(mol.GetNumAtoms())))
    reverse = Chem.RenumberAtoms(mol, order)
    old_to_new = {old: new for new, old in enumerate(order)}

    def signature(molecule, annotations, index_map):
        return {
            (
                item["name"],
                tuple(sorted(index_map[idx] for idx in item["atom_ids"])),
                tuple(
                    sorted(
                        tuple(
                            sorted(
                                (
                                    index_map[molecule.GetBondWithIdx(idx).GetBeginAtomIdx()],
                                    index_map[molecule.GetBondWithIdx(idx).GetEndAtomIdx()],
                                )
                            )
                        )
                        for idx in item["bond_ids"]
                    )
                ),
            )
            for item in annotations
        }

    assert signature(mol, annotate_groups(mol), old_to_new) == signature(
        reverse, annotate_groups(reverse), {idx: idx for idx in range(reverse.GetNumAtoms())}
    )


def test_empty_no_group_and_invalid_inputs():
    assert annotate_groups(Chem.MolFromSmiles("")) == []
    assert annotate_groups(Chem.MolFromSmiles("CCC")) == []
    assert annotate_groups(Chem.MolFromSmiles("C[NH3+]")) == []
    with pytest.raises(ValueError, match="valid RDKit molecule"):
        annotate_groups(None)
