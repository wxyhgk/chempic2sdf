"""Chemical and metadata checks for the quarantined atom/bond teaching pilot."""

from __future__ import annotations

import re

import pytest
from rdkit import Chem

from Data.sdf.chemvector_curriculum_v0_1.atoms_bonds import records


def _molecule(record: dict) -> Chem.Mol:
    mol = Chem.MolFromSmiles(record["smiles"], sanitize=True)
    assert mol is not None, record["record_id"]
    return mol


def test_records_are_stable_unique_and_independent() -> None:
    examples = records()
    assert 25 <= len(examples) <= 45
    assert examples == records()
    ids = [record["record_id"] for record in examples]
    assert len(ids) == len(set(ids))
    for record in examples:
        assert set(record) == {"record_id", "stage", "name", "smiles", "concepts"}
        assert re.fullmatch(r"atoms_bonds_[a-z0-9_]+", record["record_id"])
        assert record["stage"] == "atoms_bonds"
        assert record["name"].strip()
        assert record["concepts"]
        assert len(record["concepts"]) == len(set(record["concepts"]))
    examples[0]["concepts"].append("not_persistent")
    examples[0]["name"] = "not_persistent"
    examples.pop()
    assert "not_persistent" not in repr(records())


@pytest.mark.parametrize("record", records(), ids=lambda item: item["record_id"])
def test_each_structure_is_a_complete_small_single_component_graph(record: dict) -> None:
    mol = _molecule(record)
    assert len(Chem.GetMolFrags(mol)) == 1
    assert 2 <= mol.GetNumHeavyAtoms() <= 8
    assert mol.GetNumBonds() >= 1
    assert all(atom.GetAtomicNum() > 0 for atom in mol.GetAtoms())
    assert all(not atom.HasQuery() for atom in mol.GetAtoms())
    assert all(atom.GetNumRadicalElectrons() == 0 for atom in mol.GetAtoms())
    assert all(atom.GetAtomMapNum() == 0 for atom in mol.GetAtoms())
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    restored = Chem.MolFromSmiles(canonical, sanitize=True)
    assert restored is not None
    assert Chem.MolToSmiles(restored, canonical=True, isomericSmiles=True) == canonical


def test_canonical_structures_are_deduplicated_within_stage() -> None:
    by_smiles: dict[str, str] = {}
    for record in records():
        canonical = Chem.MolToSmiles(_molecule(record), canonical=True, isomericSmiles=True)
        assert canonical not in by_smiles, (record["record_id"], by_smiles.get(canonical))
        by_smiles[canonical] = record["record_id"]


def test_element_and_basic_bond_coverage_is_verified_from_the_graphs() -> None:
    examples = records()
    molecules = [_molecule(record) for record in examples]
    required_elements = {"N", "O", "S", "P", "B", "Si", "F", "Cl", "Br", "I"}
    observed_elements = {atom.GetSymbol() for mol in molecules for atom in mol.GetAtoms()}
    assert required_elements <= observed_elements
    observed_bonds = {bond.GetBondType() for mol in molecules for bond in mol.GetBonds()}
    assert {Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE} <= observed_bonds
    assert Chem.BondType.AROMATIC in observed_bonds
    for record, mol in zip(examples, molecules):
        symbols = {atom.GetSymbol() for atom in mol.GetAtoms()}
        for concept in record["concepts"]:
            if concept.startswith("element_"):
                assert concept.removeprefix("element_") in symbols
        types = {bond.GetBondType() for bond in mol.GetBonds()}
        for concept, bond_type in (
            ("single_bond", Chem.BondType.SINGLE),
            ("double_bond", Chem.BondType.DOUBLE),
            ("triple_bond", Chem.BondType.TRIPLE),
            ("aromatic_bond", Chem.BondType.AROMATIC),
        ):
            if concept in record["concepts"]:
                assert bond_type in types
        if "mixed_bond_orders" in record["concepts"]:
            assert len(types) >= 2


def test_structural_concepts_charges_and_isotopes_have_real_examples() -> None:
    examples = records()
    required = {
        "implicit_carbon_endpoint",
        "implicit_carbon_bend",
        "implicit_carbon_branch",
        "ring_closure",
        "heteroatom_heteroatom_bond",
        "two_letter_element",
        "positive_charge",
        "negative_charge",
        "isotope_label",
    }
    assert required <= {concept for record in examples for concept in record["concepts"]}
    for record in examples:
        mol = _molecule(record)
        concepts = set(record["concepts"])
        if "implicit_carbon_endpoint" in concepts:
            assert any(atom.GetAtomicNum() == 6 and atom.GetDegree() == 1 for atom in mol.GetAtoms())
        if "implicit_carbon_bend" in concepts:
            assert any(atom.GetAtomicNum() == 6 and atom.GetDegree() == 2 for atom in mol.GetAtoms())
        if "implicit_carbon_branch" in concepts:
            assert any(atom.GetAtomicNum() == 6 and atom.GetDegree() >= 3 for atom in mol.GetAtoms())
        if "ring_closure" in concepts:
            assert mol.GetRingInfo().NumRings() >= 1
        if "heteroatom_heteroatom_bond" in concepts:
            assert any(
                bond.GetBeginAtom().GetAtomicNum() not in {1, 6}
                and bond.GetEndAtom().GetAtomicNum() not in {1, 6}
                for bond in mol.GetBonds()
            )
        if "positive_charge" in concepts:
            assert Chem.GetFormalCharge(mol) > 0
        if "negative_charge" in concepts:
            assert Chem.GetFormalCharge(mol) < 0
        if "isotope_label" in concepts:
            assert any(atom.GetIsotope() > 0 for atom in mol.GetAtoms())
        if "two_letter_element" in concepts:
            assert any(len(atom.GetSymbol()) == 2 for atom in mol.GetAtoms())
