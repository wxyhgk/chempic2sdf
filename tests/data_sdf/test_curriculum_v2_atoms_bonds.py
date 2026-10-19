"""Scope, chemistry, determinism and actual coverage of v0.2 candidates."""

from collections import Counter
import re

import pytest
from rdkit import Chem

from Data.sdf.chemvector_curriculum_v0_1.catalog import merged_records
from Data.sdf.chemvector_curriculum_v0_2.atoms_bonds import (
    _validated_molecule,
    records,
)


@pytest.fixture(scope="module")
def examples():
    return records()


def test_deterministic_unique_fresh_records(examples):
    assert 900 <= len(examples) <= 1800
    assert examples == records()
    ids = [item["record_id"] for item in examples]
    assert ids == sorted(set(ids))
    assert len({item["smiles"] for item in examples}) == len(examples)
    for item in examples:
        assert set(item) == {"record_id", "stage", "name", "smiles", "concepts"}
        assert re.fullmatch(r"v02_ab_[a-f0-9]{16}", item["record_id"])
        assert item["stage"] == "atoms_bonds"
        assert item["name"]
        assert item["concepts"] == sorted(set(item["concepts"]))
    changed = records()
    changed[0]["name"] = "independence sentinel"
    changed[0]["concepts"].append("independence sentinel")
    changed.pop()
    assert records() == examples


def test_complete_small_legal_graphs_and_canonical_roundtrip(examples):
    for item in examples:
        mol = Chem.MolFromSmiles(item["smiles"], sanitize=True)
        assert mol is not None, item
        assert len(Chem.GetMolFrags(mol)) == 1, item
        assert 2 <= mol.GetNumHeavyAtoms() <= 20, item
        assert mol.GetNumBonds() > 0
        assert all(atom.GetAtomicNum() > 0 and not atom.HasQuery()
                   and not atom.GetNumRadicalElectrons() and not atom.GetAtomMapNum()
                   for atom in mol.GetAtoms()), item
        assert Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) == item["smiles"]
        assert all(bond.GetBondType() in {
            Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
            Chem.BondType.TRIPLE, Chem.BondType.AROMATIC,
        } for bond in mol.GetBonds())


def test_existing_merger_keeps_all_candidates_quarantined(examples):
    merged = merged_records(examples)
    assert len(merged) == len(examples)
    assert all(item["split"] == "quarantine" for item in merged)
    assert all(item["training_authorized"] is False for item in merged)
    assert all(item["external_overlap_status"] == "unchecked" for item in merged)


def test_graph_derived_element_bond_and_charge_tags(examples):
    elements, bond_instances = Counter(), Counter()
    for item in examples:
        mol = Chem.MolFromSmiles(item["smiles"])
        symbols = {atom.GetSymbol() for atom in mol.GetAtoms()}
        concepts = set(item["concepts"])
        elements.update(symbols)
        types = {bond.GetBondType() for bond in mol.GetBonds()}
        bond_instances.update(str(bond.GetBondType()) for bond in mol.GetBonds())
        assert {tag.removeprefix("element_") for tag in concepts
                if tag.startswith("element_")} == symbols
        for bond_type, tag in (
            (Chem.BondType.SINGLE, "single_bond"),
            (Chem.BondType.DOUBLE, "double_bond"),
            (Chem.BondType.TRIPLE, "triple_bond"),
            (Chem.BondType.AROMATIC, "aromatic_bond"),
        ):
            assert (tag in concepts) == (bond_type in types)
        assert ("positive_charge" in concepts) == (Chem.GetFormalCharge(mol) > 0)
        assert ("negative_charge" in concepts) == (Chem.GetFormalCharge(mol) < 0)
        assert ("isotope_label" in concepts) == any(atom.GetIsotope() for atom in mol.GetAtoms())
    assert set(elements) == {"C", "N", "O", "S", "P", "B", "Si", "F", "Cl", "Br", "I"}
    assert min(elements.values()) >= 18
    assert bond_instances["TRIPLE"] >= 200
    assert bond_instances["DOUBLE"] >= 200
    assert bond_instances["AROMATIC"] >= 500


def test_triple_bonds_have_distinct_real_attachment_environments(examples):
    coverage = Counter()
    triple_neighbors = set()
    for item in examples:
        mol = Chem.MolFromSmiles(item["smiles"])
        observed = set()
        for bond in mol.GetBonds():
            if bond.GetBondType() != Chem.BondType.TRIPLE:
                continue
            first, second = bond.GetBeginAtom(), bond.GetEndAtom()
            numbers = {first.GetAtomicNum(), second.GetAtomicNum()}
            if numbers == {6, 7}:
                observed.add("nitrile_bond")
            elif numbers == {6}:
                observed.add("terminal_alkyne_bond" if min(first.GetDegree(), second.GetDegree()) == 1
                             else "internal_alkyne_bond")
            triple_neighbors.update(atom.GetSymbol() for atom in mol.GetAtoms())
        for tag in ("nitrile_bond", "terminal_alkyne_bond", "internal_alkyne_bond"):
            assert (tag in item["concepts"]) == (tag in observed)
        coverage.update(observed)
    assert coverage["terminal_alkyne_bond"] >= 30
    assert coverage["internal_alkyne_bond"] >= 150
    assert coverage["nitrile_bond"] >= 30
    assert {"N", "O", "S", "F", "Cl", "Br", "I"} <= triple_neighbors


def test_isotopes_are_small_explicit_minority_and_stereo_is_not_fabricated(examples):
    isotope_records = [item for item in examples if "isotope_label" in item["concepts"]]
    assert len(isotope_records) == 4
    explicit_carbon = [item for item in examples if "explicit_carbon_label" in item["concepts"]]
    assert len(explicit_carbon) == 2
    for item in examples:
        mol = Chem.MolFromSmiles(item["smiles"])
        assert all(atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED for atom in mol.GetAtoms())
        assert all(bond.GetStereo() == Chem.BondStereo.STEREONONE for bond in mol.GetBonds())
        assert not ({"wedge_bond", "dash_bond"} & set(item["concepts"]))


def test_degree_two_carbon_does_not_claim_a_rendered_bend(examples):
    for item in examples:
        mol = Chem.MolFromSmiles(item["smiles"])
        degree_two_carbon = any(
            atom.GetAtomicNum() == 6 and not atom.GetIsotope() and atom.GetDegree() == 2
            for atom in mol.GetAtoms()
        )
        assert ("carbon_degree_two" in item["concepts"]) == degree_two_carbon
        assert "implicit_carbon_bend" not in item["concepts"]


def test_targeted_heteroatom_and_double_bond_gaps_have_graph_coverage(examples):
    molecules_per_motif = Counter()
    family_counts = Counter()
    for item in examples:
        mol = Chem.MolFromSmiles(item["smiles"])
        found = set()
        for bond in mol.GetBonds():
            first, second = bond.GetBeginAtom(), bond.GetEndAtom()
            symbols = tuple(sorted((first.GetSymbol(), second.GetSymbol())))
            bond_type = bond.GetBondType()
            # Neutral N-O bonds exercise hydroxylamine connectivity, rather
            # than counting charge-separated nitro N(+)-O(-) as coverage.
            if bond_type == Chem.BondType.SINGLE:
                if symbols == ("N", "N"):
                    found.add("N-N")
                elif symbols == ("N", "O") and not (
                    first.GetFormalCharge() or second.GetFormalCharge()
                ):
                    found.add("neutral_N-O")
                elif symbols == ("S", "S"):
                    found.add("S-S")
                elif symbols == ("O", "O"):
                    found.add("O-O")
            elif bond_type == Chem.BondType.DOUBLE:
                if symbols == ("C", "N"):
                    found.add("C=N")
                elif symbols == ("C", "S"):
                    found.add("C=S")
        molecules_per_motif.update(found)
        family_counts.update(item["concepts"])
    for motif in ("N-N", "neutral_N-O", "S-S", "C=N", "C=S"):
        assert molecules_per_motif[motif] >= 18, (motif, molecules_per_motif)
    for family in ("substituted_hydrazine", "substituted_hydroxylamine",
                   "disulfide", "ketimine", "thioketone"):
        assert family_counts[f"template_{family}"] == 18
    assert molecules_per_motif["O-O"] == 1  # Existing seed only, no expansion.


@pytest.mark.parametrize("smiles", [
    "C(C)(C)(C)(C)C", "C[CH2]", "C.C", "CC*", "[CH3:1]C", "C" * 21,
    "C[Mg]Br", "N->[Cu]",
])
def test_invalid_or_out_of_scope_templates_fail_instead_of_being_repaired(smiles):
    with pytest.raises(ValueError):
        _validated_molecule(smiles, "deliberate_test_case")
