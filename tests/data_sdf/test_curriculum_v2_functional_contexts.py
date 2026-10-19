"""Checks for isolated synthetic functional-context candidates; no file IO."""

from collections import Counter
from hashlib import sha256

from rdkit import Chem

from Data.sdf.chemvector_curriculum_v0_1.catalog import merged_records
from Data.sdf.chemvector_curriculum_v0_1.functional_groups import annotate_groups
from Data.sdf.chemvector_curriculum_v0_2.functional_contexts import (
    CATALOG_STATUS,
    PROVENANCE_VERSION,
    records,
)


def test_candidate_scope_determinism_uniqueness_and_independent_records():
    rows = records()
    assert CATALOG_STATUS == "draft/quarantine"
    assert 1000 <= len(rows) <= 1400
    assert rows == records()
    assert len({row["record_id"] for row in rows}) == len(rows)
    assert len({row["smiles"] for row in rows}) == len(rows)
    for row in rows:
        assert row["stage"] == "functional_groups"
        assert row["record_id"] == "v02_fg_" + sha256(row["smiles"].encode()).hexdigest()[:16]
        assert row["split"] == "quarantine"
        assert row["training_authorized"] is False
        assert row["external_overlap_status"] == "unchecked"
    rows[0]["concepts"].append("untrue")
    rows[0]["provenance"]["sources"][0]["context_id"] = "changed"
    assert rows != records()
    assert records() == records()


def test_all_structures_are_complete_valid_and_all_group_claims_are_real():
    for row in records():
        mol = Chem.MolFromSmiles(row["smiles"])
        assert mol is not None, row["record_id"]
        assert Chem.SanitizeMol(mol, catchErrors=True) == Chem.SanitizeFlags.SANITIZE_NONE
        assert 3 <= mol.GetNumHeavyAtoms() <= 24
        assert len(Chem.GetMolFrags(mol)) == 1
        assert all(atom.GetAtomicNum() > 0 for atom in mol.GetAtoms())
        assert all(not atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())
        assert all(not atom.GetAtomMapNum() for atom in mol.GetAtoms())
        assert sum(atom.GetFormalCharge() for atom in mol.GetAtoms()) == 0
        assert Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) == row["smiles"]
        assert row["concepts"] == sorted({hit["name"] for hit in annotate_groups(mol)})
        assert row["concepts"]


def test_provenance_reconstructs_candidates_without_corpus_access():
    paths = set()
    for row in records():
        provenance = row["provenance"]
        assert provenance["kind"] == "hand_authored_templates"
        assert provenance["version"] == PROVENANCE_VERSION
        assert provenance["sources"]
        for source in provenance["sources"]:
            mol = Chem.MolFromSmiles(source["constructed_smiles"])
            assert mol is not None
            assert Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) == row["smiles"]
            path = (source["template_id"], source["context_id"])
            assert path not in paths
            paths.add(path)
    assert len(paths) == 32 * 39
    assert len({template for template, _ in paths}) == 32
    assert len({context for _, context in paths}) == 39


def test_varied_functional_groups_bond_orders_and_connection_environments():
    groups, molecule_bonds, elements = Counter(), Counter(), Counter()
    triple_contexts, ring_molecules, branched_molecules = set(), 0, 0
    for row in records():
        mol = Chem.MolFromSmiles(row["smiles"])
        groups.update(row["concepts"])
        bond_types = {str(bond.GetBondType()) for bond in mol.GetBonds()}
        molecule_bonds.update(bond_types)
        elements.update({atom.GetSymbol() for atom in mol.GetAtoms()})
        ring_molecules += mol.GetRingInfo().NumRings() > 0
        branched_molecules += any(atom.GetDegree() >= 3 for atom in mol.GetAtoms())
        if "TRIPLE" in bond_types:
            triple_contexts.update(source["context_id"] for source in row["provenance"]["sources"])
    assert set(groups) == {
        "alcohol", "phenol", "ether", "aldehyde", "ketone", "carboxylic_acid",
        "ester", "amine", "amide", "nitrile", "nitro", "thiol", "thioether",
        "sulfoxide", "sulfone", "sulfonamide",
    }
    assert min(groups.values()) >= 35
    assert molecule_bonds["TRIPLE"] >= 150
    assert molecule_bonds["DOUBLE"] >= 500
    assert molecule_bonds["AROMATIC"] >= 300
    assert triple_contexts >= {"phenyl", "cyclopropyl", "isopropyl", "propargyl", "pyridin_2_yl"}
    assert ring_molecules >= 500
    assert branched_molecules >= 700
    assert set(elements) >= {"C", "N", "O", "S", "F", "Cl", "Br", "B", "P", "Si"}


def test_existing_merge_contract_preserves_quarantine_and_identity():
    source = records()
    merged = merged_records(source)
    assert len(merged) == len(source)
    assert {row["canonical_smiles"] for row in merged} == {row["smiles"] for row in source}
    assert all(row["split"] == "quarantine" for row in merged)
    assert all(row["training_authorized"] is False for row in merged)
    assert all(row["external_overlap_status"] == "unchecked" for row in merged)
