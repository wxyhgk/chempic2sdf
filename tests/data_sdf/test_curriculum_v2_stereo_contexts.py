"""Template-only checks: no training/dev/sealed data access or rendering."""

from collections import Counter, defaultdict

import pytest
from rdkit import Chem

from Data.sdf.chemvector_curriculum_v0_1.catalog import merged_records
from Data.sdf.chemvector_curriculum_v0_2 import stereo_contexts as source


@pytest.fixture(scope="module")
def candidates():
    return source.records()


def test_schema_unique_identity_stable_ids_and_fresh_records(candidates):
    assert len(candidates) >= 500
    assert len({r["record_id"] for r in candidates}) == len(candidates)
    assert len({r["smiles"] for r in candidates}) == len(candidates)
    assert all(r["record_id"].startswith("v02_st_") for r in candidates)
    assert all(r["stage"] == "stereo_contexts" for r in candidates)
    assert candidates == source.records()
    fresh = source.records()
    fresh[0]["concepts"].append("mutation")
    assert "mutation" not in source.records()[0]["concepts"]
    assert "mutation" not in candidates[0]["concepts"]


def test_complete_small_ordinary_molecules_and_canonical_roundtrip(candidates):
    allowed = {Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
               Chem.BondType.TRIPLE, Chem.BondType.AROMATIC}
    for record in candidates:
        mol = Chem.MolFromSmiles(record["smiles"])
        assert mol is not None
        assert 2 <= mol.GetNumHeavyAtoms() <= 24
        assert len(Chem.GetMolFrags(mol)) == 1
        assert all(a.GetAtomicNum() and not a.GetNumRadicalElectrons()
                   and not a.GetAtomMapNum() for a in mol.GetAtoms())
        assert {b.GetBondType() for b in mol.GetBonds()} <= allowed
        assert Chem.MolToSmiles(mol, isomericSmiles=True) == record["smiles"]


def test_real_tetrahedral_pairs_ez_pairs_and_complete_two_center_families(candidates):
    families = defaultdict(list)
    for record in candidates:
        mol = Chem.MolFromSmiles(record["smiles"])
        families[Chem.MolToSmiles(mol, isomericSmiles=False)].append(mol)
    family_sizes = Counter()
    for family in families.values():
        assert len({Chem.MolToSmiles(m) for m in family}) == len(family)
        tetra = [Chem.FindMolChiralCenters(m, includeUnassigned=True) for m in family]
        assert all(all(label in {"R", "S"} for _, label in centers) for centers in tetra)
        if len(tetra[0]) == 1:
            assert len(family) == 2
            assert {centers[0][1] for centers in tetra} == {"R", "S"}
            family_sizes["tetrahedral"] += 1
        elif len(tetra[0]) == 2:
            assert len(family) == 4
            assert {tuple(label for _, label in centers) for centers in tetra} == {
                ("R", "R"), ("R", "S"), ("S", "R"), ("S", "S")}
            family_sizes["two_center"] += 1
        else:
            assert not any(tetra)
            assert len(family) == 2
            stereos = [{b.GetStereo() for b in m.GetBonds()
                        if b.GetStereo() != Chem.BondStereo.STEREONONE} for m in family]
            assert {frozenset(s) for s in stereos} == {
                frozenset({Chem.BondStereo.STEREOE}), frozenset({Chem.BondStereo.STEREOZ})}
            family_sizes["alkene"] += 1
    assert family_sizes["tetrahedral"] >= 200
    assert family_sizes["alkene"] >= 30
    assert family_sizes["two_center"] >= 4


def test_concepts_are_verified_properties_not_wedge_promises(candidates):
    coverage = Counter()
    for record in candidates:
        mol = Chem.MolFromSmiles(record["smiles"])
        concepts = set(record["concepts"])
        coverage.update(concepts)
        assert not any("wedge" in c or "dash" in c for c in concepts)
        centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        assert ("tetrahedral_stereochemistry" in concepts) == bool(centers)
        for configuration in ("R", "S"):
            assert (f"stereocenter_{configuration}" in concepts) == any(
                label == configuration for _, label in centers)
        for code, label in ((Chem.BondStereo.STEREOE, "E"), (Chem.BondStereo.STEREOZ, "Z")):
            assert (f"double_bond_stereo_{label}" in concepts) == any(
                b.GetStereo() == code for b in mol.GetBonds())
        for kind, label in ((Chem.BondType.SINGLE, "single_bond"),
                            (Chem.BondType.DOUBLE, "double_bond"),
                            (Chem.BondType.TRIPLE, "triple_bond"),
                            (Chem.BondType.AROMATIC, "aromatic_bond")):
            assert (label in concepts) == any(b.GetBondType() == kind for b in mol.GetBonds())
    assert coverage["stereocenter_R"] == coverage["stereocenter_S"]
    assert coverage["double_bond_stereo_E"] == coverage["double_bond_stereo_Z"]
    assert coverage["aromatic_environment"] > 0
    assert coverage["heteroaromatic_environment"] > 0
    assert coverage["triple_bond"] > 0


def test_legacy_merge_keeps_quarantine_and_retains_stereo(candidates):
    merged = merged_records(candidates)
    assert len(merged) == len(candidates)
    assert {r["canonical_smiles"] for r in merged} == {r["smiles"] for r in candidates}
    assert all(r["split"] == "quarantine" and not r["training_authorized"]
               and r["external_overlap_status"] == "unchecked" for r in merged)


@pytest.mark.parametrize("smiles,tetrahedral,alkenes", [
    ("C[C@H](O)C", 1, 0),  # Identical methyl ligands: no real center.
    ("CC(O)CC", 1, 0),  # A real center left unspecified.
    ("C/C=C", 0, 1),  # Terminal alkene cannot retain E/Z.
    ("C/C=C/C", 1, 1),  # No tetrahedral center.
    ("C[C@H](O)C.*", 1, 0),  # Disconnected wildcard is not a molecule.
])
def test_invalid_or_disappearing_stereo_is_rejected(smiles, tetrahedral, alkenes):
    with pytest.raises(ValueError):
        source._validated(smiles, tetrahedral=tetrahedral, alkenes=alkenes)
