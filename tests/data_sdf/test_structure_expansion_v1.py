import pytest
from rdkit import Chem

from Data.sdf.case_diversity_v1.catalog import pool
from Data.sdf.structure_expansion_v1.catalog import records, new_candidates, connection_identity, scaffold


@pytest.fixture(scope='module')
def catalog():
    old, _ = pool()
    new, report = new_candidates(old)
    return old, new, report


def test_templates_are_unique_records_and_bounded_valid_graphs():
    rs = records()
    assert len({r['record_id'] for r in rs}) == len(rs)
    for r in rs:
        mol = Chem.MolFromSmiles(r['smiles'])
        assert mol and 2 <= mol.GetNumHeavyAtoms() <= 48
        assert len(Chem.GetMolFrags(mol)) == 1


def test_genuinely_new_connections_and_scaffolds_not_only_isotope_changes(catalog):
    old, new, report = catalog
    assert len(new) > 1000 and report['distinct_new_connections'] > 1000
    assert report['distinct_new_nonempty_scaffolds'] > 400 and report['new_acyclic_connections'] > 200
    assert not ({e['canonical_smiles'] for e in old} & {e['canonical_smiles'] for e in new})
    assert len({e['molecule_id'] for e in new}) == len(new)
    for e in new:
        assert e['split'] == 'quarantine' and not e['training_authorized']
        assert e['external_overlap_status'] == 'unchecked'


def test_novelty_does_not_count_isotope_or_stereo_as_connectivity():
    assert connection_identity('[13CH3]O') == connection_identity('CO')
    assert connection_identity('N[C@H](C)CO') == connection_identity('N[C@@H](C)CO')
    assert scaffold('CCO') == ''
    assert scaffold('c1ccccc1O') == scaffold('Cc1ccccc1')
    assert scaffold('c1ccncc1') != scaffold('c1ccccc1')


def test_new_selection_is_reproducible_against_reference_order(catalog):
    old, new, report = catalog
    again, other = new_candidates(list(reversed(old)))
    assert new == again and report == other


def test_all_candidates_sdf_identity_and_stereo_roundtrip(catalog):
    for e in catalog[1]:
        mol = Chem.MolFromSmiles(e['canonical_smiles'])
        restored = Chem.MolFromMolBlock(Chem.MolToMolBlock(mol), removeHs=False)
        assert restored is not None
        assert Chem.MolToSmiles(restored) == e['canonical_smiles']
