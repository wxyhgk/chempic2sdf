from copy import deepcopy

import pytest
from rdkit import Chem

from chemvector.experiments.id_graph.graph import from_chemmol
from chemvector.experiments.id_graph_diagnosis.controls import FIELDS, intervene, representation_check
from chemvector.experiments.id_graph_prediction.scoring import compile_score
from chemvector.experiments.id_graph.roundtrip import signature


def graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    chemical = {'molecule_id': 'fixture', 'atoms': [
        {'id': a.GetIdx(), 'element': a.GetSymbol(), 'formal_charge': a.GetFormalCharge(),
         'isotope': a.GetIsotope(), 'aromatic': a.GetIsAromatic(), 'explicit_h_count': a.GetNumExplicitHs(),
         'no_implicit': a.GetNoImplicit(), 'radical_electrons': a.GetNumRadicalElectrons(),
         'chiral_tag': str(a.GetChiralTag())} for a in mol.GetAtoms()], 'bonds': [
        {'id': b.GetIdx(), 'begin_atom': b.GetBeginAtomIdx(), 'end_atom': b.GetEndAtomIdx(),
         'type': str(b.GetBondType()), 'aromatic': b.GetIsAromatic(), 'conjugated': b.GetIsConjugated(),
         'stereo': str(b.GetStereo()), 'stereo_atoms': list(b.GetStereoAtoms())} for b in mol.GetBonds()]}
    render = {'molecule_id': 'fixture', 'variant': 'fixture', 'atoms': [
        {'atom_id': i, 'image_x': 30. * i, 'image_y': 0.} for i in range(mol.GetNumAtoms())]}
    return from_chemmol(chemical, render)


@pytest.mark.parametrize('mode', list(FIELDS))
def test_attribute_modes_only_change_declared_fields(mode):
    pred, truth = graph('CC'), graph('C[NH3+]')
    before = deepcopy(pred)
    actual, metrics = intervene(pred, truth, mode)
    assert pred == before
    assert metrics['complete_safe_bijection']
    assert actual['bonds'] == pred['bonds'] and actual['observations'] == pred['observations']
    for a, b, t in zip(actual['atoms'], pred['atoms'], truth['atoms']):
        for field in b:
            assert a[field] == (t[field] if field in FIELDS[mode] else b[field])


def test_noop_is_exact():
    pred = graph('CC')
    assert intervene(pred, graph('CN'), 'none')[0] == pred


def test_bond_control_preserves_edges_incident_to_unmatched_nodes():
    pred, truth = graph('CCC'), graph('CN')
    pred['bonds'][0]['type'] = 'DOUBLE'
    actual, metrics = intervene(pred, truth, 'none', True)
    assert len(actual['atoms']) == 3
    boundary = next(b for b in actual['bonds'] if b['atom_ids'] == [1, 2])
    assert boundary == pred['bonds'][1]
    assert next(b for b in actual['bonds'] if b['atom_ids'] == [0, 1])['type'] == 'SINGLE'
    assert metrics['preserved_boundary_bonds'] == 1 and not metrics['complete_safe_bijection']


def test_ambiguous_nodes_are_never_assigned_truth_attributes():
    pred, truth = graph('CC'), graph('CN')
    for g in (pred, truth):
        for p in g['observations']['atom_positions']:
            p['xy'] = [0., 0.]
    actual, metrics = intervene(pred, truth, 'full', True)
    assert actual == pred and metrics['safe_matched_atoms'] == 0


def test_missing_atoms_are_not_added_by_full_control():
    actual, metrics = intervene(graph('C'), graph('CN'), 'full', True)
    assert len(actual['atoms']) == 1 and actual['bonds'] == []
    assert metrics['unavailable_truth_bonds'] == 1


def test_complete_full_control_matches_truth_core_without_stereo_injection():
    pred, truth = graph('CCCC'), graph('F[C@H](Cl)Br')
    actual, metrics = intervene(pred, truth, 'full', True)
    expected = {'core': signature(Chem.MolFromSmiles('F[C@H](Cl)Br'), False),
                'stereo': signature(Chem.MolFromSmiles('F[C@H](Cl)Br'))}
    result, _ = compile_score(actual, expected)
    assert metrics['complete_safe_bijection'] and result['core_graph_exact']
    assert not result['stereo_graph_exact']


def test_stereo_prediction_is_rejected_instead_of_misapplying_references():
    with pytest.raises(ValueError, match='stereo-unspecified'):
        intervene(graph('F[C@H](Cl)Br'), graph('F[C@H](Cl)Br'), 'full', True)


def test_kekule_and_aromatic_labels_normalize_to_same_bond_types():
    truth = graph('c1ccccc1')
    pred = deepcopy(truth)
    for a in pred['atoms']:
        a['aromatic'] = False
    for i, b in enumerate(pred['bonds']):
        b['type'] = 'SINGLE' if i % 2 else 'DOUBLE'
        b['aromatic'] = False
    result = representation_check(pred, truth)
    assert result['resolved'] == 6 and result['persistent'] == result['unresolved'] == 0


def test_real_saturated_unsaturated_difference_not_erased():
    result = representation_check(graph('CC'), graph('C=C'))
    assert result['persistent'] == 1 and result['resolved'] == 0


def test_invalid_graph_equivalence_is_unresolved():
    pred = graph('CC')
    pred['bonds'][0].update(type='AROMATIC', aromatic=True)
    result = representation_check(pred, graph('CC'))
    assert result['unresolved'] == 1 and result['resolved'] == 0


def test_existing_output_is_not_reused(tmp_path):
    from chemvector.experiments.id_graph_diagnosis.__main__ import freeze

    with pytest.raises(FileExistsError):
        freeze(tmp_path)
