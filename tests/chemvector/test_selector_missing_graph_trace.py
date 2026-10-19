"""Synthetic saved-graph provenance and unchanged pair-label replay checks."""

from copy import deepcopy

import pytest
from rdkit import Chem
import torch

from tools import selector_missing_graph_trace as trace


def fixture(*, drop_first_bond=False, reverse_mol_order=False):
    targets = torch.tensor([[.2, .5], [.4, .5], [.6, .5]]).tolist()
    points = [[targets[0][0] + 7 / 768, .5], targets[1], targets[2]]
    candidates = [dict(slot=i, geometry=p + [0.] * 6) for i, p in enumerate(points)]
    geometry = dict(image_size=768, match_radius_px=6, target_anchors=targets, candidates=candidates)
    atoms = [dict(atom_id=i, element='N' if i == 0 else 'C', text='N' if i == 0 else '') for i in range(3)]
    bonds = [dict(bond_id=i, begin_atom=i, end_atom=i + 1, chemical_type='SINGLE') for i in range(2)]
    layers = dict(atoms=atoms, bonds=bonds)
    mol = Chem.RWMol()
    order = [2, 1, 0] if reverse_mol_order else [0, 1, 2]
    by_slot = {}
    for slot in order:
        by_slot[slot] = mol.AddAtom(Chem.Atom(atoms[slot]['element']))
    for b in bonds:
        if not (drop_first_bond and b['bond_id'] == 0):
            mol.AddBond(by_slot[b['begin_atom']], by_slot[b['end_atom']], Chem.BondType.SINGLE)
    conf = Chem.Conformer(3)
    for slot, index in by_slot.items():
        x, y = [round(v * 1023) / 100 for v in points[slot]]
        conf.SetAtomPosition(index, (x, -y, 0))
    mol.AddConformer(conf)
    Chem.SanitizeMol(mol)
    prediction = dict(molblock=Chem.MolToMolBlock(mol), valid=True, exact=not drop_first_bond,
                      fallback=False, error=None)
    record = dict(sample_key='synthetic', prediction=prediction,
                  metrics=dict(logits={'0': 1., '1': 1., '2': 1.}, selected_atoms=3,
                               matched_target_ids=[1, 2]))
    return geometry, record, layers


@pytest.mark.parametrize('reverse', [False, True])
def test_graph_lineage_uses_saved_coordinates_not_atom_order(reverse):
    geo, record, layers = fixture(reverse_mol_order=reverse)
    before = deepcopy((geo, record, layers))
    rng = torch.get_rng_state().clone()
    result = trace.inspect_case(geo, record, layers, [0])
    node = result['focused_nodes'][0]
    assert node['local_status'] == 'element_and_incident_bonds_correct'
    assert node['compiled_node']['element'] == 'N'
    assert node['compiled_node']['slot'] == 0
    assert node['compiled_node']['mol_atom_index'] == (2 if reverse else 0)
    assert node['selected_correspondence']['distance_px'] == pytest.approx(7)
    assert node['official_selected_target'] == -1
    assert node['expected_incident_bonds'][0]['current_negative_pool_eligible'] is True
    assert not node['missing_bond_ids']
    assert (geo, record, layers) == before
    assert torch.equal(torch.get_rng_state(), rng)


def test_preserved_element_can_have_missing_connection():
    geo, record, layers = fixture(drop_first_bond=True)
    node = trace.inspect_case(geo, record, layers, [0])['focused_nodes'][0]
    assert node['element_correct']
    assert node['missing_bond_ids'] == [0]
    assert node['local_status'] == 'incident_bond_error'


def test_no_molblock_is_unknown_not_a_deleted_atom():
    geo, record, layers = fixture()
    record['prediction'].update(molblock=None, valid=False, exact=False, error='sanitize failed')
    node = trace.inspect_case(geo, record, layers, [0])['focused_nodes'][0]
    assert node['local_status'] == 'no_saved_final_graph'
    assert node['selected_correspondence']['slot'] == 0
    assert node['compiled_node'] is None
    assert node['missing_bond_ids'] is None


def test_negative_pool_is_replayed_without_promoting_diagnostic_matches():
    geo, record, layers = fixture()
    mapping, labels = trace.replay_pair_labels(geo, record, layers)
    assert mapping == [-1, 1, 2]
    assert labels == {(0, 1): False, (0, 2): False, (1, 2): True}
    assert set(labels) == {(0, 1), (0, 2), (1, 2)}


def test_disagreed_historical_matching_fails_closed():
    geo, record, layers = fixture()
    record['metrics']['matched_target_ids'] = [0, 1, 2]
    with pytest.raises(ValueError, match='matcher replay differs'):
        trace.replay_pair_labels(geo, record, layers)


def test_duplicate_spatial_claimants_stay_unresolved():
    geo, record, _ = fixture()
    selected = trace.selected_candidates(geo, record)
    selected.append(dict(slot=3, geometry=[.201, .5] + [0.] * 6))
    rows = trace.spatial_correspondence(geo, selected)
    assert all(not r['accepted_for_diagnostic_only'] for r in rows if r['target_id'] == 0)


def test_remote_spatial_nearest_is_not_forced_to_match():
    geo, record, _ = fixture()
    selected = trace.selected_candidates(geo, record)
    selected[0]['geometry'][:2] = [.2, .8]
    row = trace.spatial_correspondence(geo, selected)[0]
    assert not row['accepted_for_diagnostic_only']
    assert 'outside_local_separation_ball' in row['unresolved_reasons']


@pytest.mark.parametrize('failure', ['moved_coordinate', 'fallback', 'collision', 'count'])
def test_ambiguous_or_changed_saved_graph_lineage_fails(failure):
    geo, record, _ = fixture()
    selected = trace.selected_candidates(geo, record)
    if failure == 'moved_coordinate':
        mol = Chem.MolFromMolBlock(record['prediction']['molblock'], removeHs=False)
        p = mol.GetConformer().GetAtomPosition(0)
        p.x += .01
        mol.GetConformer().SetAtomPosition(0, p)
        record['prediction']['molblock'] = Chem.MolToMolBlock(mol)
    elif failure == 'fallback':
        record['prediction']['fallback'] = True
    elif failure == 'collision':
        selected[1]['geometry'] = selected[0]['geometry'][:]
    elif failure == 'count':
        selected.pop()
    with pytest.raises(ValueError):
        trace.compiled_graph(record['prediction'], selected)


def test_aromatic_kekule_serialization_is_normalized_before_comparison():
    mol = Chem.MolFromSmiles('c1ccccc1')
    conf = Chem.Conformer(6)
    selected = []
    for i in range(6):
        x, y = (100 + 100 * (i % 3)), (100 + 100 * (i // 3))
        conf.SetAtomPosition(i, (x / 100, -y / 100, 0))
        selected.append(dict(slot=i, geometry=[x / 1023, y / 1023] + [0.] * 6))
    mol.AddConformer(conf)
    pred = dict(molblock=Chem.MolToMolBlock(mol), valid=True, exact=True, fallback=False)
    graph = trace.compiled_graph(pred, selected)
    assert {b['chemical_type'] for b in graph['edges']} == {'AROMATIC'}


def test_changed_element_is_not_hidden_by_spatial_association():
    geo, record, layers = fixture()
    mol = Chem.MolFromMolBlock(record['prediction']['molblock'], removeHs=False)
    mol.GetAtomWithIdx(0).SetAtomicNum(8)
    record['prediction']['molblock'] = Chem.MolToMolBlock(mol)
    node = trace.inspect_case(geo, record, layers, [0])['focused_nodes'][0]
    assert node['local_status'] == 'wrong_element'
    assert not node['element_correct']
