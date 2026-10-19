"""No real corpus/checkpoint access; target coverage is not measured accuracy."""

from copy import deepcopy

import pytest
from rdkit import Chem

from chemvector.training.curriculum import pair_readiness_audit as audit
from chemvector.training.curriculum.pair_readiness import (
    chemistry, training_inventory, reference_training_rows, geometry_negatives,
    describe_reference, summarize_references,
)


def row(smiles='CNN', key='a__reference', split='train', source='curriculum'):
    return dict(source=source, domain='curriculum' if source == 'curriculum' else 'full_molecule',
                split=split, canonical_smiles=smiles, sample_key=key,
                molecule_id=smiles, group_key=smiles)


def target(smiles, positions):
    mol = Chem.MolFromSmiles(smiles)
    graph = {'atoms': [{'atom_id': a.GetIdx(), 'element': a.GetSymbol()} for a in mol.GetAtoms()],
             'bonds': [{'begin_atom': b.GetBeginAtomIdx(), 'end_atom': b.GetEndAtomIdx()}
                       for b in mol.GetBonds()]}
    layers = {'atoms': [{'atom_id': i, 'center': xy} for i, xy in enumerate(positions)]}
    return graph, layers


def test_inventory_distinguishes_images_identities_small_and_large_motifs():
    m = {'samples': [row(), row(key='a__small'), row('N=N', 'b__reference'),
                     row('CCCCCCCCNS', 'full', source='full'), row('NN', 'v', split='validation')]}
    facts = training_inventory(m)
    assert facts['curriculum']['all']['images'] == 3
    assert facts['curriculum']['all']['molecules'] == 2
    assert facts['curriculum']['all']['motifs']['N-N:SINGLE']['images'] == 2
    assert facts['curriculum']['all']['motifs']['N-N:SINGLE']['molecules'] == 1
    assert facts['curriculum']['all']['motifs']['N-N:DOUBLE']['molecules'] == 1
    assert facts['full']['all']['motifs']['N-S:SINGLE']['molecules'] == 1
    assert facts['full']['small_2_to_7']['images'] == 0


def test_reference_selection_is_deterministic_train_only_without_style_fallback():
    m = {'samples': [row('NS', 'b__reference'), row(), row(key='a__font_small'),
                     row('NN', 'v__reference', split='validation')]}
    before = deepcopy(m)
    selected = reference_training_rows(m)
    assert [r['sample_key'] for r in selected] == ['a__reference', 'b__reference']
    m['samples'].reverse()
    assert reference_training_rows(m) == selected
    selected[0]['split'] = 'validation'
    assert m['samples'] == before['samples'][::-1]


@pytest.mark.parametrize('bad', ['empty', 'duplicate', 'identity_overlap', 'group_overlap'])
def test_reference_selection_fails_closed(bad):
    m = {'samples': [row()]}
    if bad == 'empty': m['samples'][0]['split'] = 'validation'
    elif bad == 'duplicate': m['samples'].append(row())
    elif bad == 'identity_overlap': m['samples'].append(row(key='val__reference', split='validation'))
    else: m['samples'].append({**row('NN', 'val', split='validation'), 'group_key': 'CNN'})
    with pytest.raises(ValueError): reference_training_rows(m)


def test_collinear_shared_neighbor_is_a_confirmed_negative_not_a_true_edge():
    graph, layers = target('O=S=O', [(0, 0), (72, 0), (144, 0)])
    result = geometry_negatives(graph, layers)
    assert result['structured_negative_pairs'] == [[0, 2]]
    assert result['intervening_nonedge_pairs'] == [{'pair': [0, 2], 'witnesses': [
        {'middle_atom': 1, 'fraction': .5, 'distance_px': 0., 'elements': ['O', 'S', 'O']}]}]
    record = describe_reference(row('O=S=O'), graph, layers)
    assert record['coverage_counts']['O-S-O_nonedge'] == 1
    assert record['eligible_clean_pair_stage']


@pytest.mark.parametrize('smiles,positions', [
    ('OO', [(0, 0), (72, 0)]),
    ('O.S.O', [(0, 0), (72, 0), (144, 0)]),
    ('O=S=O', [(0, 0), (72, 12), (144, 0)]),
    ('O=S=O', [(0, 0), (180, 0), (144, 0)]),
    ('O=S=O', [(0, 0), (0, 0), (0, 0)]),
])
def test_geometry_does_not_invent_crossing_negatives(smiles, positions):
    assert not geometry_negatives(*target(smiles, positions))['intervening_nonedge_pairs']


def test_real_oo_bond_is_a_required_positive_control():
    graph, layers = target('OO', [(0, 0), (72, 0)])
    record = describe_reference(row('OO'), graph, layers)
    assert record['coverage_counts']['O-O:SINGLE'] == 1
    assert record['coverage_counts']['intervening_nonedge'] == 0


@pytest.mark.parametrize('smiles', ['CCCCCCCCNS', 'C[C@H](O)N', 'C/C=C/C', 'N'])
def test_out_of_scope_records_are_reported_not_silently_counted_as_clean_coverage(smiles):
    facts = chemistry(smiles)
    graph, layers = target(smiles, [(i * 20., 0.) for i in range(Chem.MolFromSmiles(smiles).GetNumAtoms())])
    record = describe_reference(row(smiles), graph, layers)
    assert not record['eligible_clean_pair_stage'] and record['exclusion_reasons']
    summary = summarize_references([record])
    assert summary['reference_training_molecules'] == summary['excluded_molecules'] == 1
    assert summary['eligible_molecules'] == 0 and not summary['minimal_target_coverage_passed']
    assert facts['heavy_atoms'] > 0


def test_minimal_coverage_never_grants_training_or_claims_candidate_reachability():
    records = []
    for smiles in ('CNN', 'CNS', 'OO', 'O=S=O'):
        n = Chem.MolFromSmiles(smiles).GetNumAtoms()
        graph, layers = target(smiles, [(i * 72., 0.) for i in range(n)])
        records.append(describe_reference(row(smiles), graph, layers))
    summary = summarize_references(records)
    assert summary['minimal_target_coverage_passed'] and not summary['missing_required_motifs']
    assert summary['training_ready'] is False


def test_launcher_cannot_launch_training_or_read_images(monkeypatch):
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return {'status': 'completed'}
    monkeypatch.setattr(audit, 'run_budgeted', run)
    assert audit.main([]) == 0
    command, kwargs = calls[0]
    assert command[-2:] == ['chemvector.training.curriculum.pair_readiness_audit', '--worker']
    assert kwargs['budget'].gpu_devices == () and kwargs['budget'].wall_seconds == 180
    assert 'training' not in kwargs
    with pytest.raises(SystemExit): audit.main(['--train'])
