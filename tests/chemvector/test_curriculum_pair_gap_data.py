"""Synthetic pair-gap contracts; no corpus, renderer, model, or identity index."""

from collections import Counter
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import pytest
from rdkit import Chem

from chemvector.training.curriculum import pair_gap_data as gap


def canonical(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True, isomericSmiles=True)


def entry(smiles):
    smiles = canonical(smiles)
    return {'molecule_id': 'synthetic_' + sha256(smiles.encode()).hexdigest()[:16],
            'canonical_smiles': smiles}


def released_row(smiles, index=0, **changes):
    result = {'canonical_smiles': canonical(smiles), 'source': 'curriculum',
              'domain': 'curriculum', 'split': 'train',
              'sample_key': f'synthetic_{index}__reference', 'molecule_id': f'synthetic_{index}'}
    result.update(changes)
    return result


def controls_manifest():
    return {'samples': [released_row(smiles, i) for i, smiles in enumerate(gap.CONTROL_SMILES)]}


def target(smiles, positions=None, *, key=None):
    molecule = Chem.MolFromSmiles(smiles)
    positions = positions or [(float(i * 72), 0.) for i in range(molecule.GetNumAtoms())]
    assert len(positions) == molecule.GetNumAtoms()
    return {'key': key or smiles,
            'graph': {'canonical_smiles': canonical(smiles),
                      'atoms': [{'atom_id': a.GetIdx(), 'element': a.GetSymbol()}
                                for a in molecule.GetAtoms()],
                      'bonds': [{'begin_atom': b.GetBeginAtomIdx(), 'end_atom': b.GetEndAtomIdx(),
                                 'chemical_type': str(b.GetBondType())} for b in molecule.GetBonds()]},
            'layers': {'atoms': [{'atom_id': i, 'center': xy} for i, xy in enumerate(positions)]}}


def test_fixed_candidates_are_18_distinct_in_scope_quarantined_identities():
    before = deepcopy(gap.SPECS)
    entries = gap.candidate_entries()
    assert gap.SPECS == before
    assert len(gap.SPECS) == len(entries) == 18
    assert Counter(spec[2] for spec in gap.SPECS) == {
        'N-N:SINGLE': 8, 'N-S:SINGLE': 8, 'N-S:NONEDGE': 2}
    assert len({r['canonical_smiles'] for r in entries}) == len(entries)
    assert len({gap.identities(r['canonical_smiles'])['group_key'] for r in entries}) == len(entries)
    assert [r['molecule_id'] for r in entries] == sorted(r['molecule_id'] for r in entries)
    for result in entries:
        smiles = result['canonical_smiles']
        molecule = Chem.MolFromSmiles(smiles)
        assert canonical(smiles) == smiles
        assert 2 <= molecule.GetNumAtoms() <= 7
        assert len(Chem.GetMolFrags(molecule)) == 1
        assert all(a.GetAtomicNum() > 1 and not a.GetIsotope() and not a.GetFormalCharge()
                   and not a.GetNumRadicalElectrons() and not a.GetAtomMapNum()
                   for a in molecule.GetAtoms())
        assert not gap.chemistry(smiles)['specified_stereo']
        assert result['molecule_id'] == 'curr_' + sha256(smiles.encode()).hexdigest()[:16]
        assert result['split'] == 'quarantine'
        assert result['training_authorized'] is False
        assert result['external_overlap_status'] == 'unchecked'
        assert 'pair_gap_quarantine' in result['concepts']
        concept = next(c for c in result['concepts'] if c != 'pair_gap_quarantine')
        counts = gap.target_coverage(target(smiles))['counts']
        assert counts[concept] > 0
        if concept == 'N-S:NONEDGE':
            assert not any(k.startswith('N-S:') and k != concept and v for k, v in counts.items())


@pytest.mark.parametrize('smiles', [
    'not_a_smiles', 'N', 'NNCCCCCC', 'NN.C', '[NH3+]N', '[15NH2]N',
    '[NH]N', '[NH2:1]N', '[2H]NN', 'N[NH]*', 'NNC(C)O',
    'NN[C@H](C)O', 'NN/C=C/C',
])
def test_out_of_scope_or_noncanonical_candidates_are_rejected(smiles):
    with pytest.raises(ValueError):
        gap.candidate_entries([(smiles, 'synthetic invalid candidate', 'N-N:SINGLE')])


def test_total_node_limit_is_not_merely_heavy_atom_limit():
    # Isotopic explicit H survives RDKit parsing: seven heavy atoms, eight nodes.
    smiles = canonical('[2H]NNCCCCC')
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule.GetNumHeavyAtoms() == 7 and molecule.GetNumAtoms() == 8
    with pytest.raises(ValueError):
        gap.candidate_entries([(smiles, 'explicit H outside scope', 'N-N:SINGLE')])


@pytest.mark.parametrize('specs, message', [
    ([], 'Empty'),
    ([('CNN', 'first', 'N-N:SINGLE'), ('CNN', 'second', 'N-N:SINGLE')], 'unique'),
    ([('NNC1CC1', 'first', 'N-N:SINGLE'), ('CNNC1CC1', 'second', 'N-N:SINGLE')], 'unique'),
    ([('N=N', 'double not single', 'N-N:SINGLE')], 'declared positive'),
    ([('NCCS', 'nonedge not positive', 'N-S:SINGLE')], 'declared positive'),
    ([('CNS', 'positive not nonedge', 'N-S:NONEDGE')], 'without a direct'),
    ([('CCN', 'missing sulfur', 'N-S:NONEDGE')], 'without a direct'),
    ([('CNN', 'unknown concept', 'N-N:DOUBLE')], 'Unsupported'),
])
def test_duplicate_or_false_declared_concepts_fail_closed(specs, message):
    with pytest.raises(ValueError, match=message):
        gap.candidate_entries(specs)


def test_local_identity_review_detects_full_connectivity_and_group_without_file_reads(monkeypatch):
    entries = [entry('CNN'), entry('C[C@H](N)O'), entry('Cc1ccccc1'), entry('CNS')]
    manifest = {'samples': [released_row('CNN'), released_row('CNN', 1),
                           released_row('C[C@@H](N)O', 2, split='validation'),
                           released_row('Fc1ccccc1', 3)]}
    previous_entries, previous_manifest = deepcopy(entries), deepcopy(manifest)

    def forbidden(*args, **kwargs):
        raise AssertionError('Pure local identity review must not open external files')

    monkeypatch.setattr('builtins.open', forbidden)
    monkeypatch.setattr(Path, 'open', forbidden)
    monkeypatch.setattr(Path, 'read_bytes', forbidden)
    monkeypatch.setattr(Path, 'read_text', forbidden)
    result = gap.local_identity_review(entries, manifest)
    found = [set(r['released_overlap_fields']) for r in result['candidates']]
    assert found == [{'inchikey', 'connectivity_prefix', 'group_key'},
                     {'connectivity_prefix', 'group_key'}, {'group_key'}, set()]
    assert result['released_unique_smiles'] == 3
    assert not result['local_clear']
    assert result['external_overlap_status'] == 'unchecked'
    assert result['training_authorized'] is False
    assert entries == previous_entries and manifest == previous_manifest


@pytest.mark.parametrize('manifest', [{'samples': []}, {'samples': [released_row('CCO')]}])
def test_locally_clear_candidates_never_receive_training_permission(manifest):
    result = gap.local_identity_review([entry('CNN')], manifest)
    assert result['local_clear']
    assert result['candidates'][0]['released_overlap_fields'] == []
    assert result['external_overlap_status'] == 'unchecked'
    assert result['training_authorized'] is False


def test_control_rows_select_exactly_seven_training_reference_rows_without_mutation():
    manifest = controls_manifest()
    expected = deepcopy(manifest['samples'])
    manifest['samples'] += [released_row('OO', 20, split='validation'),
                            released_row('OO', 21, source='other'),
                            released_row('OO', 22, domain='full_molecule'),
                            released_row('OO', 23, sample_key='x__font_small')]
    manifest['samples'].reverse()
    before = deepcopy(manifest)
    selected = gap.control_rows(manifest)
    assert len(selected) == 7
    assert tuple(r['canonical_smiles'] for r in selected) == gap.CONTROL_SMILES
    assert selected == expected
    selected[0]['split'] = 'validation'
    assert manifest == before


@pytest.mark.parametrize('changes', [
    {'split': 'validation'}, {'source': 'full_molecule'}, {'domain': 'full_molecule'},
    {'sample_key': 'x__font_small'},
])
def test_missing_training_control_never_falls_back_to_validation_or_other_style(changes):
    manifest = controls_manifest()
    manifest['samples'][0].update(changes)
    manifest['samples'].append(released_row('OO', 99, split='validation'))
    with pytest.raises(ValueError, match='no validation fallback'):
        gap.control_rows(manifest)


def test_duplicate_training_reference_control_is_rejected():
    manifest = controls_manifest()
    manifest['samples'].append(deepcopy(manifest['samples'][0]))
    with pytest.raises(ValueError, match='duplicate'):
        gap.control_rows(manifest)


def test_target_coverage_distinguishes_true_oo_from_collinear_o_s_o_nonedge():
    true_pair = gap.target_coverage(target('OO'))
    crossing = gap.target_coverage(target('O=S=O'))
    assert true_pair['counts']['O-O:SINGLE'] == 1
    assert true_pair['counts'].get('O-O:NONEDGE', 0) == 0
    assert true_pair['counts']['O-S-O:INTERVENING_NONEDGE'] == 0
    assert crossing['counts'].get('O-O:SINGLE', 0) == 0
    assert crossing['counts']['O-O:NONEDGE'] == 1
    assert crossing['counts']['O-S-O:INTERVENING_NONEDGE'] == 1
    assert crossing['counts']['TYPE:DOUBLE'] == 2
    assert crossing['geometry']['intervening_nonedge_pairs'][0]['pair'] == [0, 2]
    noncollinear = gap.target_coverage(target('O=S=O', [(0., 0.), (72., 12.), (144., 0.)]))
    assert noncollinear['counts']['O-O:NONEDGE'] == 1
    assert noncollinear['counts']['O-S-O:INTERVENING_NONEDGE'] == 0


@pytest.mark.parametrize('smiles, concept, count', [
    ('NN', 'N-N:SINGLE', 1), ('NCN', 'N-N:NONEDGE', 1),
    ('CNS', 'N-S:SINGLE', 1), ('NCCS', 'N-S:NONEDGE', 1),
    ('NC(N)=S', 'N-S:NONEDGE', 2),
    ('CC', 'TYPE:SINGLE', 1), ('C=C', 'TYPE:DOUBLE', 1),
    ('C#C', 'TYPE:TRIPLE', 1), ('c1ccncc1', 'TYPE:AROMATIC', 6),
])
def test_target_coverage_counts_actual_edges_nonedges_and_all_four_bond_types(smiles, concept, count):
    sample = target(smiles, key='fixed_synthetic_key')
    before = deepcopy(sample)
    result = gap.target_coverage(sample)
    assert result['sample_key'] == 'fixed_synthetic_key'
    assert result['canonical_smiles'] == canonical(smiles)
    assert result['counts'][concept] == count
    assert sample == before


def coverage_records():
    return [{**gap.target_coverage(target(smiles)), 'compatible': True, 'origin': 'synthetic'}
            for smiles in ('CNN', 'CNS', 'NCN', 'NCCS', 'OO', 'O=S=O', 'C#C', 'c1ccncc1')]


def test_complete_target_coverage_is_not_training_or_prediction_readiness():
    records = coverage_records()
    before = deepcopy(records)
    report = gap.coverage_summary(records)
    assert set(report['coverage']) == set(gap.REQUIRED_COVERAGE)
    assert report['minimal_target_coverage_passed'] and report['missing_controls'] == []
    assert all(v['molecules'] > 0 and v['instances'] > 0 for v in report['coverage'].values())
    assert report['training_authorized'] is False and report['training_ready'] is False
    assert report['predicted_reachability_checked'] is False
    assert records == before


def test_repeated_images_are_not_counted_as_distinct_molecules():
    record = {**gap.target_coverage(target('CNN')), 'compatible': True}
    report = gap.coverage_summary([record, {**record, 'sample_key': 'another_variant'}])
    assert report['coverage']['N-N:SINGLE'] == {'molecules': 1, 'instances': 2}


def test_incompatible_candidates_and_controls_cannot_satisfy_coverage():
    records = coverage_records()
    for record in records:
        if record['counts'].get('N-N:SINGLE') or record['counts'].get('TYPE:TRIPLE'):
            record['compatible'] = False
    report = gap.coverage_summary(records)
    assert not report['minimal_target_coverage_passed']
    for required in ('N-N:SINGLE', 'TYPE:TRIPLE'):
        assert required in report['missing_controls']
        assert report['coverage'][required] == {'molecules': 0, 'instances': 0}
    assert report['coverage']['O-O:SINGLE'] == {'molecules': 1, 'instances': 1}
    empty = gap.coverage_summary([])
    assert set(empty['missing_controls']) == set(gap.REQUIRED_COVERAGE)
    assert not empty['minimal_target_coverage_passed']
