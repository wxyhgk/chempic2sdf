"""Synthetic fixed-panel diagnostics; no real model or private identity calls."""

from copy import deepcopy

import pytest
import torch

from chemvector.bond_object import BOND_TYPE_TO_ID
from chemvector.training.curriculum import pair_gap_diagnostic as audit, pair_gap_metrics as metrics
from tests.chemvector.test_curriculum_headroom import synthetic


def panel_inputs():
    remaining = [dict(molecule_id=f'm{i}', canonical_smiles=f'id{i}', sample_keys=[f'm{i}__reference'],
                      image_count=1) for i in range(17)]
    controls = [dict(source='curriculum', domain='curriculum', split='train', molecule_id=f'm{i}',
                     canonical_smiles=f'id{i}', sample_key=f'm{i}__reference', meta_sha256=f'h{i}')
                for i in range(17, 24)]
    panel = dict(schema_version='chemvector-pair-gap-screened-panel-0.1', quarantine_root=str(audit.DATA),
                 training_authorized=False, training_ready=False, remaining_candidate_count=17,
                 total_inspection_images=24, existing_training_controls=deepcopy(controls),
                 new_samples=[dict(molecule_id=r['molecule_id'], sample_key=r['sample_keys'][0],
                                   profile_id='reference') for r in remaining])
    manifest = dict(samples=controls, sources={'curriculum': 'synthetic/released'})
    return panel, remaining, manifest


def test_fixed_panel_preserves_order_origins_and_does_not_mutate_admission():
    args = panel_inputs()
    before = deepcopy(args)
    rows = audit.fixed_rows(*args)
    assert len(rows) == 24
    assert all(r['origin'] == 'new_quarantine' for r in rows[:17])
    assert all(r['origin'] == 'existing_training_control' for r in rows[17:])
    assert [r['sample_key'] for r in rows] == [f'm{i}__reference' for i in range(24)]
    assert args == before


@pytest.mark.parametrize('bad', ['count', 'grant', 'ready', 'root', 'duplicate_new', 'duplicate_control',
                                 'replaced', 'dev_control', 'changed_control', 'duplicate_release'])
def test_fixed_panel_rejects_expansion_replacement_and_false_admission(bad):
    panel, remaining, manifest = panel_inputs()
    if bad == 'count': remaining.pop()
    elif bad == 'grant': panel['training_authorized'] = True
    elif bad == 'ready': panel['training_ready'] = True
    elif bad == 'root': panel['quarantine_root'] = 'elsewhere'
    elif bad == 'duplicate_new': panel['new_samples'][1] = panel['new_samples'][0]
    elif bad == 'duplicate_control': panel['existing_training_controls'][1] = panel['existing_training_controls'][0]
    elif bad == 'replaced': remaining[0]['sample_keys'] = ['replacement__reference']
    elif bad == 'dev_control': manifest['samples'][0]['split'] = 'validation'
    elif bad == 'changed_control': panel['existing_training_controls'][0]['meta_sha256'] = 'changed'
    else: manifest['samples'].append(manifest['samples'][0])
    with pytest.raises(ValueError): audit.fixed_rows(panel, remaining, manifest)


def args(origin='new_quarantine', present=True, chemical='SINGLE'):
    sample, batch, raw, structure, trace, config, threshold = synthetic(present, chemical)
    sample.update(key='m__reference')
    sample['meta'].update(molecule_id='m', profile_id='reference',
        split='quarantine' if origin == 'new_quarantine' else 'train',
        training_authorized=origin != 'new_quarantine',
        external_overlap_status='unchecked' if origin == 'new_quarantine' else 'checked')
    sample['graph'].update(canonical_smiles='CO', bonds=[dict(begin_atom=0, end_atom=1, chemical_type='SINGLE')])
    for i, atom in enumerate(sample['layers']['atoms']):
        atom['center'] = (raw['atom_geometry'][0, i] * 768).tolist()
    structure['output']['chemical_logits'] = torch.zeros(1, 4)
    structure['output']['chemical_logits'][0, BOND_TYPE_TO_ID[chemical]] = 10.
    row = dict(origin=origin, sample_key='m__reference', molecule_id='m', canonical_smiles='CO')
    return [sample, row, batch, raw, structure, trace, config, threshold]


@pytest.mark.parametrize('origin', ['new_quarantine', 'existing_training_control'])
def test_perfect_pairs_are_scored_without_relabeling_quarantine(origin):
    values = args(origin)
    meta = deepcopy(values[0]['meta'])
    m = metrics.measure(*values)
    assert m['bond_stages']['reachable_correct'] == 1
    assert m['coverage']['target_atoms'] == m['coverage']['selected_matched'] == 2
    assert m['candidate_bijection'] and m['atoms'][1]['basic_field_differences'] == {}
    assert values[0]['meta'] == meta


@pytest.mark.parametrize('present,chemical,stage', [(False, 'SINGLE', 'reachable_missing'),
    (True, 'DOUBLE', 'reachable_wrong_type'), (True, 'SINGLE', 'reachable_correct')])
def test_fixed_candidates_distinguish_missing_and_wrong_chemical(present, chemical, stage):
    m = metrics.measure(*args(present=present, chemical=chemical))
    assert m['bond_stages'][stage] == 1
    assert sum(m['bond_stages'].values()) == m['coverage']['target_bonds'] == 1


@pytest.mark.parametrize('raw_gap', [False, True])
def test_unreachable_targets_keep_denominator_and_distinguish_raw_from_selection(raw_gap):
    values = args()
    if raw_gap:
        values[3]['atom_mask'][0, 1] = False
    values[4]['atoms']['atom_mask'][0, 1] = False
    values[4].update(pairs=torch.empty(0, 2, dtype=torch.long), chemistry=torch.empty(0, dtype=torch.long),
                     output={'presence_logits': torch.empty(0), 'chemical_logits': torch.empty(0, 4)})
    values[5] = {'roles': [{'candidate': 0, 'text': False}], 'ocr': []}
    m = metrics.measure(*values)
    stage = 'raw_correspondence_gap' if raw_gap else 'selection_correspondence_gap'
    assert m['bond_stages'][stage] == 1
    assert m['coverage']['target_bonds'] == 1 and m['coverage']['reachable_positive_pairs'] == 0
    assert not m['candidate_bijection']
    assert m['atoms'][1]['candidate'] is None


def test_unknown_endpoint_pairs_are_not_confirmed_false_positives():
    values = args()
    values[4]['atoms']['atom_mask'][0, 2] = True
    values[4].update(pairs=torch.tensor([[0, 1], [0, 2], [1, 2]]), chemistry=torch.zeros(3, dtype=torch.long),
                     output={'presence_logits': torch.full((3,), 10.), 'chemical_logits': torch.zeros(3, 4)})
    values[5]['roles'].append({'candidate': 2, 'text': False})
    m = metrics.measure(*values)
    assert len(m['unknown_present_pairs']) == m['pair_contract']['unknown_pairs'] == 2
    assert m['nonedges'] == []
    assert m['pair_contract']['known_negative_pairs'] == 0


def test_confirmed_nonedge_preserves_false_positive_and_coverage():
    values = args()
    values[0]['graph']['bonds'] = []
    values[2].update(bond_mask=torch.empty(1, 0, dtype=torch.bool),
                     bond_atom_refs=torch.empty(1, 0, 2, dtype=torch.long),
                     bond_chemical=torch.empty(1, 0, dtype=torch.long))
    m = metrics.measure(*values)
    assert m['bonds'] == [] and len(m['nonedges']) == 1
    assert m['nonedges'][0]['present'] and m['nonedges'][0]['reachable']
    assert m['unknown_present_pairs'] == []


@pytest.mark.parametrize('bad', ['purpose', 'origin', 'promoted', 'overlap', 'identity', 'style',
                                 'batch_training', 'pair_order', 'roles', 'ocr', 'nan'])
def test_diagnostic_scope_and_prediction_consistency_fail_closed(bad):
    values = args()
    if bad == 'purpose': values[0]['purpose'] = 'training'
    elif bad == 'origin': values[1]['origin'] = 'validation'
    elif bad == 'promoted': values[0]['meta']['training_authorized'] = True
    elif bad == 'overlap': values[0]['meta']['external_overlap_status'] = 'checked'
    elif bad == 'identity': values[1]['canonical_smiles'] = 'CN'
    elif bad == 'style': values[0]['meta']['profile_id'] = 'font_small'
    elif bad == 'batch_training': values[2]['training_authorized'] = True
    elif bad == 'pair_order': values[4]['pairs'] = values[4]['pairs'].flip(1)
    elif bad == 'roles': values[5]['roles'].pop()
    elif bad == 'ocr': values[5]['ocr'] = []
    else: values[4]['output']['chemical_logits'][0, 0] = float('nan')
    with pytest.raises(ValueError): metrics.measure(*values)


def test_summary_keeps_origins_and_invalid_predictions_in_denominator():
    cases = []
    for origin in ('new_quarantine', 'existing_training_control'):
        values = args(origin)
        cases.append({'key': origin, 'origin': origin, 'metrics': metrics.measure(*values),
                      'prediction': {'exact': False, 'valid': False, 'fallback': False}})
    s = metrics.summarize(cases)
    assert s['all']['images'] == 2 and s['all']['exact'] == s['all']['valid'] == 0
    assert s['new_quarantine']['images'] == s['existing_training_control']['images'] == 1
    assert s['all']['bond_stages']['reachable_correct'] == 2
    with pytest.raises(ValueError): metrics.summarize(cases + [cases[0]])


def test_cpu_fixed_launcher_has_no_training_threshold_or_expansion_knobs(monkeypatch):
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return {'status': 'completed'}
    monkeypatch.setattr(audit, 'run_budgeted', run)
    assert audit.main([]) == 0
    command, kwargs = calls[0]
    assert command[-2:] == ['chemvector.training.curriculum.pair_gap_diagnostic', '--worker']
    assert kwargs['budget'].wall_seconds == 300 and kwargs['budget'].gpu_devices == ()
    assert 'training' not in kwargs
    for option in ('--max-steps', '--pair-threshold', '--samples', '--gpu'):
        with pytest.raises(SystemExit): audit.main([option, '1'])


def test_bound_input_verification_rejects_modified_content(tmp_path):
    from chemvector.training.curriculum.checkpoints import file_digest
    path = tmp_path / 'input.json'
    path.write_text('{}')
    bound = {str(path): file_digest(path)}
    audit.verify_bindings(bound)
    path.write_text('{"changed": true}')
    with pytest.raises(ValueError): audit.verify_bindings(bound)
