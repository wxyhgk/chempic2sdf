"""Synthetic clean-validation tracing tests; no real images/checkpoints/GPU."""

from copy import deepcopy

import pytest
import torch

from chemvector.training.curriculum import clean_diagnostic as audit
from chemvector.training.curriculum.clean_metrics import (
    select_clean_rows, measure_clean, predicted_box, parsed_difference,
)
from tests.chemvector.test_curriculum_headroom import synthetic


def manifest():
    rows = [dict(source='curriculum', split='validation', domain='curriculum',
                 sample_key=f'm{i}__reference', molecule_id=f'm{i}', canonical_smiles=f'id{i}') for i in range(16)]
    rows.extend([{**rows[0], 'sample_key': 'training__reference', 'split': 'train'},
                 {**rows[0], 'sample_key': 'stress__font_small'},
                 {**rows[0], 'sample_key': 'full__reference', 'source': 'full_molecule', 'domain': 'full_molecule'}])
    return {'samples': rows}


def test_all_clean_validation_rows_are_selected_without_prediction_or_order_dependence():
    m = manifest()
    chosen = select_clean_rows(m)
    assert len(chosen) == 16 and all(r['split'] == 'validation' for r in chosen)
    m['samples'].reverse()
    assert select_clean_rows(m) == chosen
    chosen[0]['split'] = 'train'
    assert select_clean_rows(m)[0]['split'] == 'validation'


@pytest.mark.parametrize('bad', ['short', 'duplicate', 'same_identity', 'same_molecule'])
def test_no_replacement_from_training_or_other_styles(bad):
    m = manifest()
    if bad == 'short': m['samples'].pop(0)
    elif bad == 'duplicate': m['samples'].append(m['samples'][0])
    elif bad == 'same_identity': m['samples'][1]['canonical_smiles'] = m['samples'][0]['canonical_smiles']
    else: m['samples'][1]['molecule_id'] = m['samples'][0]['molecule_id']
    with pytest.raises(ValueError): select_clean_rows(m)


def args():
    values = list(synthetic())
    values[0]['meta'].update(split='validation', profile_id='reference')
    values[3]['output']['chemical_logits'] = torch.tensor([[10., 0., 0., 0.]])
    values[4]['ocr'][0]['score'] = -.1
    values.append({1: {'text': 'O', 'score': -.01}})
    return values


def test_baseline_trace_and_reference_ocr_are_separate_and_inert():
    values = args()
    values[4]['ocr'][0]['text'] = 'N'
    before = deepcopy(values[4])
    m = measure_clean(*values)
    label = m['atoms'][1]
    assert label['predicted_crop']['text'] == 'N' and label['reference_crop']['text'] == 'O'
    assert label['predicted_crop']['basic_field_differences']['element'] == {'expected': 'O', 'predicted': 'N'}
    assert label['reference_crop']['basic_field_differences'] == {}
    assert m['bonds'][0]['reachable'] and m['bonds'][0]['present']
    assert m['coverage']['target_bonds'] == 1 and m['candidate_bijection']
    assert values[4] == before


def test_closed_text_gate_does_not_turn_reference_ocr_into_model_prediction():
    values = args()
    values[4]['ocr'] = []
    values[4]['roles'][1]['text'] = False
    label = measure_clean(*values)['atoms'][1]
    assert label['predicted_crop'] is None and label['predicted_text'] is None
    assert label['reference_crop']['text'] == 'O' and not label['role']['text']


def test_unreachable_bond_and_near_raw_candidate_stay_visible():
    values = args()
    values[3]['atoms']['atom_mask'] = torch.tensor([[True, False, False]])
    values[3]['pairs'] = torch.empty(0, 2, dtype=torch.long)
    values[3]['output'] = {'presence_logits': torch.empty(0), 'chemical_logits': torch.empty(0, 4)}
    values[3]['chemistry'] = torch.empty(0, dtype=torch.long)
    values[4] = {'roles': [{'candidate': 0, 'text': False}], 'ocr': []}
    result = measure_clean(*values)
    assert result['coverage']['target_bonds'] == 1
    assert result['bonds'][0]['reachable'] is False and result['bonds'][0]['presence_probability'] is None
    assert result['atoms'][1]['candidate'] is None
    assert result['atoms'][1]['coverage']['near_raw_ids'] == [1]
    assert result['atoms'][1]['nearest_raw'] == {'candidate': 1, 'distance_px': 0.}


@pytest.mark.parametrize('bad', ['train', 'style', 'purpose', 'unchecked', 'reference_missing', 'role_missing'])
def test_wrong_scope_or_incomplete_traces_fail_closed(bad):
    values = args()
    if bad == 'train': values[0]['meta']['split'] = 'train'
    elif bad == 'style': values[0]['meta']['profile_id'] = 'font_small'
    elif bad == 'purpose': values[0]['purpose'] = 'training'
    elif bad == 'unchecked': values[0]['meta']['external_overlap_status'] = 'unchecked'
    elif bad == 'reference_missing': values[-1] = {}
    else: values[4]['roles'].pop()
    with pytest.raises(ValueError): measure_clean(*values)


def test_reference_boxes_use_only_visible_labels_and_fixed_production_context():
    sample = args()[0]
    assert audit.reference_crops(sample) == [(1, sample['layers']['atoms'][1]['bbox'])]
    assert predicted_box({'center': [.5, .5], 'half_extent': [.125, .25]}) == [288., 192., 480., 576.]


def test_parser_abstains_on_unsupported_label_and_preserves_charge_errors():
    expected = dict(element='N', formal_charge=1, isotope=0, radical_electrons=0)
    assert parsed_difference('HN', expected)['parsed'] is None
    assert parsed_difference('NH3', expected)['basic_field_differences'] == {'formal_charge': {'expected': 1, 'predicted': 0}}


def test_launcher_is_fixed_cpu_only_and_has_no_training_or_threshold_knob(monkeypatch):
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return {'status': 'completed'}
    monkeypatch.setattr(audit, 'run_budgeted', run)
    assert audit.main([]) == 0
    command, kwargs = calls[0]
    assert command[-2:] == ['chemvector.training.curriculum.clean_diagnostic', '--worker']
    assert kwargs['budget'].wall_seconds == 300 and kwargs['budget'].gpu_devices == ()
    assert 'training' not in kwargs
    with pytest.raises(SystemExit): audit.main(['--pair-threshold', '.1'])
