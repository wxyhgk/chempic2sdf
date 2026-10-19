"""No real models or evaluation data: fixed selection and opportunity metrics."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from chemvector.bond_object import BOND_TYPE_TO_ID
from chemvector.training.curriculum import headroom_metrics as metrics, headroom_audit


def manifest():
    rows = [dict(source='full', sample_key=f'm{i}_style{j}', molecule_id=f'm{i}', canonical_smiles=f'id{i}',
                 split='train', domain='full_molecule') for i in range(40) for j in range(2)]
    rows.extend([dict(source='dev', sample_key=f'd{i}', molecule_id=f'd{i}', canonical_smiles=f'dev{i}',
                      split='validation', domain='full_molecule') for i in range(20)])
    rows.append(dict(source='curriculum', sample_key='c0', molecule_id='c0', canonical_smiles='curr0',
                     split='train', domain='curriculum'))
    return {'samples': rows}


def test_selection_is_fixed_distinct_training_only_and_order_independent():
    m = manifest()
    chosen = metrics.select_training_rows(m)
    assert len(chosen) == len({r['molecule_id'] for r in chosen}) == len({r['canonical_smiles'] for r in chosen}) == 32
    assert all(r['split'] == 'train' and r['domain'] == 'full_molecule' for r in chosen)
    m['samples'].reverse()
    assert chosen == metrics.select_training_rows(m)
    chosen[0]['molecule_id'] = 'modified'
    assert all(r['molecule_id'] != 'modified' for r in m['samples'])


@pytest.mark.parametrize('bad', ['duplicate', 'shortage', 'same_identity'])
def test_selection_rejects_duplicates_and_cannot_fill_with_eval_or_same_molecule(bad):
    m = manifest()
    if bad == 'duplicate':
        m['samples'].append(m['samples'][0])
    elif bad == 'shortage':
        m['samples'] = m['samples'][:60] + m['samples'][80:]
    else:
        for row in m['samples']:
            row['canonical_smiles'] = 'same'
    with pytest.raises(ValueError):
        metrics.select_training_rows(m)


def synthetic(present=True, chemical='SINGLE', text='O'):
    raw = {'atom_mask': torch.tensor([[True, True, True]]), 'atom_indices': torch.tensor([[0, 1, 2]]),
           'atom_geometry': torch.tensor([[[.2, .3], [.7, .3], [.9, .9]]])}
    atoms = {**raw, 'atom_mask': torch.tensor([[True, True, False]])}
    batch = {'purpose': 'diagnostic', 'training_authorized': False,
             'atom_mask': torch.tensor([[True, True]]), 'atom_anchors': raw['atom_geometry'][:, :2].clone(),
             'bond_mask': torch.tensor([[True]]), 'bond_atom_refs': torch.tensor([[[0, 1]]]),
             'bond_chemical': torch.tensor([[BOND_TYPE_TO_ID['SINGLE']]]), 'atom_ids': [[0, 1]]}
    gt_atoms = [dict(atom_id=i, element=e, formal_charge=0, isotope=0, radical_electrons=0,
                     chiral_tag='CHI_UNSPECIFIED') for i, e in enumerate(['C', 'O'])]
    sample = {'purpose': 'diagnostic', 'meta': {'split': 'train', 'external_overlap_status': 'checked'},
              'graph': {'atoms': gt_atoms, 'bonds': [{'stereo': 'STEREONONE'}]},
              'layers': {'atoms': [dict(atom_id=0, visible_text=False, text='', bbox=None),
                                  dict(atom_id=1, visible_text=True, text='O', bbox=[529.92, 215.04, 545.28, 245.76])]}}
    trace = {'roles': [dict(candidate=0, text=False), dict(candidate=1, text=True)],
             'ocr': [dict(candidate=1, text=text, center=[.7, .3], half_extent=[.01, .02])],
             'compiled_candidate_indices': [0, 1],
             'scene_before_fallback': {'atom_objects': [{'chemical': {k: a[k] for k in ('element', 'formal_charge', 'isotope', 'radical_electrons')}} for a in gt_atoms]}}
    structure = {'atoms': atoms, 'pairs': torch.tensor([[0, 1]]),
                 'output': {'presence_logits': torch.tensor([10. if present else -10.])},
                 'chemistry': torch.tensor([BOND_TYPE_TO_ID[chemical]])}
    config = SimpleNamespace(image_size=768, atom_match_radius_px=6)
    return sample, batch, raw, structure, trace, config, .85


def record(args, exact=False):
    result = {'key': 'full:m0', 'prediction': dict(exact=exact, valid=True, fallback=False),
              'metrics': metrics.measure(*args)}
    result['flags'] = metrics.classify(result)
    return result


def test_perfect_geometry_pair_and_literal_path_do_not_imply_training_headroom():
    args = synthetic()
    old = deepcopy(args)
    r = record(args, exact=True)
    assert r['metrics']['candidate_bijection'] and r['metrics']['pair']['exact_on_fixed_candidates']
    assert r['metrics']['text']['literal_path_exact']
    assert r['metrics']['text']['rows'][1]['bbox_iou'] == pytest.approx(1.)
    assert r['flags']['baseline_correct_control'] and not r['flags']['pair_isolation_candidate']
    assert torch.equal(args[2]['atom_mask'], old[2]['atom_mask'])
    assert args[4] == old[4]


@pytest.mark.parametrize('kind', ['missing', 'wrong_chemical'])
def test_isolated_pair_errors_are_candidates_not_automatic_training_authorization(kind):
    args = synthetic(present=kind != 'missing', chemical='DOUBLE' if kind == 'wrong_chemical' else 'SINGLE')
    r = record(args)
    assert r['flags']['pair_isolation_candidate']
    assert r['metrics']['pair']['errors'][0]['kind'] == kind
    report = metrics.summarize([r])
    assert report['nonexclusive_flags']['pair_isolation_candidate'] == 1
    assert report['training_authorized'] is False


@pytest.mark.parametrize('blocker', ['text', 'role', 'stereo', 'atom_attribute', 'missing_scene'])
def test_text_role_stereo_and_atom_mismatches_block_clean_pair_flag(blocker):
    args = list(synthetic(present=False))
    if blocker == 'text':
        args[4]['ocr'][0]['text'] = 'N'
    elif blocker == 'role':
        args[4]['roles'][1]['text'] = False
        args[4]['ocr'] = []
    elif blocker == 'stereo':
        args[0]['graph']['atoms'][1]['chiral_tag'] = 'CHI_TETRAHEDRAL_CW'
    elif blocker == 'atom_attribute':
        args[4]['scene_before_fallback']['atom_objects'][1]['chemical']['element'] = 'N'
    else:
        args[4].pop('scene_before_fallback')
    assert not record(args)['flags']['pair_isolation_candidate']


def test_unreachable_bond_is_in_denominator_and_never_called_pair_only_fixable():
    args = list(synthetic())
    args[3]['atoms']['atom_mask'] = torch.tensor([[True, False, False]])
    args[3]['pairs'] = torch.empty((0, 2), dtype=torch.long)
    args[3]['output']['presence_logits'] = torch.empty(0)
    args[3]['chemistry'] = torch.empty(0, dtype=torch.long)
    args[4] = {'roles': [dict(candidate=0, text=False)], 'ocr': [],
               'compiled_candidate_indices': [0],
               'scene_before_fallback': {'atom_objects': args[4]['scene_before_fallback']['atom_objects'][:1]}}
    r = record(args)
    assert r['metrics']['pair']['unreachable_bonds'] == r['metrics']['pair']['missing_total_edges'] == 1
    assert r['flags']['selection_gap'] and not r['flags']['pair_isolation_candidate']
    assert r['metrics']['text']['missing_visible_candidates'] == 1


def test_text_difference_alone_never_justifies_bbox_training():
    r = record(synthetic(text='N'))
    assert r['flags']['text_interface_candidate']
    assert r['flags']['active_text_difference']
    assert not r['flags']['bbox_training_justified']
    # A geometrically perfect crop can still decode incorrectly.
    assert r['metrics']['text']['rows'][1]['bbox_iou'] == pytest.approx(1.)


@pytest.mark.parametrize('bad', ['purpose', 'split', 'overlap', 'pairs', 'roles'])
def test_metrics_fail_closed_on_wrong_scope_or_prediction_trace(bad):
    args = list(synthetic())
    if bad == 'purpose':
        args[0]['purpose'] = 'training'
    elif bad == 'split':
        args[0]['meta']['split'] = 'validation'
    elif bad == 'overlap':
        args[0]['meta']['external_overlap_status'] = 'unchecked'
    elif bad == 'pairs':
        args[3]['pairs'] = torch.tensor([[1, 0]])
    else:
        args[4]['roles'].pop()
    with pytest.raises(ValueError):
        metrics.measure(*args)


def test_invalid_output_stays_in_summary_and_does_not_become_correct_control():
    r = record(synthetic(present=False))
    r['prediction']['valid'] = False
    s = metrics.summarize([r])
    assert s['images'] == 1 and s['baseline_exact'] == 0 and s['baseline_valid'] == 0
    assert s['nonexclusive_flags']['baseline_correct_control'] == 0


def test_headroom_launcher_has_fixed_cpu_budget_and_no_training_knob(monkeypatch):
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return {'status': 'completed'}
    monkeypatch.setattr(headroom_audit, 'run_budgeted', run)
    assert headroom_audit.main([]) == 0
    command, kwargs = calls[0]
    assert command[-2:] == ['chemvector.training.curriculum.headroom_audit', '--worker']
    assert kwargs['budget'].wall_seconds == 300 and kwargs['budget'].gpu_devices == ()
    assert 'training' not in kwargs
    with pytest.raises(SystemExit):
        headroom_audit.main(['--max-steps', '1'])
