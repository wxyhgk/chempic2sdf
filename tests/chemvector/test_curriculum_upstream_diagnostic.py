"""Synthetic checks before the one real, fixed CPU diagnostic."""

from copy import deepcopy
from types import SimpleNamespace

from PIL import Image
import pytest
import torch

from chemvector.training.curriculum import evaluation, upstream_diagnostic as audit
from chemvector.training.curriculum.interface_diagnostic import tensor_tree_digest


def cache():
    raw = {'atom_mask': torch.tensor([[True, True, False]]),
           'atom_indices': torch.tensor([[0, 1, 2]]),
           'atom_geometry': torch.tensor([[[.2, .3], [.7, .3], [.9, .9]]]),
           'atom_features': torch.zeros(1, 3, 2), 'atom_scores': torch.ones(1, 3)}
    bonds = {k: torch.zeros(1, 1, 2) for k in ('bond_features', 'bond_endpoints', 'bond_types')}
    bonds.update(bond_scores=torch.ones(1, 1), bond_mask=torch.ones(1, 1, dtype=torch.bool))
    return {'features': torch.zeros(1, 2, 3, 3), 'raw': raw, 'bonds': bonds}


def selector(*args):
    assert not torch.is_grad_enabled()
    return {'atom_presence_logits': torch.tensor([[0., -1., 9.]])}


def test_selector_uses_fixed_prediction_inputs_and_never_resurrects_masked_candidates():
    c = cache()
    before = tensor_tree_digest(c)
    selected, _ = audit.choose_from_cache(c, selector, -.25)
    assert selected['atom_mask'].tolist() == [[True, False, False]]
    assert selected['atom_geometry'] is c['raw']['atom_geometry']
    assert tensor_tree_digest(c) == before


def test_selector_rejects_cache_mutation_and_nonfinite_outputs():
    def mutating(*args):
        args[1].add_(1)
        return selector(*args)
    with pytest.raises(ValueError, match='changed cached'):
        audit.choose_from_cache(cache(), mutating, -.25)
    with pytest.raises(ValueError, match='Non-finite'):
        audit.choose_from_cache(cache(), lambda *a: {'atom_presence_logits': torch.full((1, 3), float('nan'))}, -.25)


def inputs():
    rows, keys, cases = [], [], []
    for i in range(12):
        rows.append(dict(source='train', sample_key=str(i), molecule_id=str(i), split='train',
                         domain='full_molecule' if i < 8 else 'curriculum'))
        keys.append('train:' + str(i))
        rows.append(dict(source='dev', sample_key=str(i), molecule_id=str(i), split='validation', domain='full_molecule'))
        cases.append(dict(key='dev:' + str(i), group='old_stratum'))
    return dict(training_keys=keys, cases=cases), dict(samples=rows)


def test_selection_keeps_original_order_and_uses_no_new_predictions():
    prior, manifest = inputs()
    plan = audit.inherited_selection(prior, manifest)
    assert plan['cases'] == prior['cases'] and plan['training_keys'] == prior['training_keys']
    assert len(plan['arms']) == 6 and plan['planned_development_predictions'] == 72
    assert plan['planned_generator_forwards'] == 48
    prior['cases'][0]['group'] = 'modified'
    assert plan['cases'][0]['group'] == 'old_stratum'


@pytest.mark.parametrize('bad', ['duplicate', 'missing', 'train_split', 'dev_split', 'domain', 'molecules'])
def test_selection_rejects_changed_inputs(bad):
    prior, manifest = inputs()
    if bad == 'duplicate':
        manifest['samples'].append(manifest['samples'][0])
    elif bad == 'missing':
        prior['training_keys'].pop()
    elif bad == 'train_split':
        manifest['samples'][0]['split'] = 'validation'
    elif bad == 'dev_split':
        manifest['samples'][1]['split'] = 'train'
    elif bad == 'domain':
        manifest['samples'][0]['domain'] = 'curriculum'
    else:
        manifest['samples'][3]['molecule_id'] = '0'
    with pytest.raises(ValueError):
        audit.inherited_selection(prior, manifest)


def test_coverage_distinguishes_raw_presence_from_selection_and_keeps_missing_truth():
    c = cache()
    selected, _ = audit.choose_from_cache(c, selector, -.25)
    batch = {'purpose': 'diagnostic', 'training_authorized': False,
             'atom_mask': torch.tensor([[True, True, True]]),
             'atom_anchors': torch.tensor([[[.2, .3], [.7, .3], [.5, .8]]]),
             'bond_mask': torch.tensor([[True, True]]),
             'bond_atom_refs': torch.tensor([[[0, 1], [1, 2]]]),
             'bond_chemical': torch.tensor([[0, 0]])}
    cfg = SimpleNamespace(image_size=768, atom_match_radius_px=6)
    result = audit.labeled_coverage(c['raw'], selected, batch, cfg)
    assert (result['target_atoms'], result['raw_matched'], result['selected_matched']) == (3, 2, 1)
    assert result['raw_no_nearby'] == 1 and result['selected_no_nearby'] == 2
    assert result['reachable_positive_pairs'] == 0
    assert result['target_details'][1]['near_raw_ids'] == [1]
    assert result['target_details'][1]['near_selected_ids'] == []
    batch['purpose'] = 'training'
    with pytest.raises(ValueError, match='diagnostic-only'):
        audit.labeled_coverage(c['raw'], selected, batch, cfg)


def test_front_cache_only_changes_selector_on_each_fixed_generator(monkeypatch):
    from chemvector.training.curriculum import continuation
    calls = []
    all_selected = lambda *a: {'atom_presence_logits': torch.ones(1, 3)}
    models = {'initial': SimpleNamespace(set_head=all_selected, offset=0.),
              'last': SimpleNamespace(set_head=selector, offset=.01)}
    def forward(model, pixels, config, threshold):
        calls.append(model.offset)
        c = cache()
        c['raw']['atom_geometry'] += model.offset
        selected, output = audit.choose_from_cache(c, model.set_head, threshold)
        return c['features'], None, None, c['raw'], c['bonds'], output, selected
    monkeypatch.setattr(continuation, 'structure_forward', forward)
    caches, selections, traces = audit.cache_fronts(models, torch.zeros(1), None, -.25)
    assert calls == [0., .01]
    assert len(traces) == 4
    for g in audit.NAMES:
        assert selections[(g, 'initial')]['atom_geometry'] is selections[(g, 'last')]['atom_geometry']
        assert selections[(g, 'initial')]['atom_mask'].sum() == 2
        assert selections[(g, 'last')]['atom_mask'].sum() == 1


def synthetic_predictor(monkeypatch):
    # Real pair tensor construction and compiler, only the neural predictions are synthetic.
    atoms = {'atom_mask': torch.tensor([[True, True]]), 'atom_geometry': torch.tensor([[[.3, .5], [.7, .5]]]),
             'atom_features': torch.zeros(1, 2, 4), 'atom_scores': torch.ones(1, 2)}
    features = torch.zeros(1)
    monkeypatch.setattr(evaluation, 'structure_forward', lambda *a: (features, None, None, None, None, None, atoms))
    predictor = evaluation.FrozenOCRPredictor.__new__(evaluation.FrozenOCRPredictor)
    predictor.provenance = dict(atom_threshold=-.25, pair_threshold=.85, role_threshold=.85)
    predictor.ocr = lambda crops: torch.zeros(len(crops), 3, 2)
    monkeypatch.setattr(evaluation, 'decode_ctc_with_score', lambda row: ('N', .99))
    model = SimpleNamespace(pair=lambda *a: {'presence_logits': torch.tensor([10.]),
        'chemical_logits': torch.tensor([[10., -10., -10., -10.]])},
        role=lambda *a: torch.tensor([-10., 10.]),
        bbox=lambda pixels, anchors, indices, rows: {'centers': anchors[rows, indices],
            'half_extents': torch.ones(len(indices), 2) * .02})
    return predictor, model, features, atoms


def test_cached_atoms_path_matches_original_structure_path(monkeypatch):
    predictor, model, features, atoms = synthetic_predictor(monkeypatch)
    original = predictor.predict_structure(model, None, torch.zeros(1))
    cached = predictor.from_predicted_atoms(model.pair, features, atoms)
    assert tensor_tree_digest(original) == tensor_tree_digest(cached)


def test_failure_trace_preserves_atoms_and_edges_without_changing_prediction(monkeypatch):
    predictor, model, _, _ = synthetic_predictor(monkeypatch)
    structure = predictor.predict_structure(model, None, torch.zeros(1))
    def first_failure(scene):
        raise ValueError('first sanitize error')
    def fallback_failure(scene):
        raise ValueError('fallback error')
    monkeypatch.setattr(evaluation, 'mol_from_chemscene_v0_2', first_failure)
    monkeypatch.setattr(evaluation, 'compile_carbon_placeholder', fallback_failure)
    image, pixels, trace = Image.new('RGB', (768, 768)), torch.zeros(1, 3, 768, 768), {}
    default = predictor.from_structure(structure, model.role, model.bbox, pixels, image)
    traced = predictor.from_structure(structure, model.role, model.bbox, pixels, image, trace=trace)
    assert traced == default and traced['molblock'] is None
    assert trace['sanitize_error'] == 'first sanitize error' and trace['compile_error'] == 'fallback error'
    assert len(trace['scene_before_fallback']['atom_objects']) == 2
    assert len(trace['scene_before_fallback']['bond_objects']) == 1


def test_summary_uses_current_not_historical_groups_and_retains_invalids():
    def case(old, initial, last):
        arms = {a: dict(exact=False, valid=False, fallback=False) for a in audit.ARMS}
        for name, correct in [('initial', initial), ('last', last)]:
            arms[name] = dict(exact=correct, valid=correct, fallback=False)
        return dict(group=old, arms=arms)
    cases = [case('rescued', True, False), case('invalid', True, True), case('attributes', False, True)]
    result = audit.summarize_cases(cases)
    assert result['all']['images'] == 3
    assert result['currently_broken']['images'] == 1
    assert result['currently_rescued']['images'] == 1
    assert result['all']['arms']['initial']['exact'] == 2
    assert result['all']['arms']['last']['valid'] == 2


def test_saved_replay_mismatch_is_a_hard_failure():
    old = dict(predicted_smiles='CN', expected_smiles='CN', valid=True, exact=True, fallback=False)
    audit.check_saved_replay(old, old)
    with pytest.raises(ValueError, match='replay changed'):
        audit.check_saved_replay(dict(old, fallback=True), old)


def test_cli_has_fixed_cpu_budget_and_no_training_or_retry(monkeypatch):
    seen = []
    def run(command, **kwargs):
        seen.append((command, kwargs))
        return {'status': 'completed'}
    monkeypatch.setattr(audit, 'run_budgeted', run)
    assert audit.main([]) == 0
    command, args = seen[0]
    assert command[-2:] == ['chemvector.training.curriculum.upstream_diagnostic', '--worker']
    assert args['budget'].wall_seconds == 300 and not args['budget'].gpu_devices
    assert 'training' not in args
    with pytest.raises(SystemExit):
        audit.main(['--max-steps', '100'])
