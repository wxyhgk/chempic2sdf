"""Synthetic-only checks for the saved-coordinate independent reference."""

from copy import deepcopy

import numpy as np
import pytest

from tools import audit_curriculum_v02_selector_matching as existing
from tools import audit_selector_radius_reference as runner
from tools import selector_node_groups as groups
from tools import selector_radius_reference as reference


def fixture():
    points = {1: (.5613572001457214, .48636168241500854),
              2: (.472441166639328, .5021392107009888),
              3: (.4551868140697479, .49676835536956787),
              5: (.4400653541088104, .4872252643108368)}
    geometry = dict(image_size=768, match_radius_px=6,
        target_anchors=[[.5620723366737366, .4995112419128418], [.4682306945323944, .4995112419128418]],
        candidates=[dict(slot=s, geometry=list(p), raw_target=1 if s == 2 else -1) for s, p in points.items()])
    layers = dict(atoms=[dict(atom_id=0), dict(atom_id=1)],
                  bonds=[dict(bond_id=0, begin_atom=0, end_atom=1)])
    return geometry, layers


def test_reuses_existing_tested_radius_assignment():
    assert reference.radius_assignment is existing.radius_assignment


def test_known_triple_bond_counterexample_no_false_model_improvement():
    geo, layers = fixture()
    untouched = deepcopy((geo, layers))
    result = reference.compare_removal(geo, [1, 2, 3, 5], [1, 3], layers)
    assert result['original']['before']['matched_target_ids'] == [1]
    assert result['original']['after']['matched_target_ids'] == []
    assert result['radius_reference']['before']['matched_target_ids'] == [1]
    assert result['radius_reference']['after']['matched_target_ids'] == [1]
    assert result['radius_reference']['after']['mapping'] == {2: 1, 5: -1}
    for policy in ('original', 'radius_reference'):
        for when in ('before', 'after'):
            assert result[policy][when]['reachable_bond_ids'] == []
    assert result['original']['removal_difference']['lost_relations'] == [(2, 1)]
    assert result['radius_reference']['removal_difference']['lost_relations'] == []
    assert result['frozen_eligible_graph_identical'] and result['reference_relations_preserved']
    assert result['remaining_slots'] == [2, 5]
    assert not result['training_authorized'] and result['uses_ground_truth']
    assert (geo, layers) == untouched


@pytest.mark.parametrize('slots', [[], [1], [2], [2, 5], [1, 2, 3, 5]])
def test_original_mapping_parity(slots):
    geo, layers = fixture()
    result = reference.compare(geo, slots, layers)
    old = groups.coverage(geo, slots, layers)
    assert all(result['original'][k] == v for k, v in old.items())


@pytest.mark.parametrize('slots,removed', [([], []), ([1], [1]), ([1, 2, 3, 5], [])])
def test_empty_and_noop_removal(slots, removed):
    geo, layers = fixture()
    check = reference.compare_removal(geo, slots, removed, layers)
    assert check['frozen_eligible_graph_identical'] and check['reference_relations_preserved']
    assert not check['radius_reference']['removal_difference']['lost_target_ids']


@pytest.mark.parametrize('mutation', ['duplicate', 'negative', 'bool', 'unknown', 'radius', 'size', 'nan', 'bad_xy'])
def test_geometry_fails_closed(mutation):
    geo, _ = fixture()
    slots = [1, 2, 3, 5]
    if mutation == 'duplicate': slots.append(1)
    elif mutation == 'negative': slots.append(-1)
    elif mutation == 'bool': slots[0] = True
    elif mutation == 'unknown': slots.append(7)
    elif mutation == 'radius': geo['match_radius_px'] = 7
    elif mutation == 'size': geo['image_size'] = 767
    elif mutation == 'nan': geo['target_anchors'][0][0] = float('nan')
    else: geo['candidates'][0]['geometry'] = [1.1, .1]
    with pytest.raises(ValueError):
        reference.distance_matrix(geo, slots)


@pytest.mark.parametrize('removed', [[2], [9], [1, 1]])
def test_rejects_eligible_unknown_or_duplicate_removals(removed):
    geo, layers = fixture()
    with pytest.raises(ValueError):
        reference.compare_removal(geo, [1, 2, 3, 5], removed, layers)


def test_numerical_recompute_is_separate_from_frozen_policy(monkeypatch):
    geo, layers = fixture()
    def costs(geometry, slots):
        return np.array([[6., 20.], [20., 20.]], dtype=np.float32) if len(slots) == 2 else np.array([[6.0001, 20.]], dtype=np.float32)
    monkeypatch.setattr(reference, 'distance_matrix', costs)
    check = reference.compare_removal(geo, [1, 2], [2], layers)
    assert check['radius_reference']['after']['matched_target_ids'] == [0]
    assert check['numerical_check']['recomputed_reference']['matched_target_ids'] == []
    assert check['numerical_check']['difference_from_frozen']['lost_target_ids'] == [0]
    assert len(check['numerical_check']['eligibility_changes']) == 1
    assert check['numerical_check']['max_abs_distance_change_px'] > 0
    assert check['frozen_eligible_graph_identical']


def test_equal_cardinality_can_change_target_and_bond_coverage():
    layers = dict(atoms=[dict(atom_id=i) for i in range(3)],
                  bonds=[dict(bond_id=0, begin_atom=0, end_atom=2)])
    before = reference.coverage([[1, 2, 99], [99, 99, 1]], [0, 1], layers, policy='radius_reference')
    after = reference.coverage([[2, 1, 99], [99, 99, 1]], [0, 1], layers, policy='radius_reference')
    diff = reference.difference(before, after)
    assert diff['matched_count_delta'] == 0
    assert diff['lost_target_ids'] == [0] and diff['gained_target_ids'] == [1]
    assert diff['lost_bond_ids'] == [0] and diff['lost_relations'] == [(0, 0)]


def test_ties_do_not_assume_which_optimal_slot_or_target_wins():
    _, layers = fixture()
    result = reference.coverage([[1, 1], [1, 1], [20, 20]], [0, 1, 2], layers, policy='radius_reference')
    smaller = reference.coverage([[1, 1], [1, 1]], [0, 1], layers, policy='radius_reference')
    assert result['matched_target_ids'] == smaller['matched_target_ids'] == [0, 1]
    assert {p['slot'] for p in result['assignment_before_gate']} == {0, 1}
    assert sum(p['distance_px'] for p in result['assignment_before_gate']) == 2


@pytest.mark.parametrize('mutation', ['shape', 'nan', 'negative', 'unsorted', 'atom_ids', 'bond_ids', 'bond_end', 'policy'])
def test_coverage_contract(mutation):
    _, layers = fixture()
    cost, slots, policy = [[1, 20], [20, 1]], [0, 1], 'radius_reference'
    if mutation == 'shape': cost.pop()
    elif mutation == 'nan': cost[0][0] = float('nan')
    elif mutation == 'negative': cost[0][0] = -1
    elif mutation == 'unsorted': slots.reverse()
    elif mutation == 'atom_ids': layers['atoms'][0]['atom_id'] = 8
    elif mutation == 'bond_ids': layers['bonds'].append(deepcopy(layers['bonds'][0]))
    elif mutation == 'bond_end': layers['bonds'][0]['end_atom'] = 9
    else: policy = 'deployed'
    with pytest.raises(ValueError):
        reference.coverage(cost, slots, layers, policy=policy)


def scope_fixture():
    rows = [dict(sample_key='synthetic_' + str(i), fit_domain='witness' if i < 26 else 'old_full_molecule',
                 split='train', purpose='training') for i in range(58)]
    manifest = dict(training_samples=rows, validation_only_samples=[dict(sample_key='never_read')])
    report = dict(case_sha256={r['sample_key'] + '.json': 'unused' for r in rows},
                  images=58, saved_phase_checks=232, training_authorized=False)
    class Inputs:
        def json(self, path, expected):
            if path == runner.prior.MANIFEST:
                assert expected == runner.MANIFEST_SHA
                return manifest
            assert path == runner.prior.OUTPUT / 'report.json' and expected == runner.SOURCE_SHA
            return report
    return manifest, report, Inputs()


def test_preflight_only_manifest_and_bound_report():
    _, _, inputs = scope_fixture()
    assert len(runner.preflight(inputs)[2]) == 58


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'validation_overlap', 'domain', 'admitted', 'phase_count', 'case_key'])
def test_scope_rejects_changed_or_validation_rows(mutation):
    manifest, report, inputs = scope_fixture()
    if mutation == 'missing': manifest['training_samples'].pop()
    elif mutation == 'duplicate': manifest['training_samples'][1] = manifest['training_samples'][0]
    elif mutation == 'validation_overlap': manifest['validation_only_samples'][0]['sample_key'] = 'synthetic_0'
    elif mutation == 'domain': manifest['training_samples'][0]['fit_domain'] = 'old_full_molecule'
    elif mutation == 'admitted': report['training_authorized'] = True
    elif mutation == 'phase_count': report['saved_phase_checks'] = 1
    else: report['case_sha256']['unexpected.json'] = 'unused'
    with pytest.raises(ValueError):
        runner.preflight(inputs)


@pytest.mark.parametrize('mutation', [None, 'nan', 'alias', 'missing', 'selected_count', 'positive'])
def test_fixed_selector_cutoff_and_exact_saved_coverage(mutation):
    geo, _ = fixture()
    rec = dict(metrics=dict(logits={'1': -.25, '2': -.25001, '3': -1., '5': 1.},
        selected_atoms=2, positive_slots=[2], retained_positive_slots=[], selected_negative_slots=[1, 5]))
    if mutation == 'nan': rec['metrics']['logits']['1'] = float('nan')
    elif mutation == 'alias': rec['metrics']['logits']['01'] = rec['metrics']['logits'].pop('1')
    elif mutation == 'missing': del rec['metrics']['logits']['1']
    elif mutation == 'selected_count': rec['metrics']['selected_atoms'] = 3
    elif mutation == 'positive': rec['metrics']['positive_slots'] = []
    if mutation:
        with pytest.raises(ValueError):
            runner.check_selection(geo, rec)
    else:
        assert runner.check_selection(geo, rec) == [1, 5]
