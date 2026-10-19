import numpy as np
import pytest

from tools.selector_candidate_parity import compare


def case():
    old = dict(image_size=768, match_radius_px=6, target_anchors=[[.2, .2]], candidates=[
        dict(slot=0, grid_id=100, geometry=[.2, .2], raw_score=.9),
        dict(slot=1, grid_id=300, geometry=[.8, .8], raw_score=.02)])
    logits = {'0': .1, '1': -3.}
    new = dict(slots=np.array([0, 1]), grid_ids=np.array([100, 300]),
        geometry=np.array([[.2, .2], [.8, .8]]), raw_scores=np.array([.9, .02]), logits=np.array([.1, -3.]))
    return old, logits, new


def test_identical_candidate_identity_can_be_reused_without_gradient_claim():
    r = compare(*case())
    assert r['same_complete_candidate_grid_set'] and r['may_reuse_historical_slot_labels']
    assert not r['parameter_direction_established'] and not r['training_authorized']


def test_permuted_slots_have_equal_selected_grids_but_cannot_reuse_slot_labels():
    old, logits, new = case()
    for name in ('grid_ids', 'geometry', 'raw_scores', 'logits'): new[name] = new[name][::-1].copy()
    r = compare(old, logits, new)
    assert r['same_complete_candidate_grid_set'] and r['selected_grid_ids_equal']
    assert r['eligible_grid_target_relations_equal'] and len(r['reordered_common_candidates']) == 2
    assert not r['may_reuse_historical_slot_labels']


def test_unselected_cutoff_replacement_still_changes_global_candidate_context():
    old, logits, new = case(); new['grid_ids'][1] = 301
    r = compare(old, logits, new)
    assert r['selected_grid_ids_equal'] and r['all_changed_candidates_outside_all_target_radii']
    assert r['all_changed_candidates_at_own_cutoff']
    assert not r['same_complete_candidate_grid_set'] and not r['may_reuse_historical_slot_labels']


def test_threshold_flip_is_detected_by_grid():
    old, logits, new = case(); new['logits'][0] = -.3
    r = compare(old, logits, new)
    assert not r['selected_grid_ids_equal'] and r['max_common_logit_change'] == pytest.approx(.4)


@pytest.mark.parametrize('bad', ['duplicate', 'nan', 'missing_score'])
def test_malformed_snapshot_rejected(bad):
    old, logits, new = case()
    if bad == 'duplicate': new['grid_ids'][1] = new['grid_ids'][0]
    if bad == 'nan': new['logits'][1] = float('nan')
    if bad == 'missing_score': logits.pop('1')
    with pytest.raises(ValueError): compare(old, logits, new)
