"""Synthetic protocol bridges and detached-logit gradients; no real models."""

from copy import deepcopy

import pytest
import torch

from chemvector.vertex import build_vertex_heatmap
from tests.chemvector.test_curriculum_data import corpus, sample
from tools import selector_anchor_contract as contract


@pytest.mark.parametrize('smiles', ['CCO', 'CC=O', 'c1ccccc1', 'C[O-]', 'F[C@H](Cl)Br'])
def test_legacy_and_current_anchors_equal_after_explicit_id_alignment(corpus, smiles):
    s = sample(corpus, smiles)
    untouched = deepcopy((s['graph'], s['layers'], s['text_records'], s['meta']))
    result = contract.compare_target_paths(*untouched)
    assert result['anchors_equal'] and result['heatmaps_equal']
    assert result['roles_equal'] and result['text_mask_equal']
    assert sorted(result['legacy_spatial_slot_to_source_id']) == list(range(result['atom_count']))
    assert (s['graph'], s['layers'], s['text_records'], s['meta']) == untouched
    assert not result['training_authorized']


def test_text_center_is_a_distinct_changed_contract_not_anchor_swap(corpus):
    s = sample(corpus, 'CCO')
    result = contract.compare_target_paths(s['graph'], s['layers'], s['text_records'], s['meta'])
    assert result['source_ids_with_changed_text_center']
    for i in result['source_ids_with_changed_text_center']:
        assert torch.equal(result['legacy_text_centers'][i], result['current_anchors'][0, i])
        assert not torch.equal(result['current_text_centers'][i], result['current_anchors'][0, i])


def test_moving_ink_boxes_changes_text_targets_not_heatmap(corpus):
    s = sample(corpus, 'CCO')
    before = contract.compare_target_paths(s['graph'], s['layers'], s['text_records'], s['meta'])
    layer, records = deepcopy(s['layers']), deepcopy(s['text_records'])
    for atom in layer['atoms']:
        if atom['bbox'] is not None:
            atom['bbox'][0] += 3
            atom['bbox'][2] += 3
    for glyph in records['glyphs']:
        glyph['bbox'][0] += 3
        glyph['bbox'][2] += 3
    after = contract.compare_target_paths(s['graph'], layer, records, s['meta'])
    assert torch.equal(before['current_anchors'], after['current_anchors'])
    assert torch.equal(before['current_heatmap'], after['current_heatmap'])
    assert not torch.equal(before['current_text_centers'], after['current_text_centers'])


@pytest.mark.parametrize('change', ['canvas', 'identity', 'atom_ids', 'collision'])
def test_bridge_contract_fails_closed(corpus, change):
    s = sample(corpus, 'CCO')
    layers = deepcopy(s['layers'])
    if change == 'canvas': layers['width'] = 767
    elif change == 'identity': layers['molecule_id'] = 'different'
    elif change == 'atom_ids': layers['atoms'][0]['atom_id'] = 99
    else: layers['atoms'][1]['center'] = layers['atoms'][0]['center'][:]
    with pytest.raises(ValueError):
        contract.legacy_bridge(s['graph'], layers)


def heat_fixture(*, neighbor_positive=False):
    anchors = [[.25, .25], [49.5/192, 48.5/192] if neighbor_positive else [.75, .75]]
    logits = torch.full((1, 1, 192, 192), -4.)
    logits[0, 0, 48, 49] = 1.
    target = build_vertex_heatmap(torch.tensor([anchors]), torch.ones(1, 2, dtype=torch.bool), 192, 192, 1)
    geometry = dict(target_anchors=anchors,
                    candidates=[dict(slot=1, geometry=[49.5/192, 48.5/192])])
    layers = dict(atoms=[dict(atom_id=i, center=[v*767 for v in xy], text='O',
                              bbox=[v*767-2 for v in xy]+[v*767+2 for v in xy]) for i, xy in enumerate(anchors)])
    return dict(heatmap_logits=logits), geometry, layers, target


def test_existing_heatmap_loss_pushes_gt_up_and_false_neighbor_down():
    snapshot, geo, layers, heat = heat_fixture()
    before = snapshot['heatmap_logits'].clone()
    result = contract.saved_heatmap_probe(snapshot, geo, [0], layers, heat)
    row = result['targets'][0]
    assert row['gt_heatmap_target'] == 1 and row['gt_dloss_dlogit'] < 0
    assert len(row['local_max_neighbors']) == 1
    neighbor = row['local_max_neighbors'][0]
    assert neighbor['dloss_dlogit'] > 0
    assert neighbor['independent_descent_gt_minus_neighbor_margin_direction'] > 0
    assert neighbor['positive_source_atom_ids'] == []
    assert torch.equal(before, snapshot['heatmap_logits']) and snapshot['heatmap_logits'].grad is None
    assert result['parameter_gradients_computed'] is False


def test_other_true_atom_is_not_mislabeled_as_suppressible_negative():
    snapshot, geo, layers, heat = heat_fixture(neighbor_positive=True)
    row = contract.saved_heatmap_probe(snapshot, geo, [0], layers, heat)['targets'][0]
    neighbor = row['local_max_neighbors'][0]
    assert neighbor['positive_source_atom_ids'] == [1]
    assert neighbor['target_heatmap'] == 1 and neighbor['dloss_dlogit'] < 0


def test_cannot_build_supervision_from_failure_targets_only():
    snapshot, geo, layers, _ = heat_fixture()
    incomplete = build_vertex_heatmap(torch.tensor([geo['target_anchors'][:1]]),
                                      torch.ones(1, 1, dtype=torch.bool), 192, 192, 1)
    with pytest.raises(ValueError, match='all original'):
        contract.saved_heatmap_probe(snapshot, geo, [0], layers, incomplete)


@pytest.mark.parametrize('change', ['nan', 'grad', 'shape', 'dtype', 'target_nan', 'target_grad', 'ids'])
def test_detached_snapshot_contract(change):
    snapshot, geo, layers, heat = heat_fixture()
    focus = [0]
    if change == 'nan': snapshot['heatmap_logits'][0, 0, 0, 0] = float('nan')
    elif change == 'grad': snapshot['heatmap_logits'].requires_grad_()
    elif change == 'shape': snapshot['heatmap_logits'] = snapshot['heatmap_logits'][:, :, :191]
    elif change == 'dtype': snapshot['heatmap_logits'] = snapshot['heatmap_logits'].double()
    elif change == 'target_nan': heat[0, 0, 0, 0] = float('nan')
    elif change == 'target_grad': heat.requires_grad_()
    else: focus = [0, 0]
    with pytest.raises(ValueError):
        contract.saved_heatmap_probe(snapshot, geo, focus, layers, heat)
