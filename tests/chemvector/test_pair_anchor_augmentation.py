"""Synthetic CPU tensors only; no corpus, real model, optimizer or GPU."""

import pytest
import torch

from chemvector.direct_atom_pair_bond import direct_atom_pair_loss
from chemvector.training.curriculum import pair_anchor_augmentation as aug


def fixture():
    anchors = torch.tensor([[100., 100.], [150., 100.], [200., 100.], [250., 100.]]) / 768
    selected = torch.ones(4, dtype=torch.bool)
    mapping = torch.tensor([0, 1, 2, -1])
    targets = anchors[:3].clone()
    pairs = torch.combinations(torch.arange(4), r=2)
    known = (mapping[pairs] >= 0).all(-1)
    chosen = torch.tensor([0, 3, 1])  # two path edges and the real 0--2 non-edge
    return [anchors, selected, mapping, targets, pairs, known, chosen]


def test_fixed_views_keep_identity_order_direction_and_inputs():
    args = fixture()
    before = [t.clone() for t in args]
    views = aug.plan_views(*args)
    assert len(views) == 9 and len(set(aug.OFFSETS)) == 9
    for view, offset in zip(views, aug.OFFSETS):
        assert view['offset_pixels'] == list(offset)
        assert torch.equal(view['rows'], args[-1])
        assert not bool((~args[-2][view['rows']]).any())
        torch.testing.assert_close(view['anchors'] - args[0], torch.tensor(offset).expand(4, 2) / 768)
        torch.testing.assert_close(view['anchors'][args[4][:, 1]] - view['anchors'][args[4][:, 0]],
                                   args[0][args[4][:, 1]] - args[0][args[4][:, 0]])
    assert all(torch.equal(a, b) for a, b in zip(args, before))


@pytest.mark.parametrize('change,reason', [
    ('border', 'outside_image'), ('radius', 'outside_owner_radius'),
    ('tie', 'not_unique_nearest'), ('nearer_neighbor', 'not_unique_nearest'),
])
def test_shift_cannot_reassign_owners_or_rewrite_base_rows(change, reason):
    args = fixture()
    if change == 'border':
        args[0][0] = torch.tensor([1., 100.]) / 768
        args[3][0] = args[0][0]
        offset = (-4., 0.)
    elif change == 'radius':
        args[0][0, 0] += 4. / 768
        offset = (4., 0.)
    else:
        # Use exact float64 midpoint to test a tie without FP32 rounding noise.
        args[0] = args[0].double(); args[3] = args[3].double()
        args[0][0] = args[3][0] = torch.tensor([.5, .5], dtype=torch.float64)
        args[3][2] = args[3][0] + torch.tensor([8. if change == 'tie' else 6., 0.]) / 768
        if change == 'tie': args[3][2, 0] = .5 + 8. / 768
        offset = (4., 0.)
    views = aug.plan_views(*args)
    shifted = views[aug.OFFSETS.index(offset)]
    assert torch.equal(views[0]['rows'], args[-1])
    assert bool(shifted['exclusion_masks'][reason][0])
    assert 0 not in shifted['rows'].tolist()
    assert args[2][0] == 0  # same original lineage, never rematched


def test_ambiguous_original_rows_kept_but_marked_not_eligible():
    args = fixture()
    args[3][2] = args[3][0]
    views = aug.plan_views(*args)
    assert torch.equal(views[0]['rows'], args[-1])
    assert not views[0]['eligible_mask'][0]
    assert all(0 not in view['rows'].tolist() for view in views[1:])


@pytest.mark.parametrize('change', ['unknown_chosen', 'wrong_known', 'duplicate_owner',
                                   'duplicate_pair', 'reversed_pair', 'nan', 'integer_xy'])
def test_bad_contract_rejected(change):
    args = fixture()
    if change == 'unknown_chosen': args[-1] = torch.tensor([2])
    elif change == 'wrong_known': args[-2].fill_(True)
    elif change == 'duplicate_owner': args[2][1] = 0
    elif change == 'duplicate_pair': args[4][1] = args[4][0]
    elif change == 'reversed_pair': args[4][0] = torch.tensor([1, 0])
    elif change == 'nan': args[0][0, 0] = float('nan')
    else: args[0] = args[0].long()
    with pytest.raises(ValueError): aug.plan_views(*args)


def test_empty_and_unselected_branches_do_not_invent_supervision():
    args = fixture()
    args[1][3] = False
    args[4] = args[4][args[5]]
    args[5] = torch.ones(3, dtype=torch.bool)
    args[6] = torch.empty(0, dtype=torch.long)
    for view in aug.plan_views(*args):
        assert len(view['rows']) == 0
        assert torch.equal(view['anchors'][3], args[0][3])
    args[4] = torch.empty(0, 2, dtype=torch.long)
    args[5] = torch.empty(0, dtype=torch.bool)
    assert all(v['eligible_mask'].numel() == 0 for v in aug.plan_views(*args))


def logits(count):
    return {'presence_logits': torch.zeros(count, requires_grad=True),
            'chemical_logits': torch.zeros(count, 4, requires_grad=True)}


def test_loss_zero_weight_exact_value_and_gradient_parity():
    base, shifted = logits(3), logits(2)
    presence, chemical, positions = torch.tensor([True, True, False]), torch.tensor([0, 1, -1]), torch.tensor([0, 2])
    original = direct_atom_pair_loss(base, presence, chemical)['loss']
    result = aug.typed_augmentation_loss(base, shifted, presence, chemical, positions, weight=0)
    assert torch.equal(original, result['loss']) and not result['augmentation_applied']
    expected = torch.autograd.grad(original, list(base.values()), retain_graph=True)
    result['loss'].backward()
    for tensor, gradient in zip(base.values(), expected): assert torch.equal(tensor.grad, gradient)
    assert all(t.grad is None for t in shifted.values())


def test_loss_preserves_base_gradient_and_teaches_each_positive_type_and_nonedge():
    base, shifted = logits(5), logits(5)
    presence = torch.tensor([True, True, True, True, False])
    chemical, positions = torch.tensor([0, 1, 2, 3, -1]), torch.arange(5)
    result = aug.typed_augmentation_loss(base, shifted, presence, chemical, positions)
    expected = torch.autograd.grad(result['original']['loss'], list(base.values()), retain_graph=True)
    result['loss'].backward()
    for tensor, gradient in zip(base.values(), expected): assert torch.equal(tensor.grad, gradient)
    assert bool((shifted['presence_logits'].grad[:4] < 0).all())
    assert shifted['presence_logits'].grad[4] > 0
    assert bool((shifted['chemical_logits'].grad[torch.arange(4), torch.arange(4)] < 0).all())
    assert not bool(shifted['chemical_logits'].grad[4].any())
    assert all(bool(torch.isfinite(t.grad).all()) for t in (*base.values(), *shifted.values()))


def test_negative_only_and_empty_augmentation():
    presence, chemical = torch.zeros(2, dtype=torch.bool), torch.full((2,), -1)
    result = aug.typed_augmentation_loss(logits(2), logits(1), presence, chemical, torch.tensor([1]))
    assert result['original']['chemical_loss'] == result['augmented']['chemical_loss'] == 0
    result['loss'].backward()
    empty = aug.typed_augmentation_loss(logits(2), None, presence, chemical, torch.empty(0, dtype=torch.long))
    assert empty['loss'] is empty['original']['loss'] and not empty['augmentation_applied']
    with pytest.raises(ValueError):
        aug.typed_augmentation_loss(logits(2), logits(0), presence, chemical, torch.empty(0, dtype=torch.long))


@pytest.mark.parametrize('bad', ['chemical', 'row', 'duplicate', 'nan', 'weight'])
def test_bad_loss_alignment_rejected(bad):
    base, shifted = logits(2), logits(2)
    presence, chemical, positions, weight = torch.tensor([True, False]), torch.tensor([0, -1]), torch.arange(2), 1
    if bad == 'chemical': chemical[0] = 4
    elif bad == 'row': positions[1] = 2
    elif bad == 'duplicate': positions[1] = 0
    elif bad == 'nan': shifted['presence_logits'] = torch.tensor([float('nan'), 0.])
    else: weight = .5
    with pytest.raises(ValueError): aug.typed_augmentation_loss(base, shifted, presence, chemical, positions, weight=weight)


def test_ribbon_review_is_informational_not_a_negative_filter():
    args = fixture()
    # The intermediate atom lies between 0 and 2 but not between neighboring pairs.
    flags = aug.other_nodes_in_ribbon(args[0], args[4], args[2], args[3])
    assert flags[1] and not flags[0] and not flags[3]
    views = aug.plan_views(*args)
    assert all(1 in v['rows'].tolist() for v in views)
    args[3][1, 1] += 13 / 768
    assert not aug.other_nodes_in_ribbon(args[0], args[4], args[2], args[3])[1]
