"""Pure synthetic dense fields; no model or real image forwards."""

import pytest
import torch
from torch.nn import functional as F

from chemvector.components.graph.joint.predicted_incidence import _local_peak_topk
from tools.selector_peak_tie_contract import selection_difference, stable_local_peak_topk


def scalar_order(z, maximum, threshold):
    probabilities = z.sigmoid()
    peaks = probabilities == F.max_pool2d(probabilities, 3, 1, 1)
    scores = probabilities.masked_fill(~peaks, -1).flatten(1)
    rows = []
    for row in scores.tolist():
        ids = sorted(range(len(row)), key=lambda i: (-row[i], i))[:maximum]
        rows.append(ids+[0]*max(0, maximum-len(row)))
    return torch.tensor(rows, dtype=torch.int64)


def test_flat_plateau_uses_ascending_identity_and_no_noise_or_mutation():
    z = torch.full((1, 1, 6, 6), .8)
    before = z.clone()
    values, ids, mask = stable_local_peak_topk(z, maximum=4, threshold=.01)
    assert ids.tolist() == [[0, 1, 2, 3]] and bool(mask.all())
    assert torch.equal(values, z.sigmoid().flatten()[:4][None]) and torch.equal(z, before)


def test_multichannel_flat_identity_does_not_merge_bond_slots():
    z = torch.zeros(1, 2, 2, 2)
    _, ids, _ = stable_local_peak_topk(z, maximum=6, threshold=.01)
    assert ids.tolist() == [[0, 1, 2, 3, 4, 5]]
    assert ids[0, 0] % 4 == ids[0, 4] % 4 and ids[0, 0] != ids[0, 4]


def test_near_ties_are_not_rounded_merged_or_perturbed():
    z = torch.tensor([0., 1e-9], dtype=torch.float64).reshape(1, 2, 1, 1)
    values, ids, _ = stable_local_peak_topk(z, maximum=2, threshold=.01)
    assert ids.tolist() == [[1, 0]] and values[0, 0] > values[0, 1]
    assert torch.equal(values[0], z.sigmoid().flatten().flip(0))


def test_exact_ties_are_in_probability_space_not_original_logit_space():
    z = torch.tensor([20., 21.]).reshape(1, 2, 1, 1)
    assert z.flatten()[1] > z.flatten()[0] and z.sigmoid().flatten()[0] == z.sigmoid().flatten()[1]
    assert stable_local_peak_topk(z, maximum=1, threshold=.01)[1].tolist() == [[0]]


@pytest.mark.parametrize('shape,maximum', [((2, 1, 7, 9), 13), ((1, 3, 5, 7), 23), ((1, 1, 2, 2), 7)])
def test_scalar_lexicographic_reference_and_legacy_sorted_scores(shape, maximum):
    torch.manual_seed(141)
    z = torch.randint(-4, 4, shape).float()
    new = stable_local_peak_topk(z, maximum=maximum, threshold=.05)
    old = _local_peak_topk(z, maximum=maximum, threshold=.05)
    assert torch.equal(new[1], scalar_order(z, maximum, .05))
    assert torch.equal(new[0], old[0]) and torch.equal(new[2], old[2])
    r = selection_difference(z, old, new, maximum=maximum, threshold=.05)
    assert all(row['changed_members_only_at_exact_cutoff_tie'] for row in r['batches'])


def test_unique_active_peaks_are_unchanged():
    z = torch.full((1, 1, 7, 7), -10.)
    z[0, 0, 0, 0], z[0, 0, 0, 6], z[0, 0, 6, 6] = 3., 2., 1.
    old = _local_peak_topk(z, maximum=3, threshold=.01)
    new = stable_local_peak_topk(z, maximum=3, threshold=.01)
    assert all(torch.equal(a, b) for a, b in zip(old, new))
    assert selection_difference(z, old, new, maximum=3, threshold=.01)['batches'][0]['active_order_unchanged']


def test_peak_suppression_inclusive_threshold_and_padding_remain_unchanged():
    z = torch.tensor([[-1., 0.], [-2., -3.]])[None, None]
    values, ids, mask = stable_local_peak_topk(z, maximum=7, threshold=.5)
    assert mask.tolist() == [[True, False, False, False, False, False, False]]
    assert ids[0, 0] == 1 and values[0, 0] == .5
    assert ids[0, 4:].tolist() == [0, 0, 0] and values[0, 4:].tolist() == [-1., -1., -1.]
    assert not stable_local_peak_topk(z, maximum=7, threshold=.50001)[2].any()


def test_sorting_retained_items_cannot_recover_a_different_topk_set():
    # Two legal old top-k subsets of one same-score plateau remain different
    # after sorting. Canonicalization must happen BEFORE truncation.
    a, b = [4, 3], [2, 1]
    assert sorted(a) != sorted(b)
    z = torch.zeros(1, 1, 1, 6)
    canonical = stable_local_peak_topk(z, maximum=2, threshold=.01)
    for ids in (a, b):
        old = (torch.full((1, 2), .5), torch.tensor([ids]), torch.ones(1, 2, dtype=torch.bool))
        r = selection_difference(z, old, canonical, maximum=2, threshold=.01)
        assert r['batches'][0]['canonical_active_flat_ids'] == [0, 1]


def test_different_dense_values_are_not_mistaken_for_resolved_device_parity():
    a = torch.zeros(1, 2, 1, 1, dtype=torch.float64)
    b = a.clone(); b[0, 1] = 1e-9
    assert stable_local_peak_topk(a, maximum=1, threshold=.01)[1].item() == 0
    assert stable_local_peak_topk(b, maximum=1, threshold=.01)[1].item() == 1


@pytest.mark.parametrize('z,maximum,threshold', [
    (torch.zeros(128), 3, .01), (torch.zeros(1, 1, 0, 2), 3, .01),
    (torch.ones(1, 1, 2, 2, dtype=torch.int64), 3, .01),
    (torch.ones(1, 1, 2, 2, dtype=torch.float16), 3, .01),
    (torch.full((1, 1, 2, 2), float('nan')), 3, .01),
    (torch.full((1, 1, 2, 2), float('inf')), 3, .01),
    (torch.zeros(1, 1, 2, 2), True, .01), (torch.zeros(1, 1, 2, 2), 0, .01),
    (torch.zeros(1, 1, 2, 2), 3, float('nan')), (torch.zeros(1, 1, 2, 2), 3, -1.),
    (torch.zeros(1, 1, 2, 2), 3, True),
])
def test_invalid_or_retained_only_inputs_rejected(z, maximum, threshold):
    with pytest.raises(ValueError): stable_local_peak_topk(z, maximum=maximum, threshold=threshold)


def test_difference_rejects_non_tie_changes_and_stale_values():
    z = torch.tensor([3., 2., 1.]).reshape(1, 3, 1, 1)
    new = stable_local_peak_topk(z, maximum=2, threshold=.01)
    for wrong in ((new[0], torch.tensor([[0, 2]]), new[2]),
                  (new[0]+.01, new[1], new[2])):
        with pytest.raises(ValueError): selection_difference(z, wrong, new, maximum=2, threshold=.01)
