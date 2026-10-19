"""Synthetic mask, negative sampler and direct-logit gradient contracts only."""

from copy import deepcopy
from itertools import combinations

import pytest
import torch
import torch.nn.functional as F

from tools import pair_supervision_mask as masks
from tests.chemvector.test_selector_missing_graph_trace import fixture


def state_fixture():
    mapping = [-1, 1, 2, 3]
    bonds = [dict(begin_atom=0, end_atom=1), dict(begin_atom=1, end_atom=2)]
    labels = {p: p == (1, 2) for p in combinations(range(4), 2)}
    owners = {i: dict(target_ids=[i], source='original_match' if i else 'hypothesis') for i in range(4)}
    return mapping, labels, owners, bonds


def test_two_fixed_policies_preserve_positives_and_isolate_possible_edges():
    args = state_fixture()
    before = deepcopy(args)
    rows = masks.classify_pairs(*args)
    by_pair = {tuple(r['pair']): r for r in rows}
    assert by_pair[(0, 1)]['matched_only'] == by_pair[(0, 1)]['conditional_ownership'] == 'ignore'
    assert by_pair[(0, 2)]['matched_only'] == 'ignore'
    assert by_pair[(0, 2)]['conditional_ownership'] == 'negative'
    assert by_pair[(0, 2)]['reason'] == 'conditional_nonedge'
    assert by_pair[(1, 2)]['matched_only'] == by_pair[(1, 2)]['conditional_ownership'] == 'positive'
    summary = masks.summarize(rows)
    assert summary['matched_only']['positive'] == summary['conditional_ownership']['positive'] == 1
    assert summary['matched_only']['ignore'] == 3
    assert summary['conditional_ownership']['ignore'] == 1
    assert summary['conditional_ownership']['hypothesis_dependent_negatives'] == 2
    assert args == before


@pytest.mark.parametrize('target_ids, reason', [([], 'unresolved_owner'), ([1], 'possible_same_atom'),
                                             ([0, 3], 'possible_true_bond')])
def test_conditional_policy_abstains_on_unknown_overlap_or_any_possible_bond(target_ids, reason):
    mapping, labels, owners, bonds = state_fixture()
    owners[0]['target_ids'] = target_ids
    row = masks.classify_pairs(mapping, labels, owners, bonds)[0]
    assert row['pair'] == [0, 1]
    assert row['conditional_ownership'] == 'ignore'
    assert row['reason'] == reason


def test_two_uncertain_endpoints_cannot_be_promoted_to_positive():
    mapping = [-1, -1]
    owners = {0: dict(target_ids=[0]), 1: dict(target_ids=[1])}
    row = masks.classify_pairs(mapping, {(0, 1): False}, owners,
                               [dict(begin_atom=0, end_atom=1)])[0]
    assert not row['original_presence']
    assert row['matched_only'] == row['conditional_ownership'] == 'ignore'


@pytest.mark.parametrize('bad', ['positive_missing', 'positive_invented', 'overwrite_match', 'duplicate_owner'])
def test_inconsistent_inputs_fail_closed(bad):
    mapping, labels, owners, bonds = state_fixture()
    if bad == 'positive_missing':
        labels[(1, 2)] = False
    elif bad == 'positive_invented':
        labels[(0, 1)] = True
    elif bad == 'overwrite_match':
        owners[1]['target_ids'] = [0]
    else:
        owners[0]['target_ids'] = [0, 0]
    with pytest.raises(ValueError):
        masks.classify_pairs(mapping, labels, owners, bonds)


def test_seven_pixel_text_node_does_not_make_true_bond_a_supervised_negative():
    geo, record, layers = fixture()
    record['group'] = 'synthetic'
    before = deepcopy((geo, record, layers))
    rng = torch.get_rng_state().clone()
    result = masks.inspect_case(geo, record, layers)
    row = next(r for r in result['pairs'] if r['pair'] == [0, 1])
    assert row['diagnostic_reference'] == 'bond'
    assert not row['original_presence']
    assert row['matched_only'] == row['conditional_ownership'] == 'ignore'
    assert result['ownership_hypotheses'][0]['source'] == 'radius_and_or_diagnostic_hypothesis'
    assert result['diagnostic_reference_counts']['current']['bond_negative'] == 1
    assert result['diagnostic_reference_counts']['matched_only']['bond_ignore'] == 1
    assert (geo, record, layers) == before
    assert torch.equal(torch.get_rng_state(), rng)


def test_duplicate_candidates_are_not_declared_background():
    geo, record, _ = fixture()
    selected = masks.trace.selected_candidates(geo, record)
    selected[0]['geometry'][:2] = [.401, .5]
    mapping = [-1, 1, 2]
    owners, diagnostic = masks.ownership_hypotheses(geo, selected, mapping)
    assert not diagnostic[0]['accepted_for_diagnostic_only']
    assert owners[0]['target_ids'] == [1]


def test_remote_unmatched_candidate_stays_unresolved_not_background():
    geo, record, _ = fixture()
    selected = masks.trace.selected_candidates(geo, record)
    selected[0]['geometry'][:2] = [.2, .8]
    owners, _ = masks.ownership_hypotheses(geo, selected, [-1, 1, 2])
    assert owners[0] == dict(target_ids=[], source='unresolved')


def sampler_fixture(batch_size=1):
    base = torch.tensor(list(combinations(range(4), 2)))
    pairs = base.repeat(batch_size, 1)
    samples = torch.arange(batch_size).repeat_interleave(len(base))
    presence = torch.tensor([True, False, False, False, False, False]).repeat(batch_size)
    targets = dict(presence=presence, chemical=torch.zeros(len(pairs), dtype=torch.long),
                   structured_negative=torch.zeros(len(pairs), dtype=torch.bool))
    atoms = dict(atom_geometry=torch.tensor([[[.1, .1], [.2, .2], [.3, .3], [.4, .4]]]).repeat(batch_size, 1, 1))
    mask = torch.tensor([True, False, False, True, True, True]).repeat(batch_size)
    return pairs, samples, atoms, targets, mask


@pytest.mark.parametrize('batch_size', [1, 2])
def test_ignored_rows_removed_before_sampling_and_do_not_consume_quota(batch_size):
    pairs, samples, atoms, targets, mask = sampler_fixture(batch_size)
    with torch.random.fork_rng():
        torch.manual_seed(7)
        selected = masks.sample_with_mask(pairs, samples, atoms, targets, mask,
                                         negative_ratio=1, minimum_negatives=2)
    assert bool(mask[selected].all())
    assert len(selected) == 3 * batch_size
    for sample in range(batch_size):
        chosen = selected[samples[selected] == sample]
        assert len(chosen) == 3
        assert int(targets['presence'][chosen].sum()) == 1
    assert set(torch.where(targets['presence'])[0].tolist()) <= set(selected.tolist())


def test_all_supervised_sampler_is_exact_legacy_with_same_rng():
    pairs, samples, atoms, targets, mask = sampler_fixture(2)
    mask[:] = True
    with torch.random.fork_rng():
        torch.manual_seed(91)
        expected = masks.select_training_pairs(pairs, samples, atoms, targets, negative_ratio=1, minimum_negatives=2)
        torch.manual_seed(91)
        actual = masks.sample_with_mask(pairs, samples, atoms, targets, mask, negative_ratio=1, minimum_negatives=2)
    assert torch.equal(expected, actual)


def test_no_positive_no_supervised_rows_returns_empty():
    pairs, samples, atoms, targets, mask = sampler_fixture()
    targets['presence'].zero_()
    mask.zero_()
    selected = masks.sample_with_mask(pairs, samples, atoms, targets, mask)
    assert selected.dtype == torch.long and selected.shape == (0,)


def test_cannot_mask_positive_or_use_nonboolean_mask():
    pairs, samples, atoms, targets, mask = sampler_fixture()
    with pytest.raises(ValueError, match='boolean'):
        masks.sample_with_mask(pairs, samples, atoms, targets, mask.float())
    mask[0] = False
    with pytest.raises(ValueError, match='original positive'):
        masks.sample_with_mask(pairs, samples, atoms, targets, mask)


def test_ignored_direct_logits_have_zero_gradient_and_valid_signs_remain():
    pairs, samples, atoms, targets, mask = sampler_fixture()
    # Synthetic independent logits, not a learned model or shared-parameter claim.
    logits = torch.zeros(len(pairs), requires_grad=True)
    selected = masks.sample_with_mask(pairs, samples, atoms, targets, mask)
    loss = F.binary_cross_entropy_with_logits(logits[selected], targets['presence'][selected].float())
    loss.backward()
    assert torch.equal(logits.grad[~mask], torch.zeros_like(logits.grad[~mask]))
    assert bool((logits.grad[targets['presence']] < 0).all())
    assert bool((logits.grad[mask & ~targets['presence']] > 0).all())


@pytest.mark.parametrize('positives,negatives,expected', [(0, 0, 0), (0, 3, 3), (1, 80, 48), (20, 500, 120)])
def test_capacity_uses_unchanged_sampler_budget(positives, negatives, expected):
    assert masks.sampling_capacity(positives, negatives) == expected
