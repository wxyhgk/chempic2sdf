"""Synthetic CPU tests only, including tiny toy-optimizer parity checks."""

from copy import deepcopy

import pytest
import torch
from torch import nn

from chemvector.direct_atom_pair_bond import direct_atom_pair_loss
from chemvector.training.curriculum import pair_anchor_wiring as wiring
from tests.chemvector.test_curriculum_supervision_contract import pair_fixture


def test_independent_visit_counters_break_global_step_aliasing():
    keys = [str(i) for i in range(58)]
    cycle = wiring.PerImageOffsetCycle(keys)
    seen = {k: [] for k in keys}
    for _ in range(8):
        for key in keys:
            seen[key].append(cycle.next(key)['view_index'])
    assert all(values == list(range(1, 9)) for values in seen.values())
    assert {attempt % 8 for attempt in range(0, 58 * 8, 58)} == {0, 2, 4, 6}


def test_empty_or_irregular_visits_advance_without_reading_labels():
    cycle = wiring.PerImageOffsetCycle(['empty', 'known'])
    assert cycle.next('empty')['view_index'] == 1
    assert cycle.next('empty')['view_index'] == 2
    assert cycle.next('known')['view_index'] == 1
    assert cycle.next('empty')['view_index'] == 3
    with pytest.raises(ValueError): cycle.next('new')


@pytest.mark.parametrize('keys', [[], ['x', 'x'], [1], ['']])
def test_invalid_schedule_scope_rejected(keys):
    with pytest.raises(ValueError): wiring.PerImageOffsetCycle(keys)


def test_sparse_shuffled_row_ids_map_to_positions_not_ids():
    chosen, rows = torch.tensor([31, 2, 1523, 7]), torch.tensor([1523, 31, 7])
    result = wiring.sampled_positions(chosen, rows)
    assert result.tolist() == [2, 0, 3]
    assert torch.equal(chosen[result], rows)
    assert wiring.sampled_positions(chosen, torch.empty(0, dtype=torch.long)).numel() == 0


@pytest.mark.parametrize('chosen,rows', [([3, 3], [3]), ([3, 8], [8, 8]),
                                      ([3, 8], [4]), ([3, 8], [9]), ([], [0]), ([-1], [])])
def test_invalid_mapping_rejected(chosen, rows):
    with pytest.raises(ValueError):
        wiring.sampled_positions(torch.tensor(chosen, dtype=torch.long), torch.tensor(rows, dtype=torch.long))


class ToyPair(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 5)
        self.calls = []

    def forward(self, features, atom_features, anchors, scores, pairs, samples):
        self.calls.append(tuple(t.detach().clone() for t in (features, atom_features, anchors, scores, pairs, samples)))
        x = torch.stack((features.mean().expand(len(pairs)),
                         atom_features[samples, pairs[:, 0], 0],
                         anchors[samples, pairs[:, 1], 0],
                         scores[samples, pairs[:, 0]]), -1)
        logits = self.linear(x)
        return dict(presence_logits=logits[:, 0], chemical_logits=logits[:, 1:])


def fixture():
    atoms, mapping, targets = pair_fixture()
    atoms['atom_features'] = torch.arange(8.).reshape(1, 4, 2).requires_grad_()
    atoms['atom_scores'] = torch.tensor([[.7, .8, .9, .1]], requires_grad=True)
    atoms['atom_geometry'].requires_grad_()
    features = torch.ones(1, 2, 3, 3, requires_grad=True)
    return features, atoms, mapping, targets, torch.tensor([3, 0, 1])


def test_actual_call_wiring_keeps_features_scores_and_row_labels():
    data = fixture()
    head = ToyPair().eval()
    before = deepcopy(data)
    result = wiring.paired_objective(head, *data, view_index=1)
    assert result['pair_forwards'] == 2
    assert result['planned_sampled_positions'].tolist() == [0, 1, 2]
    first, second = head.calls
    assert first[4].tolist() == [[1, 2], [0, 1], [0, 2]]
    for index in (0, 1, 3, 4, 5): assert torch.equal(first[index], second[index])
    torch.testing.assert_close(second[2] - first[2], torch.full_like(first[2], -4 / 768))
    result['losses']['loss'].backward()
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in head.parameters())
    assert data[0].grad is None
    assert all(data[1][k].grad is None for k in ('atom_features', 'atom_scores', 'atom_geometry'))
    assert torch.equal(data[0], before[0])
    assert all(torch.equal(data[1][k], before[1][k]) for k in data[1])


def test_partial_view_uses_selected_positions_and_source_types():
    data = fixture()
    with torch.no_grad(): data[1]['atom_geometry'][0, 0, 0] += 4 / 768
    head = ToyPair().eval()
    result = wiring.paired_objective(head, *data, view_index=5)  # +4,0: atom 0 now outside radius
    assert result['planned_augmented_row_ids'].tolist() == [3]
    assert result['planned_sampled_positions'].tolist() == [0]
    assert head.calls[1][4].tolist() == [[1, 2]]  # original DOUBLE, not the row-id=0 SINGLE


def test_empty_augmentation_never_calls_empty_forward_and_keeps_base():
    data = fixture()
    with torch.no_grad(): data[1]['atom_geometry'][0, :3, 0] += 4 / 768
    result = wiring.paired_objective(ToyPair().eval(), *data, view_index=5)
    assert result['pair_forwards'] == 1
    assert not result['losses']['augmentation_applied']


def test_unknown_empty_base_and_negative_only_remain_distinct():
    data = list(fixture())
    data[2].fill_(-1); data[-1] = torch.empty(0, dtype=torch.long)
    result = wiring.paired_objective(ToyPair().eval(), *data, view_index=1)
    assert result['losses'] is None and result['pair_forwards'] == 0 and result['skipped_empty_base']
    data = fixture()
    data[3]['bond_mask'].zero_()
    result = wiring.paired_objective(ToyPair().eval(), *data, view_index=1)
    assert result['pair_forwards'] == 2 and not result['skipped_empty_base']
    assert result['losses']['original']['chemical_loss'] == 0
    assert result['losses']['augmented']['chemical_loss'] == 0


def assert_nested_equal(a, b):
    if isinstance(a, torch.Tensor): assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a: assert_nested_equal(a[key], b[key])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b): assert_nested_equal(x, y)
    else: assert a == b


def test_zero_weight_multistep_shared_parameter_optimizer_and_rng_parity():
    data = fixture()
    reference, wired = ToyPair().eval(), ToyPair().eval()
    wired.load_state_dict(reference.state_dict())
    optimizers = [torch.optim.AdamW(h.parameters(), lr=.001) for h in (reference, wired)]
    contract = wiring.pair_targets_with_unknown(data[1], data[2], data[3])
    chosen = data[-1]
    labels = contract['legacy_targets']
    rng = torch.get_rng_state().clone()
    for _ in range(3):
        for opt in optimizers: opt.zero_grad(set_to_none=True)
        out = reference(data[0].detach(), data[1]['atom_features'].detach(),
                        data[1]['atom_geometry'][..., :2].detach(), data[1]['atom_scores'].detach(),
                        contract['pair_indices'][chosen], contract['sample_indices'][chosen])
        loss = direct_atom_pair_loss(out, labels['presence'][chosen], labels['chemical'][chosen])['loss']
        result = wiring.paired_objective(wired, *data, view_index=1, weight=0)
        assert result['pair_forwards'] == 1
        assert torch.equal(result['losses']['loss'], loss)
        loss.backward(); result['losses']['loss'].backward()
        for x, y in zip(reference.parameters(), wired.parameters()): assert torch.equal(x.grad, y.grad)
        for opt in optimizers: opt.step()  # toy CPU parameters only, not real training
        assert_nested_equal(reference.state_dict(), wired.state_dict())
        assert_nested_equal(optimizers[0].state_dict(), optimizers[1].state_dict())
    assert torch.equal(rng, torch.get_rng_state())


def test_invalid_view_or_train_mode_rejected_before_forward():
    head = ToyPair()
    with pytest.raises(ValueError): wiring.paired_objective(head, *fixture(), view_index=1)
    assert head.calls == []
    head.eval()
    with pytest.raises(ValueError): wiring.paired_objective(head, *fixture(), view_index=0)
    with pytest.raises(ValueError): wiring.paired_objective(head, *fixture(), view_index=1, weight=.5)


def test_unknown_chosen_row_rejected_before_any_pair_call():
    data = list(fixture())
    data[-1] = torch.tensor([2])  # candidate 3 is unmatched
    head = ToyPair().eval()
    with pytest.raises(ValueError): wiring.paired_objective(head, *data, view_index=1)
    assert head.calls == []
