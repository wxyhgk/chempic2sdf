import numpy as np
import pytest
import torch

from chemvector.experiments.recurrent_depth.data import assign
from chemvector.experiments.recurrent_depth.metrics import decide, score_records
from chemvector.experiments.recurrent_depth.model import CandidateRefiner


@pytest.fixture
def inputs():
    torch.manual_seed(11)
    return torch.randn(2, 6, 10), torch.tensor([[True] * 4 + [False] * 2, [True] * 6]), torch.randn(2, 6)


def active_model(depth=1):
    torch.manual_seed(5)
    model = CandidateRefiner(10, 16, 4, depth).eval()
    torch.nn.init.normal_(model.output.weight, std=.1)
    return model


def test_zero_residual_reproduces_baseline(inputs):
    x, mask, base = inputs
    model = CandidateRefiner(10, 16)
    for loops in (1, 2, 4, 8):
        result = model(x, mask, base, loops)
        torch.testing.assert_close(result[mask], base[mask], rtol=0, atol=0)
        assert (result[~mask] == -1e4).all()


def test_parameter_count_does_not_grow_with_recurrence(inputs):
    tied, untied = active_model(), active_model(4)
    before = sum(p.numel() for p in tied.parameters())
    tied(*inputs, loops=8)
    assert sum(p.numel() for p in tied.parameters()) == before
    assert sum(p.numel() for p in untied.parameters()) > before
    assert untied.blocks[0].linear1.weight.data_ptr() != untied.blocks[1].linear1.weight.data_ptr()
    torch.testing.assert_close(untied.blocks[0].linear1.weight, untied.blocks[1].linear1.weight)


def test_tied_and_untied_have_identical_initial_forward(inputs):
    torch.testing.assert_close(active_model()(*inputs, loops=4), active_model(4)(*inputs, loops=4))


def test_recurrence_updates_state_not_identical_reexecution(inputs):
    model = active_model()
    assert not torch.allclose(model(*inputs, loops=1), model(*inputs, loops=4))


def test_candidate_permutation_equivariance(inputs):
    x, mask, base = inputs
    model = active_model()
    order = torch.tensor([4, 1, 5, 0, 2, 3])
    torch.testing.assert_close(model(x[:, order], mask[:, order], base[:, order], 4), model(x, mask, base, 4)[:, order], atol=1e-6, rtol=1e-5)


def test_padding_cannot_change_valid_predictions(inputs):
    x, mask, base = inputs
    model = active_model()
    original = model(x, mask, base, 4)
    changed = x.clone()
    changed[~mask] = 1e6
    torch.testing.assert_close(model(changed, mask, base, 4), original)


def test_all_empty_mask_is_finite(inputs):
    x, mask, base = inputs
    result = active_model()(x, torch.zeros_like(mask), base, 4)
    assert torch.isfinite(result).all() and (result == -1e4).all()


def test_inputs_unchanged_and_core_has_gradient(inputs):
    model = active_model()
    original = [t.clone() for t in inputs]
    model(*inputs, loops=4).sum().backward()
    assert model.blocks[0].linear1.weight.grad.abs().sum() > 0
    for before, after in zip(original, inputs):
        assert torch.equal(before, after)


@pytest.mark.parametrize('depth,loops', [(1, 0), (4, 2)])
def test_invalid_loop_count_rejected(inputs, depth, loops):
    with pytest.raises(ValueError):
        active_model(depth)(*inputs, loops=loops)


def test_matching_prioritizes_cardinality_and_masks():
    anchors = np.array([[0., 0.], [5., 0.], [90., 90.]])
    targets = np.array([[1., 0.], [-5., 0.]])
    assert set(assign(anchors, np.ones(3, dtype=bool), targets)) == {0, 1}
    assert assign(anchors, np.array([False, False, True]), targets).size == 0
    assert assign(anchors, np.zeros(3, dtype=bool), targets).size == 0


def test_scoring_counts_unreachable_targets_and_duplicate_proposals():
    record = {'key': 'one', 'mask': torch.ones(3, dtype=torch.bool),
              'anchors': torch.tensor([[0., 0.], [1., 0.], [10., 0.]]),
              'targets': torch.tensor([[0., 0.], [100., 100.]])}
    metrics, rows = score_records([record], np.array([[1., 1., -1.]]))
    assert metrics['tp'] == 1 and metrics['fp'] == 1 and metrics['fn'] == 1
    assert metrics['candidate_oracle_recall'] == .5 and metrics['exact'] == 0
    # A different duplicate can recover the object; scoring is not frozen to a training label.
    metrics, _ = score_records([record], np.array([[-1., 1., -1.]]))
    assert metrics['tp'] == 1 and metrics['fp'] == 0


def test_gate_requires_compute_control_and_source_guard():
    trials = []
    for seed in (17, 29, 43):
        for arm, loops, exact in [('tied1', 1, .5), ('tied4', 4, .55), ('untied4', 4, .7)]:
            trials.append({'seed': seed, 'arm': arm, 'trained_loops': loops, 'evaluation_loops': loops,
                           'metrics': {'exact': exact, 'fp_per_image': 1., 'recall': .9}})
    gate = {'min_exact_gain_vs_tied1': .01, 'max_exact_drop_vs_untied4': .01,
            'max_fp_increase_per_image': .1, 'min_positive_seeds': 2, 'max_exact_drop_vs_source': .01}
    result = decide(trials, {'exact': .8}, gate)
    assert not result['passed'] and not result['checks']['same_unrolled_compute_control']
    assert not result['checks']['frozen_source_guard']
