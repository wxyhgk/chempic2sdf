"""Synthetic leaf-logit math only: no real samples, checkpoint, or model."""

from dataclasses import replace
import itertools
import math

import pytest
import torch
from torch.nn import functional as F

from tools.selector_group_objective import (
    GroupSpec,
    SELECTION_THRESHOLD,
    evaluate_group_objective,
    exactly_one_nll,
)


def leaf(values):
    return torch.tensor(values, dtype=torch.float64, requires_grad=True)


def evaluate(values, groups):
    logits = leaf(values)
    result = evaluate_group_objective(logits, groups)
    gradient, = torch.autograd.grad(result.total, logits)
    return result, gradient


def group(slots=(0, 1), representatives=(0,), fragments=()):
    return GroupSpec("node", slots, representatives, fragments)


def test_gradcheck_and_gradgradcheck():
    groups = [group(fragments=(2,)), GroupSpec("second", (3,), (3,))]
    logits = leaf([-0.9, 0.3, 1.8, -2.1, 4.0])
    objective = lambda z: evaluate_group_objective(z, groups).total
    assert torch.autograd.gradcheck(objective, (logits,))
    assert torch.autograd.gradgradcheck(objective, (logits,))


@pytest.mark.parametrize("count", [1, 2, 3, 5])
def test_exact_probability_by_enumerating_bernoulli_outcomes(count):
    logits = torch.linspace(-1.4, 2.0, count, dtype=torch.float64)
    probabilities = torch.sigmoid(logits)
    exactly_one = 0.0
    for bits in itertools.product((0, 1), repeat=count):
        if sum(bits) == 1:
            probability = math.prod(float(p if bit else 1-p) for p, bit in zip(probabilities, bits))
            exactly_one += probability
    assert float(exactly_one_nll(logits)) == pytest.approx(-math.log(exactly_one))


def test_gradient_matches_closed_form_not_raw_positive_labels():
    logits = leaf([-1.2, 0.8, -0.4])
    result = evaluate_group_objective(logits, [group((0, 1, 2), (0,))])
    gradient, = torch.autograd.grad(result.total, logits)
    expected = torch.sigmoid(logits-SELECTION_THRESHOLD) - torch.softmax(logits, dim=0)
    assert torch.allclose(gradient, expected, atol=1e-14)
    changed_label, changed_gradient = evaluate(logits.detach().tolist(), [group((0, 1, 2), (1,))])
    assert torch.equal(gradient, changed_gradient)
    assert torch.equal(result.total, changed_label.total)


def test_two_candidate_missing_group_raises_both_valid_representatives():
    _, gradient = evaluate([-3.0, -3.0], [group()])
    assert bool((gradient < 0).all())  # Gradient descent raises both.


@pytest.mark.parametrize("offsets", [[-0.1, -0.1, -0.1], [-0.11, -0.10, -0.09],
                                      [-0.12, -0.11, -0.10, -0.09, -0.08]])
def test_open_neighborhood_missing_all_is_pushed_down_all_contract_failure(offsets):
    values = [SELECTION_THRESHOLD+offset for offset in offsets]
    _, gradient = evaluate(values, [group(tuple(range(len(values))))])
    assert all(value < SELECTION_THRESHOLD for value in values)
    assert bool((gradient > 0).all())
    # This is the undesired direction, even with distinct scores. The test
    # exposes failure of "missing -> at least one representative rises";
    # it must NOT be interpreted as a passed training-readiness check.


def test_duplicates_above_threshold_both_decrease_initially():
    _, gradient = evaluate([3.0, 3.0], [group()])
    assert bool((gradient > 0).all())


def test_alternative_can_be_retained_and_raw_representative_suppressed():
    _, gradient = evaluate([-2.0, 2.0], [group()])
    assert gradient[0] > 0 and gradient[1] < 0


def test_explicit_fragments_down_unknowns_zero_and_no_input_mutation():
    logits = leaf([-1.0, 1.0, 4.0, 0.5, -0.5, 0.7])
    before = logits.detach().clone()
    excluded = GroupSpec("sensitive", (3,), (), (4,), "projection_sensitive")
    result = evaluate_group_objective(logits, [group(fragments=(2,)), excluded])
    gradient, = torch.autograd.grad(result.total, logits)
    assert gradient[2] > 0 and torch.equal(gradient[3:], torch.zeros(3, dtype=torch.float64))
    assert torch.equal(before, logits.detach())
    assert not result.training_authorized
    assert result.supervised_slots == (0, 1, 2)
    assert result.excluded_nodes == (("sensitive", "projection_sensitive"),)


@pytest.mark.parametrize("reason", ["unknown", "quantization_sensitive", "no_raw_representative"])
def test_exclusion_with_no_representative_ignores_even_obvious_fragment(reason):
    logits = leaf([3.0, 5.0])
    result = evaluate_group_objective(logits, [GroupSpec("node", (0,), (), (1,), reason)])
    gradient, = torch.autograd.grad(result.total, logits)
    assert result.total.item() == 0 and torch.equal(gradient, torch.zeros_like(logits))


def test_empty_groups_have_finite_differentiable_zero():
    logits = leaf([2.0, -1.0])
    result = evaluate_group_objective(logits, [])
    gradient, = torch.autograd.grad(result.total, logits)
    assert result.total.item() == 0 and torch.equal(gradient, torch.zeros_like(logits))


def test_single_candidate_is_plain_positive_softplus():
    logits = leaf([-2.1])
    result = evaluate_group_objective(logits, [group((0,), (0,))])
    assert torch.equal(result.node_loss, F.softplus(-(logits[0]-SELECTION_THRESHOLD)))


@pytest.mark.parametrize("size", [2, 3, 5])
def test_symmetric_stationary_point_is_explicit_failure_not_success(size):
    value = SELECTION_THRESHOLD - math.log(size-1)
    result, gradient = evaluate([value]*size, [group(tuple(range(size)))])
    assert bool((gradient.abs() < 1e-14).all())
    hard_count = sum(value >= SELECTION_THRESHOLD for _ in range(size))
    assert hard_count != 1  # Two selected for size=2; none for size>2.
    assert result.node_loss.item() > 0


def test_symmetric_stationary_point_is_saddle_with_negative_contrast_curvature():
    logits = leaf([SELECTION_THRESHOLD, SELECTION_THRESHOLD])
    hessian = torch.autograd.functional.hessian(
        lambda z: evaluate_group_objective(z, [group()]).total, logits
    )
    contrast = torch.tensor([1.0, -1.0], dtype=torch.float64)
    shared = torch.tensor([1.0, 1.0], dtype=torch.float64)
    assert contrast @ hessian @ contrast < 0
    assert shared @ hessian @ shared > 0


def test_tied_gradient_cannot_choose_one_candidate():
    _, gradient = evaluate([0.0, 0.0], [group()])
    assert gradient[0] == gradient[1]


def test_group_and_candidate_order_invariance():
    values = [-1.1, 1.3, 0.2, -3.0, 2.0, 0.5]
    groups = [group((0, 1, 2), (0,), (3,)), GroupSpec("other", (4,), (4,), (5,))]
    baseline, gradient = evaluate(values, groups)
    reverse, reverse_gradient = evaluate(values, [groups[1], replace(groups[0], acceptable_slots=(2, 0, 1))])
    assert torch.allclose(baseline.total, reverse.total, atol=1e-14)
    assert torch.allclose(gradient, reverse_gradient, atol=1e-14)
    permutation = [3, 2, 0, 5, 1, 4]
    inverse = {old: new for new, old in enumerate(permutation)}
    permuted_groups = [replace(g,
        acceptable_slots=tuple(inverse[s] for s in g.acceptable_slots),
        representative_slots=tuple(inverse[s] for s in g.representative_slots),
        fragment_slots=tuple(inverse[s] for s in g.fragment_slots)) for g in groups]
    permuted, permuted_gradient = evaluate([values[s] for s in permutation], permuted_groups)
    assert torch.allclose(baseline.total, permuted.total, atol=1e-14)
    assert torch.allclose(gradient[permutation], permuted_gradient, atol=1e-14)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("values", [[1000., 1000.], [-1000., -1000.], [-1000., 1000.], [1000.], [-1000.]])
def test_extreme_scores_have_finite_nonnegative_loss_and_gradients(dtype, values):
    logits = torch.tensor(values, dtype=dtype, requires_grad=True)
    spec = group(tuple(range(len(values))))
    result = evaluate_group_objective(logits, [spec])
    gradient, = torch.autograd.grad(result.total, logits)
    assert torch.isfinite(result.total) and result.total >= 0
    assert bool(torch.isfinite(gradient).all())


@pytest.mark.parametrize("specs,match", [
    ([group((0, 0))], "Duplicate slot"),
    ([group(representatives=(0, 0))], "Duplicate slot"),
    ([group(fragments=(2, 2))], "Duplicate slot"),
    ([group(fragments=(1,))], "cannot also be a fragment"),
    ([group(representatives=(2,))], "must belong"),
    ([group(representatives=())], "explicitly exclude"),
    ([group((), ())], "explicitly exclude"),
    ([group((0, 9))], "in-range"),
    ([group((0, True))], "in-range"),
    ([group(), GroupSpec("node", (2,), (2,))], "Duplicate node"),
    ([group(), GroupSpec("other", (1,), (1,))], "Overlapping"),
    ([group(), GroupSpec("excluded", (1,), (), (), "unknown")], "Overlapping"),
    ([replace(group(), exclusion_reason="")], "nonempty reason"),
])
def test_malformed_or_ambiguous_groups_fail_closed(specs, match):
    with pytest.raises(ValueError, match=match):
        evaluate_group_objective(leaf([0.0, 0.0, 0.0]), specs)


@pytest.mark.parametrize("logits", [torch.zeros(2, 2), torch.tensor([1, 2]),
                                     torch.tensor([float("nan")]), torch.tensor([float("inf")]),
                                     torch.tensor([1.0], dtype=torch.float16)])
def test_invalid_logits_rejected(logits):
    with pytest.raises(ValueError, match="CPU float32/float64"):
        evaluate_group_objective(logits, [])


def test_no_optimization_model_or_data_dependency_in_module():
    import inspect
    import tools.selector_group_objective as module
    source = inspect.getsource(module)
    assert "torch.optim" not in source and "torch.load" not in source
    assert "open(" not in source and "read_text(" not in source
    assert "cuda(" not in source and ".backward(" not in source
