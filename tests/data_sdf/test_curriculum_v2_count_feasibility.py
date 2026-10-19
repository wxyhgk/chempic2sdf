"""Closed-form scalar intervals, no model access or coefficient search."""

import pytest

from tools.check_curriculum_v02_count_feasibility import feasible_interval


def row(intercept, slope, positive=True):
    return dict(intercept=intercept, slope=slope, positive=positive)


def test_conflicting_foreground_background_bounds():
    result = feasible_interval([row(1, -10), row(2, -10, False)])
    assert not result['feasible']
    assert result['required_upper'] == pytest.approx(.1)
    assert result['required_lower'] == pytest.approx(.2)
    assert result['upper_witness']['positive'] and not result['lower_witness']['positive']


def test_feasible_closed_interval_and_exact_zero_boundary():
    result = feasible_interval([row(5, -10), row(2, -10, False)])
    assert result['feasible'] and result['required_lower'] == .2 and result['required_upper'] == .5
    result = feasible_interval([row(2, -10), row(2, -10, False)])
    assert result['feasible'] and result['required_lower'] == result['required_upper'] == .2


@pytest.mark.parametrize('constraint', [row(-1, 0), row(1, 0, False), row(-2, 1), row(2, 1, False)])
def test_constant_or_out_of_domain_contradictions(constraint):
    assert not feasible_interval([constraint])['feasible']


def test_empty_or_nonstrict_zero_constraints():
    for constraints in ([], [row(0, 0)], [row(0, 0, False)]):
        result = feasible_interval(constraints)
        assert result['feasible'] and result['required_lower'] == 0 and result['required_upper'] == 1


@pytest.mark.parametrize('constraint', [row(float('nan'), 0), row(0, float('inf')), row(0, 1, 1)])
def test_invalid_constraints(constraint):
    with pytest.raises(ValueError):
        feasible_interval([constraint])
