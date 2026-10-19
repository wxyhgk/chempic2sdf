"""Independent loss arithmetic; no real sample/model/optimizer access."""

import numpy as np
import pytest
import torch

from tools.audit_curriculum_v02_selector_matching import atom_loss_probe
from tools.check_curriculum_v02_selector_loss_replay import analytic_derivatives


@pytest.mark.parametrize("values,labels", [
    ([-1, -.1] + [-5] * 126, [True, True] + [False] * 126),
    ([-1, -1, 0, 0], [True, True, False, False]),
    ([5, -5], [True, False]),
    ([0, 0, 0], [True, True, True]),
    ([0, 0, 0], [False, False, False]),
    ([-.2, .5, -20], [True, True, False]),
])
def test_analytic_and_autograd_agree(values, labels):
    z, target = torch.tensor([values], dtype=torch.float32), torch.tensor([labels])
    _, gradients = atom_loss_probe(z, target, torch.ones_like(target))
    expected = analytic_derivatives(values, labels)
    for key, values in expected.items():
        np.testing.assert_allclose(gradients[key][0].numpy(), values, atol=2e-7, rtol=1e-6)


def test_count_pressure_can_dominate_a_retained_correct_candidate():
    # Focal pushes the second true candidate up, but summed background mass
    # makes the count term push it down; ranking touches only the worse first.
    values = [-1., -.1] + [-3.] * 126
    labels = [True, True] + [False] * 126
    result = analytic_derivatives(values, labels)
    assert -.25 <= values[1] < 0
    assert result["focal"][1] < 0 and result["weighted_count"][1] > 0
    assert result["weighted_ranking"][1] == 0 and result["atom_only_total"][1] > 0
    assert result["atom_only_total"][0] < 0  # Worst rejected positive still pushed up.


@pytest.mark.parametrize("values,labels", [([], []), ([1], [True, False]),
    ([float("nan")], [True]), ([[1]], [[True]])])
def test_invalid_scalar_cache(values, labels):
    with pytest.raises(ValueError):
        analytic_derivatives(values, labels)
