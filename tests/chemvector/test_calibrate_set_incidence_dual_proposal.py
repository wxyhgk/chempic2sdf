import pytest
import torch

from chemvector.calibrate_set_incidence_dual_proposal import interpolated_state


def test_interpolated_state_scales_floating_tensors() -> None:
    base = {"weight": torch.tensor([1.0, 3.0]), "count": torch.tensor(2)}
    trained = {"weight": torch.tensor([5.0, 7.0]), "count": torch.tensor(4)}

    result = interpolated_state(base, trained, 0.25)

    torch.testing.assert_close(result["weight"], torch.tensor([2.0, 4.0]))
    assert result["count"].item() == 2


def test_interpolated_state_rejects_invalid_scale() -> None:
    with pytest.raises(ValueError):
        interpolated_state({"x": torch.ones(1)}, {"x": torch.ones(1)}, 1.1)
