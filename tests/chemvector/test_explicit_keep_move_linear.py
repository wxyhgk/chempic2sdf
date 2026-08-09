import torch

from chemvector.components.recovery.explicit_keep_move_linear import (
    EXPLICIT_KEEP_MOVE_FEATURES,
    ExplicitKeepMoveLinear,
    fit_explicit_keep_move_linear,
)


def test_explicit_linear_has_eight_parameters() -> None:
    model = ExplicitKeepMoveLinear()
    assert model.parameter_count == 8
    assert model(torch.zeros(3, EXPLICIT_KEEP_MOVE_FEATURES)).shape == (3,)


def test_explicit_linear_fit_is_deterministic_and_separates_signal() -> None:
    torch.manual_seed(31)
    features = torch.randn(128, EXPLICIT_KEEP_MOVE_FEATURES)
    labels = features[:, 0] + 0.5 * features[:, 1] > 0.0
    first, first_audit = fit_explicit_keep_move_linear(features, labels)
    second, second_audit = fit_explicit_keep_move_linear(features, labels)
    for name, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[name], atol=0, rtol=0)
    assert first_audit == second_audit
    assert first_audit["training_accuracy"] > 0.95
