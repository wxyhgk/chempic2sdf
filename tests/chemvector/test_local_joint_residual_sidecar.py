import pytest
import torch

from tools.train_local_joint_residual_sidecar import (
    ResidualPairSidecar,
    _choose_threshold,
    _evaluate,
)


def test_sidecar_parameter_budget_and_finite_forward():
    model = ResidualPairSidecar()
    assert sum(p.numel() for p in model.parameters()) < 10_000
    output = model(torch.randn(3, 11))
    assert output.shape == (3,) and bool(torch.isfinite(output).all())


def test_threshold_is_selected_from_train_metrics_only():
    scores = torch.tensor([.99, .98, .20, .10])
    labels = torch.tensor([True, True, False, False])
    image_indices = torch.tensor([0, 0, 1, 1])
    result = _choose_threshold(scores, labels, image_indices)
    assert result["selected"] is not None
    selected = result["selected"]["threshold"]
    assert selected in (.50, .60, .70, .80, .85, .90, .95, .98)
    holdout = _evaluate(
        torch.tensor([.99, .1]), torch.tensor([False, True]),
        torch.tensor([0, 0]), selected,
    )
    assert holdout["threshold"] == selected


def test_evaluate_none_threshold_is_non_admissible_without_selection():
    result = _evaluate(torch.tensor([.1]), torch.tensor([False]), torch.tensor([0]), None)
    assert result["selected"] == 0 and result["precision"] == 0.0


def test_residual_sidecar_rejects_wrong_feature_width():
    model = ResidualPairSidecar()
    with pytest.raises(RuntimeError):
        model(torch.zeros(1, 10))

