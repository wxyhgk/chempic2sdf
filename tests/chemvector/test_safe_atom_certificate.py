import torch

from chemvector.safe_atom_action_model import ACTION_FEATURE_DIM
from chemvector.safe_atom_certificate import (
    UnsafeActionCertificate,
    molecule_fold,
    strict_unsafe_threshold,
    unsafe_certificate_loss,
)


def test_certificate_is_small_and_backpropagates() -> None:
    model = UnsafeActionCertificate()
    assert model.parameter_count() < 100_000
    logits = model(torch.randn(8, ACTION_FEATURE_DIM))
    loss = unsafe_certificate_loss(
        logits, torch.tensor([True, False, True, False, True, False, False, True])
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_molecule_folds_are_stable_and_grouped() -> None:
    assert molecule_fold("chembl:CHEMBL1") == molecule_fold("chembl:CHEMBL1")
    assert 0 <= molecule_fold("chembl:CHEMBL2") < 5


def test_strict_threshold_rejects_every_calibration_unsafe() -> None:
    probabilities = torch.tensor([0.1, 0.3, 0.2], dtype=torch.float32)
    threshold = strict_unsafe_threshold(probabilities)
    assert threshold < float(probabilities.min())
    assert not bool((probabilities < threshold).any())
