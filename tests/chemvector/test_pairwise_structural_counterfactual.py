import pytest
import torch

from chemvector.components.recovery.pairwise_structural_counterfactual import (
    PAIR_EQUIVALENT,
    PAIR_HARMFUL,
    PAIR_RESCUE,
    PAIRWISE_ARMS,
    PairwiseStructuralCounterfactual,
    PairwiseStructuralCounterfactualConfig,
    pairwise_counterfactual_loss,
    sample_centered_patches,
    sample_relation_bands,
)


def _inputs(batch: int = 4) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(7)
    image = torch.rand(batch, 1, 96, 96)
    score = torch.rand(batch, 1, 96, 96)
    source = torch.zeros(batch, 2)
    prior = torch.tensor([[0.01, -0.005]]).repeat(batch, 1)
    alternative = torch.tensor([[-0.008, 0.012]]).repeat(batch, 1)
    mask = torch.ones(batch, dtype=torch.bool)
    return image, score, source, prior, alternative, mask


@pytest.mark.parametrize("arm", PAIRWISE_ARMS)
def test_pairwise_model_is_exactly_antisymmetric(arm: str) -> None:
    model = PairwiseStructuralCounterfactual(
        PairwiseStructuralCounterfactualConfig(arm=arm)
    )
    with torch.no_grad():
        model.utility[-1].weight.normal_()
    image, score, source, prior, alternative, mask = _inputs()
    forward = model(image, score, source, prior, alternative, mask)
    reverse = model(image, score, source, alternative, prior, mask)
    torch.testing.assert_close(forward, -reverse, atol=1e-6, rtol=1e-6)
    assert model.parameter_count < 300_000


def test_pairwise_explicit_score_evidence_remains_antisymmetric() -> None:
    model = PairwiseStructuralCounterfactual(
        PairwiseStructuralCounterfactualConfig(
            arm="patch-only", explicit_score_evidence=True
        )
    )
    with torch.no_grad():
        model.utility[-1].weight.normal_()
    image, score, source, prior, alternative, mask = _inputs()
    forward = model(image, score, source, prior, alternative, mask)
    reverse = model(image, score, source, alternative, prior, mask)
    torch.testing.assert_close(forward, -reverse, atol=1e-6, rtol=1e-6)
    assert model.config_dict()["explicit_score_evidence"] is True


def test_pairwise_mask_and_gradient() -> None:
    model = PairwiseStructuralCounterfactual()
    image, score, source, prior, alternative, mask = _inputs()
    mask[-1] = False
    delta = model(image, score, source, prior, alternative, mask)
    assert delta[-1].item() == 0.0
    labels = torch.tensor(
        [PAIR_RESCUE, PAIR_HARMFUL, PAIR_EQUIVALENT, PAIR_HARMFUL]
    )
    losses = pairwise_counterfactual_loss(
        delta, torch.zeros_like(delta), labels, mask
    )
    losses["loss"].backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_pairwise_visual_samplers_have_fixed_shapes() -> None:
    image, score, source, prior, _, _ = _inputs(batch=2)
    values = torch.cat((image, score), dim=1)
    patches = sample_centered_patches(values, prior, 25)
    bands = sample_relation_bands(
        values, source, prior, width=17, samples=48
    )
    assert patches.shape == (2, 2, 25, 25)
    assert bands.shape == (2, 2, 17, 48)
    with pytest.raises(ValueError, match="band offsets"):
        sample_relation_bands(
            values, source[:1], prior, width=17, samples=48
        )


def test_pairwise_rejects_invalid_inputs() -> None:
    model = PairwiseStructuralCounterfactual()
    image, score, source, prior, alternative, mask = _inputs()
    with pytest.raises(ValueError, match="pair_mask"):
        model(image, score, source, prior, alternative, mask.float())
