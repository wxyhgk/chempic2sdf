import torch

from chemvector.components.recovery.structural_peak_residual import (
    StructuralPeakResidual, candidate_centered_patches, normalized_peak_prior, residual_selection_loss,
)


def _inputs(batch: int = 2):
    return (torch.rand(batch,1,96,96),torch.rand(batch,1,96,96),(torch.rand(batch,8,2)-0.5)*24/768,torch.rand(batch,8),torch.ones(batch,8,dtype=torch.bool),torch.zeros(batch),torch.zeros(batch))


def test_residual_step0_exactly_reproduces_prior() -> None:
    model=StructuralPeakResidual(arm="multiscale",cap=2.0).eval(); inputs=_inputs()
    details=model.forward_details(*inputs)
    torch.testing.assert_close(details["scores"],normalized_peak_prior(inputs[3],inputs[4]))
    assert not bool(details["residual"].any())


def test_candidate_patches_have_fixed_shape() -> None:
    inputs=_inputs(); patches=candidate_centered_patches(torch.cat(inputs[:2],dim=1),inputs[2],17)
    assert patches.shape==(2,8,2,17,17)


def test_residual_loss_backpropagates_with_hard_and_easy() -> None:
    model=StructuralPeakResidual(arm="global",cap=2.0); inputs=list(_inputs()); inputs[2].zero_(); inputs[2][:,1,0]=10/768
    targets=torch.stack((inputs[2][0,1],inputs[2][1,0])); details=model.forward_details(*inputs)
    losses=residual_selection_loss(details,inputs[2],inputs[4],targets,torch.tensor([True,False])); losses["loss"].backward()
    assert all(value.grad is not None for value in model.parameters())
