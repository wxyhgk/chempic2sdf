import torch

from chemvector.components.bonds.objects.proposal_endpoint_residual import (
    ProposalEndpointResidualConfig,
    ProposalEndpointResidualHead,
    proposal_endpoint_residual_loss,
)


def _inputs():
    return (
        torch.randn(2, 5, 8),
        torch.rand(2, 5, 2, 2),
        torch.rand(2, 5, 4),
        torch.rand(2, 5),
        torch.tensor([[True] * 5, [True, True, True, False, False]]),
    )


def test_zero_initialization_preserves_endpoints():
    config = ProposalEndpointResidualConfig(feature_dim=8, type_dim=4, hidden_dim=16)
    head = ProposalEndpointResidualHead(config)
    features, endpoints, types, scores, mask = _inputs()
    output = head(features, endpoints, types, scores, mask)
    assert torch.equal(output["endpoints"], endpoints)
    assert torch.count_nonzero(output["endpoint_delta"]) == 0


def test_endpoint_delta_is_bounded_and_masked():
    config = ProposalEndpointResidualConfig(
        feature_dim=8,
        type_dim=4,
        hidden_dim=16,
        maximum_delta_pixels=24,
        image_size=768,
    )
    head = ProposalEndpointResidualHead(config)
    with torch.no_grad():
        head.network[-1].bias.fill_(10.0)
    features, endpoints, types, scores, mask = _inputs()
    output = head(features, endpoints, types, scores, mask)
    delta_px = output["endpoint_delta"].abs() * config.image_size
    assert float(delta_px.max().detach()) <= config.maximum_delta_pixels + 1e-4
    assert torch.equal(output["endpoints"][~mask], endpoints[~mask])


def test_loss_trains_positive_and_preserves_negative():
    config = ProposalEndpointResidualConfig(feature_dim=8, type_dim=4, hidden_dim=16)
    head = ProposalEndpointResidualHead(config)
    features, endpoints, types, scores, mask = _inputs()
    output = head(features, endpoints, types, scores, mask)
    targets = endpoints.clone()
    targets[:, 0, 0, 0] += 4.0 / 768.0
    supervision = torch.zeros_like(mask)
    supervision[:, 0] = True
    losses = proposal_endpoint_residual_loss(
        output, endpoints, targets, supervision, mask
    )
    losses["loss"].backward()
    assert float(losses["endpoint_loss"].detach()) > 0
    assert head.network[-1].weight.grad is not None
