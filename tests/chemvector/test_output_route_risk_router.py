import pytest
import torch

from chemvector.components.graph.correction.output_route_risk_router import (
    OutputRouteRiskRouter,
    OutputRouteRiskRouterConfig,
)


def test_output_route_risk_router_shape_and_zero_initialization():
    router = OutputRouteRiskRouter(
        OutputRouteRiskRouterConfig(input_dim=7, hidden_dim=8)
    )
    logits = router(torch.randn(3, 7))
    assert logits.shape == (3, 2, 4)
    assert torch.equal(logits, torch.zeros_like(logits))


def test_output_route_risk_router_rejects_wrong_shapes():
    with pytest.raises(ValueError):
        OutputRouteRiskRouterConfig(input_dim=0)
    router = OutputRouteRiskRouter(OutputRouteRiskRouterConfig(input_dim=7))
    with pytest.raises(ValueError):
        router(torch.randn(3, 6))
