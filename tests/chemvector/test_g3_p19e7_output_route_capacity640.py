import torch

from chemvector.components.graph.correction.output_route_router import (
    OutputRouteRouter,
    OutputRouteRouterConfig,
)
from chemvector.training.graph.correction import (
    train_g3_p19e7_output_route_capacity640 as p19e7,
)


def test_router_initially_defaults_to_keep_fusion():
    router = OutputRouteRouter(OutputRouteRouterConfig(input_dim=5, hidden_dim=8))
    logits = router(torch.randn(4, 5))
    assert logits.argmax(-1).tolist() == [0, 0, 0, 0]


def test_infer_atom_count_round_trip():
    assert p19e7.infer_atom_count(45) == 10
    try:
        p19e7.infer_atom_count(44)
    except ValueError as error:
        assert "not nC2" in str(error)
    else:
        raise AssertionError("invalid pair count was accepted")


def test_observable_features_do_not_depend_on_targets():
    new = torch.linspace(0.01, 0.99, 10)
    old = torch.linspace(0.99, 0.01, 10)
    names, features, selected = p19e7.build_observable_features(new, old)
    assert len(names) == len(features)
    assert len(names) == len(set(names))
    assert set(selected) == {"fusion", "new", "old"}
    assert not any("target" in name or "false_positive" in name for name in names)


def test_route_label_defaults_to_fusion_and_uses_rescue_route():
    assert p19e7._route_label(torch.tensor([True, False, False])) == 0
    assert p19e7._route_label(torch.tensor([False, True, False])) == 1
    assert p19e7._route_label(torch.tensor([False, False, True])) == 2
    assert p19e7._route_label(torch.tensor([False, False, False])) == 0
