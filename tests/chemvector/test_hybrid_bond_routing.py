from __future__ import annotations

import torch

import chemvector.hybrid_bond_routing as routing


def _runtime(**overrides) -> routing.HybridBondRoutingRuntime:
    values = {
        "device": torch.device("cpu"),
        "base": None,
        "detector": "base",
        "expert_detector": None,
        "style_gate": None,
        "style_threshold": 0.9,
        "utility_router": None,
        "utility_feature_names": None,
        "utility_feature_mean": None,
        "utility_feature_std": None,
        "utility_threshold": None,
    }
    values.update(overrides)
    return routing.HybridBondRoutingRuntime(**values)


def test_base_detector_filters_by_threshold_and_reports_no_route(monkeypatch) -> None:
    monkeypatch.setattr(routing, "_apply_detector", lambda model, features: model)
    monkeypatch.setattr(
        routing,
        "decode_dense_bond_objects",
        lambda output, **kwargs: [[{"score": 0.8}, {"score": 0.2}]],
    )
    selected, diagnostics = routing.decode_routed_bond_objects(
        torch.zeros((1, 4, 8, 8)), threshold=0.5, runtime=_runtime()
    )
    assert selected == [[{"score": 0.8}]]
    assert diagnostics == {
        "bond_style_expert_probability": 0.0,
        "selected_bond_expert": "base",
    }


def test_style_gate_can_select_expert_detector(monkeypatch) -> None:
    monkeypatch.setattr(routing, "_apply_detector", lambda model, features: model)

    def decode(output, **kwargs):
        score = 0.7 if output == "base" else 0.95
        return [[{"source": output, "score": score}]]

    monkeypatch.setattr(routing, "decode_dense_bond_objects", decode)
    class Gate:
        def __call__(self, stride4):
            return torch.tensor((10.0,))

    selected, diagnostics = routing.decode_routed_bond_objects(
        torch.zeros((1, 4, 8, 8)),
        threshold=0.5,
        runtime=_runtime(expert_detector="expert", style_gate=Gate()),
    )
    assert selected == [[{"source": "expert", "score": 0.95}]]
    assert diagnostics["selected_bond_expert"] == "style_expert"
    assert diagnostics["bond_style_expert_routes"] == 1
    assert diagnostics["bond_expert_route_kind"] == "style"
