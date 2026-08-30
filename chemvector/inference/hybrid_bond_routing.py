"""Run and route base and style-expert BondObject detectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from chemvector.bond_expert_utility import combined_utility_features
from chemvector.dense_bond_object import (
    MultiScaleDenseBondObjectDetector,
    decode_dense_bond_objects,
)
from chemvector.pretrain_dense_bond_object import _apply_detector


@dataclass(frozen=True, slots=True)
class HybridBondRoutingRuntime:
    device: torch.device
    base: Any
    detector: Any
    expert_detector: Any
    style_gate: Any
    style_threshold: float
    utility_router: Any
    utility_feature_names: Any
    utility_feature_mean: Any
    utility_feature_std: Any
    utility_threshold: Any


def bond_detector_features(
    stride4: torch.Tensor, runtime: HybridBondRoutingRuntime
) -> tuple[torch.Tensor, ...]:
    if not isinstance(runtime.detector, MultiScaleDenseBondObjectDetector):
        return (stride4,)
    stride8 = runtime.base.vision_encoder.forward_high_resolution_from_stroke(
        stride4
    )
    stride16 = stride8
    for stage in runtime.base.vision_encoder.stages[3:5]:
        stride16 = stage(stride16)
    return stride4, stride8, stride16


def decode_routed_bond_objects(
    stride4: torch.Tensor,
    *,
    threshold: float,
    runtime: HybridBondRoutingRuntime,
) -> tuple[list[list[dict]], dict]:
    """Decode BondObjects and conservatively select the optional expert."""

    detector_features = bond_detector_features(stride4, runtime)
    base_ranked_batches = decode_dense_bond_objects(
        _apply_detector(runtime.detector, detector_features),
        threshold=0.0,
        maximum_objects=128,
    )
    if runtime.expert_detector is None:
        selected = [
            [item for item in ranked if float(item["score"]) >= threshold]
            for ranked in base_ranked_batches
        ]
        return selected, {
            "bond_style_expert_probability": 0.0,
            "selected_bond_expert": "base",
        }

    probabilities = runtime.style_gate(stride4).float().sigmoid()
    expert_ranked_batches = decode_dense_bond_objects(
        _apply_detector(runtime.expert_detector, detector_features),
        threshold=0.0,
        maximum_objects=128,
    )
    if runtime.utility_router is None:
        route_probabilities = probabilities
        route_threshold = runtime.style_threshold
        route_kind = "style"
    else:
        feature_rows = []
        for base_ranked, expert_ranked, style_probability in zip(
            base_ranked_batches, expert_ranked_batches, probabilities
        ):
            values = combined_utility_features(
                base_ranked,
                expert_ranked,
                style_probability=float(style_probability),
            )
            feature_rows.append(
                [values[name] for name in runtime.utility_feature_names]
            )
        feature_tensor = torch.tensor(
            feature_rows, device=runtime.device, dtype=torch.float32
        )
        feature_tensor = (
            feature_tensor - runtime.utility_feature_mean
        ) / runtime.utility_feature_std
        route_probabilities = runtime.utility_router(
            feature_tensor
        ).float().sigmoid()
        route_threshold = runtime.utility_threshold
        route_kind = "utility"
    routes = [float(value) >= route_threshold for value in route_probabilities]
    selected_ranked = [
        expert if use_expert else base
        for base, expert, use_expert in zip(
            base_ranked_batches, expert_ranked_batches, routes
        )
    ]
    selected = [
        [item for item in ranked if float(item["score"]) >= threshold]
        for ranked in selected_ranked
    ]
    diagnostics = {
        "bond_style_expert_probability": (
            float(probabilities[0]) if len(probabilities) == 1 else None
        ),
        "bond_utility_expert_probability": (
            float(route_probabilities[0])
            if route_kind == "utility" and len(route_probabilities) == 1
            else None
        ),
        "selected_bond_expert": (
            "style_expert" if routes[0] else "base"
        ) if len(routes) == 1 else None,
        "bond_style_expert_routes": sum(routes),
        "bond_expert_route_kind": route_kind,
        "bond_expert_route_threshold": route_threshold,
    }
    return selected, diagnostics
