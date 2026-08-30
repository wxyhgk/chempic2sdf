"""Conservative utility routing between primary and expert OCR predictions."""

from __future__ import annotations

from typing import Any, Protocol, Sequence

import torch

from chemvector.ctc_ocr import ctc_text_log_score
from chemvector.ocr_utility_router import (
    add_ocr_cross_score_features,
    add_ocr_visual_features,
    ocr_utility_features,
)


class OCRUtilityRoutingRuntime(Protocol):
    """The narrow runtime contract required by OCR utility routing."""

    device: torch.device
    utility_router: Any
    utility_feature_names: Sequence[str]
    utility_feature_mean: torch.Tensor
    utility_feature_std: torch.Tensor
    utility_threshold: float


def route_ocr_predictions(
    primary: list[tuple[str, float]],
    expert: list[tuple[str, float]],
    *,
    runtime: OCRUtilityRoutingRuntime,
    primary_visual: torch.Tensor | None = None,
    expert_visual: torch.Tensor | None = None,
    primary_logits: torch.Tensor | None = None,
    expert_logits: torch.Tensor | None = None,
) -> tuple[list[tuple[str, float]], int, int]:
    """Conservatively replace primary OCR predictions with expert predictions."""

    if len(primary) != len(expert) or runtime.utility_router is None:
        raise ValueError("OCR utility routing inputs differ")
    selected = list(primary)
    disagreements = [
        index
        for index, (old, new) in enumerate(zip(primary, expert))
        if old[0] != new[0]
    ]
    if not disagreements:
        return selected, 0, 0
    uses_visual = any(
        name.startswith("old_visual_") for name in runtime.utility_feature_names
    )
    uses_cross_scores = (
        "old_model_new_text_score" in runtime.utility_feature_names
    )
    rows = []
    for index in disagreements:
        row = ocr_utility_features(*primary[index], *expert[index])
        if uses_visual:
            if primary_visual is None or expert_visual is None:
                raise ValueError("Visual OCR router requires paired descriptors")
            row = add_ocr_visual_features(
                row, primary_visual[index], expert_visual[index]
            )
        if uses_cross_scores:
            if primary_logits is None or expert_logits is None:
                raise ValueError("Cross-score OCR router requires paired logits")
            row = add_ocr_cross_score_features(
                row,
                old_model_new_text_score=ctc_text_log_score(
                    primary_logits[index],
                    expert[index][0],
                    normalize_by_bytes=True,
                ),
                new_model_old_text_score=ctc_text_log_score(
                    expert_logits[index],
                    primary[index][0],
                    normalize_by_bytes=True,
                ),
            )
        rows.append(row)
    features = torch.tensor(
        [[row[name] for name in runtime.utility_feature_names] for row in rows],
        device=runtime.device,
        dtype=torch.float32,
    )
    features = (
        features - runtime.utility_feature_mean
    ) / runtime.utility_feature_std
    with torch.autocast(device_type=runtime.device.type, dtype=torch.bfloat16):
        probabilities = runtime.utility_router(features).float().sigmoid()
    routes = probabilities >= runtime.utility_threshold
    for row, route in zip(disagreements, routes.tolist()):
        if route:
            selected[row] = expert[row]
    return selected, len(disagreements), int(routes.sum())
