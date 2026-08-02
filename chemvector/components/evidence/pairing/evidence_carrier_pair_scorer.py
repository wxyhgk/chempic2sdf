"""Small symmetric scorer for conditional Evidence Carrier boundary pairs.

The scorer consumes only the eight target-free geometry/raster measurements
emitted by ``evidence_carrier_conditional_pair_data``.  It never sees target
IDs, absolute coordinates, molecule metadata, or render identity.  Its scalar
output is an uncalibrated gain for pairing two anchors relative to keeping both
as zero-gain singletons.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from chemvector.evidence_carrier_conditional_pair_data import (
    PAIR_FEATURE_DIM,
    PAIR_FEATURE_NAMES,
    PAIR_NEGATIVE_LABEL,
    PAIR_POSITIVE_LABEL,
)


PAIR_SCORER_PARAMETER_COUNT = 833
PAIR_SCORER_NORMALIZATION_SOURCE = "train-fit"


@dataclass(frozen=True, slots=True)
class EvidenceCarrierPairFeatureTransformConfig:
    """Fixed target-free feature scales, independent of fitted data statistics."""

    maximum_candidate_angle_degrees: float = 8.0
    maximum_candidate_normal_separation_pixels: float = 8.0
    reference_image_size_pixels: int = 768
    minimum_normalization_scale: float = 1e-6

    def __post_init__(self) -> None:
        numeric = (
            self.maximum_candidate_angle_degrees,
            self.maximum_candidate_normal_separation_pixels,
            self.minimum_normalization_scale,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in numeric
        ):
            raise ValueError("Pair feature transform scales must be finite and positive")
        if (
            isinstance(self.reference_image_size_pixels, bool)
            or not isinstance(self.reference_image_size_pixels, int)
            or self.reference_image_size_pixels < 2
        ):
            raise ValueError("Pair feature reference image size must be an integer >= 2")

    @property
    def maximum_reference_overlap_pixels(self) -> float:
        side = float(self.reference_image_size_pixels - 1)
        return math.sqrt(2.0) * side


def _validate_pair_feature_inputs(
    pair_features: torch.Tensor,
    pair_mask: torch.Tensor,
) -> tuple[int, int]:
    if pair_features.ndim != 3 or pair_features.size(-1) != PAIR_FEATURE_DIM:
        raise ValueError(
            "Conditional pair features must have shape [batch,pairs,8]"
        )
    if not pair_features.is_floating_point():
        raise ValueError("Conditional pair features must be floating point")
    if pair_mask.shape != pair_features.shape[:2] or pair_mask.dtype != torch.bool:
        raise ValueError("Conditional pair mask shape or dtype is invalid")
    if pair_mask.device != pair_features.device:
        raise ValueError("Conditional pair features and mask must share one device")
    valid = pair_features[pair_mask]
    if len(valid) and not bool(torch.isfinite(valid).all()):
        raise ValueError("Valid conditional pair features contain non-finite values")
    return int(pair_features.size(0)), int(pair_features.size(1))


def transform_evidence_carrier_pair_features(
    pair_features: torch.Tensor,
    pair_mask: torch.Tensor,
    config: EvidenceCarrierPairFeatureTransformConfig = (
        EvidenceCarrierPairFeatureTransformConfig()
    ),
) -> torch.Tensor:
    """Apply fixed, label-free scaling before fit-only normalization.

    Angle and normal separation are expressed relative to the frozen wide
    candidate profile.  Overlap uses a log scale normalized by the 768px
    reference-canvas diagonal.  The remaining five inputs are already bounded
    fractions or darkness measurements.  Padding is canonicalized to zero.
    """

    _validate_pair_feature_inputs(pair_features, pair_mask)
    valid = pair_features[pair_mask]
    if len(valid):
        if bool((valid[:, :3] < 0.0).any()):
            raise ValueError("Pair angle, separation, and overlap cannot be negative")
        fractions = valid[:, 3:]
        tolerance = 1e-5
        if bool(((fractions < -tolerance) | (fractions > 1.0 + tolerance)).any()):
            raise ValueError("Pair overlap ratios and ink measurements must lie in [0,1]")

    result = pair_features.clone()
    result[..., 0] = (
        result[..., 0] / config.maximum_candidate_angle_degrees
    )
    result[..., 1] = (
        result[..., 1] / config.maximum_candidate_normal_separation_pixels
    )
    overlap_denominator = math.log1p(config.maximum_reference_overlap_pixels)
    result[..., 2] = torch.log1p(result[..., 2].clamp_min(0.0)) / overlap_denominator
    result[..., 3:] = result[..., 3:].clamp(0.0, 1.0)
    return torch.where(pair_mask[..., None], result, torch.zeros_like(result))


class EvidenceCarrierPairScorer(nn.Module):
    """Map symmetric target-free pair descriptors to relative pair gains."""

    model_type = "evidence-carrier-conditional-pair-scorer-v0.1"

    def __init__(
        self,
        transform_config: EvidenceCarrierPairFeatureTransformConfig = (
            EvidenceCarrierPairFeatureTransformConfig()
        ),
    ) -> None:
        super().__init__()
        self.transform_config = transform_config
        self.mlp = nn.Sequential(
            nn.Linear(PAIR_FEATURE_DIM, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        self.register_buffer(
            "feature_mean",
            torch.zeros(PAIR_FEATURE_DIM, dtype=torch.float32),
        )
        self.register_buffer(
            "feature_scale",
            torch.ones(PAIR_FEATURE_DIM, dtype=torch.float32),
        )
        self.register_buffer(
            "normalization_fitted",
            torch.tensor(False, dtype=torch.bool),
        )
        self.register_buffer(
            "normalization_sample_count",
            torch.tensor(0, dtype=torch.long),
        )
        if self.parameter_count() != PAIR_SCORER_PARAMETER_COUNT:
            raise RuntimeError("Conditional pair scorer parameter contract changed")

    def fit_feature_normalization(
        self,
        pair_features: torch.Tensor,
        pair_mask: torch.Tensor,
        *,
        source_partition: str,
    ) -> dict[str, Any]:
        """Fit normalization exactly once from the declared train-fit partition."""

        if source_partition != PAIR_SCORER_NORMALIZATION_SOURCE:
            raise ValueError(
                "Pair normalization may only be fitted from source_partition='train-fit'"
            )
        if bool(self.normalization_fitted):
            raise RuntimeError("Pair feature normalization is already fitted")
        transformed = transform_evidence_carrier_pair_features(
            pair_features,
            pair_mask,
            self.transform_config,
        )
        selected = transformed[pair_mask]
        if len(selected) < 2:
            raise ValueError("Pair normalization requires at least two train-fit rows")
        mean = selected.float().mean(dim=0)
        scale = selected.float().std(dim=0, unbiased=False)
        scale = torch.where(
            scale >= self.transform_config.minimum_normalization_scale,
            scale,
            torch.ones_like(scale),
        )
        with torch.no_grad():
            self.feature_mean.copy_(mean.to(self.feature_mean))
            self.feature_scale.copy_(scale.to(self.feature_scale))
            self.normalization_sample_count.fill_(len(selected))
            self.normalization_fitted.fill_(True)
        return {
            "source_partition": source_partition,
            "sample_count": len(selected),
            "feature_mean": self.feature_mean.detach().cpu().tolist(),
            "feature_scale": self.feature_scale.detach().cpu().tolist(),
        }

    def forward(
        self,
        pair_features: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if not bool(self.normalization_fitted):
            raise RuntimeError("Pair feature normalization has not been fitted")
        transformed = transform_evidence_carrier_pair_features(
            pair_features,
            pair_mask,
            self.transform_config,
        )
        mean = self.feature_mean.to(
            device=transformed.device,
            dtype=transformed.dtype,
        )
        scale = self.feature_scale.to(
            device=transformed.device,
            dtype=transformed.dtype,
        )
        normalized = (transformed - mean) / scale
        normalized = torch.where(
            pair_mask[..., None],
            normalized,
            torch.zeros_like(normalized),
        )
        pair_gains = self.mlp(normalized).squeeze(-1)
        pair_gains = torch.where(
            pair_mask,
            pair_gains,
            torch.zeros_like(pair_gains),
        )
        return {
            "pair_gains": pair_gains,
            "normalized_pair_features": normalized,
            "pair_mask": pair_mask,
        }

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def config_dict(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "pair_feature_names": list(PAIR_FEATURE_NAMES),
            "pair_feature_dim": PAIR_FEATURE_DIM,
            "parameter_count": self.parameter_count(),
            "normalization_source": PAIR_SCORER_NORMALIZATION_SOURCE,
            "transform": {
                "maximum_candidate_angle_degrees": (
                    self.transform_config.maximum_candidate_angle_degrees
                ),
                "maximum_candidate_normal_separation_pixels": (
                    self.transform_config.maximum_candidate_normal_separation_pixels
                ),
                "reference_image_size_pixels": (
                    self.transform_config.reference_image_size_pixels
                ),
                "minimum_normalization_scale": (
                    self.transform_config.minimum_normalization_scale
                ),
            },
        }


def _validate_pair_loss_inputs(
    pair_gains: torch.Tensor,
    pair_indices: torch.Tensor,
    pair_labels: torch.Tensor,
    pair_mask: torch.Tensor,
) -> tuple[int, int]:
    if pair_gains.ndim != 2:
        raise ValueError("Conditional pair gains must have shape [batch,pairs]")
    batch, pairs = pair_gains.shape
    if not pair_gains.is_floating_point():
        raise ValueError("Conditional pair gains must be floating point")
    if pair_indices.shape != (batch, pairs, 2) or pair_indices.dtype != torch.long:
        raise ValueError("Conditional pair indices shape or dtype is invalid")
    if pair_labels.shape != (batch, pairs) or pair_labels.dtype != torch.long:
        raise ValueError("Conditional pair labels shape or dtype is invalid")
    if pair_mask.shape != (batch, pairs) or pair_mask.dtype != torch.bool:
        raise ValueError("Conditional pair loss mask shape or dtype is invalid")
    if not (
        pair_gains.device
        == pair_indices.device
        == pair_labels.device
        == pair_mask.device
    ):
        raise ValueError("Conditional pair loss tensors must share one device")
    if bool(pair_mask.any()):
        valid_gains = pair_gains[pair_mask]
        valid_indices = pair_indices[pair_mask]
        valid_labels = pair_labels[pair_mask]
        if not bool(torch.isfinite(valid_gains).all()):
            raise ValueError("Valid conditional pair gains contain non-finite values")
        if bool((valid_indices < 0).any()) or bool(
            (valid_indices[:, 0] == valid_indices[:, 1]).any()
        ):
            raise ValueError("Valid conditional pair indices must be nonnegative and distinct")
        if bool(
            (
                (valid_labels != PAIR_NEGATIVE_LABEL)
                & (valid_labels != PAIR_POSITIVE_LABEL)
            ).any()
        ):
            raise ValueError("Valid conditional pair labels must be binary")
    return batch, pairs


def evidence_carrier_pair_scorer_loss(
    pair_gains: torch.Tensor,
    pair_indices: torch.Tensor,
    pair_labels: torch.Tensor,
    pair_mask: torch.Tensor,
    *,
    classification_weight: float = 1.0,
    ranking_weight: float = 0.5,
    ranking_margin: float = 1.0,
) -> dict[str, Any]:
    """Balanced per-image BCE plus ranking of shared-anchor conflicts."""

    numeric = (classification_weight, ranking_weight, ranking_margin)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for value in numeric
    ):
        raise ValueError("Conditional pair loss weights and margin must be finite >= 0")
    batch, _ = _validate_pair_loss_inputs(
        pair_gains,
        pair_indices,
        pair_labels,
        pair_mask,
    )
    zero = pair_gains.sum() * 0.0
    image_bce_parts = []
    positive_bce_parts = []
    negative_bce_parts = []
    ranking_parts = []
    positive_count = negative_count = ranking_pair_count = 0

    for batch_index in range(batch):
        valid = pair_mask[batch_index]
        if not bool(valid.any()):
            continue
        gains = pair_gains[batch_index, valid].float()
        labels = pair_labels[batch_index, valid]
        indices = pair_indices[batch_index, valid]
        positive = labels == PAIR_POSITIVE_LABEL
        negative = labels == PAIR_NEGATIVE_LABEL
        class_parts = []
        if bool(positive.any()):
            current = F.softplus(-gains[positive]).mean()
            positive_bce_parts.append(current)
            class_parts.append(current)
            positive_count += int(positive.sum())
        if bool(negative.any()):
            current = F.softplus(gains[negative]).mean()
            negative_bce_parts.append(current)
            class_parts.append(current)
            negative_count += int(negative.sum())
        if class_parts:
            image_bce_parts.append(torch.stack(class_parts).mean())

        if bool(positive.any()) and bool(negative.any()):
            positive_indices = indices[positive]
            negative_indices = indices[negative]
            shared = (
                positive_indices[:, None, :, None]
                == negative_indices[None, :, None, :]
            ).any(dim=-1).any(dim=-1)
            if bool(shared.any()):
                differences = (
                    float(ranking_margin)
                    + gains[negative].unsqueeze(0)
                    - gains[positive].unsqueeze(1)
                )
                ranking_parts.append(F.softplus(differences[shared]).mean())
                ranking_pair_count += int(shared.sum())

    balanced_bce = (
        torch.stack(image_bce_parts).mean() if image_bce_parts else zero
    )
    positive_bce = (
        torch.stack(positive_bce_parts).mean() if positive_bce_parts else zero
    )
    negative_bce = (
        torch.stack(negative_bce_parts).mean() if negative_bce_parts else zero
    )
    ranking_loss = torch.stack(ranking_parts).mean() if ranking_parts else zero
    loss = (
        float(classification_weight) * balanced_bce
        + float(ranking_weight) * ranking_loss
    )
    return {
        "loss": loss,
        "balanced_bce": balanced_bce,
        "positive_bce": positive_bce,
        "negative_bce": negative_bce,
        "ranking_loss": ranking_loss,
        "images_with_pairs": len(image_bce_parts),
        "images_with_positive_pairs": len(positive_bce_parts),
        "images_with_negative_pairs": len(negative_bce_parts),
        "images_with_ranking_pairs": len(ranking_parts),
        "positive_pair_count": positive_count,
        "negative_pair_count": negative_count,
        "ranking_pair_count": ranking_pair_count,
    }


__all__ = [
    "PAIR_SCORER_NORMALIZATION_SOURCE",
    "PAIR_SCORER_PARAMETER_COUNT",
    "EvidenceCarrierPairFeatureTransformConfig",
    "EvidenceCarrierPairScorer",
    "evidence_carrier_pair_scorer_loss",
    "transform_evidence_carrier_pair_features",
]
