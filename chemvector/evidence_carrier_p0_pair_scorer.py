"""Symmetric pair scorer conditioned on frozen P0 carrier evidence.

The module keeps the original eight target-free pair measurements, adds a
shared symmetric projection of each frozen P0 ``carrier_features`` vector, and
adds explicit consistency measurements between the two P0-decoded lines and
the target-free overlap centerline.  It contains no P0 parameters and accepts
no target IDs or absolute-position embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal

import torch
from torch import nn

from chemvector.evidence_carrier_conditional_pair_data import PAIR_FEATURE_DIM
from chemvector.evidence_carrier_extent import (
    EvidenceCarrierExtentPredictor,
    decode_evidence_carrier_extent,
)
from chemvector.evidence_carrier_pair_scorer import (
    PAIR_SCORER_NORMALIZATION_SOURCE,
    EvidenceCarrierPairFeatureTransformConfig,
    transform_evidence_carrier_pair_features,
)


P0_PAIR_CONSISTENCY_FEATURE_NAMES = (
    "decoded_mutual_angle_degrees",
    "decoded_mutual_normal_distance_pixels",
    "decoded_mutual_interval_disagreement",
    "decoded_mutual_endpoint_error_q",
    "decoded_to_centerline_angle_min_degrees",
    "decoded_to_centerline_normal_distance_min_pixels",
    "decoded_to_centerline_interval_disagreement_min",
    "decoded_to_centerline_endpoint_error_min_q",
    "decoded_to_centerline_angle_max_degrees",
    "decoded_to_centerline_normal_distance_max_pixels",
    "decoded_to_centerline_interval_disagreement_max",
    "decoded_to_centerline_endpoint_error_max_q",
)
P0_PAIR_CONSISTENCY_FEATURE_DIM = len(P0_PAIR_CONSISTENCY_FEATURE_NAMES)
P0_PAIR_DESCRIPTOR_DIM = PAIR_FEATURE_DIM + P0_PAIR_CONSISTENCY_FEATURE_DIM
P0_PAIR_SCORER_PARAMETER_COUNT = 11873


@dataclass(frozen=True, slots=True)
class EvidenceCarrierP0PairScorerConfig:
    anchor_feature_dim: int = 128
    anchor_projection_dim: int = 32
    pair_hidden_dim: int = 64
    coordinate_max: float = 2047.0
    normal_distance_scale_pixels: float = 8.0
    minimum_line_length_pixels: float = 1e-6
    minimum_normalization_scale: float = 1e-6

    def __post_init__(self) -> None:
        for name in (
            "anchor_feature_dim",
            "anchor_projection_dim",
            "pair_hidden_dim",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "coordinate_max",
            "normal_distance_scale_pixels",
            "minimum_line_length_pixels",
            "minimum_normalization_scale",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive")


def _validate_pair_geometry_inputs(
    decoded_segments_pixels: torch.Tensor,
    pair_indices: torch.Tensor,
    pair_centerlines_pixels: torch.Tensor,
    pair_mask: torch.Tensor,
    canvas_size_pixels: torch.Tensor,
) -> tuple[int, int, int]:
    if (
        decoded_segments_pixels.ndim != 4
        or decoded_segments_pixels.shape[-2:] != (2, 2)
        or not decoded_segments_pixels.is_floating_point()
    ):
        raise ValueError("P0 decoded segments must have shape [batch,anchors,2,2]")
    batch, anchors = decoded_segments_pixels.shape[:2]
    if pair_indices.ndim != 3 or pair_indices.shape[0] != batch or pair_indices.size(-1) != 2:
        raise ValueError("P0 pair indices must have shape [batch,pairs,2]")
    if pair_indices.dtype != torch.long:
        raise ValueError("P0 pair indices must use torch.long")
    pairs = pair_indices.size(1)
    if pair_centerlines_pixels.shape != (batch, pairs, 2, 2) or not (
        pair_centerlines_pixels.is_floating_point()
    ):
        raise ValueError("P0 pair centerlines must have shape [batch,pairs,2,2]")
    if pair_mask.shape != (batch, pairs) or pair_mask.dtype != torch.bool:
        raise ValueError("P0 pair consistency mask shape or dtype is invalid")
    if canvas_size_pixels.shape != (batch, 2) or not canvas_size_pixels.is_floating_point():
        raise ValueError("P0 canvas_size_pixels must contain [width,height] per image")
    devices = {
        decoded_segments_pixels.device,
        pair_indices.device,
        pair_centerlines_pixels.device,
        pair_mask.device,
        canvas_size_pixels.device,
    }
    if len(devices) != 1:
        raise ValueError("P0 pair consistency tensors must share one device")
    if bool(pair_mask.any()):
        indices = pair_indices[pair_mask]
        if (
            anchors < 1
            or bool((indices < 0).any())
            or bool((indices >= anchors).any())
            or bool((indices[:, 0] == indices[:, 1]).any())
        ):
            raise ValueError("Active P0 pair indices reference invalid anchors")
        if not bool(torch.isfinite(decoded_segments_pixels).all()) or not bool(
            torch.isfinite(pair_centerlines_pixels[pair_mask]).all()
        ):
            raise ValueError("Active P0 pair geometry contains non-finite values")
    if not bool(torch.isfinite(canvas_size_pixels).all()) or bool(
        (canvas_size_pixels < 2.0).any()
    ):
        raise ValueError("P0 pair canvas dimensions must be finite and >= 2")
    return batch, anchors, pairs


def _gather_pair_segments(
    segments: torch.Tensor,
    pair_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, pairs = pair_indices.shape[:2]
    safe = pair_indices.clamp_min(0)
    expanded = segments[:, None].expand(-1, pairs, -1, -1, -1)
    gather = safe[..., None, None].expand(-1, -1, -1, 2, 2)
    selected = torch.gather(expanded, 2, gather)
    return selected[:, :, 0], selected[:, :, 1]


def _gather_pair_features(
    features: torch.Tensor,
    pair_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, pairs = pair_indices.shape[:2]
    safe = pair_indices.clamp_min(0)
    expanded = features[:, None].expand(-1, pairs, -1, -1)
    gather = safe[..., None].expand(-1, -1, -1, features.size(-1))
    selected = torch.gather(expanded, 2, gather)
    return selected[:, :, 0], selected[:, :, 1]


def _line_pair_components(
    first: torch.Tensor,
    second: torch.Tensor,
    canvas_size_pixels: torch.Tensor,
    *,
    config: EvidenceCarrierP0PairScorerConfig,
) -> torch.Tensor:
    first_delta = first[..., 1, :] - first[..., 0, :]
    second_delta = second[..., 1, :] - second[..., 0, :]
    first_length = first_delta.norm(dim=-1)
    second_length = second_delta.norm(dim=-1)
    minimum = config.minimum_line_length_pixels
    if bool((first_length <= minimum).any()) or bool((second_length <= minimum).any()):
        raise ValueError("Active P0 decoded pair contains a degenerate segment")
    first_axis = first_delta / first_length.unsqueeze(-1)
    second_axis = second_delta / second_length.unsqueeze(-1)
    dot = (first_axis * second_axis).sum(dim=-1).clamp(-1.0, 1.0)
    angle = torch.rad2deg(torch.acos(dot.abs()))

    first_normal = torch.stack((-first_axis[..., 1], first_axis[..., 0]), dim=-1)
    second_normal = torch.stack((-second_axis[..., 1], second_axis[..., 0]), dim=-1)
    center_delta = second.mean(dim=-2) - first.mean(dim=-2)
    normal_distance = torch.maximum(
        (center_delta * first_normal).sum(dim=-1).abs(),
        (center_delta * second_normal).sum(dim=-1).abs(),
    )

    aligned_second = torch.where(
        (dot >= 0.0).unsqueeze(-1), second_axis, -second_axis
    )
    common = first_axis + aligned_second
    common_length = common.norm(dim=-1)
    common_axis = torch.where(
        (common_length > minimum).unsqueeze(-1),
        common / common_length.clamp_min(minimum).unsqueeze(-1),
        first_axis,
    )
    first_projection = (first * common_axis.unsqueeze(-2)).sum(dim=-1)
    second_projection = (second * common_axis.unsqueeze(-2)).sum(dim=-1)
    first_low, first_high = first_projection.min(dim=-1).values, first_projection.max(dim=-1).values
    second_low, second_high = second_projection.min(dim=-1).values, second_projection.max(dim=-1).values
    intersection = (
        torch.minimum(first_high, second_high) - torch.maximum(first_low, second_low)
    ).clamp_min(0.0)
    union = torch.maximum(first_high, second_high) - torch.minimum(first_low, second_low)
    interval_iou = intersection / union.clamp_min(minimum)

    direct = (first - second).norm(dim=-1).max(dim=-1).values
    reverse = (first - second.flip(-2)).norm(dim=-1).max(dim=-1).values
    endpoint_error_pixels = torch.minimum(direct, reverse)
    q_per_pixel = torch.maximum(
        config.coordinate_max / (canvas_size_pixels[..., 0] - 1.0),
        config.coordinate_max / (canvas_size_pixels[..., 1] - 1.0),
    )
    endpoint_error_q = endpoint_error_pixels * q_per_pixel
    return torch.stack(
        (angle, normal_distance, 1.0 - interval_iou, endpoint_error_q),
        dim=-1,
    )


def p0_decoded_pair_consistency_features(
    decoded_segments_pixels: torch.Tensor,
    pair_indices: torch.Tensor,
    pair_centerlines_pixels: torch.Tensor,
    pair_mask: torch.Tensor,
    canvas_size_pixels: torch.Tensor,
    config: EvidenceCarrierP0PairScorerConfig = EvidenceCarrierP0PairScorerConfig(),
) -> torch.Tensor:
    """Return 12 target-free symmetric decoded-line consistency measurements."""

    batch, _, pairs = _validate_pair_geometry_inputs(
        decoded_segments_pixels,
        pair_indices,
        pair_centerlines_pixels,
        pair_mask,
        canvas_size_pixels,
    )
    if pairs == 0:
        return decoded_segments_pixels.new_empty(
            (batch, 0, P0_PAIR_CONSISTENCY_FEATURE_DIM)
        )
    first, second = _gather_pair_segments(decoded_segments_pixels, pair_indices)
    size = canvas_size_pixels[:, None, :].expand(-1, pairs, -1)
    valid_geometry = pair_mask[..., None, None]
    safe_line = decoded_segments_pixels.new_tensor(((0.0, 0.0), (1.0, 0.0)))
    first = torch.where(valid_geometry, first, safe_line)
    second = torch.where(valid_geometry, second, safe_line)
    centerline = torch.where(valid_geometry, pair_centerlines_pixels, safe_line)
    mutual = _line_pair_components(first, second, size, config=config)
    first_center = _line_pair_components(first, centerline, size, config=config)
    second_center = _line_pair_components(second, centerline, size, config=config)
    minimum = torch.minimum(first_center, second_center)
    maximum = torch.maximum(first_center, second_center)
    result = torch.cat((mutual, minimum, maximum), dim=-1)
    return torch.where(pair_mask[..., None], result, torch.zeros_like(result))


def transform_p0_pair_consistency_features(
    features: torch.Tensor,
    pair_mask: torch.Tensor,
    config: EvidenceCarrierP0PairScorerConfig = EvidenceCarrierP0PairScorerConfig(),
) -> torch.Tensor:
    if features.ndim != 3 or features.size(-1) != P0_PAIR_CONSISTENCY_FEATURE_DIM:
        raise ValueError("P0 consistency features must have shape [batch,pairs,12]")
    if pair_mask.shape != features.shape[:2] or pair_mask.dtype != torch.bool:
        raise ValueError("P0 consistency feature mask shape or dtype is invalid")
    valid = features[pair_mask]
    if len(valid) and (
        not bool(torch.isfinite(valid).all()) or bool((valid < 0.0).any())
    ):
        raise ValueError("Active P0 consistency features must be finite and nonnegative")
    result = features.clone()
    angle_indices = (0, 4, 8)
    normal_indices = (1, 5, 9)
    disagreement_indices = (2, 6, 10)
    endpoint_indices = (3, 7, 11)
    result[..., list(angle_indices)] = result[..., list(angle_indices)] / 90.0
    result[..., list(normal_indices)] = (
        result[..., list(normal_indices)] / config.normal_distance_scale_pixels
    )
    result[..., list(disagreement_indices)] = result[
        ..., list(disagreement_indices)
    ].clamp(0.0, 1.0)
    denominator = math.log1p(config.coordinate_max)
    result[..., list(endpoint_indices)] = torch.log1p(
        result[..., list(endpoint_indices)]
    ) / denominator
    return torch.where(pair_mask[..., None], result, torch.zeros_like(result))


class EvidenceCarrierP0PairScorer(nn.Module):
    """Score pairs from frozen P0 anchor embeddings and explicit consistency."""

    model_type = "evidence-carrier-p0-pair-scorer-v0.1"

    def __init__(
        self,
        config: EvidenceCarrierP0PairScorerConfig = EvidenceCarrierP0PairScorerConfig(),
        pair_transform_config: EvidenceCarrierPairFeatureTransformConfig = (
            EvidenceCarrierPairFeatureTransformConfig()
        ),
    ) -> None:
        super().__init__()
        self.config = config
        self.pair_transform_config = pair_transform_config
        self.anchor_projector = nn.Sequential(
            nn.Linear(config.anchor_feature_dim, config.anchor_projection_dim),
            nn.GELU(),
            nn.LayerNorm(config.anchor_projection_dim),
        )
        symmetric_dim = 3 * config.anchor_projection_dim
        self.pair_head = nn.Sequential(
            nn.Linear(symmetric_dim + P0_PAIR_DESCRIPTOR_DIM, config.pair_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.pair_hidden_dim),
            nn.Linear(config.pair_hidden_dim, 1),
        )
        self.register_buffer(
            "descriptor_mean", torch.zeros(P0_PAIR_DESCRIPTOR_DIM, dtype=torch.float32)
        )
        self.register_buffer(
            "descriptor_scale", torch.ones(P0_PAIR_DESCRIPTOR_DIM, dtype=torch.float32)
        )
        self.register_buffer(
            "normalization_fitted", torch.tensor(False, dtype=torch.bool)
        )
        self.register_buffer(
            "normalization_sample_count", torch.tensor(0, dtype=torch.long)
        )
        if config == EvidenceCarrierP0PairScorerConfig() and (
            self.parameter_count() != P0_PAIR_SCORER_PARAMETER_COUNT
        ):
            raise RuntimeError("Frozen-P0 pair scorer parameter contract changed")

    def _descriptors(
        self,
        pair_features: torch.Tensor,
        consistency_features: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        base = transform_evidence_carrier_pair_features(
            pair_features, pair_mask, self.pair_transform_config
        )
        consistency = transform_p0_pair_consistency_features(
            consistency_features, pair_mask, self.config
        )
        return torch.cat((base, consistency), dim=-1)

    def fit_feature_normalization(
        self,
        pair_features: torch.Tensor,
        consistency_features: torch.Tensor,
        pair_mask: torch.Tensor,
        *,
        source_partition: str,
    ) -> dict[str, Any]:
        if source_partition != PAIR_SCORER_NORMALIZATION_SOURCE:
            raise ValueError("Frozen-P0 pair normalization may only use train-fit")
        if bool(self.normalization_fitted):
            raise RuntimeError("Frozen-P0 pair normalization is already fitted")
        descriptors = self._descriptors(pair_features, consistency_features, pair_mask)
        selected = descriptors[pair_mask]
        if len(selected) < 2:
            raise ValueError("Frozen-P0 pair normalization needs at least two rows")
        mean = selected.float().mean(dim=0)
        scale = selected.float().std(dim=0, unbiased=False)
        scale = torch.where(
            scale >= self.config.minimum_normalization_scale,
            scale,
            torch.ones_like(scale),
        )
        with torch.no_grad():
            self.descriptor_mean.copy_(mean.to(self.descriptor_mean))
            self.descriptor_scale.copy_(scale.to(self.descriptor_scale))
            self.normalization_sample_count.fill_(len(selected))
            self.normalization_fitted.fill_(True)
        return {
            "source_partition": source_partition,
            "sample_count": len(selected),
            "descriptor_mean": self.descriptor_mean.detach().cpu().tolist(),
            "descriptor_scale": self.descriptor_scale.detach().cpu().tolist(),
        }

    def forward(
        self,
        anchor_features: torch.Tensor,
        pair_indices: torch.Tensor,
        pair_features: torch.Tensor,
        consistency_features: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if not bool(self.normalization_fitted):
            raise RuntimeError("Frozen-P0 pair normalization has not been fitted")
        if anchor_features.ndim != 3 or anchor_features.size(-1) != self.config.anchor_feature_dim:
            raise ValueError("Frozen P0 anchor features have an invalid shape")
        if not anchor_features.is_floating_point():
            raise ValueError("Frozen P0 anchor features must be floating point")
        batch, anchors = anchor_features.shape[:2]
        if pair_indices.ndim != 3 or pair_indices.shape[0] != batch or pair_indices.size(-1) != 2:
            raise ValueError("Frozen-P0 pair indices have an invalid shape")
        if pair_indices.dtype != torch.long:
            raise ValueError("Frozen-P0 pair indices must use torch.long")
        if pair_mask.shape != pair_indices.shape[:2] or pair_mask.dtype != torch.bool:
            raise ValueError("Frozen-P0 pair mask shape or dtype is invalid")
        if pair_features.shape[:2] != pair_mask.shape or consistency_features.shape[:2] != pair_mask.shape:
            raise ValueError("Frozen-P0 pair feature batches differ")
        if bool(pair_mask.any()):
            active = pair_indices[pair_mask]
            if (
                anchors < 1
                or bool((active < 0).any())
                or bool((active >= anchors).any())
                or bool((active[:, 0] == active[:, 1]).any())
            ):
                raise ValueError("Frozen-P0 pair indices reference invalid anchors")

        projected = self.anchor_projector(anchor_features)
        first, second = _gather_pair_features(projected, pair_indices)
        symmetric = torch.cat(
            (first + second, (first - second).abs(), first * second), dim=-1
        )
        descriptors = self._descriptors(pair_features, consistency_features, pair_mask)
        mean = self.descriptor_mean.to(descriptors)
        scale = self.descriptor_scale.to(descriptors)
        normalized = (descriptors - mean) / scale
        normalized = torch.where(
            pair_mask[..., None], normalized, torch.zeros_like(normalized)
        )
        combined = torch.cat((symmetric, normalized), dim=-1)
        gains = self.pair_head(combined).squeeze(-1)
        gains = torch.where(pair_mask, gains, torch.zeros_like(gains))
        return {
            "pair_gains": gains,
            "projected_anchor_features": projected,
            "normalized_pair_descriptors": normalized,
            "pair_mask": pair_mask,
        }

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def config_dict(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "config": {
                name: getattr(self.config, name)
                for name in self.config.__dataclass_fields__
            },
            "pair_transform": {
                name: getattr(self.pair_transform_config, name)
                for name in self.pair_transform_config.__dataclass_fields__
            },
            "consistency_feature_names": list(P0_PAIR_CONSISTENCY_FEATURE_NAMES),
            "parameter_count": self.parameter_count(),
        }


@torch.inference_mode()
def extract_frozen_p0_anchor_evidence(
    p0_model: EvidenceCarrierExtentPredictor,
    pixel_values: torch.Tensor,
    anchor_segments: torch.Tensor,
    anchor_mask: torch.Tensor,
    *,
    coordinate_mode: Literal["normalized", "pixels", "pixel"] = "normalized",
) -> dict[str, torch.Tensor]:
    """Extract detached P0 embeddings and decoded lines from a frozen model."""

    if any(parameter.requires_grad for parameter in p0_model.parameters()):
        raise ValueError("P0 parameters must be frozen before pair feature extraction")
    p0_model.eval()
    output = p0_model(
        pixel_values,
        anchor_segments,
        anchor_mask,
        coordinate_mode=coordinate_mode,
    )
    carrier_features = output.get("carrier_features")
    if not isinstance(carrier_features, torch.Tensor):
        raise ValueError("Frozen P0 output lacks carrier_features")
    decoded = decode_evidence_carrier_extent(output, keep_threshold=0.0)
    return {
        "anchor_features": carrier_features.detach(),
        "decoded_segments_pixels": decoded["refined_segments_pixels"].detach(),
        "anchor_mask": anchor_mask,
    }


__all__ = [
    "P0_PAIR_CONSISTENCY_FEATURE_DIM",
    "P0_PAIR_CONSISTENCY_FEATURE_NAMES",
    "P0_PAIR_DESCRIPTOR_DIM",
    "P0_PAIR_SCORER_PARAMETER_COUNT",
    "EvidenceCarrierP0PairScorer",
    "EvidenceCarrierP0PairScorerConfig",
    "extract_frozen_p0_anchor_evidence",
    "p0_decoded_pair_consistency_features",
    "transform_p0_pair_consistency_features",
]
