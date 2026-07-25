"""Tiny target-free per-image threshold predictor for frozen pair scores.

The predictor treats candidate pairs as an unordered set.  Each token contains
the frozen scorer's normalized 20-dimensional descriptor followed by its raw
gain.  A shared DeepSets encoder is pooled with mean and max, then combined
with four target-free count features to predict an offset from the frozen
global threshold.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from chemvector.evidence_carrier_conditional_pair_data import PAIR_FEATURE_NAMES
from chemvector.evidence_carrier_p0_pair_scorer import (
    P0_PAIR_CONSISTENCY_FEATURE_NAMES,
    P0_PAIR_DESCRIPTOR_DIM,
)


SAMPLE_THRESHOLD_BASELINE_TAU = 0.7209501005709171
SAMPLE_THRESHOLD_PAIR_DESCRIPTOR_NAMES = (
    *(f"normalized_{name}" for name in PAIR_FEATURE_NAMES),
    *(f"normalized_{name}" for name in P0_PAIR_CONSISTENCY_FEATURE_NAMES),
)
SAMPLE_THRESHOLD_PAIR_TOKEN_NAMES = (
    *SAMPLE_THRESHOLD_PAIR_DESCRIPTOR_NAMES,
    "raw_pair_gain",
)
SAMPLE_THRESHOLD_GLOBAL_COUNT_FEATURE_NAMES = (
    "log1p_anchor_count",
    "log1p_candidate_pair_count",
    "log1p_gain_above_tau0_count",
    "log1p_gain_above_tau0_incident_unique_anchor_count",
)
SAMPLE_THRESHOLD_PAIR_TOKEN_DIM = P0_PAIR_DESCRIPTOR_DIM + 1
SAMPLE_THRESHOLD_GLOBAL_COUNT_FEATURE_DIM = len(
    SAMPLE_THRESHOLD_GLOBAL_COUNT_FEATURE_NAMES
)
SAMPLE_THRESHOLD_PREDICTOR_PARAMETER_COUNT = 1_841
SAMPLE_THRESHOLD_INTERVAL_INWARD_MARGIN_FRACTION = 0.05


def sample_threshold_global_count_features(
    anchor_count: int,
    pair_indices: torch.Tensor,
    raw_pair_gains: torch.Tensor,
    *,
    tau0: float = SAMPLE_THRESHOLD_BASELINE_TAU,
) -> torch.Tensor:
    """Build the fixed four target-free log-count features.

    ``gain > tau0`` is intentionally strict.  This helper is also used by the
    manifest loader to verify that serialized count features did not drift
    from the pre-registered contract.
    """

    if isinstance(anchor_count, bool) or not isinstance(anchor_count, int):
        raise ValueError("anchor_count must be an integer")
    if anchor_count < 0:
        raise ValueError("anchor_count cannot be negative")
    if pair_indices.ndim != 2 or pair_indices.size(-1) != 2:
        raise ValueError("pair_indices must have shape [pairs,2]")
    if pair_indices.dtype != torch.long:
        raise ValueError("pair_indices must use torch.long")
    if raw_pair_gains.ndim != 1 or len(raw_pair_gains) != len(pair_indices):
        raise ValueError("raw_pair_gains must have one value per pair")
    if not raw_pair_gains.is_floating_point():
        raise ValueError("raw_pair_gains must be floating point")
    if not bool(torch.isfinite(raw_pair_gains).all()):
        raise ValueError("raw_pair_gains contain non-finite values")
    if (
        isinstance(tau0, bool)
        or not isinstance(tau0, (int, float))
        or not math.isfinite(float(tau0))
    ):
        raise ValueError("tau0 must be finite")
    if len(pair_indices) and (
        anchor_count < 1
        or bool((pair_indices < 0).any())
        or bool((pair_indices >= anchor_count).any())
        or bool((pair_indices[:, 0] == pair_indices[:, 1]).any())
    ):
        raise ValueError("pair_indices reference invalid anchors")

    above = raw_pair_gains > float(tau0)
    if bool(above.any()):
        incident_count = int(torch.unique(pair_indices[above].reshape(-1)).numel())
    else:
        incident_count = 0
    raw_counts = raw_pair_gains.new_tensor(
        (
            anchor_count,
            len(pair_indices),
            int(above.sum()),
            incident_count,
        )
    )
    return torch.log1p(raw_counts)


def _validate_predictor_inputs(
    pair_tokens: torch.Tensor,
    pair_mask: torch.Tensor,
    global_count_features: torch.Tensor,
) -> tuple[int, int]:
    if (
        pair_tokens.ndim != 3
        or pair_tokens.size(-1) != SAMPLE_THRESHOLD_PAIR_TOKEN_DIM
        or not pair_tokens.is_floating_point()
    ):
        raise ValueError("pair_tokens must have shape [batch,pairs,21]")
    batch, pairs = pair_tokens.shape[:2]
    if pair_mask.shape != (batch, pairs) or pair_mask.dtype != torch.bool:
        raise ValueError("pair_mask shape or dtype is invalid")
    if (
        global_count_features.shape
        != (batch, SAMPLE_THRESHOLD_GLOBAL_COUNT_FEATURE_DIM)
        or not global_count_features.is_floating_point()
    ):
        raise ValueError("global_count_features must have shape [batch,4]")
    if len(
        {pair_tokens.device, pair_mask.device, global_count_features.device}
    ) != 1:
        raise ValueError("predictor inputs must share one device")
    if bool(pair_mask.any()) and not bool(
        torch.isfinite(pair_tokens[pair_mask]).all()
    ):
        raise ValueError("active pair tokens contain non-finite values")
    if not bool(torch.isfinite(global_count_features).all()):
        raise ValueError("global count features contain non-finite values")
    if bool((global_count_features < 0.0).any()):
        raise ValueError("global log-count features cannot be negative")
    return batch, pairs


class EvidenceCarrierSampleThresholdPredictor(nn.Module):
    """Predict one scalar pair-selection threshold per image."""

    model_type = "evidence-carrier-sample-threshold-predictor-v0.1"

    def __init__(self, *, tau0: float = SAMPLE_THRESHOLD_BASELINE_TAU) -> None:
        super().__init__()
        if (
            isinstance(tau0, bool)
            or not isinstance(tau0, (int, float))
            or not math.isfinite(float(tau0))
        ):
            raise ValueError("tau0 must be finite")
        self.pair_encoder = nn.Sequential(
            nn.Linear(SAMPLE_THRESHOLD_PAIR_TOKEN_DIM, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
        )
        self.threshold_head = nn.Sequential(
            nn.Linear(16 * 2 + SAMPLE_THRESHOLD_GLOBAL_COUNT_FEATURE_DIM, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        self.register_buffer("tau0", torch.tensor(float(tau0), dtype=torch.float64))
        if self.parameter_count() != SAMPLE_THRESHOLD_PREDICTOR_PARAMETER_COUNT:
            raise RuntimeError("Sample-threshold predictor parameter contract changed")

    def forward(
        self,
        pair_tokens: torch.Tensor,
        pair_mask: torch.Tensor,
        global_count_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, _ = _validate_predictor_inputs(
            pair_tokens,
            pair_mask,
            global_count_features,
        )
        encoded = self.pair_encoder(pair_tokens)
        active = pair_mask[..., None]
        encoded_sum = torch.where(
            active, encoded, torch.zeros_like(encoded)
        ).sum(dim=1)
        counts = pair_mask.sum(dim=1)
        pooled_mean = encoded_sum / counts.clamp_min(1).to(
            encoded.dtype
        ).unsqueeze(-1)

        empty_pool = torch.zeros(
            (batch, encoded.size(-1)),
            device=encoded.device,
            dtype=encoded.dtype,
        )
        if encoded.size(1):
            negative_infinity = torch.full_like(encoded, -torch.inf)
            pooled_max = torch.where(active, encoded, negative_infinity).amax(dim=1)
            pooled_max = torch.where(
                (counts > 0).unsqueeze(-1), pooled_max, empty_pool
            )
        else:
            pooled_max = empty_pool
        pooled = torch.cat((pooled_mean, pooled_max, global_count_features), dim=-1)
        delta = self.threshold_head(pooled).squeeze(-1)
        tau = self.tau0.to(device=delta.device) + delta.to(dtype=self.tau0.dtype)
        return {
            "tau": tau,
            "delta": delta,
            "pooled_pair_mean": pooled_mean,
            "pooled_pair_max": pooled_max,
            "pair_mask": pair_mask,
        }

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def config_dict(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "tau0": float(self.tau0),
            "pair_token_names": list(SAMPLE_THRESHOLD_PAIR_TOKEN_NAMES),
            "pair_token_dim": SAMPLE_THRESHOLD_PAIR_TOKEN_DIM,
            "global_count_feature_names": list(
                SAMPLE_THRESHOLD_GLOBAL_COUNT_FEATURE_NAMES
            ),
            "global_count_feature_dim": SAMPLE_THRESHOLD_GLOBAL_COUNT_FEATURE_DIM,
            "pair_encoder": [21, 32, 16],
            "pooling": ["mean", "max"],
            "threshold_head": [36, 16, 1],
            "parameter_count": self.parameter_count(),
        }


def fp32_open_interval_contract(
    interval_lower: torch.Tensor,
    interval_upper: torch.Tensor,
    interval_normalization_scale: torch.Tensor,
    *,
    inward_margin_fraction: float = (
        SAMPLE_THRESHOLD_INTERVAL_INWARD_MARGIN_FRACTION
    ),
) -> torch.Tensor:
    """Return which FP32 intervals satisfy the predictor loss contract."""

    if (
        interval_lower.dtype != torch.float32
        or interval_upper.dtype != torch.float32
        or interval_normalization_scale.dtype != torch.float32
        or interval_lower.shape != interval_upper.shape
        or interval_lower.shape != interval_normalization_scale.shape
    ):
        raise ValueError("interval contract tensors must share one FP32 shape")
    if (
        isinstance(inward_margin_fraction, bool)
        or not isinstance(inward_margin_fraction, (int, float))
        or not math.isfinite(float(inward_margin_fraction))
        or not 0.0 <= float(inward_margin_fraction) < 0.5
    ):
        raise ValueError("inward_margin_fraction must lie in [0,0.5)")

    lower = interval_lower
    upper = interval_upper
    scale = interval_normalization_scale
    finite_both = torch.isfinite(lower) & torch.isfinite(upper)
    width = upper - lower
    machine_epsilon = torch.finfo(torch.float32).eps
    roundoff_tolerance = (
        2.0
        * machine_epsilon
        * (lower.abs() + upper.abs() + scale.abs()).clamp_min(1.0)
    )
    requested_tolerance = 1e-7 + 1e-6 * width.abs()
    scale_tolerance = torch.maximum(roundoff_tolerance, requested_tolerance)
    scale_matches = (~finite_both) | ((scale - width).abs() <= scale_tolerance)
    margin = scale * float(inward_margin_fraction)
    inner_lower = torch.where(torch.isfinite(lower), lower + margin, lower)
    inner_upper = torch.where(torch.isfinite(upper), upper - margin, upper)
    return (
        ~torch.isnan(lower)
        & ~torch.isnan(upper)
        & ~torch.isposinf(lower)
        & ~torch.isneginf(upper)
        & (lower < upper)
        & torch.isfinite(scale)
        & (scale > 0.0)
        & scale_matches
        & (inner_lower < inner_upper)
    )


def robust_open_interval_loss(
    predicted_tau: torch.Tensor,
    interval_lower: torch.Tensor,
    interval_upper: torch.Tensor,
    interval_mask: torch.Tensor,
    interval_normalization_scale: torch.Tensor,
    *,
    inward_margin_fraction: float = (
        SAMPLE_THRESHOLD_INTERVAL_INWARD_MARGIN_FRACTION
    ),
) -> dict[str, torch.Tensor]:
    """Return the minimum normalized hinge over valid oracle intervals.

    Finite intervals use their exact width as the normalization scale.  A
    one-sided interval retains its infinite side and moves the finite boundary
    inward by five percent of a manifest-provided finite scale.  This keeps the
    loss well-defined for the extreme states of a parametric matching hull.
    """

    if predicted_tau.ndim != 1 or not predicted_tau.is_floating_point():
        raise ValueError("predicted_tau must be a floating-point batch vector")
    expected = interval_lower.shape
    if (
        interval_lower.ndim != 2
        or interval_upper.shape != expected
        or interval_mask.shape != expected
        or interval_normalization_scale.shape != expected
    ):
        raise ValueError("interval tensors must share shape [batch,intervals]")
    if expected[0] != len(predicted_tau):
        raise ValueError("interval batch size differs from predicted_tau")
    if not interval_lower.is_floating_point() or not interval_upper.is_floating_point():
        raise ValueError("interval bounds must be floating point")
    if not interval_normalization_scale.is_floating_point():
        raise ValueError("interval normalization scale must be floating point")
    if interval_mask.dtype != torch.bool:
        raise ValueError("interval_mask must be boolean")
    devices = {
        predicted_tau.device,
        interval_lower.device,
        interval_upper.device,
        interval_mask.device,
        interval_normalization_scale.device,
    }
    if len(devices) != 1:
        raise ValueError("interval loss tensors must share one device")
    if (
        isinstance(inward_margin_fraction, bool)
        or not isinstance(inward_margin_fraction, (int, float))
        or not math.isfinite(float(inward_margin_fraction))
        or not 0.0 <= float(inward_margin_fraction) < 0.5
    ):
        raise ValueError("inward_margin_fraction must lie in [0,0.5)")
    if not bool(interval_mask.any(dim=1).all()):
        raise ValueError("each loss-eligible sample needs at least one interval")

    active_lower = interval_lower[interval_mask]
    active_upper = interval_upper[interval_mask]
    active_scale = interval_normalization_scale[interval_mask]
    if bool(torch.isnan(active_lower).any()) or bool(torch.isnan(active_upper).any()):
        raise ValueError("active interval bounds cannot be NaN")
    if bool(torch.isposinf(active_lower).any()) or bool(
        torch.isneginf(active_upper).any()
    ):
        raise ValueError("active interval infinities point in the wrong direction")
    if not bool((active_lower < active_upper).all()):
        raise ValueError("active oracle intervals must be non-degenerate and open")
    if not bool(torch.isfinite(active_scale).all()) or bool(
        (active_scale <= 0.0).any()
    ):
        raise ValueError("active interval normalization scales must be finite and positive")

    contract = fp32_open_interval_contract(
        interval_lower,
        interval_upper,
        interval_normalization_scale,
        inward_margin_fraction=inward_margin_fraction,
    )
    if bool(interval_mask.logical_and(~contract).any()):
        raise ValueError("active interval violates the FP32 loss contract")

    margin = interval_normalization_scale * float(inward_margin_fraction)
    inner_lower = torch.where(
        torch.isfinite(interval_lower), interval_lower + margin, interval_lower
    )
    inner_upper = torch.where(
        torch.isfinite(interval_upper), interval_upper - margin, interval_upper
    )
    if bool(interval_mask.logical_and(inner_lower >= inner_upper).any()):
        raise ValueError("inward margin collapses an active oracle interval")

    tau = predicted_tau[:, None]
    lower_hinge = torch.where(
        torch.isfinite(inner_lower),
        torch.relu(inner_lower - tau),
        torch.zeros_like(inner_lower),
    )
    upper_hinge = torch.where(
        torch.isfinite(inner_upper),
        torch.relu(tau - inner_upper),
        torch.zeros_like(inner_upper),
    )
    normalized = (lower_hinge + upper_hinge) / interval_normalization_scale
    normalized = torch.where(
        interval_mask,
        normalized,
        torch.full_like(normalized, torch.inf),
    )
    per_sample_loss, selected_interval_index = normalized.min(dim=1)
    return {
        "loss": per_sample_loss.mean(),
        "per_sample_loss": per_sample_loss,
        "selected_interval_index": selected_interval_index,
        "per_interval_loss": normalized,
        "inner_lower": inner_lower,
        "inner_upper": inner_upper,
    }


__all__ = [
    "SAMPLE_THRESHOLD_BASELINE_TAU",
    "SAMPLE_THRESHOLD_GLOBAL_COUNT_FEATURE_DIM",
    "SAMPLE_THRESHOLD_GLOBAL_COUNT_FEATURE_NAMES",
    "SAMPLE_THRESHOLD_INTERVAL_INWARD_MARGIN_FRACTION",
    "SAMPLE_THRESHOLD_PAIR_DESCRIPTOR_NAMES",
    "SAMPLE_THRESHOLD_PAIR_TOKEN_DIM",
    "SAMPLE_THRESHOLD_PAIR_TOKEN_NAMES",
    "SAMPLE_THRESHOLD_PREDICTOR_PARAMETER_COUNT",
    "EvidenceCarrierSampleThresholdPredictor",
    "fp32_open_interval_contract",
    "robust_open_interval_loss",
    "sample_threshold_global_count_features",
]
