"""Pairwise visual counterfactual over a frozen atom candidate prior."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


IMAGE_SIZE = 768.0
CONTEXT_SIZE = 96
PAIRWISE_ARMS = ("patch-only", "patch-band")
PAIR_RESCUE = 1
PAIR_HARMFUL = 0
PAIR_EQUIVALENT = 2


@dataclass(frozen=True, slots=True)
class PairwiseStructuralCounterfactualConfig:
    arm: str = "patch-band"
    patch_size: int = 25
    band_width: int = 17
    band_samples: int = 48
    residual_cap: float = 2.0

    def __post_init__(self) -> None:
        if self.arm not in PAIRWISE_ARMS:
            raise ValueError(f"unknown pairwise arm: {self.arm}")
        if self.patch_size < 9 or self.patch_size % 2 == 0:
            raise ValueError("patch_size must be an odd integer of at least 9")
        if min(self.band_width, self.band_samples) < 3:
            raise ValueError("band dimensions must be at least 3")
        if self.residual_cap <= 0:
            raise ValueError("residual_cap must be positive")


def _validate_inputs(
    image_crops: torch.Tensor,
    score_crops: torch.Tensor,
    source_offsets: torch.Tensor,
    prior_offsets: torch.Tensor,
    alternative_offsets: torch.Tensor,
    pair_mask: torch.Tensor,
) -> None:
    if image_crops.ndim != 4 or image_crops.shape[1:] != (
        1,
        CONTEXT_SIZE,
        CONTEXT_SIZE,
    ):
        raise ValueError("image_crops must have shape [B,1,96,96]")
    if score_crops.shape != image_crops.shape:
        raise ValueError("score_crops must match image_crops")
    batch = len(image_crops)
    for name, value in (
        ("source_offsets", source_offsets),
        ("prior_offsets", prior_offsets),
        ("alternative_offsets", alternative_offsets),
    ):
        if value.shape != (batch, 2) or not value.is_floating_point():
            raise ValueError(f"{name} must have floating shape [B,2]")
    if pair_mask.shape != (batch,) or pair_mask.dtype != torch.bool:
        raise ValueError("pair_mask must have boolean shape [B]")
    devices = {
        image_crops.device,
        score_crops.device,
        source_offsets.device,
        prior_offsets.device,
        alternative_offsets.device,
        pair_mask.device,
    }
    if len(devices) != 1:
        raise ValueError("pairwise inputs must share one device")
    active = pair_mask
    if bool(active.any()):
        tensors = (
            image_crops[active],
            score_crops[active],
            source_offsets[active],
            prior_offsets[active],
            alternative_offsets[active],
        )
        if not all(bool(torch.isfinite(value).all()) for value in tensors):
            raise ValueError("active pairwise inputs contain non-finite values")


def _to_grid(points: torch.Tensor) -> torch.Tensor:
    """Convert full-image normalized offsets around a crop center to grid coords."""

    return points.float() * (IMAGE_SIZE / (CONTEXT_SIZE * 0.5))


def sample_centered_patches(
    values: torch.Tensor,
    offsets: torch.Tensor,
    patch_size: int,
) -> torch.Tensor:
    if values.ndim != 4 or values.size(1) != 2:
        raise ValueError("patch source must have shape [B,2,H,W]")
    if offsets.shape != (len(values), 2):
        raise ValueError("patch offsets must have shape [B,2]")
    axis = torch.arange(
        patch_size, device=values.device, dtype=torch.float32
    ) - (patch_size - 1) * 0.5
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    local = torch.stack((x, y), dim=-1) / (CONTEXT_SIZE * 0.5)
    grid = _to_grid(offsets)[:, None, None, :] + local[None]
    return F.grid_sample(
        values.float(),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )


def sample_relation_bands(
    values: torch.Tensor,
    source_offsets: torch.Tensor,
    candidate_offsets: torch.Tensor,
    *,
    width: int,
    samples: int,
) -> torch.Tensor:
    """Sample a rotation-normalized 2-D band from source to candidate."""

    if values.ndim != 4 or values.size(1) != 2:
        raise ValueError("band source must have shape [B,2,H,W]")
    if (
        source_offsets.shape != (len(values), 2)
        or candidate_offsets.shape != (len(values), 2)
    ):
        raise ValueError("band offsets must have shape [B,2]")
    begin = _to_grid(source_offsets)
    end = _to_grid(candidate_offsets)
    delta = end - begin
    unit = delta / torch.linalg.vector_norm(delta, dim=-1, keepdim=True).clamp_min(
        1e-6
    )
    perpendicular = torch.stack((-unit[:, 1], unit[:, 0]), dim=-1)
    along = torch.linspace(0.0, 1.0, samples, device=values.device)
    across = torch.linspace(
        -(width - 1) * 0.5,
        (width - 1) * 0.5,
        width,
        device=values.device,
    ) / (CONTEXT_SIZE * 0.5)
    base = begin[:, None, :] + delta[:, None, :] * along[None, :, None]
    grid = base[:, None, :, :] + perpendicular[:, None, None, :] * across[
        None, :, None, None
    ]
    return F.grid_sample(
        values.float(),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )


class _RasterEncoder(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(2, 12, 3, padding=1),
            nn.GroupNorm(3, 12),
            nn.GELU(),
            nn.Conv2d(12, 24, 3, stride=2, padding=1),
            nn.GroupNorm(6, 24),
            nn.GELU(),
            nn.Conv2d(24, 32, 3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, output_dim),
            nn.GELU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values.float())


class PairwiseStructuralCounterfactual(nn.Module):
    """Return an exactly antisymmetric residual utility delta."""

    model_type = "pairwise-structural-counterfactual-v0.1"

    def __init__(
        self,
        config: PairwiseStructuralCounterfactualConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or PairwiseStructuralCounterfactualConfig()
        self.patch_encoder = _RasterEncoder(40)
        self.band_encoder = (
            _RasterEncoder(24) if self.config.arm == "patch-band" else None
        )
        candidate_dim = 40 + (24 if self.band_encoder is not None else 0) + 4
        self.utility = nn.Sequential(
            nn.Linear(candidate_dim * 3, 96),
            nn.GELU(),
            nn.Linear(96, 48),
            nn.GELU(),
            nn.Linear(48, 1),
        )
        output = self.utility[-1]
        if not isinstance(output, nn.Linear):
            raise RuntimeError("pairwise output layer contract changed")
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        if self.parameter_count >= 300_000:
            raise RuntimeError("pairwise model exceeds the 300k parameter budget")

    @property
    def parameter_count(self) -> int:
        return sum(value.numel() for value in self.parameters())

    def _candidate_features(
        self,
        values: torch.Tensor,
        source: torch.Tensor,
        candidate: torch.Tensor,
    ) -> torch.Tensor:
        patch = sample_centered_patches(
            values, candidate, self.config.patch_size
        )
        parts = [self.patch_encoder(patch)]
        if self.band_encoder is not None:
            band = sample_relation_bands(
                values,
                source,
                candidate,
                width=self.config.band_width,
                samples=self.config.band_samples,
            )
            parts.append(self.band_encoder(band))
        geometry = torch.cat(
            (candidate * IMAGE_SIZE, (candidate - source) * IMAGE_SIZE), dim=-1
        ) / CONTEXT_SIZE
        parts.append(geometry.float())
        return torch.cat(parts, dim=-1)

    def _ordered_utility(
        self, candidate: torch.Tensor, other: torch.Tensor
    ) -> torch.Tensor:
        interaction = torch.cat(
            (candidate, other, candidate - other), dim=-1
        )
        return self.utility(interaction).squeeze(-1)

    def forward(
        self,
        image_crops: torch.Tensor,
        score_crops: torch.Tensor,
        source_offsets: torch.Tensor,
        prior_offsets: torch.Tensor,
        alternative_offsets: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        _validate_inputs(
            image_crops,
            score_crops,
            source_offsets,
            prior_offsets,
            alternative_offsets,
            pair_mask,
        )
        values = torch.cat((image_crops.float(), score_crops.float()), dim=1)
        prior = self._candidate_features(values, source_offsets, prior_offsets)
        alternative = self._candidate_features(
            values, source_offsets, alternative_offsets
        )
        forward = self._ordered_utility(alternative, prior)
        reverse = self._ordered_utility(prior, alternative)
        delta = self.config.residual_cap * torch.tanh((forward - reverse) * 0.5)
        return torch.where(pair_mask, delta, torch.zeros_like(delta))

    def config_dict(self) -> dict[str, object]:
        return {
            "model_type": self.model_type,
            "arm": self.config.arm,
            "patch_size": self.config.patch_size,
            "band_width": self.config.band_width,
            "band_samples": self.config.band_samples,
            "residual_cap": self.config.residual_cap,
            "parameter_count": self.parameter_count,
            "exact_antisymmetry": True,
        }


def swap_pair_inputs(
    prior_offsets: torch.Tensor,
    alternative_offsets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if prior_offsets.shape != alternative_offsets.shape:
        raise ValueError("pair offsets must have matching shapes")
    return alternative_offsets, prior_offsets


def pairwise_counterfactual_loss(
    utility_delta: torch.Tensor,
    prior_margin: torch.Tensor,
    labels: torch.Tensor,
    pair_mask: torch.Tensor,
    *,
    hard_negative_top_k: int = 8,
    equivalent_weight: float = 0.25,
    residual_l2_weight: float = 0.01,
) -> dict[str, torch.Tensor]:
    if utility_delta.ndim != 1 or prior_margin.shape != utility_delta.shape:
        raise ValueError("pairwise deltas and margins must have shape [B]")
    if labels.shape != utility_delta.shape or labels.dtype != torch.long:
        raise ValueError("pairwise labels must have long shape [B]")
    if pair_mask.shape != utility_delta.shape or pair_mask.dtype != torch.bool:
        raise ValueError("pair_mask must have boolean shape [B]")
    if hard_negative_top_k < 1:
        raise ValueError("hard_negative_top_k must be positive")
    corrected = prior_margin.float() + utility_delta.float()
    directed = pair_mask & ((labels == PAIR_RESCUE) | (labels == PAIR_HARMFUL))
    target = torch.where(labels == PAIR_RESCUE, 1.0, -1.0)
    ranking = F.softplus(-target[directed] * corrected[directed]).mean()
    harmful = pair_mask & (labels == PAIR_HARMFUL)
    hardest = corrected[harmful].topk(
        min(hard_negative_top_k, int(harmful.sum()))
    ).values
    hard_negative = F.softplus(hardest).mean()
    equivalent = pair_mask & (labels == PAIR_EQUIVALENT)
    equivalent_loss = (
        utility_delta[equivalent].square().mean()
        if bool(equivalent.any())
        else utility_delta.sum() * 0.0
    )
    residual_l2 = utility_delta[pair_mask].square().mean()
    loss = (
        ranking
        + 0.5 * hard_negative
        + equivalent_weight * equivalent_loss
        + residual_l2_weight * residual_l2
    )
    return {
        "loss": loss,
        "ranking": ranking,
        "hard_negative": hard_negative,
        "equivalent": equivalent_loss,
        "residual_l2": residual_l2,
        "corrected_margin": corrected,
    }


__all__ = [
    "PAIRWISE_ARMS",
    "PAIR_EQUIVALENT",
    "PAIR_HARMFUL",
    "PAIR_RESCUE",
    "PairwiseStructuralCounterfactual",
    "PairwiseStructuralCounterfactualConfig",
    "pairwise_counterfactual_loss",
    "sample_centered_patches",
    "sample_relation_bands",
    "swap_pair_inputs",
]
