"""Pairwise visual counterfactual over a frozen atom candidate prior."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


IMAGE_SIZE = 768.0
CONTEXT_SIZE = 96
LOCAL_PEAK_TOKEN_DIM = 9
GLOBAL_SET_TOKEN_DIM = 8
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
    explicit_score_evidence: bool = False
    local_peak_set_evidence: bool = False
    local_peak_count: int = 16
    local_peak_kernel: int = 5
    global_candidate_set_evidence: bool = False
    global_candidate_count: int = 129

    def __post_init__(self) -> None:
        if self.arm not in PAIRWISE_ARMS:
            raise ValueError(f"unknown pairwise arm: {self.arm}")
        if self.patch_size < 9 or self.patch_size % 2 == 0:
            raise ValueError("patch_size must be an odd integer of at least 9")
        if min(self.band_width, self.band_samples) < 3:
            raise ValueError("band dimensions must be at least 3")
        if self.residual_cap <= 0:
            raise ValueError("residual_cap must be positive")
        if self.local_peak_count < 1 or self.local_peak_count > CONTEXT_SIZE**2:
            raise ValueError("local_peak_count must fit inside the context crop")
        if self.local_peak_kernel < 3 or self.local_peak_kernel % 2 == 0:
            raise ValueError("local_peak_kernel must be an odd integer of at least 3")
        if self.global_candidate_count < 2:
            raise ValueError("global_candidate_count must include candidates and source")


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


def extract_local_peak_tokens(
    image_crops: torch.Tensor,
    score_crops: torch.Tensor,
    source_offsets: torch.Tensor,
    candidate_offsets: torch.Tensor,
    *,
    peak_count: int,
    pooling_kernel: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract a fixed-size set of regional center-field maxima."""

    if image_crops.ndim != 4 or image_crops.shape[1:] != (
        1,
        CONTEXT_SIZE,
        CONTEXT_SIZE,
    ):
        raise ValueError("image_crops must have shape [B,1,96,96]")
    if score_crops.shape != image_crops.shape:
        raise ValueError("score_crops must match image_crops")
    batch = len(image_crops)
    if source_offsets.shape != (batch, 2) or candidate_offsets.shape != (batch, 2):
        raise ValueError("peak offsets must have shape [B,2]")
    if peak_count < 1 or peak_count > CONTEXT_SIZE**2:
        raise ValueError("peak_count must fit inside the context crop")
    if pooling_kernel < 3 or pooling_kernel % 2 == 0:
        raise ValueError("pooling_kernel must be an odd integer of at least 3")

    score = score_crops[:, 0].float()
    # The deterministic sub-ULP ramp makes flat plateaus choose one stable pixel.
    flat_rank = torch.arange(
        CONTEXT_SIZE**2, device=score.device, dtype=torch.float32
    ).reshape(1, CONTEXT_SIZE, CONTEXT_SIZE)
    priority = score + flat_rank * torch.finfo(torch.float32).eps
    maxima = priority == F.max_pool2d(
        priority[:, None],
        pooling_kernel,
        stride=1,
        padding=pooling_kernel // 2,
    )[:, 0]
    ranked = priority.masked_fill(~maxima, -torch.inf).flatten(1)
    _, flat_ids = ranked.topk(peak_count, dim=1, largest=True, sorted=True)
    flat_score = score.flatten(1)
    peak_score = flat_score.gather(1, flat_ids)
    peak_ink = image_crops[:, 0].float().flatten(1).gather(1, flat_ids)
    peak_mask = torch.isfinite(peak_score)

    peak_x = (flat_ids.remainder(CONTEXT_SIZE).float() + 0.5) - CONTEXT_SIZE * 0.5
    peak_y = (flat_ids.div(CONTEXT_SIZE, rounding_mode="floor").float() + 0.5) - CONTEXT_SIZE * 0.5
    peak_offsets = torch.stack((peak_x, peak_y), dim=-1) / IMAGE_SIZE
    relative_source = (
        (peak_offsets - source_offsets[:, None].float()) * IMAGE_SIZE / CONTEXT_SIZE
    )
    relative_candidate = (
        (peak_offsets - candidate_offsets[:, None].float()) * IMAGE_SIZE
        / CONTEXT_SIZE
    )
    distance_source = torch.linalg.vector_norm(relative_source, dim=-1)
    distance_candidate = torch.linalg.vector_norm(relative_candidate, dim=-1)
    rank = torch.arange(peak_count, device=score.device, dtype=torch.float32)
    rank = rank[None].expand(batch, -1) / max(peak_count - 1, 1)
    tokens = torch.cat(
        (
            relative_source,
            relative_candidate,
            torch.tanh(peak_score / 8.0)[..., None],
            peak_ink[..., None],
            distance_source[..., None],
            distance_candidate[..., None],
            rank[..., None],
        ),
        dim=-1,
    )
    return torch.where(peak_mask[..., None], tokens, torch.zeros_like(tokens)), peak_mask


def masked_deepset_pool(
    encoded_tokens: torch.Tensor,
    token_mask: torch.Tensor,
) -> torch.Tensor:
    """Permutation-invariant mean/max pooling over valid set members."""

    if encoded_tokens.ndim != 3:
        raise ValueError("encoded_tokens must have shape [B,K,D]")
    if token_mask.shape != encoded_tokens.shape[:2] or token_mask.dtype != torch.bool:
        raise ValueError("token_mask must have boolean shape [B,K]")
    valid = token_mask[..., None]
    count = valid.sum(dim=1).clamp_min(1)
    mean = (encoded_tokens * valid).sum(dim=1) / count
    floor = torch.finfo(encoded_tokens.dtype).min
    maximum = encoded_tokens.masked_fill(~valid, floor).amax(dim=1)
    any_valid = token_mask.any(dim=1, keepdim=True)
    maximum = torch.where(any_valid, maximum, torch.zeros_like(maximum))
    return torch.cat((mean, maximum), dim=-1)


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
        self.local_peak_encoder = (
            nn.Sequential(
                nn.Linear(LOCAL_PEAK_TOKEN_DIM, 32),
                nn.GELU(),
                nn.Linear(32, 32),
                nn.GELU(),
            )
            if self.config.local_peak_set_evidence
            else None
        )
        self.global_set_encoder = (
            nn.Sequential(
                nn.Linear(GLOBAL_SET_TOKEN_DIM, 32),
                nn.GELU(),
                nn.Linear(32, 32),
                nn.GELU(),
            )
            if self.config.global_candidate_set_evidence
            else None
        )
        candidate_dim = (
            40
            + (24 if self.band_encoder is not None else 0)
            + 4
            + (2 if self.config.explicit_score_evidence else 0)
            + (64 if self.local_peak_encoder is not None else 0)
            + (64 if self.global_set_encoder is not None else 0)
        )
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
        global_offsets: torch.Tensor | None = None,
        global_scores: torch.Tensor | None = None,
        global_source_mask: torch.Tensor | None = None,
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
        if self.config.explicit_score_evidence:
            score_field = values[:, 1:2].float()
            source_value = F.grid_sample(
                score_field,
                _to_grid(source)[:, None, None, :],
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            ).flatten()
            candidate_value = F.grid_sample(
                score_field,
                _to_grid(candidate)[:, None, None, :],
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            ).flatten()
            parts.append(
                torch.stack(
                    (candidate_value, candidate_value - source_value), dim=-1
                )
            )
        if self.local_peak_encoder is not None:
            tokens, token_mask = extract_local_peak_tokens(
                values[:, 0:1],
                values[:, 1:2],
                source,
                candidate,
                peak_count=self.config.local_peak_count,
                pooling_kernel=self.config.local_peak_kernel,
            )
            parts.append(masked_deepset_pool(self.local_peak_encoder(tokens), token_mask))
        if self.global_set_encoder is not None:
            if (
                global_offsets is None
                or global_scores is None
                or global_source_mask is None
            ):
                raise ValueError("global candidate set evidence is required")
            count = self.config.global_candidate_count
            batch = len(values)
            if global_offsets.shape != (batch, count, 2):
                raise ValueError("global_offsets have invalid shape")
            if global_scores.shape != (batch, count):
                raise ValueError("global_scores have invalid shape")
            if (
                global_source_mask.shape != (batch, count)
                or global_source_mask.dtype != torch.bool
            ):
                raise ValueError("global_source_mask has invalid shape or dtype")
            if bool((global_source_mask.sum(dim=1) != 1).any()):
                raise ValueError("global candidate set requires one explicit source token")
            if not bool(
                torch.isfinite(global_offsets).all()
                & torch.isfinite(global_scores).all()
            ):
                raise ValueError("global candidate set contains non-finite values")
            relative_source = (
                (global_offsets.float() - source[:, None].float())
                * IMAGE_SIZE
                / CONTEXT_SIZE
            )
            relative_candidate = (
                (global_offsets.float() - candidate[:, None].float())
                * IMAGE_SIZE
                / CONTEXT_SIZE
            )
            score_rank = (
                global_scores[:, :, None].float()
                < global_scores[:, None, :].float()
            ).sum(dim=-1).float() / max(count - 1, 1)
            global_tokens = torch.cat(
                (
                    relative_source,
                    relative_candidate,
                    global_scores.float()[..., None],
                    score_rank[..., None],
                    global_source_mask.float()[..., None],
                    torch.linalg.vector_norm(relative_candidate, dim=-1)[..., None],
                ),
                dim=-1,
            )
            valid = torch.ones_like(global_source_mask)
            parts.append(
                masked_deepset_pool(
                    self.global_set_encoder(global_tokens),
                    valid,
                )
            )
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
        global_offsets: torch.Tensor | None = None,
        global_scores: torch.Tensor | None = None,
        global_source_mask: torch.Tensor | None = None,
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
        prior = self._candidate_features(
            values,
            source_offsets,
            prior_offsets,
            global_offsets,
            global_scores,
            global_source_mask,
        )
        alternative = self._candidate_features(
            values,
            source_offsets,
            alternative_offsets,
            global_offsets,
            global_scores,
            global_source_mask,
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
            "explicit_score_evidence": self.config.explicit_score_evidence,
            "local_peak_set_evidence": self.config.local_peak_set_evidence,
            "local_peak_count": self.config.local_peak_count,
            "local_peak_kernel": self.config.local_peak_kernel,
            "global_candidate_set_evidence": self.config.global_candidate_set_evidence,
            "global_candidate_count": self.config.global_candidate_count,
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
    "extract_local_peak_tokens",
    "masked_deepset_pool",
    "sample_centered_patches",
    "sample_relation_bands",
    "swap_pair_inputs",
]
