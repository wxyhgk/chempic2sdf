"""Candidate-relative evidence refinement for frozen line carriers.

The model never predicts free coordinates.  Every input carrier defines a
canonical unoriented line.  That infinite line is clipped to the image, and an
oriented RGB ink strip is sampled over the complete clipped interval.  The
original proposal segment is encoded only by its relative support interval and
endpoint maps on that strip.  The heads then classify whether to keep the
carrier, a bounded normal offset, and two ordered extent fractions inside the
clipped interval.

There is deliberately no absolute-position embedding or image-wide query in
this module.  Translating an image pattern together with its carrier therefore
translates the reconstructed line while preserving the candidate-relative
evidence representation (away from crop changes).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch
from torch import nn
from torch.nn import functional as F


CarrierCoordinateMode = Literal["normalized", "pixels", "pixel"]


@dataclass(frozen=True, slots=True)
class EvidenceCarrierExtentConfig:
    """Geometry and head sizes for evidence-carrier extent refinement."""

    longitudinal_samples: int = 385
    band_samples: int = 17
    band_radius_pixels: float = 8.0
    normal_offset_radius_pixels: float = 4.0
    normal_offset_step_pixels: float = 0.5
    extent_bins: int = 385
    hidden_dim: int = 128
    dropout: float = 0.0
    minimum_carrier_length_pixels: float = 1e-3
    anchor_edge_softness_bins: float = 0.75
    anchor_endpoint_sigma_bins: float = 1.5

    def __post_init__(self) -> None:
        dimensions = (
            self.longitudinal_samples,
            self.band_samples,
            self.extent_bins,
            self.hidden_dim,
        )
        if min(dimensions) < 1:
            raise ValueError("Evidence Carrier dimensions must be positive")
        if self.longitudinal_samples < 2:
            raise ValueError("Evidence Carrier needs at least two line samples")
        if self.band_samples % 2 == 0:
            raise ValueError("Evidence Carrier band_samples must be odd")
        if self.extent_bins < 2:
            raise ValueError("Evidence Carrier needs at least two extent bins")
        positive = (
            self.band_radius_pixels,
            self.normal_offset_step_pixels,
            self.minimum_carrier_length_pixels,
            self.anchor_edge_softness_bins,
            self.anchor_endpoint_sigma_bins,
        )
        if min(positive) <= 0.0:
            raise ValueError("Evidence Carrier geometric scales must be positive")
        if self.normal_offset_radius_pixels < 0.0:
            raise ValueError("Evidence Carrier normal-offset radius cannot be negative")
        steps = 2.0 * (
            self.normal_offset_radius_pixels / self.normal_offset_step_pixels
        )
        if abs(steps - round(steps)) > 1e-6:
            raise ValueError(
                "Evidence Carrier normal-offset range must contain an integer "
                "number of steps"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("Evidence Carrier dropout must be in [0, 1)")

    @property
    def normal_offset_bins(self) -> int:
        return int(
            round(
                2.0
                * self.normal_offset_radius_pixels
                / self.normal_offset_step_pixels
            )
        ) + 1

    def normal_offset_values(
        self,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        return torch.linspace(
            -self.normal_offset_radius_pixels,
            self.normal_offset_radius_pixels,
            self.normal_offset_bins,
            device=device,
            dtype=dtype,
        )

    def extent_fraction_values(
        self,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        return torch.linspace(
            0.0,
            1.0,
            self.extent_bins,
            device=device,
            dtype=dtype,
        )


@dataclass(frozen=True, slots=True)
class EvidenceCarrierStripGeometry:
    """Candidate-relative strips and the geometry needed for decoding."""

    oriented_strips: torch.Tensor
    canonical_segments_pixels: torch.Tensor
    carrier_centers_pixels: torch.Tensor
    carrier_axes: torch.Tensor
    carrier_normals: torch.Tensor
    clipped_segments_pixels: torch.Tensor
    anchor_interval_fractions: torch.Tensor
    anchor_support_mask: torch.Tensor
    anchor_endpoint_maps: torch.Tensor
    intersection_mask: torch.Tensor
    image_size_pixels: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


def _validate_coordinate_mode(mode: str) -> str:
    if mode == "pixel":
        return "pixels"
    if mode not in ("normalized", "pixels"):
        raise ValueError(
            "Evidence Carrier coordinate_mode must be 'normalized' or 'pixels'"
        )
    return mode


def _validate_image_and_carriers(
    pixel_values: torch.Tensor,
    carrier_segments: torch.Tensor,
    carrier_mask: torch.Tensor,
    *,
    coordinate_mode: str,
    minimum_length_pixels: float,
) -> tuple[torch.Tensor, str]:
    if pixel_values.ndim != 4 or pixel_values.size(1) != 3:
        raise ValueError(
            "Evidence Carrier pixel_values must have shape [batch, 3, height, width]"
        )
    if not pixel_values.is_floating_point():
        raise ValueError("Evidence Carrier pixel_values must be floating point")
    if pixel_values.size(-2) < 2 or pixel_values.size(-1) < 2:
        raise ValueError("Evidence Carrier images must be at least 2x2 pixels")
    if carrier_segments.ndim != 4 or carrier_segments.shape[-2:] != (2, 2):
        raise ValueError(
            "Evidence Carrier segments must have shape [batch, carriers, 2, 2]"
        )
    if carrier_segments.size(0) != pixel_values.size(0):
        raise ValueError("Evidence Carrier image and segment batches differ")
    if not carrier_segments.is_floating_point():
        raise ValueError("Evidence Carrier segments must be floating point")
    expected_mask_shape = carrier_segments.shape[:2]
    if carrier_mask.shape != expected_mask_shape or carrier_mask.dtype != torch.bool:
        raise ValueError(
            "Evidence Carrier mask must be boolean with shape [batch, carriers]"
        )
    if not (
        pixel_values.device == carrier_segments.device == carrier_mask.device
    ):
        raise ValueError("Evidence Carrier inputs must be on the same device")
    if not bool(torch.isfinite(pixel_values).all()):
        raise ValueError("Evidence Carrier pixel_values contain non-finite values")
    if not bool(torch.isfinite(carrier_segments).all()):
        raise ValueError("Evidence Carrier segments contain non-finite values")

    mode = _validate_coordinate_mode(coordinate_mode)
    height, width = pixel_values.shape[-2:]
    segments = carrier_segments.to(dtype=pixel_values.dtype)
    if mode == "normalized":
        valid_values = segments[carrier_mask]
        if valid_values.numel() and bool(
            ((valid_values < 0.0) | (valid_values > 1.0)).any()
        ):
            raise ValueError(
                "Normalized Evidence Carrier coordinates must lie in [0, 1]"
            )
        scale = segments.new_tensor((float(width - 1), float(height - 1)))
        segments = segments * scale
    else:
        valid_values = segments[carrier_mask]
        if valid_values.numel():
            invalid_x = (valid_values[..., 0] < 0.0) | (
                valid_values[..., 0] > float(width - 1)
            )
            invalid_y = (valid_values[..., 1] < 0.0) | (
                valid_values[..., 1] > float(height - 1)
            )
            if bool((invalid_x | invalid_y).any()):
                raise ValueError(
                    "Pixel Evidence Carrier coordinates must lie inside the image"
                )

    lengths = (segments[..., 1, :] - segments[..., 0, :]).norm(dim=-1)
    if bool((carrier_mask & (lengths < minimum_length_pixels)).any()):
        raise ValueError("Evidence Carrier contains a zero-length valid segment")
    return segments, mode


def canonicalize_carrier_segments(
    segments_pixels: torch.Tensor,
    carrier_mask: torch.Tensor,
    *,
    minimum_length_pixels: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Canonicalize endpoint order and return segments, centers, axes, normals.

    The canonical axis points toward positive x.  Exactly vertical carriers
    point toward positive y.  Reversing either input segment therefore produces
    bitwise-identical geometry apart from ordinary floating-point arithmetic.
    """

    if segments_pixels.ndim != 4 or segments_pixels.shape[-2:] != (2, 2):
        raise ValueError("Carrier segments must have shape [batch, carriers, 2, 2]")
    if carrier_mask.shape != segments_pixels.shape[:2] or carrier_mask.dtype != torch.bool:
        raise ValueError("Carrier mask shape or dtype is invalid")
    if not segments_pixels.is_floating_point():
        raise ValueError("Carrier segments must be floating point")
    if segments_pixels.device != carrier_mask.device:
        raise ValueError("Carrier segments and mask must share one device")
    if minimum_length_pixels <= 0.0:
        raise ValueError("Minimum carrier length must be positive")
    if not bool(torch.isfinite(segments_pixels).all()):
        raise ValueError("Carrier segments contain non-finite values")

    delta = segments_pixels[..., 1, :] - segments_pixels[..., 0, :]
    lengths = delta.norm(dim=-1)
    if bool((carrier_mask & (lengths < minimum_length_pixels)).any()):
        raise ValueError("Carrier contains a zero-length valid segment")
    vertical = delta[..., 0].abs() <= torch.finfo(delta.dtype).eps * 8.0
    reverse = (delta[..., 0] < 0.0) | (vertical & (delta[..., 1] < 0.0))
    canonical = torch.where(
        reverse[..., None, None],
        segments_pixels.flip(-2),
        segments_pixels,
    )
    canonical_delta = canonical[..., 1, :] - canonical[..., 0, :]
    safe_lengths = canonical_delta.norm(dim=-1).clamp_min(minimum_length_pixels)
    axes = canonical_delta / safe_lengths.unsqueeze(-1)
    normals = torch.stack((-axes[..., 1], axes[..., 0]), dim=-1)
    centers = canonical.mean(dim=-2)

    valid = carrier_mask[..., None]
    canonical = torch.where(carrier_mask[..., None, None], canonical, 0.0)
    centers = torch.where(valid, centers, 0.0)
    axes = torch.where(valid, axes, 0.0)
    normals = torch.where(valid, normals, 0.0)
    return canonical, centers, axes, normals


def clip_carrier_lines_to_image(
    centers_pixels: torch.Tensor,
    carrier_axes: torch.Tensor,
    carrier_mask: torch.Tensor,
    *,
    image_height: int,
    image_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clip infinite canonical carrier lines to the continuous image rectangle."""

    if centers_pixels.ndim != 3 or centers_pixels.size(-1) != 2:
        raise ValueError("Carrier centers must have shape [batch, carriers, 2]")
    if carrier_axes.shape != centers_pixels.shape:
        raise ValueError("Carrier axes and centers must have identical shapes")
    if carrier_mask.shape != centers_pixels.shape[:2] or carrier_mask.dtype != torch.bool:
        raise ValueError("Carrier mask shape or dtype is invalid")
    if not centers_pixels.is_floating_point() or not carrier_axes.is_floating_point():
        raise ValueError("Carrier centers and axes must be floating point")
    if not (
        centers_pixels.device == carrier_axes.device == carrier_mask.device
    ):
        raise ValueError("Carrier clipping tensors must share one device")
    if image_height < 2 or image_width < 2:
        raise ValueError("Carrier clipping image dimensions must be at least two")
    if not bool(torch.isfinite(centers_pixels).all()) or not bool(
        torch.isfinite(carrier_axes).all()
    ):
        raise ValueError("Carrier clipping tensors contain non-finite values")

    dtype = centers_pixels.dtype
    epsilon = torch.finfo(dtype).eps * 16.0
    axis_lengths = carrier_axes.norm(dim=-1)
    if bool((carrier_mask & (axis_lengths <= epsilon)).any()):
        raise ValueError("Valid carrier axes must be nonzero")
    low = centers_pixels.new_full(carrier_mask.shape, -torch.inf)
    high = centers_pixels.new_full(carrier_mask.shape, torch.inf)
    intersects = carrier_mask.clone()
    bounds = (float(image_width - 1), float(image_height - 1))
    for dimension, maximum in enumerate(bounds):
        coordinate = centers_pixels[..., dimension]
        component = carrier_axes[..., dimension]
        moving = component.abs() > epsilon
        safe_component = torch.where(moving, component, 1.0)
        first = -coordinate / safe_component
        second = (maximum - coordinate) / safe_component
        dimension_low = torch.minimum(first, second)
        dimension_high = torch.maximum(first, second)
        low = torch.maximum(low, torch.where(moving, dimension_low, -torch.inf))
        high = torch.minimum(high, torch.where(moving, dimension_high, torch.inf))
        inside = (coordinate >= -epsilon) & (coordinate <= maximum + epsilon)
        intersects &= moving | inside

    intersects &= low <= high + epsilon
    safe_low = torch.where(intersects, low, 0.0)
    safe_high = torch.where(intersects, high, 0.0)
    start = centers_pixels + carrier_axes * safe_low.unsqueeze(-1)
    end = centers_pixels + carrier_axes * safe_high.unsqueeze(-1)
    start = torch.stack(
        (
            start[..., 0].clamp(0.0, float(image_width - 1)),
            start[..., 1].clamp(0.0, float(image_height - 1)),
        ),
        dim=-1,
    )
    end = torch.stack(
        (
            end[..., 0].clamp(0.0, float(image_width - 1)),
            end[..., 1].clamp(0.0, float(image_height - 1)),
        ),
        dim=-1,
    )
    clipped = torch.stack((start, end), dim=-2)
    clipped = torch.where(intersects[..., None, None], clipped, 0.0)
    return clipped, intersects


def _carrier_anchor_channels(
    canonical_segments_pixels: torch.Tensor,
    clipped_segments_pixels: torch.Tensor,
    intersection_mask: torch.Tensor,
    config: EvidenceCarrierExtentConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode the proposal segment only in its clipped-line coordinate frame."""

    clipped_start = clipped_segments_pixels[..., 0, :]
    clipped_vector = (
        clipped_segments_pixels[..., 1, :]
        - clipped_start
    )
    denominator = clipped_vector.square().sum(dim=-1).clamp_min(
        config.minimum_carrier_length_pixels**2
    )
    projected = (
        (canonical_segments_pixels - clipped_start.unsqueeze(-2))
        * clipped_vector.unsqueeze(-2)
    ).sum(dim=-1) / denominator.unsqueeze(-1)
    left = projected.min(dim=-1).values.clamp(0.0, 1.0)
    right = projected.max(dim=-1).values.clamp(0.0, 1.0)
    intervals = torch.stack((left, right), dim=-1)
    intervals = torch.where(
        intersection_mask[..., None],
        intervals,
        0.0,
    )

    positions = torch.linspace(
        0.0,
        1.0,
        config.longitudinal_samples,
        device=canonical_segments_pixels.device,
        dtype=canonical_segments_pixels.dtype,
    )
    bin_fraction = 1.0 / float(config.longitudinal_samples - 1)
    edge_scale = config.anchor_edge_softness_bins * bin_fraction
    support = torch.sigmoid(
        (positions[None, None, :] - left[..., None]) / edge_scale
    ) * torch.sigmoid(
        (right[..., None] - positions[None, None, :]) / edge_scale
    )
    endpoint_scale = config.anchor_endpoint_sigma_bins * bin_fraction
    left_endpoint = torch.exp(
        -0.5
        * (
            (positions[None, None, :] - left[..., None])
            / endpoint_scale
        ).square()
    )
    right_endpoint = torch.exp(
        -0.5
        * (
            (positions[None, None, :] - right[..., None])
            / endpoint_scale
        ).square()
    )
    support = support * intersection_mask[..., None].to(support.dtype)
    endpoints = torch.stack((left_endpoint, right_endpoint), dim=2)
    endpoints = endpoints * intersection_mask[..., None, None].to(endpoints.dtype)
    support = support[:, :, None, None, :].expand(
        -1,
        -1,
        1,
        config.band_samples,
        -1,
    )
    endpoints = endpoints[:, :, :, None, :].expand(
        -1,
        -1,
        -1,
        config.band_samples,
        -1,
    )
    return intervals, support, endpoints


def extract_evidence_carrier_strips(
    pixel_values: torch.Tensor,
    carrier_segments: torch.Tensor,
    carrier_mask: torch.Tensor,
    config: EvidenceCarrierExtentConfig = EvidenceCarrierExtentConfig(),
    *,
    coordinate_mode: CarrierCoordinateMode = "normalized",
) -> EvidenceCarrierStripGeometry:
    """Sample candidate-relative RGB ink strips over complete clipped lines."""

    segments_pixels, _ = _validate_image_and_carriers(
        pixel_values,
        carrier_segments,
        carrier_mask,
        coordinate_mode=coordinate_mode,
        minimum_length_pixels=config.minimum_carrier_length_pixels,
    )
    canonical, centers, axes, normals = canonicalize_carrier_segments(
        segments_pixels,
        carrier_mask,
        minimum_length_pixels=config.minimum_carrier_length_pixels,
    )
    height, width = pixel_values.shape[-2:]
    clipped, intersection_mask = clip_carrier_lines_to_image(
        centers,
        axes,
        carrier_mask,
        image_height=height,
        image_width=width,
    )
    anchor_intervals, anchor_support, anchor_endpoints = _carrier_anchor_channels(
        canonical,
        clipped,
        intersection_mask,
        config,
    )
    batch, carriers = carrier_mask.shape
    if carriers == 0:
        strips = pixel_values.new_empty(
            (
                batch,
                0,
                3,
                config.band_samples,
                config.longitudinal_samples,
            )
        )
    else:
        fractions = torch.linspace(
            0.0,
            1.0,
            config.longitudinal_samples,
            device=pixel_values.device,
            dtype=pixel_values.dtype,
        )
        offsets = torch.linspace(
            -config.band_radius_pixels,
            config.band_radius_pixels,
            config.band_samples,
            device=pixel_values.device,
            dtype=pixel_values.dtype,
        )
        start = clipped[..., 0, :]
        delta = clipped[..., 1, :] - start
        base = start[..., None, :] + delta[..., None, :] * fractions[None, None, :, None]
        grid_pixels = base[:, :, None, :, :] + (
            normals[:, :, None, None, :]
            * offsets[None, None, :, None, None]
        )
        grid = torch.stack(
            (
                2.0 * grid_pixels[..., 0] / float(width - 1) - 1.0,
                2.0 * grid_pixels[..., 1] / float(height - 1) - 1.0,
            ),
            dim=-1,
        ).reshape(
            batch,
            carriers * config.band_samples,
            config.longitudinal_samples,
            2,
        )
        # White normalized RGB maps to zero ink, matching grid_sample padding.
        ink_rgb = pixel_values.neg().add(1.0).mul(0.5).clamp(0.0, 1.0)
        sampled = F.grid_sample(
            ink_rgb,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        strips = sampled.reshape(
            batch,
            3,
            carriers,
            config.band_samples,
            config.longitudinal_samples,
        ).permute(0, 2, 1, 3, 4)
        strips = strips * intersection_mask[..., None, None, None].to(strips.dtype)

    image_size = pixel_values.new_tensor((float(height), float(width)))
    return EvidenceCarrierStripGeometry(
        oriented_strips=strips,
        canonical_segments_pixels=canonical,
        carrier_centers_pixels=centers,
        carrier_axes=axes,
        carrier_normals=normals,
        clipped_segments_pixels=clipped,
        anchor_interval_fractions=anchor_intervals,
        anchor_support_mask=anchor_support,
        anchor_endpoint_maps=anchor_endpoints,
        intersection_mask=intersection_mask,
        image_size_pixels=image_size,
    )


class EvidenceCarrierExtentPredictor(nn.Module):
    """Classify keep, normal offset, and bounded extents for each carrier."""

    model_type = "evidence-carrier-extent-v0.1"

    def __init__(
        self,
        config: EvidenceCarrierExtentConfig = EvidenceCarrierExtentConfig(),
    ) -> None:
        super().__init__()
        self.config = config
        self.strip_encoder = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, 32),
            nn.GELU(),
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                stride=(2, 1),
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(1, 64),
            nn.GELU(),
            nn.Conv2d(
                64,
                96,
                kernel_size=3,
                stride=(2, 1),
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(1, 96),
            nn.GELU(),
        )
        encoded_band = (config.band_samples + 1) // 2
        encoded_band = (encoded_band + 1) // 2
        self.longitudinal_encoder = nn.Sequential(
            nn.Conv1d(96 * encoded_band, config.hidden_dim, kernel_size=1),
            nn.GroupNorm(1, config.hidden_dim),
            nn.GELU(),
            nn.Conv1d(
                config.hidden_dim,
                config.hidden_dim,
                kernel_size=5,
                padding=2,
                bias=False,
            ),
            nn.GroupNorm(1, config.hidden_dim),
            nn.GELU(),
            nn.Conv1d(
                config.hidden_dim,
                config.hidden_dim,
                kernel_size=5,
                padding=2,
                bias=False,
            ),
            nn.GroupNorm(1, config.hidden_dim),
            nn.GELU(),
        )
        self.global_shared = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
        )
        self.keep_head = nn.Linear(config.hidden_dim, 1)
        self.normal_offset_head = nn.Linear(
            config.hidden_dim,
            config.normal_offset_bins,
        )
        self.left_extent_head = nn.Conv1d(config.hidden_dim, 1, kernel_size=1)
        self.right_extent_head = nn.Conv1d(config.hidden_dim, 1, kernel_size=1)
        self.register_buffer(
            "normal_offset_values_pixels",
            config.normal_offset_values(),
            persistent=False,
        )
        self.register_buffer(
            "extent_fraction_values",
            config.extent_fraction_values(),
            persistent=False,
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        carrier_segments: torch.Tensor,
        carrier_mask: torch.Tensor,
        *,
        coordinate_mode: CarrierCoordinateMode = "normalized",
    ) -> dict[str, torch.Tensor]:
        geometry = extract_evidence_carrier_strips(
            pixel_values,
            carrier_segments,
            carrier_mask,
            self.config,
            coordinate_mode=coordinate_mode,
        )
        batch, carriers = carrier_mask.shape
        carrier_input_strips = torch.cat(
            (
                geometry.oriented_strips,
                geometry.anchor_support_mask,
                geometry.anchor_endpoint_maps,
            ),
            dim=2,
        )
        if carriers:
            flat = carrier_input_strips.reshape(
                batch * carriers,
                6,
                self.config.band_samples,
                self.config.longitudinal_samples,
            )
            encoded = self.strip_encoder(flat)
            encoded = encoded.flatten(1, 2)
            longitudinal = self.longitudinal_encoder(encoded)
            hidden = self.global_shared(longitudinal.mean(dim=-1)).reshape(
                batch,
                carriers,
                self.config.hidden_dim,
            )
            keep_logits = self.keep_head(hidden).squeeze(-1)
            normal_logits = self.normal_offset_head(hidden)
            left_logits = self.left_extent_head(longitudinal).squeeze(1)
            right_logits = self.right_extent_head(longitudinal).squeeze(1)
            if self.config.extent_bins != self.config.longitudinal_samples:
                left_logits = F.interpolate(
                    left_logits.unsqueeze(1),
                    size=self.config.extent_bins,
                    mode="linear",
                    align_corners=True,
                ).squeeze(1)
                right_logits = F.interpolate(
                    right_logits.unsqueeze(1),
                    size=self.config.extent_bins,
                    mode="linear",
                    align_corners=True,
                ).squeeze(1)
            left_logits = left_logits.reshape(batch, carriers, self.config.extent_bins)
            right_logits = right_logits.reshape(
                batch,
                carriers,
                self.config.extent_bins,
            )
            longitudinal = longitudinal.reshape(
                batch,
                carriers,
                self.config.hidden_dim,
                self.config.longitudinal_samples,
            )
            valid = geometry.intersection_mask
            keep_logits = torch.where(valid, keep_logits, 0.0)
            normal_logits = torch.where(valid[..., None], normal_logits, 0.0)
            left_logits = torch.where(valid[..., None], left_logits, 0.0)
            right_logits = torch.where(valid[..., None], right_logits, 0.0)
            hidden = torch.where(valid[..., None], hidden, 0.0)
            longitudinal = torch.where(
                valid[..., None, None],
                longitudinal,
                0.0,
            )
        else:
            # Keep an all-padding batch connected to the module parameters so
            # a normal training step can still call backward on its zero loss.
            anchor = self.keep_head.weight.sum() * 0.0
            hidden = pixel_values.new_empty(
                (batch, 0, self.config.hidden_dim)
            ) + anchor
            longitudinal = pixel_values.new_empty(
                (
                    batch,
                    0,
                    self.config.hidden_dim,
                    self.config.longitudinal_samples,
                )
            ) + anchor
            keep_logits = pixel_values.new_empty((batch, 0)) + anchor
            normal_logits = pixel_values.new_empty(
                (batch, 0, self.config.normal_offset_bins)
            ) + anchor
            left_logits = pixel_values.new_empty(
                (batch, 0, self.config.extent_bins)
            ) + anchor
            right_logits = pixel_values.new_empty(
                (batch, 0, self.config.extent_bins)
            ) + anchor

        return {
            "keep_logits": keep_logits,
            "normal_offset_logits": normal_logits,
            "left_extent_logits": left_logits,
            "right_extent_logits": right_logits,
            "carrier_features": hidden,
            "longitudinal_features": longitudinal,
            "carrier_input_strips": carrier_input_strips,
            "carrier_mask": carrier_mask,
            "normal_offset_values_pixels": self.normal_offset_values_pixels.to(
                dtype=pixel_values.dtype
            ),
            "extent_fraction_values": self.extent_fraction_values.to(
                dtype=pixel_values.dtype
            ),
            **geometry.as_dict(),
        }

    def config_dict(self) -> dict[str, Any]:
        return {"evidence_carrier_extent_config": asdict(self.config)}

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def decode_ordered_extent_bins(
    left_extent_logits: torch.Tensor,
    right_extent_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Jointly decode the best pair under the hard constraint left < right."""

    if left_extent_logits.shape != right_extent_logits.shape:
        raise ValueError("Left and right extent logits must have identical shapes")
    if left_extent_logits.ndim < 1 or left_extent_logits.size(-1) < 2:
        raise ValueError("Ordered extent decoding needs at least two bins")
    if not left_extent_logits.is_floating_point() or not right_extent_logits.is_floating_point():
        raise ValueError("Extent logits must be floating point")
    if not bool(torch.isfinite(left_extent_logits).all()) or not bool(
        torch.isfinite(right_extent_logits).all()
    ):
        raise ValueError("Extent logits contain non-finite values")

    prefix_scores, prefix_indices = torch.cummax(
        left_extent_logits[..., :-1],
        dim=-1,
    )
    pair_scores = prefix_scores + right_extent_logits[..., 1:]
    right_indices = pair_scores.argmax(dim=-1) + 1
    left_indices = torch.gather(
        prefix_indices,
        -1,
        (right_indices - 1).unsqueeze(-1),
    ).squeeze(-1)
    return left_indices, right_indices


def _validate_decode_output(output: dict[str, torch.Tensor]) -> None:
    required = {
        "keep_logits",
        "normal_offset_logits",
        "left_extent_logits",
        "right_extent_logits",
        "carrier_mask",
        "carrier_centers_pixels",
        "carrier_axes",
        "carrier_normals",
        "normal_offset_values_pixels",
        "extent_fraction_values",
        "image_size_pixels",
    }
    missing = sorted(required - output.keys())
    if missing:
        raise ValueError(f"Evidence Carrier decode fields are missing: {missing}")
    keep = output["keep_logits"]
    if keep.ndim != 2:
        raise ValueError("Evidence Carrier keep logits must have shape [batch, carriers]")
    batch, carriers = keep.shape
    mask = output["carrier_mask"]
    if mask.shape != keep.shape or mask.dtype != torch.bool:
        raise ValueError("Evidence Carrier decode mask shape or dtype is invalid")
    for name in (
        "normal_offset_logits",
        "left_extent_logits",
        "right_extent_logits",
    ):
        tensor = output[name]
        if tensor.ndim != 3 or tensor.shape[:2] != (batch, carriers):
            raise ValueError(f"Evidence Carrier {name} has an invalid shape")
    geometry_shape = (batch, carriers, 2)
    for name in (
        "carrier_centers_pixels",
        "carrier_axes",
        "carrier_normals",
    ):
        if output[name].shape != geometry_shape:
            raise ValueError(f"Evidence Carrier {name} has an invalid shape")
    normal_values = output["normal_offset_values_pixels"]
    extent_values = output["extent_fraction_values"]
    if normal_values.ndim != 1 or normal_values.numel() != output[
        "normal_offset_logits"
    ].size(-1):
        raise ValueError("Evidence Carrier normal-offset bin values are invalid")
    if extent_values.ndim != 1 or extent_values.numel() != output[
        "left_extent_logits"
    ].size(-1):
        raise ValueError("Evidence Carrier extent bin values are invalid")
    if output["right_extent_logits"].size(-1) != extent_values.numel():
        raise ValueError("Evidence Carrier left and right extent bins differ")
    if extent_values.numel() < 2 or bool((extent_values[1:] <= extent_values[:-1]).any()):
        raise ValueError("Evidence Carrier extent fractions must be strictly increasing")
    if float(extent_values[0]) < 0.0 or float(extent_values[-1]) > 1.0:
        raise ValueError("Evidence Carrier extent fractions must lie in [0, 1]")
    image_size = output["image_size_pixels"]
    if image_size.shape != (2,) or not bool(torch.isfinite(image_size).all()):
        raise ValueError("Evidence Carrier image_size_pixels must contain [height, width]")


@torch.no_grad()
def decode_evidence_carrier_extent(
    output: dict[str, torch.Tensor],
    *,
    keep_threshold: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Reconstruct finite lines after offsetting and re-clipping each carrier."""

    if not 0.0 <= keep_threshold <= 1.0:
        raise ValueError("Evidence Carrier keep threshold must be in [0, 1]")
    _validate_decode_output(output)
    left_indices, right_indices = decode_ordered_extent_bins(
        output["left_extent_logits"],
        output["right_extent_logits"],
    )
    normal_indices = output["normal_offset_logits"].argmax(dim=-1)
    normal_offsets = output["normal_offset_values_pixels"][normal_indices]
    shifted_centers = output["carrier_centers_pixels"] + (
        output["carrier_normals"] * normal_offsets.unsqueeze(-1)
    )
    image_height = int(round(float(output["image_size_pixels"][0])))
    image_width = int(round(float(output["image_size_pixels"][1])))
    offset_clipped, intersection_mask = clip_carrier_lines_to_image(
        shifted_centers,
        output["carrier_axes"],
        output["carrier_mask"],
        image_height=image_height,
        image_width=image_width,
    )
    fraction_values = output["extent_fraction_values"]
    left_fractions = fraction_values[left_indices]
    right_fractions = fraction_values[right_indices]
    start = offset_clipped[..., 0, :]
    interval = offset_clipped[..., 1, :] - start
    refined_start = start + interval * left_fractions.unsqueeze(-1)
    refined_end = start + interval * right_fractions.unsqueeze(-1)
    refined_pixels = torch.stack((refined_start, refined_end), dim=-2)
    active = (
        output["carrier_mask"]
        & intersection_mask
        & (output["keep_logits"].sigmoid() >= keep_threshold)
    )
    refined_pixels = torch.where(active[..., None, None], refined_pixels, 0.0)
    scale = refined_pixels.new_tensor(
        (float(image_width - 1), float(image_height - 1))
    )
    refined_normalized = refined_pixels / scale
    return {
        "keep_scores": output["keep_logits"].sigmoid(),
        "keep_mask": active,
        "normal_offset_bins": normal_indices,
        "normal_offsets_pixels": normal_offsets,
        "left_extent_bins": left_indices,
        "right_extent_bins": right_indices,
        "left_extent_fractions": left_fractions,
        "right_extent_fractions": right_fractions,
        "offset_clipped_segments_pixels": offset_clipped,
        "offset_intersection_mask": intersection_mask,
        "refined_segments_pixels": refined_pixels,
        "refined_segments_normalized": refined_normalized,
    }


def _validate_class_targets(
    name: str,
    targets: torch.Tensor,
    expected_shape: torch.Size,
    classes: int,
    active: torch.Tensor,
) -> None:
    if targets.shape != expected_shape:
        raise ValueError(f"Evidence Carrier {name} targets have an invalid shape")
    if targets.dtype == torch.bool or targets.is_floating_point():
        raise ValueError(f"Evidence Carrier {name} targets must use an integer dtype")
    selected = targets[active]
    if selected.numel() and bool(((selected < 0) | (selected >= classes)).any()):
        raise ValueError(f"Evidence Carrier {name} target is outside the class range")


def _optional_supervision_mask(
    name: str,
    value: torch.Tensor | None,
    carrier_mask: torch.Tensor,
) -> torch.Tensor:
    if value is None:
        return carrier_mask
    if value.shape != carrier_mask.shape or value.dtype != torch.bool:
        raise ValueError(
            f"Evidence Carrier {name} must be boolean with shape [batch, carriers]"
        )
    if value.device != carrier_mask.device:
        raise ValueError("Evidence Carrier supervision masks must share the logits device")
    if bool((value & ~carrier_mask).any()):
        raise ValueError(f"Evidence Carrier {name} includes padded carriers")
    return value


def _carrier_loss_weights(
    sample_weight: torch.Tensor | None,
    equivalence_weight: torch.Tensor | None,
    carrier_mask: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if sample_weight is not None and equivalence_weight is not None:
        raise ValueError(
            "Evidence Carrier accepts either sample_weight or equivalence_weight, not both"
        )
    weights = equivalence_weight if equivalence_weight is not None else sample_weight
    if weights is None:
        return None
    if weights.shape != carrier_mask.shape or not weights.is_floating_point():
        raise ValueError(
            "Evidence Carrier sample weights must be floating point with shape "
            "[batch, carriers]"
        )
    if weights.device != carrier_mask.device:
        raise ValueError("Evidence Carrier sample weights must share the logits device")
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0.0).any()):
        raise ValueError("Evidence Carrier sample weights must be finite and nonnegative")
    return weights.to(dtype=dtype)


def _masked_weighted_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor | None,
    zero: torch.Tensor,
) -> torch.Tensor:
    if not bool(mask.any()):
        return zero
    selected = values[mask]
    if weights is None:
        return selected.mean()
    selected_weights = weights[mask]
    denominator = selected_weights.sum()
    if float(denominator.detach()) <= 0.0:
        return zero
    return (selected * selected_weights).sum() / denominator


def evidence_carrier_extent_loss(
    output: dict[str, torch.Tensor],
    keep_labels: torch.Tensor,
    normal_offset_bin: torch.Tensor,
    extent_left_bin: torch.Tensor,
    extent_right_bin: torch.Tensor,
    *,
    keep_loss_mask: torch.Tensor | None = None,
    geometry_loss_mask: torch.Tensor | None = None,
    sample_weight: torch.Tensor | None = None,
    equivalence_weight: torch.Tensor | None = None,
    keep_weight: float = 1.0,
    normal_offset_weight: float = 1.0,
    extent_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Mask-aware classification loss for bounded carrier reconstruction."""

    weights = (keep_weight, normal_offset_weight, extent_weight)
    if min(weights) < 0.0:
        raise ValueError("Evidence Carrier loss weights must be nonnegative")
    _validate_decode_output(output)
    keep_logits = output["keep_logits"]
    carrier_mask = output["carrier_mask"]
    if keep_labels.shape != keep_logits.shape:
        raise ValueError("Evidence Carrier keep targets have an invalid shape")
    if keep_labels.device != keep_logits.device:
        raise ValueError("Evidence Carrier targets and logits must share one device")
    if keep_labels.dtype == torch.bool:
        keep_float = keep_labels.to(keep_logits.dtype)
        keep_positive = keep_labels
        keep_defined = torch.ones_like(carrier_mask)
    elif keep_labels.is_floating_point():
        if not bool(torch.isfinite(keep_labels).all()):
            raise ValueError("Evidence Carrier keep targets contain non-finite values")
        selected = keep_labels[carrier_mask]
        if selected.numel() and bool(((selected < 0.0) | (selected > 1.0)).any()):
            invalid = (selected != -1.0) & ((selected < 0.0) | (selected > 1.0))
            if bool(invalid.any()):
                raise ValueError(
                    "Evidence Carrier keep targets must be -1 or lie in [0, 1]"
                )
        keep_defined = keep_labels >= 0.0
        keep_float = keep_labels.clamp_min(0.0).to(keep_logits.dtype)
        keep_positive = keep_labels >= 0.5
    else:
        selected = keep_labels[carrier_mask]
        invalid = (selected != -1) & (selected != 0) & (selected != 1)
        if bool(invalid.any()):
            raise ValueError(
                "Evidence Carrier integer keep targets must be -1, zero, or one"
            )
        keep_defined = keep_labels != -1
        keep_float = keep_labels.clamp_min(0).to(keep_logits.dtype)
        keep_positive = keep_labels == 1

    keep_supervision = (
        carrier_mask
        & keep_defined
        & _optional_supervision_mask(
            "keep_loss_mask",
            keep_loss_mask,
            carrier_mask,
        )
    )
    geometry_supervision = (
        carrier_mask
        & keep_defined
        & keep_positive
        & _optional_supervision_mask(
            "geometry_loss_mask",
            geometry_loss_mask,
            carrier_mask,
        )
    )
    carrier_weights = _carrier_loss_weights(
        sample_weight,
        equivalence_weight,
        carrier_mask,
        keep_logits.dtype,
    )
    for target in (normal_offset_bin, extent_left_bin, extent_right_bin):
        if target.device != keep_logits.device:
            raise ValueError("Evidence Carrier targets and logits must share one device")
    _validate_class_targets(
        "normal-offset",
        normal_offset_bin,
        keep_logits.shape,
        output["normal_offset_logits"].size(-1),
        geometry_supervision,
    )
    _validate_class_targets(
        "left-extent",
        extent_left_bin,
        keep_logits.shape,
        output["left_extent_logits"].size(-1),
        geometry_supervision,
    )
    _validate_class_targets(
        "right-extent",
        extent_right_bin,
        keep_logits.shape,
        output["right_extent_logits"].size(-1),
        geometry_supervision,
    )
    if bool(
        (
            extent_left_bin[geometry_supervision]
            >= extent_right_bin[geometry_supervision]
        ).any()
    ):
        raise ValueError("Evidence Carrier positive targets must satisfy left < right")

    zero = (
        keep_logits.sum()
        + output["normal_offset_logits"].sum()
        + output["left_extent_logits"].sum()
        + output["right_extent_logits"].sum()
    ) * 0.0
    keep_losses = F.binary_cross_entropy_with_logits(
        keep_logits,
        keep_float,
        reduction="none",
    )
    keep_loss = _masked_weighted_mean(
        keep_losses,
        keep_supervision,
        carrier_weights,
        zero,
    )
    if bool(geometry_supervision.any()):
        normal_values = F.cross_entropy(
            output["normal_offset_logits"][geometry_supervision],
            normal_offset_bin[geometry_supervision].long(),
            reduction="none",
        )
        left_values = F.cross_entropy(
            output["left_extent_logits"][geometry_supervision],
            extent_left_bin[geometry_supervision].long(),
            reduction="none",
        )
        right_values = F.cross_entropy(
            output["right_extent_logits"][geometry_supervision],
            extent_right_bin[geometry_supervision].long(),
            reduction="none",
        )
        geometry_weights = (
            carrier_weights[geometry_supervision]
            if carrier_weights is not None
            else None
        )
        if geometry_weights is None:
            normal_loss = normal_values.mean()
            left_loss = left_values.mean()
            right_loss = right_values.mean()
        else:
            denominator = geometry_weights.sum()
            if float(denominator.detach()) <= 0.0:
                normal_loss = left_loss = right_loss = zero
            else:
                normal_loss = (normal_values * geometry_weights).sum() / denominator
                left_loss = (left_values * geometry_weights).sum() / denominator
                right_loss = (right_values * geometry_weights).sum() / denominator
    else:
        normal_loss = left_loss = right_loss = zero
    extent_loss = 0.5 * (left_loss + right_loss)
    loss = (
        keep_weight * keep_loss
        + normal_offset_weight * normal_loss
        + extent_weight * extent_loss
    )
    return {
        "loss": loss,
        "keep_loss": keep_loss,
        "normal_offset_loss": normal_loss,
        "extent_loss": extent_loss,
        "left_extent_loss": left_loss,
        "right_extent_loss": right_loss,
        "active_carrier_count": carrier_mask.sum(),
        "keep_supervision_count": keep_supervision.sum(),
        "positive_carrier_count": geometry_supervision.sum(),
        "geometry_supervision_count": geometry_supervision.sum(),
        "keep_supervision_weight": (
            carrier_weights[keep_supervision].sum()
            if carrier_weights is not None
            else keep_supervision.sum().to(keep_logits.dtype)
        ),
        "geometry_supervision_weight": (
            carrier_weights[geometry_supervision].sum()
            if carrier_weights is not None
            else geometry_supervision.sum().to(keep_logits.dtype)
        ),
    }


__all__ = [
    "CarrierCoordinateMode",
    "EvidenceCarrierExtentConfig",
    "EvidenceCarrierExtentPredictor",
    "EvidenceCarrierStripGeometry",
    "canonicalize_carrier_segments",
    "clip_carrier_lines_to_image",
    "decode_evidence_carrier_extent",
    "decode_ordered_extent_bins",
    "evidence_carrier_extent_loss",
    "extract_evidence_carrier_strips",
]
