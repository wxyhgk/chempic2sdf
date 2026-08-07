"""Lightweight Atom-center field over frozen P1-A spatial hidden features."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from chemvector.components.vision.vertex import focal_heatmap_loss


G3_ATOM_CENTER_FIELD_CHANNELS = 256
G3_STROKE_FEATURE_CHANNELS = 96


class G3AtomCenterField(nn.Module):
    """Predict one center logit and sub-cell offset from frozen hidden input."""

    def __init__(self) -> None:
        super().__init__()
        self.center = nn.Conv2d(G3_ATOM_CENTER_FIELD_CHANNELS, 1, kernel_size=1)
        self.offset = nn.Conv2d(G3_ATOM_CENTER_FIELD_CHANNELS, 2, kernel_size=1)
        nn.init.constant_(self.center.bias, -4.6)

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        if (
            hidden.ndim != 4
            or hidden.size(1) != G3_ATOM_CENTER_FIELD_CHANNELS
            or min(hidden.shape[-2:]) < 1
        ):
            raise ValueError("G3 AtomCenterField hidden must have shape [batch,256,h,w]")
        if not hidden.is_floating_point() or not bool(torch.isfinite(hidden).all()):
            raise ValueError("G3 AtomCenterField hidden must be finite floating point")
        frozen = hidden.detach()
        center_logits = self.center(frozen)
        offset_logits = self.offset(frozen)
        return {
            "center_logits": center_logits,
            "offset_logits": offset_logits,
            "offsets": offset_logits.sigmoid(),
        }

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class G3AtomCenterFieldLocalResidual(nn.Module):
    """Frozen P1-A slot-0 center field with a zero-start local adapter."""

    def __init__(self, atom_head: nn.Module) -> None:
        super().__init__()
        presence, anchor_offset = _validate_p1a_atom_head(atom_head)
        self.register_buffer(
            "base_center_weight", presence.weight[0:1].detach().clone()
        )
        self.register_buffer(
            "base_center_bias", presence.bias[0:1].detach().clone()
        )
        self.register_buffer(
            "base_offset_weight",
            anchor_offset.weight.detach()
            .reshape(3, 2, G3_ATOM_CENTER_FIELD_CHANNELS, 1, 1)[0]
            .clone(),
        )
        self.register_buffer(
            "base_offset_bias",
            anchor_offset.bias.detach().reshape(3, 2)[0].clone(),
        )
        self.adapter = nn.Sequential(
            nn.Conv2d(G3_ATOM_CENTER_FIELD_CHANNELS, 32, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32),
            nn.GELU(),
            nn.Conv2d(32, 3, kernel_size=1),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        if (
            hidden.ndim != 4
            or hidden.size(1) != G3_ATOM_CENTER_FIELD_CHANNELS
            or min(hidden.shape[-2:]) < 1
        ):
            raise ValueError(
                "G3 local-residual AtomCenterField hidden must have shape "
                "[batch,256,h,w]"
            )
        if not hidden.is_floating_point() or not bool(torch.isfinite(hidden).all()):
            raise ValueError(
                "G3 local-residual AtomCenterField hidden must be finite floating point"
            )
        frozen = hidden.detach()
        base_center_logits = F.conv2d(
            frozen, self.base_center_weight, self.base_center_bias
        )
        base_offset_logits = F.conv2d(
            frozen, self.base_offset_weight, self.base_offset_bias
        )
        residual = self.adapter(frozen)
        center_residual = residual[:, 0:1]
        offset_residual = residual[:, 1:3]
        center_logits = base_center_logits + center_residual
        offset_logits = base_offset_logits + offset_residual
        return {
            "center_logits": center_logits,
            "offset_logits": offset_logits,
            "offsets": offset_logits.sigmoid(),
            "base_center_logits": base_center_logits,
            "base_offset_logits": base_offset_logits,
            "center_residual": center_residual,
            "offset_residual": offset_residual,
        }

    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )


class G3AtomCenterFieldHighResolutionResidual(nn.Module):
    """Two-times-resolution center field over frozen 192x192 P1-A features."""

    def __init__(self, atom_head: nn.Module) -> None:
        super().__init__()
        presence, _ = _validate_p1a_atom_head(atom_head)
        self.register_buffer(
            "base_center_weight", presence.weight[0:1].detach().clone()
        )
        self.register_buffer(
            "base_center_bias", presence.bias[0:1].detach().clone()
        )
        self.input_projection = nn.Conv2d(
            G3_ATOM_CENTER_FIELD_CHANNELS, 32, kernel_size=1
        )
        self.activation = nn.GELU()
        self.local = nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32)
        self.output_projection = nn.Conv2d(32, 3, kernel_size=1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        if hidden.ndim != 4 or hidden.shape[1:] != (
            G3_ATOM_CENTER_FIELD_CHANNELS,
            192,
            192,
        ):
            raise ValueError(
                "G3 high-resolution AtomCenterField hidden must have shape "
                "[batch,256,192,192]"
            )
        if not hidden.is_floating_point() or not bool(torch.isfinite(hidden).all()):
            raise ValueError(
                "G3 high-resolution AtomCenterField hidden must be finite floating point"
            )
        frozen = hidden.detach()
        base_center_low_resolution = F.conv2d(
            frozen, self.base_center_weight, self.base_center_bias
        )
        base_center_logits = F.interpolate(
            base_center_low_resolution,
            scale_factor=2.0,
            mode="bilinear",
            align_corners=False,
        )
        adapted = self.activation(self.input_projection(frozen))
        adapted = F.interpolate(
            adapted,
            scale_factor=2.0,
            mode="bilinear",
            align_corners=False,
        )
        residual = self.output_projection(self.activation(self.local(adapted)))
        center_residual = residual[:, 0:1]
        offset_logits = residual[:, 1:3]
        center_logits = base_center_logits + center_residual
        return {
            "center_logits": center_logits,
            "offset_logits": offset_logits,
            "offsets": offset_logits.sigmoid(),
            "base_center_logits": base_center_logits,
            "base_center_low_resolution": base_center_low_resolution,
            "center_residual": center_residual,
            "offset_residual": offset_logits,
        }

    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )


class VirtualStride2AtomCenterField(nn.Module):
    """Predict a virtual stride-2 center field from frozen stride-4 features."""

    def __init__(self) -> None:
        super().__init__()
        self.input_projection = nn.Conv2d(
            G3_STROKE_FEATURE_CHANNELS, 64, kernel_size=1
        )
        self.activation = nn.GELU()
        self.local = nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64)
        self.subpixel_projection = nn.Conv2d(64, 12, kernel_size=1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        with torch.no_grad():
            self.subpixel_projection.bias[:4].fill_(-4.6)
            self.subpixel_projection.bias[4:].zero_()

    def forward(self, stroke_features: torch.Tensor) -> dict[str, torch.Tensor]:
        if stroke_features.ndim != 4 or stroke_features.shape[1:] != (
            G3_STROKE_FEATURE_CHANNELS,
            192,
            192,
        ):
            raise ValueError(
                "Virtual stride-2 AtomCenterField features must have shape "
                "[batch,96,192,192]"
            )
        if not stroke_features.is_floating_point() or not bool(
            torch.isfinite(stroke_features).all()
        ):
            raise ValueError(
                "Virtual stride-2 AtomCenterField features must be finite floating point"
            )
        frozen = stroke_features.detach()
        hidden = self.activation(self.input_projection(frozen))
        hidden = self.activation(self.local(hidden))
        logits = self.pixel_shuffle(self.subpixel_projection(hidden))
        center_logits = logits[:, 0:1]
        offset_logits = logits[:, 1:3]
        return {
            "center_logits": center_logits,
            "offset_logits": offset_logits,
            "offsets": offset_logits.sigmoid(),
        }

    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )


class VirtualStride2AtomCenterResidual(nn.Module):
    """Add phase-aware stride-2 residuals to the frozen P1-A center field."""

    def __init__(self, atom_head: nn.Module) -> None:
        super().__init__()
        presence, _ = _validate_p1a_atom_head(atom_head)
        self.register_buffer(
            "base_center_weight", presence.weight[0:1].detach().clone()
        )
        self.register_buffer(
            "base_center_bias", presence.bias[0:1].detach().clone()
        )
        self.input_projection = nn.Conv2d(
            G3_STROKE_FEATURE_CHANNELS, 64, kernel_size=1
        )
        self.activation = nn.GELU()
        self.local = nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64)
        self.subpixel_projection = nn.Conv2d(64, 12, kernel_size=1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        nn.init.zeros_(self.subpixel_projection.weight)
        nn.init.zeros_(self.subpixel_projection.bias)

    def forward(
        self,
        object_features: torch.Tensor,
        stroke_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if object_features.ndim != 4 or object_features.shape[1:] != (
            G3_ATOM_CENTER_FIELD_CHANNELS,
            192,
            192,
        ):
            raise ValueError(
                "Virtual stride-2 residual object features must have shape "
                "[batch,256,192,192]"
            )
        if stroke_features.ndim != 4 or stroke_features.shape != (
            object_features.size(0),
            G3_STROKE_FEATURE_CHANNELS,
            192,
            192,
        ):
            raise ValueError(
                "Virtual stride-2 residual stroke features must have shape "
                "[batch,96,192,192]"
            )
        if (
            not object_features.is_floating_point()
            or not stroke_features.is_floating_point()
            or not bool(torch.isfinite(object_features).all())
            or not bool(torch.isfinite(stroke_features).all())
        ):
            raise ValueError(
                "Virtual stride-2 residual features must be finite floating point"
            )
        frozen_object = object_features.detach()
        frozen_stroke = stroke_features.detach()
        base_center_low_resolution = F.conv2d(
            frozen_object, self.base_center_weight, self.base_center_bias
        )
        base_center_logits = F.interpolate(
            base_center_low_resolution,
            scale_factor=2.0,
            mode="bilinear",
            align_corners=False,
        )
        hidden = self.activation(self.input_projection(frozen_stroke))
        hidden = self.activation(self.local(hidden))
        residual = self.pixel_shuffle(self.subpixel_projection(hidden))
        center_residual = residual[:, 0:1]
        offset_logits = residual[:, 1:3]
        center_logits = base_center_logits + center_residual
        return {
            "center_logits": center_logits,
            "offset_logits": offset_logits,
            "offsets": offset_logits.sigmoid(),
            "base_center_logits": base_center_logits,
            "base_center_low_resolution": base_center_low_resolution,
            "center_residual": center_residual,
            "offset_residual": offset_logits,
        }

    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )


class ContinuousStride2AtomCenterField(nn.Module):
    """Decode a continuous stride-2 feature map before predicting one center field."""

    def __init__(self, atom_head: nn.Module) -> None:
        super().__init__()
        presence, _ = _validate_p1a_atom_head(atom_head)
        self.register_buffer(
            "base_center_weight", presence.weight[0:1].detach().clone()
        )
        self.register_buffer(
            "base_center_bias", presence.bias[0:1].detach().clone()
        )
        self.subpixel_features = nn.Conv2d(
            G3_STROKE_FEATURE_CHANNELS, 128, kernel_size=1
        )
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.activation = nn.GELU()
        self.local = nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32)
        self.output_projection = nn.Conv2d(32, 3, kernel_size=1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        object_features: torch.Tensor,
        stroke_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if object_features.ndim != 4 or object_features.shape[1:] != (
            G3_ATOM_CENTER_FIELD_CHANNELS,
            192,
            192,
        ):
            raise ValueError(
                "Continuous stride-2 object features must have shape "
                "[batch,256,192,192]"
            )
        if stroke_features.ndim != 4 or stroke_features.shape != (
            object_features.size(0),
            G3_STROKE_FEATURE_CHANNELS,
            192,
            192,
        ):
            raise ValueError(
                "Continuous stride-2 stroke features must have shape "
                "[batch,96,192,192]"
            )
        if (
            not object_features.is_floating_point()
            or not stroke_features.is_floating_point()
            or not bool(torch.isfinite(object_features).all())
            or not bool(torch.isfinite(stroke_features).all())
        ):
            raise ValueError(
                "Continuous stride-2 features must be finite floating point"
            )
        base_center_low_resolution = F.conv2d(
            object_features.detach(), self.base_center_weight, self.base_center_bias
        )
        base_center_logits = F.interpolate(
            base_center_low_resolution,
            scale_factor=2.0,
            mode="bilinear",
            align_corners=False,
        )
        high_resolution = self.pixel_shuffle(self.subpixel_features(stroke_features))
        high_resolution = self.activation(high_resolution)
        high_resolution = self.activation(self.local(high_resolution))
        residual = self.output_projection(high_resolution)
        center_residual = residual[:, 0:1]
        offset_logits = residual[:, 1:3]
        center_logits = base_center_logits + center_residual
        return {
            "center_logits": center_logits,
            "offset_logits": offset_logits,
            "offsets": offset_logits.sigmoid(),
            "base_center_logits": base_center_logits,
            "base_center_low_resolution": base_center_low_resolution,
            "high_resolution_features": high_resolution,
            "center_residual": center_residual,
            "offset_residual": offset_logits,
        }

    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )


class ContinuousStride1AtomCenterField(nn.Module):
    """Decode continuous 384 and 768 feature maps before center prediction."""

    def __init__(self, atom_head: nn.Module) -> None:
        super().__init__()
        presence, _ = _validate_p1a_atom_head(atom_head)
        self.register_buffer(
            "base_center_weight", presence.weight[0:1].detach().clone()
        )
        self.register_buffer(
            "base_center_bias", presence.bias[0:1].detach().clone()
        )
        self.stride2_features = nn.Conv2d(
            G3_STROKE_FEATURE_CHANNELS, 128, kernel_size=1
        )
        self.stride2_shuffle = nn.PixelShuffle(2)
        self.stride2_local = nn.Conv2d(
            32, 32, kernel_size=3, padding=1, groups=32
        )
        self.stride1_features = nn.Conv2d(32, 128, kernel_size=1)
        self.stride1_shuffle = nn.PixelShuffle(2)
        self.stride1_local = nn.Conv2d(
            32, 32, kernel_size=3, padding=1, groups=32
        )
        self.activation = nn.GELU()
        self.output_projection = nn.Conv2d(32, 3, kernel_size=1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        object_features: torch.Tensor,
        stroke_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if object_features.ndim != 4 or object_features.shape[1:] != (
            G3_ATOM_CENTER_FIELD_CHANNELS,
            192,
            192,
        ):
            raise ValueError(
                "Continuous stride-1 object features must have shape "
                "[batch,256,192,192]"
            )
        if stroke_features.ndim != 4 or stroke_features.shape != (
            object_features.size(0),
            G3_STROKE_FEATURE_CHANNELS,
            192,
            192,
        ):
            raise ValueError(
                "Continuous stride-1 stroke features must have shape "
                "[batch,96,192,192]"
            )
        if (
            not object_features.is_floating_point()
            or not stroke_features.is_floating_point()
            or not bool(torch.isfinite(object_features).all())
            or not bool(torch.isfinite(stroke_features).all())
        ):
            raise ValueError(
                "Continuous stride-1 features must be finite floating point"
            )
        base_center_low_resolution = F.conv2d(
            object_features.detach(), self.base_center_weight, self.base_center_bias
        )
        base_center_logits = F.interpolate(
            base_center_low_resolution,
            scale_factor=4.0,
            mode="bilinear",
            align_corners=False,
        )
        stride2 = self.stride2_shuffle(self.stride2_features(stroke_features))
        stride2 = self.activation(stride2)
        stride2 = self.activation(self.stride2_local(stride2))
        stride1 = self.stride1_shuffle(self.stride1_features(stride2))
        stride1 = self.activation(stride1)
        stride1 = self.activation(self.stride1_local(stride1))
        residual = self.output_projection(stride1)
        center_residual = residual[:, 0:1]
        offset_logits = residual[:, 1:3]
        center_logits = base_center_logits + center_residual
        return {
            "center_logits": center_logits,
            "offset_logits": offset_logits,
            "offsets": offset_logits.sigmoid(),
            "base_center_logits": base_center_logits,
            "base_center_low_resolution": base_center_low_resolution,
            "stride2_features": stride2,
            "stride1_features": stride1,
            "center_residual": center_residual,
            "offset_residual": offset_logits,
        }

    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )


def _validate_p1a_atom_head(atom_head: nn.Module) -> tuple[nn.Conv2d, nn.Conv2d]:
    presence = getattr(atom_head, "presence", None)
    anchor_offset = getattr(atom_head, "anchor_offset", None)
    if not isinstance(presence, nn.Conv2d) or not isinstance(
        anchor_offset, nn.Conv2d
    ):
        raise ValueError("P1-A Atom head lacks convolutional presence/anchor_offset")
    if (
        presence.kernel_size != (1, 1)
        or anchor_offset.kernel_size != (1, 1)
        or presence.in_channels != G3_ATOM_CENTER_FIELD_CHANNELS
        or anchor_offset.in_channels != G3_ATOM_CENTER_FIELD_CHANNELS
        or presence.out_channels != 3
        or anchor_offset.out_channels != 6
        or presence.bias is None
        or anchor_offset.bias is None
    ):
        raise ValueError("P1-A Atom head dimensions differ from the three-slot protocol")
    return presence, anchor_offset


@torch.no_grad()
def warmstart_atom_center_field_from_p1a(
    field: G3AtomCenterField,
    atom_head: nn.Module,
) -> None:
    """Copy P1-A slot-0 presence and x/y offset heads into the center field."""

    presence, anchor_offset = _validate_p1a_atom_head(atom_head)
    field.center.weight.copy_(presence.weight[0:1])
    field.center.bias.copy_(presence.bias[0:1])
    offset_weight = anchor_offset.weight.reshape(
        3, 2, G3_ATOM_CENTER_FIELD_CHANNELS, 1, 1
    )
    offset_bias = anchor_offset.bias.reshape(3, 2)
    field.offset.weight.copy_(offset_weight[0])
    field.offset.bias.copy_(offset_bias[0])


def _p75_nearest_neighbor_cells(
    anchors: torch.Tensor,
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    if len(anchors) < 2:
        return anchors.new_tensor(10.0)
    coordinates = torch.stack(
        (anchors[:, 0] * float(width), anchors[:, 1] * float(height)), dim=-1
    )
    distances = torch.cdist(coordinates.float(), coordinates.float())
    distances.fill_diagonal_(float("inf"))
    nearest = distances.amin(dim=1).sort().values
    position = 0.75 * (len(nearest) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return nearest[lower]
    fraction = position - lower
    return nearest[lower] * (1.0 - fraction) + nearest[upper] * fraction


@torch.no_grad()
def build_atom_center_field_targets(
    atom_anchors: torch.Tensor,
    atom_mask: torch.Tensor,
    *,
    height: int,
    width: int,
    adaptive_gaussian: bool = False,
) -> dict[str, torch.Tensor]:
    """Build one-hot or scale-adaptive Gaussian center and offset targets.

    Adaptive sigma uses the order-independent p75 of per-Atom nearest-neighbor
    distance in output-cell units: ``clamp(p75 / 10, 1, 4)``. The default is a
    one-hot target for the first pilot. Multiple Atoms in one cell share one
    center target and use the lexicographically smallest anchor's offset.
    """

    if atom_anchors.ndim != 3 or atom_anchors.size(-1) != 2:
        raise ValueError("G3 AtomCenterField anchors must have shape [batch,atoms,2]")
    if atom_mask.shape != atom_anchors.shape[:2] or atom_mask.dtype != torch.bool:
        raise ValueError("G3 AtomCenterField anchor mask differs")
    if atom_mask.device != atom_anchors.device:
        raise ValueError("G3 AtomCenterField anchors and mask must share one device")
    if min(height, width) < 1:
        raise ValueError("G3 AtomCenterField target dimensions must be positive")
    valid_anchors = atom_anchors[atom_mask]
    if not atom_anchors.is_floating_point() or not bool(
        torch.isfinite(valid_anchors).all()
    ):
        raise ValueError("G3 AtomCenterField anchors must be finite floating point")
    if bool(((valid_anchors < 0.0) | (valid_anchors > 1.0)).any()):
        raise ValueError("G3 AtomCenterField anchors must be normalized")

    anchors = atom_anchors.detach().float()
    batch = len(anchors)
    heatmap = torch.zeros(batch, 1, height, width, device=anchors.device)
    offset_sum = torch.zeros(batch, 2, height, width, device=anchors.device)
    occupancy = torch.zeros(
        batch, 1, height, width, device=anchors.device, dtype=torch.long
    )
    sigma_cells = torch.zeros(batch, device=anchors.device)

    for batch_index in range(batch):
        indices = torch.where(atom_mask[batch_index])[0]
        current = anchors[batch_index, indices]
        if adaptive_gaussian:
            sigma = (
                _p75_nearest_neighbor_cells(
                    current, height=height, width=width
                )
                / 10.0
            ).clamp(1.0, 4.0)
            sigma_cells[batch_index] = sigma
        else:
            sigma = anchors.new_zeros(())
        ordered = sorted(
            range(len(current)),
            key=lambda index: tuple(current[index].cpu().tolist()),
        )
        for local_index in ordered:
            anchor = current[local_index]
            cell_x_float = (anchor[0] * width).clamp(0.0, width - 1e-4)
            cell_y_float = (anchor[1] * height).clamp(0.0, height - 1e-4)
            cell_x = int(cell_x_float)
            cell_y = int(cell_y_float)
            if occupancy[batch_index, 0, cell_y, cell_x] == 0:
                offset_sum[batch_index, 0, cell_y, cell_x] = (
                    cell_x_float - cell_x
                )
                offset_sum[batch_index, 1, cell_y, cell_x] = (
                    cell_y_float - cell_y
                )
            occupancy[batch_index, 0, cell_y, cell_x] += 1
            if not adaptive_gaussian:
                heatmap[batch_index, 0, cell_y, cell_x] = 1.0
                continue
            radius = int(math.ceil(3.0 * float(sigma)))
            left = max(0, cell_x - radius)
            right = min(width, cell_x + radius + 1)
            top = max(0, cell_y - radius)
            bottom = min(height, cell_y + radius + 1)
            grid_y, grid_x = torch.meshgrid(
                torch.arange(top, bottom, device=anchors.device),
                torch.arange(left, right, device=anchors.device),
                indexing="ij",
            )
            gaussian = torch.exp(
                -(
                    (grid_x.float() - cell_x).square()
                    + (grid_y.float() - cell_y).square()
                )
                / (2.0 * sigma.square())
            )
            heatmap[batch_index, 0, top:bottom, left:right] = torch.maximum(
                heatmap[batch_index, 0, top:bottom, left:right], gaussian
            )

    positive = occupancy > 0
    return {
        "heatmap": heatmap,
        "offsets": offset_sum,
        "positive": positive,
        "occupancy": occupancy,
        "sigma_cells": sigma_cells,
        "adaptive_gaussian": adaptive_gaussian,
    }


@torch.no_grad()
def decode_atom_center_field(
    output: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Expose all dense center cells without thresholding or maxima selection."""

    center_logits = output.get("center_logits")
    offset_logits = output.get("offset_logits")
    if (
        not isinstance(center_logits, torch.Tensor)
        or center_logits.ndim != 4
        or center_logits.size(1) != 1
    ):
        raise ValueError("G3 AtomCenterField center logits have invalid shape")
    if not isinstance(offset_logits, torch.Tensor) or offset_logits.shape != (
        center_logits.size(0),
        2,
        center_logits.size(2),
        center_logits.size(3),
    ):
        raise ValueError("G3 AtomCenterField offset logits have invalid shape")
    if center_logits.device != offset_logits.device or not bool(
        torch.isfinite(center_logits).all() & torch.isfinite(offset_logits).all()
    ):
        raise ValueError("G3 AtomCenterField dense outputs must be finite and colocated")
    batch, _, height, width = center_logits.shape
    offsets = offset_logits.detach().float().sigmoid()
    y, x = torch.meshgrid(
        torch.arange(height, device=center_logits.device, dtype=torch.float32),
        torch.arange(width, device=center_logits.device, dtype=torch.float32),
        indexing="ij",
    )
    anchors = torch.stack(
        (
            (x[None] + offsets[:, 0]) / float(width),
            (y[None] + offsets[:, 1]) / float(height),
        ),
        dim=1,
    )
    source_ids = torch.arange(
        height * width, device=center_logits.device, dtype=torch.long
    ).reshape(1, height, width).expand(batch, -1, -1)
    return {
        "scores": center_logits.detach().float().sigmoid()[:, 0],
        "anchors": anchors,
        "source_ids": source_ids,
    }


def atom_center_field_loss(
    output: dict[str, torch.Tensor],
    atom_anchors: torch.Tensor,
    atom_mask: torch.Tensor,
    *,
    adaptive_gaussian: bool = False,
    peak_margin_weight: float = 0.0,
    peak_margin: float = 1.0,
    global_peak_rank_weight: float = 0.0,
    global_peak_candidate_budget: int = 128,
    global_peak_boundary_window: int = 16,
    global_peak_margin: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Apply center focal loss and positive-cell sub-pixel offset loss."""

    if not math.isfinite(float(peak_margin_weight)) or peak_margin_weight < 0.0:
        raise ValueError("G3 AtomCenterField peak margin weight must be non-negative")
    if (
        not math.isfinite(float(global_peak_rank_weight))
        or global_peak_rank_weight < 0.0
    ):
        raise ValueError(
            "G3 AtomCenterField global peak rank weight must be non-negative"
        )

    center_logits = output.get("center_logits")
    offset_logits = output.get("offset_logits")
    if (
        not isinstance(center_logits, torch.Tensor)
        or center_logits.ndim != 4
        or center_logits.size(1) != 1
    ):
        raise ValueError("G3 AtomCenterField center logits have invalid shape")
    if not isinstance(offset_logits, torch.Tensor) or offset_logits.shape != (
        center_logits.size(0),
        2,
        center_logits.size(2),
        center_logits.size(3),
    ):
        raise ValueError("G3 AtomCenterField offset logits have invalid shape")
    targets = build_atom_center_field_targets(
        atom_anchors,
        atom_mask,
        height=center_logits.size(2),
        width=center_logits.size(3),
        adaptive_gaussian=adaptive_gaussian,
    )
    heatmap_loss = focal_heatmap_loss(center_logits, targets["heatmap"])
    positive = targets["positive"].expand(-1, 2, -1, -1)
    if bool(positive.any()):
        offset_loss = F.smooth_l1_loss(
            offset_logits.sigmoid()[positive], targets["offsets"][positive]
        )
    else:
        offset_loss = heatmap_loss * 0.0
    peak_margin_loss = (
        atom_center_peak_margin_loss(
            center_logits, targets["positive"], margin=peak_margin
        )
        if peak_margin_weight
        else heatmap_loss * 0.0
    )
    global_peak_rank_loss = (
        atom_center_global_peak_rank_loss(
            center_logits,
            targets["positive"],
            candidate_budget=global_peak_candidate_budget,
            boundary_window=global_peak_boundary_window,
            margin=global_peak_margin,
        )
        if global_peak_rank_weight
        else heatmap_loss * 0.0
    )
    return {
        "loss": (
            heatmap_loss
            + 8.0 * offset_loss
            + float(peak_margin_weight) * peak_margin_loss
            + float(global_peak_rank_weight) * global_peak_rank_loss
        ),
        "heatmap_loss": heatmap_loss,
        "offset_loss": offset_loss,
        "peak_margin_loss": peak_margin_loss,
        "global_peak_rank_loss": global_peak_rank_loss,
        "sigma_cells_mean": targets["sigma_cells"].mean(),
    }


def atom_center_peak_margin_loss(
    center_logits: torch.Tensor,
    positive_mask: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """Rank each GT center above its non-GT 8-neighborhood maximum.

    Positive neighbors are excluded from the competing set, so adjacent GT
    centers never penalize one another. Image padding is excluded as well.
    """

    if (
        center_logits.ndim != 4
        or center_logits.size(1) != 1
        or min(center_logits.shape[-2:]) < 1
    ):
        raise ValueError("G3 AtomCenterField center logits have invalid shape")
    if (
        positive_mask.shape != center_logits.shape
        or positive_mask.dtype != torch.bool
    ):
        raise ValueError("G3 AtomCenterField positive mask differs")
    if positive_mask.device != center_logits.device:
        raise ValueError(
            "G3 AtomCenterField logits and positive mask must share one device"
        )
    if not center_logits.is_floating_point() or not bool(
        torch.isfinite(center_logits).all()
    ):
        raise ValueError("G3 AtomCenterField center logits must be finite floating point")
    if not math.isfinite(float(margin)):
        raise ValueError("G3 AtomCenterField peak margin must be finite")

    batch, _, height, width = center_logits.shape
    logits = center_logits[:, 0]
    neighborhoods = F.unfold(
        F.pad(logits[:, None], (1, 1, 1, 1), value=-torch.inf),
        kernel_size=3,
    ).reshape(batch, 9, height, width)
    neighbor_positive = F.unfold(
        F.pad(positive_mask.float(), (1, 1, 1, 1), value=1.0),
        kernel_size=3,
    ).reshape(batch, 9, height, width).bool()
    valid_negative = ~neighbor_positive
    negative_max = neighborhoods.masked_fill(~valid_negative, -torch.inf).amax(dim=1)
    eligible = positive_mask[:, 0] & valid_negative.any(dim=1)
    if not bool(eligible.any()):
        return center_logits.sum() * 0.0
    return F.softplus(float(margin) + negative_max[eligible] - logits[eligible]).mean()


def atom_center_global_peak_rank_loss(
    center_logits: torch.Tensor,
    positive_mask: torch.Tensor,
    *,
    candidate_budget: int = 128,
    boundary_window: int = 16,
    margin: float = 1.0,
) -> torch.Tensor:
    """Rank GT centers above false regional maxima at the candidate boundary.

    For ``G`` occupied GT cells and a candidate budget ``K``, every GT center
    must outrank the ``K - G + 1``-th false maximum to fit in the top ``K``.
    Positions of false maxima are mined from detached logits, while scores at
    the selected positions are gathered from the live tensor and keep their
    gradients. The final window of false maxima above the boundary supplies a
    less brittle signal than one order statistic without pushing GT centers
    above every false maximum in the image.

    Equal-valued local plateaus can contribute more than one mining position.
    This is deliberate for the training-only approximation: it suppresses a
    plateau instead of introducing an inference-time pixel clustering rule.
    """

    if (
        center_logits.ndim != 4
        or center_logits.size(1) != 1
        or min(center_logits.shape[-2:]) < 1
    ):
        raise ValueError("G3 AtomCenterField center logits have invalid shape")
    if positive_mask.shape != center_logits.shape or positive_mask.dtype != torch.bool:
        raise ValueError("G3 AtomCenterField positive mask differs")
    if positive_mask.device != center_logits.device:
        raise ValueError("G3 AtomCenterField logits and positive mask must share one device")
    if not center_logits.is_floating_point() or not bool(
        torch.isfinite(center_logits).all()
    ):
        raise ValueError("G3 AtomCenterField center logits must be finite floating point")
    if not isinstance(candidate_budget, int) or isinstance(candidate_budget, bool):
        raise TypeError("G3 global peak candidate budget must be an integer")
    if candidate_budget < 1:
        raise ValueError("G3 global peak candidate budget must be positive")
    if not isinstance(boundary_window, int) or isinstance(boundary_window, bool):
        raise TypeError("G3 global peak boundary window must be an integer")
    if boundary_window < 1:
        raise ValueError("G3 global peak boundary window must be positive")
    if not math.isfinite(float(margin)):
        raise ValueError("G3 global peak margin must be finite")

    live_logits = center_logits[:, 0].float()
    selection_logits = live_logits.detach()
    regional_maxima = selection_logits == F.max_pool2d(
        selection_logits[:, None], kernel_size=3, stride=1, padding=1
    )[:, 0]
    false_maxima = regional_maxima & ~positive_mask[:, 0]

    image_losses: list[torch.Tensor] = []
    for image_index in range(center_logits.size(0)):
        positive_indices = torch.where(positive_mask[image_index, 0].flatten())[0]
        positive_count = int(positive_indices.numel())
        if positive_count == 0:
            continue

        boundary_index = max(candidate_budget - positive_count, 0)
        false_count = int(false_maxima[image_index].sum())
        if false_count <= boundary_index:
            continue

        detached_false_scores = selection_logits[image_index].flatten().masked_fill(
            ~false_maxima[image_index].flatten(), -torch.inf
        )
        ranked_false_indices = detached_false_scores.topk(
            boundary_index + 1, sorted=True
        ).indices
        window_start = max(0, len(ranked_false_indices) - boundary_window)
        hard_negative_indices = ranked_false_indices[window_start:]

        flat_live = live_logits[image_index].flatten()
        positive_scores = flat_live[positive_indices]
        hard_negative_scores = flat_live[hard_negative_indices]
        image_losses.append(
            F.softplus(
                float(margin)
                + hard_negative_scores[None, :]
                - positive_scores[:, None]
            ).mean()
        )

    if not image_losses:
        return center_logits.sum() * 0.0
    return torch.stack(image_losses).mean()


__all__ = [
    "G3_ATOM_CENTER_FIELD_CHANNELS",
    "G3_STROKE_FEATURE_CHANNELS",
    "ContinuousStride2AtomCenterField",
    "ContinuousStride1AtomCenterField",
    "G3AtomCenterField",
    "G3AtomCenterFieldHighResolutionResidual",
    "G3AtomCenterFieldLocalResidual",
    "VirtualStride2AtomCenterField",
    "VirtualStride2AtomCenterResidual",
    "atom_center_field_loss",
    "atom_center_global_peak_rank_loss",
    "atom_center_peak_margin_loss",
    "build_atom_center_field_targets",
    "decode_atom_center_field",
    "warmstart_atom_center_field_from_p1a",
]
