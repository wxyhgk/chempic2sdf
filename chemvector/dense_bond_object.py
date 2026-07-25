"""Spatially grounded dense Bond Object detection for ChemScene v0.8."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations

import torch
from torch import nn
from torch.nn import functional as F

from chemvector.bond_object import BOND_TYPES, DIRECTIONS
from chemvector.vertex import focal_heatmap_loss


@dataclass(frozen=True, slots=True)
class DenseBondObjectConfig:
    input_channels: int = 96
    hidden_channels: int = 128
    slots: int = 3
    max_endpoint_offset: float = 0.5

    def __post_init__(self) -> None:
        if min(self.input_channels, self.hidden_channels, self.slots) < 1:
            raise ValueError("Dense Bond Object dimensions must be positive")
        if not 0 < self.max_endpoint_offset <= 1:
            raise ValueError("max_endpoint_offset must be in (0, 1]")


class DilatedResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(1, channels),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            ),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.block(inputs)


class DenseBondObjectDetector(nn.Module):
    """Predict complete bond objects from spatially stable stride-4 cells."""

    def __init__(self, config: DenseBondObjectConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Sequential(
            nn.Conv2d(config.input_channels, config.hidden_channels, 3, padding=1),
            nn.GroupNorm(1, config.hidden_channels),
            nn.GELU(),
        )
        self.context = nn.Sequential(
            *(DilatedResidualBlock(config.hidden_channels, dilation) for dilation in (1, 2, 4, 8))
        )
        self.heatmap = nn.Conv2d(config.hidden_channels, config.slots, 1)
        self.center_offset = nn.Conv2d(config.hidden_channels, config.slots * 2, 1)
        self.orientation = nn.Conv2d(config.hidden_channels, config.slots * 2, 1)
        self.half_length = nn.Conv2d(config.hidden_channels, config.slots, 1)
        self.display = nn.Conv2d(
            config.hidden_channels, config.slots * len(BOND_TYPES), 1
        )
        self.chemical = nn.Conv2d(
            config.hidden_channels, config.slots * len(BOND_TYPES), 1
        )
        self.direction = nn.Conv2d(
            config.hidden_channels, config.slots * len(DIRECTIONS), 1
        )
        nn.init.constant_(self.heatmap.bias, -4.6)
        nn.init.constant_(self.half_length.bias, -1.8)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.context(self.input_projection(features))
        batch, _, height, width = hidden.shape
        slots = self.config.slots
        center_offsets = self.center_offset(hidden).reshape(batch, slots, 2, height, width).sigmoid()
        y, x = torch.meshgrid(
            torch.arange(height, device=hidden.device, dtype=hidden.dtype),
            torch.arange(width, device=hidden.device, dtype=hidden.dtype),
            indexing="ij",
        )
        centers = torch.stack(
            (
                (x[None, None] + center_offsets[:, :, 0]) / width,
                (y[None, None] + center_offsets[:, :, 1]) / height,
            ),
            dim=2,
        )
        doubled_orientation = F.normalize(
            self.orientation(hidden).reshape(batch, slots, 2, height, width),
            dim=2,
            eps=1e-6,
        )
        angle = 0.5 * torch.atan2(
            doubled_orientation[:, :, 1], doubled_orientation[:, :, 0]
        )
        direction = torch.stack((angle.cos(), angle.sin()), dim=2)
        half_length = (
            self.half_length(hidden).sigmoid() * self.config.max_endpoint_offset
        )
        delta = direction * half_length[:, :, None]
        endpoints = torch.stack((centers - delta, centers + delta), dim=2).clamp(0.0, 1.0)
        return {
            "object_features": hidden,
            "heatmap_logits": self.heatmap(hidden),
            "center_offsets": center_offsets,
            "centers": centers,
            "endpoints": endpoints,
            "doubled_orientation": doubled_orientation,
            "half_length": half_length,
            "display_logits": self.display(hidden).reshape(
                batch, slots, len(BOND_TYPES), height, width
            ),
            "chemical_logits": self.chemical(hidden).reshape(
                batch, slots, len(BOND_TYPES), height, width
            ),
            "direction_logits": self.direction(hidden).reshape(
                batch, slots, len(DIRECTIONS), height, width
            ),
        }


class MultiScaleDenseBondObjectDetector(nn.Module):
    """Add zero-initialized stride-8/16 context without changing the base output."""

    def __init__(
        self,
        config: DenseBondObjectConfig,
        *,
        stride8_channels: int = 192,
        stride16_channels: int = 384,
        preserve_base_types: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.preserve_base_types = preserve_base_types
        self.type_base: DenseBondObjectDetector | None = None
        self.base = DenseBondObjectDetector(config)
        self.stride8_adapter = self._adapter(stride8_channels, config.input_channels)
        self.stride16_adapter = self._adapter(stride16_channels, config.input_channels)

    @staticmethod
    def _adapter(input_channels: int, output_channels: int) -> nn.Sequential:
        adapter = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
        )
        nn.init.zeros_(adapter[-1].weight)
        nn.init.zeros_(adapter[-1].bias)
        return adapter

    def forward(
        self,
        stride4: torch.Tensor,
        stride8: torch.Tensor,
        stride16: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        size = stride4.shape[-2:]
        context8 = F.interpolate(
            self.stride8_adapter(stride8), size=size, mode="bilinear", align_corners=False
        )
        context16 = F.interpolate(
            self.stride16_adapter(stride16), size=size, mode="bilinear", align_corners=False
        )
        fused = self.base(stride4 + context8 + context16)
        if self.preserve_base_types or self.type_base is not None:
            base = (self.type_base or self.base)(stride4)
            for key in ("display_logits", "chemical_logits", "direction_logits"):
                fused[key] = base[key]
        return fused

    def install_frozen_type_base(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.type_base = DenseBondObjectDetector(self.config)
        self.type_base.load_state_dict(state_dict)
        self.type_base.requires_grad_(False)
        self.type_base.eval()

    def adapter_parameters(self) -> list[nn.Parameter]:
        return list(self.stride8_adapter.parameters()) + list(
            self.stride16_adapter.parameters()
        )


def build_dense_bond_targets(
    targets: dict[str, torch.Tensor],
    *,
    height: int,
    width: int,
    slots: int,
) -> dict[str, torch.Tensor]:
    device = targets["endpoints"].device
    heatmap = torch.zeros(
        (targets["endpoints"].size(0), slots, height, width), device=device
    )
    center_offsets = torch.zeros(
        (targets["endpoints"].size(0), slots, 2, height, width), device=device
    )
    endpoints = torch.zeros(
        (targets["endpoints"].size(0), slots, 2, 2, height, width), device=device
    )
    display = torch.zeros(
        (targets["endpoints"].size(0), slots, height, width),
        device=device,
        dtype=torch.long,
    )
    chemical = torch.zeros_like(display)
    direction = torch.zeros_like(display)
    doubled_orientation = torch.zeros(
        (targets["endpoints"].size(0), slots, 2, height, width), device=device
    )
    half_length = torch.zeros_like(heatmap)
    positive = torch.zeros_like(heatmap, dtype=torch.bool)
    occupancy = torch.zeros(
        (targets["endpoints"].size(0), height, width),
        device=device,
        dtype=torch.long,
    )
    for batch_index in range(targets["endpoints"].size(0)):
        valid_indices = torch.where(targets["mask"][batch_index])[0]
        cells: dict[tuple[int, int], list[int]] = {}
        for target_index in valid_indices.tolist():
            center = targets["endpoints"][batch_index, target_index].mean(dim=0)
            cell_x = min(width - 1, max(0, int(float(center[0]) * width)))
            cell_y = min(height - 1, max(0, int(float(center[1]) * height)))
            cells.setdefault((cell_y, cell_x), []).append(target_index)
        for (cell_y, cell_x), indices in cells.items():
            occupancy[batch_index, cell_y, cell_x] = len(indices)
            indices.sort(
                key=lambda index: tuple(
                    targets["endpoints"][batch_index, index].flatten().tolist()
                )
            )
            if len(indices) > slots:
                raise ValueError(
                    f"Dense Bond Object slot capacity {slots} is smaller than "
                    f"{len(indices)} bonds in cell ({cell_x}, {cell_y})"
                )
            for slot, target_index in enumerate(indices):
                target_endpoints = targets["endpoints"][batch_index, target_index]
                center = target_endpoints.mean(dim=0)
                heatmap[batch_index, slot, cell_y, cell_x] = 1.0
                positive[batch_index, slot, cell_y, cell_x] = True
                center_offsets[batch_index, slot, :, cell_y, cell_x] = torch.stack(
                    (center[0] * width - cell_x, center[1] * height - cell_y)
                )
                endpoints[batch_index, slot, :, :, cell_y, cell_x] = target_endpoints
                delta = target_endpoints[1] - target_endpoints[0]
                length = torch.linalg.vector_norm(delta).clamp_min(1e-8)
                unit = delta / length
                doubled_orientation[batch_index, slot, :, cell_y, cell_x] = torch.stack(
                    (unit[0].square() - unit[1].square(), 2.0 * unit[0] * unit[1])
                )
                half_length[batch_index, slot, cell_y, cell_x] = length / 2.0
                display[batch_index, slot, cell_y, cell_x] = targets["display"][
                    batch_index, target_index
                ]
                chemical[batch_index, slot, cell_y, cell_x] = targets["chemical"][
                    batch_index, target_index
                ]
                direction[batch_index, slot, cell_y, cell_x] = targets["direction"][
                    batch_index, target_index
                ]
    return {
        "heatmap": heatmap,
        "center_offsets": center_offsets,
        "endpoints": endpoints,
        "display": display,
        "chemical": chemical,
        "direction": direction,
        "doubled_orientation": doubled_orientation,
        "half_length": half_length,
        "positive": positive,
        "occupancy": occupancy,
    }


@lru_cache(maxsize=None)
def _injective_slot_permutations(slots: int, targets: int) -> torch.Tensor:
    if min(slots, targets) < 1 or targets > slots:
        raise ValueError("Slot assignment dimensions are invalid")
    return torch.tensor(tuple(permutations(range(slots), targets)), dtype=torch.long)


def _unordered_endpoint_l1_cost(
    predicted: torch.Tensor, expected: torch.Tensor
) -> torch.Tensor:
    forward = (
        predicted[:, None] - expected[None]
    ).abs().sum(dim=(-1, -2))
    reverse = (
        predicted[:, None] - expected.flip(1)[None]
    ).abs().sum(dim=(-1, -2))
    return torch.minimum(forward, reverse)


def align_dense_bond_targets_to_output(
    output: dict[str, torch.Tensor],
    dense: dict[str, torch.Tensor],
    *,
    slot_matching: str,
) -> dict[str, torch.Tensor]:
    """Assign unordered same-cell GT Bonds to prediction slots inside the loss."""

    if slot_matching == "fixed":
        return dense
    if slot_matching != "hungarian":
        raise ValueError(f"Unknown dense Bond slot matching mode: {slot_matching}")

    slots = int(output["heatmap_logits"].size(1))
    aligned = {
        "heatmap": torch.zeros_like(dense["heatmap"]),
        "center_offsets": torch.zeros_like(dense["center_offsets"]),
        "endpoints": torch.zeros_like(dense["endpoints"]),
        "display": torch.zeros_like(dense["display"]),
        "chemical": torch.zeros_like(dense["chemical"]),
        "direction": torch.zeros_like(dense["direction"]),
        "doubled_orientation": torch.zeros_like(dense["doubled_orientation"]),
        "half_length": torch.zeros_like(dense["half_length"]),
        "positive": torch.zeros_like(dense["positive"]),
        "occupancy": dense["occupancy"],
    }
    occupied = torch.where(dense["occupancy"] > 0)
    for batch_index, cell_y, cell_x in zip(*(item.tolist() for item in occupied)):
        count = int(dense["occupancy"][batch_index, cell_y, cell_x])
        if count > slots:
            raise ValueError("Dense Bond occupancy exceeds detector slot count")
        expected_endpoints = dense["endpoints"][
            batch_index, :count, :, :, cell_y, cell_x
        ]
        predicted_endpoints = output["endpoints"][
            batch_index, :, :, :, cell_y, cell_x
        ]
        endpoint_cost = _unordered_endpoint_l1_cost(
            predicted_endpoints.float(), expected_endpoints.float()
        )
        predicted_centers = output["center_offsets"][
            batch_index, :, :, cell_y, cell_x
        ].float()
        expected_centers = dense["center_offsets"][
            batch_index, :count, :, cell_y, cell_x
        ].float()
        center_cost = (
            predicted_centers[:, None] - expected_centers[None]
        ).abs().sum(-1)
        predicted_orientation = output["doubled_orientation"][
            batch_index, :, :, cell_y, cell_x
        ].float()
        expected_orientation = dense["doubled_orientation"][
            batch_index, :count, :, cell_y, cell_x
        ].float()
        orientation_cost = 1.0 - torch.einsum(
            "sd,kd->sk", predicted_orientation, expected_orientation
        )
        predicted_length = output["half_length"][
            batch_index, :, cell_y, cell_x
        ].float()
        expected_length = dense["half_length"][
            batch_index, :count, cell_y, cell_x
        ].float()
        length_cost = (
            predicted_length[:, None] - expected_length[None]
        ).abs()
        cost = (
            8.0 * endpoint_cost
            + 2.0 * center_cost
            + 2.0 * orientation_cost
            + 8.0 * length_cost
        ).detach()
        assignments = _injective_slot_permutations(slots, count).to(cost.device)
        target_indices = torch.arange(count, device=cost.device)
        assignment_costs = cost[assignments, target_indices].sum(-1)
        selected_slots = assignments[assignment_costs.argmin()]

        aligned["heatmap"][batch_index, selected_slots, cell_y, cell_x] = 1.0
        aligned["positive"][batch_index, selected_slots, cell_y, cell_x] = True
        for target_index, slot in enumerate(selected_slots.tolist()):
            for name in (
                "center_offsets",
                "endpoints",
                "display",
                "chemical",
                "direction",
                "doubled_orientation",
                "half_length",
            ):
                aligned[name][batch_index, slot, ..., cell_y, cell_x] = dense[name][
                    batch_index, target_index, ..., cell_y, cell_x
                ]
    return aligned


def dense_slot_duplicate_loss(
    output: dict[str, torch.Tensor],
    positive: torch.Tensor,
    occupancy: torch.Tensor,
    *,
    image_size: float,
    sigma_pixels: float,
) -> torch.Tensor:
    """Suppress same-cell duplicate slots without repelling two matched GT Bonds."""

    if sigma_pixels <= 0.0:
        raise ValueError("Dense slot duplicate sigma must be positive")
    probabilities = output["heatmap_logits"].float().sigmoid()
    endpoints = output["endpoints"].float()
    occupied = occupancy > 0
    losses = []
    for first in range(probabilities.size(1)):
        for second in range(first + 1, probabilities.size(1)):
            eligible = occupied & ~(positive[:, first] & positive[:, second])
            if not bool(eligible.any()):
                continue
            first_endpoints = endpoints[:, first]
            second_endpoints = endpoints[:, second]
            forward = torch.linalg.vector_norm(
                (first_endpoints - second_endpoints) * image_size, dim=2
            ).sum(1)
            reverse = torch.linalg.vector_norm(
                (first_endpoints - second_endpoints.flip(1)) * image_size,
                dim=2,
            ).sum(1)
            similarity = torch.exp(
                -torch.minimum(forward, reverse).detach() / sigma_pixels
            )
            pair_mass = probabilities[:, first] * probabilities[:, second]
            losses.append((pair_mass * similarity)[eligible])
    if not losses:
        return probabilities.sum() * 0.0
    return torch.cat(losses).mean()


def dense_bond_object_loss(
    output: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    endpoint_barrier_weight: float = 0.02,
    endpoint_barrier_margin_pixels: float = 4.0,
    heatmap_set_weight: float = 0.0,
    heatmap_set_margin: float = 1.0,
    slot_matching: str = "fixed",
    cell_count_weight: float = 0.0,
    duplicate_weight: float = 0.0,
    duplicate_sigma_pixels: float = 8.0,
    image_size: float = 768.0,
) -> dict[str, torch.Tensor]:
    height, width = output["heatmap_logits"].shape[-2:]
    dense = build_dense_bond_targets(
        targets,
        height=height,
        width=width,
        slots=output["heatmap_logits"].size(1),
    )
    dense = align_dense_bond_targets_to_output(
        output, dense, slot_matching=slot_matching
    )
    positive = dense["positive"]
    heatmap_loss = focal_heatmap_loss(output["heatmap_logits"], dense["heatmap"])
    zero = heatmap_loss * 0.0
    set_losses = []
    expanded_positive = F.max_pool2d(
        positive.float(), kernel_size=3, stride=1, padding=1
    ).bool()
    for batch_index in range(positive.size(0)):
        positive_logits = output["heatmap_logits"][batch_index][positive[batch_index]]
        negative_logits = output["heatmap_logits"][batch_index][
            ~expanded_positive[batch_index]
        ]
        if len(positive_logits) and len(negative_logits):
            set_losses.append(
                F.relu(
                    heatmap_set_margin
                    - positive_logits.min()
                    + negative_logits.max()
                )
            )
    heatmap_set_loss = torch.stack(set_losses).mean() if set_losses else zero
    occupied = dense["occupancy"] > 0
    if bool(occupied.any()):
        cell_count_loss = F.smooth_l1_loss(
            output["heatmap_logits"].float().sigmoid().sum(1)[occupied],
            dense["occupancy"][occupied].float(),
        )
    else:
        cell_count_loss = zero
    duplicate_loss = (
        dense_slot_duplicate_loss(
            output,
            positive,
            dense["occupancy"],
            image_size=image_size,
            sigma_pixels=duplicate_sigma_pixels,
        )
        if duplicate_weight > 0.0
        else zero
    )
    if positive.any():
        predicted_centers = output["center_offsets"].permute(0, 1, 3, 4, 2)[positive]
        expected_centers = dense["center_offsets"].permute(0, 1, 3, 4, 2)[positive]
        center_loss = F.smooth_l1_loss(predicted_centers, expected_centers)
        predicted_orientation = output["doubled_orientation"].permute(
            0, 1, 3, 4, 2
        )[positive]
        expected_orientation = dense["doubled_orientation"].permute(
            0, 1, 3, 4, 2
        )[positive]
        orientation_loss = (1.0 - (predicted_orientation * expected_orientation).sum(-1)).mean()
        predicted_length = output["half_length"][positive]
        expected_length = dense["half_length"][positive]
        length_loss = F.smooth_l1_loss(predicted_length, expected_length)
        predicted_endpoints = output["endpoints"].permute(0, 1, 4, 5, 2, 3)[positive]
        expected_endpoints = dense["endpoints"].permute(0, 1, 4, 5, 2, 3)[positive]
        forward = (predicted_endpoints - expected_endpoints).abs().sum(dim=(-1, -2))
        reverse = (
            predicted_endpoints - expected_endpoints.flip(1)
        ).abs().sum(dim=(-1, -2))
        use_reverse = (reverse < forward)[:, None, None]
        aligned = torch.where(use_reverse, expected_endpoints.flip(1), expected_endpoints)
        endpoint_loss = torch.minimum(forward, reverse).mean()
        distances = torch.linalg.vector_norm(
            (predicted_endpoints - aligned) * image_size, dim=-1
        )
        barrier_loss = F.relu(
            distances - endpoint_barrier_margin_pixels
        ).square().mean()
        display_logits = output["display_logits"].permute(0, 1, 3, 4, 2)[positive]
        chemical_logits = output["chemical_logits"].permute(0, 1, 3, 4, 2)[positive]
        direction_logits = output["direction_logits"].permute(0, 1, 3, 4, 2)[positive]
        display_loss = F.cross_entropy(display_logits, dense["display"][positive])
        chemical_loss = F.cross_entropy(chemical_logits, dense["chemical"][positive])
        direction_loss = F.cross_entropy(direction_logits, dense["direction"][positive])
    else:
        center_loss = orientation_loss = length_loss = endpoint_loss = barrier_loss = zero
        display_loss = chemical_loss = direction_loss = zero
    return {
        "loss": (
            heatmap_loss
            + heatmap_set_weight * heatmap_set_loss
            + cell_count_weight * cell_count_loss
            + duplicate_weight * duplicate_loss
            + 4.0 * center_loss
            + 2.0 * orientation_loss
            + 8.0 * length_loss
            + 8.0 * endpoint_loss
            + endpoint_barrier_weight * barrier_loss
            + display_loss
            + chemical_loss
            + 0.5 * direction_loss
        ),
        "heatmap_loss": heatmap_loss,
        "heatmap_set_loss": heatmap_set_loss,
        "cell_count_loss": cell_count_loss,
        "duplicate_loss": duplicate_loss,
        "center_loss": center_loss,
        "orientation_loss": orientation_loss,
        "length_loss": length_loss,
        "endpoint_loss": endpoint_loss,
        "endpoint_barrier_loss": barrier_loss,
        "display_loss": display_loss,
        "chemical_loss": chemical_loss,
        "direction_loss": direction_loss,
    }


@torch.no_grad()
def decode_dense_bond_objects(
    output: dict[str, torch.Tensor],
    *,
    threshold: float = 0.3,
    maximum_objects: int = 128,
) -> list[list[dict[str, torch.Tensor | float | int]]]:
    scores = output["heatmap_logits"].sigmoid()
    peaks = scores == F.max_pool2d(scores, 3, stride=1, padding=1)
    scores = scores * peaks
    batch, slots, height, width = scores.shape
    decoded = []
    for batch_index in range(batch):
        values, flat_indices = scores[batch_index].flatten().topk(
            min(maximum_objects, slots * height * width)
        )
        valid = values >= threshold
        values = values[valid]
        flat_indices = flat_indices[valid]
        spatial_size = height * width
        feature_rows = (
            output["object_features"][batch_index]
            .permute(1, 2, 0)
            .reshape(spatial_size, -1)
        )
        spatial_indices = flat_indices.remainder(spatial_size)
        selected_features = feature_rows[spatial_indices]
        selected_endpoints = (
            output["endpoints"][batch_index]
            .permute(0, 3, 4, 1, 2)
            .reshape(slots * spatial_size, 2, 2)[flat_indices]
        )
        selected_display = (
            output["display_logits"][batch_index]
            .permute(0, 2, 3, 1)
            .reshape(slots * spatial_size, -1)[flat_indices]
            .argmax(-1)
        )
        selected_chemical = (
            output["chemical_logits"][batch_index]
            .permute(0, 2, 3, 1)
            .reshape(slots * spatial_size, -1)[flat_indices]
            .argmax(-1)
        )
        selected_direction = (
            output["direction_logits"][batch_index]
            .permute(0, 2, 3, 1)
            .reshape(slots * spatial_size, -1)[flat_indices]
            .argmax(-1)
        )
        score_values = values.float().cpu().tolist()
        display_values = selected_display.cpu().tolist()
        chemical_values = selected_chemical.cpu().tolist()
        direction_values = selected_direction.cpu().tolist()
        objects = []
        for score, feature, endpoints, display, chemical, direction in zip(
            score_values,
            selected_features.unbind(0),
            selected_endpoints.unbind(0),
            display_values,
            chemical_values,
            direction_values,
        ):
            objects.append(
                {
                    "score": score,
                    "feature": feature,
                    "endpoints": endpoints,
                    "display": display,
                    "chemical": chemical,
                    "direction": direction,
                }
            )
        decoded.append(objects)
    return decoded
