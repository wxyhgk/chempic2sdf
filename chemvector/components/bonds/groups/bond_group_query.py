"""Proposal-conditioned group queries for complete Bond Objects."""

from __future__ import annotations

from dataclasses import dataclass

from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
from torch.nn import functional as F

from chemvector.atom_query import poisson_binomial_log_probs, position_encoding_2d
from chemvector.bond_object import BOND_TYPES, DIRECTIONS, unordered_endpoint_cost


@dataclass(frozen=True, slots=True)
class BondGroupQueryConfig:
    input_channels: int = 96
    proposal_channels: int = 128
    d_model: int = 192
    queries: int = 128
    layers: int = 2
    heads: int = 6
    ffn_dim: int = 768
    memory_stride: int = 4
    endpoint_delta_scale: float = 1.0
    endpoint_local_channels: int = 0
    endpoint_patch_radius_cells: int = 2
    endpoint_line_samples: int = 0
    endpoint_line_radius_cells: int = 2
    endpoint_line_feature_dim: int = 64
    endpoint_ink_line_samples: int = 0
    endpoint_ink_line_band: int = 25
    endpoint_ink_line_radius_pixels: float = 12.0
    endpoint_ink_line_feature_dim: int = 64
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if min(
            self.input_channels,
            self.proposal_channels,
            self.d_model,
            self.queries,
            self.layers,
            self.heads,
            self.ffn_dim,
            self.memory_stride,
        ) < 1:
            raise ValueError("Bond Group Query dimensions must be positive")
        if self.d_model % self.heads:
            raise ValueError("Bond Group Query heads must divide d_model")
        if self.d_model % 4:
            raise ValueError("Bond Group Query d_model must be divisible by four")
        if self.endpoint_delta_scale <= 0.0:
            raise ValueError("Bond Group Query endpoint delta scale must be positive")
        if self.endpoint_local_channels < 0:
            raise ValueError("Bond Group Query local endpoint channels must be nonnegative")
        if self.endpoint_patch_radius_cells < 0:
            raise ValueError("Bond Group Query endpoint patch radius must be nonnegative")
        if self.endpoint_line_samples < 0 or self.endpoint_line_samples == 1:
            raise ValueError("Bond Group Query line samples must be zero or at least two")
        if self.endpoint_line_radius_cells < 0:
            raise ValueError("Bond Group Query line radius must be nonnegative")
        if self.endpoint_line_feature_dim < 1:
            raise ValueError("Bond Group Query line feature dimension must be positive")
        if self.endpoint_line_samples and not self.endpoint_local_channels:
            raise ValueError("Bond Group Query line bands require endpoint-local features")
        if self.endpoint_ink_line_samples < 0 or self.endpoint_ink_line_samples == 1:
            raise ValueError("Bond Group Query ink-line samples must be zero or at least two")
        if self.endpoint_ink_line_band < 1:
            raise ValueError("Bond Group Query ink-line width must be positive")
        if self.endpoint_ink_line_radius_pixels <= 0.0:
            raise ValueError("Bond Group Query ink-line radius must be positive")
        if self.endpoint_ink_line_feature_dim < 1:
            raise ValueError("Bond Group Query ink-line feature dimension must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("Bond Group Query dropout must be in [0, 1)")


def sample_proposal_endpoint_patches(
    feature_map: torch.Tensor,
    endpoints: torch.Tensor,
    mask: torch.Tensor,
    *,
    radius_cells: int,
) -> torch.Tensor:
    """Sample an axis-aligned local feature patch around each proposal endpoint."""

    if feature_map.ndim != 4:
        raise ValueError("Endpoint feature map must have shape [batch, channels, h, w]")
    if endpoints.ndim != 4 or endpoints.shape[-2:] != (2, 2):
        raise ValueError("Bond endpoints must have shape [batch, queries, 2, 2]")
    if mask.shape != endpoints.shape[:2]:
        raise ValueError("Bond endpoint patch mask must match batch and query dimensions")
    if feature_map.size(0) != endpoints.size(0):
        raise ValueError("Endpoint feature map and proposals have different batches")
    if radius_cells < 0:
        raise ValueError("Endpoint patch radius must be nonnegative")
    if not bool(torch.isfinite(endpoints).all()):
        raise ValueError("Bond endpoint coordinates must be finite")

    batch, channels, height, width = feature_map.shape
    queries = endpoints.size(1)
    patch_size = 2 * radius_cells + 1
    offsets_y = torch.arange(
        -radius_cells,
        radius_cells + 1,
        device=endpoints.device,
        dtype=endpoints.dtype,
    ) / float(height)
    offsets_x = torch.arange(
        -radius_cells,
        radius_cells + 1,
        device=endpoints.device,
        dtype=endpoints.dtype,
    ) / float(width)
    grid_y, grid_x = torch.meshgrid(offsets_y, offsets_x, indexing="ij")
    offsets = torch.stack((grid_x, grid_y), dim=-1)
    grid = endpoints[..., None, None, :] + offsets
    grid = grid.clamp(0.0, 1.0).mul(2.0).sub(1.0)
    grid = grid.to(feature_map.dtype).reshape(
        batch, queries * 2 * patch_size, patch_size, 2
    )
    sampled = F.grid_sample(
        feature_map,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    sampled = sampled.reshape(
        batch, channels, queries, 2, patch_size, patch_size
    ).permute(0, 2, 3, 1, 4, 5)
    return sampled * mask[:, :, None, None, None, None].to(sampled.dtype)


def sample_proposal_line_bands(
    feature_map: torch.Tensor,
    endpoints: torch.Tensor,
    mask: torch.Tensor,
    *,
    line_samples: int,
    radius_cells: int,
) -> torch.Tensor:
    """Sample a direction-aligned feature band along every proposal segment."""

    if feature_map.ndim != 4:
        raise ValueError("Line-band feature map must have shape [batch, channels, h, w]")
    if endpoints.ndim != 4 or endpoints.shape[-2:] != (2, 2):
        raise ValueError("Bond endpoints must have shape [batch, queries, 2, 2]")
    if mask.shape != endpoints.shape[:2]:
        raise ValueError("Bond line-band mask must match batch and query dimensions")
    if feature_map.size(0) != endpoints.size(0):
        raise ValueError("Line-band feature map and proposals have different batches")
    if line_samples < 2 or radius_cells < 0:
        raise ValueError("Bond line-band dimensions are invalid")
    if not bool(torch.isfinite(endpoints).all()):
        raise ValueError("Bond endpoint coordinates must be finite")

    batch, channels, height, width = feature_map.shape
    queries = endpoints.size(1)
    band_samples = 2 * radius_cells + 1
    begin = endpoints[:, :, 0]
    delta = endpoints[:, :, 1] - begin
    delta_pixels = torch.stack(
        (delta[..., 0] * width, delta[..., 1] * height), dim=-1
    )
    direction = delta_pixels / delta_pixels.norm(dim=-1, keepdim=True).clamp_min(
        1e-6
    )
    perpendicular = torch.stack((-direction[..., 1], direction[..., 0]), dim=-1)
    perpendicular = torch.stack(
        (perpendicular[..., 0] / width, perpendicular[..., 1] / height), dim=-1
    )
    positions = torch.linspace(
        0.0,
        1.0,
        line_samples,
        device=endpoints.device,
        dtype=endpoints.dtype,
    )
    offsets = torch.arange(
        -radius_cells,
        radius_cells + 1,
        device=endpoints.device,
        dtype=endpoints.dtype,
    )
    center_line = begin[:, :, None, :] + delta[:, :, None, :] * positions[
        None, None, :, None
    ]
    grid = center_line[:, :, None, :, :] + (
        perpendicular[:, :, None, None, :]
        * offsets[None, None, :, None, None]
    )
    grid = grid.clamp(0.0, 1.0).mul(2.0).sub(1.0)
    grid = grid.to(feature_map.dtype).reshape(
        batch, queries * band_samples, line_samples, 2
    )
    sampled = F.grid_sample(
        feature_map,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    sampled = sampled.reshape(
        batch, channels, queries, band_samples, line_samples
    ).permute(0, 2, 1, 3, 4)
    return sampled * mask[:, :, None, None, None].to(sampled.dtype)


def sample_proposal_ink_line_views(
    pixel_values: torch.Tensor,
    endpoints: torch.Tensor,
    mask: torch.Tensor,
    *,
    line_samples: int,
    band_samples: int,
    radius_pixels: float,
) -> torch.Tensor:
    """Sample raw-ink line bands in endpoint-oriented forward/reverse views."""

    if pixel_values.ndim != 4 or pixel_values.size(1) not in (1, 3):
        raise ValueError("Raw line-band pixels must be BCHW grayscale or RGB")
    if endpoints.ndim != 4 or endpoints.shape[-2:] != (2, 2):
        raise ValueError("Bond endpoints must have shape [batch, queries, 2, 2]")
    if mask.shape != endpoints.shape[:2]:
        raise ValueError("Bond raw line-band mask must match batch and query dimensions")
    if pixel_values.size(0) != endpoints.size(0):
        raise ValueError("Raw line-band pixels and proposals have different batches")
    if line_samples < 2 or band_samples < 1 or radius_pixels <= 0.0:
        raise ValueError("Bond raw line-band dimensions are invalid")
    if not bool(torch.isfinite(endpoints).all()):
        raise ValueError("Bond endpoint coordinates must be finite")

    batch, _, height, width = pixel_values.shape
    queries = endpoints.size(1)
    begin = endpoints[:, :, 0]
    delta = endpoints[:, :, 1] - begin
    delta_pixels = torch.stack(
        (delta[..., 0] * width, delta[..., 1] * height), dim=-1
    )
    direction = delta_pixels / delta_pixels.norm(dim=-1, keepdim=True).clamp_min(
        1e-6
    )
    perpendicular = torch.stack((-direction[..., 1], direction[..., 0]), dim=-1)
    perpendicular = torch.stack(
        (perpendicular[..., 0] / width, perpendicular[..., 1] / height), dim=-1
    )
    positions = torch.linspace(
        0.04,
        0.96,
        line_samples,
        device=endpoints.device,
        dtype=endpoints.dtype,
    )
    offsets = torch.linspace(
        -radius_pixels,
        radius_pixels,
        band_samples,
        device=endpoints.device,
        dtype=endpoints.dtype,
    )
    center_line = begin[:, :, None, :] + delta[:, :, None, :] * positions[
        None, None, :, None
    ]
    grid = center_line[:, :, None, :, :] + (
        perpendicular[:, :, None, None, :]
        * offsets[None, None, :, None, None]
    )
    grid = grid.mul(2.0).sub(1.0).to(pixel_values.dtype).reshape(
        batch, queries * band_samples, line_samples, 2
    )
    if pixel_values.size(1) == 3:
        ink = 1.0 - pixel_values.add(1.0).mul(0.5).mean(dim=1, keepdim=True)
    else:
        ink = 1.0 - pixel_values.add(1.0).mul(0.5)
    sampled = F.grid_sample(
        ink,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).reshape(batch, queries, 1, band_samples, line_samples)
    sampled = sampled * mask[:, :, None, None, None].to(sampled.dtype)
    return torch.stack((sampled, sampled.flip((-2, -1))), dim=2)


def _flatten_slot_spatial(tensor: torch.Tensor) -> torch.Tensor:
    """Move slot and spatial dimensions into one dense-candidate dimension."""

    if tensor.ndim < 4:
        raise ValueError("Slot-spatial tensors must have at least four dimensions")
    batch, slots = tensor.shape[:2]
    height, width = tensor.shape[-2:]
    middle = tensor.shape[2:-2]
    order = (0, 1, tensor.ndim - 2, tensor.ndim - 1, *range(2, tensor.ndim - 2))
    return tensor.permute(order).reshape(batch, slots * height * width, *middle)


def _gather_candidates(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    flattened = _flatten_slot_spatial(tensor)
    if flattened.ndim == 2:
        return flattened.gather(1, indices)
    gather_indices = indices.reshape(
        indices.size(0), indices.size(1), *([1] * (flattened.ndim - 2))
    ).expand(indices.size(0), indices.size(1), *flattened.shape[2:])
    return flattened.gather(1, gather_indices)


def extract_dense_bond_proposals(
    dense_output: dict[str, torch.Tensor],
    *,
    maximum_queries: int = 128,
) -> dict[str, torch.Tensor]:
    """Extract the same per-slot 3x3 peaks and global ranking as dense decoding."""

    if maximum_queries < 1:
        raise ValueError("Bond proposal count must be positive")
    required = {
        "heatmap_logits",
        "object_features",
        "endpoints",
        "display_logits",
        "chemical_logits",
        "direction_logits",
    }
    missing = sorted(required - dense_output.keys())
    if missing:
        raise ValueError(f"Dense Bond output is missing fields: {missing}")

    heatmap_logits = dense_output["heatmap_logits"]
    if heatmap_logits.ndim != 4:
        raise ValueError("Dense Bond heatmap must have shape [batch, slots, h, w]")
    batch, slots, height, width = heatmap_logits.shape
    expected_shapes = {
        "endpoints": (batch, slots, 2, 2, height, width),
        "display_logits": (batch, slots, len(BOND_TYPES), height, width),
        "chemical_logits": (batch, slots, len(BOND_TYPES), height, width),
        "direction_logits": (batch, slots, len(DIRECTIONS), height, width),
    }
    for name, expected in expected_shapes.items():
        if dense_output[name].shape != expected:
            raise ValueError(
                f"Dense Bond {name} has shape {tuple(dense_output[name].shape)}, "
                f"expected {expected}"
            )
    object_features = dense_output["object_features"]
    if object_features.ndim != 4 or object_features.shape[0] != batch:
        raise ValueError(
            "Dense Bond object features must have shape [batch, channels, h, w]"
        )
    if object_features.shape[-2:] != (height, width):
        raise ValueError("Dense Bond object features and heatmap spatial sizes differ")

    scores = heatmap_logits.sigmoid()
    peaks = scores == F.max_pool2d(scores, kernel_size=3, stride=1, padding=1)
    ranked_scores = scores * peaks
    query_count = min(maximum_queries, slots * height * width)
    values, flat_indices = ranked_scores.flatten(1).topk(query_count, dim=1)

    raw_logits = heatmap_logits.flatten(1).gather(1, flat_indices)
    selected_peaks = peaks.flatten(1).gather(1, flat_indices)
    base_logits = torch.where(
        selected_peaks,
        raw_logits,
        torch.full_like(raw_logits, -30.0),
    )
    endpoints = _gather_candidates(dense_output["endpoints"], flat_indices)
    display_logits = _gather_candidates(
        dense_output["display_logits"], flat_indices
    )
    chemical_logits = _gather_candidates(
        dense_output["chemical_logits"], flat_indices
    )
    direction_logits = _gather_candidates(
        dense_output["direction_logits"], flat_indices
    )

    spatial_indices = flat_indices.remainder(height * width)
    feature_map = object_features.flatten(2).transpose(1, 2)
    feature_indices = spatial_indices[..., None].expand(
        batch, query_count, feature_map.size(-1)
    )
    proposal_features = feature_map.gather(1, feature_indices)
    slot_indices = flat_indices.div(height * width, rounding_mode="floor")
    y_indices = spatial_indices.div(width, rounding_mode="floor")
    x_indices = spatial_indices.remainder(width)

    return {
        "base_scores": values,
        "base_logits": base_logits,
        "base_endpoints": endpoints,
        "base_display_logits": display_logits,
        "base_chemical_logits": chemical_logits,
        "base_direction_logits": direction_logits,
        "base_features": proposal_features,
        "mask": torch.ones_like(values, dtype=torch.bool),
        "indices": torch.stack((slot_indices, y_indices, x_indices), dim=-1),
    }


class BondGroupQueryDetector(nn.Module):
    """Refine dense proposals as permutation-equivariant three-token Bond groups."""

    GROUP_SIZE = 3

    def __init__(self, config: BondGroupQueryConfig) -> None:
        super().__init__()
        self.config = config
        self.memory_projection = nn.Sequential(
            nn.Conv2d(
                config.input_channels,
                config.d_model,
                kernel_size=3,
                stride=config.memory_stride,
                padding=1,
            ),
            nn.GroupNorm(1, config.d_model),
        )
        proposal_dim = (
            config.proposal_channels
            + 1
            + 4
            + len(BOND_TYPES)
            + len(BOND_TYPES)
            + len(DIRECTIONS)
        )
        self.proposal_projection = nn.Sequential(
            nn.LayerNorm(proposal_dim),
            nn.Linear(proposal_dim, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
        )
        self.position_projection = nn.Sequential(
            nn.Linear(2, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.role_embedding = nn.Embedding(self.GROUP_SIZE, config.d_model)
        layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            layer,
            num_layers=config.layers,
            norm=nn.LayerNorm(config.d_model),
        )
        self.presence_residual = nn.Linear(config.d_model, 1)
        self.endpoint_residual = nn.Linear(config.d_model, 2)
        self.display_residual = nn.Linear(config.d_model, len(BOND_TYPES))
        self.chemical_residual = nn.Linear(config.d_model, len(BOND_TYPES))
        for head in (
            self.presence_residual,
            self.endpoint_residual,
            self.display_residual,
            self.chemical_residual,
        ):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.endpoint_local_projection: nn.Module | None = None
        self.endpoint_local_residual: nn.Module | None = None
        self.endpoint_line_encoder: nn.Module | None = None
        self.endpoint_ink_line_encoder: nn.Module | None = None
        self.endpoint_ink_line_residual: nn.Module | None = None
        if config.endpoint_local_channels:
            patch_size = 2 * config.endpoint_patch_radius_cells + 1
            local_dim = config.endpoint_local_channels * patch_size * patch_size
            line_dim = 0
            self.endpoint_local_projection = nn.Sequential(
                nn.Conv2d(
                    config.input_channels,
                    config.endpoint_local_channels,
                    kernel_size=1,
                ),
                nn.GroupNorm(1, config.endpoint_local_channels),
                nn.GELU(),
            )
            if config.endpoint_line_samples:
                self.endpoint_line_encoder = nn.Sequential(
                    nn.Conv2d(
                        config.endpoint_local_channels,
                        config.endpoint_local_channels,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.GroupNorm(1, config.endpoint_local_channels),
                    nn.GELU(),
                    nn.AdaptiveAvgPool2d((1, 4)),
                    nn.Flatten(),
                    nn.Linear(
                        4 * config.endpoint_local_channels,
                        config.endpoint_line_feature_dim,
                    ),
                    nn.GELU(),
                    nn.LayerNorm(config.endpoint_line_feature_dim),
                )
                line_dim = config.endpoint_line_feature_dim
            self.endpoint_local_residual = nn.Sequential(
                nn.LayerNorm(config.d_model + local_dim + line_dim),
                nn.Linear(config.d_model + local_dim + line_dim, config.d_model),
                nn.GELU(),
                nn.LayerNorm(config.d_model),
                nn.Linear(config.d_model, 2),
            )
            nn.init.zeros_(self.endpoint_local_residual[-1].weight)
            nn.init.zeros_(self.endpoint_local_residual[-1].bias)
        if config.endpoint_ink_line_samples:
            self.endpoint_ink_line_encoder = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, stride=(1, 2), padding=1),
                nn.GELU(),
                nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool2d((3, 6)),
                nn.Flatten(),
                nn.Linear(32 * 3 * 6, config.endpoint_ink_line_feature_dim),
                nn.GELU(),
                nn.LayerNorm(config.endpoint_ink_line_feature_dim),
            )
            self.endpoint_ink_line_residual = nn.Sequential(
                nn.Linear(
                    config.endpoint_ink_line_feature_dim,
                    config.endpoint_ink_line_feature_dim,
                ),
                nn.GELU(),
                nn.Linear(config.endpoint_ink_line_feature_dim, 2),
            )
            nn.init.zeros_(self.endpoint_ink_line_residual[-1].weight)
            nn.init.zeros_(self.endpoint_ink_line_residual[-1].bias)

    def _validate_proposals(
        self, features: torch.Tensor, proposals: dict[str, torch.Tensor]
    ) -> tuple[int, int]:
        if features.ndim != 4 or features.size(1) != self.config.input_channels:
            raise ValueError("Bond Group Query feature map has an incompatible shape")
        required = {
            "base_logits",
            "base_endpoints",
            "base_display_logits",
            "base_chemical_logits",
            "base_direction_logits",
            "base_features",
            "mask",
        }
        missing = sorted(required - proposals.keys())
        if missing:
            raise ValueError(f"Bond proposals are missing fields: {missing}")
        batch, queries = proposals["base_logits"].shape
        if batch != features.size(0) or queries > self.config.queries:
            raise ValueError("Bond proposal batch or query count is incompatible")
        expected_shapes = {
            "base_endpoints": (batch, queries, 2, 2),
            "base_display_logits": (batch, queries, len(BOND_TYPES)),
            "base_chemical_logits": (batch, queries, len(BOND_TYPES)),
            "base_direction_logits": (batch, queries, len(DIRECTIONS)),
            "base_features": (batch, queries, self.config.proposal_channels),
            "mask": (batch, queries),
        }
        for name, expected in expected_shapes.items():
            if proposals[name].shape != expected:
                raise ValueError(
                    f"Bond proposal {name} has shape {tuple(proposals[name].shape)}, "
                    f"expected {expected}"
                )
        return batch, queries

    def forward(
        self,
        features: torch.Tensor,
        proposals: dict[str, torch.Tensor],
        *,
        pixel_values: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch, queries = self._validate_proposals(features, proposals)
        memory_map = self.memory_projection(features)
        memory_height, memory_width = memory_map.shape[-2:]
        memory = memory_map.flatten(2).transpose(1, 2)
        memory = memory + position_encoding_2d(
            memory_height,
            memory_width,
            self.config.d_model,
            memory.device,
            memory.dtype,
        ).unsqueeze(0)

        base_endpoints = proposals["base_endpoints"]
        proposal_dtype = memory.dtype
        proposal_input = torch.cat(
            (
                proposals["base_features"].to(proposal_dtype),
                proposals["base_logits"][..., None].to(proposal_dtype),
                base_endpoints.flatten(2).to(proposal_dtype),
                proposals["base_display_logits"]
                .float()
                .softmax(-1)
                .to(proposal_dtype),
                proposals["base_chemical_logits"]
                .float()
                .softmax(-1)
                .to(proposal_dtype),
                proposals["base_direction_logits"]
                .float()
                .softmax(-1)
                .to(proposal_dtype),
            ),
            dim=-1,
        )
        shared = self.proposal_projection(proposal_input)
        centers = base_endpoints.mean(dim=2)
        roles = self.role_embedding.weight.to(shared.dtype)
        object_tokens = (
            shared
            + self.position_projection(centers.to(shared.dtype))
            + roles[0]
        )
        endpoint_tokens = (
            shared[:, :, None]
            + self.position_projection(base_endpoints.to(shared.dtype))
            + roles[1:][None, None]
        )
        group_tokens = torch.cat(
            (object_tokens[:, :, None], endpoint_tokens), dim=2
        )
        token_mask = ~proposals["mask"].bool().repeat_interleave(
            self.GROUP_SIZE, dim=1
        )
        decoded = self.decoder(
            group_tokens.reshape(
                batch, queries * self.GROUP_SIZE, self.config.d_model
            ),
            memory,
            tgt_key_padding_mask=token_mask,
        ).reshape(batch, queries, self.GROUP_SIZE, self.config.d_model)
        object_hidden = decoded[:, :, 0]
        endpoint_hidden = decoded[:, :, 1:]

        endpoint_delta = self.endpoint_residual(endpoint_hidden)
        endpoint_local_features = None
        endpoint_line_features = None
        if self.endpoint_local_projection is not None:
            local_map = self.endpoint_local_projection(features)
            endpoint_local_features = sample_proposal_endpoint_patches(
                local_map,
                base_endpoints.to(local_map.dtype),
                proposals["mask"].bool(),
                radius_cells=self.config.endpoint_patch_radius_cells,
            ).flatten(3)
            local_inputs = [
                endpoint_hidden,
                endpoint_local_features.to(endpoint_hidden.dtype),
            ]
            if self.endpoint_line_encoder is not None:
                line_bands = sample_proposal_line_bands(
                    local_map,
                    base_endpoints.to(local_map.dtype),
                    proposals["mask"].bool(),
                    line_samples=self.config.endpoint_line_samples,
                    radius_cells=self.config.endpoint_line_radius_cells,
                )
                line_bands = 0.5 * (
                    line_bands + line_bands.flip((-2, -1))
                )
                endpoint_line_features = self.endpoint_line_encoder(
                    line_bands.reshape(
                        batch * queries,
                        self.config.endpoint_local_channels,
                        2 * self.config.endpoint_line_radius_cells + 1,
                        self.config.endpoint_line_samples,
                    )
                ).reshape(batch, queries, self.config.endpoint_line_feature_dim)
                endpoint_line_features = endpoint_line_features * proposals[
                    "mask"
                ][..., None].to(endpoint_line_features.dtype)
                local_inputs.append(
                    endpoint_line_features[:, :, None]
                    .expand(-1, -1, 2, -1)
                    .to(endpoint_hidden.dtype)
                )
            local_input = torch.cat(local_inputs, dim=-1)
            endpoint_delta = endpoint_delta + self.endpoint_local_residual(
                local_input
            )
        endpoint_delta = endpoint_delta.tanh()
        endpoints = (
            base_endpoints
            + endpoint_delta.to(base_endpoints.dtype) * self.config.endpoint_delta_scale
        )
        endpoint_ink_line_features = None
        if self.endpoint_ink_line_encoder is not None:
            if pixel_values is None:
                raise ValueError("Raw ink-line Bond queries require pixel values")
            ink_views = sample_proposal_ink_line_views(
                pixel_values,
                base_endpoints.to(pixel_values.dtype),
                proposals["mask"].bool(),
                line_samples=self.config.endpoint_ink_line_samples,
                band_samples=self.config.endpoint_ink_line_band,
                radius_pixels=self.config.endpoint_ink_line_radius_pixels,
            )
            endpoint_ink_line_features = self.endpoint_ink_line_encoder(
                ink_views.reshape(
                    batch * queries * 2,
                    1,
                    self.config.endpoint_ink_line_band,
                    self.config.endpoint_ink_line_samples,
                )
            ).reshape(batch, queries, 2, self.config.endpoint_ink_line_feature_dim)
            endpoint_ink_line_features = endpoint_ink_line_features * proposals[
                "mask"
            ][:, :, None, None].to(endpoint_ink_line_features.dtype)
            local_ink_delta = self.endpoint_ink_line_residual(
                endpoint_ink_line_features
            ).tanh()
            height, width = pixel_values.shape[-2:]
            base_delta = base_endpoints[:, :, 1] - base_endpoints[:, :, 0]
            base_delta_pixels = torch.stack(
                (base_delta[..., 0] * width, base_delta[..., 1] * height),
                dim=-1,
            )
            direction = base_delta_pixels / base_delta_pixels.norm(
                dim=-1, keepdim=True
            ).clamp_min(1e-6)
            endpoint_directions = torch.stack((direction, -direction), dim=2)
            endpoint_perpendiculars = torch.stack(
                (-endpoint_directions[..., 1], endpoint_directions[..., 0]),
                dim=-1,
            )
            pixel_delta = (
                local_ink_delta[..., :1] * endpoint_directions
                + local_ink_delta[..., 1:] * endpoint_perpendiculars
            )
            image_scale = float(max(height, width))
            normalized_delta = torch.stack(
                (
                    pixel_delta[..., 0] * image_scale / width,
                    pixel_delta[..., 1] * image_scale / height,
                ),
                dim=-1,
            )
            endpoints = endpoints + normalized_delta.to(endpoints.dtype) * (
                self.config.endpoint_delta_scale
            )
        endpoints = endpoints.clamp(0.0, 1.0)
        return {
            "presence_logits": proposals["base_logits"]
            + self.presence_residual(object_hidden).squeeze(-1).to(
                proposals["base_logits"].dtype
            ),
            "endpoints": endpoints,
            "display_logits": proposals["base_display_logits"]
            + self.display_residual(object_hidden).to(
                proposals["base_display_logits"].dtype
            ),
            "chemical_logits": proposals["base_chemical_logits"]
            + self.chemical_residual(object_hidden).to(
                proposals["base_chemical_logits"].dtype
            ),
            # Direction remains a frozen visual attribute in v0.8.29. Endpoint
            # reversal is therefore not coupled to a contradictory direction loss.
            "direction_logits": proposals["base_direction_logits"],
            "proposal_mask": proposals["mask"].bool(),
            "object_features": object_hidden,
            "endpoint_features": endpoint_hidden,
            "endpoint_local_features": endpoint_local_features,
            "endpoint_line_features": endpoint_line_features,
            "endpoint_ink_line_features": endpoint_ink_line_features,
        }


def set_endpoint_local_only_trainable(detector: BondGroupQueryDetector) -> None:
    """Freeze the global group-query path and train only endpoint-local modules."""

    if (
        detector.endpoint_local_projection is None
        or detector.endpoint_local_residual is None
    ):
        raise ValueError("Endpoint-local modules are disabled")
    detector.requires_grad_(False)
    detector.endpoint_local_projection.requires_grad_(True)
    detector.endpoint_local_residual.requires_grad_(True)
    if detector.endpoint_line_encoder is not None:
        detector.endpoint_line_encoder.requires_grad_(True)
    if detector.endpoint_ink_line_encoder is not None:
        detector.endpoint_ink_line_encoder.requires_grad_(True)
        detector.endpoint_ink_line_residual.requires_grad_(True)


def bond_group_query_assignments(
    output: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    matching_endpoints: torch.Tensor | None = None,
    presence_cost_weight: float | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Globally match Bond groups by geometry and optional positive confidence."""

    if presence_cost_weight is not None and presence_cost_weight < 0.0:
        raise ValueError("Bond matching presence cost weight must be nonnegative")
    endpoints = (
        output["endpoints"] if matching_endpoints is None else matching_endpoints
    ).detach().float()
    if endpoints.shape != output["endpoints"].shape:
        raise ValueError("Bond matching endpoints and predicted endpoints differ")
    proposal_mask = output.get(
        "proposal_mask",
        torch.ones(endpoints.shape[:2], device=endpoints.device, dtype=torch.bool),
    )
    if proposal_mask.shape != endpoints.shape[:2]:
        raise ValueError("Bond proposal mask and endpoint queries differ")
    if targets["mask"].shape[:1] != endpoints.shape[:1]:
        raise ValueError("Bond query and target batch sizes differ")
    presence_logits = output.get("presence_logits")
    if presence_cost_weight is not None and (
        presence_logits is None or presence_logits.shape != endpoints.shape[:2]
    ):
        raise ValueError("Confidence-aware Bond matching requires presence logits")

    assignments = []
    for batch_index in range(endpoints.size(0)):
        valid_queries = torch.where(proposal_mask[batch_index])[0]
        valid_targets = torch.where(targets["mask"][batch_index])[0]
        if len(valid_targets) > len(valid_queries):
            raise ValueError("Bond targets exceed available group queries")
        if not len(valid_targets):
            empty = torch.empty(0, device=endpoints.device, dtype=torch.long)
            assignments.append((empty, empty))
            continue
        predicted = endpoints[batch_index, valid_queries]
        expected = targets["endpoints"][batch_index, valid_targets].float()
        endpoint_cost = unordered_endpoint_cost(predicted, expected)
        center_cost = torch.cdist(
            predicted.mean(dim=1), expected.mean(dim=1), p=1
        )
        cost = 8.0 * endpoint_cost + 2.0 * center_cost
        if presence_cost_weight is not None:
            positive_presence_cost = F.softplus(
                -presence_logits[batch_index, valid_queries].detach().float()
            )
            cost = cost + presence_cost_weight * positive_presence_cost[:, None]
        rows, columns = linear_sum_assignment(cost.cpu().numpy())
        assignments.append(
            (
                valid_queries[
                    torch.as_tensor(rows, device=endpoints.device, dtype=torch.long)
                ],
                valid_targets[
                    torch.as_tensor(columns, device=endpoints.device, dtype=torch.long)
                ],
            )
        )
    return assignments


def _validate_bond_group_query_assignments(
    output: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    assignments: list[tuple[torch.Tensor, torch.Tensor]],
) -> None:
    if len(assignments) != output["endpoints"].size(0):
        raise ValueError("Precomputed Bond assignments have an incompatible batch size")
    proposal_mask = output.get(
        "proposal_mask",
        torch.ones_like(output["presence_logits"], dtype=torch.bool),
    ).bool()
    for batch_index, (queries, target_indices) in enumerate(assignments):
        if queries.ndim != 1 or target_indices.shape != queries.shape:
            raise ValueError("Precomputed Bond assignment indices have incompatible shapes")
        invalid_dtype = (
            queries.dtype == torch.bool
            or queries.dtype.is_floating_point
            or queries.dtype.is_complex
            or target_indices.dtype == torch.bool
            or target_indices.dtype.is_floating_point
            or target_indices.dtype.is_complex
        )
        if invalid_dtype:
            raise ValueError("Precomputed Bond assignment indices must be integers")
        if (
            queries.device != output["endpoints"].device
            or target_indices.device != output["endpoints"].device
        ):
            raise ValueError("Precomputed Bond assignment indices use the wrong device")
        queries = queries.long()
        target_indices = target_indices.long()
        expected_targets = torch.where(targets["mask"][batch_index])[0]
        if (
            len(queries.unique()) != len(queries)
            or len(target_indices.unique()) != len(target_indices)
            or bool((queries < 0).any())
            or bool((queries >= proposal_mask.size(1)).any())
            or bool((target_indices < 0).any())
            or bool((target_indices >= targets["mask"].size(1)).any())
            or (len(queries) and not bool(proposal_mask[batch_index, queries].all()))
            or set(target_indices.tolist()) != set(expected_targets.tolist())
        ):
            raise ValueError("Precomputed Bond assignments violate query/target identity")


def bond_group_query_geometry_loss(
    output: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    matching_endpoints: torch.Tensor | None = None,
    target_weights: torch.Tensor | None = None,
    endpoint_weight: float = 8.0,
    endpoint_barrier_weight: float = 0.0,
    endpoint_barrier_margin_pixels: float = 4.0,
    endpoint_huber_beta_pixels: float = 4.0,
    image_size: float = 768.0,
) -> dict[str, torch.Tensor]:
    """Isolate endpoint learning from existence, count, and type objectives."""

    if min(endpoint_weight, endpoint_barrier_weight) < 0.0:
        raise ValueError("Bond Group Query geometry weights must be nonnegative")
    if (
        endpoint_barrier_margin_pixels < 0.0
        or endpoint_huber_beta_pixels <= 0.0
        or image_size <= 0.0
    ):
        raise ValueError("Bond Group Query endpoint geometry is invalid")
    if target_weights is not None and target_weights.shape != targets["mask"].shape:
        raise ValueError("Bond geometry target weights must match the target mask")
    assignments = bond_group_query_assignments(
        output,
        targets,
        matching_endpoints=matching_endpoints,
    )
    endpoint_losses = []
    barrier_losses = []
    for batch_index, (query_indices, target_indices) in enumerate(assignments):
        if not len(query_indices):
            continue
        predicted = output["endpoints"][batch_index, query_indices]
        expected = targets["endpoints"][batch_index, target_indices]
        forward = (predicted - expected).abs().sum(dim=(-1, -2))
        reverse = (predicted - expected.flip(1)).abs().sum(dim=(-1, -2))
        aligned = torch.where(
            (reverse < forward)[:, None, None], expected.flip(1), expected
        )
        pixel_errors = (predicted.float() - aligned.float()) * image_size
        per_object_endpoint_loss = F.smooth_l1_loss(
            pixel_errors,
            torch.zeros_like(pixel_errors),
            beta=endpoint_huber_beta_pixels,
            reduction="none",
        ).mean(dim=(-1, -2)) / image_size
        weights = (
            torch.ones_like(per_object_endpoint_loss)
            if target_weights is None
            else target_weights[batch_index, target_indices].to(
                per_object_endpoint_loss.dtype
            )
        )
        endpoint_losses.append(
            (per_object_endpoint_loss * weights).sum() / weights.sum().clamp_min(1.0)
        )
        distances = torch.linalg.vector_norm(
            pixel_errors,
            dim=-1,
        )
        per_object_barrier = F.relu(
            distances - endpoint_barrier_margin_pixels
        ).square().mean(dim=-1)
        barrier_losses.append(
            (per_object_barrier * weights).sum() / weights.sum().clamp_min(1.0)
        )
    zero = output["endpoints"].sum() * 0.0
    endpoint_loss = torch.stack(endpoint_losses).mean() if endpoint_losses else zero
    endpoint_barrier_loss = (
        torch.stack(barrier_losses).mean() if barrier_losses else zero
    )
    return {
        "loss": (
            endpoint_weight * endpoint_loss
            + endpoint_barrier_weight * endpoint_barrier_loss
        ),
        "endpoint_loss": endpoint_loss,
        "endpoint_barrier_loss": endpoint_barrier_loss,
    }


def bond_group_query_loss(
    output: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    no_object_weight: float = 0.25,
    ranking_weight: float = 0.5,
    set_count_weight: float = 1.0,
    soft_count_weight: float = 0.1,
    endpoint_weight: float = 8.0,
    endpoint_barrier_weight: float = 0.02,
    endpoint_barrier_margin_pixels: float = 4.0,
    display_weight: float = 1.0,
    chemical_weight: float = 1.0,
    image_size: float = 768.0,
    assignments: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> dict[str, torch.Tensor]:
    """Train one complete Bond Object per globally matched query group."""

    weights = (
        no_object_weight,
        ranking_weight,
        set_count_weight,
        soft_count_weight,
        endpoint_weight,
        endpoint_barrier_weight,
        display_weight,
        chemical_weight,
    )
    if min(weights) < 0.0 or no_object_weight == 0.0:
        raise ValueError("Bond Group Query loss weights are invalid")
    if endpoint_barrier_margin_pixels < 0.0 or image_size <= 0.0:
        raise ValueError("Bond Group Query endpoint geometry is invalid")

    if assignments is None:
        assignments = bond_group_query_assignments(output, targets)
    else:
        _validate_bond_group_query_assignments(output, targets, assignments)
    proposal_mask = output.get(
        "proposal_mask",
        torch.ones_like(output["presence_logits"], dtype=torch.bool),
    ).bool()
    presence_targets = torch.zeros_like(output["presence_logits"])
    presence_weights = torch.full_like(
        output["presence_logits"], no_object_weight
    ) * proposal_mask.to(output["presence_logits"].dtype)
    endpoint_losses = []
    barrier_losses = []
    display_logits = []
    display_targets = []
    chemical_logits = []
    chemical_targets = []

    for batch_index, (query_indices, target_indices) in enumerate(assignments):
        if not len(query_indices):
            continue
        presence_targets[batch_index, query_indices] = 1.0
        presence_weights[batch_index, query_indices] = 1.0
        predicted = output["endpoints"][batch_index, query_indices]
        expected = targets["endpoints"][batch_index, target_indices]
        forward = (predicted - expected).abs().sum(dim=(-1, -2))
        reverse = (predicted - expected.flip(1)).abs().sum(dim=(-1, -2))
        use_reverse = (reverse < forward)[:, None, None]
        aligned = torch.where(use_reverse, expected.flip(1), expected)
        endpoint_losses.append(F.smooth_l1_loss(predicted, aligned))
        distances = torch.linalg.vector_norm(
            (predicted.float() - aligned.float()) * image_size,
            dim=-1,
        )
        barrier_losses.append(
            F.relu(distances - endpoint_barrier_margin_pixels).square().mean()
        )
        display_logits.append(output["display_logits"][batch_index, query_indices])
        display_targets.append(targets["display"][batch_index, target_indices])
        chemical_logits.append(output["chemical_logits"][batch_index, query_indices])
        chemical_targets.append(targets["chemical"][batch_index, target_indices])

    presence_loss = (
        F.binary_cross_entropy_with_logits(
            output["presence_logits"], presence_targets, reduction="none"
        )
        * presence_weights
    ).sum() / presence_weights.sum().clamp_min(1.0)
    zero = presence_loss * 0.0
    endpoint_loss = torch.stack(endpoint_losses).mean() if endpoint_losses else zero
    endpoint_barrier_loss = (
        torch.stack(barrier_losses).mean() if barrier_losses else zero
    )
    if display_logits:
        display_loss = F.cross_entropy(
            torch.cat(display_logits), torch.cat(display_targets).long()
        )
        chemical_loss = F.cross_entropy(
            torch.cat(chemical_logits), torch.cat(chemical_targets).long()
        )
    else:
        display_loss = chemical_loss = zero

    ranking_losses = []
    for batch_index in range(len(proposal_mask)):
        valid = proposal_mask[batch_index]
        positive = valid & presence_targets[batch_index].bool()
        negative = valid & ~presence_targets[batch_index].bool()
        positive_logits = output["presence_logits"][batch_index, positive]
        negative_logits = output["presence_logits"][batch_index, negative]
        if not len(positive_logits) or not len(negative_logits):
            continue
        hard_negative_count = min(len(negative_logits), 2 * len(positive_logits))
        hard_negatives = negative_logits.topk(hard_negative_count).values
        ranking_losses.append(
            F.softplus(hard_negatives[:, None] - positive_logits[None]).mean()
        )
    ranking_loss = torch.stack(ranking_losses).mean() if ranking_losses else zero

    masked_logits = output["presence_logits"].float().masked_fill(
        ~proposal_mask, -30.0
    )
    count_targets = targets["mask"].sum(dim=1).long()
    set_count_loss = F.nll_loss(
        poisson_binomial_log_probs(masked_logits), count_targets
    )
    soft_count_loss = F.smooth_l1_loss(
        masked_logits.sigmoid().sum(dim=1), count_targets.float()
    )
    return {
        "loss": (
            presence_loss
            + ranking_weight * ranking_loss
            + set_count_weight * set_count_loss
            + soft_count_weight * soft_count_loss
            + endpoint_weight * endpoint_loss
            + endpoint_barrier_weight * endpoint_barrier_loss
            + display_weight * display_loss
            + chemical_weight * chemical_loss
        ),
        "presence_loss": presence_loss,
        "ranking_loss": ranking_loss,
        "set_count_loss": set_count_loss,
        "soft_count_loss": soft_count_loss,
        "endpoint_loss": endpoint_loss,
        "endpoint_barrier_loss": endpoint_barrier_loss,
        "display_loss": display_loss,
        "chemical_loss": chemical_loss,
    }


@torch.no_grad()
def decode_bond_group_queries(
    output: dict[str, torch.Tensor],
    *,
    mode: str = "set_count",
    threshold: float = 0.25,
) -> list[list[dict[str, torch.Tensor | float | int]]]:
    """Decode raw Bond groups without cell-local slots or geometric pruning."""

    if mode not in {"set_count", "threshold"}:
        raise ValueError(f"Unknown Bond Group Query decode mode: {mode}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Bond Group Query threshold must be in [0, 1]")
    mask = output.get(
        "proposal_mask",
        torch.ones_like(output["presence_logits"], dtype=torch.bool),
    ).bool()
    raw_logits = output["presence_logits"].masked_fill(~mask, -30.0)
    probabilities = raw_logits.sigmoid()
    logits = raw_logits.float()
    counts = (
        poisson_binomial_log_probs(logits).argmax(-1)
        if mode == "set_count"
        else None
    )
    decoded = []
    for batch_index in range(len(logits)):
        order = logits[batch_index].argsort(descending=True)
        objects = []
        desired = int(counts[batch_index]) if counts is not None else None
        for query_index in order.tolist():
            if not bool(mask[batch_index, query_index]):
                continue
            score = float(probabilities[batch_index, query_index])
            if desired is None and score < threshold:
                break
            if desired is not None and len(objects) >= desired:
                break
            objects.append(
                {
                    "score": score,
                    "feature": output["object_features"][batch_index, query_index],
                    "endpoints": output["endpoints"][batch_index, query_index],
                    "display": int(
                        output["display_logits"][batch_index, query_index].argmax()
                    ),
                    "chemical": int(
                        output["chemical_logits"][batch_index, query_index].argmax()
                    ),
                    "direction": int(
                        output["direction_logits"][batch_index, query_index].argmax()
                    ),
                }
            )
        decoded.append(objects)
    return decoded
