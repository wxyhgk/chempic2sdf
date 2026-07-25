"""Full-adjacency node-edge message passing for dual vertex candidates."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from chemvector.structured_selector import GroundedStructuredSelector


@dataclass(frozen=True, slots=True)
class DualGraphDenoiserConfig:
    d_model: int = 192
    edge_dim: int = 64
    layers: int = 2
    dropout: float = 0.0
    residual_scale: float = 8.0
    refine_edges: bool = False
    edge_residual_scale: float = 8.0
    edge_auxiliary_dim: int = 0
    straight_through_topk: int = 0
    patch_mode: str = "none"
    patch_size: int = 32
    patch_residual_scale: float = 4.0

    def __post_init__(self) -> None:
        if (
            min(self.d_model, self.edge_dim, self.layers) < 1
            or self.edge_auxiliary_dim < 0
            or self.straight_through_topk < 0
        ):
            raise ValueError("Graph denoiser dimensions must be positive")
        if self.straight_through_topk == 1:
            raise ValueError("Straight-through top-k must be zero or at least two")
        if self.straight_through_topk and self.edge_auxiliary_dim < 4:
            raise ValueError("Straight-through features require endpoint ray evidence")
        if self.patch_mode not in {"none", "feature", "pixel", "semantic_pixel"}:
            raise ValueError(
                "Patch mode must be none, feature, pixel, or semantic_pixel"
            )
        if self.patch_size < 4 or self.patch_size % 2:
            raise ValueError("Patch size must be an even integer of at least four")
        if (
            not 0.0 <= self.dropout < 1.0
            or self.residual_scale <= 0.0
            or self.edge_residual_scale <= 0.0
            or self.patch_residual_scale <= 0.0
        ):
            raise ValueError("Graph denoiser dropout or residual scale is invalid")


class CandidatePatchResidual(nn.Module):
    """Encode either the existing feature patch or a larger raw-pixel patch."""

    def __init__(
        self,
        mode: str,
        *,
        visual_dim: int,
        d_model: int,
        patch_size: int,
        residual_scale: float,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.patch_size = patch_size
        self.residual_scale = residual_scale
        if mode == "feature":
            if visual_dim < 1:
                raise ValueError("Feature patch mode requires candidate visual features")
            self.encoder = nn.Sequential(
                nn.LayerNorm(visual_dim),
                nn.Linear(visual_dim, 208),
                nn.GELU(),
                nn.Linear(208, d_model),
                nn.GELU(),
            )
        elif mode == "pixel":
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 24, 3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(24, 48, 3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(48, 96, 3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(96, d_model, 3, stride=2, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
        else:
            raise ValueError("Candidate patch residual requires feature or pixel mode")
        self.residual = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(
        self,
        hidden: torch.Tensor,
        candidate_mask: torch.Tensor,
        candidate_visual: torch.Tensor | None,
        candidate_pixel_patches: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.mode == "feature":
            if candidate_visual is None:
                raise ValueError("Feature patch mode requires candidate visual features")
            encoded = self.encoder(candidate_visual.float())
        else:
            if candidate_pixel_patches is None:
                raise ValueError("Pixel patch mode requires candidate pixel patches")
            expected = (
                hidden.size(0),
                hidden.size(1),
                1,
                self.patch_size,
                self.patch_size,
            )
            if candidate_pixel_patches.shape != expected:
                raise ValueError("Candidate pixel patches do not match graph config")
            patches = candidate_pixel_patches.reshape(-1, 1, self.patch_size, self.patch_size)
            patches = patches.float()
            if candidate_pixel_patches.dtype == torch.uint8:
                patches = patches / 255.0
            encoded = self.encoder(patches).reshape(hidden.shape)
        raw_residual = self.residual(encoded)
        bounded = self.residual_scale * torch.tanh(
            raw_residual / self.residual_scale
        )
        output = self.norm(hidden + bounded.to(hidden.dtype))
        return output.masked_fill(~candidate_mask.unsqueeze(-1), 0.0)


class CandidatePassThroughContext(nn.Module):
    """Classify local pass-through geometry before exposing detached context."""

    def __init__(
        self, *, d_model: int, patch_size: int, residual_scale: float
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.residual_scale = residual_scale
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 24, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(48, 96, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(96, d_model, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(d_model, 1)
        self.context_projection = nn.Linear(1, d_model)
        self.norm = nn.LayerNorm(d_model)
        nn.init.zeros_(self.context_projection.weight)
        nn.init.zeros_(self.context_projection.bias)

    def forward(
        self,
        hidden: torch.Tensor,
        candidate_mask: torch.Tensor,
        candidate_pixel_patches: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if candidate_pixel_patches is None:
            raise ValueError("Semantic pixel mode requires candidate pixel patches")
        expected = (
            hidden.size(0),
            hidden.size(1),
            1,
            self.patch_size,
            self.patch_size,
        )
        if candidate_pixel_patches.shape != expected:
            raise ValueError("Candidate pixel patches do not match graph config")
        patches = candidate_pixel_patches.reshape(
            -1, 1, self.patch_size, self.patch_size
        ).float()
        if candidate_pixel_patches.dtype == torch.uint8:
            patches = patches / 255.0
        encoded = self.encoder(patches).reshape(hidden.shape)
        pass_through_logits = self.classifier(encoded).squeeze(-1)
        probability = pass_through_logits.detach().sigmoid().unsqueeze(-1)
        raw_context = self.context_projection(probability)
        bounded = self.residual_scale * torch.tanh(
            raw_context / self.residual_scale
        )
        output = self.norm(hidden + bounded.to(hidden.dtype))
        output = output.masked_fill(~candidate_mask.unsqueeze(-1), 0.0)
        pass_through_logits = pass_through_logits.masked_fill(~candidate_mask, -30.0)
        return output, pass_through_logits


class FullAdjacencyMessageLayer(nn.Module):
    def __init__(self, config: DualGraphDenoiserConfig) -> None:
        super().__init__()
        edge_inputs = config.d_model * 2 + 5 + config.edge_auxiliary_dim
        self.edge_auxiliary_dim = config.edge_auxiliary_dim
        self.straight_through_topk = config.straight_through_topk
        self.edge_encoder = nn.Sequential(
            nn.LayerNorm(edge_inputs),
            nn.Linear(edge_inputs, config.edge_dim),
            nn.GELU(),
            nn.Linear(config.edge_dim, config.edge_dim),
            nn.GELU(),
        )
        self.edge_message = nn.Linear(config.edge_dim, config.d_model)
        self.edge_residual = (
            nn.Linear(config.edge_dim, 1) if config.refine_edges else None
        )
        self.edge_residual_scale = config.edge_residual_scale
        if self.edge_residual is not None:
            nn.init.zeros_(self.edge_residual.weight)
            nn.init.zeros_(self.edge_residual.bias)
        self.node_update = nn.Sequential(
            nn.LayerNorm(
                config.d_model * 2 + 2 + (2 if config.straight_through_topk else 0)
            ),
            nn.Linear(
                config.d_model * 2 + 2 + (2 if config.straight_through_topk else 0),
                config.d_model,
            ),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model),
        )
        self.node_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        hidden: torch.Tensor,
        points: torch.Tensor,
        candidate_mask: torch.Tensor,
        edge_logits: torch.Tensor,
        pair_indices: torch.Tensor,
        edge_auxiliary: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        begin = pair_indices[:, 0]
        end = pair_indices[:, 1]
        pair_mask = candidate_mask[:, begin] & candidate_mask[:, end]
        begin_hidden = hidden[:, begin]
        end_hidden = hidden[:, end]
        delta = points[:, end].float() - points[:, begin].float()
        length = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        edge_probability = edge_logits.float().sigmoid() * pair_mask
        edge_scalar = torch.cat(
            (
                edge_logits.float().clamp(-12.0, 12.0).unsqueeze(-1) / 12.0,
                edge_probability.unsqueeze(-1),
                delta,
                length,
            ),
            dim=-1,
        )
        if self.edge_auxiliary_dim:
            if edge_auxiliary is None or edge_auxiliary.shape != (
                hidden.size(0), pair_indices.size(0), self.edge_auxiliary_dim
            ):
                raise ValueError("Edge auxiliary features do not match graph config")
            edge_scalar = torch.cat((edge_scalar, edge_auxiliary.float()), dim=-1)
        edge_hidden = self.edge_encoder(
            torch.cat(
                (
                    begin_hidden + end_hidden,
                    (begin_hidden - end_hidden).abs(),
                    edge_scalar,
                ),
                dim=-1,
            )
        )
        if self.edge_residual is not None:
            raw_edge_residual = self.edge_residual(edge_hidden).squeeze(-1)
            bounded_edge_residual = self.edge_residual_scale * torch.tanh(
                raw_edge_residual / self.edge_residual_scale
            )
            edge_logits = (edge_logits.float() + bounded_edge_residual).masked_fill(
                ~pair_mask, -30.0
            )
            edge_probability = edge_logits.sigmoid() * pair_mask
        messages = self.edge_message(edge_hidden) * edge_probability.unsqueeze(-1)
        aggregated = torch.zeros_like(hidden)
        degree = torch.zeros(
            (*candidate_mask.shape, 1), device=hidden.device, dtype=hidden.dtype
        )
        confident_degree = torch.zeros_like(degree)
        batch_size = hidden.size(0)
        node_dim = hidden.size(-1)
        scatter_begin = begin[None, :, None].expand(batch_size, -1, node_dim)
        scatter_end = end[None, :, None].expand_as(scatter_begin)
        aggregated.scatter_add_(1, scatter_begin, messages)
        aggregated.scatter_add_(1, scatter_end, messages)
        degree.scatter_add_(
            1,
            begin[None, :, None].expand(batch_size, -1, 1),
            edge_probability.unsqueeze(-1),
        )
        degree.scatter_add_(
            1,
            end[None, :, None].expand(batch_size, -1, 1),
            edge_probability.unsqueeze(-1),
        )
        confident = ((edge_probability >= 0.8) & pair_mask).to(hidden.dtype).unsqueeze(-1)
        confident_degree.scatter_add_(
            1, begin[None, :, None].expand(batch_size, -1, 1), confident
        )
        confident_degree.scatter_add_(
            1, end[None, :, None].expand(batch_size, -1, 1), confident
        )
        aggregated = aggregated / degree.clamp_min(1.0)
        node_scalars = [
            (degree / 8.0).clamp_max(4.0) / 4.0,
            (confident_degree / 4.0).clamp_max(4.0) / 4.0,
        ]
        if self.straight_through_topk:
            node_scalars.append(
                straight_through_node_features(
                    points,
                    candidate_mask,
                    edge_probability,
                    pair_indices,
                    edge_auxiliary,
                    topk=self.straight_through_topk,
                )
            )
        update = self.node_update(
            torch.cat(
                (
                    hidden,
                    aggregated,
                    *node_scalars,
                ),
                dim=-1,
            )
        )
        output = self.node_norm(hidden + update)
        return output.masked_fill(~candidate_mask.unsqueeze(-1), 0.0), edge_logits


def straight_through_node_features(
    points: torch.Tensor,
    candidate_mask: torch.Tensor,
    edge_probability: torch.Tensor,
    pair_indices: torch.Tensor,
    edge_auxiliary: torch.Tensor | None,
    *,
    topk: int,
) -> torch.Tensor:
    """Summarize opposite incident edges and ink continuing through a node."""

    if edge_auxiliary is None or edge_auxiliary.size(-1) < 4:
        raise ValueError("Straight-through summary requires four endpoint ray features")
    node_count = points.size(1)
    k = min(topk, node_count)
    begin = pair_indices[:, 0]
    end = pair_indices[:, 1]
    incident = edge_probability.new_zeros(points.size(0), node_count, node_count)
    outward = edge_probability.new_zeros(points.size(0), node_count, node_count)
    incident[:, begin, end] = edge_probability
    incident[:, end, begin] = edge_probability
    outward[:, begin, end] = edge_auxiliary[..., 1].float()
    outward[:, end, begin] = edge_auxiliary[..., 3].float()
    probabilities, neighbors = incident.topk(k, dim=-1)
    neighbor_points = torch.gather(
        points[:, None].expand(-1, node_count, -1, -1),
        2,
        neighbors[..., None].expand(-1, -1, -1, 2),
    )
    directions = neighbor_points.float() - points[:, :, None].float()
    directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    cosine = torch.einsum("bnkd,bnld->bnkl", directions, directions)
    opposition = ((-cosine - 0.8) / 0.2).clamp(0.0, 1.0)
    pair_support = (
        probabilities[:, :, :, None]
        * probabilities[:, :, None, :]
        * opposition
    )
    continuation = torch.gather(outward, 2, neighbors)
    continuation_pair = torch.minimum(
        continuation[:, :, :, None], continuation[:, :, None, :]
    )
    opposite_support = pair_support.amax(dim=(-1, -2))
    pass_through_support = (pair_support * continuation_pair).amax(dim=(-1, -2))
    features = torch.stack((opposite_support, pass_through_support), dim=-1)
    return features.masked_fill(~candidate_mask.unsqueeze(-1), 0.0)


class GroundedDualGraphDenoiser(nn.Module):
    """Refine a deployed group selector using every candidate-pair edge."""

    def __init__(
        self,
        base_selector: GroundedStructuredSelector,
        config: DualGraphDenoiserConfig,
    ) -> None:
        super().__init__()
        if base_selector.config.d_model != config.d_model:
            raise ValueError("Base selector and graph denoiser dimensions differ")
        self.base_selector = base_selector
        self.config = config
        self.message_layers = nn.ModuleList(
            FullAdjacencyMessageLayer(config) for _ in range(config.layers)
        )
        self.patch_residual = (
            CandidatePatchResidual(
                config.patch_mode,
                visual_dim=base_selector.config.visual_dim,
                d_model=config.d_model,
                patch_size=config.patch_size,
                residual_scale=config.patch_residual_scale,
            )
            if config.patch_mode in {"feature", "pixel"}
            else None
        )
        self.pass_through_context = (
            CandidatePassThroughContext(
                d_model=config.d_model,
                patch_size=config.patch_size,
                residual_scale=config.patch_residual_scale,
            )
            if config.patch_mode == "semantic_pixel"
            else None
        )
        self.residual_score = nn.Linear(config.d_model, 1)
        nn.init.zeros_(self.residual_score.weight)
        nn.init.zeros_(self.residual_score.bias)

    def forward(
        self,
        candidate_hidden: torch.Tensor,
        base_logits: torch.Tensor,
        points: torch.Tensor,
        candidate_mask: torch.Tensor,
        edge_logits: torch.Tensor,
        pair_indices: torch.Tensor,
        candidate_auxiliary: torch.Tensor,
        candidate_visual: torch.Tensor | None = None,
        candidate_roles: torch.Tensor | None = None,
        edge_auxiliary: torch.Tensor | None = None,
        candidate_pixel_patches: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_with_edges(
            candidate_hidden,
            base_logits,
            points,
            candidate_mask,
            edge_logits,
            pair_indices,
            candidate_auxiliary,
            candidate_visual,
            candidate_roles,
            edge_auxiliary,
            candidate_pixel_patches,
        )["node_logits"]

    def forward_with_edges(
        self,
        candidate_hidden: torch.Tensor,
        base_logits: torch.Tensor,
        points: torch.Tensor,
        candidate_mask: torch.Tensor,
        edge_logits: torch.Tensor,
        pair_indices: torch.Tensor,
        candidate_auxiliary: torch.Tensor,
        candidate_visual: torch.Tensor | None = None,
        candidate_roles: torch.Tensor | None = None,
        edge_auxiliary: torch.Tensor | None = None,
        candidate_pixel_patches: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if pair_indices.shape != (edge_logits.size(1), 2):
            raise ValueError("Graph denoiser pair indices do not match edge logits")
        hidden = self.base_selector.encode_context(
            candidate_hidden,
            base_logits,
            points,
            candidate_mask,
            edge_logits,
            pair_indices,
            candidate_auxiliary,
            candidate_visual,
            candidate_roles,
        )
        if self.patch_residual is not None:
            hidden = self.patch_residual(
                hidden,
                candidate_mask,
                candidate_visual,
                candidate_pixel_patches,
            )
        pass_through_logits = None
        if self.pass_through_context is not None:
            hidden, pass_through_logits = self.pass_through_context(
                hidden,
                candidate_mask,
                candidate_pixel_patches,
            )
        deployed_logits = self.base_selector.score_context(
            hidden, base_logits, candidate_mask
        )
        refined_edge_logits = edge_logits.float()
        for layer in self.message_layers:
            hidden, refined_edge_logits = layer(
                hidden,
                points,
                candidate_mask,
                refined_edge_logits,
                pair_indices,
                edge_auxiliary,
            )
        raw_residual = self.residual_score(hidden).squeeze(-1)
        residual = self.config.residual_scale * torch.tanh(
            raw_residual / self.config.residual_scale
        )
        node_logits = (deployed_logits + residual).masked_fill(~candidate_mask, -30.0)
        output = {
            "node_logits": node_logits,
            "edge_logits": refined_edge_logits,
        }
        if pass_through_logits is not None:
            output["pass_through_logits"] = pass_through_logits
        return output
