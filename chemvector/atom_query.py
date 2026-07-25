"""Set-based atom detection with learned queries and Hungarian matching."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from chemvector.geometry import stable_pairwise_distances
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True, slots=True)
class AtomQueryConfig:
    input_channels: int = 192
    d_model: int = 192
    queries: int = 128
    layers: int = 3
    heads: int = 6
    ffn_dim: int = 768
    dropout: float = 0.0
    memory_stride: int = 2
    local_feature_dim: int = 48
    local_radius: float = 32.0 / 768.0
    cardinality_layers: int = 2
    selection_layers: int = 2

    def __post_init__(self) -> None:
        if min(
            self.input_channels,
            self.d_model,
            self.queries,
            self.layers,
            self.heads,
            self.ffn_dim,
            self.local_feature_dim,
            self.cardinality_layers,
            self.selection_layers,
        ) < 1:
            raise ValueError("Atom Query dimensions must be positive")
        if self.d_model % self.heads:
            raise ValueError("d_model must be divisible by heads")
        if self.d_model % 4:
            raise ValueError("d_model must be divisible by 4 for 2D positions")
        if self.memory_stride not in {1, 2, 4}:
            raise ValueError("memory_stride must be 1, 2, or 4")
        if not 0.0 < self.local_radius < 0.5:
            raise ValueError("local_radius must be in (0, 0.5)")


class CardinalityTokenLayer(nn.Module):
    """Update one global token from atom queries and visual memory."""

    def __init__(self, config: AtomQueryConfig) -> None:
        super().__init__()
        self.token_query_norm = nn.LayerNorm(config.d_model)
        self.atom_norm = nn.LayerNorm(config.d_model)
        self.atom_attention = nn.MultiheadAttention(
            config.d_model,
            config.heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.memory_query_norm = nn.LayerNorm(config.d_model)
        self.memory_norm = nn.LayerNorm(config.d_model)
        self.memory_attention = nn.MultiheadAttention(
            config.d_model,
            config.heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.ffn_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_dim, config.d_model),
        )

    def forward(
        self,
        token: torch.Tensor,
        atom_queries: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        normalized_atoms = self.atom_norm(atom_queries)
        update, _ = self.atom_attention(
            self.token_query_norm(token),
            normalized_atoms,
            normalized_atoms,
            need_weights=False,
        )
        token = token + update
        normalized_memory = self.memory_norm(memory)
        update, _ = self.memory_attention(
            self.memory_query_norm(token),
            normalized_memory,
            normalized_memory,
            need_weights=False,
        )
        token = token + update
        return token + self.ffn(self.ffn_norm(token))


def position_encoding_2d(
    height: int,
    width: int,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    quarter = channels // 4
    frequencies = torch.exp(
        torch.arange(quarter, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / max(quarter - 1, 1))
    )
    y = torch.arange(height, device=device, dtype=torch.float32)[:, None]
    x = torch.arange(width, device=device, dtype=torch.float32)[:, None]
    y_angles = y * frequencies[None, :]
    x_angles = x * frequencies[None, :]
    y_encoding = torch.cat((y_angles.sin(), y_angles.cos()), dim=-1)
    x_encoding = torch.cat((x_angles.sin(), x_angles.cos()), dim=-1)
    encoding = torch.cat(
        (
            y_encoding[:, None, :].expand(height, width, -1),
            x_encoding[None, :, :].expand(height, width, -1),
        ),
        dim=-1,
    )
    return encoding.reshape(height * width, channels).to(dtype=dtype)


def query_reference_points(queries: int) -> torch.Tensor:
    columns = math.ceil(math.sqrt(queries))
    rows = math.ceil(queries / columns)
    y, x = torch.meshgrid(
        (torch.arange(rows, dtype=torch.float32) + 0.5) / rows,
        (torch.arange(columns, dtype=torch.float32) + 0.5) / columns,
        indexing="ij",
    )
    return torch.stack((x.flatten(), y.flatten()), dim=-1)[:queries]


class AtomQueryDetector(nn.Module):
    def __init__(self, config: AtomQueryConfig) -> None:
        super().__init__()
        self.config = config
        kernel_size = 1 if config.memory_stride == 1 else 3
        padding = 0 if config.memory_stride == 1 else 1
        self.input_projection = nn.Sequential(
            nn.Conv2d(
                config.input_channels,
                config.d_model,
                kernel_size=kernel_size,
                stride=config.memory_stride,
                padding=padding,
            ),
            nn.GroupNorm(1, config.d_model),
        )
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
        self.queries = nn.Embedding(config.queries, config.d_model)
        self.reference_projection = nn.Linear(2, config.d_model)
        self.register_buffer(
            "reference_points",
            query_reference_points(config.queries),
            persistent=True,
        )
        self.presence = nn.Linear(config.d_model, 1)
        self.ranking_presence = deepcopy(self.presence)
        self.cardinality_token = nn.Embedding(1, config.d_model)
        self.cardinality_decoder = nn.ModuleList(
            CardinalityTokenLayer(config) for _ in range(config.cardinality_layers)
        )
        self.cardinality_norm = nn.LayerNorm(config.d_model)
        self.count_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.queries + 1),
        )
        self.count_regression = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, 1),
        )
        self.center = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, 2),
        )
        self.aromatic = nn.Linear(config.d_model, 1)
        self.local_projection = nn.Conv2d(
            config.input_channels,
            config.local_feature_dim,
            kernel_size=1,
        )
        self.selection_local_projection = deepcopy(self.local_projection)
        self.local_refinement = nn.Sequential(
            nn.Linear(config.d_model + config.local_feature_dim * 9, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, 2),
        )
        self.second_local_refinement = deepcopy(self.local_refinement)
        self.role = nn.Sequential(
            nn.Linear(config.d_model + config.local_feature_dim * 9, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, 2),
        )
        self.selection = nn.Sequential(
            nn.Linear(config.d_model + config.local_feature_dim * 9, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, 1),
        )
        self.selection_context_projection = nn.Sequential(
            nn.Linear(config.d_model + config.local_feature_dim * 9, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
        )
        self.selection_position_projection = nn.Linear(2, config.d_model)
        selection_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.selection_context = nn.TransformerEncoder(
            selection_layer,
            num_layers=config.selection_layers,
            norm=nn.LayerNorm(config.d_model),
            enable_nested_tensor=False,
        )
        self.selection_context_score = nn.Linear(config.d_model, 1)
        self.duplicate_projection = nn.Linear(config.d_model, 64)
        self.duplicate_geometry = nn.Sequential(
            nn.Linear(3, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        self.duplicate_similarity_scale = nn.Parameter(torch.tensor(1.0))
        nn.init.constant_(self.presence.bias, -2.0)
        nn.init.constant_(self.ranking_presence.bias, -2.0)
        nn.init.zeros_(self.count_regression[-1].weight)
        nn.init.constant_(self.count_regression[-1].bias, 20.0)
        nn.init.zeros_(self.center[-1].weight)
        nn.init.zeros_(self.center[-1].bias)
        nn.init.zeros_(self.local_refinement[-1].weight)
        nn.init.zeros_(self.local_refinement[-1].bias)
        nn.init.zeros_(self.second_local_refinement[-1].weight)
        nn.init.zeros_(self.second_local_refinement[-1].bias)
        nn.init.zeros_(self.selection[-1].weight)
        nn.init.zeros_(self.selection[-1].bias)
        nn.init.zeros_(self.selection_context_score.weight)
        nn.init.zeros_(self.selection_context_score.bias)
        auxiliary_layers = config.layers - 1
        self.auxiliary_presence = nn.ModuleList(
            deepcopy(self.presence) for _ in range(auxiliary_layers)
        )
        self.auxiliary_count_heads = nn.ModuleList(
            deepcopy(self.count_head) for _ in range(auxiliary_layers)
        )
        self.auxiliary_centers = nn.ModuleList(
            deepcopy(self.center) for _ in range(auxiliary_layers)
        )
        self.auxiliary_aromatic = nn.ModuleList(
            deepcopy(self.aromatic) for _ in range(auxiliary_layers)
        )
        self.auxiliary_role = nn.ModuleList(
            nn.Linear(config.d_model, 2) for _ in range(auxiliary_layers)
        )

    def forward(self, feature_map: torch.Tensor) -> dict[str, object]:
        if feature_map.ndim != 4 or feature_map.size(1) != self.config.input_channels:
            raise ValueError("Atom Query feature map has incompatible shape")
        memory = self.input_projection(feature_map)
        height, width = memory.shape[-2:]
        memory = memory.flatten(2).transpose(1, 2)
        memory = memory + position_encoding_2d(
            height,
            width,
            memory.size(-1),
            memory.device,
            memory.dtype,
        ).unsqueeze(0)
        query_content = self.queries.weight + self.reference_projection(
            self.reference_points
        )
        queries = query_content.unsqueeze(0).expand(feature_map.size(0), -1, -1)
        hidden = queries
        intermediate = []
        for layer in self.decoder.layers:
            hidden = layer(hidden, memory)
            intermediate.append(
                self.decoder.norm(hidden) if self.decoder.norm is not None else hidden
            )
        hidden = intermediate[-1]
        cardinality_hidden = self.cardinality_token.weight.unsqueeze(0).expand(
            feature_map.size(0), -1, -1
        )
        for layer in self.cardinality_decoder:
            cardinality_hidden = layer(cardinality_hidden, hidden, memory)
        cardinality_hidden = self.cardinality_norm(cardinality_hidden).squeeze(1)
        reference_logits = torch.logit(
            self.reference_points.clamp(1e-4, 1.0 - 1e-4)
        )
        coarse_centers = (
            reference_logits.unsqueeze(0) + self.center(hidden)
        ).sigmoid()
        local_features = self.local_projection(feature_map)
        offsets = torch.tensor(
            (
                (-1.0, -1.0),
                (0.0, -1.0),
                (1.0, -1.0),
                (-1.0, 0.0),
                (0.0, 0.0),
                (1.0, 0.0),
                (-1.0, 1.0),
                (0.0, 1.0),
                (1.0, 1.0),
            ),
            device=feature_map.device,
            dtype=feature_map.dtype,
        ) * self.config.local_radius
        sample_points = coarse_centers[:, :, None, :] + offsets[None, None, :, :]
        sample_grid = sample_points.clamp(0.0, 1.0).mul(2.0).sub(1.0)
        sampled = F.grid_sample(
            local_features,
            sample_grid.reshape(feature_map.size(0), -1, 1, 2),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        sampled = sampled.squeeze(-1).transpose(1, 2).reshape(
            feature_map.size(0),
            self.config.queries,
            self.config.local_feature_dim * 9,
        )
        refinement = self.local_refinement(torch.cat((hidden, sampled), dim=-1))
        role_logits = self.role(torch.cat((hidden, sampled), dim=-1))
        first_centers = (
            coarse_centers + refinement.tanh() * self.config.local_radius
        ).clamp(0.0, 1.0)
        second_radius = self.config.local_radius * 2.0
        second_sample_points = (
            first_centers[:, :, None, :] + 2.0 * offsets[None, None, :, :]
        )
        second_sample_grid = second_sample_points.clamp(0.0, 1.0).mul(2.0).sub(1.0)
        second_sampled = F.grid_sample(
            local_features,
            second_sample_grid.reshape(feature_map.size(0), -1, 1, 2),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        second_sampled = second_sampled.squeeze(-1).transpose(1, 2).reshape(
            feature_map.size(0),
            self.config.queries,
            self.config.local_feature_dim * 9,
        )
        second_refinement = self.second_local_refinement(
            torch.cat((hidden, second_sampled), dim=-1)
        )
        centers = (
            first_centers + second_refinement.tanh() * second_radius
        ).clamp(0.0, 1.0)
        selection_local_features = self.selection_local_projection(feature_map)
        selection_points = (
            centers[:, :, None, :] + 0.25 * offsets[None, None, :, :]
        )
        selection_grid = selection_points.clamp(0.0, 1.0).mul(2.0).sub(1.0)
        selection_sampled = F.grid_sample(
            selection_local_features,
            selection_grid.reshape(feature_map.size(0), -1, 1, 2),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        selection_sampled = selection_sampled.squeeze(-1).transpose(1, 2).reshape(
            feature_map.size(0),
            self.config.queries,
            self.config.local_feature_dim * 9,
        )
        selection_input = torch.cat((hidden, selection_sampled), dim=-1)
        selection_context = self.selection_context_projection(selection_input)
        selection_context = selection_context + self.selection_position_projection(
            centers
        )
        selection_context = self.selection_context(selection_context)
        duplicate_features = F.normalize(
            self.duplicate_projection(selection_context),
            dim=-1,
        )
        duplicate_similarity = torch.matmul(
            duplicate_features,
            duplicate_features.transpose(1, 2),
        )
        coordinate_delta = (
            centers[:, :, None, :] - centers[:, None, :, :]
        ).abs()
        duplicate_geometry = torch.cat(
            (
                coordinate_delta,
                coordinate_delta
                .square()
                .sum(dim=-1, keepdim=True)
                .clamp_min(1e-12)
                .sqrt(),
            ),
            dim=-1,
        )
        duplicate_logits = (
            self.duplicate_similarity_scale * duplicate_similarity
            + self.duplicate_geometry(duplicate_geometry).squeeze(-1)
        )
        base_presence_logits = self.presence(hidden).squeeze(-1)
        ranking_presence_logits = self.ranking_presence(hidden).squeeze(-1)
        local_presence_logits = self.selection(selection_input).squeeze(-1)
        selection_context_logits = self.selection_context_score(
            selection_context
        ).squeeze(-1)
        result = {
            "presence_logits": base_presence_logits,
            "base_presence_logits": base_presence_logits,
            "ranking_presence_logits": ranking_presence_logits,
            "local_presence_logits": local_presence_logits,
            "selection_context_logits": selection_context_logits,
            "count_logits": self.count_head(cardinality_hidden),
            "count_value": self.count_regression(cardinality_hidden).squeeze(-1),
            "cardinality_hidden": cardinality_hidden,
            "centers": centers,
            "first_centers": first_centers,
            "coarse_centers": coarse_centers,
            "aromatic_logits": self.aromatic(hidden).squeeze(-1),
            "role_logits": role_logits,
            "selection_logits": (
                ranking_presence_logits
                + local_presence_logits
                + selection_context_logits
            ),
            "duplicate_logits": duplicate_logits,
            "selection_features": selection_context,
        }
        auxiliary_outputs = []
        for stage, presence, count_head, center, aromatic, role in zip(
            intermediate[:-1],
            self.auxiliary_presence,
            self.auxiliary_count_heads,
            self.auxiliary_centers,
            self.auxiliary_aromatic,
            self.auxiliary_role,
        ):
            stage_centers = (
                reference_logits.unsqueeze(0) + center(stage)
            ).sigmoid()
            auxiliary_outputs.append(
                {
                    "presence_logits": presence(stage).squeeze(-1),
                    "count_logits": count_head(stage.mean(dim=1)),
                    "centers": stage_centers,
                    "coarse_centers": stage_centers,
                    "aromatic_logits": aromatic(stage).squeeze(-1),
                    "role_logits": role(stage),
                }
            )
        result["auxiliary_outputs"] = tuple(auxiliary_outputs)
        return result


def hungarian_assignments(
    output: dict[str, torch.Tensor],
    anchor_points: torch.Tensor,
    anchor_mask: torch.Tensor,
    role_targets: torch.Tensor | None = None,
    *,
    role_matching_cost_weight: float = 0.0,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Match each target to the closest unique final query center.

    Query confidence must not influence positive-sample identity. Otherwise an
    already confident but less accurate query can repeatedly steal the target
    from a closer query and reinforce its own score.
    """

    if role_matching_cost_weight < 0:
        raise ValueError("Role matching cost weight must be nonnegative")
    if role_matching_cost_weight and role_targets is None:
        raise ValueError("Role matching requires role targets")
    assignments = []
    centers = output["centers"].detach()
    for batch_index in range(centers.size(0)):
        targets = anchor_points[batch_index][anchor_mask[batch_index]]
        if not len(targets):
            empty = torch.empty(0, dtype=torch.long, device=centers.device)
            assignments.append((empty, empty))
            continue
        cost = stable_pairwise_distances(centers[batch_index], targets)
        if role_matching_cost_weight:
            target_roles = role_targets[batch_index][anchor_mask[batch_index]].long()
            role_probabilities = output["role_logits"][batch_index].detach().softmax(-1)
            cost = cost - role_matching_cost_weight * role_probabilities[:, target_roles]
        query_indices, target_indices = linear_sum_assignment(cost.cpu().numpy())
        assignments.append(
            (
                torch.as_tensor(query_indices, dtype=torch.long, device=centers.device),
                torch.as_tensor(target_indices, dtype=torch.long, device=centers.device),
            )
        )
    return assignments


def coordinate_hungarian_assignments(
    output: dict[str, torch.Tensor],
    anchor_points: torch.Tensor,
    anchor_mask: torch.Tensor,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Assign the closest unique final query centers without using scores."""

    assignments = []
    centers = output["centers"].detach()
    for batch_index in range(centers.size(0)):
        targets = anchor_points[batch_index][anchor_mask[batch_index]]
        if not len(targets):
            empty = torch.empty(0, dtype=torch.long, device=centers.device)
            assignments.append((empty, empty))
            continue
        cost = stable_pairwise_distances(centers[batch_index], targets)
        query_indices, target_indices = linear_sum_assignment(cost.cpu().numpy())
        assignments.append(
            (
                torch.as_tensor(query_indices, dtype=torch.long, device=centers.device),
                torch.as_tensor(target_indices, dtype=torch.long, device=centers.device),
            )
        )
    return assignments


def poisson_binomial_log_probs(presence_logits: torch.Tensor) -> torch.Tensor:
    """Return the count distribution induced by independent query probabilities."""

    if presence_logits.ndim != 2:
        raise ValueError("Presence logits must have shape [batch, queries]")
    probabilities = presence_logits.float().sigmoid()
    distribution = torch.zeros(
        (presence_logits.size(0), presence_logits.size(1) + 1),
        device=presence_logits.device,
        dtype=torch.float32,
    )
    distribution[:, 0] = 1.0
    for query_index in range(presence_logits.size(1)):
        probability = probabilities[:, query_index : query_index + 1]
        previous = distribution.clone()
        distribution[:, 0] = previous[:, 0] * (1.0 - probability[:, 0])
        distribution[:, 1:] = (
            previous[:, 1:] * (1.0 - probability)
            + previous[:, :-1] * probability
        )
    return distribution.clamp_min(1e-30).log()


def atom_query_loss(
    output: dict[str, torch.Tensor],
    anchor_points: torch.Tensor,
    anchor_mask: torch.Tensor,
    aromatic_targets: torch.Tensor,
    *,
    role_targets: torch.Tensor | None = None,
    no_object_weight: float = 1.0,
    center_loss_weight: float = 10.0,
    coarse_center_loss_weight: float = 2.0,
    center_tail_loss_weight: float = 0.0,
    role_loss_weight: float = 0.0,
    role_matching_cost_weight: float = 0.0,
    cardinality_loss_weight: float = 0.01,
    count_loss_weight: float = 0.0,
    count_regression_loss_weight: float = 0.0,
    selection_loss_weight: float = 0.0,
    selection_ranking_loss_weight: float = 0.0,
    duplicate_loss_weight: float = 0.0,
    set_count_loss_weight: float = 0.0,
    auxiliary_loss_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    if (
        no_object_weight <= 0
        or center_loss_weight < 0
        or coarse_center_loss_weight < 0
        or center_tail_loss_weight < 0
        or role_loss_weight < 0
        or role_matching_cost_weight < 0
        or cardinality_loss_weight < 0
        or count_loss_weight < 0
        or count_regression_loss_weight < 0
        or selection_loss_weight < 0
        or selection_ranking_loss_weight < 0
        or duplicate_loss_weight < 0
        or set_count_loss_weight < 0
        or auxiliary_loss_weight < 0
    ):
        raise ValueError("Atom Query loss weights are invalid")
    if role_loss_weight and role_targets is None:
        raise ValueError("Role loss requires role targets")
    assignments = hungarian_assignments(
        output,
        anchor_points,
        anchor_mask,
        role_targets,
        role_matching_cost_weight=role_matching_cost_weight,
    )
    presence_targets = torch.zeros_like(output["presence_logits"])
    presence_weights = torch.full_like(output["presence_logits"], no_object_weight)
    center_losses = []
    center_tail_losses = []
    coarse_center_losses = []
    aromatic_losses = []
    role_losses = []
    role_correct = output["role_logits"].sum() * 0.0
    role_total = 0
    for batch_index, (query_indices, target_indices) in enumerate(assignments):
        if not len(query_indices):
            continue
        presence_targets[batch_index, query_indices] = 1.0
        presence_weights[batch_index, query_indices] = 1.0
        targets = anchor_points[batch_index][anchor_mask[batch_index]][target_indices]
        target_aromatic = aromatic_targets[batch_index][anchor_mask[batch_index]][
            target_indices
        ]
        center_errors = output["centers"][batch_index, query_indices] - targets
        center_losses.append(center_errors.abs().mean())
        center_distances = center_errors.float().square().sum(dim=-1).sqrt()
        center_tail_losses.append(
            center_distances.topk(min(3, len(center_distances))).values.mean()
        )
        coarse_center_losses.append(
            F.l1_loss(
                output["coarse_centers"][batch_index, query_indices],
                targets,
            )
        )
        aromatic_losses.append(
            F.binary_cross_entropy_with_logits(
                output["aromatic_logits"][batch_index, query_indices],
                target_aromatic,
            )
        )
        if role_targets is not None:
            target_roles = role_targets[batch_index][anchor_mask[batch_index]][
                target_indices
            ].long()
            role_losses.append(
                F.cross_entropy(
                    output["role_logits"][batch_index, query_indices],
                    target_roles,
                )
            )
            role_correct = role_correct + (
                output["role_logits"][batch_index, query_indices].argmax(dim=-1)
                == target_roles
            ).sum()
            role_total += len(target_roles)
    presence_loss = (
        F.binary_cross_entropy_with_logits(
            output["presence_logits"],
            presence_targets,
            reduction="none",
        )
        * presence_weights
    ).sum() / presence_weights.sum().clamp_min(1.0)
    center_loss = (
        torch.stack(center_losses).mean()
        if center_losses
        else output["centers"].sum() * 0.0
    )
    center_tail_loss = (
        torch.stack(center_tail_losses).mean()
        if center_tail_losses
        else output["centers"].sum() * 0.0
    )
    aromatic_loss = (
        torch.stack(aromatic_losses).mean()
        if aromatic_losses
        else output["aromatic_logits"].sum() * 0.0
    )
    role_loss = (
        torch.stack(role_losses).mean()
        if role_losses
        else output["role_logits"].sum() * 0.0
    )
    role_accuracy = role_correct / max(role_total, 1)
    coarse_center_loss = (
        torch.stack(coarse_center_losses).mean()
        if coarse_center_losses
        else output["coarse_centers"].sum() * 0.0
    )
    expected_counts = anchor_mask.sum(dim=1).to(output["presence_logits"].dtype)
    if int(expected_counts.max()) > output["count_logits"].size(-1) - 1:
        raise ValueError("Atom count exceeds the configured query capacity")
    predicted_counts = output["presence_logits"].sigmoid().sum(dim=1)
    cardinality_loss = (
        (predicted_counts - expected_counts).abs() / expected_counts.clamp_min(1.0)
    ).mean()
    count_loss = F.cross_entropy(
        output["count_logits"],
        expected_counts.long(),
    )
    count_regression_loss = (
        F.smooth_l1_loss(output["count_value"], expected_counts, beta=1.0)
        if "count_value" in output
        else count_loss * 0.0
    )
    set_count_loss = F.nll_loss(
        poisson_binomial_log_probs(output["presence_logits"]),
        expected_counts.long(),
    )
    selection_loss = count_loss * 0.0
    selection_ranking_loss = count_loss * 0.0
    duplicate_loss = count_loss * 0.0
    if "selection_logits" in output:
        selection_targets = torch.zeros_like(output["selection_logits"])
        selection_assignments = coordinate_hungarian_assignments(
            output,
            anchor_points,
            anchor_mask,
        )
        ranking_losses = []
        for batch_index, (query_indices, _) in enumerate(selection_assignments):
            selection_targets[batch_index, query_indices] = 1.0
            negative_mask = torch.ones(
                output["selection_logits"].size(1),
                dtype=torch.bool,
                device=output["selection_logits"].device,
            )
            negative_mask[query_indices] = False
            positive_scores = output["selection_logits"][batch_index, query_indices]
            negative_scores = output["selection_logits"][batch_index, negative_mask]
            if len(positive_scores) and len(negative_scores):
                hard_negative_count = min(len(negative_scores), 2 * len(positive_scores))
                hard_negative_scores = negative_scores.topk(hard_negative_count).values
                ranking_losses.append(
                    F.softplus(
                        hard_negative_scores[:, None] - positive_scores[None, :]
                    ).mean()
                )
        positive_count = selection_targets.sum().clamp_min(1.0)
        negative_count = (selection_targets.numel() - positive_count).clamp_min(1.0)
        positive_weight = (negative_count / positive_count).clamp(1.0, 20.0)
        selection_loss = F.binary_cross_entropy_with_logits(
            output["selection_logits"],
            selection_targets,
            pos_weight=positive_weight,
        )
        if ranking_losses:
            selection_ranking_loss = torch.stack(ranking_losses).mean()
    if "duplicate_logits" in output:
        duplicate_losses = []
        for batch_index, (query_indices, _) in enumerate(
            coordinate_hungarian_assignments(output, anchor_points, anchor_mask)
        ):
            targets = anchor_points[batch_index][anchor_mask[batch_index]]
            if not len(targets):
                continue
            nearest_targets = stable_pairwise_distances(
                output["centers"][batch_index].detach(),
                targets,
            ).argmin(dim=1)
            assigned = torch.zeros(
                output["duplicate_logits"].size(1),
                dtype=torch.bool,
                device=output["duplicate_logits"].device,
            )
            assigned[query_indices] = True
            pair_mask = assigned[:, None] | assigned[None, :]
            pair_mask = pair_mask & torch.triu(
                torch.ones_like(pair_mask),
                diagonal=1,
            )
            labels = nearest_targets[:, None] == nearest_targets[None, :]
            pair_logits = output["duplicate_logits"][batch_index][pair_mask]
            pair_labels = labels[pair_mask].to(pair_logits.dtype)
            if not len(pair_logits):
                continue
            positives = pair_labels.sum().clamp_min(1.0)
            negatives = (pair_labels.numel() - positives).clamp_min(1.0)
            duplicate_losses.append(
                F.binary_cross_entropy_with_logits(
                    pair_logits,
                    pair_labels,
                    pos_weight=(negatives / positives).clamp(1.0, 20.0),
                )
            )
        if duplicate_losses:
            duplicate_loss = torch.stack(duplicate_losses).mean()
    loss = (
        presence_loss
        + center_loss_weight * center_loss
        + coarse_center_loss_weight * coarse_center_loss
        + center_tail_loss_weight * center_tail_loss
        + 0.25 * aromatic_loss
        + role_loss_weight * role_loss
        + cardinality_loss_weight * cardinality_loss
        + count_loss_weight * count_loss
        + count_regression_loss_weight * count_regression_loss
        + selection_loss_weight * selection_loss
        + selection_ranking_loss_weight * selection_ranking_loss
        + duplicate_loss_weight * duplicate_loss
        + set_count_loss_weight * set_count_loss
    )
    auxiliary_losses = []
    for auxiliary_output in output.get("auxiliary_outputs", ()):
        auxiliary_losses.append(
            atom_query_loss(
                auxiliary_output,
                anchor_points,
                anchor_mask,
                aromatic_targets,
                role_targets=role_targets,
                no_object_weight=no_object_weight,
                center_loss_weight=center_loss_weight,
                coarse_center_loss_weight=coarse_center_loss_weight,
                center_tail_loss_weight=center_tail_loss_weight,
                role_loss_weight=0.0,
                role_matching_cost_weight=role_matching_cost_weight,
                cardinality_loss_weight=cardinality_loss_weight,
                count_loss_weight=count_loss_weight,
                count_regression_loss_weight=count_regression_loss_weight,
                selection_loss_weight=0.0,
                selection_ranking_loss_weight=0.0,
                duplicate_loss_weight=0.0,
                set_count_loss_weight=set_count_loss_weight,
                auxiliary_loss_weight=0.0,
            )["loss"]
        )
    auxiliary_loss = (
        torch.stack(auxiliary_losses).mean()
        if auxiliary_losses
        else loss * 0.0
    )
    loss = loss + auxiliary_loss_weight * auxiliary_loss
    return {
        "loss": loss,
        "presence_loss": presence_loss,
        "center_loss": center_loss,
        "center_tail_loss": center_tail_loss,
        "coarse_center_loss": coarse_center_loss,
        "aromatic_loss": aromatic_loss,
        "role_loss": role_loss,
        "role_accuracy": role_accuracy,
        "cardinality_loss": cardinality_loss,
        "count_loss": count_loss,
        "count_regression_loss": count_regression_loss,
        "selection_loss": selection_loss,
        "selection_ranking_loss": selection_ranking_loss,
        "duplicate_loss": duplicate_loss,
        "set_count_loss": set_count_loss,
        "auxiliary_loss": auxiliary_loss,
    }


def apply_atom_query_vision_updates(base_model, payload: dict) -> None:
    """Apply the optional trainable vision subset stored with a query checkpoint."""

    updates = payload.get("base_vision_updates") or {}
    if not updates:
        return
    state = base_model.vision_encoder.state_dict()
    unexpected = sorted(set(updates) - set(state))
    if unexpected:
        raise ValueError(f"Atom Query vision checkpoint has unknown tensors: {unexpected}")
    for name, value in updates.items():
        if state[name].shape != value.shape:
            raise ValueError(f"Atom Query vision tensor shape differs for {name}")
        state[name] = value
    base_model.vision_encoder.load_state_dict(state)


def apply_atom_query_dense_head(base_model, payload: dict) -> None:
    """Load an optional shared-backbone dense vertex head from a query checkpoint."""

    updates = payload.get("dense_vertex_head")
    if updates is None:
        raise ValueError("Atom Query checkpoint does not contain a dense vertex head")
    if base_model.vertex_head is None:
        raise ValueError("Base model does not have a dense vertex head")
    state = base_model.vertex_head.state_dict()
    unexpected = sorted(set(updates) - set(state))
    missing = sorted(set(state) - set(updates))
    if unexpected or missing:
        raise ValueError(
            "Atom Query dense head architecture mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    for name, value in updates.items():
        if state[name].shape != value.shape:
            raise ValueError(f"Atom Query dense head tensor shape differs for {name}")
    base_model.vertex_head.load_state_dict(updates)


def atom_query_ranking_logits(output: dict[str, torch.Tensor]) -> torch.Tensor:
    """Return final-coordinate-aware ranking scores when they are available."""

    return output.get("selection_logits", output["presence_logits"])


@torch.no_grad()
def decode_atom_queries(
    output: dict[str, torch.Tensor],
    *,
    threshold: float = 0.5,
    maximum_vertices: int = 128,
) -> list[list[dict[str, float | bool]]]:
    probabilities = atom_query_ranking_logits(output).sigmoid()
    aromatic = output["aromatic_logits"].sigmoid()
    decoded = []
    for batch_index in range(probabilities.size(0)):
        scores, indices = probabilities[batch_index].topk(
            min(maximum_vertices, probabilities.size(1))
        )
        vertices = []
        for score, index in zip(scores.tolist(), indices.tolist()):
            if score < threshold:
                break
            vertices.append(
                {
                    "x": float(output["centers"][batch_index, index, 0]),
                    "y": float(output["centers"][batch_index, index, 1]),
                    "score": score,
                    "aromatic": bool(aromatic[batch_index, index] >= 0.5),
                }
            )
        decoded.append(vertices)
    return decoded


@torch.no_grad()
def decode_atom_queries_by_cardinality(
    output: dict[str, torch.Tensor],
    *,
    maximum_vertices: int = 128,
) -> list[list[dict[str, float | bool]]]:
    count_probabilities = output["presence_logits"].sigmoid()
    ranking_probabilities = atom_query_ranking_logits(output).sigmoid()
    aromatic = output["aromatic_logits"].sigmoid()
    decoded = []
    for batch_index in range(count_probabilities.size(0)):
        predicted_count = round(float(count_probabilities[batch_index].sum()))
        predicted_count = min(
            max(predicted_count, 0),
            maximum_vertices,
            ranking_probabilities.size(1),
        )
        scores, indices = ranking_probabilities[batch_index].topk(predicted_count)
        decoded.append(
            [
                {
                    "x": float(output["centers"][batch_index, index, 0]),
                    "y": float(output["centers"][batch_index, index, 1]),
                    "score": float(score),
                    "aromatic": bool(aromatic[batch_index, index] >= 0.5),
                }
                for score, index in zip(scores.tolist(), indices.tolist())
            ]
        )
    return decoded


@torch.no_grad()
def decode_atom_queries_by_count_head(
    output: dict[str, torch.Tensor],
    *,
    maximum_vertices: int = 128,
) -> list[list[dict[str, float | bool]]]:
    probabilities = atom_query_ranking_logits(output).sigmoid()
    aromatic = output["aromatic_logits"].sigmoid()
    predicted_counts = output["count_logits"].argmax(dim=-1)
    decoded = []
    for batch_index, predicted_count in enumerate(predicted_counts.tolist()):
        predicted_count = min(
            max(int(predicted_count), 0),
            maximum_vertices,
            probabilities.size(1),
        )
        scores, indices = probabilities[batch_index].topk(predicted_count)
        decoded.append(
            [
                {
                    "x": float(output["centers"][batch_index, index, 0]),
                    "y": float(output["centers"][batch_index, index, 1]),
                    "score": float(score),
                    "aromatic": bool(aromatic[batch_index, index] >= 0.5),
                }
                for score, index in zip(scores.tolist(), indices.tolist())
            ]
        )
    return decoded


@torch.no_grad()
def decode_atom_queries_by_count_regression(
    output: dict[str, torch.Tensor],
    *,
    maximum_vertices: int = 128,
) -> list[list[dict[str, float | bool]]]:
    probabilities = atom_query_ranking_logits(output).sigmoid()
    aromatic = output["aromatic_logits"].sigmoid()
    predicted_counts = output["count_value"].round().long()
    decoded = []
    for batch_index, predicted_count in enumerate(predicted_counts.tolist()):
        predicted_count = min(
            max(int(predicted_count), 0),
            maximum_vertices,
            probabilities.size(1),
        )
        scores, indices = probabilities[batch_index].topk(predicted_count)
        decoded.append(
            [
                {
                    "x": float(output["centers"][batch_index, index, 0]),
                    "y": float(output["centers"][batch_index, index, 1]),
                    "score": float(score),
                    "aromatic": bool(aromatic[batch_index, index] >= 0.5),
                }
                for score, index in zip(scores.tolist(), indices.tolist())
            ]
        )
    return decoded


@torch.no_grad()
def decode_atom_queries_by_known_counts(
    output: dict[str, torch.Tensor],
    counts: torch.Tensor,
    *,
    maximum_vertices: int = 128,
) -> list[list[dict[str, float | bool]]]:
    """Decode top-k queries with externally supplied counts for evaluation only."""

    probabilities = output["presence_logits"].sigmoid()
    aromatic = output["aromatic_logits"].sigmoid()
    if counts.ndim != 1 or counts.size(0) != probabilities.size(0):
        raise ValueError("Known counts must have shape [batch]")
    decoded = []
    for batch_index, count in enumerate(counts.tolist()):
        count = min(
            max(int(count), 0),
            maximum_vertices,
            probabilities.size(1),
        )
        scores, indices = probabilities[batch_index].topk(count)
        decoded.append(
            [
                {
                    "x": float(output["centers"][batch_index, index, 0]),
                    "y": float(output["centers"][batch_index, index, 1]),
                    "score": float(score),
                    "aromatic": bool(aromatic[batch_index, index] >= 0.5),
                }
                for score, index in zip(scores.tolist(), indices.tolist())
            ]
        )
    return decoded


@torch.no_grad()
def decode_atom_queries_by_selection_known_counts(
    output: dict[str, torch.Tensor],
    counts: torch.Tensor,
    *,
    maximum_vertices: int = 128,
) -> list[list[dict[str, float | bool]]]:
    """Decode top-k queries by the learned local selection score."""

    probabilities = output["selection_logits"].sigmoid()
    aromatic = output["aromatic_logits"].sigmoid()
    if counts.ndim != 1 or counts.size(0) != probabilities.size(0):
        raise ValueError("Known counts must have shape [batch]")
    decoded = []
    for batch_index, count in enumerate(counts.tolist()):
        count = min(
            max(int(count), 0),
            maximum_vertices,
            probabilities.size(1),
        )
        scores, indices = probabilities[batch_index].topk(count)
        decoded.append(
            [
                {
                    "x": float(output["centers"][batch_index, index, 0]),
                    "y": float(output["centers"][batch_index, index, 1]),
                    "score": float(score),
                    "aromatic": bool(aromatic[batch_index, index] >= 0.5),
                }
                for score, index in zip(scores.tolist(), indices.tolist())
            ]
        )
    return decoded


@torch.no_grad()
def decode_atom_queries_by_duplicate_penalty_known_counts(
    output: dict[str, torch.Tensor],
    counts: torch.Tensor,
    *,
    duplicate_penalty: float = 1.0,
    maximum_vertices: int = 128,
) -> list[list[dict[str, float | bool]]]:
    """Greedily rerank selection scores with a learned continuous duplicate penalty."""

    if duplicate_penalty < 0:
        raise ValueError("Duplicate penalty must be nonnegative")
    selection_logits = output["selection_logits"].float()
    duplicate_probabilities = output["duplicate_logits"].float().sigmoid()
    aromatic = output["aromatic_logits"].sigmoid()
    if counts.ndim != 1 or counts.size(0) != selection_logits.size(0):
        raise ValueError("Known counts must have shape [batch]")
    decoded = []
    for batch_index, count in enumerate(counts.tolist()):
        count = min(
            max(int(count), 0),
            maximum_vertices,
            selection_logits.size(1),
        )
        available = torch.ones(
            selection_logits.size(1),
            dtype=torch.bool,
            device=selection_logits.device,
        )
        selected = []
        for _ in range(count):
            adjusted = selection_logits[batch_index].clone()
            if selected:
                duplicate_probability = duplicate_probabilities[
                    batch_index, :, selected
                ].max(dim=1).values
                adjusted = adjusted - duplicate_penalty * duplicate_probability
            adjusted = adjusted.masked_fill(~available, float("-inf"))
            index = int(adjusted.argmax())
            selected.append(index)
            available[index] = False
        decoded.append(
            [
                {
                    "x": float(output["centers"][batch_index, index, 0]),
                    "y": float(output["centers"][batch_index, index, 1]),
                    "score": float(selection_logits[batch_index, index].sigmoid()),
                    "aromatic": bool(aromatic[batch_index, index] >= 0.5),
                }
                for index in selected
            ]
        )
    return decoded


@torch.no_grad()
def decode_atom_queries_by_assignment_oracle(
    output: dict[str, torch.Tensor],
    target_points: torch.Tensor,
    target_mask: torch.Tensor,
) -> list[list[dict[str, float | bool]]]:
    """Select the globally closest unique queries to truth for coverage diagnosis."""

    centers = output["centers"]
    probabilities = output["presence_logits"].sigmoid()
    aromatic = output["aromatic_logits"].sigmoid()
    if target_points.ndim != 3 or target_points.size(0) != centers.size(0):
        raise ValueError("Target points must have shape [batch, targets, 2]")
    if target_mask.shape != target_points.shape[:2]:
        raise ValueError("Target mask shape does not match target points")
    decoded = []
    for batch_index in range(centers.size(0)):
        targets = target_points[batch_index][target_mask[batch_index]].to(
            device=centers.device,
            dtype=centers.dtype,
        )
        if not len(targets):
            decoded.append([])
            continue
        cost = stable_pairwise_distances(centers[batch_index], targets)
        query_indices, _ = linear_sum_assignment(cost.cpu().numpy())
        decoded.append(
            [
                {
                    "x": float(centers[batch_index, index, 0]),
                    "y": float(centers[batch_index, index, 1]),
                    "score": float(probabilities[batch_index, index]),
                    "aromatic": bool(aromatic[batch_index, index] >= 0.5),
                }
                for index in query_indices.tolist()
            ]
        )
    return decoded


@torch.no_grad()
def decode_atom_queries_by_set_count(
    output: dict[str, torch.Tensor],
    *,
    maximum_vertices: int = 128,
) -> list[list[dict[str, float | bool]]]:
    ranking_probabilities = atom_query_ranking_logits(output).sigmoid()
    aromatic = output["aromatic_logits"].sigmoid()
    predicted_counts = poisson_binomial_log_probs(
        output["presence_logits"]
    ).argmax(dim=-1)
    decoded = []
    for batch_index, predicted_count in enumerate(predicted_counts.tolist()):
        predicted_count = min(
            max(int(predicted_count), 0),
            maximum_vertices,
            ranking_probabilities.size(1),
        )
        scores, indices = ranking_probabilities[batch_index].topk(predicted_count)
        decoded.append(
            [
                {
                    "x": float(output["centers"][batch_index, index, 0]),
                    "y": float(output["centers"][batch_index, index, 1]),
                    "score": float(score),
                    "aromatic": bool(aromatic[batch_index, index] >= 0.5),
                }
                for score, index in zip(scores.tolist(), indices.tolist())
            ]
        )
    return decoded


def load_atom_query_state(
    detector: AtomQueryDetector,
    state_dict: dict[str, torch.Tensor],
) -> None:
    incompatible = detector.load_state_dict(state_dict, strict=False)
    allowed_missing = {
        key
        for key in detector.state_dict()
        if key.startswith("count_head.") or key.startswith("auxiliary_")
        or key.startswith("count_regression.")
        or key.startswith("role.")
        or key.startswith("second_local_refinement.")
        or key.startswith("cardinality_token.")
        or key.startswith("cardinality_decoder.")
        or key.startswith("cardinality_norm.")
        or key.startswith("selection.")
        or key.startswith("selection_local_projection.")
        or key.startswith("selection_context_projection.")
        or key.startswith("selection_position_projection.")
        or key.startswith("selection_context.")
        or key.startswith("selection_context_score.")
        or key.startswith("ranking_presence.")
        or key.startswith("duplicate_projection.")
        or key.startswith("duplicate_geometry.")
        or key == "duplicate_similarity_scale"
    }
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    if unexpected or missing - allowed_missing:
        raise ValueError(
            "Atom Query checkpoint architecture mismatch: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    if any(key.startswith("selection_local_projection.") for key in missing):
        detector.selection_local_projection.load_state_dict(
            detector.local_projection.state_dict()
        )
    if any(key.startswith("ranking_presence.") for key in missing):
        detector.ranking_presence.load_state_dict(detector.presence.state_dict())


def snap_queries_to_dense_candidates(
    queries: list[dict[str, float | bool]],
    candidates: list[dict[str, float | bool]],
    *,
    image_size: int = 768,
    maximum_distance_pixels: float = 64.0,
    score_weight_pixels: float = 2.0,
) -> list[dict[str, float | bool]]:
    if not queries or not candidates:
        return [dict(item) for item in queries]
    query_points = torch.tensor(
        [(float(item["x"]), float(item["y"])) for item in queries],
        dtype=torch.float32,
    )
    candidate_points = torch.tensor(
        [(float(item["x"]), float(item["y"])) for item in candidates],
        dtype=torch.float32,
    )
    distances = stable_pairwise_distances(query_points, candidate_points).mul(image_size)
    scores = torch.tensor(
        [float(item["score"]) for item in candidates],
        dtype=torch.float32,
    )
    cost = distances - score_weight_pixels * scores[None, :]
    query_indices, candidate_indices = linear_sum_assignment(cost.numpy())
    snapped = [dict(item) for item in queries]
    for query_index, candidate_index in zip(query_indices, candidate_indices):
        if float(distances[query_index, candidate_index]) > maximum_distance_pixels:
            continue
        candidate = candidates[int(candidate_index)]
        snapped[int(query_index)]["x"] = float(candidate["x"])
        snapped[int(query_index)]["y"] = float(candidate["y"])
        snapped[int(query_index)]["dense_score"] = float(candidate["score"])
    return snapped
