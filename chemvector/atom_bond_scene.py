"""Joint AtomObject/BondObject set prediction with explicit incidence pointers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
from torch.nn import functional as F

from chemvector.atom_query import position_encoding_2d
from chemvector.bond_object import BOND_TYPES, DIRECTIONS
from chemvector.model import ChemVectorConfig, ChemVisionEncoder


ATOM_ROLES = ("IMPLICIT", "TEXT")


@dataclass(frozen=True, slots=True)
class AtomBondSceneConfig:
    atom_queries: int = 128
    bond_queries: int = 128
    d_model: int = 192
    decoder_layers: int = 2
    decoder_heads: int = 6
    decoder_ffn_dim: int = 768
    memory_grid_size: int = 32
    dropout: float = 0.0
    incidence_dim: int = 96
    local_feature_dim: int = 32
    local_radius: float = 32.0 / 768.0

    def __post_init__(self) -> None:
        if min(
            self.atom_queries,
            self.bond_queries,
            self.d_model,
            self.decoder_layers,
            self.decoder_heads,
            self.decoder_ffn_dim,
            self.memory_grid_size,
            self.incidence_dim,
            self.local_feature_dim,
        ) < 1:
            raise ValueError("Atom/Bond Scene dimensions must be positive")
        if self.d_model % self.decoder_heads:
            raise ValueError("decoder_heads must divide d_model")
        if self.d_model % 4:
            raise ValueError("d_model must be divisible by four")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 < self.local_radius < 0.25:
            raise ValueError("local_radius must be in (0, 0.25)")


class AtomBondScenePredictor(nn.Module):
    """Predict joint object sets; bond endpoints point directly to atom queries."""

    model_type = "atom-bond-scene-v0.1"

    def __init__(
        self,
        vision_config: ChemVectorConfig,
        config: AtomBondSceneConfig = AtomBondSceneConfig(),
    ) -> None:
        super().__init__()
        if len(vision_config.vision_dims) < 3:
            raise ValueError("Atom/Bond Scene requires a stride-16 vision stage")
        self.vision_config = vision_config
        self.config = config
        self.vision_encoder = ChemVisionEncoder(vision_config)
        high_channels = vision_config.vision_dims[1]
        self.memory_projection = nn.Sequential(
            nn.Conv2d(high_channels, config.d_model, kernel_size=1),
            nn.GroupNorm(1, config.d_model),
            nn.GELU(),
            nn.Conv2d(config.d_model, config.d_model, kernel_size=3, padding=1),
            nn.GroupNorm(1, config.d_model),
        )
        self.raw_ink_projection = nn.Conv2d(2, config.d_model, kernel_size=1, bias=False)

        atom_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.decoder_heads,
            dim_feedforward=config.decoder_ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        bond_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.decoder_heads,
            dim_feedforward=config.decoder_ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.atom_decoder = nn.TransformerDecoder(
            atom_layer,
            num_layers=config.decoder_layers,
            norm=nn.LayerNorm(config.d_model),
        )
        self.bond_decoder = nn.TransformerDecoder(
            bond_layer,
            num_layers=config.decoder_layers,
            norm=nn.LayerNorm(config.d_model),
        )
        self.atom_query_embedding = nn.Embedding(config.atom_queries, config.d_model)
        self.bond_query_embedding = nn.Embedding(config.bond_queries, config.d_model)
        self.atom_reference_projection = nn.Linear(2, config.d_model)
        self.bond_reference_projection = nn.Linear(2, config.d_model)
        self.register_buffer(
            "atom_reference_points",
            self._reference_grid(config.atom_queries),
            persistent=True,
        )
        self.register_buffer(
            "bond_reference_points",
            self._reference_grid(config.bond_queries),
            persistent=True,
        )

        self.atom_presence = nn.Linear(config.d_model, 1)
        self.atom_anchor = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, 2),
        )
        self.atom_role = nn.Linear(config.d_model, len(ATOM_ROLES))
        self.atom_text_center = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, 2),
        )
        self.atom_text_bbox = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, 4),
        )
        self.atom_count = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.atom_queries + 1),
        )
        self.atom_local_projection = nn.Conv2d(
            vision_config.vision_dims[0], config.local_feature_dim, kernel_size=1
        )
        atom_local_input = config.d_model + 9 * config.local_feature_dim
        self.atom_refinement = nn.Sequential(
            nn.Linear(atom_local_input, config.decoder_ffn_dim),
            nn.GELU(),
            nn.Linear(config.decoder_ffn_dim, 2),
        )
        self.atom_second_refinement = nn.Sequential(
            nn.Linear(atom_local_input, config.decoder_ffn_dim),
            nn.GELU(),
            nn.Linear(config.decoder_ffn_dim, 2),
        )

        self.bond_presence = nn.Linear(config.d_model, 1)
        self.bond_center_vector = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, 4),
        )
        self.bond_display = nn.Linear(config.d_model, len(BOND_TYPES))
        self.bond_chemical = nn.Linear(config.d_model, len(BOND_TYPES))
        self.bond_direction = nn.Linear(config.d_model, len(DIRECTIONS))
        self.bond_count = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.bond_queries + 1),
        )
        self.bond_local_projection = nn.Conv2d(
            vision_config.vision_dims[0], config.local_feature_dim, kernel_size=1
        )
        bond_local_input = config.d_model + 18 * config.local_feature_dim
        self.bond_refinement = nn.Sequential(
            nn.Linear(bond_local_input, config.decoder_ffn_dim),
            nn.GELU(),
            nn.Linear(config.decoder_ffn_dim, 4),
        )
        self.bond_second_refinement = nn.Sequential(
            nn.Linear(bond_local_input, config.decoder_ffn_dim),
            nn.GELU(),
            nn.Linear(config.decoder_ffn_dim, 4),
        )

        self.endpoint_embedding = nn.Embedding(2, config.d_model)
        self.incidence_query = nn.Linear(config.d_model, config.incidence_dim)
        self.incidence_key = nn.Linear(config.d_model, config.incidence_dim)
        self.incidence_geometry_scale = nn.Parameter(torch.tensor(2.0))
        self._reset_parameters()

    @staticmethod
    def _reference_grid(queries: int) -> torch.Tensor:
        columns = math.ceil(math.sqrt(queries))
        rows = math.ceil(queries / columns)
        y, x = torch.meshgrid(
            (torch.arange(rows, dtype=torch.float32) + 0.5) / rows,
            (torch.arange(columns, dtype=torch.float32) + 0.5) / columns,
            indexing="ij",
        )
        return torch.stack((x.flatten(), y.flatten()), dim=-1)[:queries]

    @staticmethod
    def _inverse_sigmoid(values: torch.Tensor) -> torch.Tensor:
        values = values.clamp(1e-4, 1.0 - 1e-4)
        return torch.log(values / (1.0 - values))

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.atom_query_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.bond_query_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.endpoint_embedding.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.atom_anchor[-1].weight)
        nn.init.zeros_(self.atom_anchor[-1].bias)
        nn.init.zeros_(self.bond_center_vector[-1].weight)
        nn.init.zeros_(self.bond_center_vector[-1].bias)
        for head in (
            self.atom_refinement,
            self.atom_second_refinement,
            self.bond_refinement,
            self.bond_second_refinement,
        ):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        nn.init.constant_(self.atom_presence.bias, -2.0)
        nn.init.constant_(self.bond_presence.bias, -2.0)
        with torch.no_grad():
            self.atom_count[-1].bias.zero_()
            self.bond_count[-1].bias.zero_()

    def _memory(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        stroke = self.vision_encoder.forward_stroke_resolution(pixel_values)
        high = self.vision_encoder.forward_high_resolution_from_stroke(stroke)
        high = F.adaptive_avg_pool2d(
            high,
            (self.config.memory_grid_size, self.config.memory_grid_size),
        )
        memory_map = self.memory_projection(high)
        grayscale = pixel_values.add(1.0).mul(0.5).mean(dim=1, keepdim=True)
        ink = (1.0 - grayscale).clamp(0.0, 1.0)
        average = F.adaptive_avg_pool2d(
            ink,
            (self.config.memory_grid_size, self.config.memory_grid_size),
        )
        maximum = F.adaptive_max_pool2d(
            ink,
            (self.config.memory_grid_size, self.config.memory_grid_size),
        )
        memory_map = memory_map + self.raw_ink_projection(
            torch.cat((average, maximum), dim=1)
        )
        memory = memory_map.flatten(2).transpose(1, 2)
        memory = memory + position_encoding_2d(
            self.config.memory_grid_size,
            self.config.memory_grid_size,
            self.config.d_model,
            memory.device,
            memory.dtype,
        ).unsqueeze(0)
        return memory, stroke

    def _sample_local(
        self,
        feature_map: torch.Tensor,
        points: torch.Tensor,
        *,
        radius: float,
    ) -> torch.Tensor:
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
            device=points.device,
            dtype=points.dtype,
        ) * radius
        sample_points = points[..., None, :] + offsets
        grid = sample_points.clamp(0.0, 1.0).mul(2.0).sub(1.0)
        batch = points.size(0)
        sampled = F.grid_sample(
            feature_map,
            grid.reshape(batch, -1, 1, 2),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return sampled.squeeze(-1).transpose(1, 2).reshape(
            *points.shape[:-1], 9 * feature_map.size(1)
        )

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        if pixel_values.ndim != 4 or pixel_values.size(1) != 3:
            raise ValueError("Atom/Bond Scene images must have shape [batch, 3, h, w]")
        memory, stroke = self._memory(pixel_values)
        atom_references = self.atom_reference_points.to(memory.dtype)
        bond_references = self.bond_reference_points.to(memory.dtype)
        atom_queries = (
            self.atom_query_embedding.weight.to(memory.dtype)
            + self.atom_reference_projection(atom_references)
        ).unsqueeze(0).expand(pixel_values.size(0), -1, -1)
        bond_queries = (
            self.bond_query_embedding.weight.to(memory.dtype)
            + self.bond_reference_projection(bond_references)
        ).unsqueeze(0).expand(pixel_values.size(0), -1, -1)
        atom_hidden = self.atom_decoder(atom_queries, memory)
        bond_hidden = self.bond_decoder(bond_queries, memory)

        coarse_atom_anchors = torch.sigmoid(
            self._inverse_sigmoid(atom_references)[None] + self.atom_anchor(atom_hidden)
        )
        atom_local_map = self.atom_local_projection(stroke)
        atom_local = self._sample_local(
            atom_local_map,
            coarse_atom_anchors,
            radius=self.config.local_radius,
        )
        atom_anchors = (
            coarse_atom_anchors
            + self.atom_refinement(torch.cat((atom_hidden, atom_local), dim=-1)).tanh()
            * self.config.local_radius
        ).clamp(0.0, 1.0)
        atom_second_local = self._sample_local(
            atom_local_map,
            atom_anchors,
            radius=self.config.local_radius * 0.5,
        )
        atom_anchors = (
            atom_anchors
            + self.atom_second_refinement(
                torch.cat((atom_hidden, atom_second_local), dim=-1)
            ).tanh()
            * (self.config.local_radius * 0.5)
        ).clamp(0.0, 1.0)
        text_center = torch.sigmoid(
            self._inverse_sigmoid(atom_anchors) + self.atom_text_center(atom_hidden)
        )
        text_bbox_raw = self.atom_text_bbox(atom_hidden)
        text_bbox = torch.cat(
            (text_center, torch.sigmoid(text_bbox_raw[..., 2:] - 3.0)),
            dim=-1,
        )

        center_vector = self.bond_center_vector(bond_hidden)
        bond_centers = torch.sigmoid(
            self._inverse_sigmoid(bond_references)[None] + center_vector[..., :2]
        )
        maximum_half_vector = torch.minimum(bond_centers, 1.0 - bond_centers)
        half_vector = torch.tanh(center_vector[..., 2:]) * maximum_half_vector
        coarse_bond_ports = torch.stack(
            (bond_centers - half_vector, bond_centers + half_vector),
            dim=2,
        )
        bond_local_map = self.bond_local_projection(stroke)
        bond_local = self._sample_local(
            bond_local_map,
            coarse_bond_ports,
            radius=self.config.local_radius,
        ).flatten(2)
        bond_ports = (
            coarse_bond_ports
            + self.bond_refinement(
                torch.cat((bond_hidden, bond_local), dim=-1)
            ).tanh().reshape(pixel_values.size(0), self.config.bond_queries, 2, 2)
            * self.config.local_radius
        ).clamp(0.0, 1.0)
        bond_second_local = self._sample_local(
            bond_local_map,
            bond_ports,
            radius=self.config.local_radius * 0.5,
        ).flatten(2)
        bond_ports = (
            bond_ports
            + self.bond_second_refinement(
                torch.cat((bond_hidden, bond_second_local), dim=-1)
            ).tanh().reshape(pixel_values.size(0), self.config.bond_queries, 2, 2)
            * (self.config.local_radius * 0.5)
        ).clamp(0.0, 1.0)

        endpoint_hidden = (
            bond_hidden[:, :, None, :]
            + self.endpoint_embedding.weight.to(bond_hidden.dtype)[None, None, :, :]
        )
        incidence_queries = self.incidence_query(endpoint_hidden)
        incidence_keys = self.incidence_key(atom_hidden)
        incidence_logits = torch.einsum(
            "bqed,bad->bqea", incidence_queries, incidence_keys
        ) / math.sqrt(self.config.incidence_dim)
        distances = (
            bond_ports[:, :, :, None, :] - atom_anchors[:, None, None, :, :]
        ).square().sum(dim=-1)
        incidence_logits = incidence_logits - F.softplus(
            self.incidence_geometry_scale
        ) * distances * 32.0

        return {
            "atom_presence_logits": self.atom_presence(atom_hidden).squeeze(-1),
            "atom_anchors": atom_anchors,
            "coarse_atom_anchors": coarse_atom_anchors,
            "atom_role_logits": self.atom_role(atom_hidden),
            "atom_text_center": text_center,
            "atom_text_bbox": text_bbox,
            "atom_count_logits": self.atom_count(atom_hidden.mean(dim=1)),
            "bond_presence_logits": self.bond_presence(bond_hidden).squeeze(-1),
            "bond_ports": bond_ports,
            "coarse_bond_ports": coarse_bond_ports,
            "bond_display_logits": self.bond_display(bond_hidden),
            "bond_chemical_logits": self.bond_chemical(bond_hidden),
            "bond_direction_logits": self.bond_direction(bond_hidden),
            "bond_count_logits": self.bond_count(bond_hidden.mean(dim=1)),
            "incidence_logits": incidence_logits,
            "atom_features": atom_hidden,
            "bond_features": bond_hidden,
        }

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def config_dict(self) -> dict:
        return {
            "vision_config": asdict(self.vision_config),
            "atom_bond_scene_config": asdict(self.config),
        }


def unordered_port_cost(predicted: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    forward = (predicted[:, None] - expected[None]).abs().sum(dim=(-1, -2))
    reverse = (predicted[:, None] - expected[None].flip(2)).abs().sum(dim=(-1, -2))
    return torch.minimum(forward, reverse)


def atom_assignments(
    output: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    assignments = []
    for batch_index in range(output["atom_anchors"].size(0)):
        valid = targets["atom_mask"][batch_index]
        anchors = targets["atom_anchors"][batch_index, valid]
        roles = targets["atom_roles"][batch_index, valid]
        cost = 12.0 * torch.cdist(
            output["atom_anchors"][batch_index].float(), anchors.float(), p=1
        )
        cost = cost - output["atom_role_logits"][batch_index].softmax(-1)[:, roles]
        rows, columns = linear_sum_assignment(cost.detach().cpu().numpy())
        assignments.append(
            (
                torch.as_tensor(rows, device=cost.device, dtype=torch.long),
                torch.as_tensor(columns, device=cost.device, dtype=torch.long),
            )
        )
    return assignments


def _target_atom_query_map(
    atom_assignment: tuple[torch.Tensor, torch.Tensor], atom_count: int
) -> torch.Tensor:
    queries, target_offsets = atom_assignment
    result = torch.full(
        (atom_count,), -1, device=queries.device, dtype=torch.long
    )
    result[target_offsets] = queries
    return result


def bond_assignments(
    output: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    atom_matches: list[tuple[torch.Tensor, torch.Tensor]],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    assignments = []
    log_incidence = output["incidence_logits"].log_softmax(-1)
    for batch_index in range(output["bond_ports"].size(0)):
        valid = targets["bond_mask"][batch_index]
        ports = targets["bond_ports"][batch_index, valid]
        display = targets["bond_display"][batch_index, valid]
        chemical = targets["bond_chemical"][batch_index, valid]
        atom_refs = targets["bond_atom_refs"][batch_index, valid]
        atom_count = int(targets["atom_mask"][batch_index].sum())
        target_to_query = _target_atom_query_map(atom_matches[batch_index], atom_count)
        ref_queries = target_to_query[atom_refs]

        predicted_ports = output["bond_ports"][batch_index]
        forward_geometry = (
            predicted_ports[:, None] - ports[None]
        ).abs().sum(dim=(-1, -2))
        reverse_geometry = (
            predicted_ports[:, None] - ports[None].flip(2)
        ).abs().sum(dim=(-1, -2))
        forward_incidence = -(
            log_incidence[batch_index, :, 0][:, ref_queries[:, 0]]
            + log_incidence[batch_index, :, 1][:, ref_queries[:, 1]]
        )
        reverse_incidence = -(
            log_incidence[batch_index, :, 0][:, ref_queries[:, 1]]
            + log_incidence[batch_index, :, 1][:, ref_queries[:, 0]]
        )
        cost = torch.minimum(
            10.0 * forward_geometry + 0.5 * forward_incidence,
            10.0 * reverse_geometry + 0.5 * reverse_incidence,
        )
        cost = cost - output["bond_display_logits"][batch_index].softmax(-1)[:, display]
        cost = cost - output["bond_chemical_logits"][batch_index].softmax(-1)[:, chemical]
        rows, columns = linear_sum_assignment(cost.detach().cpu().numpy())
        assignments.append(
            (
                torch.as_tensor(rows, device=cost.device, dtype=torch.long),
                torch.as_tensor(columns, device=cost.device, dtype=torch.long),
            )
        )
    return assignments


def _balanced_presence_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    positives = targets.sum().clamp_min(1.0)
    negatives = (targets.numel() - positives).clamp_min(1.0)
    return F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=(negatives / positives).clamp_max(16.0),
    )


def atom_bond_scene_loss(
    output: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    atom_matches = atom_assignments(output, targets)
    bond_matches = bond_assignments(output, targets, atom_matches)
    atom_presence_targets = torch.zeros_like(output["atom_presence_logits"])
    bond_presence_targets = torch.zeros_like(output["bond_presence_logits"])
    atom_anchor_losses = []
    atom_role_losses = []
    text_center_losses = []
    text_bbox_losses = []
    bond_port_losses = []
    bond_display_losses = []
    bond_chemical_losses = []
    bond_direction_losses = []
    incidence_losses = []

    for batch_index, (queries, target_offsets) in enumerate(atom_matches):
        atom_presence_targets[batch_index, queries] = 1.0
        valid_indices = torch.where(targets["atom_mask"][batch_index])[0]
        target_indices = valid_indices[target_offsets]
        atom_anchor_losses.append(
            F.smooth_l1_loss(
                output["atom_anchors"][batch_index, queries],
                targets["atom_anchors"][batch_index, target_indices],
                beta=4.0 / 768.0,
            )
        )
        atom_role_losses.append(
            F.cross_entropy(
                output["atom_role_logits"][batch_index, queries],
                targets["atom_roles"][batch_index, target_indices],
            )
        )
        text = targets["atom_text_mask"][batch_index, target_indices]
        if bool(text.any()):
            text_queries = queries[text]
            text_targets = target_indices[text]
            text_center_losses.append(
                F.smooth_l1_loss(
                    output["atom_text_center"][batch_index, text_queries],
                    targets["atom_text_center"][batch_index, text_targets],
                    beta=4.0 / 768.0,
                )
            )
            text_bbox_losses.append(
                F.smooth_l1_loss(
                    output["atom_text_bbox"][batch_index, text_queries],
                    targets["atom_text_bbox"][batch_index, text_targets],
                    beta=4.0 / 768.0,
                )
            )

    log_incidence = output["incidence_logits"].log_softmax(-1)
    for batch_index, (queries, target_offsets) in enumerate(bond_matches):
        bond_presence_targets[batch_index, queries] = 1.0
        valid_indices = torch.where(targets["bond_mask"][batch_index])[0]
        target_indices = valid_indices[target_offsets]
        expected_ports = targets["bond_ports"][batch_index, target_indices]
        predicted_ports = output["bond_ports"][batch_index, queries]
        atom_count = int(targets["atom_mask"][batch_index].sum())
        target_to_query = _target_atom_query_map(atom_matches[batch_index], atom_count)
        atom_refs = targets["bond_atom_refs"][batch_index, target_indices]
        ref_queries = target_to_query[atom_refs]
        forward_geometry = (predicted_ports - expected_ports).abs().sum(dim=(-1, -2))
        reverse_geometry = (
            predicted_ports - expected_ports.flip(1)
        ).abs().sum(dim=(-1, -2))
        forward_incidence = -(
            log_incidence[batch_index, queries, 0, ref_queries[:, 0]]
            + log_incidence[batch_index, queries, 1, ref_queries[:, 1]]
        )
        reverse_incidence = -(
            log_incidence[batch_index, queries, 0, ref_queries[:, 1]]
            + log_incidence[batch_index, queries, 1, ref_queries[:, 0]]
        )
        reverse = (10.0 * reverse_geometry + reverse_incidence) < (
            10.0 * forward_geometry + forward_incidence
        )
        aligned_ports = torch.where(
            reverse[:, None, None], expected_ports.flip(1), expected_ports
        )
        aligned_refs = torch.where(reverse[:, None], ref_queries.flip(1), ref_queries)
        bond_port_losses.append(
            F.smooth_l1_loss(
                predicted_ports,
                aligned_ports,
                beta=4.0 / 768.0,
            )
        )
        incidence_losses.append(
            F.nll_loss(
                log_incidence[batch_index, queries].reshape(-1, log_incidence.size(-1)),
                aligned_refs.reshape(-1),
            )
        )
        bond_display_losses.append(
            F.cross_entropy(
                output["bond_display_logits"][batch_index, queries],
                targets["bond_display"][batch_index, target_indices],
            )
        )
        bond_chemical_losses.append(
            F.cross_entropy(
                output["bond_chemical_logits"][batch_index, queries],
                targets["bond_chemical"][batch_index, target_indices],
            )
        )
        bond_direction_losses.append(
            F.cross_entropy(
                output["bond_direction_logits"][batch_index, queries],
                targets["bond_direction"][batch_index, target_indices],
            )
        )

    atom_presence_loss = _balanced_presence_loss(
        output["atom_presence_logits"], atom_presence_targets
    )
    bond_presence_loss = _balanced_presence_loss(
        output["bond_presence_logits"], bond_presence_targets
    )
    zero = atom_presence_loss * 0.0

    def mean(values: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(values).mean() if values else zero

    atom_count_targets = targets["atom_mask"].sum(dim=1).long()
    bond_count_targets = targets["bond_mask"].sum(dim=1).long()
    atom_count_loss = F.cross_entropy(output["atom_count_logits"], atom_count_targets)
    bond_count_loss = F.cross_entropy(output["bond_count_logits"], bond_count_targets)
    losses = {
        "atom_presence_loss": atom_presence_loss,
        "atom_anchor_loss": mean(atom_anchor_losses),
        "atom_role_loss": mean(atom_role_losses),
        "text_center_loss": mean(text_center_losses),
        "text_bbox_loss": mean(text_bbox_losses),
        "atom_count_loss": atom_count_loss,
        "bond_presence_loss": bond_presence_loss,
        "bond_port_loss": mean(bond_port_losses),
        "bond_display_loss": mean(bond_display_losses),
        "bond_chemical_loss": mean(bond_chemical_losses),
        "bond_direction_loss": mean(bond_direction_losses),
        "bond_count_loss": bond_count_loss,
        "incidence_loss": mean(incidence_losses),
    }
    losses["loss"] = (
        losses["atom_presence_loss"]
        + 64.0 * losses["atom_anchor_loss"]
        + losses["atom_role_loss"]
        + 2.0 * losses["text_center_loss"]
        + 2.0 * losses["text_bbox_loss"]
        + losses["atom_count_loss"]
        + losses["bond_presence_loss"]
        + 64.0 * losses["bond_port_loss"]
        + losses["bond_display_loss"]
        + losses["bond_chemical_loss"]
        + 0.5 * losses["bond_direction_loss"]
        + losses["bond_count_loss"]
        + 2.0 * losses["incidence_loss"]
    )
    return losses
