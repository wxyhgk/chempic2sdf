"""Joint node-edge reasoning over high-recall dense vertex proposals."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F

from chemvector.geometry import stable_pairwise_distances
from chemvector.proposal_keep import ProposalKeepConfig, proposal_indices
from chemvector.vertex import VertexCollator


@dataclass(frozen=True, slots=True)
class ProposalGraphConfig:
    input_channels: int
    node_dim: int = 64
    edge_dim: int = 32
    maximum_proposals: int = 192
    proposal_threshold: float = 0.05
    proposal_nms_kernel_size: int = 3
    matching_radius_pixels: float = 16.0
    line_samples: int = 24
    line_band: int = 3
    line_band_radius_pixels: float = 3.0
    incident_topk: int = 4
    global_layers: int = 0
    global_heads: int = 4
    global_ffn_dim: int = 256
    joint_node_edge_gradients: bool = False

    def __post_init__(self) -> None:
        if min(
            self.input_channels,
            self.node_dim,
            self.edge_dim,
            self.maximum_proposals,
            self.line_samples,
            self.line_band,
            self.incident_topk,
            self.global_heads,
            self.global_ffn_dim,
        ) < 1:
            raise ValueError("Proposal graph dimensions must be positive")
        if self.global_layers < 0:
            raise ValueError("Proposal graph global layers must be nonnegative")
        if self.node_dim % self.global_heads:
            raise ValueError("Proposal graph node dim must be divisible by global heads")
        if not 0.0 < self.proposal_threshold < 1.0:
            raise ValueError("Proposal graph threshold must be in (0, 1)")
        if self.proposal_nms_kernel_size < 1 or self.proposal_nms_kernel_size % 2 == 0:
            raise ValueError("Proposal graph NMS kernel must be a positive odd integer")
        if min(self.matching_radius_pixels, self.line_band_radius_pixels) <= 0:
            raise ValueError("Proposal graph radii must be positive")
        if self.incident_topk >= self.maximum_proposals:
            raise ValueError("Incident top-k must be smaller than maximum proposals")

    def keep_config(self) -> ProposalKeepConfig:
        return ProposalKeepConfig(
            input_channels=self.input_channels,
            maximum_proposals=self.maximum_proposals,
            proposal_threshold=self.proposal_threshold,
            proposal_nms_kernel_size=self.proposal_nms_kernel_size,
            matching_radius_pixels=self.matching_radius_pixels,
        )


class ProposalGraphCollator(VertexCollator):
    """Add a padded atom adjacency matrix to the shared vertex batch."""

    def __call__(self, samples: list[dict]) -> dict[str, torch.Tensor | list[str]]:
        batch = super().__call__(samples)
        maximum_atoms = batch["anchor_points"].size(1)
        adjacency = torch.zeros(
            (len(samples), maximum_atoms, maximum_atoms),
            dtype=torch.bool,
        )
        for sample_index, sample in enumerate(samples):
            for bond in sample["chemmol"]["bonds"]:
                begin = int(bond["begin_atom"])
                end = int(bond["end_atom"])
                if begin == end:
                    raise ValueError(f"Self bond in sample {sample['key']}: atom {begin}")
                if not 0 <= begin < maximum_atoms or not 0 <= end < maximum_atoms:
                    raise ValueError(
                        f"Bond atom outside padded batch in sample {sample['key']}: "
                        f"{begin}-{end}"
                    )
                adjacency[sample_index, begin, end] = True
                adjacency[sample_index, end, begin] = True
        batch["adjacency"] = adjacency
        return batch


def padded_proposal_indices(
    dense_output: dict[str, torch.Tensor],
    config: ProposalGraphConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = proposal_indices(dense_output, config.keep_config())
    padded = torch.full(
        (len(indices), config.maximum_proposals),
        -1,
        dtype=torch.long,
        device=dense_output["vertex_heatmap_logits"].device,
    )
    mask = torch.zeros_like(padded, dtype=torch.bool)
    for batch_index, values in enumerate(indices):
        count = min(len(values), config.maximum_proposals)
        padded[batch_index, :count] = values[:count]
        mask[batch_index, :count] = True
    return padded, mask


def proposal_graph_points(
    dense_output: dict[str, torch.Tensor],
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    height, width = dense_output["vertex_heatmap_logits"].shape[-2:]
    safe = indices.clamp_min(0)
    y = torch.div(safe, width, rounding_mode="floor")
    x = safe % width
    offsets = dense_output["vertex_offset_logits"].sigmoid()
    batch = torch.arange(indices.size(0), device=indices.device)[:, None]
    offset_x = offsets[batch, 0, y, x]
    offset_y = offsets[batch, 1, y, x]
    points = torch.stack(((x + offset_x) / width, (y + offset_y) / height), dim=-1)
    return points, y, x


def proposal_graph_targets(
    points: torch.Tensor,
    proposal_mask: torch.Tensor,
    pair_indices: torch.Tensor,
    anchor_points: torch.Tensor,
    anchor_mask: torch.Tensor,
    adjacency: torch.Tensor,
    *,
    matching_radius_pixels: float,
    image_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assignments = torch.full_like(proposal_mask, -1, dtype=torch.long)
    node_targets = torch.zeros_like(proposal_mask, dtype=torch.float32)
    for batch_index in range(points.size(0)):
        candidate_ids = torch.where(proposal_mask[batch_index])[0]
        target_ids = torch.where(anchor_mask[batch_index])[0]
        if not len(candidate_ids) or not len(target_ids):
            continue
        distances = stable_pairwise_distances(
            points[batch_index, candidate_ids],
            anchor_points[batch_index, target_ids],
        ).mul(image_size)
        proposal_offsets, target_offsets = linear_sum_assignment(distances.cpu().numpy())
        proposal_offsets = torch.as_tensor(
            proposal_offsets,
            device=points.device,
            dtype=torch.long,
        )
        target_offsets = torch.as_tensor(
            target_offsets,
            device=points.device,
            dtype=torch.long,
        )
        accepted = distances[proposal_offsets, target_offsets] <= matching_radius_pixels
        matched_proposals = candidate_ids[proposal_offsets[accepted]]
        matched_targets = target_ids[target_offsets[accepted]]
        assignments[batch_index, matched_proposals] = matched_targets
        node_targets[batch_index, matched_proposals] = 1.0

    begin = pair_indices[:, 0]
    end = pair_indices[:, 1]
    begin_targets = assignments[:, begin]
    end_targets = assignments[:, end]
    mapped = (begin_targets >= 0) & (end_targets >= 0)
    batch = torch.arange(points.size(0), device=points.device)[:, None]
    edge_targets = torch.zeros(
        (points.size(0), pair_indices.size(0)),
        device=points.device,
        dtype=torch.float32,
    )
    safe_begin = begin_targets.clamp_min(0)
    safe_end = end_targets.clamp_min(0)
    edge_targets[mapped] = adjacency[batch.expand_as(mapped), safe_begin, safe_end][mapped].float()
    return node_targets, edge_targets, assignments


class ProposalGraphRefiner(nn.Module):
    """Use visible pair ink to update node keep probabilities."""

    def __init__(self, config: ProposalGraphConfig) -> None:
        super().__init__()
        self.config = config
        node_inputs = config.input_channels + 7
        self.node_encoder = nn.Sequential(
            nn.Linear(node_inputs, config.node_dim),
            nn.GELU(),
            nn.Linear(config.node_dim, config.node_dim),
            nn.GELU(),
        )
        self.global_encoder = None
        if config.global_layers:
            layer = nn.TransformerEncoderLayer(
                d_model=config.node_dim,
                nhead=config.global_heads,
                dim_feedforward=config.global_ffn_dim,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.global_encoder = nn.TransformerEncoder(
                layer,
                num_layers=config.global_layers,
                enable_nested_tensor=False,
            )
        line_inputs = config.line_samples * config.line_band + 3
        self.line_encoder = nn.Sequential(
            nn.Linear(line_inputs, config.edge_dim),
            nn.GELU(),
            nn.Linear(config.edge_dim, config.edge_dim),
            nn.GELU(),
        )
        self.endpoint_encoder = nn.Sequential(
            nn.Linear(config.node_dim * 2, config.edge_dim),
            nn.GELU(),
        )
        self.edge_classifier = nn.Linear(config.edge_dim, 1)
        self.edge_message = nn.Linear(config.edge_dim, config.node_dim)
        self.node_local_classifier = nn.Sequential(
            nn.Linear(config.node_dim + 2, config.node_dim),
            nn.GELU(),
            nn.Linear(config.node_dim, 1),
        )
        self.node_rescue_classifier = nn.Sequential(
            nn.Linear(
                config.node_dim + config.incident_topk + 1,
                config.node_dim,
            ),
            nn.GELU(),
            nn.Linear(config.node_dim, 1),
        )
        nn.init.zeros_(self.node_rescue_classifier[-1].weight)
        nn.init.zeros_(self.node_rescue_classifier[-1].bias)

    def _line_bands(
        self,
        pixel_values: torch.Tensor,
        points: torch.Tensor,
        pair_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        begin = points[:, pair_indices[:, 0]]
        end = points[:, pair_indices[:, 1]]
        delta = end - begin
        pixel_height, pixel_width = pixel_values.shape[-2:]
        delta_pixels = torch.stack(
            (delta[..., 0] * pixel_width, delta[..., 1] * pixel_height),
            dim=-1,
        )
        lengths = delta_pixels.norm(dim=-1).clamp_min(1e-6)
        direction = delta_pixels / lengths.unsqueeze(-1)
        perpendicular = torch.stack((-direction[..., 1], direction[..., 0]), dim=-1)
        perpendicular = torch.stack(
            (
                perpendicular[..., 0] / pixel_width,
                perpendicular[..., 1] / pixel_height,
            ),
            dim=-1,
        )
        positions = torch.linspace(
            0.08,
            0.92,
            self.config.line_samples,
            device=points.device,
            dtype=points.dtype,
        )
        offsets = torch.linspace(
            -self.config.line_band_radius_pixels,
            self.config.line_band_radius_pixels,
            self.config.line_band,
            device=points.device,
            dtype=points.dtype,
        )
        base = begin[:, :, None, :] + delta[:, :, None, :] * positions[None, None, :, None]
        grid = base[:, :, None, :, :] + (
            perpendicular[:, :, None, None, :] * offsets[None, None, :, None, None]
        )
        grid = grid.mul(2.0).sub(1.0)
        grid = grid.reshape(
            pixel_values.size(0),
            pair_indices.size(0) * self.config.line_band,
            self.config.line_samples,
            2,
        )
        gray = pixel_values.mean(dim=1, keepdim=True)
        ink = (1.0 - gray) * 0.5
        sampled = F.grid_sample(
            ink,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled = sampled.reshape(
            pixel_values.size(0),
            pair_indices.size(0),
            self.config.line_band * self.config.line_samples,
        )
        geometry = torch.cat(
            (
                direction,
                (lengths / max(pixel_height, pixel_width)).unsqueeze(-1),
            ),
            dim=-1,
        )
        return sampled, geometry

    def forward(
        self,
        feature_map: torch.Tensor,
        pixel_values: torch.Tensor,
        dense_output: dict[str, torch.Tensor],
        *,
        base_keep_logits: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        indices, proposal_mask = padded_proposal_indices(dense_output, self.config)
        points, y, x = proposal_graph_points(dense_output, indices)
        batch = torch.arange(indices.size(0), device=indices.device)[:, None]
        spatial = feature_map.permute(0, 2, 3, 1)[batch, y, x]
        heatmap = dense_output["vertex_heatmap_logits"].sigmoid()[batch, 0, y, x]
        aromatic = dense_output["vertex_aromatic_logits"].sigmoid()[batch, 0, y, x]
        offsets = dense_output["vertex_offset_logits"].sigmoid().permute(0, 2, 3, 1)[
            batch,
            y,
            x,
        ]
        if base_keep_logits is None:
            base_keep = torch.zeros_like(heatmap)
        else:
            base_keep = base_keep_logits.sigmoid()[batch, 0, y, x]
        node_inputs = torch.cat(
            (
                spatial,
                heatmap.unsqueeze(-1),
                aromatic.unsqueeze(-1),
                offsets,
                points,
                base_keep.unsqueeze(-1),
            ),
            dim=-1,
        )
        node_hidden = self.node_encoder(node_inputs)
        if self.global_encoder is not None:
            node_hidden = self.global_encoder(
                node_hidden,
                src_key_padding_mask=~proposal_mask,
            )
            node_hidden = node_hidden.masked_fill(~proposal_mask.unsqueeze(-1), 0.0)
        pair_indices = torch.triu_indices(
            self.config.maximum_proposals,
            self.config.maximum_proposals,
            offset=1,
            device=indices.device,
        ).transpose(0, 1)
        begin = pair_indices[:, 0]
        end = pair_indices[:, 1]
        pair_mask = proposal_mask[:, begin] & proposal_mask[:, end]
        line_bands, geometry = self._line_bands(pixel_values, points, pair_indices)
        line_hidden = self.line_encoder(torch.cat((line_bands, geometry), dim=-1))
        endpoint_sum = node_hidden[:, begin] + node_hidden[:, end]
        endpoint_difference = (node_hidden[:, begin] - node_hidden[:, end]).abs()
        edge_hidden = line_hidden + self.endpoint_encoder(
            torch.cat((endpoint_sum, endpoint_difference), dim=-1)
        )
        edge_logits = self.edge_classifier(edge_hidden).squeeze(-1)
        edge_logits = edge_logits.masked_fill(~pair_mask, -20.0)
        edge_probability = edge_logits.sigmoid() * pair_mask
        # The legacy path isolates the edge semantics from the node loss. The optional
        # joint path lets global node selection and visible-line connectivity co-adapt.
        message_hidden = (
            edge_hidden
            if self.config.joint_node_edge_gradients
            else edge_hidden.detach()
        )
        message_probability = (
            edge_probability
            if self.config.joint_node_edge_gradients
            else edge_probability.detach()
        )
        message = self.edge_message(message_hidden) * message_probability.unsqueeze(-1)
        aggregated = torch.zeros_like(node_hidden)
        degree = torch.zeros(
            (*proposal_mask.shape, 1),
            device=node_hidden.device,
            dtype=node_hidden.dtype,
        )
        scatter_begin = begin[None, :, None].expand(indices.size(0), -1, self.config.node_dim)
        scatter_end = end[None, :, None].expand_as(scatter_begin)
        aggregated.scatter_add_(1, scatter_begin, message)
        aggregated.scatter_add_(1, scatter_end, message)
        degree.scatter_add_(1, begin[None, :, None].expand(indices.size(0), -1, 1), edge_probability.unsqueeze(-1))
        degree.scatter_add_(1, end[None, :, None].expand(indices.size(0), -1, 1), edge_probability.unsqueeze(-1))
        aggregated = aggregated / degree.clamp_min(1.0)
        incident = torch.zeros(
            (
                indices.size(0),
                self.config.maximum_proposals,
                self.config.maximum_proposals,
            ),
            device=edge_probability.device,
            dtype=edge_probability.dtype,
        )
        incident[:, begin, end] = edge_probability.detach()
        incident[:, end, begin] = edge_probability.detach()
        strongest = incident.topk(self.config.incident_topk, dim=-1).values
        local_logits = self.node_local_classifier(
            torch.cat(
                (node_hidden, heatmap.unsqueeze(-1), base_keep.unsqueeze(-1)),
                dim=-1,
            )
        ).squeeze(-1)
        rescue_logits = self.node_rescue_classifier(
            torch.cat((aggregated, strongest, degree.detach()), dim=-1)
        ).squeeze(-1)
        node_logits = local_logits + rescue_logits
        node_logits = node_logits.masked_fill(~proposal_mask, -20.0)
        return {
            "proposal_indices": indices,
            "proposal_mask": proposal_mask,
            "points": points,
            "pair_indices": pair_indices,
            "pair_mask": pair_mask,
            "node_logits": node_logits,
            "edge_logits": edge_logits,
            "incident_edge_support": strongest,
            "local_node_logits": local_logits,
            "rescue_node_logits": rescue_logits,
            "heatmap_scores": heatmap,
            "aromatic_scores": aromatic,
        }


def load_proposal_graph_checkpoint(
    path: str | Path,
    device: torch.device,
) -> tuple[ProposalGraphRefiner, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = ProposalGraphConfig(**payload["proposal_graph_config"])
    head = ProposalGraphRefiner(config).to(device)
    head.load_state_dict(payload["proposal_graph"])
    return head.eval(), payload


def proposal_graph_loss(
    output: dict[str, torch.Tensor],
    anchor_points: torch.Tensor,
    anchor_mask: torch.Tensor,
    adjacency: torch.Tensor,
    *,
    matching_radius_pixels: float,
    image_size: int,
    edge_loss_weight: float = 1.0,
    negative_ratio: int = 8,
) -> dict[str, torch.Tensor | float]:
    node_targets, edge_targets, assignments = proposal_graph_targets(
        output["points"].detach(),
        output["proposal_mask"],
        output["pair_indices"],
        anchor_points,
        anchor_mask,
        adjacency,
        matching_radius_pixels=matching_radius_pixels,
        image_size=image_size,
    )
    valid_nodes = output["proposal_mask"]
    node_logits = output["node_logits"][valid_nodes].float()
    node_labels = node_targets[valid_nodes]
    positive_nodes = int(node_labels.sum())
    negative_nodes = len(node_labels) - positive_nodes
    node_weight = min(negative_nodes / max(positive_nodes, 1), 10.0)
    if len(node_logits):
        node_loss = F.binary_cross_entropy_with_logits(
            node_logits,
            node_labels,
            pos_weight=node_logits.new_tensor(node_weight),
        )
    else:
        node_loss = output["node_logits"].sum() * 0.0

    valid_edges = output["pair_mask"]
    edge_logits = output["edge_logits"][valid_edges].float()
    edge_labels = edge_targets[valid_edges]
    positive_indices = torch.where(edge_labels > 0.5)[0]
    negative_indices = torch.where(edge_labels <= 0.5)[0]
    negative_count = min(len(negative_indices), max(len(positive_indices) * negative_ratio, 32))
    if negative_count:
        hard = torch.topk(edge_logits[negative_indices].detach(), negative_count).indices
        selected = torch.cat((positive_indices, negative_indices[hard]))
    else:
        selected = positive_indices
    if len(selected):
        selected_labels = edge_labels[selected]
        positive_edges = int(selected_labels.sum())
        selected_negatives = len(selected_labels) - positive_edges
        edge_weight = min(selected_negatives / max(positive_edges, 1), 16.0)
        edge_loss = F.binary_cross_entropy_with_logits(
            edge_logits[selected],
            selected_labels,
            pos_weight=edge_logits.new_tensor(edge_weight),
        )
    else:
        edge_loss = output["edge_logits"].sum() * 0.0
    loss = node_loss + edge_loss_weight * edge_loss
    return {
        "loss": loss,
        "node_loss": node_loss,
        "edge_loss": edge_loss,
        "node_targets": node_targets,
        "edge_targets": edge_targets,
        "assignments": assignments,
        "proposal_recall": positive_nodes / max(int(anchor_mask.sum()), 1),
        "selected_edges": float(len(selected)),
    }


@torch.no_grad()
def decode_proposal_graph(
    output: dict[str, torch.Tensor],
    *,
    threshold: float,
    maximum_vertices: int = 128,
) -> list[list[dict[str, float | bool]]]:
    if not 0.0 < threshold < 1.0:
        raise ValueError("Proposal graph threshold must be in (0, 1)")
    probabilities = output["node_logits"].sigmoid()
    decoded = []
    for batch_index in range(probabilities.size(0)):
        valid = torch.where(output["proposal_mask"][batch_index])[0]
        scores = probabilities[batch_index, valid]
        order = scores.argsort(descending=True)
        vertices = []
        for offset in order[:maximum_vertices].tolist():
            score = float(scores[offset])
            if score < threshold:
                break
            proposal_id = int(valid[offset])
            vertices.append(
                {
                    "proposal_id": proposal_id,
                    "x": float(output["points"][batch_index, proposal_id, 0]),
                    "y": float(output["points"][batch_index, proposal_id, 1]),
                    "score": score,
                    "dense_score": float(output["heatmap_scores"][batch_index, proposal_id]),
                    "aromatic": bool(output["aromatic_scores"][batch_index, proposal_id] >= 0.5),
                }
            )
        decoded.append(vertices)
    return decoded
