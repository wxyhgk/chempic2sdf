"""Graph-level correction for a predicted ChemScene bond set."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from chemvector.bond_object import BOND_TYPES


ELEMENT_GROUPS = ("C", "N", "O", "F", "Cl", "Br", "B", "P", "S", "OTHER")
ELEMENT_TO_ID = {name: index for index, name in enumerate(ELEMENT_GROUPS)}


@dataclass(frozen=True, slots=True)
class ChemSceneGraphCorrectorConfig:
    d_model: int = 96
    layers: int = 3
    element_dim: int = 16
    residual_scale: float = 6.0
    type_prior: float = 2.0

    def __post_init__(self) -> None:
        if min(self.d_model, self.layers, self.element_dim) < 1:
            raise ValueError("Graph corrector dimensions must be positive")
        if self.residual_scale <= 0.0 or self.type_prior < 0.0:
            raise ValueError("Graph corrector residual configuration is invalid")


def nearest_bond_pairs(
    points: torch.Tensor, bond_endpoints: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if points.ndim != 2 or points.size(-1) != 2 or not len(points):
        raise ValueError("Junction points must have shape [nodes, 2]")
    if bond_endpoints.ndim != 3 or bond_endpoints.shape[1:] != (2, 2):
        raise ValueError("Bond endpoints must have shape [edges, 2, 2]")
    distances = torch.cdist(bond_endpoints.float(), points.float())
    return distances.argmin(-1), distances.min(-1).values


class NodeEdgeMessageLayer(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.edge_update = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.edge_norm = nn.LayerNorm(d_model)
        self.edge_message = nn.Linear(d_model, d_model)
        self.node_update = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.node_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
        pairs: torch.Tensor,
        edge_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        begin, end = pairs[:, 0], pairs[:, 1]
        begin_hidden = nodes[begin]
        end_hidden = nodes[end]
        edge_delta = self.edge_update(
            torch.cat(
                (edges, begin_hidden + end_hidden, (begin_hidden - end_hidden).abs()),
                dim=-1,
            )
        )
        edges = self.edge_norm(edges + edge_delta)
        messages = self.edge_message(edges) * edge_weights[:, None]
        aggregate = torch.zeros_like(nodes)
        degree = torch.zeros((len(nodes), 1), device=nodes.device, dtype=nodes.dtype)
        aggregate.index_add_(0, begin, messages)
        aggregate.index_add_(0, end, messages)
        degree.index_add_(0, begin, edge_weights[:, None])
        degree.index_add_(0, end, edge_weights[:, None])
        aggregate = aggregate / degree.clamp_min(1.0)
        nodes = self.node_norm(
            nodes + self.node_update(torch.cat((nodes, aggregate), dim=-1))
        )
        return nodes, edges


class ChemSceneGraphCorrector(nn.Module):
    """Jointly contextualize Junction nodes and predicted BondObjects."""

    node_scalar_dim = 6
    edge_scalar_dim = 21

    def __init__(self, config: ChemSceneGraphCorrectorConfig | None = None) -> None:
        super().__init__()
        self.config = config or ChemSceneGraphCorrectorConfig()
        self.element_embedding = nn.Embedding(len(ELEMENT_GROUPS), self.config.element_dim)
        self.node_projection = nn.Sequential(
            nn.Linear(self.node_scalar_dim + self.config.element_dim, self.config.d_model),
            nn.GELU(),
            nn.LayerNorm(self.config.d_model),
        )
        self.edge_projection = nn.Sequential(
            nn.Linear(self.edge_scalar_dim, self.config.d_model),
            nn.GELU(),
            nn.LayerNorm(self.config.d_model),
        )
        self.layers = nn.ModuleList(
            NodeEdgeMessageLayer(self.config.d_model) for _ in range(self.config.layers)
        )
        self.node_keep = nn.Linear(self.config.d_model, 1)
        self.edge_keep_residual = nn.Linear(self.config.d_model, 1)
        self.display_residual = nn.Linear(self.config.d_model, len(BOND_TYPES))
        self.chemical_residual = nn.Linear(self.config.d_model, len(BOND_TYPES))
        for head in (self.edge_keep_residual, self.display_residual, self.chemical_residual):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        points: torch.Tensor,
        roles: torch.Tensor,
        elements: torch.Tensor,
        bond_endpoints: torch.Tensor,
        bond_scores: torch.Tensor,
        display_ids: torch.Tensor,
        chemical_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if roles.shape != elements.shape or roles.shape != (len(points),):
            raise ValueError("Junction role/element features differ from points")
        edges = len(bond_endpoints)
        if not (
            bond_scores.shape == display_ids.shape == chemical_ids.shape == (edges,)
        ):
            raise ValueError("BondObject feature lengths differ")
        pairs, endpoint_distances = nearest_bond_pairs(points, bond_endpoints)
        edge_probability = bond_scores.float().clamp(1e-5, 1.0 - 1e-5)
        orders = torch.tensor(
            (1.0, 2.0, 3.0, 1.5), device=points.device, dtype=torch.float32
        )[chemical_ids.long()]
        degree = torch.zeros((len(points),), device=points.device)
        order_sum = torch.zeros_like(degree)
        for endpoint in range(2):
            degree.index_add_(0, pairs[:, endpoint], edge_probability)
            order_sum.index_add_(0, pairs[:, endpoint], edge_probability * orders)
        role_one_hot = F.one_hot(roles.long(), num_classes=2).float()
        node_scalars = torch.cat(
            (
                points.float(),
                role_one_hot,
                (degree / 4.0).clamp_max(2.0)[:, None],
                (order_sum / 4.0).clamp_max(2.0)[:, None],
            ),
            dim=-1,
        )
        nodes = self.node_projection(
            torch.cat((node_scalars, self.element_embedding(elements.long())), dim=-1)
        )
        center = bond_endpoints.mean(1).float()
        delta = (bond_endpoints[:, 1] - bond_endpoints[:, 0]).float()
        length = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        unit = delta / length.clamp_min(1e-8)
        edge_scalars = torch.cat(
            (
                bond_endpoints.reshape(edges, 4).float(),
                center,
                unit.abs(),
                length,
                edge_probability[:, None],
                endpoint_distances.float(),
                pairs[:, :1].eq(pairs[:, 1:]).float(),
                F.one_hot(display_ids.long(), num_classes=len(BOND_TYPES)).float(),
                F.one_hot(chemical_ids.long(), num_classes=len(BOND_TYPES)).float(),
            ),
            dim=-1,
        )
        edge_hidden = self.edge_projection(edge_scalars)
        for layer in self.layers:
            nodes, edge_hidden = layer(
                nodes, edge_hidden, pairs, edge_probability.to(nodes.dtype)
            )
        keep_residual = self.config.residual_scale * torch.tanh(
            self.edge_keep_residual(edge_hidden).squeeze(-1) / self.config.residual_scale
        )
        base_keep = torch.logit(edge_probability)
        display_prior = F.one_hot(
            display_ids.long(), num_classes=len(BOND_TYPES)
        ).float() * self.config.type_prior
        chemical_prior = F.one_hot(
            chemical_ids.long(), num_classes=len(BOND_TYPES)
        ).float() * self.config.type_prior
        return {
            "node_keep_logits": self.node_keep(nodes).squeeze(-1),
            "edge_keep_logits": base_keep + keep_residual,
            "display_logits": display_prior + self.display_residual(edge_hidden),
            "chemical_logits": chemical_prior + self.chemical_residual(edge_hidden),
            "pairs": pairs,
        }


def graph_corrector_loss(
    output: dict[str, torch.Tensor],
    node_targets: torch.Tensor,
    edge_targets: torch.Tensor,
    display_targets: torch.Tensor,
    chemical_targets: torch.Tensor,
    *,
    node_weight: float = 0.25,
    edge_weight: float = 1.0,
    type_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    node_loss = F.binary_cross_entropy_with_logits(
        output["node_keep_logits"], node_targets.float()
    )
    positives = edge_targets.float().sum().clamp_min(1.0)
    negatives = (1.0 - edge_targets.float()).sum().clamp_min(1.0)
    edge_loss = F.binary_cross_entropy_with_logits(
        output["edge_keep_logits"],
        edge_targets.float(),
        pos_weight=(negatives / positives).clamp_max(16.0),
    )
    matched = edge_targets.bool()
    if bool(matched.any()):
        display_loss = F.cross_entropy(
            output["display_logits"][matched], display_targets[matched]
        )
        chemical_loss = F.cross_entropy(
            output["chemical_logits"][matched], chemical_targets[matched]
        )
    else:
        display_loss = output["display_logits"].sum() * 0.0
        chemical_loss = output["chemical_logits"].sum() * 0.0
    return {
        "loss": node_weight * node_loss + edge_weight * edge_loss
        + type_weight * (display_loss + chemical_loss),
        "node_loss": node_loss,
        "edge_loss": edge_loss,
        "display_loss": display_loss,
        "chemical_loss": chemical_loss,
    }
