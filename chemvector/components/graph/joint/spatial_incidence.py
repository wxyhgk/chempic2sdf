"""Learned BondObject endpoint to AtomObject proposal incidence."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class SpatialIncidenceConfig:
    atom_feature_dim: int = 128
    bond_feature_dim: int = 128
    bond_type_dim: int = 8
    hidden_dim: int = 128
    pair_hidden_dim: int = 64

    def __post_init__(self) -> None:
        if min(
            self.atom_feature_dim,
            self.bond_feature_dim,
            self.bond_type_dim,
            self.hidden_dim,
            self.pair_hidden_dim,
        ) < 1:
            raise ValueError("Spatial incidence dimensions must be positive")


class SpatialIncidenceHead(nn.Module):
    """Score every predicted Bond endpoint against complete AtomObject proposals."""

    def __init__(self, config: SpatialIncidenceConfig) -> None:
        super().__init__()
        self.config = config
        self.atom_projection = nn.Sequential(
            nn.Linear(config.atom_feature_dim + 8, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.endpoint_embedding = nn.Embedding(2, config.hidden_dim)
        self.endpoint_projection = nn.Sequential(
            nn.Linear(
                config.bond_feature_dim + config.bond_type_dim + 4,
                config.hidden_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.pair_geometry = nn.Sequential(
            nn.Linear(7, config.pair_hidden_dim),
            nn.GELU(),
            nn.Linear(config.pair_hidden_dim, config.pair_hidden_dim),
            nn.GELU(),
            nn.Linear(config.pair_hidden_dim, 1),
        )

    def forward(
        self,
        atom_features: torch.Tensor,
        atom_geometry: torch.Tensor,
        atom_mask: torch.Tensor,
        bond_features: torch.Tensor,
        bond_endpoints: torch.Tensor,
        bond_types: torch.Tensor,
        bond_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if atom_geometry.shape[-1] != 8:
            raise ValueError("AtomObject geometry must contain eight values")
        if bond_endpoints.shape[-2:] != (2, 2):
            raise ValueError("BondObject endpoints must have shape [..., 2, 2]")
        atom_tokens = self.atom_projection(
            torch.cat((atom_features, atom_geometry), dim=-1)
        )
        bond_count = bond_features.size(1)
        expanded_bond = bond_features[:, :, None, :].expand(-1, -1, 2, -1)
        expanded_types = bond_types[:, :, None, :].expand(-1, -1, 2, -1)
        other_endpoints = bond_endpoints.flip(2)
        endpoint_inputs = torch.cat(
            (expanded_bond, expanded_types, bond_endpoints, other_endpoints),
            dim=-1,
        )
        endpoint_tokens = self.endpoint_projection(endpoint_inputs)
        endpoint_tokens = endpoint_tokens + self.endpoint_embedding.weight.to(
            endpoint_tokens.dtype
        )[None, None, :, :]
        content_logits = torch.einsum(
            "bqed,bad->bqea", endpoint_tokens, atom_tokens
        ) / math.sqrt(self.config.hidden_dim)

        delta = (
            atom_geometry[:, None, None, :, :2]
            - bond_endpoints[:, :, :, None, :]
        )
        absolute = delta.abs()
        distance = delta.square().sum(dim=-1, keepdim=True).sqrt()
        bond_vector = (
            other_endpoints - bond_endpoints
        )[:, :, :, None, :].expand(-1, -1, -1, atom_features.size(1), -1)
        bond_unit = F.normalize(bond_vector, dim=-1, eps=1e-6)
        longitudinal = (delta * bond_unit).sum(dim=-1, keepdim=True)
        cross = (
            delta[..., 0:1] * bond_unit[..., 1:2]
            - delta[..., 1:2] * bond_unit[..., 0:1]
        )
        geometry_features = torch.cat(
            (delta, absolute, distance, longitudinal, cross), dim=-1
        )
        geometry_logits = self.pair_geometry(geometry_features).squeeze(-1)
        logits = content_logits + geometry_logits
        logits = logits.masked_fill(~atom_mask[:, None, None, :], -1e4)
        logits = logits.masked_fill(~bond_mask[:, :, None, None], -1e4)
        return {
            "incidence_logits": logits,
            "atom_tokens": atom_tokens,
            "endpoint_tokens": endpoint_tokens,
        }


def spatial_incidence_loss(
    output: dict[str, torch.Tensor],
    target_atom_refs: torch.Tensor,
    bond_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    logits = output["incidence_logits"][bond_mask]
    targets = target_atom_refs[bond_mask]
    if not len(logits):
        zero = output["incidence_logits"].sum() * 0.0
        return {"loss": zero, "endpoint_accuracy": zero, "bond_accuracy": zero}
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    predicted = logits.argmax(dim=-1)
    endpoint_accuracy = (predicted == targets).float().mean()
    bond_accuracy = (predicted == targets).all(dim=-1).float().mean()
    return {
        "loss": loss,
        "endpoint_accuracy": endpoint_accuracy,
        "bond_accuracy": bond_accuracy,
    }
