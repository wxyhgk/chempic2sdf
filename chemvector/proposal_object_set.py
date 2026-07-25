"""Joint Atom/Bond proposal presence and incidence consistency head."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from chemvector.spatial_incidence import (
    SpatialIncidenceConfig,
    SpatialIncidenceHead,
    spatial_incidence_loss,
)


@dataclass(frozen=True, slots=True)
class ProposalObjectSetConfig:
    incidence: SpatialIncidenceConfig = SpatialIncidenceConfig()
    hidden_dim: int = 128
    stable_bond_hidden_dim: int = 0

    def __post_init__(self) -> None:
        if self.hidden_dim < 1:
            raise ValueError("Proposal object set hidden_dim must be positive")
        if self.stable_bond_hidden_dim < 0:
            raise ValueError("stable_bond_hidden_dim must be non-negative")


class ProposalObjectSetHead(nn.Module):
    """Select complete object sets while retaining learned endpoint incidence."""

    def __init__(self, config: ProposalObjectSetConfig) -> None:
        super().__init__()
        self.config = config
        self.incidence = SpatialIncidenceHead(config.incidence)
        incidence_hidden = config.incidence.hidden_dim
        endpoint_hidden = 2 * incidence_hidden
        self.atom_pre_presence = nn.Sequential(
            nn.Linear(incidence_hidden + 9, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, 1),
        )
        self.bond_pre_presence = nn.Sequential(
            nn.Linear(endpoint_hidden + 17, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, 1),
        )
        self.atom_presence_residual = nn.Sequential(
            nn.Linear(incidence_hidden + 4, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
        )
        self.bond_presence_residual = nn.Sequential(
            nn.Linear(endpoint_hidden + 6, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
        )
        self.stable_bond_presence = None
        if config.stable_bond_hidden_dim:
            self.stable_bond_presence = nn.Sequential(
                nn.Linear(30, config.stable_bond_hidden_dim),
                nn.GELU(),
                nn.LayerNorm(config.stable_bond_hidden_dim),
                nn.Linear(config.stable_bond_hidden_dim, 1),
            )
            nn.init.zeros_(self.stable_bond_presence[-1].weight)
            nn.init.zeros_(self.stable_bond_presence[-1].bias)

    @staticmethod
    def _incidence_statistics(
        logits: torch.Tensor, atom_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        masked = logits.masked_fill(~atom_mask[:, None, None, :], -1e4)
        probabilities = masked.softmax(-1)
        top = probabilities.topk(min(2, probabilities.size(-1)), dim=-1).values
        maximum = top[..., 0]
        second = top[..., 1] if top.size(-1) > 1 else torch.zeros_like(maximum)
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(-1)
        normalizer = math.log(max(probabilities.size(-1), 2))
        statistics = torch.stack((maximum, maximum - second, entropy / normalizer), dim=-1)
        return probabilities, statistics

    def forward(
        self,
        atom_features: torch.Tensor,
        atom_geometry: torch.Tensor,
        atom_scores: torch.Tensor,
        atom_mask: torch.Tensor,
        bond_features: torch.Tensor,
        bond_endpoints: torch.Tensor,
        bond_types: torch.Tensor,
        bond_scores: torch.Tensor,
        bond_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        incidence = self.incidence(
            atom_features,
            atom_geometry,
            atom_mask,
            bond_features,
            bond_endpoints,
            bond_types,
            bond_mask,
        )
        atom_tokens = incidence["atom_tokens"]
        endpoint_tokens = incidence["endpoint_tokens"]
        probabilities, incidence_statistics = self._incidence_statistics(
            incidence["incidence_logits"], atom_mask
        )
        atom_pre = self.atom_pre_presence(
            torch.cat((atom_tokens, atom_geometry, atom_scores[..., None]), dim=-1)
        ).squeeze(-1)
        atom_pre_probabilities = atom_pre.sigmoid()
        expected_atom_presence = torch.einsum(
            "bqea,ba->bqe", probabilities, atom_pre_probabilities
        )
        bond_input = torch.cat(
            (
                endpoint_tokens.flatten(2),
                bond_types,
                bond_scores[..., None],
                incidence_statistics.flatten(2),
                expected_atom_presence,
            ),
            dim=-1,
        )
        bond_pre = self.bond_pre_presence(bond_input).squeeze(-1)
        weighted_incidence = probabilities * bond_pre.sigmoid()[:, :, None, None]
        atom_support_sum = weighted_incidence.sum(dim=(1, 2))
        atom_support_max = weighted_incidence.amax(dim=(1, 2))
        atom_residual_input = torch.cat(
            (
                atom_tokens,
                atom_scores[..., None],
                atom_pre[..., None],
                atom_support_sum[..., None],
                atom_support_max[..., None],
            ),
            dim=-1,
        )
        atom_logits = atom_pre + self.atom_presence_residual(
            atom_residual_input
        ).squeeze(-1)
        expected_refined_presence = torch.einsum(
            "bqea,ba->bqe", probabilities, atom_logits.sigmoid()
        )
        bond_residual_input = torch.cat(
            (
                endpoint_tokens.flatten(2),
                bond_scores[..., None],
                bond_pre[..., None],
                expected_refined_presence,
                incidence_statistics[..., 0],
            ),
            dim=-1,
        )
        base_bond_logits = bond_pre + self.bond_presence_residual(
            bond_residual_input
        ).squeeze(-1)
        stable_residual = torch.zeros_like(base_bond_logits)
        if self.stable_bond_presence is not None:
            vectors = bond_endpoints[:, :, 1] - bond_endpoints[:, :, 0]
            lengths = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
            units = F.normalize(vectors, dim=-1, eps=1e-6)
            doubled_orientation = torch.stack(
                (
                    units[..., 0].square() - units[..., 1].square(),
                    2.0 * units[..., 0] * units[..., 1],
                ),
                dim=-1,
            )
            stable_input = torch.cat(
                (
                    bond_endpoints.flatten(2),
                    bond_endpoints.mean(dim=2),
                    vectors,
                    lengths,
                    doubled_orientation,
                    bond_types,
                    bond_scores[..., None],
                    incidence_statistics.flatten(2),
                    expected_refined_presence,
                    bond_pre[..., None],
                    base_bond_logits[..., None],
                ),
                dim=-1,
            )
            stable_residual = self.stable_bond_presence(stable_input).squeeze(-1)
        bond_logits = base_bond_logits + stable_residual
        atom_logits = atom_logits.masked_fill(~atom_mask, -1e4)
        bond_logits = bond_logits.masked_fill(~bond_mask, -1e4)
        return {
            **incidence,
            "atom_presence_logits": atom_logits,
            "bond_presence_logits": bond_logits,
            "atom_pre_presence_logits": atom_pre,
            "bond_pre_presence_logits": bond_pre,
            "base_bond_presence_logits": base_bond_logits,
            "stable_bond_presence_residual": stable_residual,
            "incidence_probabilities": probabilities,
        }


def _focal_presence_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    gamma: float = 2.0,
) -> torch.Tensor:
    logits = logits[mask].float()
    targets = targets[mask].float()
    if not len(logits):
        return logits.sum() * 0.0
    probabilities = logits.sigmoid()
    cross_entropy = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    correct_probability = torch.where(targets.bool(), probabilities, 1.0 - probabilities)
    positive = targets.sum().clamp_min(1.0)
    negative = (1.0 - targets).sum().clamp_min(1.0)
    positive_weight = (negative / positive).clamp(1.0, 8.0)
    class_weight = torch.where(targets.bool(), positive_weight, 1.0)
    return (class_weight * (1.0 - correct_probability).pow(gamma) * cross_entropy).mean()


def _set_ranking_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    margin: float = 1.0,
) -> torch.Tensor:
    losses = []
    for batch_index in range(logits.size(0)):
        positive = logits[batch_index][mask[batch_index] & targets[batch_index]]
        negative = logits[batch_index][mask[batch_index] & ~targets[batch_index]]
        if len(positive) and len(negative):
            losses.append(F.relu(margin - positive.min() + negative.max()))
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


def _topology_closure_loss(
    output: dict[str, torch.Tensor],
    atom_mask: torch.Tensor,
    bond_mask: torch.Tensor,
) -> torch.Tensor:
    """Align Atom/Bond presence without moving learned incidence assignments."""
    atom_probabilities = output["atom_presence_logits"].sigmoid()
    bond_logits = output["bond_presence_logits"]
    bond_probabilities = bond_logits.sigmoid()
    incidence = output["incidence_probabilities"].detach()
    endpoint_presence = torch.einsum(
        "bqea,ba->bqe", incidence, atom_probabilities
    ).clamp(1e-6, 1.0)
    valid = bond_mask[:, :, None].expand_as(endpoint_presence)
    if not valid.any():
        return bond_logits.sum() * 0.0
    # A selected bond pushes its endpoint atoms up. Missing endpoint support pushes
    # the bond down. Detaching the opposite side prevents trivial joint collapse.
    atom_pressure = (
        bond_probabilities.detach()[:, :, None] * -endpoint_presence.log()
    )
    missing_endpoint = (1.0 - endpoint_presence.detach()).clamp(0.0, 1.0)
    bond_pressure = missing_endpoint * F.softplus(bond_logits[:, :, None])
    return (atom_pressure + bond_pressure)[valid].mean()


def _graph_exact_surrogate_loss(
    output: dict[str, torch.Tensor],
    atom_presence_targets: torch.Tensor,
    bond_presence_targets: torch.Tensor,
    atom_mask: torch.Tensor,
    bond_mask: torch.Tensor,
    target_atom_refs: torch.Tensor,
    bond_supervision_mask: torch.Tensor,
    *,
    atom_threshold: float,
    bond_threshold: float,
    temperature: float = 0.25,
) -> torch.Tensor:
    """Smooth maximum over all decisions that must be correct for a whole graph."""
    if temperature <= 0.0:
        raise ValueError("Graph exact surrogate temperature must be positive")
    sample_losses = []
    for batch_index in range(atom_mask.size(0)):
        errors = []
        for logits, targets, mask, threshold in (
            (
                output["atom_presence_logits"][batch_index],
                atom_presence_targets[batch_index],
                atom_mask[batch_index],
                atom_threshold,
            ),
            (
                output["bond_presence_logits"][batch_index],
                bond_presence_targets[batch_index],
                bond_mask[batch_index],
                bond_threshold,
            ),
        ):
            selected_logits = logits[mask].float() - threshold
            selected_targets = targets[mask]
            signed_margin = torch.where(
                selected_targets, selected_logits, -selected_logits
            )
            errors.append(F.softplus(-signed_margin))
        supervised = bond_supervision_mask[batch_index]
        if supervised.any():
            incidence_logits = output["incidence_logits"][batch_index, supervised]
            incidence_targets = target_atom_refs[batch_index, supervised]
            errors.append(
                F.cross_entropy(
                    incidence_logits.flatten(0, 1).float(),
                    incidence_targets.flatten(),
                    reduction="none",
                )
            )
        flat_errors = torch.cat(errors)
        sample_losses.append(
            temperature * torch.logsumexp(flat_errors / temperature, dim=0)
        )
    return (
        torch.stack(sample_losses).mean()
        if sample_losses
        else output["atom_presence_logits"].sum() * 0.0
    )


def proposal_object_set_loss(
    output: dict[str, torch.Tensor],
    atom_presence_targets: torch.Tensor,
    bond_presence_targets: torch.Tensor,
    atom_mask: torch.Tensor,
    bond_mask: torch.Tensor,
    target_atom_refs: torch.Tensor,
    bond_supervision_mask: torch.Tensor,
    *,
    topology_closure_weight: float = 0.0,
    graph_exact_surrogate_weight: float = 0.0,
    atom_decision_threshold: float = 0.0,
    bond_decision_threshold: float = -0.5,
) -> dict[str, torch.Tensor]:
    if topology_closure_weight < 0.0:
        raise ValueError("Topology closure weight must be non-negative")
    if graph_exact_surrogate_weight < 0.0:
        raise ValueError("Graph exact surrogate weight must be non-negative")
    atom_presence_loss = _focal_presence_loss(
        output["atom_presence_logits"], atom_presence_targets, atom_mask
    )
    bond_presence_loss = _focal_presence_loss(
        output["bond_presence_logits"], bond_presence_targets, bond_mask
    )
    atom_ranking_loss = _set_ranking_loss(
        output["atom_presence_logits"], atom_presence_targets, atom_mask
    )
    bond_ranking_loss = _set_ranking_loss(
        output["bond_presence_logits"], bond_presence_targets, bond_mask
    )
    atom_count_loss = F.smooth_l1_loss(
        (output["atom_presence_logits"].sigmoid() * atom_mask).sum(-1),
        atom_presence_targets.sum(-1).float(),
    )
    bond_count_loss = F.smooth_l1_loss(
        (output["bond_presence_logits"].sigmoid() * bond_mask).sum(-1),
        bond_presence_targets.sum(-1).float(),
    )
    incidence_losses = spatial_incidence_loss(
        output, target_atom_refs, bond_supervision_mask
    )
    topology_closure_loss = _topology_closure_loss(output, atom_mask, bond_mask)
    graph_exact_surrogate_loss = _graph_exact_surrogate_loss(
        output,
        atom_presence_targets,
        bond_presence_targets,
        atom_mask,
        bond_mask,
        target_atom_refs,
        bond_supervision_mask,
        atom_threshold=atom_decision_threshold,
        bond_threshold=bond_decision_threshold,
    )
    loss = (
        atom_presence_loss
        + bond_presence_loss
        + 0.5 * (atom_ranking_loss + bond_ranking_loss)
        + 0.1 * (atom_count_loss + bond_count_loss)
        + incidence_losses["loss"]
        + topology_closure_weight * topology_closure_loss
        + graph_exact_surrogate_weight * graph_exact_surrogate_loss
    )
    return {
        "loss": loss,
        "atom_presence_loss": atom_presence_loss,
        "bond_presence_loss": bond_presence_loss,
        "atom_ranking_loss": atom_ranking_loss,
        "bond_ranking_loss": bond_ranking_loss,
        "atom_count_loss": atom_count_loss,
        "bond_count_loss": bond_count_loss,
        "incidence_loss": incidence_losses["loss"],
        "topology_closure_loss": topology_closure_loss,
        "graph_exact_surrogate_loss": graph_exact_surrogate_loss,
        "endpoint_accuracy": incidence_losses["endpoint_accuracy"],
        "incidence_bond_accuracy": incidence_losses["bond_accuracy"],
    }
