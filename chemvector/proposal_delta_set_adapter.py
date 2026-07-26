"""Zero-initialized set/incidence corrections conditioned on Bond proposal deltas."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class ProposalDeltaSetConfig:
    hidden_dim: int = 64

    def __post_init__(self) -> None:
        if self.hidden_dim < 1:
            raise ValueError("Proposal delta hidden dimension must be positive")


def gather_original_bond_evidence(
    proposals: dict[str, torch.Tensor],
    original_dense: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Read original detector evidence at each candidate proposal's dense index."""

    indices = proposals["bond_indices"]
    batch = torch.arange(indices.size(0), device=indices.device)[:, None]
    scores = original_dense["heatmap_logits"].sigmoid().flatten(1)[batch, indices]
    endpoints = original_dense["endpoints"].permute(0, 1, 4, 5, 2, 3).flatten(1, 3)[
        batch, indices
    ]
    return {"scores": scores, "endpoints": endpoints}


def proposal_deltas(
    proposals: dict[str, torch.Tensor],
    original_dense: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    original = gather_original_bond_evidence(proposals, original_dense)
    return {
        "score": proposals["bond_scores"] - original["scores"],
        "endpoints": proposals["bond_endpoints"] - original["endpoints"],
        "original_endpoints": original["endpoints"],
    }


class ProposalDeltaSetAdapter(nn.Module):
    """Correct only the residual-proposal path; the original path bypasses this head."""

    def __init__(self, config: ProposalDeltaSetConfig) -> None:
        super().__init__()
        self.config = config
        self.bond_presence = self._head(16, config.hidden_dim)
        self.incidence = self._head(12, config.hidden_dim)

    @staticmethod
    def _head(input_dim: int, hidden_dim: int) -> nn.Sequential:
        head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(head[-1].weight)
        nn.init.zeros_(head[-1].bias)
        return head

    def forward(
        self,
        base: dict[str, torch.Tensor],
        proposals: dict[str, torch.Tensor],
        deltas: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        atom_mask = proposals["atom_mask"]
        bond_mask = proposals["bond_mask"]
        masked_incidence = base["incidence_logits"].masked_fill(
            ~atom_mask[:, None, None, :], -1e4
        )
        probabilities = masked_incidence.softmax(-1)
        top = probabilities.topk(min(2, probabilities.size(-1)), dim=-1).values
        maximum = top[..., 0]
        second = top[..., 1] if top.size(-1) > 1 else torch.zeros_like(maximum)
        incidence_summary = torch.stack((maximum, maximum - second), dim=-1)

        score_delta = deltas["score"][..., None]
        endpoint_delta = deltas["endpoints"].flatten(2)
        presence_input = torch.cat(
            (
                score_delta,
                score_delta.abs(),
                endpoint_delta,
                endpoint_delta.abs(),
                proposals["bond_scores"][..., None],
                base["bond_presence_logits"][..., None],
                incidence_summary.flatten(2),
            ),
            dim=-1,
        )
        presence_residual = self.bond_presence(presence_input).squeeze(-1)
        presence_residual = presence_residual.masked_fill(~bond_mask, 0.0)

        atom_anchors = proposals["atom_geometry"][..., :2]
        candidate_endpoints = proposals["bond_endpoints"]
        original_endpoints = deltas["original_endpoints"]
        candidate_relative = (
            candidate_endpoints[:, :, :, None, :] - atom_anchors[:, None, None]
        )
        original_relative = (
            original_endpoints[:, :, :, None, :] - atom_anchors[:, None, None]
        )
        endpoint_delta_expanded = deltas["endpoints"][:, :, :, None, :].expand_as(
            candidate_relative
        )
        atom_scores = proposals["atom_scores"][:, None, None, :, None].expand(
            *candidate_relative.shape[:-1], 1
        )
        bond_scores = proposals["bond_scores"][:, :, None, None, None].expand(
            *candidate_relative.shape[:-1], 1
        )
        score_delta_expanded = deltas["score"][:, :, None, None, None].expand(
            *candidate_relative.shape[:-1], 1
        )
        incidence_input = torch.cat(
            (
                endpoint_delta_expanded,
                endpoint_delta_expanded.abs(),
                candidate_relative,
                original_relative,
                base["incidence_logits"][..., None],
                atom_scores,
                score_delta_expanded,
                bond_scores,
            ),
            dim=-1,
        )
        incidence_residual = self.incidence(incidence_input).squeeze(-1)
        valid_incidence = bond_mask[:, :, None, None] & atom_mask[:, None, None, :]
        incidence_residual = incidence_residual.masked_fill(~valid_incidence, 0.0)
        incidence_logits = base["incidence_logits"] + incidence_residual
        adjusted_probabilities = incidence_logits.masked_fill(
            ~atom_mask[:, None, None, :], -1e4
        ).softmax(-1).to(base["atom_presence_logits"].dtype)
        return {
            **base,
            "bond_presence_logits": base["bond_presence_logits"] + presence_residual,
            "incidence_logits": incidence_logits,
            "incidence_probabilities": adjusted_probabilities,
            "proposal_delta_bond_presence_residual": presence_residual,
            "proposal_delta_incidence_residual": incidence_residual,
        }


__all__ = [
    "ProposalDeltaSetAdapter",
    "ProposalDeltaSetConfig",
    "gather_original_bond_evidence",
    "proposal_deltas",
]
