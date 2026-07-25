"""Candidate-level exactness verifier for frozen chemical RETYPE actions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from chemvector.atom_pair_chemical_retype import (
    RETYPE_FEATURE_DIM,
    AtomPairChemicalRetypeHead,
    AtomPairRetypeAction,
)
from chemvector.audit_atom_pair_chemical_reachability import chemical_confidence_margin
from chemvector.bond_object import BOND_TYPES


VERIFIER_SCALAR_DIM = 8


@dataclass(frozen=True, slots=True)
class AtomPairChemicalRetypeVerifierConfig:
    input_dim: int = RETYPE_FEATURE_DIM
    hidden_dim: int = 64
    type_embedding_dim: int = 16
    maximum_groups: int = 256
    ranking_margin: float = 1.0

    def __post_init__(self) -> None:
        if min(
            self.input_dim,
            self.hidden_dim,
            self.type_embedding_dim,
            self.maximum_groups,
        ) < 1:
            raise ValueError("Chemical verifier dimensions must be positive")
        if self.ranking_margin <= 0.0:
            raise ValueError("Chemical verifier ranking margin must be positive")


class AtomPairChemicalRetypeVerifier(nn.Module):
    def __init__(self, config: AtomPairChemicalRetypeVerifierConfig) -> None:
        super().__init__()
        self.config = config
        self.edge_encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
        )
        self.type_embedding = nn.Embedding(len(BOND_TYPES), config.type_embedding_dim)
        global_dim = config.hidden_dim * 2 + 2
        candidate_dim = (
            config.hidden_dim
            + global_dim
            + config.type_embedding_dim * 2
            + VERIFIER_SCALAR_DIM
        )
        self.verifier = nn.Sequential(
            nn.Linear(candidate_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, 1),
        )
        nn.init.zeros_(self.verifier[-1].weight)
        nn.init.zeros_(self.verifier[-1].bias)

    def score_sample(
        self,
        edge_features: torch.Tensor,
        chemical_logits: torch.Tensor,
        selected: torch.Tensor,
        gate_logit: torch.Tensor,
        action_scores: torch.Tensor,
        actions: list[AtomPairRetypeAction],
    ) -> torch.Tensor:
        if not (len(edge_features) == len(chemical_logits) == len(selected)):
            raise ValueError("Chemical verifier sample tensors differ in length")
        if len(action_scores) != len(actions):
            raise ValueError("Chemical verifier scores and actions differ in length")
        if selected.dtype != torch.bool:
            raise ValueError("Chemical verifier selection must be boolean")
        if not len(edge_features):
            raise ValueError("Chemical verifier requires at least one group")
        if len(edge_features) > self.config.maximum_groups:
            raise ValueError("Chemical verifier group count exceeds capacity")
        if not actions:
            return edge_features.new_empty((0,))

        hidden = self.edge_encoder(edge_features)
        global_feature = torch.cat(
            (
                hidden.mean(0),
                hidden.amax(0),
                hidden.new_tensor(
                    (
                        float(selected.sum()) / self.config.maximum_groups,
                        float(len(selected)) / self.config.maximum_groups,
                    )
                ),
            )
        )
        predicted = chemical_logits.argmax(-1)
        chemical_margin = chemical_confidence_margin(chemical_logits)
        top_action = action_scores.float().amax()
        if len(action_scores) > 1:
            top_two = action_scores.float().topk(2).values
            action_margin = top_two[0] - top_two[1]
        else:
            action_margin = top_action.new_zeros(())
        rows = []
        for action_index, action in enumerate(actions):
            group = action.group_index
            current_state = int(predicted[group])
            target_state = action.target_state
            target_logit = chemical_logits[group, target_state].float()
            current_logit = chemical_logits[group, current_state].float()
            prior_score = action_scores[action_index].float()
            scalars = torch.stack(
                (
                    gate_logit.float().clamp(-8.0, 8.0) / 8.0,
                    prior_score.clamp(-8.0, 8.0) / 8.0,
                    (prior_score - top_action).clamp(-8.0, 0.0) / 8.0,
                    action_margin.clamp(0.0, 8.0) / 8.0,
                    chemical_margin[group].float().clamp(0.0, 8.0) / 8.0,
                    target_logit.clamp(-8.0, 8.0) / 8.0,
                    (target_logit - current_logit).clamp(-8.0, 8.0) / 8.0,
                    selected[group].float(),
                )
            )
            rows.append(
                torch.cat(
                    (
                        hidden[group],
                        global_feature,
                        self.type_embedding.weight[current_state],
                        self.type_embedding.weight[target_state],
                        scalars,
                    )
                )
            )
        return self.verifier(torch.stack(rows)).squeeze(-1)


@torch.no_grad()
def initialize_verifier_from_retype(
    verifier: AtomPairChemicalRetypeVerifier,
    retype: AtomPairChemicalRetypeHead,
) -> None:
    if verifier.config.hidden_dim != retype.config.hidden_dim:
        raise ValueError("Verifier and retype hidden dimensions must match")
    verifier.edge_encoder.load_state_dict(retype.edge_encoder.state_dict())
    verifier.type_embedding.load_state_dict(retype.type_embedding.state_dict())


def candidate_exact_labels(
    actions: list[AtomPairRetypeAction],
    current_states: torch.Tensor,
    expected_states: torch.Tensor,
) -> torch.Tensor:
    labels = current_states.new_zeros((len(actions),), dtype=torch.float32)
    for index, action in enumerate(actions):
        revised = current_states.clone()
        revised[action.group_index] = action.target_state
        labels[index] = float(bool((revised == expected_states).all()))
    return labels


def chemical_retype_verifier_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    ranking_margin: float,
) -> dict[str, torch.Tensor]:
    if scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError("Chemical verifier scores and labels differ in shape")
    if not len(scores):
        raise ValueError("Chemical verifier loss requires candidate actions")
    if not bool(((labels == 0.0) | (labels == 1.0)).all()):
        raise ValueError("Chemical verifier labels must be binary")
    positive = labels > 0.5
    negative = ~positive
    hardest_negative = scores[negative].amax() if bool(negative.any()) else scores.new_tensor(-8.0)
    if bool(positive.any()):
        if int(positive.sum()) != 1:
            raise ValueError("A RETYPE sample must have at most one exact action")
        positive_score = scores[positive][0]
        positive_loss = F.softplus(-positive_score)
        negative_loss = F.softplus(hardest_negative)
        ranking_loss = F.relu(float(ranking_margin) - positive_score + hardest_negative)
        loss = positive_loss + negative_loss + ranking_loss
    else:
        positive_loss = scores.sum() * 0.0
        ranking_loss = scores.sum() * 0.0
        negative_loss = F.softplus(scores.amax())
        loss = negative_loss
    return {
        "loss": loss,
        "positive_loss": positive_loss,
        "negative_loss": negative_loss,
        "ranking_loss": ranking_loss,
    }


def choose_verified_retype_action(
    scores: torch.Tensor,
    actions: list[AtomPairRetypeAction],
    *,
    minimum_margin: float,
) -> AtomPairRetypeAction | None:
    if len(scores) != len(actions):
        raise ValueError("Chemical verifier scores and actions differ in length")
    if not actions:
        return None
    index = int(scores.argmax())
    if float(scores[index].detach()) <= minimum_margin:
        return None
    return actions[index]


def load_atom_pair_chemical_retype_verifier(
    path: str | Path,
    device: torch.device,
) -> tuple[AtomPairChemicalRetypeVerifier, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("checkpoint_schema_version")
        != "atom-pair-chemical-retype-verifier-0.1"
    ):
        raise ValueError("Unsupported chemical retype verifier checkpoint schema")
    model = AtomPairChemicalRetypeVerifier(
        AtomPairChemicalRetypeVerifierConfig(**payload["config"])
    ).to(device)
    model.load_state_dict(payload["head"], strict=True)
    model.eval().requires_grad_(False)
    return model, payload
