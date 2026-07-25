"""Two-stage chemical retype gate and action over selected AtomPair groups."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from chemvector.atom_pair_one_move import AtomPairOneMoveHead
from chemvector.audit_atom_pair_chemical_reachability import (
    ABSENT_STATE,
    chemical_confidence_margin,
    typed_edit_kinds,
)
from chemvector.bond_object import BOND_TYPES


RETYPE_FEATURE_DIM = 18 + 1 + len(BOND_TYPES) + 3
TARGET_STATE_FEATURE_DIM = 2


@dataclass(frozen=True, slots=True)
class AtomPairChemicalRetypeConfig:
    input_dim: int = RETYPE_FEATURE_DIM
    hidden_dim: int = 64
    type_embedding_dim: int = 16
    retype_candidates: int = 16
    maximum_groups: int = 256

    def __post_init__(self) -> None:
        if min(
            self.input_dim,
            self.hidden_dim,
            self.type_embedding_dim,
            self.retype_candidates,
            self.maximum_groups,
        ) < 1:
            raise ValueError("Chemical retype dimensions must be positive")


@dataclass(frozen=True, slots=True)
class AtomPairRetypeAction:
    group_index: int
    target_state: int

    def __post_init__(self) -> None:
        if self.group_index < 0:
            raise ValueError("Retype action must reference a group")
        if not 0 <= self.target_state < len(BOND_TYPES):
            raise ValueError("Retype target must be a chemical bond type")


def build_retype_features(
    relation_features: torch.Tensor,
    base_logits: torch.Tensor,
    chemical_logits: torch.Tensor,
    selected: torch.Tensor,
) -> torch.Tensor:
    if not (
        len(relation_features)
        == len(base_logits)
        == len(chemical_logits)
        == len(selected)
    ):
        raise ValueError("Chemical retype feature tensors differ in length")
    if selected.dtype != torch.bool:
        raise ValueError("Chemical retype selection must be boolean")
    changed = selected != (base_logits >= 0.0)
    margin = chemical_confidence_margin(chemical_logits)
    return torch.cat(
        (
            relation_features.float(),
            base_logits.float()[:, None],
            chemical_logits.float(),
            margin[:, None],
            selected.float()[:, None],
            changed.float()[:, None],
        ),
        dim=-1,
    )


class AtomPairChemicalRetypeHead(nn.Module):
    def __init__(self, config: AtomPairChemicalRetypeConfig) -> None:
        super().__init__()
        self.config = config
        self.edge_encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
        )
        global_dim = config.hidden_dim * 2 + 5
        self.gate_head = nn.Sequential(
            nn.Linear(global_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
        )
        action_dim = (
            config.hidden_dim
            + global_dim
            + config.type_embedding_dim
            + TARGET_STATE_FEATURE_DIM
        )
        self.type_embedding = nn.Embedding(len(BOND_TYPES), config.type_embedding_dim)
        self.action_head = nn.Sequential(
            nn.Linear(action_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
        )
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.zeros_(self.gate_head[-1].bias)
        nn.init.zeros_(self.action_head[-1].weight)
        nn.init.zeros_(self.action_head[-1].bias)

    def score_sample(
        self,
        edge_features: torch.Tensor,
        chemical_logits: torch.Tensor,
        selected: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[AtomPairRetypeAction]]:
        if not (len(edge_features) == len(chemical_logits) == len(selected)):
            raise ValueError("Chemical retype sample tensors differ in length")
        if not len(edge_features):
            raise ValueError("Chemical retype scoring requires at least one group")
        if len(edge_features) > self.config.maximum_groups:
            raise ValueError("Chemical retype group count exceeds capacity")
        hidden = self.edge_encoder(edge_features)
        margins = chemical_confidence_margin(chemical_logits)
        changed = edge_features[:, -1] > 0.5
        global_feature = torch.cat(
            (
                hidden.mean(0),
                hidden.amax(0),
                hidden.new_tensor(
                    (
                        float(selected.sum()) / self.config.maximum_groups,
                        float(len(selected)) / self.config.maximum_groups,
                        float(changed.sum()) / self.config.maximum_groups,
                        float(margins.mean()) / 8.0,
                        float(margins.min()) / 8.0,
                    )
                ),
            )
        )
        gate_logit = self.gate_head(global_feature).squeeze(-1)
        selected_indices = torch.where(selected)[0]
        selected_indices = selected_indices[
            torch.argsort(margins[selected_indices], stable=True)[
                : self.config.retype_candidates
            ]
        ]
        predicted = chemical_logits.argmax(-1)
        top_logits = chemical_logits.float().amax(-1)
        rows = []
        actions = []
        for index in selected_indices.tolist():
            for target_state in range(len(BOND_TYPES)):
                if target_state == int(predicted[index]):
                    continue
                target_logit = chemical_logits[index, target_state].float()
                target_features = torch.stack(
                    (
                        target_logit.clamp(-8.0, 8.0) / 8.0,
                        (target_logit - top_logits[index]).clamp(-8.0, 0.0) / 8.0,
                    )
                )
                rows.append(
                    torch.cat(
                        (
                            hidden[index],
                            global_feature,
                            self.type_embedding.weight[target_state],
                            target_features,
                        )
                    )
                )
                actions.append(AtomPairRetypeAction(index, target_state))
        if not rows:
            return gate_logit, hidden.new_empty((0,)), actions
        return gate_logit, self.action_head(torch.stack(rows)).squeeze(-1), actions


@torch.no_grad()
def initialize_retype_encoder_from_one_move(
    retype: AtomPairChemicalRetypeHead,
    first: AtomPairOneMoveHead,
) -> None:
    if retype.config.hidden_dim != first.config.hidden_dim:
        raise ValueError("Retype and one-move hidden dimensions must match")
    retype.edge_encoder[0].weight.zero_()
    retype.edge_encoder[0].weight[:, : first.config.input_dim].copy_(
        first.edge_encoder[0].weight
    )
    retype.edge_encoder[0].bias.copy_(first.edge_encoder[0].bias)
    retype.edge_encoder[2].load_state_dict(first.edge_encoder[2].state_dict())


def target_retype_action_index(
    actions: list[AtomPairRetypeAction],
    current_states: torch.Tensor,
    expected_states: torch.Tensor,
) -> tuple[int, int | None] | None:
    edits = typed_edit_kinds(current_states, expected_states)
    if not edits:
        return 0, None
    if len(edits) != 1 or edits[0][1] != "retype":
        return None
    index, _ = edits[0]
    target = AtomPairRetypeAction(index, int(expected_states[index]))
    try:
        return 1, actions.index(target)
    except ValueError:
        return None


def apply_retype_action(
    states: torch.Tensor,
    action: AtomPairRetypeAction | None,
) -> torch.Tensor:
    output = states.clone()
    if action is not None:
        if int(output[action.group_index]) == ABSENT_STATE:
            raise ValueError("Retype action cannot add an absent edge")
        output[action.group_index] = action.target_state
    return output


def choose_retype_action(
    gate_logit: torch.Tensor,
    action_scores: torch.Tensor,
    actions: list[AtomPairRetypeAction],
    *,
    gate_threshold: float,
) -> AtomPairRetypeAction | None:
    if len(action_scores) != len(actions):
        raise ValueError("Retype scores and actions differ in length")
    if float(gate_logit.detach()) <= gate_threshold or not actions:
        return None
    return actions[int(action_scores.argmax())]


def joint_retype_logits(
    gate_logit: torch.Tensor,
    action_scores: torch.Tensor,
) -> torch.Tensor:
    if gate_logit.ndim != 0 or action_scores.ndim != 1:
        raise ValueError("Joint retype logits require a scalar gate and action vector")
    keep = gate_logit.new_zeros((1,))
    return torch.cat((keep, gate_logit + action_scores))


def choose_joint_retype_action(
    gate_logit: torch.Tensor,
    action_scores: torch.Tensor,
    actions: list[AtomPairRetypeAction],
    *,
    minimum_margin: float,
) -> AtomPairRetypeAction | None:
    if len(action_scores) != len(actions):
        raise ValueError("Retype scores and actions differ in length")
    if not actions:
        return None
    joint = gate_logit + action_scores
    index = int(joint.argmax())
    if float(joint[index].detach()) <= minimum_margin:
        return None
    return actions[index]


def chemical_retype_loss(
    gate_logit: torch.Tensor,
    gate_target: int,
    action_scores: torch.Tensor,
    action_target: int | None,
    *,
    gate_positive_weight: float,
    action_weight: float,
) -> dict[str, torch.Tensor]:
    target = gate_logit.new_tensor(float(gate_target))
    positive_weight = gate_logit.new_tensor(float(gate_positive_weight))
    gate = F.binary_cross_entropy_with_logits(
        gate_logit.float(), target.float(), pos_weight=positive_weight.float()
    )
    if action_target is None:
        action = gate * 0.0
    else:
        action = F.cross_entropy(
            action_scores[None].float(),
            torch.tensor((action_target,), device=action_scores.device),
        )
    return {
        "loss": gate + float(action_weight) * action,
        "gate_loss": gate,
        "action_loss": action,
    }


def chemical_retype_joint_loss(
    gate_logit: torch.Tensor,
    gate_target: int,
    action_scores: torch.Tensor,
    action_target: int | None,
    *,
    retype_weight: float,
) -> dict[str, torch.Tensor]:
    if gate_target not in {0, 1}:
        raise ValueError("Joint retype gate target must be binary")
    if gate_target == 0:
        if action_target is not None:
            raise ValueError("KEEP target cannot reference a retype action")
        target = 0
    else:
        if action_target is None:
            raise ValueError("RETYPE target must reference an action")
        if not 0 <= action_target < len(action_scores):
            raise ValueError("Joint retype action target is out of range")
        target = action_target + 1
    logits = joint_retype_logits(gate_logit, action_scores)
    loss = F.cross_entropy(
        logits[None].float(),
        torch.tensor((target,), device=logits.device),
    )
    if gate_target:
        loss = loss * float(retype_weight)
    return {"loss": loss, "joint_loss": loss}


def chemical_retype_safety_loss(
    gate_logit: torch.Tensor,
    action_scores: torch.Tensor,
    safe_target: int,
    *,
    positive_weight: float,
) -> dict[str, torch.Tensor]:
    if safe_target not in {0, 1}:
        raise ValueError("Retype safety target must be binary")
    if not len(action_scores):
        raise ValueError("Retype safety loss requires at least one action")
    safe_logit = gate_logit + action_scores.max()
    target = safe_logit.new_tensor(float(safe_target))
    weight = safe_logit.new_tensor(float(positive_weight))
    loss = F.binary_cross_entropy_with_logits(
        safe_logit.float(), target.float(), pos_weight=weight.float()
    )
    return {"loss": loss, "safety_loss": loss}


def load_atom_pair_chemical_retype_head(
    path: str | Path,
    device: torch.device,
) -> tuple[AtomPairChemicalRetypeHead, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("checkpoint_schema_version") != "atom-pair-chemical-retype-0.1":
        raise ValueError("Unsupported chemical retype checkpoint schema")
    model = AtomPairChemicalRetypeHead(
        AtomPairChemicalRetypeConfig(**payload["config"])
    ).to(device)
    model.load_state_dict(payload["head"], strict=True)
    model.eval().requires_grad_(False)
    return model, payload
