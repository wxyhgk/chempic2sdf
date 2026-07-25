"""Chemical-aware one-edit actions over frozen AtomPair groups."""

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
    typed_candidate_indices,
    typed_edit_kinds,
)
from chemvector.bond_object import BOND_TYPES


CHEMICAL_ACTION_FEATURE_DIM = 18 + 1 + len(BOND_TYPES) + 3


@dataclass(frozen=True, slots=True)
class AtomPairChemicalActionConfig:
    input_dim: int = CHEMICAL_ACTION_FEATURE_DIM
    hidden_dim: int = 64
    type_embedding_dim: int = 16
    delete_candidates: int = 4
    add_candidates: int = 16
    retype_candidates: int = 16
    maximum_groups: int = 256

    def __post_init__(self) -> None:
        if min(
            self.input_dim,
            self.hidden_dim,
            self.type_embedding_dim,
            self.delete_candidates,
            self.add_candidates,
            self.retype_candidates,
            self.maximum_groups,
        ) < 1:
            raise ValueError("Chemical action dimensions must be positive")


@dataclass(frozen=True, slots=True)
class AtomPairChemicalAction:
    kind: str
    group_index: int = -1
    target_state: int = ABSENT_STATE

    def __post_init__(self) -> None:
        if self.kind not in {"keep", "delete", "add", "retype"}:
            raise ValueError(f"Unsupported chemical action kind: {self.kind}")
        if self.kind == "keep":
            if self.group_index != -1:
                raise ValueError("KEEP must not reference a group")
        elif self.group_index < 0:
            raise ValueError("Chemical edit must reference a group")
        if not 0 <= self.target_state <= ABSENT_STATE:
            raise ValueError("Chemical action target state is outside the vocabulary")


def build_chemical_action_features(
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
        raise ValueError("Chemical action feature tensors differ in length")
    if selected.dtype != torch.bool:
        raise ValueError("Chemical action selection must be boolean")
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


class AtomPairChemicalActionHead(nn.Module):
    def __init__(self, config: AtomPairChemicalActionConfig) -> None:
        super().__init__()
        self.config = config
        self.edge_encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
        )
        global_dim = config.hidden_dim * 2 + 5
        self.keep_head = nn.Sequential(
            nn.Linear(global_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
        )
        edge_action_dim = config.hidden_dim + global_dim
        typed_action_dim = edge_action_dim + config.type_embedding_dim
        self.type_embedding = nn.Embedding(len(BOND_TYPES), config.type_embedding_dim)
        self.delete_head = nn.Sequential(
            nn.Linear(edge_action_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
        )
        self.add_head = nn.Sequential(
            nn.Linear(typed_action_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
        )
        self.retype_head = nn.Sequential(
            nn.Linear(typed_action_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
        )
        for head in (self.keep_head, self.delete_head, self.add_head, self.retype_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def score_sample(
        self,
        edge_features: torch.Tensor,
        base_logits: torch.Tensor,
        chemical_logits: torch.Tensor,
        selected: torch.Tensor,
    ) -> tuple[torch.Tensor, list[AtomPairChemicalAction]]:
        if not (
            len(edge_features)
            == len(base_logits)
            == len(chemical_logits)
            == len(selected)
        ):
            raise ValueError("Chemical action sample tensors differ in length")
        if not len(edge_features):
            raise ValueError("Chemical action scoring requires at least one group")
        if len(edge_features) > self.config.maximum_groups:
            raise ValueError("Chemical action group count exceeds capacity")
        hidden = self.edge_encoder(edge_features)
        changed = selected != (base_logits >= 0.0)
        margins = chemical_confidence_margin(chemical_logits)
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
        scores = [self.keep_head(global_feature).squeeze(-1)]
        actions = [AtomPairChemicalAction("keep")]
        delete, add, retype = typed_candidate_indices(
            base_logits,
            chemical_logits,
            selected,
            delete_candidates=self.config.delete_candidates,
            add_candidates=self.config.add_candidates,
            retype_candidates=self.config.retype_candidates,
        )
        delete_indices = torch.tensor(
            sorted(delete), device=hidden.device, dtype=torch.long
        )
        if len(delete_indices):
            values = torch.cat(
                (
                    hidden[delete_indices],
                    global_feature.expand(len(delete_indices), -1),
                ),
                dim=-1,
            )
            scores.extend(self.delete_head(values).squeeze(-1).unbind())
            actions.extend(
                AtomPairChemicalAction(
                    "delete", group_index=int(index), target_state=ABSENT_STATE
                )
                for index in delete_indices.tolist()
            )

        predicted_types = chemical_logits.argmax(-1)
        for kind, indices, head in (
            ("add", sorted(add), self.add_head),
            ("retype", sorted(retype), self.retype_head),
        ):
            action_rows = []
            action_values = []
            for index in indices:
                for target_state in range(len(BOND_TYPES)):
                    if kind == "retype" and target_state == int(predicted_types[index]):
                        continue
                    action_rows.append((index, target_state))
                    action_values.append(
                        torch.cat(
                            (
                                hidden[index],
                                global_feature,
                                self.type_embedding.weight[target_state],
                            )
                        )
                    )
            if action_values:
                values = torch.stack(action_values)
                scores.extend(head(values).squeeze(-1).unbind())
                actions.extend(
                    AtomPairChemicalAction(
                        kind, group_index=index, target_state=target_state
                    )
                    for index, target_state in action_rows
                )
        return torch.stack(scores), actions


@torch.no_grad()
def initialize_chemical_action_from_one_move(
    chemical: AtomPairChemicalActionHead,
    first: AtomPairOneMoveHead,
) -> None:
    if chemical.config.hidden_dim != first.config.hidden_dim:
        raise ValueError("Chemical and one-move hidden dimensions must match")
    chemical.edge_encoder[0].weight.zero_()
    chemical.edge_encoder[0].weight[:, : first.config.input_dim].copy_(
        first.edge_encoder[0].weight
    )
    chemical.edge_encoder[0].bias.copy_(first.edge_encoder[0].bias)
    chemical.edge_encoder[2].load_state_dict(first.edge_encoder[2].state_dict())

    def copy_head_prefix(target: nn.Sequential, source: nn.Sequential) -> None:
        target[0].weight.zero_()
        target[0].weight[:, : source[0].weight.shape[1]].copy_(source[0].weight)
        target[0].bias.copy_(source[0].bias)
        target[-1].load_state_dict(source[-1].state_dict())

    copy_head_prefix(chemical.keep_head, first.keep_head)
    copy_head_prefix(chemical.delete_head, first.delete_head)
    copy_head_prefix(chemical.add_head, first.add_head)
    copy_head_prefix(chemical.retype_head, first.add_head)


def target_chemical_action_index(
    actions: list[AtomPairChemicalAction],
    current_states: torch.Tensor,
    expected_states: torch.Tensor,
) -> int | None:
    edits = typed_edit_kinds(current_states, expected_states)
    if not edits:
        target = AtomPairChemicalAction("keep")
    elif len(edits) == 1:
        index, kind = edits[0]
        target = AtomPairChemicalAction(
            kind,
            group_index=index,
            target_state=int(expected_states[index]),
        )
    else:
        return None
    try:
        return actions.index(target)
    except ValueError:
        return None


def apply_chemical_action(
    states: torch.Tensor,
    action: AtomPairChemicalAction,
) -> torch.Tensor:
    output = states.clone()
    if action.kind != "keep":
        output[action.group_index] = action.target_state
    return output


def choose_chemical_action(
    scores: torch.Tensor,
    actions: list[AtomPairChemicalAction],
    *,
    minimum_margin: float,
) -> tuple[AtomPairChemicalAction, float]:
    if len(scores) != len(actions) or not actions:
        raise ValueError("Chemical action scores and actions differ in length")
    if len(scores) == 1:
        return actions[0], 0.0
    best_edit = int(scores[1:].argmax()) + 1
    margin = float((scores[best_edit] - scores[0]).detach())
    if margin <= minimum_margin:
        return actions[0], margin
    return actions[best_edit], margin


def chemical_action_loss(
    scores: torch.Tensor,
    target_index: int,
    *,
    edit_weight: float,
) -> torch.Tensor:
    target = torch.tensor((target_index,), device=scores.device, dtype=torch.long)
    loss = F.cross_entropy(scores[None].float(), target)
    return loss * (edit_weight if target_index else 1.0)


def load_atom_pair_chemical_action_head(
    path: str | Path,
    device: torch.device,
) -> tuple[AtomPairChemicalActionHead, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("checkpoint_schema_version") != "atom-pair-chemical-action-0.1":
        raise ValueError("Unsupported chemical action checkpoint schema")
    model = AtomPairChemicalActionHead(
        AtomPairChemicalActionConfig(**payload["config"])
    ).to(device)
    model.load_state_dict(payload["head"], strict=True)
    model.eval().requires_grad_(False)
    return model, payload
