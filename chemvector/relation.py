"""Factorized edge-existence and bond-type supervision."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from chemvector.stroke import STROKE_CLASSES


BOND_CLASSES = ("SINGLE", "DOUBLE", "TRIPLE")
AROMATIC_CLASS_ID = STROKE_CLASSES.index("AROMATIC")


def apply_safe_hydrogen_constraints(
    edge_logits: torch.Tensor,
    bond_logits: torch.Tensor,
    pair_indices: torch.Tensor,
    hydrogen_like: torch.Tensor,
    atom_confidences: torch.Tensor,
    *,
    confidence_threshold: float = 0.95,
    disabled_logit: float = -30.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply only valence rules that are safe for high-confidence H/D/T labels."""

    if edge_logits.ndim != 2 or bond_logits.shape != (*edge_logits.shape, 3):
        raise ValueError("Batched edge and bond logits have incompatible shapes")
    if pair_indices.shape != (edge_logits.size(1), 2):
        raise ValueError("Pair indices do not match edge logits")
    if hydrogen_like.shape != atom_confidences.shape or hydrogen_like.ndim != 2:
        raise ValueError("Hydrogen mask and atom confidences must have shape [batch, atoms]")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("Hydrogen confidence threshold must be in [0, 1]")
    if pair_indices.numel() and int(pair_indices.max()) >= hydrogen_like.size(1):
        raise ValueError("Pair index is outside the atom dimension")

    constrained_edges = edge_logits.clone()
    constrained_bonds = bond_logits.clone()
    confident_hydrogen = hydrogen_like.bool() & (
        atom_confidences >= confidence_threshold
    )
    begin = pair_indices[:, 0]
    end = pair_indices[:, 1]
    hydrogen_pair = confident_hydrogen[:, begin] | confident_hydrogen[:, end]
    constrained_bonds[..., 1:] = constrained_bonds[..., 1:].masked_fill(
        hydrogen_pair.unsqueeze(-1), disabled_logit
    )

    for batch_index in range(edge_logits.size(0)):
        for atom_index in torch.where(confident_hydrogen[batch_index])[0].tolist():
            incident = torch.where((begin == atom_index) | (end == atom_index))[0]
            if len(incident) <= 1:
                continue
            keep = incident[edge_logits[batch_index, incident].argmax()]
            suppress = incident[incident != keep]
            constrained_edges[batch_index, suppress] = disabled_logit
    return constrained_edges, constrained_bonds


def factorized_relation_probabilities(
    edge_logits: torch.Tensor,
    bond_logits: torch.Tensor,
) -> torch.Tensor:
    """Compose P(NONE) and P(edge) * P(bond type) into five classes."""

    if edge_logits.shape != bond_logits.shape[:-1]:
        raise ValueError("Edge and bond logits have incompatible shapes")
    if bond_logits.size(-1) != len(BOND_CLASSES):
        raise ValueError("Bond logits do not match the bond vocabulary")
    edge_probability = edge_logits.float().sigmoid()
    bond_probabilities = bond_logits.float().softmax(dim=-1)
    visible_probabilities = torch.cat(
        (
            (1.0 - edge_probability).unsqueeze(-1),
            edge_probability.unsqueeze(-1) * bond_probabilities,
        ),
        dim=-1,
    )
    aromatic_probability = visible_probabilities.new_zeros(
        (*visible_probabilities.shape[:-1], 1)
    )
    return torch.cat((visible_probabilities, aromatic_probability), dim=-1)


def factorized_relation_loss(
    edge_logits: torch.Tensor,
    bond_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    edge_pos_weight: float = 4.0,
    edge_focal_gamma: float = 2.0,
    bond_class_weights: tuple[float, ...] = (1.0, 2.0, 6.0),
    edge_loss_weight: float = 1.0,
    bond_loss_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Train edge existence on all pairs and bond type only on true edges."""

    if edge_logits.ndim != 1 or bond_logits.ndim != 2 or labels.ndim != 1:
        raise ValueError("Relation logits and labels must be flattened")
    if edge_logits.size(0) != labels.size(0) or bond_logits.size(0) != labels.size(0):
        raise ValueError("Relation logits and labels have different lengths")
    if bond_logits.size(1) != len(BOND_CLASSES):
        raise ValueError("Bond logits do not match the bond vocabulary")
    if edge_pos_weight <= 0 or edge_focal_gamma < 0:
        raise ValueError("Edge loss weights are invalid")
    if len(bond_class_weights) != len(BOND_CLASSES):
        raise ValueError("One class weight is required per bond type")

    edge_targets = (labels != 0).to(dtype=edge_logits.dtype)
    binary_loss = F.binary_cross_entropy_with_logits(
        edge_logits,
        edge_targets,
        pos_weight=edge_logits.new_tensor(edge_pos_weight),
        reduction="none",
    )
    if edge_focal_gamma:
        probabilities = edge_logits.sigmoid()
        target_probabilities = torch.where(
            edge_targets.bool(),
            probabilities,
            1.0 - probabilities,
        )
        binary_loss = binary_loss * (1.0 - target_probabilities).pow(edge_focal_gamma)
    edge_loss = binary_loss.mean()

    visible_bond_mask = (labels > 0) & (labels < AROMATIC_CLASS_ID)
    if bool(visible_bond_mask.any()):
        bond_targets = labels[visible_bond_mask] - 1
        bond_loss = F.cross_entropy(
            bond_logits[visible_bond_mask],
            bond_targets,
            weight=bond_logits.new_tensor(bond_class_weights),
        )
    else:
        bond_loss = bond_logits.sum() * 0.0
    loss = edge_loss_weight * edge_loss + bond_loss_weight * bond_loss
    return {
        "loss": loss,
        "edge_loss": edge_loss,
        "bond_loss": bond_loss,
    }


def hard_mined_factorized_relation_loss(
    edge_logits: torch.Tensor,
    bond_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    negative_ratio: int = 8,
    minimum_negatives: int = 32,
    ranking_weight: float = 0.2,
    ranking_margin: float = 1.0,
    **loss_kwargs,
) -> dict[str, torch.Tensor]:
    """Keep every true edge and train on the currently hardest non-edges."""

    if negative_ratio < 1 or minimum_negatives < 1:
        raise ValueError("Hard-negative counts must be positive")
    if ranking_weight < 0 or ranking_margin < 0:
        raise ValueError("Ranking loss settings must be nonnegative")
    if edge_logits.ndim != 1 or labels.ndim != 1:
        raise ValueError("Hard-negative mining expects flattened relation tensors")
    if edge_logits.size(0) != labels.size(0):
        raise ValueError("Relation logits and labels have different lengths")

    positive_indices = torch.where(labels != 0)[0]
    negative_indices = torch.where(labels == 0)[0]
    negative_count = min(
        int(negative_indices.numel()),
        max(minimum_negatives, int(positive_indices.numel()) * negative_ratio),
    )
    if negative_count:
        hard_offsets = torch.topk(
            edge_logits[negative_indices].detach(),
            negative_count,
        ).indices
        hard_negative_indices = negative_indices[hard_offsets]
    else:
        hard_negative_indices = negative_indices
    selected = torch.cat((positive_indices, hard_negative_indices))
    if not len(selected):
        selected = torch.arange(labels.numel(), device=labels.device)
    losses = factorized_relation_loss(
        edge_logits[selected],
        bond_logits[selected],
        labels[selected],
        **loss_kwargs,
    )

    ranking_loss = edge_logits.sum() * 0.0
    if len(positive_indices) and len(hard_negative_indices):
        ranking_pairs = min(len(positive_indices), len(hard_negative_indices))
        weakest_positive = -torch.topk(
            -edge_logits[positive_indices],
            ranking_pairs,
        ).values
        strongest_negative = torch.topk(
            edge_logits[hard_negative_indices],
            ranking_pairs,
        ).values
        ranking_loss = F.softplus(
            ranking_margin + strongest_negative - weakest_positive
        ).mean()
    losses["ranking_loss"] = ranking_loss
    losses["loss"] = losses["loss"] + ranking_weight * ranking_loss
    losses["selected_pairs"] = edge_logits.new_tensor(float(len(selected)))
    losses["selected_negatives"] = edge_logits.new_tensor(
        float(len(hard_negative_indices))
    )
    return losses


def relation_teacher_consistency_loss(
    output: dict[str, torch.Tensor],
    teacher_output: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Preserve frozen edge-existence and conditional bond logits on replay pairs."""
    edge_logits = output["edge_logits"]
    bond_logits = output["bond_logits"]
    teacher_edge = teacher_output["edge_logits"].detach()
    teacher_bond = teacher_output["bond_logits"].detach()
    if edge_logits.shape != teacher_edge.shape or bond_logits.shape != teacher_bond.shape:
        raise ValueError("Teacher relation output shapes differ")
    edge_loss = F.smooth_l1_loss(edge_logits, teacher_edge)
    teacher_edge_weights = teacher_edge.sigmoid().square().unsqueeze(-1)
    bond_loss = (
        F.smooth_l1_loss(bond_logits, teacher_bond, reduction="none")
        * teacher_edge_weights
    ).sum() / (
        teacher_edge_weights.sum().clamp_min(1e-6) * bond_logits.size(-1)
    )
    return {
        "teacher_loss": edge_loss + bond_loss,
        "teacher_edge_loss": edge_loss,
        "teacher_bond_loss": bond_loss,
    }


def new_relation_totals() -> dict[str, int]:
    totals = {
        "pairs": 0,
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 0,
        "visible_bonds": 0,
        "bond_correct": 0,
        "aromatic_edges": 0,
    }
    for name in BOND_CLASSES:
        totals[f"{name}_count"] = 0
        totals[f"{name}_correct"] = 0
    return totals


@torch.no_grad()
def update_relation_metrics(
    totals: dict[str, int],
    edge_logits: torch.Tensor,
    bond_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    edge_threshold: float = 0.5,
) -> None:
    if not 0.0 <= edge_threshold <= 1.0:
        raise ValueError("edge_threshold must be in [0, 1]")
    predicted_edges = edge_logits.sigmoid() >= edge_threshold
    actual_edges = labels != 0
    predicted_bonds = bond_logits.argmax(dim=-1) + 1
    totals["pairs"] += int(labels.numel())
    totals["true_positive"] += int((predicted_edges & actual_edges).sum())
    totals["false_positive"] += int((predicted_edges & ~actual_edges).sum())
    totals["false_negative"] += int((~predicted_edges & actual_edges).sum())
    visible_bonds = (labels > 0) & (labels < AROMATIC_CLASS_ID)
    totals["visible_bonds"] += int(visible_bonds.sum())
    totals["aromatic_edges"] += int((labels == AROMATIC_CLASS_ID).sum())
    totals["bond_correct"] += int(
        ((predicted_bonds == labels) & visible_bonds).sum()
    )
    for class_id, name in enumerate(BOND_CLASSES, start=1):
        class_mask = labels == class_id
        totals[f"{name}_count"] += int(class_mask.sum())
        totals[f"{name}_correct"] += int(
            ((predicted_bonds == class_id) & class_mask).sum()
        )


def summarize_relation_metrics(totals: dict[str, int]) -> dict[str, float]:
    precision = totals["true_positive"] / max(
        totals["true_positive"] + totals["false_positive"], 1
    )
    recall = totals["true_positive"] / max(
        totals["true_positive"] + totals["false_negative"], 1
    )
    result = {
        "edge_precision": precision,
        "edge_recall": recall,
        "edge_f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "visible_bond_accuracy": totals["bond_correct"]
        / max(totals["visible_bonds"], 1),
        "aromatic_edges_excluded": float(totals["aromatic_edges"]),
    }
    for name in BOND_CLASSES:
        result[f"{name.lower()}_recall"] = totals[f"{name}_correct"] / max(
            totals[f"{name}_count"], 1
        )
    return result
