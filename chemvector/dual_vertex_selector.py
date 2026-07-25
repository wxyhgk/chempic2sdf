"""Role-aware candidate utilities for independent text and implicit vertex heads."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from chemvector.geometry import stable_pairwise_distances
from chemvector.grounded_graph import radius_aware_linear_assignment


IMPLICIT_ROLE = 0
TEXT_ROLE = 1
ROLE_COUNT = 2


def merge_dual_candidates(
    text_candidates: list[dict],
    implicit_candidates: list[dict],
    *,
    maximum_candidates: int,
) -> tuple[list[dict], list[int]]:
    """Keep all high-value text objects, then fill capacity with implicit peaks."""

    if maximum_candidates < 1:
        raise ValueError("maximum_candidates must be positive")
    text = sorted(
        text_candidates,
        key=lambda item: float(item["score"]),
        reverse=True,
    )[:maximum_candidates]
    remaining = maximum_candidates - len(text)
    implicit = sorted(
        implicit_candidates,
        key=lambda item: float(item["score"]),
        reverse=True,
    )[:remaining]
    merged = [dict(item, candidate_role=TEXT_ROLE) for item in text]
    merged.extend(dict(item, candidate_role=IMPLICIT_ROLE) for item in implicit)
    return merged, [TEXT_ROLE] * len(text) + [IMPLICIT_ROLE] * len(implicit)


def dual_candidate_auxiliary(
    base_auxiliary: torch.Tensor,
    merged_candidates: list[list[dict]],
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    """Append text-object extents to the existing dense candidate descriptors."""

    if base_auxiliary.shape[:2] != candidate_mask.shape:
        raise ValueError("Candidate auxiliary and mask shapes differ")
    extents = torch.zeros(
        (*candidate_mask.shape, 2),
        device=base_auxiliary.device,
        dtype=base_auxiliary.dtype,
    )
    for batch_index, candidates in enumerate(merged_candidates):
        if len(candidates) > candidate_mask.size(1):
            raise ValueError("Merged candidate count exceeds padded capacity")
        for candidate_index, item in enumerate(candidates):
            extents[batch_index, candidate_index, 0] = float(
                item.get("half_width", 0.0)
            )
            extents[batch_index, candidate_index, 1] = float(
                item.get("half_height", 0.0)
            )
    return torch.cat((base_auxiliary, extents), dim=-1)


def padded_candidate_roles(
    role_batches: list[list[int]],
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    if len(role_batches) != candidate_mask.size(0):
        raise ValueError("Candidate role batch size differs")
    roles = torch.zeros_like(candidate_mask, dtype=torch.long)
    for batch_index, values in enumerate(role_batches):
        if len(values) > candidate_mask.size(1):
            raise ValueError("Candidate role count exceeds padded capacity")
        if values:
            roles[batch_index, : len(values)] = torch.tensor(
                values,
                device=candidate_mask.device,
                dtype=torch.long,
            )
    return roles


def role_aware_candidate_targets(
    candidate_points: torch.Tensor,
    candidate_mask: torch.Tensor,
    candidate_roles: torch.Tensor,
    target_points: torch.Tensor,
    target_mask: torch.Tensor,
    target_roles: torch.Tensor,
    *,
    matching_radius_pixels: float,
    image_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match each role independently so text and implicit peaks cannot trade credit."""

    if candidate_points.shape != (*candidate_mask.shape, 2):
        raise ValueError("Candidate point shape differs from mask")
    if candidate_roles.shape != candidate_mask.shape:
        raise ValueError("Candidate role shape differs from mask")
    if target_points.shape != (*target_mask.shape, 2):
        raise ValueError("Target point shape differs from mask")
    if target_roles.shape != target_mask.shape:
        raise ValueError("Target role shape differs from mask")
    if matching_radius_pixels <= 0 or image_size < 1:
        raise ValueError("Matching radius and image size must be positive")

    targets = torch.zeros_like(candidate_mask, dtype=torch.float32)
    assignments = torch.full_like(candidate_roles, -1)
    for batch_index in range(candidate_points.size(0)):
        for role in range(ROLE_COUNT):
            candidate_ids = torch.where(
                candidate_mask[batch_index]
                & (candidate_roles[batch_index] == role)
            )[0]
            target_ids = torch.where(
                target_mask[batch_index] & (target_roles[batch_index] == role)
            )[0]
            if not len(candidate_ids) or not len(target_ids):
                continue
            distances = stable_pairwise_distances(
                candidate_points[batch_index, candidate_ids],
                target_points[batch_index, target_ids],
            ).mul(image_size)
            candidate_offsets, target_offsets = radius_aware_linear_assignment(
                distances,
                matching_radius_pixels,
            )
            accepted = (
                distances[candidate_offsets, target_offsets]
                <= matching_radius_pixels
            )
            matched_candidates = candidate_ids[candidate_offsets[accepted]]
            matched_targets = target_ids[target_offsets[accepted]]
            targets[batch_index, matched_candidates] = 1.0
            assignments[batch_index, matched_candidates] = matched_targets
    return targets, assignments


def role_aware_candidate_groups(
    candidate_points: torch.Tensor,
    candidate_mask: torch.Tensor,
    candidate_roles: torch.Tensor,
    target_points: torch.Tensor,
    target_mask: torch.Tensor,
    target_roles: torch.Tensor,
    *,
    matching_radius_pixels: float,
    image_size: int,
) -> torch.Tensor:
    """Assign every in-radius alternative candidate to its nearest target group."""

    groups = torch.full_like(candidate_roles, -1)
    for batch_index in range(candidate_points.size(0)):
        for role in range(ROLE_COUNT):
            candidate_ids = torch.where(
                candidate_mask[batch_index]
                & (candidate_roles[batch_index] == role)
            )[0]
            target_ids = torch.where(
                target_mask[batch_index] & (target_roles[batch_index] == role)
            )[0]
            if not len(candidate_ids) or not len(target_ids):
                continue
            distances = stable_pairwise_distances(
                candidate_points[batch_index, candidate_ids],
                target_points[batch_index, target_ids],
            ).mul(image_size)
            nearest_distances, nearest_offsets = distances.min(dim=1)
            accepted = nearest_distances <= matching_radius_pixels
            groups[batch_index, candidate_ids[accepted]] = target_ids[
                nearest_offsets[accepted]
            ]
    return groups


def line_interior_candidate_mask(
    candidate_points: torch.Tensor,
    candidate_mask: torch.Tensor,
    group_ids: torch.Tensor,
    target_points: torch.Tensor,
    target_mask: torch.Tensor,
    adjacency: torch.Tensor,
    *,
    image_size: int,
    maximum_line_distance_pixels: float = 3.0,
    minimum_endpoint_distance_pixels: float = 8.0,
    ambiguous_atom_radius_pixels: float = 10.0,
    minimum_projection: float = 0.15,
    maximum_projection: float = 0.85,
) -> torch.Tensor:
    """Mark unambiguous false candidates lying inside a true bond segment."""

    if candidate_points.shape != (*candidate_mask.shape, 2):
        raise ValueError("Candidate point shape differs from mask")
    if group_ids.shape != candidate_mask.shape:
        raise ValueError("Candidate groups and mask shapes differ")
    if target_points.shape != (*target_mask.shape, 2):
        raise ValueError("Target point shape differs from mask")
    if adjacency.shape != (
        target_points.size(0), target_points.size(1), target_points.size(1)
    ):
        raise ValueError("Adjacency shape differs from target points")
    if (
        image_size < 1
        or min(
            maximum_line_distance_pixels,
            minimum_endpoint_distance_pixels,
            ambiguous_atom_radius_pixels,
        )
        <= 0
        or not 0.0 <= minimum_projection < maximum_projection <= 1.0
    ):
        raise ValueError("Line-interior supervision parameters are invalid")

    output = torch.zeros_like(candidate_mask)
    for batch_index in range(candidate_points.size(0)):
        valid_candidates = candidate_mask[batch_index] & (group_ids[batch_index] < 0)
        candidate_ids = torch.where(valid_candidates)[0]
        target_ids = torch.where(target_mask[batch_index])[0]
        if not len(candidate_ids) or not len(target_ids):
            continue
        local_adjacency = adjacency[batch_index][target_ids][:, target_ids]
        bond_offsets = torch.triu(local_adjacency, diagonal=1).nonzero()
        if not len(bond_offsets):
            continue

        candidates = candidate_points[batch_index, candidate_ids].float() * image_size
        targets = target_points[batch_index, target_ids].float() * image_size
        starts = targets[bond_offsets[:, 0]]
        ends = targets[bond_offsets[:, 1]]
        segments = ends - starts
        squared_lengths = segments.square().sum(dim=-1).clamp_min(1e-6)
        relative = candidates[:, None, :] - starts[None, :, :]
        projection = (relative * segments[None]).sum(dim=-1) / squared_lengths[None]
        projected = starts[None] + projection[..., None] * segments[None]
        line_distance = torch.linalg.vector_norm(
            candidates[:, None, :] - projected, dim=-1
        )
        start_distance = torch.linalg.vector_norm(
            candidates[:, None, :] - starts[None], dim=-1
        )
        end_distance = torch.linalg.vector_norm(
            candidates[:, None, :] - ends[None], dim=-1
        )
        atom_distance = torch.cdist(candidates, targets).amin(dim=-1)
        on_line = (
            (line_distance <= maximum_line_distance_pixels)
            & (projection >= minimum_projection)
            & (projection <= maximum_projection)
            & (start_distance >= minimum_endpoint_distance_pixels)
            & (end_distance >= minimum_endpoint_distance_pixels)
        ).any(dim=-1)
        unambiguous = atom_distance > ambiguous_atom_radius_pixels
        output[batch_index, candidate_ids[on_line & unambiguous]] = True
    return output


def grouped_candidate_selection_loss(
    logits: torch.Tensor,
    group_ids: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    ranking_weight: float = 4.0,
    ranking_margin: float = 1.0,
    background_weight: float = 1.0,
    absolute_boundary_weight: float = 0.0,
    positive_floor_probability: float = 0.8,
    background_ceiling_probability: float = 0.2,
    hard_background_count: int = 16,
    line_interior_mask: torch.Tensor | None = None,
    line_interior_weight: float = 0.0,
    line_interior_margin: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Train one REAL representative per target group without arbitrary identity labels."""

    if logits.shape != group_ids.shape or logits.shape != candidate_mask.shape:
        raise ValueError("Grouped candidate tensors must have matching shapes")
    if (
        min(
            ranking_weight,
            background_weight,
            absolute_boundary_weight,
            line_interior_weight,
        )
        < 0
        or min(ranking_margin, line_interior_margin) <= 0
    ):
        raise ValueError("Grouped candidate loss weights or margin are invalid")
    if line_interior_mask is not None and line_interior_mask.shape != logits.shape:
        raise ValueError("Line-interior mask must match candidate logits")
    if not 0.0 < positive_floor_probability < 1.0:
        raise ValueError("Positive floor probability must lie in (0, 1)")
    if not 0.0 < background_ceiling_probability < 1.0:
        raise ValueError("Background ceiling probability must lie in (0, 1)")
    if hard_background_count < 1:
        raise ValueError("Hard background count must be positive")
    positive_losses = []
    background_losses = []
    ranking_losses = []
    boundary_losses = []
    line_interior_losses = []
    positive_floor = torch.logit(logits.new_tensor(positive_floor_probability))
    background_ceiling = torch.logit(
        logits.new_tensor(background_ceiling_probability)
    )
    masks = (
        line_interior_mask
        if line_interior_mask is not None
        else torch.zeros_like(candidate_mask)
    )
    for scores, groups, valid, line_interior in zip(
        logits, group_ids, candidate_mask, masks
    ):
        valid_scores = scores[valid]
        valid_groups = groups[valid]
        background = valid_scores[valid_groups < 0]
        if len(background):
            background_losses.append(F.softplus(background).mean())
        group_scores = []
        for group_id in valid_groups[valid_groups >= 0].unique():
            group_scores.append(valid_scores[valid_groups == group_id].max())
        if group_scores:
            group_scores = torch.stack(group_scores)
            positive_losses.append(F.softplus(-group_scores).mean())
            boundary = F.softplus(positive_floor - group_scores).mean()
            if len(background):
                hard_background = background.topk(
                    min(hard_background_count, len(background))
                ).values
                ranking_losses.append(
                    F.softplus(
                        ranking_margin
                        + hard_background[:, None]
                        - group_scores[None, :]
                    ).mean()
                )
                boundary = boundary + F.softplus(
                    hard_background - background_ceiling
                ).mean()
            boundary_losses.append(boundary)
            explicit_line_background = scores[
                valid & line_interior & (groups < 0)
            ]
            if len(explicit_line_background):
                hardest_line_background = explicit_line_background.max()
                line_interior_losses.append(
                    F.softplus(
                        line_interior_margin
                        + hardest_line_background
                        - group_scores
                    ).mean()
                )
    zero = logits.sum() * 0.0
    positive_loss = torch.stack(positive_losses).mean() if positive_losses else zero
    background_loss = (
        torch.stack(background_losses).mean() if background_losses else zero
    )
    ranking_loss = torch.stack(ranking_losses).mean() if ranking_losses else zero
    boundary_loss = torch.stack(boundary_losses).mean() if boundary_losses else zero
    line_interior_loss = (
        torch.stack(line_interior_losses).mean() if line_interior_losses else zero
    )
    selection_loss = positive_loss + background_weight * background_loss
    return {
        "loss": selection_loss
        + ranking_weight * ranking_loss
        + absolute_boundary_weight * boundary_loss
        + line_interior_weight * line_interior_loss,
        "selection_loss": selection_loss,
        "whole_set_loss": ranking_loss,
        "extreme_set_loss": zero,
        "positive_group_loss": positive_loss,
        "background_loss": background_loss,
        "absolute_boundary_loss": boundary_loss,
        "line_interior_ranking_loss": line_interior_loss,
    }


def role_aware_match_count(
    predicted_points: torch.Tensor,
    predicted_roles: torch.Tensor,
    target_points: torch.Tensor,
    target_roles: torch.Tensor,
    *,
    matching_radius_pixels: float,
    image_size: int,
) -> int:
    """Count spatial matches while preventing text/implicit cross-role credit."""

    if predicted_points.ndim != 2 or predicted_points.size(-1) != 2:
        raise ValueError("Predicted points must have shape [points, 2]")
    if target_points.ndim != 2 or target_points.size(-1) != 2:
        raise ValueError("Target points must have shape [points, 2]")
    if predicted_roles.shape != (len(predicted_points),):
        raise ValueError("Predicted roles do not match points")
    if target_roles.shape != (len(target_points),):
        raise ValueError("Target roles do not match points")
    matches = 0
    for role in range(ROLE_COUNT):
        predicted = predicted_points[predicted_roles == role]
        targets = target_points[target_roles == role]
        if not len(predicted) or not len(targets):
            continue
        distances = stable_pairwise_distances(predicted, targets).mul(image_size)
        predicted_offsets, target_offsets = radius_aware_linear_assignment(
            distances, matching_radius_pixels
        )
        matches += int(
            (
                distances[predicted_offsets, target_offsets]
                <= matching_radius_pixels
            ).sum()
        )
    return matches


def role_suppressed_topk_indices(
    scores: torch.Tensor,
    points: torch.Tensor,
    candidate_mask: torch.Tensor,
    candidate_roles: torch.Tensor,
    counts_by_role: torch.Tensor,
    *,
    suppression_radius_pixels: float,
    image_size: int,
) -> torch.Tensor:
    """Select per-role top-k while suppressing duplicate peaks for one vertex."""

    if scores.ndim != 1 or points.shape != (len(scores), 2):
        raise ValueError("Scores and points must describe one candidate set")
    if candidate_mask.shape != scores.shape or candidate_roles.shape != scores.shape:
        raise ValueError("Candidate mask or roles differ from scores")
    if counts_by_role.shape != (ROLE_COUNT,):
        raise ValueError("Role counts have wrong shape")
    if suppression_radius_pixels <= 0 or image_size < 1:
        raise ValueError("Suppression radius and image size must be positive")

    selected: list[int] = []
    for role, requested in enumerate(counts_by_role.tolist()):
        role_indices = torch.where(candidate_mask & (candidate_roles == role))[0]
        if not len(role_indices) or requested <= 0:
            continue
        ordered = role_indices[scores[role_indices].argsort(descending=True)]
        accepted: list[int] = []
        for index in ordered.tolist():
            if accepted:
                distances = (
                    stable_pairwise_distances(
                        points[index].unsqueeze(0), points[accepted]
                    )
                    * image_size
                )
                if bool((distances <= suppression_radius_pixels).any()):
                    continue
            accepted.append(index)
            if len(accepted) >= int(requested):
                break
        selected.extend(accepted)
    return torch.tensor(selected, device=scores.device, dtype=torch.long)


def role_suppressed_threshold_indices(
    logits: torch.Tensor,
    points: torch.Tensor,
    candidate_mask: torch.Tensor,
    candidate_roles: torch.Tensor,
    *,
    probability_threshold: float,
    suppression_radius_pixels: float,
    image_size: int,
) -> torch.Tensor:
    """Decode calibrated REAL logits with same-role duplicate suppression."""

    if not 0.0 < probability_threshold < 1.0:
        raise ValueError("REAL probability threshold must lie in (0, 1)")
    return role_suppressed_role_threshold_indices(
        logits,
        points,
        candidate_mask,
        candidate_roles,
        probability_thresholds=logits.new_full((ROLE_COUNT,), probability_threshold),
        suppression_radius_pixels=suppression_radius_pixels,
        image_size=image_size,
    )


def role_suppressed_role_threshold_indices(
    logits: torch.Tensor,
    points: torch.Tensor,
    candidate_mask: torch.Tensor,
    candidate_roles: torch.Tensor,
    *,
    probability_thresholds: torch.Tensor,
    suppression_radius_pixels: float,
    image_size: int,
) -> torch.Tensor:
    """Decode REAL logits with one calibrated threshold per visual role."""

    if probability_thresholds.shape != (ROLE_COUNT,):
        raise ValueError("Role probability thresholds have wrong shape")
    if bool(
        ((probability_thresholds <= 0.0) | (probability_thresholds >= 1.0)).any()
    ):
        raise ValueError("Role probability thresholds must lie in (0, 1)")
    threshold_logits = torch.logit(
        probability_thresholds.to(device=logits.device, dtype=logits.dtype)
    )
    eligible = candidate_mask & (
        logits >= threshold_logits[candidate_roles.long()]
    )
    counts = []
    for role in range(ROLE_COUNT):
        counts.append(
            int(
                (
                    eligible
                    & (candidate_roles == role)
                ).sum()
            )
        )
    selected = role_suppressed_topk_indices(
        logits,
        points,
        eligible,
        candidate_roles,
        torch.tensor(counts, device=logits.device, dtype=torch.long),
        suppression_radius_pixels=suppression_radius_pixels,
        image_size=image_size,
    )
    return selected
