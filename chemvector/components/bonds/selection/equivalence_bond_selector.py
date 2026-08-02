"""GT atom-pair equivalence supervision for deployable Bond selection."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def build_equivalence_labels(base_example: dict) -> dict[str, torch.Tensor | int]:
    """Build training-only candidate groups without changing model features."""

    mask = base_example["mask"].bool()
    pairs = base_example["pairs"].long()
    keep_targets = base_example["keep_targets"].bool()
    if pairs.shape != (*mask.shape, 2) or keep_targets.shape != mask.shape:
        raise ValueError("Bond equivalence label inputs have incompatible shapes")

    expected_pairs = [tuple(sorted(map(int, pair))) for pair in base_example["expected_pairs"]]
    expected_display = {
        pair: int(target)
        for pair, target in zip(expected_pairs, base_example["expected_display"])
    }
    expected_chemical = {
        pair: int(target)
        for pair, target in zip(expected_pairs, base_example["expected_chemical"])
    }
    expected_set = set(expected_pairs)

    maximum_groups = int(mask.numel())
    group_ids = torch.full(mask.shape, -1, dtype=torch.long)
    group_mask = torch.zeros(maximum_groups, dtype=torch.bool)
    group_targets = torch.zeros(maximum_groups, dtype=torch.bool)
    group_pairs = torch.full((maximum_groups, 2), -1, dtype=torch.long)
    group_display_targets = torch.zeros(maximum_groups, dtype=torch.long)
    group_chemical_targets = torch.zeros(maximum_groups, dtype=torch.long)

    pair_to_group: dict[tuple[int, int], int] = {}
    for candidate_index in torch.where(mask)[0].tolist():
        pair = tuple(sorted(map(int, pairs[candidate_index].tolist())))
        group_index = pair_to_group.setdefault(pair, len(pair_to_group))
        group_ids[candidate_index] = group_index

    for pair, group_index in pair_to_group.items():
        group_mask[group_index] = True
        group_pairs[group_index] = torch.tensor(pair)
        if pair in expected_set:
            group_targets[group_index] = True
            group_display_targets[group_index] = expected_display[pair]
            group_chemical_targets[group_index] = expected_chemical[pair]

    canonical_targets = keep_targets & mask
    for group_index in torch.where(group_targets)[0].tolist():
        members = mask & group_ids.eq(group_index)
        canonical_count = int((canonical_targets & members).sum())
        if canonical_count != 1:
            pair = tuple(map(int, group_pairs[group_index].tolist()))
            raise ValueError(
                f"Representable GT Bond {pair} has {canonical_count} canonical candidates"
            )
    if bool((canonical_targets & ~group_targets[group_ids.clamp_min(0)] & mask).any()):
        raise ValueError("Canonical Bond candidate belongs to a negative equivalence group")

    representable_count = int(group_targets.sum())
    if representable_count != int(canonical_targets.sum()):
        raise ValueError("Representable Bond count differs from canonical target count")
    if representable_count > int(base_example["count_target"]):
        raise ValueError("Representable Bond count exceeds the complete GT count")

    return {
        "group_ids": group_ids,
        "group_mask": group_mask,
        "group_targets": group_targets,
        "group_pairs": group_pairs,
        "canonical_targets": canonical_targets,
        "group_display_targets": group_display_targets,
        "group_chemical_targets": group_chemical_targets,
        "representable_count": representable_count,
        "gt_count": int(base_example["count_target"]),
    }


def attach_equivalence_labels(junction_example: dict, base_example: dict) -> dict:
    """Attach Oracle labels as side targets while preserving deployable features."""

    if junction_example["key"] != base_example["key"]:
        raise ValueError("Junction and base Bond caches have different sample order")
    if junction_example["mask"].shape != base_example["mask"].shape:
        raise ValueError("Junction and base Bond caches have different candidate counts")
    features = junction_example["features"]
    output = dict(junction_example)
    output.update(build_equivalence_labels(base_example))
    if output["features"] is not features:
        raise AssertionError("Equivalence labels must not replace model features")
    return output


def normalized_logmeanexp(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Aggregate one equivalence group without rewarding duplicate candidates."""

    if logits.ndim != 1 or not logits.numel():
        raise ValueError("Equivalence group logits must be a non-empty vector")
    if temperature <= 0.0:
        raise ValueError("Equivalence aggregation temperature must be positive")
    scaled = logits.float() / temperature
    return temperature * (torch.logsumexp(scaled, dim=0) - math.log(logits.numel()))


def equivalence_bond_selector_loss(
    output: dict[str, torch.Tensor],
    mask: torch.Tensor,
    group_ids: torch.Tensor,
    group_mask: torch.Tensor,
    group_targets: torch.Tensor,
    canonical_targets: torch.Tensor,
    representable_counts: torch.Tensor,
    group_display_targets: torch.Tensor,
    group_chemical_targets: torch.Tensor,
    *,
    positive_weight: float = 1.0,
    negative_weight: float = 0.5,
    exactly_one_weight: float = 1.0,
    representative_weight: float = 0.0,
    cardinality_weight: float = 0.0,
    margin_weight: float = 0.0,
    margin: float = 1.0,
    type_weight: float = 0.0,
    temperature: float = 1.0,
    positive_mode: str = "canonical",
    negative_mode: str = "max",
    cardinality_mode: str = "none",
    display_class_weights: torch.Tensor | None = None,
    chemical_class_weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Train one decision per GT atom pair and one representative per true Bond."""

    keep_logits = output["keep_logits"].float()
    expected_shape = mask.shape
    if keep_logits.shape != expected_shape or group_ids.shape != expected_shape:
        raise ValueError("Candidate logits and equivalence IDs have incompatible shapes")
    if canonical_targets.shape != expected_shape:
        raise ValueError("Canonical Bond targets have an incompatible shape")
    if group_mask.shape != group_targets.shape or group_mask.shape[:1] != mask.shape[:1]:
        raise ValueError("Equivalence group targets have incompatible shapes")
    if group_display_targets.shape != group_mask.shape:
        raise ValueError("Display group targets have an incompatible shape")
    if group_chemical_targets.shape != group_mask.shape:
        raise ValueError("Chemical group targets have an incompatible shape")
    if representable_counts.shape != mask.shape[:1]:
        raise ValueError("Representable Bond counts have an incompatible shape")
    weights = (
        positive_weight,
        negative_weight,
        exactly_one_weight,
        representative_weight,
        cardinality_weight,
        margin_weight,
        type_weight,
    )
    if min(weights) < 0.0 or margin < 0.0:
        raise ValueError("Equivalence Bond loss weights and margin must be nonnegative")
    if cardinality_mode not in {"candidate", "group", "none"}:
        raise ValueError(f"Unknown equivalence cardinality mode: {cardinality_mode}")
    if positive_mode not in {"canonical", "group"}:
        raise ValueError(f"Unknown positive equivalence mode: {positive_mode}")
    if negative_mode not in {"max", "group"}:
        raise ValueError(f"Unknown negative equivalence mode: {negative_mode}")

    positive_scores: list[torch.Tensor] = []
    negative_scores: list[torch.Tensor] = []
    exactly_one_losses: list[torch.Tensor] = []
    representative_losses: list[torch.Tensor] = []
    cardinality_losses: list[torch.Tensor] = []
    margin_losses: list[torch.Tensor] = []
    display_losses: list[torch.Tensor] = []
    chemical_losses: list[torch.Tensor] = []

    for batch_index in range(len(mask)):
        sample_positive: list[torch.Tensor] = []
        sample_negative: list[torch.Tensor] = []
        sample_scores: list[torch.Tensor] = []
        for group_index in torch.where(group_mask[batch_index])[0].tolist():
            members = mask[batch_index] & group_ids[batch_index].eq(group_index)
            if not bool(members.any()):
                raise ValueError("Active equivalence group has no Bond candidates")
            member_logits = keep_logits[batch_index, members]
            group_score = normalized_logmeanexp(member_logits, temperature)
            sample_scores.append(group_score)
            if not bool(group_targets[batch_index, group_index]):
                negative_score = (
                    member_logits.max() if negative_mode == "max" else group_score
                )
                negative_scores.append(negative_score)
                sample_negative.append(negative_score)
                continue

            member_canonical = canonical_targets[batch_index, members]
            if int(member_canonical.sum()) != 1:
                raise ValueError("Positive equivalence group must have one canonical candidate")
            representative_target = int(torch.where(member_canonical)[0][0])
            positive_score = (
                member_logits[representative_target]
                if positive_mode == "canonical"
                else group_score
            )
            positive_scores.append(positive_score)
            sample_positive.append(positive_score)
            exactly_one_losses.append(
                F.smooth_l1_loss(
                    member_logits.sigmoid().sum(),
                    member_logits.new_tensor(1.0),
                )
            )
            representative_losses.append(
                F.cross_entropy(
                    member_logits[None],
                    torch.tensor(
                        [representative_target],
                        device=member_logits.device,
                        dtype=torch.long,
                    ),
                )
            )
            if type_weight > 0.0:
                member_indices = torch.where(members)[0]
                canonical_index = int(member_indices[representative_target])
                display_losses.append(
                    F.cross_entropy(
                        output["display_logits"][batch_index, canonical_index][None].float(),
                        group_display_targets[batch_index, group_index][None].long(),
                        weight=display_class_weights,
                    )
                )
                chemical_losses.append(
                    F.cross_entropy(
                        output["chemical_logits"][batch_index, canonical_index][None].float(),
                        group_chemical_targets[batch_index, group_index][None].long(),
                        weight=chemical_class_weights,
                    )
                )

        if cardinality_mode == "candidate":
            soft_count = keep_logits[batch_index, mask[batch_index]].sigmoid().sum()
            cardinality_losses.append(
                F.smooth_l1_loss(
                    soft_count,
                    representable_counts[batch_index].float(),
                )
            )
        elif cardinality_mode == "group" and sample_scores:
            soft_count = torch.stack(sample_scores).sigmoid().sum()
            cardinality_losses.append(
                F.smooth_l1_loss(
                    soft_count,
                    representable_counts[batch_index].float(),
                )
            )
        if sample_positive and sample_negative:
            margin_losses.append(
                F.softplus(
                    keep_logits.new_tensor(margin)
                    - torch.stack(sample_positive).min()
                    + torch.stack(sample_negative).max()
                )
            )

    zero = keep_logits.sum() * 0.0

    def mean_or_zero(values: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(values).mean() if values else zero

    positive_loss = (
        F.binary_cross_entropy_with_logits(
            torch.stack(positive_scores), torch.ones_like(torch.stack(positive_scores))
        )
        if positive_scores
        else zero
    )
    negative_loss = (
        F.binary_cross_entropy_with_logits(
            torch.stack(negative_scores), torch.zeros_like(torch.stack(negative_scores))
        )
        if negative_scores
        else zero
    )
    exactly_one_loss = mean_or_zero(exactly_one_losses)
    representative_loss = mean_or_zero(representative_losses)
    cardinality_loss = mean_or_zero(cardinality_losses)
    group_margin_loss = mean_or_zero(margin_losses)
    display_loss = mean_or_zero(display_losses)
    chemical_loss = mean_or_zero(chemical_losses)
    loss = (
        positive_weight * positive_loss
        + negative_weight * negative_loss
        + exactly_one_weight * exactly_one_loss
        + representative_weight * representative_loss
        + cardinality_weight * cardinality_loss
        + margin_weight * group_margin_loss
        + type_weight * (display_loss + chemical_loss)
    )
    return {
        "loss": loss,
        "positive_group_loss": positive_loss,
        "negative_group_loss": negative_loss,
        "exactly_one_loss": exactly_one_loss,
        "representative_loss": representative_loss,
        "cardinality_loss": cardinality_loss,
        "group_margin_loss": group_margin_loss,
        "display_loss": display_loss,
        "chemical_loss": chemical_loss,
    }


@torch.no_grad()
def equivalence_selection_metrics(
    keep_logits: torch.Tensor,
    mask: torch.Tensor,
    group_ids: torch.Tensor,
    group_mask: torch.Tensor,
    group_targets: torch.Tensor,
    *,
    threshold: float,
) -> dict[str, float | int]:
    """Measure group recall, exclusivity, and false selection before grouping."""

    if keep_logits.shape != mask.shape or group_ids.shape != mask.shape:
        raise ValueError("Equivalence selection metric inputs have incompatible shapes")
    if group_mask.shape != group_targets.shape or group_mask.shape[:1] != mask.shape[:1]:
        raise ValueError("Equivalence selection group targets have incompatible shapes")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Equivalence selection threshold must be in [0, 1]")

    selected = mask & keep_logits.float().sigmoid().ge(threshold)
    positive_groups = recalled_groups = exactly_one_groups = 0
    selected_group_true = selected_group_false = 0
    group_set_exact = strict_group_exact = 0
    duplicate_candidates = negative_candidates = 0

    for batch_index in range(len(mask)):
        expected = set(torch.where(group_mask[batch_index] & group_targets[batch_index])[0].tolist())
        selected_positive: set[int] = set()
        selected_negative: set[int] = set()
        sample_strict = True
        for group_index in torch.where(group_mask[batch_index])[0].tolist():
            members = mask[batch_index] & group_ids[batch_index].eq(group_index)
            selected_count = int((selected[batch_index] & members).sum())
            is_positive = bool(group_targets[batch_index, group_index])
            if is_positive:
                positive_groups += 1
                recalled_groups += int(selected_count > 0)
                exactly_one_groups += int(selected_count == 1)
                duplicate_candidates += max(selected_count - 1, 0)
                sample_strict &= selected_count == 1
                if selected_count:
                    selected_positive.add(group_index)
            elif selected_count:
                selected_negative.add(group_index)
                negative_candidates += selected_count
        selected_group_true += len(selected_positive)
        selected_group_false += len(selected_negative)
        group_exact = selected_positive == expected and not selected_negative
        group_set_exact += int(group_exact)
        strict_group_exact += int(group_exact and sample_strict)

    samples = max(len(mask), 1)
    group_precision = selected_group_true / max(selected_group_true + selected_group_false, 1)
    group_recall = recalled_groups / max(positive_groups, 1)
    exactly_one_rate = exactly_one_groups / max(positive_groups, 1)
    return {
        "samples": len(mask),
        "positive_groups": positive_groups,
        "group_precision": group_precision,
        "group_recall": group_recall,
        "exactly_one_rate": exactly_one_rate,
        "miss_rate": 1.0 - group_recall,
        "multi_selected_group_rate": group_recall - exactly_one_rate,
        "group_set_exact_rate": group_set_exact / samples,
        "strict_group_selection_exact_rate": strict_group_exact / samples,
        "same_group_extra_candidates_per_image": duplicate_candidates / samples,
        "selected_negative_groups_per_image": selected_group_false / samples,
        "selected_negative_candidates_per_image": negative_candidates / samples,
    }
