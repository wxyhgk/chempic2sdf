"""Pipeline adapters for proposal-anchored bond-group pointers.

The frozen flat proposal pool and GT association are intentionally delegated to
the v0.9.5 membership pipeline.  This module only adapts those targets to the
pointer contract and decodes a partition with explicit root/member candidate
masks.  No coordinates are generated or replaced here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from chemvector.bond_group_proposal_membership import (
    FIXED_MEMBER_COUNTS,
    HASHED_WEDGE,
    PHYSICAL_GROUP_NAMES,
)
from chemvector.bond_group_proposal_membership_pipeline import (
    PROPOSAL_DESCRIPTOR_DIM,
    prepare_proposal_membership_pipeline,
)


POINTER_TARGET_FIELDS = (
    "group_types",
    "group_membership",
    "group_mask",
    "member_counts",
    "group_weights",
    "proposal_keep",
    "source_group_ids",
    "known_counts",
)


@dataclass(frozen=True, slots=True)
class PointerCountResolution:
    """Raw and feasible group-count predictions for one batch."""

    raw_counts: torch.Tensor
    feasible_counts: torch.Tensor
    infeasible: torch.Tensor


def _validate_target_shapes(
    targets: dict[str, torch.Tensor],
    proposal_mask: torch.Tensor,
) -> tuple[int, int, int]:
    missing = [name for name in POINTER_TARGET_FIELDS if name not in targets]
    if missing:
        raise ValueError(f"Pointer target adapter is missing fields: {missing}")
    if proposal_mask.ndim != 2:
        raise ValueError("Pointer proposal mask must have shape [batch, proposals]")
    batch, proposals = proposal_mask.shape
    group_types = targets["group_types"]
    if group_types.ndim != 2 or group_types.size(0) != batch:
        raise ValueError("Pointer group types must have shape [batch, groups]")
    groups = group_types.size(1)
    expected = {
        "group_membership": (batch, groups, proposals),
        "group_mask": (batch, groups),
        "member_counts": (batch, groups),
        "group_weights": (batch, groups),
        "proposal_keep": (batch, proposals),
        "source_group_ids": (batch, groups),
        "known_counts": (batch,),
    }
    for name, shape in expected.items():
        if targets[name].shape != shape:
            raise ValueError(
                f"Pointer target {name} has shape {tuple(targets[name].shape)}, "
                f"expected {shape}"
            )
    return batch, groups, proposals


def adapt_proposal_pointer_targets(
    targets: dict[str, torch.Tensor],
    proposal_mask: torch.Tensor,
    *,
    max_group_count: int,
) -> dict[str, torch.Tensor]:
    """Validate and retain the exact v0.9.5 oracle association targets."""

    if max_group_count < 1:
        raise ValueError("Pointer max_group_count must be positive")
    _, groups, _ = _validate_target_shapes(targets, proposal_mask)
    if groups != max_group_count:
        raise ValueError(
            f"Pointer target slots {groups} differ from max_group_count="
            f"{max_group_count}"
        )
    active_groups = targets["group_mask"].bool()
    known_counts = targets["known_counts"].long()
    if not torch.equal(known_counts, active_groups.sum(dim=-1).long()):
        raise ValueError("Pointer known counts differ from active target groups")
    if bool((known_counts > max_group_count).any()):
        raise ValueError("Pointer known count exceeds group capacity")

    valid = proposal_mask.bool()
    usable = targets["proposal_keep"].bool()
    if bool((usable & ~valid).any()):
        raise ValueError("Pointer usable proposal mask references padding")
    membership = targets["group_membership"].bool() & active_groups.unsqueeze(-1)
    if bool((membership & ~valid.unsqueeze(1)).any()):
        raise ValueError("Pointer group membership references padding")
    membership_union = membership.any(dim=1)
    if not torch.equal(membership_union, usable):
        raise ValueError(
            "Pointer proposal_keep must equal the active membership union"
        )
    overlap = membership.sum(dim=1)
    if bool((overlap > 1).any()):
        raise ValueError("One pointer proposal belongs to multiple target groups")
    if bool((active_groups & (targets["group_weights"].float() <= 0.0)).any()):
        raise ValueError("Active pointer groups require positive weights")

    return {name: targets[name] for name in POINTER_TARGET_FIELDS}


def prepare_proposal_pointer_pipeline(
    batch: dict[str, Any],
    flat_output: dict[str, torch.Tensor],
    query_reference_points: torch.Tensor,
    *,
    max_group_count: int,
) -> dict[str, Any]:
    """Reuse the v0.9.5 pool and expose pointer targets plus usable masks."""

    prepared = prepare_proposal_membership_pipeline(
        batch,
        flat_output,
        query_reference_points,
        group_slots=max_group_count,
    )
    proposal_mask = prepared["model_inputs"]["proposal_mask"].bool()
    targets = adapt_proposal_pointer_targets(
        prepared["targets"],
        proposal_mask,
        max_group_count=max_group_count,
    )
    usable = proposal_mask & targets["proposal_keep"].bool()
    query_indices = batch.get("proposal_query_index")
    if not isinstance(query_indices, torch.Tensor) or query_indices.shape != (
        *proposal_mask.shape,
    ):
        raise ValueError("Pointer pipeline requires proposal_query_index")
    active_queries = query_indices[proposal_mask]
    if len(active_queries) and bool((active_queries < 0).any()):
        raise ValueError("Active pointer proposals require flat query indices")
    return {
        "model_inputs": prepared["model_inputs"],
        "targets": targets,
        "known_counts": targets["known_counts"],
        "oracle_coverage": prepared["oracle_coverage"],
        "oracle_usable_mask": usable,
        "proposal_query_indices": query_indices.detach(),
    }


def _validate_candidate_masks(
    proposal_mask: torch.Tensor,
    root_candidate_mask: torch.Tensor,
    member_candidate_mask: torch.Tensor,
) -> tuple[int, int]:
    if proposal_mask.ndim != 2:
        raise ValueError("Pointer proposal mask must be two-dimensional")
    shape = proposal_mask.shape
    if root_candidate_mask.shape != shape or member_candidate_mask.shape != shape:
        raise ValueError("Pointer candidate masks must match proposal_mask")
    valid = proposal_mask.bool()
    roots = root_candidate_mask.bool()
    members = member_candidate_mask.bool()
    if bool((members & ~valid).any()):
        raise ValueError("Pointer member candidates reference padding")
    if bool((roots & ~members).any()):
        raise ValueError("Pointer root candidates must be member candidates")
    return shape


def resolve_pointer_group_counts(
    output: dict[str, torch.Tensor],
    root_candidate_mask: torch.Tensor,
    member_candidate_mask: torch.Tensor,
    *,
    mode: str,
    known_counts: torch.Tensor | None = None,
) -> PointerCountResolution:
    """Resolve counts while masking learned predictions outside feasibility."""

    if mode not in {"known_count", "learned_count"}:
        raise ValueError("Pointer decode mode must be known_count or learned_count")
    if root_candidate_mask.shape != member_candidate_mask.shape:
        raise ValueError("Pointer root/member masks differ in shape")
    batch = root_candidate_mask.size(0)
    root_capacity = root_candidate_mask.bool().sum(dim=-1).long()
    member_count = member_candidate_mask.bool().sum(dim=-1).long()
    if bool(((member_count > 0) & (root_capacity == 0)).any()):
        raise ValueError("Pointer members require at least one root candidate")

    if mode == "known_count":
        if known_counts is None or known_counts.shape != (batch,):
            raise ValueError("Known pointer decoding requires one count per image")
        counts = known_counts.long()
        if bool(((counts < 0) | (counts > root_capacity)).any()):
            raise ValueError("Known pointer count exceeds root candidates")
        if bool(((member_count > 0) & (counts == 0)).any()):
            raise ValueError("Nonempty pointer members require a root")
        zeros = torch.zeros_like(counts, dtype=torch.bool)
        return PointerCountResolution(counts, counts, zeros)

    logits = output.get("group_count_logits")
    if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
        raise ValueError("Learned pointer decoding requires group_count_logits")
    if logits.size(0) != batch:
        raise ValueError("Pointer count logits and candidate masks differ")
    raw = logits.float().argmax(dim=-1)
    feasible = torch.zeros(batch, device=logits.device, dtype=torch.long)
    for batch_index in range(batch):
        maximum = min(int(root_capacity[batch_index]), logits.size(-1) - 1)
        minimum = 1 if int(member_count[batch_index]) else 0
        if maximum < minimum:
            raise ValueError("Pointer count head cannot represent a feasible count")
        feasible[batch_index] = (
            logits[batch_index, minimum : maximum + 1].float().argmax()
            + minimum
        )
    return PointerCountResolution(raw, feasible, raw != feasible)


@torch.no_grad()
def decode_proposal_pointer_partition(
    output: dict[str, torch.Tensor],
    proposal_mask: torch.Tensor,
    *,
    root_candidate_mask: torch.Tensor,
    member_candidate_mask: torch.Tensor,
    mode: str,
    known_counts: torch.Tensor | None = None,
) -> tuple[list[list[dict[str, Any]]], PointerCountResolution]:
    """Decode roots and members with independent, explicit candidate masks."""

    batch, proposals = _validate_candidate_masks(
        proposal_mask,
        root_candidate_mask,
        member_candidate_mask,
    )
    keep_logits = output.get("keep_logits")
    pointer_logits = output.get("pointer_logits")
    type_logits = output.get("group_type_logits")
    count_logits = output.get("member_count_logits")
    if not isinstance(keep_logits, torch.Tensor) or keep_logits.shape != (
        batch,
        proposals,
    ):
        raise ValueError("Pointer keep logits have an invalid shape")
    if not isinstance(pointer_logits, torch.Tensor) or pointer_logits.shape != (
        batch,
        proposals,
        proposals,
    ):
        raise ValueError("Pointer ownership logits have an invalid shape")
    if not isinstance(type_logits, torch.Tensor) or type_logits.shape != (
        batch,
        proposals,
        len(PHYSICAL_GROUP_NAMES),
    ):
        raise ValueError("Pointer group type logits have an invalid shape")
    if not isinstance(count_logits, torch.Tensor) or count_logits.shape[:2] != (
        batch,
        proposals,
    ):
        raise ValueError("Pointer member count logits have an invalid shape")

    resolution = resolve_pointer_group_counts(
        output,
        root_candidate_mask,
        member_candidate_mask,
        mode=mode,
        known_counts=known_counts,
    )
    keep_scores = keep_logits.float().sigmoid()
    group_types = type_logits.float()[..., 1:].argmax(dim=-1) + 1
    raw_member_counts = count_logits.float().argmax(dim=-1)
    if count_logits.size(-1) < 3:
        raise ValueError("Pointer member count head must represent count two")
    hashed_counts = count_logits.float()[..., 2:].argmax(dim=-1) + 2

    decoded: list[list[dict[str, Any]]] = []
    for batch_index in range(batch):
        root_candidates = torch.where(root_candidate_mask[batch_index].bool())[0]
        members = torch.where(member_candidate_mask[batch_index].bool())[0]
        count = int(resolution.feasible_counts[batch_index])
        if count == 0:
            if len(members):
                raise ValueError("Pointer members cannot be decoded without roots")
            decoded.append([])
            continue
        ranked = root_candidates[
            torch.argsort(
                keep_scores[batch_index, root_candidates],
                descending=True,
                stable=True,
            )
        ]
        roots = ranked[:count]
        ownership = pointer_logits[batch_index][members][:, roots]
        assigned = ownership.argmax(dim=-1)
        root_offsets = {
            int(root): offset for offset, root in enumerate(roots.tolist())
        }
        member_offsets = {
            int(member): offset for offset, member in enumerate(members.tolist())
        }
        for root, root_offset in root_offsets.items():
            assigned[member_offsets[root]] = root_offset

        rows = []
        used: list[int] = []
        for root_offset, root in enumerate(roots.tolist()):
            proposal_indices = members[assigned == root_offset].tolist()
            used.extend(proposal_indices)
            group_type = int(group_types[batch_index, root])
            raw_count = int(raw_member_counts[batch_index, root])
            member_count = (
                int(hashed_counts[batch_index, root])
                if group_type == HASHED_WEDGE
                else FIXED_MEMBER_COUNTS[group_type]
            )
            support_count = len(proposal_indices)
            valid_membership = (
                2 <= support_count <= member_count
                if group_type == HASHED_WEDGE
                else support_count == member_count
            )
            rows.append(
                {
                    "root_index": root,
                    "query_index": root,
                    "score": float(keep_scores[batch_index, root]),
                    "type": group_type,
                    "kind": PHYSICAL_GROUP_NAMES[group_type],
                    "raw_member_count": raw_count,
                    "member_count": member_count,
                    "support_count": support_count,
                    "proposal_indices": proposal_indices,
                    "valid_membership": valid_membership,
                }
            )
        if sorted(used) != sorted(members.tolist()) or len(used) != len(set(used)):
            raise RuntimeError("Pointer decoder did not partition members exactly once")
        decoded.append(rows)
    return decoded, resolution


def latent_root_assignment_map(
    sample_keys: list[str],
    targets: dict[str, torch.Tensor],
    selected_root_indices: torch.Tensor,
    proposal_query_indices: torch.Tensor,
) -> dict[str, int]:
    """Map each stable sample/group identity to its frozen flat root query."""

    group_mask = targets["group_mask"].bool()
    source_group_ids = targets["source_group_ids"].long()
    if selected_root_indices.shape != group_mask.shape:
        raise ValueError("Selected pointer roots and target groups differ")
    if proposal_query_indices.ndim != 2 or proposal_query_indices.size(0) != len(
        sample_keys
    ):
        raise ValueError("Pointer query indices and sample keys differ")
    result: dict[str, int] = {}
    for batch_index, sample_key in enumerate(sample_keys):
        for slot in torch.where(group_mask[batch_index])[0].tolist():
            source_group = int(source_group_ids[batch_index, slot])
            root = int(selected_root_indices[batch_index, slot])
            if source_group < 0 or not 0 <= root < proposal_query_indices.size(1):
                raise ValueError("Pointer latent root assignment is invalid")
            query_index = int(proposal_query_indices[batch_index, root])
            if query_index < 0:
                raise ValueError("Pointer latent root references a padded query")
            key = f"{sample_key}::{source_group}"
            if key in result:
                raise ValueError(f"Duplicate latent root identity: {key}")
            result[key] = query_index
    return result


def compare_latent_root_assignments(
    previous: dict[str, int] | None,
    current: dict[str, int],
) -> dict[str, int | float | None]:
    """Measure root identity retention between consecutive validations."""

    if previous is None:
        return {"compared": 0, "stable": 0, "rate": None}
    common = sorted(set(previous) & set(current))
    stable = sum(previous[key] == current[key] for key in common)
    return {
        "compared": len(common),
        "stable": stable,
        "rate": stable / len(common) if common else None,
    }


def pointer_root_selection_diagnostics(
    output: dict[str, torch.Tensor],
    decoded: list[list[dict[str, Any]]],
    targets: dict[str, torch.Tensor],
    member_candidate_mask: torch.Tensor,
) -> dict[str, int]:
    """Audit keep-selected roots against GT groups and raw pointer affinity."""

    pointer_logits = output.get("pointer_logits")
    group_membership = targets["group_membership"].bool()
    group_mask = targets["group_mask"].bool()
    if not isinstance(pointer_logits, torch.Tensor) or pointer_logits.ndim != 3:
        raise ValueError("Root diagnostics require batched pointer logits")
    batch, proposals, _ = pointer_logits.shape
    if len(decoded) != batch or member_candidate_mask.shape != (batch, proposals):
        raise ValueError("Root diagnostics batch shapes differ")
    if group_membership.shape != (*group_mask.shape, proposals):
        raise ValueError("Root diagnostics target membership has an invalid shape")

    totals = {
        "samples": 0,
        "expected_groups": 0,
        "selected_roots": 0,
        "distinct_target_groups": 0,
        "root_collisions": 0,
        "distinct_group_coverage_exact_samples": 0,
        "affinity_self_preferred_roots": 0,
        "affinity_majority_agrees_roots": 0,
        "assigned_members": 0,
        "assigned_members_same_as_root_group": 0,
    }
    for batch_index, rows in enumerate(decoded):
        active_slots = torch.where(group_mask[batch_index])[0]
        expected_groups = len(active_slots)
        member_mask = member_candidate_mask[batch_index].bool()
        active_membership = group_membership[batch_index, active_slots]
        ownership_count = active_membership[:, member_mask].sum(dim=0)
        if len(ownership_count) and not bool((ownership_count == 1).all()):
            raise ValueError("Every diagnostic member must belong to one target group")
        proposal_to_slot = torch.full(
            (proposals,),
            -1,
            device=pointer_logits.device,
            dtype=torch.long,
        )
        for local_slot, target_slot in enumerate(active_slots.tolist()):
            proposal_to_slot[active_membership[local_slot]] = int(target_slot)

        roots = [int(row["root_index"]) for row in rows]
        root_target_slots = [int(proposal_to_slot[root]) for root in roots]
        if any(slot < 0 for slot in root_target_slots):
            raise ValueError("A selected pointer root has no target group")
        distinct = len(set(root_target_slots))
        totals["samples"] += 1
        totals["expected_groups"] += expected_groups
        totals["selected_roots"] += len(roots)
        totals["distinct_target_groups"] += distinct
        totals["root_collisions"] += len(roots) - distinct
        totals["distinct_group_coverage_exact_samples"] += int(
            distinct == expected_groups
        )

        if roots:
            root_tensor = torch.tensor(
                roots,
                device=pointer_logits.device,
                dtype=torch.long,
            )
            root_affinity = pointer_logits[batch_index][root_tensor][:, root_tensor]
            preferred = root_affinity.argmax(dim=-1)
            totals["affinity_self_preferred_roots"] += int(
                (preferred == torch.arange(len(roots), device=preferred.device)).sum()
            )

        for row, root_target_slot in zip(rows, root_target_slots):
            indices = [int(value) for value in row["proposal_indices"]]
            assigned_slots = proposal_to_slot[indices]
            if bool((assigned_slots < 0).any()):
                raise ValueError("A decoded pointer member has no target group")
            counts = torch.bincount(
                assigned_slots,
                minlength=group_mask.size(1),
            )
            dominant = int(counts.argmax()) if len(indices) else -1
            totals["affinity_majority_agrees_roots"] += int(
                dominant == root_target_slot
            )
            totals["assigned_members"] += len(indices)
            totals["assigned_members_same_as_root_group"] += int(
                (assigned_slots == root_target_slot).sum()
            )
    return totals


__all__ = [
    "POINTER_TARGET_FIELDS",
    "PROPOSAL_DESCRIPTOR_DIM",
    "PointerCountResolution",
    "adapt_proposal_pointer_targets",
    "compare_latent_root_assignments",
    "decode_proposal_pointer_partition",
    "latent_root_assignment_map",
    "pointer_root_selection_diagnostics",
    "prepare_proposal_pointer_pipeline",
    "resolve_pointer_group_counts",
]
