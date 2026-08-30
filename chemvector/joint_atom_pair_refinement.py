"""Refine frozen base AtomPair set outputs into ranked decode attempts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from PIL import Image
import torch

from chemvector.atom_pair_bond_group import (
    build_atom_pair_groups,
    collapse_atom_pair_group_outputs,
)
from chemvector.atom_pair_graph_context import (
    flatten_atom_pairs,
    refine_atom_pair_group_output,
)
from chemvector.atom_pair_deletion_veto import (
    AtomPairDeletionVetoConfig,
    build_atom_pair_relation_features,
)
from chemvector.atom_pair_one_move import (
    apply_atom_pair_action,
    choose_atom_pair_action,
)
from chemvector.atom_pair_chemical_retype import (
    apply_retype_to_chemical_logits,
    build_retype_features,
)
from chemvector.atom_pair_chemical_retype_verifier import choose_verified_retype_action
from chemvector.line_segment_group_rescue import (
    LineSegmentGroupRescueConfig,
    rescue_disconnected_atom_pair_groups,
)
from chemvector.line_segment_rescue import detect_atom_pair_line_candidates
from chemvector.line_atom_pair_head import build_line_atom_pair_features
from chemvector.line_atom_pair_inference import append_line_atom_pair_proposals


@dataclass(frozen=True, slots=True)
class JointAtomPairExperts:
    """Frozen AtomPair expert heads with their training payloads."""

    group_head: Any = None
    group_payload: dict | None = None
    line_atom_pair_head: Any = None
    line_atom_pair_payload: dict | None = None
    graph_context_head: Any = None
    graph_context_payload: dict | None = None
    one_move_head: Any = None
    one_move_payload: dict | None = None
    one_move_relation_config: AtomPairDeletionVetoConfig | None = None
    retype_verifier_head: Any = None
    retype_verifier_payload: dict | None = None
    retype_head: Any = None
    retype_payload: dict | None = None


def build_atom_pair_attempts(
    name: str,
    model_output: dict,
    proposals: dict,
    bond_output: dict,
    pixels: torch.Tensor,
    image: Image.Image,
    experts: JointAtomPairExperts,
    *,
    atom_threshold: float,
    group_threshold: float,
    line_segment_group_rescue: bool,
    line_rescue_minimum_support: int,
) -> list[tuple]:
    """Build the decode attempts for one frozen base set output."""

    if experts.group_head is None:
        return []
    attempts: list[tuple] = []
    groups = build_atom_pair_groups(model_output, proposals)
    group_output = experts.group_head(groups["features"])
    graph_context_diagnostics = None
    if experts.graph_context_head is not None:
        group_output, graph_context_diagnostics = refine_atom_pair_group_output(
            experts.graph_context_head,
            group_output,
            groups,
            batch_size=1,
            residual_scale=float(
                experts.graph_context_payload.get("residual_scale", 1.0)
            ),
            base_positive_protection_threshold=(
                experts.graph_context_payload.get(
                    "base_positive_protection_threshold"
                )
            ),
        )
    one_move_diagnostics = None
    chemical_retype_diagnostics = None
    if experts.one_move_head is not None:
        pairs = flatten_atom_pairs(groups["atom_pairs"], device=pixels.device)
        relation = build_atom_pair_relation_features(
            pixels,
            proposals["atom_geometry"][..., :2],
            proposals["atom_mask"],
            model_output["atom_presence_logits"],
            groups["sample_indices"],
            pairs,
            group_output["presence_logits"],
            batch_size=1,
            config=experts.one_move_relation_config,
        )
        edge_features = torch.cat(
            (relation, group_output["presence_logits"].float()[:, None]),
            dim=-1,
        )
        action_scores, actions = experts.one_move_head.score_sample(
            edge_features,
            relation,
            group_output["presence_logits"],
            pairs,
        )
        action, action_margin = choose_atom_pair_action(
            action_scores,
            actions,
            minimum_margin=float(experts.one_move_payload["minimum_margin"]),
        )
        base_presence_logits = group_output["presence_logits"]
        selected_before = base_presence_logits >= 0.0
        selected_after = apply_atom_pair_action(selected_before, action)
        refined_logits = base_presence_logits.clone()
        changed = selected_after != selected_before
        refined_logits[changed & selected_after] = 1e-4
        refined_logits[changed & ~selected_after] = -1e-4
        group_output = {**group_output, "presence_logits": refined_logits}
        one_move_diagnostics = {
            "action": action.kind,
            "delete_index": action.delete_index,
            "add_index": action.add_index,
            "delete_atom_pair": (
                pairs[action.delete_index].tolist()
                if action.delete_index >= 0
                else None
            ),
            "add_atom_pair": (
                pairs[action.add_index].tolist()
                if action.add_index >= 0
                else None
            ),
            "score_margin": action_margin,
            "minimum_margin": float(experts.one_move_payload["minimum_margin"]),
            "candidate_actions": len(actions),
        }
        if experts.retype_verifier_head is not None:
            retype_features = build_retype_features(
                relation,
                base_presence_logits,
                group_output["chemical_logits"],
                selected_after,
            )
            (
                retype_gate,
                retype_prior_scores,
                retype_actions,
            ) = experts.retype_head.score_sample(
                retype_features,
                group_output["chemical_logits"],
                selected_after,
            )
            verifier_scores = experts.retype_verifier_head.score_sample(
                retype_features,
                group_output["chemical_logits"],
                selected_after,
                retype_gate,
                retype_prior_scores,
                retype_actions,
            )
            retype_action = choose_verified_retype_action(
                verifier_scores,
                retype_actions,
                minimum_margin=float(
                    experts.retype_verifier_payload["gate_threshold"]
                ),
            )
            group_output = {
                **group_output,
                "chemical_logits": apply_retype_to_chemical_logits(
                    group_output["chemical_logits"], retype_action
                ),
            }
            chemical_retype_diagnostics = {
                "action": "KEEP" if retype_action is None else "RETYPE",
                "group_index": (
                    retype_action.group_index
                    if retype_action is not None
                    else None
                ),
                "atom_pair": (
                    pairs[retype_action.group_index].tolist()
                    if retype_action is not None
                    else None
                ),
                "target_state": (
                    retype_action.target_state
                    if retype_action is not None
                    else None
                ),
                "score": (
                    float(verifier_scores.max()) if len(verifier_scores) else None
                ),
                "minimum_margin": float(
                    experts.retype_verifier_payload["gate_threshold"]
                ),
                "candidate_actions": len(retype_actions),
            }
    plain_proposals, plain_output, plain_grouping = collapse_atom_pair_group_outputs(
        proposals, model_output, group_output, groups
    )
    if graph_context_diagnostics is not None:
        plain_grouping[0]["graph_context"] = graph_context_diagnostics
    if one_move_diagnostics is not None:
        plain_grouping[0]["one_move"] = one_move_diagnostics
    if chemical_retype_diagnostics is not None:
        plain_grouping[0]["chemical_retype"] = chemical_retype_diagnostics
    sample_line_candidates = None
    if line_segment_group_rescue or experts.line_atom_pair_head is not None:
        predicted_pixels = (
            proposals["atom_geometry"][0, :, :2].float().cpu().numpy() * 768.0
        )
        selected_atoms = (
            proposals["atom_mask"][0]
            & (model_output["atom_presence_logits"][0] >= atom_threshold)
        ).cpu().numpy()
        sample_line_candidates = tuple(
            candidate
            for candidate in detect_atom_pair_line_candidates(
                image,
                predicted_pixels,
                atom_mask=selected_atoms,
            )
            if len(candidate.support_segment_indices)
            >= line_rescue_minimum_support
        )
    grouped_proposals = plain_proposals
    grouped_output = plain_output
    grouping = plain_grouping
    if line_segment_group_rescue:
        rescued_group_output, line_rescue = rescue_disconnected_atom_pair_groups(
            group_output,
            groups,
            atom_presence_logits=model_output["atom_presence_logits"],
            atom_mask=proposals["atom_mask"],
            line_candidates=[sample_line_candidates or ()],
            atom_threshold=atom_threshold,
            group_threshold=group_threshold,
            config=LineSegmentGroupRescueConfig(
                minimum_support_segments=line_rescue_minimum_support
            ),
        )
        grouped_proposals, grouped_output, grouping = collapse_atom_pair_group_outputs(
            proposals, model_output, rescued_group_output, groups
        )
        grouping[0]["line_segment_rescue"] = line_rescue[0]
        attempts.append(
            (
                f"line_atom_pair_{name}",
                grouped_proposals,
                grouped_output,
                grouping[0],
                group_threshold,
            )
        )
    if experts.line_atom_pair_head is not None:
        existing_pairs = set(groups["atom_pairs"][0])
        line_only_candidates = tuple(
            candidate
            for candidate in (sample_line_candidates or ())
            if candidate.atom_pair not in existing_pairs
        )
        line_features, line_sample_indices = build_line_atom_pair_features(
            bond_output["object_features"].float(),
            proposals["atom_features"].float(),
            proposals["atom_geometry"][..., :2].float(),
            [line_only_candidates],
            config=experts.line_atom_pair_head.config,
        )
        line_output = experts.line_atom_pair_head(line_features)
        line_proposals, line_model_output, line_diagnostics = (
            append_line_atom_pair_proposals(
                grouped_proposals,
                grouped_output,
                line_output,
                [line_only_candidates],
                line_sample_indices,
                line_threshold=float(experts.line_atom_pair_payload["threshold"]),
            )
        )
        line_grouping = deepcopy(grouping[0])
        line_grouping["line_atom_pair"] = line_diagnostics[0]
        attempts.insert(
            len(attempts) - int(line_segment_group_rescue),
            (
                f"first_class_line_atom_pair_{name}",
                line_proposals,
                line_model_output,
                line_grouping,
                group_threshold,
            ),
        )
    attempts.append(
        (
            f"atom_pair_{name}",
            plain_proposals,
            plain_output,
            plain_grouping[0],
            group_threshold,
        )
    )
    return attempts