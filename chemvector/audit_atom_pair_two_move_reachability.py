#!/usr/bin/env python3
"""Audit bounded second-stage edits after frozen v0.9.89 one-move inference."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.atom_pair_deletion_veto import (
    AtomPairDeletionVetoConfig,
    build_atom_pair_relation_features,
)
from chemvector.atom_pair_graph_context import (
    flatten_atom_pairs,
    load_atom_pair_graph_context_head,
)
from chemvector.atom_pair_one_move import (
    apply_atom_pair_action,
    choose_atom_pair_action,
    load_atom_pair_one_move_head,
)
from chemvector.predicted_incidence import PredictedProposalConfig
from chemvector.train_atom_bond_scene import AtomBondSceneCollator, stack_targets
from chemvector.train_atom_pair_bond_group import VALIDATION_DOMAINS
from chemvector.train_atom_pair_deletion_veto import (
    frozen_base_output,
    frozen_relation_groups,
)
from chemvector.train_atom_pair_graph_context import load_group_head
from chemvector.train_bond_presence_calibration import load_components
from chemvector.train_joint_proposal_objects_full import load_validation_records
from chemvector.train_spatial_incidence import DEFAULT_GRAPH


DEFAULT_ONE_MOVE = (
    "outputs/checkpoints/chemvector-v0989-atom-pair-one-move-final/best.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--one-move-checkpoint", default=DEFAULT_ONE_MOVE)
    parser.add_argument("--grounded-graph-checkpoint", default=DEFAULT_GRAPH)
    parser.add_argument("--samples-per-domain", type=int, default=498)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--delete-candidates", type=int, default=4)
    parser.add_argument("--add-candidates", type=int, default=16)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def residual_kind(false_positive: int, false_negative: int) -> str:
    if false_positive == 0 and false_negative == 0:
        return "exact"
    names = {
        (1, 0): "one_delete",
        (0, 1): "one_add",
        (1, 1): "one_swap",
        (2, 0): "two_delete",
        (0, 2): "two_add",
        (2, 1): "two_delete_one_add",
        (1, 2): "one_delete_two_add",
        (2, 2): "two_delete_two_add",
    }
    return names.get((false_positive, false_negative), "larger")


def error_ranks(
    logits: torch.Tensor,
    selected: torch.Tensor,
    expected: torch.Tensor,
) -> tuple[list[int], list[int]]:
    selected_indices = torch.where(selected)[0]
    unselected_indices = torch.where(~selected)[0]
    selected_order = selected_indices[
        torch.argsort(logits[selected_indices], stable=True)
    ]
    unselected_order = unselected_indices[
        torch.argsort(logits[unselected_indices], descending=True, stable=True)
    ]
    delete_rank = {
        int(index): rank for rank, index in enumerate(selected_order.tolist(), start=1)
    }
    add_rank = {
        int(index): rank for rank, index in enumerate(unselected_order.tolist(), start=1)
    }
    false_positive = torch.where(selected & ~expected)[0].tolist()
    false_negative = torch.where(~selected & expected)[0].tolist()
    return (
        [delete_rank[index] for index in false_positive],
        [add_rank[index] for index in false_negative],
    )


def bounded_reachable(
    delete_ranks: list[int],
    add_ranks: list[int],
    *,
    maximum_deletes: int,
    maximum_adds: int,
    delete_candidates: int,
    add_candidates: int,
) -> bool:
    return (
        len(delete_ranks) <= maximum_deletes
        and len(add_ranks) <= maximum_adds
        and all(rank <= delete_candidates for rank in delete_ranks)
        and all(rank <= add_candidates for rank in add_ranks)
    )


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("AtomPair two-move reachability audit requires CUDA")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")

    action_head, action_payload = load_atom_pair_one_move_head(
        args.one_move_checkpoint, device
    )
    context_head, context_payload = load_atom_pair_graph_context_head(
        action_payload["base_context_checkpoint"], device
    )
    group_head, group_payload = load_group_head(
        context_payload["group_checkpoint"], device
    )
    vision, atom, bond, set_head, source = load_components(
        group_payload["source_checkpoint"], args.grounded_graph_checkpoint, device
    )
    for module in (
        vision,
        atom,
        bond,
        set_head,
        group_head,
        context_head,
        action_head,
    ):
        module.eval().requires_grad_(False)

    proposal_config = PredictedProposalConfig(**source["proposal_config"])
    scene_config = AtomBondSceneConfig(
        atom_queries=proposal_config.maximum_atoms,
        bond_queries=proposal_config.maximum_bonds,
    )
    relation_config = AtomPairDeletionVetoConfig(**action_payload["relation_config"])
    collator = AtomBondSceneCollator(scene_config)
    minimum_margin = float(action_payload["minimum_margin"])
    domains = {}

    for domain, shard in VALIDATION_DOMAINS.items():
        rows = load_validation_records(
            [shard], per_shard=args.samples_per_domain, config=scene_config
        )[0]
        totals = Counter()
        residuals = Counter()
        action_counts = Counter()
        delete_rank_values = []
        add_rank_values = []
        for start in range(0, len(rows), args.batch_size):
            batch = collator(rows[start : start + args.batch_size])
            pixels = batch["pixel_values"].to(device)
            targets = stack_targets(batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                proposals, set_output, groups = frozen_relation_groups(
                    vision, atom, bond, set_head, pixels, targets, proposal_config
                )
                base = frozen_base_output(
                    group_head,
                    context_head,
                    context_payload,
                    groups,
                    batch_size=len(pixels),
                )
            pairs = flatten_atom_pairs(groups["atom_pairs"], device=device)
            relation = build_atom_pair_relation_features(
                pixels,
                proposals["atom_geometry"][..., :2],
                proposals["atom_mask"],
                set_output["atom_presence_logits"],
                groups["sample_indices"],
                pairs,
                base["presence_logits"],
                batch_size=len(pixels),
                config=relation_config,
            )
            edge_features = torch.cat(
                (relation, base["presence_logits"].float()[:, None]), dim=-1
            )
            for batch_index in range(len(pixels)):
                sample = groups["sample_indices"] == batch_index
                expected = groups["targets"][sample]
                sample_logits = base["presence_logits"][sample]
                baseline_selected = sample_logits >= 0.0
                target_count = int(targets["bond_mask"][batch_index].sum())
                matched_count = int(
                    proposals["bond_target_matched"][batch_index].sum()
                )
                representable = (
                    matched_count == target_count
                    and int(expected.sum()) == target_count
                )
                totals["samples"] += 1
                totals["representable"] += int(representable)
                if not representable:
                    totals["unrepresentable"] += 1
                    continue

                scores, actions = action_head.score_sample(
                    edge_features[sample],
                    relation[sample],
                    sample_logits,
                    pairs[sample],
                )
                action, _ = choose_atom_pair_action(
                    scores, actions, minimum_margin=minimum_margin
                )
                selected = apply_atom_pair_action(baseline_selected, action)
                action_counts[action.kind] += 1
                false_positive = int((selected & ~expected).sum())
                false_negative = int((~selected & expected).sum())
                kind = residual_kind(false_positive, false_negative)
                residuals[kind] += 1
                totals["exact"] += int(kind == "exact")
                delete_ranks, add_ranks = error_ranks(
                    sample_logits, selected, expected
                )
                delete_rank_values.extend(delete_ranks)
                add_rank_values.extend(add_ranks)
                totals["second_one_move_reachable"] += int(
                    bounded_reachable(
                        delete_ranks,
                        add_ranks,
                        maximum_deletes=1,
                        maximum_adds=1,
                        delete_candidates=args.delete_candidates,
                        add_candidates=args.add_candidates,
                    )
                )
                totals["bounded_two_move_top8_reachable"] += int(
                    bounded_reachable(
                        delete_ranks,
                        add_ranks,
                        maximum_deletes=2,
                        maximum_adds=2,
                        delete_candidates=args.delete_candidates,
                        add_candidates=min(args.add_candidates, 8),
                    )
                )
                totals["bounded_two_move_top16_reachable"] += int(
                    bounded_reachable(
                        delete_ranks,
                        add_ranks,
                        maximum_deletes=2,
                        maximum_adds=2,
                        delete_candidates=args.delete_candidates,
                        add_candidates=args.add_candidates,
                    )
                )
        domains[domain] = {
            **dict(totals),
            "residual_types": dict(residuals),
            "first_actions": dict(action_counts),
            "delete_rank": {
                "values": delete_rank_values,
                "top1": sum(rank <= 1 for rank in delete_rank_values),
                "top4": sum(rank <= 4 for rank in delete_rank_values),
            },
            "add_rank": {
                "values": add_rank_values,
                "top1": sum(rank <= 1 for rank in add_rank_values),
                "top4": sum(rank <= 4 for rank in add_rank_values),
                "top8": sum(rank <= 8 for rank in add_rank_values),
                "top16": sum(rank <= 16 for rank in add_rank_values),
            },
        }

    pooled = Counter()
    pooled_residuals = Counter()
    pooled_actions = Counter()
    delete_rank_values = []
    add_rank_values = []
    for values in domains.values():
        for key in (
            "samples",
            "representable",
            "unrepresentable",
            "exact",
            "second_one_move_reachable",
            "bounded_two_move_top8_reachable",
            "bounded_two_move_top16_reachable",
        ):
            pooled[key] += values.get(key, 0)
        pooled_residuals.update(values["residual_types"])
        pooled_actions.update(values["first_actions"])
        delete_rank_values.extend(values["delete_rank"]["values"])
        add_rank_values.extend(values["add_rank"]["values"])

    result = {
        "one_move_checkpoint": args.one_move_checkpoint,
        "minimum_margin": minimum_margin,
        "candidate_limits": {
            "delete": args.delete_candidates,
            "add": args.add_candidates,
        },
        "domains": domains,
        "pooled": {
            **dict(pooled),
            "residual_types": dict(pooled_residuals),
            "first_actions": dict(pooled_actions),
            "delete_rank": {
                "values": delete_rank_values,
                "top1": sum(rank <= 1 for rank in delete_rank_values),
                "top4": sum(rank <= 4 for rank in delete_rank_values),
            },
            "add_rank": {
                "values": add_rank_values,
                "top1": sum(rank <= 1 for rank in add_rank_values),
                "top4": sum(rank <= 4 for rank in add_rank_values),
                "top8": sum(rank <= 8 for rank in add_rank_values),
                "top16": sum(rank <= 16 for rank in add_rank_values),
            },
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
