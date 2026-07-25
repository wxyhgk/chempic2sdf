#!/usr/bin/env python3
"""Audit train-only chemical reachability after frozen v0.9.89 inference."""

from __future__ import annotations

import argparse
from collections import Counter
import itertools
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
from chemvector.bond_object import BOND_TYPES
from chemvector.predicted_incidence import PredictedProposalConfig
from chemvector.train_atom_bond_scene import AtomBondSceneCollator, stack_targets
from chemvector.train_atom_pair_bond_group import TRAIN_DOMAINS
from chemvector.train_atom_pair_deletion_veto import (
    frozen_base_output,
    frozen_relation_groups,
)
from chemvector.train_atom_pair_graph_context import load_group_head
from chemvector.train_bond_presence_calibration import load_components
from chemvector.train_spatial_incidence import DEFAULT_GRAPH
from chemvector.train_stable_bond_presence import valid_stream


ABSENT_STATE = len(BOND_TYPES)
DEFAULT_ONE_MOVE = (
    "outputs/checkpoints/chemvector-v0989-atom-pair-one-move-final/best.pt"
)


def typed_states(
    selected: torch.Tensor,
    chemical_logits: torch.Tensor,
) -> torch.Tensor:
    if selected.dtype != torch.bool:
        raise ValueError("Typed-state selection must be boolean")
    if len(selected) != len(chemical_logits):
        raise ValueError("Typed-state tensors differ in length")
    chemical = chemical_logits.argmax(-1)
    return torch.where(selected, chemical, chemical.new_full(chemical.shape, ABSENT_STATE))


def target_typed_states(
    expected_presence: torch.Tensor,
    chemical_targets: torch.Tensor,
) -> torch.Tensor:
    if expected_presence.dtype != torch.bool:
        raise ValueError("Typed target presence must be boolean")
    if len(expected_presence) != len(chemical_targets):
        raise ValueError("Typed target tensors differ in length")
    return torch.where(
        expected_presence,
        chemical_targets,
        chemical_targets.new_full(chemical_targets.shape, ABSENT_STATE),
    )


def chemical_confidence_margin(chemical_logits: torch.Tensor) -> torch.Tensor:
    if chemical_logits.ndim != 2 or chemical_logits.size(-1) != len(BOND_TYPES):
        raise ValueError("Chemical logits have an unsupported shape")
    top = chemical_logits.float().topk(2, dim=-1).values
    return top[:, 0] - top[:, 1]


def typed_candidate_indices(
    base_logits: torch.Tensor,
    chemical_logits: torch.Tensor,
    selected: torch.Tensor,
    *,
    delete_candidates: int,
    add_candidates: int,
    retype_candidates: int,
) -> tuple[set[int], set[int], set[int]]:
    if min(delete_candidates, add_candidates, retype_candidates) < 1:
        raise ValueError("Typed candidate limits must be positive")
    if not (len(base_logits) == len(chemical_logits) == len(selected)):
        raise ValueError("Typed candidate tensors differ in length")
    selected_indices = torch.where(selected)[0]
    unselected_indices = torch.where(~selected)[0]
    delete = selected_indices[
        torch.argsort(base_logits[selected_indices], stable=True)[:delete_candidates]
    ]
    add = unselected_indices[
        torch.argsort(
            base_logits[unselected_indices], descending=True, stable=True
        )[:add_candidates]
    ]
    margins = chemical_confidence_margin(chemical_logits)
    retype = selected_indices[
        torch.argsort(margins[selected_indices], stable=True)[:retype_candidates]
    ]
    return set(delete.tolist()), set(add.tolist()), set(retype.tolist())


def typed_edit_kinds(
    current_states: torch.Tensor,
    expected_states: torch.Tensor,
) -> list[tuple[int, str]]:
    if len(current_states) != len(expected_states):
        raise ValueError("Typed edit states differ in length")
    edits = []
    for index in torch.where(current_states != expected_states)[0].tolist():
        current = int(current_states[index])
        expected = int(expected_states[index])
        if expected == ABSENT_STATE:
            kind = "delete"
        elif current == ABSENT_STATE:
            kind = "add"
        else:
            kind = "retype"
        edits.append((index, kind))
    return edits


def bounded_typed_reachable(
    edits: list[tuple[int, str]],
    candidates: tuple[set[int], set[int], set[int]],
    *,
    maximum_edits: int,
) -> bool:
    if maximum_edits < 0:
        raise ValueError("maximum_edits must be non-negative")
    if len(edits) > maximum_edits:
        return False
    delete, add, retype = candidates
    allowed = {"delete": delete, "add": add, "retype": retype}
    return all(index in allowed[kind] for index, kind in edits)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--one-move-checkpoint", default=DEFAULT_ONE_MOVE)
    parser.add_argument("--grounded-graph-checkpoint", default=DEFAULT_GRAPH)
    parser.add_argument("--samples-per-domain", type=int, default=2_250)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--shuffle-buffer", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--delete-candidates", type=int, default=4)
    parser.add_argument("--add-candidates", type=int, default=16)
    parser.add_argument("--retype-candidates", type=int, default=16)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Chemical reachability audit requires CUDA")
    if min(args.samples_per_domain, args.batch_size) < 1:
        raise ValueError("Audit sample counts must be positive")
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

    for domain_offset, (domain, shards) in enumerate(TRAIN_DOMAINS.items()):
        stream = valid_stream(
            shards,
            config=scene_config,
            seed=args.seed + domain_offset,
            shuffle_buffer=args.shuffle_buffer,
        )
        rows = list(itertools.islice(stream, args.samples_per_domain))
        if len(rows) != args.samples_per_domain:
            raise RuntimeError(
                f"Domain {domain} yielded {len(rows)} of {args.samples_per_domain} rows"
            )
        totals = Counter()
        distances = Counter()
        edit_kinds = Counter()
        first_actions = Counter()
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
                expected_presence = groups["targets"][sample]
                type_mask = groups["type_mask"][sample]
                target_count = int(targets["bond_mask"][batch_index].sum())
                matched_count = int(
                    proposals["bond_target_matched"][batch_index].sum()
                )
                representable = (
                    matched_count == target_count
                    and int(expected_presence.sum()) == target_count
                    and int((expected_presence & type_mask).sum()) == target_count
                )
                totals["samples"] += 1
                totals["representable"] += int(representable)
                if not representable:
                    totals["unrepresentable"] += 1
                    continue

                sample_logits = base["presence_logits"][sample]
                scores, actions = action_head.score_sample(
                    edge_features[sample],
                    relation[sample],
                    sample_logits,
                    pairs[sample],
                )
                action, _ = choose_atom_pair_action(
                    scores, actions, minimum_margin=minimum_margin
                )
                first_actions[action.kind] += 1
                selected = apply_atom_pair_action(sample_logits >= 0.0, action)
                current_states = typed_states(
                    selected, base["chemical_logits"][sample]
                )
                expected_states = target_typed_states(
                    expected_presence, groups["chemical_targets"][sample]
                )
                edits = typed_edit_kinds(current_states, expected_states)
                distance = len(edits)
                distances[str(min(distance, 5)) if distance < 5 else "5+"] += 1
                edit_kinds.update(kind for _, kind in edits)
                presence_exact = bool((selected == expected_presence).all())
                typed_exact = distance == 0
                totals["presence_exact"] += int(presence_exact)
                totals["typed_exact"] += int(typed_exact)
                totals["presence_exact_chemical_wrong"] += int(
                    presence_exact and not typed_exact
                )
                totals["unrestricted_one_edit_reachable"] += int(distance <= 1)
                totals["unrestricted_two_edit_reachable"] += int(distance <= 2)
                candidates = typed_candidate_indices(
                    sample_logits,
                    base["chemical_logits"][sample],
                    selected,
                    delete_candidates=args.delete_candidates,
                    add_candidates=args.add_candidates,
                    retype_candidates=args.retype_candidates,
                )
                totals["bounded_one_edit_reachable"] += int(
                    bounded_typed_reachable(edits, candidates, maximum_edits=1)
                )
                totals["bounded_two_edit_reachable"] += int(
                    bounded_typed_reachable(edits, candidates, maximum_edits=2)
                )
                one_retype = len(edits) == 1 and edits[0][1] == "retype"
                totals["unrestricted_one_retype_reachable"] += int(
                    distance == 0 or one_retype
                )
                totals["bounded_one_retype_reachable"] += int(
                    distance == 0
                    or (
                        one_retype
                        and edits[0][0] in candidates[2]
                    )
                )
        domains[domain] = {
            **dict(totals),
            "typed_edit_distance": dict(distances),
            "typed_edit_kinds": dict(edit_kinds),
            "first_actions": dict(first_actions),
        }

    pooled = Counter()
    pooled_distances = Counter()
    pooled_edit_kinds = Counter()
    pooled_first_actions = Counter()
    for values in domains.values():
        for key, value in values.items():
            if isinstance(value, int):
                pooled[key] += value
        pooled_distances.update(values["typed_edit_distance"])
        pooled_edit_kinds.update(values["typed_edit_kinds"])
        pooled_first_actions.update(values["first_actions"])
    result = {
        "one_move_checkpoint": args.one_move_checkpoint,
        "minimum_margin": minimum_margin,
        "split_protocol": "train-only deterministic stream prefix",
        "samples_per_domain": args.samples_per_domain,
        "typed_states": [*BOND_TYPES, "ABSENT"],
        "candidate_limits": {
            "delete": args.delete_candidates,
            "add": args.add_candidates,
            "retype": args.retype_candidates,
        },
        "domains": domains,
        "pooled": {
            **dict(pooled),
            "typed_edit_distance": dict(pooled_distances),
            "typed_edit_kinds": dict(pooled_edit_kinds),
            "first_actions": dict(pooled_first_actions),
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
