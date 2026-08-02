#!/usr/bin/env python3
"""Audit edit reachability of frozen v0.9.87 AtomPair group predictions."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.atom_pair_graph_context import load_atom_pair_graph_context_head
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


DEFAULT_BASE_CONTEXT = (
    "outputs/checkpoints/"
    "chemvector-v0987-atom-pair-graph-context-positive-margin-calibrated/best.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-context-checkpoint", default=DEFAULT_BASE_CONTEXT)
    parser.add_argument("--grounded-graph-checkpoint", default=DEFAULT_GRAPH)
    parser.add_argument("--samples-per-domain", type=int, default=498)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _edit_bucket(false_positive: int, false_negative: int) -> str:
    distance = false_positive + false_negative
    if distance >= 5:
        return "5+"
    return str(distance)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("AtomPair reachability audit requires CUDA")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    context_head, context_payload = load_atom_pair_graph_context_head(
        args.base_context_checkpoint, device
    )
    group_head, group_payload = load_group_head(context_payload["group_checkpoint"], device)
    vision, atom, bond, set_head, source = load_components(
        group_payload["source_checkpoint"], args.grounded_graph_checkpoint, device
    )
    for module in (vision, atom, bond, set_head, group_head, context_head):
        module.eval().requires_grad_(False)
    proposal_config = PredictedProposalConfig(**source["proposal_config"])
    scene_config = AtomBondSceneConfig(
        atom_queries=proposal_config.maximum_atoms,
        bond_queries=proposal_config.maximum_bonds,
    )
    collator = AtomBondSceneCollator(scene_config)
    domains = {}
    for domain, shard in VALIDATION_DOMAINS.items():
        rows = load_validation_records(
            [shard], per_shard=args.samples_per_domain, config=scene_config
        )[0]
        totals = Counter()
        edit_buckets = Counter()
        move_types = Counter()
        add_ranks = []
        delete_ranks = []
        for start in range(0, len(rows), args.batch_size):
            batch = collator(rows[start : start + args.batch_size])
            pixels = batch["pixel_values"].to(device)
            targets = stack_targets(batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                proposals, _, groups = frozen_relation_groups(
                    vision, atom, bond, set_head, pixels, targets, proposal_config
                )
                base = frozen_base_output(
                    group_head,
                    context_head,
                    context_payload,
                    groups,
                    batch_size=len(pixels),
                )
            selected_all = base["presence_logits"] >= 0.0
            chemical_correct = (
                base["chemical_logits"].argmax(-1) == groups["chemical_targets"]
            )
            for batch_index in range(len(pixels)):
                sample = groups["sample_indices"] == batch_index
                expected = groups["targets"][sample]
                selected = selected_all[sample]
                false_positive = int((selected & ~expected).sum())
                false_negative = int((~selected & expected).sum())
                target_count = int(targets["bond_mask"][batch_index].sum())
                matched_count = int(
                    proposals["bond_target_matched"][batch_index].sum()
                )
                positive_count = int(expected.sum())
                representable = (
                    matched_count == target_count and positive_count == target_count
                )
                type_mask = groups["type_mask"][sample]
                chemical_representable = (
                    representable
                    and int(type_mask.sum()) == target_count
                    and bool(chemical_correct[sample][type_mask].all())
                )
                totals["samples"] += 1
                totals["representable"] += int(representable)
                totals["chemical_representable"] += int(chemical_representable)
                totals["base_exact"] += int(
                    representable and false_positive == 0 and false_negative == 0
                )
                totals["base_count_exact"] += int(
                    int(selected.sum()) == positive_count
                )
                if not representable:
                    totals["unrepresentable"] += 1
                    continue
                edit_buckets[_edit_bucket(false_positive, false_negative)] += 1
                move = (false_positive, false_negative)
                if move == (1, 0):
                    move_types["one_delete"] += 1
                    false_index = torch.where(selected & ~expected)[0][0]
                    selected_logits = base["presence_logits"][sample][selected]
                    false_logit = base["presence_logits"][sample][false_index]
                    delete_ranks.append(
                        1 + int((selected_logits < false_logit).sum())
                    )
                elif move == (0, 1):
                    move_types["one_add"] += 1
                    missed_index = torch.where(~selected & expected)[0][0]
                    unselected_logits = base["presence_logits"][sample][~selected]
                    missed_logit = base["presence_logits"][sample][missed_index]
                    add_ranks.append(
                        1 + int((unselected_logits > missed_logit).sum())
                    )
                elif move == (1, 1):
                    move_types["one_swap"] += 1
                    false_index = torch.where(selected & ~expected)[0][0]
                    missed_index = torch.where(~selected & expected)[0][0]
                    sample_logits = base["presence_logits"][sample]
                    selected_logits = sample_logits[selected]
                    unselected_logits = sample_logits[~selected]
                    delete_ranks.append(
                        1 + int((selected_logits < sample_logits[false_index]).sum())
                    )
                    add_ranks.append(
                        1 + int((unselected_logits > sample_logits[missed_index]).sum())
                    )
                elif false_positive + false_negative:
                    move_types["multi_edit"] += 1
        domains[domain] = {
            **dict(totals),
            "edit_distance": dict(sorted(edit_buckets.items())),
            "move_types": dict(move_types),
            "add_rank": {
                "values": add_ranks,
                "top1": sum(rank <= 1 for rank in add_ranks),
                "top4": sum(rank <= 4 for rank in add_ranks),
                "top8": sum(rank <= 8 for rank in add_ranks),
                "top16": sum(rank <= 16 for rank in add_ranks),
            },
            "delete_rank": {
                "values": delete_ranks,
                "top1": sum(rank <= 1 for rank in delete_ranks),
                "top4": sum(rank <= 4 for rank in delete_ranks),
                "top8": sum(rank <= 8 for rank in delete_ranks),
                "top16": sum(rank <= 16 for rank in delete_ranks),
            },
            "representable_rate": totals["representable"] / max(totals["samples"], 1),
            "chemical_representable_rate": totals["chemical_representable"]
            / max(totals["samples"], 1),
        }
    pooled = Counter()
    pooled_edits = Counter()
    pooled_moves = Counter()
    pooled_add_ranks = []
    pooled_delete_ranks = []
    for values in domains.values():
        for key in (
            "samples",
            "representable",
            "chemical_representable",
            "base_exact",
            "base_count_exact",
            "unrepresentable",
        ):
            pooled[key] += values.get(key, 0)
        pooled_edits.update(values["edit_distance"])
        pooled_moves.update(values["move_types"])
        pooled_add_ranks.extend(values["add_rank"]["values"])
        pooled_delete_ranks.extend(values["delete_rank"]["values"])
    result = {
        "base_context_checkpoint": args.base_context_checkpoint,
        "domains": domains,
        "pooled": {
            **dict(pooled),
            "edit_distance": dict(sorted(pooled_edits.items())),
            "move_types": dict(pooled_moves),
            "add_rank": {
                "values": pooled_add_ranks,
                "top1": sum(rank <= 1 for rank in pooled_add_ranks),
                "top4": sum(rank <= 4 for rank in pooled_add_ranks),
                "top8": sum(rank <= 8 for rank in pooled_add_ranks),
                "top16": sum(rank <= 16 for rank in pooled_add_ranks),
            },
            "delete_rank": {
                "values": pooled_delete_ranks,
                "top1": sum(rank <= 1 for rank in pooled_delete_ranks),
                "top4": sum(rank <= 4 for rank in pooled_delete_ranks),
                "top8": sum(rank <= 8 for rank in pooled_delete_ranks),
                "top16": sum(rank <= 16 for rank in pooled_delete_ranks),
            },
            "representable_rate": pooled["representable"] / max(pooled["samples"], 1),
            "chemical_representable_rate": pooled["chemical_representable"]
            / max(pooled["samples"], 1),
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
