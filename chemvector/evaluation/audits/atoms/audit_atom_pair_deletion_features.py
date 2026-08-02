#!/usr/bin/env python3
"""Audit whether relation features separate v0.9.87 false-positive edges."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.atom_pair_deletion_veto import (
    AtomPairDeletionVetoConfig,
    RELATION_FEATURE_DIM,
    build_atom_pair_relation_features,
)
from chemvector.atom_pair_graph_context import (
    flatten_atom_pairs,
    load_atom_pair_graph_context_head,
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


FEATURE_NAMES = (
    "normalized_length",
    "direction_cos2",
    "direction_sin2",
    "ink_mean",
    "ink_min",
    "ink_std",
    "ink_coverage",
    "explained_mean",
    "explained_max",
    "exclusive_ink",
    "exclusive_ratio",
    "intermediate_max",
    "intermediate_sum",
    "common_neighbor_max",
    "common_neighbor_sum",
    "common_collinear",
    "endpoint_degree_mean",
    "endpoint_degree_difference",
)


def distribution(values: torch.Tensor) -> dict:
    if not len(values):
        return {"count": 0}
    values = values.float().cpu()
    return {
        "count": len(values),
        "mean": float(values.mean()),
        "q10": float(values.quantile(0.1)),
        "q50": float(values.quantile(0.5)),
        "q90": float(values.quantile(0.9)),
    }


def rule_metrics(rule: torch.Tensor, false_edges: torch.Tensor) -> dict:
    selected = int(rule.sum())
    true_selected = int((rule & false_edges).sum())
    false_count = int(false_edges.sum())
    return {
        "selected": selected,
        "false_edges_selected": true_selected,
        "precision": true_selected / max(selected, 1),
        "false_edge_recall": true_selected / max(false_count, 1),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-context-checkpoint",
        default=(
            "outputs/checkpoints/"
            "chemvector-v0987-atom-pair-graph-context-positive-margin-calibrated/"
            "best.pt"
        ),
    )
    parser.add_argument("--grounded-graph-checkpoint", default=DEFAULT_GRAPH)
    parser.add_argument("--samples-per-domain", type=int, default=498)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Deletion feature audit requires CUDA")
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
    relation_config = AtomPairDeletionVetoConfig(
        input_dim=group_head.config.input_dim,
        maximum_atoms=proposal_config.maximum_atoms,
        image_size=proposal_config.image_size,
    )
    collator = AtomBondSceneCollator(scene_config)
    all_features = []
    all_targets = []
    all_domains = []
    exact = {
        domain: {
            "samples": 0,
            "baseline_presence": 0,
            "baseline_chemical": 0,
            "rules": {},
        }
        for domain in VALIDATION_DOMAINS
    }
    for domain, shard in VALIDATION_DOMAINS.items():
        rows = load_validation_records(
            [shard], per_shard=args.samples_per_domain, config=scene_config
        )[0]
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
            selected = base["presence_logits"] >= 0.0
            intermediate = relation[:, 11]
            common = relation[:, 13]
            collinear = relation[:, 15]
            ink = relation[:, 3]
            exclusive = relation[:, 9]
            batch_rules = {
                "intermediate_0_5": intermediate >= 0.5,
                "common_collinear_0_4": collinear >= 0.4,
                "spanning_two_hop": (intermediate >= 0.5) & (collinear >= 0.4),
                "unsupported_shared_neighbor": (common >= 0.5) & (ink < 0.1),
                "low_exclusive_shared_neighbor": (common >= 0.5) & (exclusive < 0.05),
            }
            chemical_correct = (
                base["chemical_logits"].argmax(-1) == groups["chemical_targets"]
            )
            for batch_index in range(len(pixels)):
                sample = groups["sample_indices"] == batch_index
                expected = groups["targets"][sample]
                target_count = int(targets["bond_mask"][batch_index].sum())
                matched_count = int(
                    proposals["bond_target_matched"][batch_index].sum()
                )
                positive_count = int(expected.sum())
                type_mask = groups["type_mask"][sample]
                chemistry_ok = (
                    int(type_mask.sum()) == target_count
                    and bool(chemical_correct[sample][type_mask].all())
                )
                baseline_ok = (
                    bool((selected[sample] == expected).all())
                    and matched_count == target_count
                    and positive_count == target_count
                )
                exact[domain]["samples"] += 1
                exact[domain]["baseline_presence"] += int(baseline_ok)
                exact[domain]["baseline_chemical"] += int(
                    baseline_ok and chemistry_ok
                )
                for name, rule in batch_rules.items():
                    row = exact[domain]["rules"].setdefault(
                        name, {"presence": 0, "chemical": 0, "deletions": 0}
                    )
                    revised = selected[sample] & ~rule[sample]
                    revised_ok = (
                        bool((revised == expected).all())
                        and matched_count == target_count
                        and positive_count == target_count
                    )
                    row["presence"] += int(revised_ok)
                    row["chemical"] += int(revised_ok and chemistry_ok)
                    row["deletions"] += int((selected[sample] & rule[sample]).sum())
            all_features.append(relation[selected].cpu())
            all_targets.append(groups["targets"][selected].cpu())
            all_domains.extend([domain] * int(selected.sum()))
    features = torch.cat(all_features)
    targets = torch.cat(all_targets).bool()
    false_edges = ~targets
    if features.size(-1) != RELATION_FEATURE_DIM:
        raise RuntimeError("Relation feature audit dimension mismatch")
    feature_summary = {}
    for index, name in enumerate(FEATURE_NAMES):
        feature_summary[name] = {
            "true_edges": distribution(features[targets, index]),
            "false_edges": distribution(features[false_edges, index]),
        }
    intermediate = features[:, 11]
    common = features[:, 13]
    collinear = features[:, 15]
    ink = features[:, 3]
    exclusive = features[:, 9]
    rules = {
        "intermediate_0_5": intermediate >= 0.5,
        "common_collinear_0_4": collinear >= 0.4,
        "spanning_two_hop": (intermediate >= 0.5) & (collinear >= 0.4),
        "unsupported_shared_neighbor": (common >= 0.5) & (ink < 0.1),
        "low_exclusive_shared_neighbor": (common >= 0.5) & (exclusive < 0.05),
    }
    result = {
        "base_context_checkpoint": args.base_context_checkpoint,
        "samples_per_domain": args.samples_per_domain,
        "base_positive_edges": len(targets),
        "true_edges": int(targets.sum()),
        "false_edges": int(false_edges.sum()),
        "features": feature_summary,
        "rules": {
            name: rule_metrics(rule, false_edges) for name, rule in rules.items()
        },
        "whole_graph_exact": exact,
        "domains": {
            domain: {
                "base_positive_edges": sum(item == domain for item in all_domains),
                "false_edges": int(
                    false_edges[
                        torch.tensor([item == domain for item in all_domains])
                    ].sum()
                ),
            }
            for domain in VALIDATION_DOMAINS
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
