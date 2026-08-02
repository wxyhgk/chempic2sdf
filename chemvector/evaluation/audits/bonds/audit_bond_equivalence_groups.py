#!/usr/bin/env python3
"""Audit target-independent endpoint grouping for duplicate Bond proposals."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.audit_bond_presence_failures import (
    DOMAINS,
    target_bond_endpoints,
    unordered_max_endpoint_distance,
)
from chemvector.predicted_incidence import (
    PredictedProposalConfig,
    associate_predicted_proposals,
    decode_predicted_atom_proposals,
    decode_predicted_bond_proposals,
)
from chemvector.train_atom_bond_scene import AtomBondSceneCollator, stack_targets
from chemvector.train_bond_presence_calibration import load_components
from chemvector.train_joint_proposal_objects_full import load_validation_records
from chemvector.train_multidomain_set_head import VALIDATION_DOMAINS
from chemvector.train_spatial_incidence import DEFAULT_GRAPH, forward_frozen_bond_head


def parse_values(text: str) -> list[float]:
    return [float(value) for value in text.split(",")]


def connected_groups(endpoints: torch.Tensor, radius_px: float) -> list[list[int]]:
    count = len(endpoints)
    if not count:
        return []
    distances = unordered_max_endpoint_distance(endpoints.float(), endpoints.float())
    adjacency = distances <= radius_px
    seen = set()
    groups = []
    for start in range(count):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        group = []
        while stack:
            index = stack.pop()
            group.append(index)
            for neighbor in torch.where(adjacency[index])[0].tolist():
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        groups.append(group)
    return groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=(
            "outputs/checkpoints/chemvector-v0972-bond-presence-calibration-10k/"
            "best.pt"
        ),
    )
    parser.add_argument("--grounded-graph-checkpoint", default=DEFAULT_GRAPH)
    parser.add_argument("--samples-per-domain", type=int, default=498)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bond-threshold", type=float, default=-0.5)
    parser.add_argument("--radii", default="4,6,8,10,12")
    parser.add_argument("--include-molgrapher", action="store_true")
    parser.add_argument(
        "--output",
        default="outputs/audits/chemvector-v0972-bond-equivalence-groups/summary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Bond equivalence audit requires CUDA")
    radii = parse_values(args.radii)
    device = torch.device("cuda")
    vision, atom, bond, set_head, source = load_components(
        args.checkpoint, args.grounded_graph_checkpoint, device
    )
    proposal_config = PredictedProposalConfig(**source["proposal_config"])
    scene_config = AtomBondSceneConfig(
        atom_queries=proposal_config.maximum_atoms,
        bond_queries=proposal_config.maximum_bonds,
    )
    collator = AtomBondSceneCollator(scene_config)
    totals = {
        radius: defaultdict(lambda: defaultdict(int)) for radius in radii
    }
    semantic_totals = defaultdict(lambda: defaultdict(int))

    domains = VALIDATION_DOMAINS if args.include_molgrapher else DOMAINS
    for domain, shard in domains.items():
        records, _ = load_validation_records(
            [shard],
            per_shard=args.samples_per_domain,
            config=scene_config,
        )
        for start in range(0, len(records), args.batch_size):
            batch = collator(records[start : start + args.batch_size])
            pixels = batch["pixel_values"].to(device)
            targets = stack_targets(batch, device)
            expected_endpoints = target_bond_endpoints(targets)
            with torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                features = vision.forward_stroke_resolution(pixels)
                atom_output = atom(features)
                bond_output = forward_frozen_bond_head(vision, bond, features)
                proposals = associate_predicted_proposals(
                    decode_predicted_atom_proposals(atom_output, proposal_config),
                    decode_predicted_bond_proposals(bond_output, proposal_config),
                    targets,
                    proposal_config,
                )
                output = set_head(
                    proposals["atom_features"],
                    proposals["atom_geometry"],
                    proposals["atom_scores"],
                    proposals["atom_mask"],
                    proposals["bond_features"],
                    proposals["bond_endpoints"],
                    proposals["bond_types"],
                    proposals["bond_scores"],
                    proposals["bond_mask"],
                )
            for batch_index in range(len(pixels)):
                mask = proposals["bond_mask"][batch_index]
                labels = proposals["bond_presence_targets"][batch_index, mask]
                logits = output["bond_presence_logits"][batch_index, mask]
                endpoints = proposals["bond_endpoints"][batch_index, mask].float()
                target_count = int(targets["bond_mask"][batch_index].sum())
                expected = expected_endpoints[batch_index, :target_count].float()
                distances = unordered_max_endpoint_distance(endpoints, expected)
                nearest_distance, nearest_target = distances.min(dim=1)
                nearest_target = torch.where(
                    nearest_distance <= proposal_config.bond_match_radius_px,
                    nearest_target,
                    torch.full_like(nearest_target, -1),
                )
                matched_targets = int(
                    proposals["bond_target_matched"][batch_index].sum()
                )

                for radius in radii:
                    groups = connected_groups(endpoints, radius)
                    group_exact = True
                    collision = False
                    selected_groups = 0
                    positive_groups = 0
                    suppressed_selected = 0
                    for group in groups:
                        indices = torch.tensor(group, device=device)
                        group_positive = bool(labels[indices].any())
                        group_selected_members = logits[indices] >= args.bond_threshold
                        group_selected = bool(group_selected_members.any())
                        group_exact = group_exact and group_positive == group_selected
                        positive_groups += int(group_positive)
                        selected_groups += int(group_selected)
                        suppressed_selected += max(
                            int(group_selected_members.sum()) - int(group_selected), 0
                        )
                        target_ids = {
                            int(value)
                            for value in nearest_target[indices].tolist()
                            if int(value) >= 0
                        }
                        if len(target_ids) > 1:
                            collision = True
                    deployed_exact = (
                        group_exact
                        and not collision
                        and matched_targets == target_count
                        and positive_groups == target_count
                    )
                    bucket = totals[radius][domain]
                    bucket["samples"] += 1
                    bucket["group_candidate_exact"] += int(group_exact)
                    bucket["group_deployed_exact"] += int(deployed_exact)
                    bucket["collision_images"] += int(collision)
                    bucket["groups"] += len(groups)
                    bucket["raw_candidates"] += len(endpoints)
                    bucket["selected_groups"] += selected_groups
                    bucket["positive_groups"] += positive_groups
                    bucket["suppressed_selected_duplicates"] += suppressed_selected

                predicted_refs = output["incidence_logits"][batch_index, mask].argmax(-1)
                semantic_map: dict[tuple[int, int], list[int]] = {}
                for proposal_index, refs in enumerate(predicted_refs.tolist()):
                    key = tuple(sorted((int(refs[0]), int(refs[1]))))
                    semantic_map.setdefault(key, []).append(proposal_index)
                semantic_groups = list(semantic_map.values())
                group_exact = True
                collision = False
                selected_groups = 0
                positive_groups = 0
                suppressed_selected = 0
                for group in semantic_groups:
                    indices = torch.tensor(group, device=device)
                    group_positive = bool(labels[indices].any())
                    selected_members = logits[indices] >= args.bond_threshold
                    group_selected = bool(selected_members.any())
                    group_exact = group_exact and group_positive == group_selected
                    positive_groups += int(group_positive)
                    selected_groups += int(group_selected)
                    suppressed_selected += max(
                        int(selected_members.sum()) - int(group_selected), 0
                    )
                    target_ids = {
                        int(value)
                        for value in nearest_target[indices].tolist()
                        if int(value) >= 0
                    }
                    if len(target_ids) > 1:
                        collision = True
                deployed_exact = (
                    group_exact
                    and not collision
                    and matched_targets == target_count
                    and positive_groups == target_count
                )
                bucket = semantic_totals[domain]
                bucket["samples"] += 1
                bucket["group_candidate_exact"] += int(group_exact)
                bucket["group_deployed_exact"] += int(deployed_exact)
                bucket["collision_images"] += int(collision)
                bucket["groups"] += len(semantic_groups)
                bucket["raw_candidates"] += len(endpoints)
                bucket["selected_groups"] += selected_groups
                bucket["positive_groups"] += positive_groups
                bucket["suppressed_selected_duplicates"] += suppressed_selected

    rows = []
    for radius, domain_counts in totals.items():
        domains = {}
        for domain, counts in domain_counts.items():
            samples = counts["samples"]
            domains[domain] = {
                **dict(counts),
                "group_candidate_exact_rate": counts["group_candidate_exact"]
                / samples,
                "group_deployed_exact_rate": counts["group_deployed_exact"]
                / samples,
                "collision_image_rate": counts["collision_images"] / samples,
                "candidate_reduction_rate": 1.0
                - counts["groups"] / max(counts["raw_candidates"], 1),
            }
        rows.append(
            {
                "radius_px": radius,
                "domains": domains,
                "minimum_domain_group_deployed_exact_rate": min(
                    item["group_deployed_exact_rate"] for item in domains.values()
                ),
                "pooled_group_deployed_exact_rate": sum(
                    item["group_deployed_exact"] for item in domains.values()
                )
                / sum(item["samples"] for item in domains.values()),
                "pooled_collision_image_rate": sum(
                    item["collision_images"] for item in domains.values()
                )
                / sum(item["samples"] for item in domains.values()),
            }
        )
    rows.sort(
        key=lambda row: (
            row["minimum_domain_group_deployed_exact_rate"],
            row["pooled_group_deployed_exact_rate"],
            -row["pooled_collision_image_rate"],
        ),
        reverse=True,
    )
    semantic_domains = {}
    for domain, counts in semantic_totals.items():
        samples = counts["samples"]
        semantic_domains[domain] = {
            **dict(counts),
            "group_candidate_exact_rate": counts["group_candidate_exact"] / samples,
            "group_deployed_exact_rate": counts["group_deployed_exact"] / samples,
            "collision_image_rate": counts["collision_images"] / samples,
            "candidate_reduction_rate": 1.0
            - counts["groups"] / max(counts["raw_candidates"], 1),
        }
    semantic = {
        "mode": "incidence_argmax_atom_pair",
        "domains": semantic_domains,
        "minimum_domain_group_deployed_exact_rate": min(
            item["group_deployed_exact_rate"] for item in semantic_domains.values()
        ),
        "pooled_group_deployed_exact_rate": sum(
            item["group_deployed_exact"] for item in semantic_domains.values()
        )
        / sum(item["samples"] for item in semantic_domains.values()),
        "pooled_collision_image_rate": sum(
            item["collision_images"] for item in semantic_domains.values()
        )
        / sum(item["samples"] for item in semantic_domains.values()),
    }
    result = {
        "checkpoint": args.checkpoint,
        "semantic_incidence_grouping": semantic,
        "best_geometry": rows[0],
        "geometry_rows": rows,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
