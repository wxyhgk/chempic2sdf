#!/usr/bin/env python3
"""Decompose Bond presence failures into ranking and duplicate-proposal errors."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.predicted_incidence import (
    PredictedProposalConfig,
    associate_predicted_proposals,
    decode_predicted_atom_proposals,
    decode_predicted_bond_proposals,
)
from chemvector.train_atom_bond_scene import AtomBondSceneCollator, stack_targets
from chemvector.train_bond_presence_calibration import load_components
from chemvector.train_joint_proposal_objects_full import load_validation_records
from chemvector.train_spatial_incidence import DEFAULT_GRAPH, forward_frozen_bond_head


DOMAINS = {
    "chemvision": (
        "Data/sdf/chemvision_layers_v0_1/pilot_10k/shards/validation-000000.tar"
    ),
    "full_v0": "Data/sdf/webdataset_full_v0/shards/validation-000000.tar",
}


def target_bond_endpoints(targets: dict[str, torch.Tensor]) -> torch.Tensor:
    batch = torch.arange(
        targets["bond_atom_refs"].size(0),
        device=targets["bond_atom_refs"].device,
    )[:, None, None]
    return targets["atom_anchors"][batch, targets["bond_atom_refs"]]


def unordered_max_endpoint_distance(
    predicted: torch.Tensor, expected: torch.Tensor
) -> torch.Tensor:
    forward = torch.linalg.vector_norm(
        (predicted[:, None] - expected[None]) * 768.0,
        dim=-1,
    ).amax(-1)
    reverse = torch.linalg.vector_norm(
        (predicted[:, None] - expected[None].flip(1)) * 768.0,
        dim=-1,
    ).amax(-1)
    return torch.minimum(forward, reverse)


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
    parser.add_argument(
        "--output",
        default="outputs/audits/chemvector-v0972-bond-presence-failures/summary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Bond presence audit requires CUDA")
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
    domain_totals = {}

    for domain, shard in DOMAINS.items():
        records, _ = load_validation_records(
            [shard],
            per_shard=args.samples_per_domain,
            config=scene_config,
        )
        totals = defaultdict(float)
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
                scores = proposals["bond_scores"][batch_index, mask]
                endpoints = proposals["bond_endpoints"][batch_index, mask].float()
                target_count = int(targets["bond_mask"][batch_index].sum())
                matched_count = int(labels.sum())
                selected = logits >= args.bond_threshold
                current_exact = bool((selected == labels).all())
                deployed_current = current_exact and matched_count == target_count

                top_count = torch.zeros_like(labels)
                if matched_count:
                    top_count[logits.topk(matched_count).indices] = True
                top_count_exact = bool((top_count == labels).all())
                raw_top_count = torch.zeros_like(labels)
                if matched_count:
                    raw_top_count[scores.topk(matched_count).indices] = True
                raw_top_count_exact = bool((raw_top_count == labels).all())
                positives = logits[labels]
                negatives = logits[~labels]
                margin = (
                    float(positives.min() - negatives.max())
                    if len(positives) and len(negatives)
                    else 0.0
                )

                false_positive = selected & ~labels
                false_negative = ~selected & labels
                expected = expected_endpoints[batch_index, :target_count].float()
                distances = (
                    unordered_max_endpoint_distance(endpoints, expected)
                    if len(endpoints) and target_count
                    else endpoints.new_empty((len(endpoints), 0))
                )
                nearest = (
                    distances.amin(dim=1)
                    if distances.numel()
                    else endpoints.new_full((len(endpoints),), float("inf"))
                )
                fp_nearest = nearest[false_positive]

                totals["samples"] += 1
                totals["targets"] += target_count
                totals["matched_targets"] += matched_count
                totals["candidate_exact"] += int(current_exact)
                totals["deployed_exact"] += int(deployed_current)
                totals["oracle_count_candidate_exact"] += int(top_count_exact)
                totals["oracle_count_deployed_exact"] += int(
                    top_count_exact and matched_count == target_count
                )
                totals["raw_score_oracle_count_exact"] += int(raw_top_count_exact)
                totals["rank_separable"] += int(margin > 0.0)
                totals["rank_margin_sum"] += margin
                totals["false_positive"] += int(false_positive.sum())
                totals["false_negative"] += int(false_negative.sum())
                totals["fp_duplicate_12px"] += int((fp_nearest <= 12.0).sum())
                totals["fp_near_20px"] += int((fp_nearest <= 20.0).sum())

        samples = int(totals["samples"])
        false_positive = max(int(totals["false_positive"]), 1)
        domain_totals[domain] = {
            **{name: int(value) if name != "rank_margin_sum" else value for name, value in totals.items()},
            "proposal_recall": totals["matched_targets"] / max(totals["targets"], 1),
            "candidate_exact_rate": totals["candidate_exact"] / samples,
            "deployed_exact_rate": totals["deployed_exact"] / samples,
            "oracle_count_candidate_exact_rate": totals[
                "oracle_count_candidate_exact"
            ]
            / samples,
            "oracle_count_deployed_exact_rate": totals[
                "oracle_count_deployed_exact"
            ]
            / samples,
            "raw_score_oracle_count_exact_rate": totals[
                "raw_score_oracle_count_exact"
            ]
            / samples,
            "rank_separable_rate": totals["rank_separable"] / samples,
            "mean_rank_margin": totals["rank_margin_sum"] / samples,
            "false_positive_duplicate_12px_rate": totals["fp_duplicate_12px"]
            / false_positive,
            "false_positive_near_20px_rate": totals["fp_near_20px"]
            / false_positive,
        }

    result = {
        "checkpoint": args.checkpoint,
        "bond_threshold": args.bond_threshold,
        "domains": domain_totals,
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
