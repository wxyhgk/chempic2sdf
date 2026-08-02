#!/usr/bin/env python3
"""Audit target-independent Atom/Bond proposal coverage and frozen incidence."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from itertools import islice
import json
from pathlib import Path

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.data import ChemVectorIterableDataset
from chemvector.predicted_incidence import (
    PredictedProposalConfig,
    associate_predicted_proposals,
    decode_predicted_atom_proposals,
    decode_predicted_bond_proposals,
)
from chemvector.spatial_incidence import SpatialIncidenceConfig, SpatialIncidenceHead
from chemvector.train_atom_bond_scene import AtomBondSceneCollator, stack_targets
from chemvector.train_spatial_incidence import (
    DEFAULT_ATOM,
    DEFAULT_BOND,
    DEFAULT_GRAPH,
    load_frozen_components,
)


DEFAULT_VALIDATION_SHARD = (
    "Data/sdf/chemvision_layers_v0_1/pilot_10k/shards/validation-000000.tar"
)
DEFAULT_INCIDENCE = (
    "outputs/checkpoints/chemvector-v0945-spatial-incidence-capacity100-step1000/"
    "best.pt"
)


def evaluate_records(
    vision,
    atom_head,
    bond_head,
    incidence_head,
    records: list[dict],
    collator: AtomBondSceneCollator,
    config: PredictedProposalConfig,
    device: torch.device,
    batch_size: int,
) -> dict:
    totals = {
        "samples": 0,
        "atom_targets": 0,
        "atom_matched": 0,
        "atom_images_covered": 0,
        "bond_targets": 0,
        "bond_matched": 0,
        "bond_images_covered": 0,
        "incidence_eligible": 0,
        "incidence_images_covered": 0,
        "incidence_endpoints": 0,
        "incidence_endpoint_correct": 0,
        "incidence_bonds": 0,
        "incidence_bond_correct": 0,
        "incidence_images_exact": 0,
        "atom_proposals": 0,
        "bond_proposals": 0,
    }
    for start in range(0, len(records), batch_size):
        batch = collator(records[start : start + batch_size])
        pixels = batch["pixel_values"].to(device)
        targets = stack_targets(batch, device)
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16
        ):
            features = vision.forward_stroke_resolution(pixels)
            atom_output = atom_head(features)
            bond_output = bond_head(features)
            atom_proposals = decode_predicted_atom_proposals(atom_output, config)
            bond_proposals = decode_predicted_bond_proposals(bond_output, config)
            proposals = associate_predicted_proposals(
                atom_proposals, bond_proposals, targets, config
            )
            incidence_output = incidence_head(
                proposals["atom_features"],
                proposals["atom_geometry"],
                proposals["atom_mask"],
                proposals["bond_features"],
                proposals["bond_endpoints"],
                proposals["bond_types"],
                proposals["bond_mask"],
            )
        for batch_index in range(len(pixels)):
            atom_targets = int(targets["atom_mask"][batch_index].sum())
            bond_targets = int(targets["bond_mask"][batch_index].sum())
            atom_matched = int(proposals["atom_target_matched"][batch_index].sum())
            bond_matched = int(proposals["bond_target_matched"][batch_index].sum())
            eligible = int(
                proposals["incidence_target_eligible"][batch_index].sum()
            )
            supervised = proposals["bond_supervision_mask"][batch_index]
            predicted = incidence_output["incidence_logits"][
                batch_index, supervised
            ].argmax(-1)
            expected = proposals["target_atom_refs"][batch_index, supervised]
            correct = predicted == expected
            totals["samples"] += 1
            totals["atom_targets"] += atom_targets
            totals["atom_matched"] += atom_matched
            totals["atom_images_covered"] += int(atom_matched == atom_targets)
            totals["bond_targets"] += bond_targets
            totals["bond_matched"] += bond_matched
            totals["bond_images_covered"] += int(bond_matched == bond_targets)
            totals["incidence_eligible"] += eligible
            totals["incidence_images_covered"] += int(eligible == bond_targets)
            totals["incidence_endpoints"] += correct.numel()
            totals["incidence_endpoint_correct"] += int(correct.sum())
            totals["incidence_bonds"] += len(correct)
            totals["incidence_bond_correct"] += int(correct.all(-1).sum())
            totals["incidence_images_exact"] += int(
                eligible == bond_targets and bool(correct.all())
            )
            totals["atom_proposals"] += int(
                proposals["atom_mask"][batch_index].sum()
            )
            totals["bond_proposals"] += int(
                proposals["bond_mask"][batch_index].sum()
            )
    samples = max(totals["samples"], 1)
    return {
        **totals,
        "atom_recall": totals["atom_matched"] / max(totals["atom_targets"], 1),
        "atom_image_coverage": totals["atom_images_covered"] / samples,
        "bond_recall": totals["bond_matched"] / max(totals["bond_targets"], 1),
        "bond_image_coverage": totals["bond_images_covered"] / samples,
        "incidence_eligible_recall": totals["incidence_eligible"]
        / max(totals["bond_targets"], 1),
        "incidence_image_coverage": totals["incidence_images_covered"] / samples,
        "eligible_endpoint_accuracy": totals["incidence_endpoint_correct"]
        / max(totals["incidence_endpoints"], 1),
        "eligible_bond_accuracy": totals["incidence_bond_correct"]
        / max(totals["incidence_bonds"], 1),
        "end_to_end_incidence_image_exact": totals["incidence_images_exact"] / samples,
        "mean_atom_proposals": totals["atom_proposals"] / samples,
        "mean_bond_proposals": totals["bond_proposals"] / samples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", default=DEFAULT_VALIDATION_SHARD)
    parser.add_argument("--expected-split", default="validation")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grounded-graph-checkpoint", default=DEFAULT_GRAPH)
    parser.add_argument("--atom-checkpoint", default=DEFAULT_ATOM)
    parser.add_argument("--bond-checkpoint", default=DEFAULT_BOND)
    parser.add_argument("--incidence-checkpoint", default=DEFAULT_INCIDENCE)
    parser.add_argument("--maximum-atoms", type=int, default=128)
    parser.add_argument("--maximum-bonds", type=int, default=128)
    parser.add_argument("--atom-threshold", type=float, default=0.01)
    parser.add_argument("--bond-threshold", type=float, default=0.01)
    parser.add_argument("--atom-match-radius-px", type=float, default=6.0)
    parser.add_argument("--bond-match-radius-px", type=float, default=12.0)
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Predicted proposal audit requires CUDA")
    device = torch.device("cuda")
    vision, atom_head, bond_head, sources = load_frozen_components(
        Path(args.grounded_graph_checkpoint),
        Path(args.atom_checkpoint),
        Path(args.bond_checkpoint),
        device,
    )
    incidence_payload = torch.load(
        args.incidence_checkpoint, map_location="cpu", weights_only=False
    )
    incidence_head = SpatialIncidenceHead(
        SpatialIncidenceConfig(**incidence_payload["config"])
    ).to(device)
    incidence_head.load_state_dict(incidence_payload["head"], strict=True)
    incidence_head.eval().requires_grad_(False)
    records = list(
        islice(
            iter(ChemVectorIterableDataset([args.shard], sequence_version="0.4")),
            args.samples,
        )
    )
    if len(records) != args.samples:
        raise ValueError("Audit shard contains fewer samples than requested")
    if any(
        sample.get("meta", {}).get(
            "source_split", sample.get("meta", {}).get("split")
        )
        != args.expected_split
        for sample in records
    ):
        raise ValueError("Audit shard contains an unexpected source split")
    config = PredictedProposalConfig(
        maximum_atoms=args.maximum_atoms,
        maximum_bonds=args.maximum_bonds,
        atom_threshold=args.atom_threshold,
        bond_threshold=args.bond_threshold,
        atom_match_radius_px=args.atom_match_radius_px,
        bond_match_radius_px=args.bond_match_radius_px,
    )
    metrics = evaluate_records(
        vision,
        atom_head,
        bond_head,
        incidence_head,
        records,
        AtomBondSceneCollator(AtomBondSceneConfig()),
        config,
        device,
        args.batch_size,
    )
    report = {
        "schema_version": "predicted-incidence-audit-0.1",
        "protocol": "target-independent-topk-proposals-gt-matched-for-metrics",
        "shard": args.shard,
        "expected_split": args.expected_split,
        "proposal_config": asdict(config),
        "sources": sources,
        "incidence_checkpoint": args.incidence_checkpoint,
        "metrics": metrics,
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
