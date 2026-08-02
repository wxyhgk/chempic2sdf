#!/usr/bin/env python3
"""Sweep Atom/Bond decision thresholds for a frozen joint checkpoint."""

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
from chemvector.train_bond_presence_calibration import DEFAULT_SOURCE, load_components
from chemvector.train_joint_proposal_objects_full import load_validation_records
from chemvector.train_proposal_object_set import _sample_set_metrics
from chemvector.train_spatial_incidence import DEFAULT_GRAPH, forward_frozen_bond_head


DEFAULT_DOMAINS = {
    "chemvision": (
        "Data/sdf/chemvision_layers_v0_1/pilot_10k/shards/validation-000000.tar"
    ),
    "full_v0": "Data/sdf/webdataset_full_v0/shards/validation-000000.tar",
}


def parse_values(text: str) -> list[float]:
    values = [float(value) for value in text.split(",")]
    if not values:
        raise ValueError("Threshold grid cannot be empty")
    return values


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
    parser.add_argument("--atom-thresholds", default="-0.75,-0.5,-0.25,0.0,0.25")
    parser.add_argument("--bond-thresholds", default="-1.0,-0.75,-0.5,-0.25,0.0,0.25")
    parser.add_argument(
        "--output",
        default="outputs/audits/chemvector-v0972-joint-threshold-sweep/summary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Joint threshold audit requires CUDA")
    atom_thresholds = parse_values(args.atom_thresholds)
    bond_thresholds = parse_values(args.bond_thresholds)
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
        (atom_threshold, bond_threshold): defaultdict(lambda: defaultdict(int))
        for atom_threshold in atom_thresholds
        for bond_threshold in bond_thresholds
    }
    domain_audits = {}

    for domain, shard in DEFAULT_DOMAINS.items():
        records, audit = load_validation_records(
            [shard],
            per_shard=args.samples_per_domain,
            config=scene_config,
        )
        domain_audits[domain] = audit
        for start in range(0, len(records), args.batch_size):
            batch = collator(records[start : start + args.batch_size])
            pixels = batch["pixel_values"].to(device)
            targets = stack_targets(batch, device)
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
                for thresholds, domain_totals in totals.items():
                    atom_threshold, bond_threshold = thresholds
                    metrics = _sample_set_metrics(
                        output,
                        proposals,
                        targets,
                        batch_index,
                        atom_threshold=atom_threshold,
                        bond_threshold=bond_threshold,
                    )
                    bucket = domain_totals[domain]
                    bucket["samples"] += 1
                    for name in (
                        "atom_deployed_exact",
                        "bond_deployed_exact",
                        "incidence_exact",
                        "full_object_exact",
                    ):
                        bucket[name] += int(metrics[name])

    rows = []
    for (atom_threshold, bond_threshold), domain_totals in totals.items():
        domains = {}
        pooled = defaultdict(int)
        for domain, counts in domain_totals.items():
            samples = counts["samples"]
            rates = {
                name + "_rate": counts[name] / samples
                for name in (
                    "atom_deployed_exact",
                    "bond_deployed_exact",
                    "incidence_exact",
                    "full_object_exact",
                )
            }
            domains[domain] = {**dict(counts), **rates}
            for name, value in counts.items():
                pooled[name] += value
        pooled_samples = pooled["samples"]
        pooled_rates = {
            name + "_rate": pooled[name] / pooled_samples
            for name in (
                "atom_deployed_exact",
                "bond_deployed_exact",
                "incidence_exact",
                "full_object_exact",
            )
        }
        rows.append(
            {
                "atom_threshold": atom_threshold,
                "bond_threshold": bond_threshold,
                "domains": domains,
                "pooled": {**dict(pooled), **pooled_rates},
                "minimum_domain_full_object_exact_rate": min(
                    item["full_object_exact_rate"] for item in domains.values()
                ),
            }
        )
    rows.sort(
        key=lambda row: (
            row["pooled"]["full_object_exact_rate"],
            row["minimum_domain_full_object_exact_rate"],
            row["pooled"]["bond_deployed_exact_rate"],
        ),
        reverse=True,
    )
    result = {
        "checkpoint": args.checkpoint,
        "samples_per_domain": args.samples_per_domain,
        "domain_audits": domain_audits,
        "best": rows[0],
        "rows": rows,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["best"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
