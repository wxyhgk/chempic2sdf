#!/usr/bin/env python3
"""Evaluate saved AtomPair group type checkpoints on the fixed two domains."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.atom_pair_bond_group import AtomPairBondGroupConfig, AtomPairBondGroupHead
from chemvector.predicted_incidence import PredictedProposalConfig
from chemvector.train_atom_bond_scene import AtomBondSceneCollator
from chemvector.train_atom_pair_bond_group import VALIDATION_DOMAINS
from chemvector.train_atom_pair_bond_group_types import validate
from chemvector.train_bond_presence_calibration import load_components
from chemvector.train_joint_proposal_objects_full import load_validation_records
from chemvector.train_spatial_incidence import DEFAULT_GRAPH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--grounded-graph-checkpoint", default=DEFAULT_GRAPH)
    parser.add_argument("--validation-samples-per-domain", type=int, default=498)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--output",
        default="outputs/evaluations/chemvector-v0983-group-types-internal/summary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("AtomPair group type evaluation requires CUDA")
    device = torch.device("cuda")
    first = torch.load(args.checkpoints[0], map_location="cpu", weights_only=False)
    vision, atom, bond, set_head, source = load_components(
        first["source_checkpoint"], args.grounded_graph_checkpoint, device
    )
    proposal_config = PredictedProposalConfig(**source["proposal_config"])
    scene_config = AtomBondSceneConfig(
        atom_queries=proposal_config.maximum_atoms,
        bond_queries=proposal_config.maximum_bonds,
    )
    collator = AtomBondSceneCollator(scene_config)
    records = {}
    for name, shard in VALIDATION_DOMAINS.items():
        rows, _ = load_validation_records(
            [shard], per_shard=args.validation_samples_per_domain, config=scene_config
        )
        records[name] = rows
    results = []
    for checkpoint in args.checkpoints:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload["source_checkpoint"] != first["source_checkpoint"]:
            raise ValueError("Evaluation checkpoints use different proposal sources")
        head = AtomPairBondGroupHead(
            AtomPairBondGroupConfig(**payload["config"])
        ).to(device)
        head.load_state_dict(payload["head"], strict=True)
        head.eval().requires_grad_(False)
        metrics = validate(
            vision, atom, bond, set_head, head, records, collator,
            proposal_config, device, args.batch_size,
        )
        results.append(
            {
                "checkpoint": checkpoint,
                "samples_seen": payload["samples_seen"],
                "metrics": metrics,
            }
        )
    result = {"results": results}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    compact = {
        row["samples_seen"]: {
            domain: {
                "chemical_accuracy": values["chemical_accuracy"],
                "chemical_deployed_exact_rate": values[
                    "chemical_deployed_exact_rate"
                ],
            }
            for domain, values in row["metrics"]["domains"].items()
        }
        for row in results
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
