#!/usr/bin/env python3
"""Evaluate a proposal-delta set adapter on fixed validation domains."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.dense_bond_object import BondProposalResidualAdapter
from chemvector.evaluate_style_sensitivity import load_style_records
from chemvector.predicted_incidence import PredictedProposalConfig
from chemvector.train_atom_bond_scene import AtomBondSceneCollator
from chemvector.train_bond_presence_calibration import load_components
from chemvector.train_proposal_delta_set_adapter import (
    domain_delta_validation,
    load_proposal_delta_set_adapter,
)
from chemvector.train_spatial_incidence import DEFAULT_GRAPH


DEFAULT_CHECKPOINT = (
    "outputs/checkpoints/chemvector-v0997-proposal-delta-set-adapter-pilot250/"
    "best.pt"
)
DEFAULT_SHARDS = {
    "chemvision": (
        "Data/sdf/chemvision_layers_v0_1/pilot_10k/shards/"
        "internal_test-000000.tar"
    ),
    "full_v0": "Data/sdf/webdataset_full_v0/shards/internal_test-000000.tar",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--grounded-graph-checkpoint", default=DEFAULT_GRAPH)
    parser.add_argument("--split", choices=("validation", "internal_test"), default="internal_test")
    parser.add_argument("--samples-per-domain", type=int, default=498)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Proposal-delta adapter evaluation requires CUDA")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    adapter, adapter_payload = load_proposal_delta_set_adapter(args.checkpoint, device)
    vision, atom, candidate_bond, set_head, source = load_components(
        adapter_payload["source_checkpoint"], args.grounded_graph_checkpoint, device
    )
    if not isinstance(candidate_bond, BondProposalResidualAdapter):
        raise ValueError("Source checkpoint does not contain a Bond proposal residual")
    original_bond = candidate_bond.base_detector
    for module in (vision, atom, candidate_bond, original_bond, set_head, adapter):
        module.eval().requires_grad_(False)
    proposal_config = PredictedProposalConfig(**source["proposal_config"])
    scene_config = AtomBondSceneConfig(
        atom_queries=proposal_config.maximum_atoms,
        bond_queries=proposal_config.maximum_bonds,
    )
    records = {
        domain: load_style_records(
            shard,
            split=args.split,
            samples=args.samples_per_domain,
            config=scene_config,
        )[0]
        for domain, shard in DEFAULT_SHARDS.items()
    }
    result = domain_delta_validation(
        records,
        vision=vision,
        atom=atom,
        candidate_bond=candidate_bond,
        original_bond=original_bond,
        set_head=set_head,
        adapter=adapter,
        collator=AtomBondSceneCollator(scene_config),
        proposal_config=proposal_config,
        device=device,
        batch_size=args.batch_size,
        atom_threshold=float(source["training_args"]["atom_decision_threshold"]),
        bond_threshold=float(source["training_args"]["bond_decision_threshold"]),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": args.checkpoint,
        "source_checkpoint": adapter_payload["source_checkpoint"],
        "split": args.split,
        "samples_per_domain": args.samples_per_domain,
        "metrics": result,
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "domains": {
                    domain: {
                        "bond_deployed_exact": metrics["bond_deployed_exact"],
                        "full_object_exact": metrics["full_object_exact"],
                    }
                    for domain, metrics in result["domains"].items()
                },
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
