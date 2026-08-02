#!/usr/bin/env python3
"""Evaluate a frozen proposal object set checkpoint with fixed thresholds."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.predicted_incidence import PredictedProposalConfig
from chemvector.proposal_object_set import ProposalObjectSetConfig, ProposalObjectSetHead
from chemvector.spatial_incidence import SpatialIncidenceConfig
from chemvector.train_atom_bond_scene import AtomBondSceneCollator
from chemvector.train_predicted_incidence import _load_records
from chemvector.train_proposal_object_set import evaluate
from chemvector.train_spatial_incidence import (
    DEFAULT_ATOM,
    DEFAULT_BOND,
    DEFAULT_GRAPH,
    load_frozen_components,
)


DEFAULT_CHECKPOINT = (
    "outputs/checkpoints/chemvector-v0948-proposal-object-set-train1000-cont1200/"
    "best.pt"
)
DEFAULT_INTERNAL_TEST_SHARD = (
    "Data/sdf/chemvision_layers_v0_1/pilot_10k/shards/internal_test-000000.tar"
)
DEFAULT_OUTPUT = (
    "outputs/evaluations/chemvector-v0948-proposal-object-set-internal-test100/"
    "evaluation.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--shard", default=DEFAULT_INTERNAL_TEST_SHARD)
    parser.add_argument("--expected-split", default="internal_test")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--atom-decision-threshold", type=float, default=0.0)
    parser.add_argument("--bond-decision-threshold", type=float, default=-0.5)
    parser.add_argument("--grounded-graph-checkpoint", default=DEFAULT_GRAPH)
    parser.add_argument("--atom-checkpoint", default=DEFAULT_ATOM)
    parser.add_argument("--bond-checkpoint", default=DEFAULT_BOND)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Proposal object set evaluation requires CUDA")
    device = torch.device("cuda")
    vision, atom_head, bond_head, sources = load_frozen_components(
        Path(args.grounded_graph_checkpoint),
        Path(args.atom_checkpoint),
        Path(args.bond_checkpoint),
        device,
    )
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    incidence_config = SpatialIncidenceConfig(**payload["config"]["incidence"])
    config = ProposalObjectSetConfig(
        incidence=incidence_config,
        hidden_dim=payload["config"]["hidden_dim"],
    )
    set_head = ProposalObjectSetHead(config).to(device)
    set_head.load_state_dict(payload["head"], strict=True)
    proposal_config = PredictedProposalConfig(**payload["proposal_config"])
    records = _load_records(args.shard, args.samples, args.expected_split)
    metrics = evaluate(
        vision,
        atom_head,
        bond_head,
        set_head,
        records,
        AtomBondSceneCollator(AtomBondSceneConfig()),
        proposal_config,
        device,
        args.batch_size,
        args.atom_decision_threshold,
        args.bond_decision_threshold,
    )
    report = {
        "schema_version": "proposal-object-set-evaluation-0.1",
        "protocol": "fixed-global-thresholds-no-test-calibration",
        "checkpoint": args.checkpoint,
        "checkpoint_step": payload["global_step"],
        "shard": args.shard,
        "expected_split": args.expected_split,
        "samples": args.samples,
        "atom_decision_threshold": args.atom_decision_threshold,
        "bond_decision_threshold": args.bond_decision_threshold,
        "proposal_config": asdict(proposal_config),
        "sources": sources,
        "metrics": metrics,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    concise = dict(report)
    concise["sources"] = sources["vision"]
    print(json.dumps(concise, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
