#!/usr/bin/env python3
"""Evaluate a frozen chemical RETYPE verifier without recalibration."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.atom_pair_chemical_retype import load_atom_pair_chemical_retype_head
from chemvector.atom_pair_chemical_retype_verifier import (
    load_atom_pair_chemical_retype_verifier,
)
from chemvector.atom_pair_deletion_veto import AtomPairDeletionVetoConfig
from chemvector.atom_pair_graph_context import load_atom_pair_graph_context_head
from chemvector.atom_pair_one_move import load_atom_pair_one_move_head
from chemvector.evaluate_atom_pair_graph_context import SPLIT_DOMAINS, load_split_records
from chemvector.predicted_incidence import PredictedProposalConfig
from chemvector.train_atom_bond_scene import AtomBondSceneCollator
from chemvector.train_atom_pair_chemical_retype import validate
from chemvector.train_atom_pair_graph_context import load_group_head
from chemvector.train_bond_presence_calibration import load_components
from chemvector.train_spatial_incidence import DEFAULT_GRAPH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=tuple(SPLIT_DOMAINS), default="validation")
    parser.add_argument("--grounded-graph-checkpoint", default=DEFAULT_GRAPH)
    parser.add_argument("--samples-per-domain", type=int, default=498)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Chemical RETYPE verifier evaluation requires CUDA")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")

    verifier_head, verifier_payload = load_atom_pair_chemical_retype_verifier(
        args.checkpoint, device
    )
    retype_head, retype_payload = load_atom_pair_chemical_retype_head(
        verifier_payload["retype_init_checkpoint"], device
    )
    first_head, first_payload = load_atom_pair_one_move_head(
        retype_payload["first_move_checkpoint"], device
    )
    context_head, context_payload = load_atom_pair_graph_context_head(
        first_payload["base_context_checkpoint"], device
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
        first_head,
        retype_head,
        verifier_head,
    ):
        module.eval().requires_grad_(False)

    proposal_config = PredictedProposalConfig(**source["proposal_config"])
    scene_config = AtomBondSceneConfig(
        atom_queries=proposal_config.maximum_atoms,
        bond_queries=proposal_config.maximum_bonds,
    )
    records = {
        name: load_split_records(
            shard,
            split=args.split,
            samples=args.samples_per_domain,
            config=scene_config,
        )
        for name, shard in SPLIT_DOMAINS[args.split].items()
    }
    margin = float(verifier_payload["gate_threshold"])
    metrics = validate(
        vision,
        atom,
        bond,
        set_head,
        group_head,
        context_head,
        context_payload,
        first_head,
        float(first_payload["minimum_margin"]),
        retype_head,
        AtomPairDeletionVetoConfig(**verifier_payload["relation_config"]),
        records,
        AtomBondSceneCollator(scene_config),
        proposal_config,
        device,
        args.batch_size,
        thresholds=(margin,),
        objective="verifier",
        verifier_head=verifier_head,
    )
    result = {
        "checkpoint": args.checkpoint,
        "samples_seen": verifier_payload["samples_seen"],
        "split": args.split,
        "uses_internal_test": args.split == "internal_test",
        "minimum_margin": margin,
        "baseline": metrics["baseline"],
        "metrics": metrics["best"],
        "diagnostics": metrics["diagnostics"],
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
