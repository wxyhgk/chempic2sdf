#!/usr/bin/env python3
"""Calibrate deletion-veto threshold on the fixed two-domain validation split."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.atom_pair_deletion_veto import load_atom_pair_deletion_veto_head
from chemvector.atom_pair_graph_context import load_atom_pair_graph_context_head
from chemvector.predicted_incidence import PredictedProposalConfig
from chemvector.train_atom_bond_scene import AtomBondSceneCollator
from chemvector.train_atom_pair_bond_group import VALIDATION_DOMAINS
from chemvector.train_atom_pair_deletion_veto import selection_key, validate
from chemvector.train_atom_pair_graph_context import load_group_head
from chemvector.train_bond_presence_calibration import load_components
from chemvector.train_joint_proposal_objects_full import load_validation_records
from chemvector.train_spatial_incidence import DEFAULT_GRAPH


def parse_thresholds(value: str) -> tuple[float, ...]:
    thresholds = tuple(float(item) for item in value.split(",") if item.strip())
    if not thresholds:
        raise argparse.ArgumentTypeError("At least one threshold is required")
    return thresholds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--grounded-graph-checkpoint", default=DEFAULT_GRAPH)
    parser.add_argument("--samples-per-domain", type=int, default=498)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--thresholds",
        type=parse_thresholds,
        default=parse_thresholds(
            "-2.50,-2.40,-2.30,-2.20,-2.10,-2.00,-1.90,-1.80,-1.70,-1.60,"
            "-1.50,-1.40,-1.30,-1.20,-1.10,-1.00"
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--calibrated-checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("AtomPair deletion-veto calibration requires CUDA")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    veto_head, veto_payload = load_atom_pair_deletion_veto_head(args.checkpoint, device)
    context_head, context_payload = load_atom_pair_graph_context_head(
        veto_payload["base_context_checkpoint"], device
    )
    group_head, group_payload = load_group_head(context_payload["group_checkpoint"], device)
    vision, atom, bond, set_head, source = load_components(
        group_payload["source_checkpoint"], args.grounded_graph_checkpoint, device
    )
    for module in (vision, atom, bond, set_head, group_head, context_head, veto_head):
        module.eval().requires_grad_(False)
    proposal_config = PredictedProposalConfig(**source["proposal_config"])
    scene_config = AtomBondSceneConfig(
        atom_queries=proposal_config.maximum_atoms,
        bond_queries=proposal_config.maximum_bonds,
    )
    collator = AtomBondSceneCollator(scene_config)
    records = {
        name: load_validation_records(
            [shard], per_shard=args.samples_per_domain, config=scene_config
        )[0]
        for name, shard in VALIDATION_DOMAINS.items()
    }
    metrics = validate(
        vision,
        atom,
        bond,
        set_head,
        group_head,
        context_head,
        context_payload,
        veto_head,
        records,
        collator,
        proposal_config,
        device,
        args.batch_size,
        thresholds=args.thresholds,
    )
    baseline_key = selection_key(metrics["baseline"])
    best_key = selection_key(metrics["best"])
    result = {
        "checkpoint": args.checkpoint,
        "selection_protocol": (
            "minimum-domain chemical, pooled chemical, minimum-domain presence, "
            "pooled presence, false negatives, false positives"
        ),
        "baseline": metrics["baseline"],
        "thresholds": metrics["thresholds"],
        "best_threshold": metrics["best_threshold"],
        "best": metrics["best"],
        "improves_selection_key": best_key > baseline_key,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.calibrated_checkpoint:
        calibrated = dict(veto_payload)
        calibrated["deletion_threshold"] = metrics["best_threshold"]
        calibrated["calibration"] = {
            "source_checkpoint": args.checkpoint,
            "output": args.output,
            "best": metrics["best"],
        }
        target = Path(args.calibrated_checkpoint)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(calibrated, target)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
