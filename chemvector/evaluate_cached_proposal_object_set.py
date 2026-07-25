#!/usr/bin/env python3
"""Evaluate a cached proposal object set checkpoint with fixed thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from chemvector.proposal_object_set import ProposalObjectSetConfig, ProposalObjectSetHead
from chemvector.spatial_incidence import SpatialIncidenceConfig
from chemvector.train_cached_proposal_object_set import evaluate_cached, load_cache


DEFAULT_CHECKPOINT = (
    "outputs/checkpoints/chemvector-v0949-cached-proposal-object-set-train8974-step3000/"
    "best.pt"
)
DEFAULT_CACHE = (
    "outputs/checkpoints/chemvector-v0949-proposal-cache-internal-test100"
)
DEFAULT_OUTPUT = (
    "outputs/evaluations/chemvector-v0949-proposal-object-set-internal-test100/"
    "evaluation.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--expected-split", default="internal_test")
    parser.add_argument("--atom-decision-threshold", type=float, default=-0.25)
    parser.add_argument("--bond-decision-threshold", type=float, default=-0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Cached proposal object set evaluation requires CUDA")
    records, manifest = load_cache(args.cache, args.expected_split)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload["proposal_config"] != manifest["proposal_config"]:
        raise ValueError("Checkpoint and evaluation cache proposal configurations differ")
    incidence_config = SpatialIncidenceConfig(**payload["config"]["incidence"])
    config = ProposalObjectSetConfig(
        incidence=incidence_config,
        hidden_dim=payload["config"]["hidden_dim"],
    )
    device = torch.device("cuda")
    set_head = ProposalObjectSetHead(config).to(device)
    set_head.load_state_dict(payload["head"], strict=True)
    metrics = evaluate_cached(
        set_head,
        records,
        device,
        args.batch_size,
        args.atom_decision_threshold,
        args.bond_decision_threshold,
    )
    report = {
        "schema_version": "cached-proposal-object-set-evaluation-0.1",
        "protocol": "fixed-global-thresholds-no-test-calibration",
        "checkpoint": args.checkpoint,
        "checkpoint_step": payload["global_step"],
        "cache": args.cache,
        "expected_split": args.expected_split,
        "atom_decision_threshold": args.atom_decision_threshold,
        "bond_decision_threshold": args.bond_decision_threshold,
        "metrics": metrics,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
