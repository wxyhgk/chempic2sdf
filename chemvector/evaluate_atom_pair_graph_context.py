#!/usr/bin/env python3
"""Evaluate a frozen AtomPair graph-context checkpoint on both internal domains."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.atom_pair_graph_context import load_atom_pair_graph_context_head
from chemvector.data import ChemVectorIterableDataset
from chemvector.predicted_incidence import PredictedProposalConfig
from chemvector.train_atom_bond_scene import AtomBondSceneCollator
from chemvector.train_atom_pair_bond_group import VALIDATION_DOMAINS
from chemvector.train_atom_pair_graph_context import (
    load_group_head,
    validate,
)
from chemvector.train_bond_presence_calibration import load_components
from chemvector.train_joint_proposal_objects_full import (
    rejection_reason,
    sample_split,
)
from chemvector.train_spatial_incidence import DEFAULT_GRAPH


SPLIT_DOMAINS = {
    "validation": VALIDATION_DOMAINS,
    "internal_test": {
        "chemvision": (
            "Data/sdf/chemvision_layers_v0_1/pilot_10k/shards/"
            "internal_test-000000.tar"
        ),
        "full_v0": "Data/sdf/webdataset_full_v0/shards/internal_test-000000.tar",
    },
}


def load_split_records(
    shard: str,
    *,
    split: str,
    samples: int,
    config: AtomBondSceneConfig,
) -> list[dict]:
    rows = []
    for sample in ChemVectorIterableDataset([shard], sequence_version="0.4"):
        if sample_split(sample) != split:
            raise ValueError(f"{shard} contains a non-{split} sample")
        if rejection_reason(sample, config) is not None:
            continue
        rows.append(sample)
        if len(rows) >= samples:
            break
    if len(rows) != samples:
        raise ValueError(f"{shard} yielded {len(rows)}, expected {samples}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=tuple(SPLIT_DOMAINS), default="validation")
    parser.add_argument("--grounded-graph-checkpoint", default=DEFAULT_GRAPH)
    parser.add_argument("--validation-samples-per-domain", type=int, default=498)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("AtomPair graph-context evaluation requires CUDA")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    context_head, context_payload = load_atom_pair_graph_context_head(
        args.checkpoint, device
    )
    group_head, group_payload = load_group_head(
        context_payload["group_checkpoint"], device
    )
    vision, atom, bond, set_head, source = load_components(
        group_payload["source_checkpoint"], args.grounded_graph_checkpoint, device
    )
    proposal_config = PredictedProposalConfig(**source["proposal_config"])
    scene_config = AtomBondSceneConfig(
        atom_queries=proposal_config.maximum_atoms,
        bond_queries=proposal_config.maximum_bonds,
    )
    collator = AtomBondSceneCollator(scene_config)
    records = {}
    for name, shard in SPLIT_DOMAINS[args.split].items():
        rows = load_split_records(
            shard,
            split=args.split,
            samples=args.validation_samples_per_domain,
            config=scene_config,
        )
        records[name] = rows
    baseline = validate(
        vision,
        atom,
        bond,
        set_head,
        group_head,
        context_head,
        records,
        collator,
        proposal_config,
        device,
        args.batch_size,
        residual_scale=0.0,
    )
    metrics = validate(
        vision,
        atom,
        bond,
        set_head,
        group_head,
        context_head,
        records,
        collator,
        proposal_config,
        device,
        args.batch_size,
        residual_scale=float(context_payload.get("residual_scale", 1.0)),
        base_positive_protection_threshold=context_payload.get(
            "base_positive_protection_threshold"
        ),
    )
    result = {
        "checkpoint": args.checkpoint,
        "samples_seen": context_payload["samples_seen"],
        "split": args.split,
        "uses_internal_test": args.split == "internal_test",
        "residual_scale": float(context_payload.get("residual_scale", 1.0)),
        "base_positive_protection_threshold": context_payload.get(
            "base_positive_protection_threshold"
        ),
        "baseline": baseline,
        "metrics": metrics,
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
