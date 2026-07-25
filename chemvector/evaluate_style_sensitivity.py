#!/usr/bin/env python3
"""Audit a frozen chemical model across paired rendering styles."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import tarfile

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
from chemvector.data import ChemVectorIterableDataset
from chemvector.predicted_incidence import PredictedProposalConfig
from chemvector.style_sensitivity import summarize_style_sensitivity
from chemvector.train_atom_bond_scene import AtomBondSceneCollator
from chemvector.train_atom_pair_chemical_retype import validate
from chemvector.train_atom_pair_graph_context import load_group_head
from chemvector.train_bond_presence_calibration import load_components
from chemvector.train_joint_proposal_objects_full import rejection_reason, sample_split
from chemvector.train_spatial_incidence import DEFAULT_GRAPH


DEFAULT_VALIDATION_SHARD = (
    "Data/sdf/chemvision_layers_v0_2/font_pilot_10k/shards/"
    "validation-000000.tar"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--shard", default=DEFAULT_VALIDATION_SHARD)
    parser.add_argument(
        "--split", choices=("validation", "internal_test"), default="validation"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=0,
        help="Accepted sample limit; 0 reads the complete shard",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grounded-graph-checkpoint", default=DEFAULT_GRAPH)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_style_records(
    shard: str,
    *,
    split: str,
    samples: int,
    config: AtomBondSceneConfig,
) -> tuple[list[dict], dict]:
    if samples < 0:
        raise ValueError("Style audit sample limit must be non-negative")
    rows = []
    rejection_counts = Counter()
    source_samples = 0
    with tarfile.open(shard, "r:*") as archive:
        shard_samples = sum(
            member.name.endswith(".meta.json") for member in archive.getmembers()
        )
    for sample in ChemVectorIterableDataset([shard], sequence_version="0.4"):
        if sample_split(sample) != split:
            raise ValueError(f"{shard} contains a non-{split} sample")
        source_samples += 1
        reason = rejection_reason(sample, config)
        if reason is not None:
            rejection_counts[reason] += 1
            continue
        rows.append(sample)
        if samples and len(rows) >= samples:
            break
    if samples and len(rows) != samples:
        raise ValueError(f"{shard} yielded {len(rows)}, expected {samples}")
    return rows, {
        "source_samples_in_shard": shard_samples,
        "loader_samples": source_samples,
        "loader_skipped_coordinate_collisions": (
            shard_samples - source_samples if samples == 0 else None
        ),
        "accepted_samples": len(rows),
        "rejected_samples": sum(rejection_counts.values()),
        "rejection_reasons": dict(rejection_counts),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Style-sensitivity evaluation requires CUDA")
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
    rows, input_audit = load_style_records(
        args.shard,
        split=args.split,
        samples=args.samples,
        config=scene_config,
    )
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
        {"font_pilot": rows},
        AtomBondSceneCollator(scene_config),
        proposal_config,
        device,
        args.batch_size,
        thresholds=(margin,),
        objective="verifier",
        verifier_head=verifier_head,
        collect_sample_details=True,
    )
    sample_details = metrics.pop("samples")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    samples_output = output.with_name(f"{output.stem}-samples.jsonl")
    samples_output.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in sample_details
        ),
        encoding="utf-8",
    )
    result = {
        "checkpoint": args.checkpoint,
        "samples_seen": verifier_payload["samples_seen"],
        "shard": args.shard,
        "split": args.split,
        "uses_internal_test": args.split == "internal_test",
        "samples": len(sample_details),
        "input_audit": input_audit,
        "minimum_margin": margin,
        "frozen_metrics": metrics,
        "baseline_style": summarize_style_sensitivity(
            sample_details, baseline=True
        ),
        "candidate_style": summarize_style_sensitivity(
            sample_details, baseline=False
        ),
        "sample_details": str(samples_output),
    }
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "samples": len(sample_details),
                "baseline": result["baseline_style"]["overall"],
                "candidate": result["candidate_style"]["overall"],
                "paired": result["candidate_style"]["paired_renderings"],
                "maximum_bucket_gaps": result["candidate_style"][
                    "maximum_bucket_gaps"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
