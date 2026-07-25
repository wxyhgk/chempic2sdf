#!/usr/bin/env python3
"""Build compact frozen Atom/Bond proposal caches for object-set training."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import glob
import hashlib
import json
from pathlib import Path
import time

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.data import ChemVectorIterableDataset
from chemvector.predicted_incidence import (
    PredictedProposalConfig,
    associate_predicted_proposals,
    decode_predicted_atom_proposals,
    decode_predicted_bond_proposals,
    fuse_predicted_atom_proposals,
)
from chemvector.spatial_atom_object import SpatialAtomObjectConfig, SpatialAtomObjectDetector
from chemvector.train_atom_bond_scene import AtomBondSceneCollator, stack_targets
from chemvector.train_predicted_incidence import _sample_split
from chemvector.train_spatial_incidence import (
    DEFAULT_ATOM,
    DEFAULT_BOND,
    DEFAULT_GRAPH,
    load_frozen_components,
)


TENSOR_FIELDS = (
    "atom_features",
    "atom_geometry",
    "atom_scores",
    "atom_mask",
    "bond_features",
    "bond_endpoints",
    "bond_types",
    "bond_scores",
    "bond_mask",
    "atom_presence_targets",
    "bond_presence_targets",
    "bond_supervision_mask",
    "target_atom_refs",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expand_shards(patterns: list[str]) -> list[str]:
    expanded = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        expanded.extend(matches or [pattern])
    unique = list(dict.fromkeys(expanded))
    if not unique or any(not Path(path).is_file() for path in unique):
        raise FileNotFoundError("One or more proposal cache input shards do not exist")
    return unique


def _cache_record(
    sample: dict,
    proposals: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    batch_index: int,
) -> dict:
    record = {
        "key": sample["key"],
        "source_split": _sample_split(sample),
        "atom_target_count": int(targets["atom_mask"][batch_index].sum()),
        "bond_target_count": int(targets["bond_mask"][batch_index].sum()),
        "atom_matched_count": int(
            proposals["atom_target_matched"][batch_index].sum()
        ),
        "bond_matched_count": int(
            proposals["bond_target_matched"][batch_index].sum()
        ),
        "incidence_eligible_count": int(
            proposals["incidence_target_eligible"][batch_index].sum()
        ),
    }
    for name in TENSOR_FIELDS:
        value = proposals[name][batch_index].detach().cpu().contiguous()
        if value.is_floating_point():
            value = value.to(torch.bfloat16)
        record[name] = value
    return record


@torch.no_grad()
def _process_batch(
    samples,
    collator,
    vision,
    atom_head,
    bond_head,
    rescue_atom_head,
    proposal_config,
    device,
    primary_atom_capacity,
    rescue_atom_capacity,
    atom_fusion_radius_px,
) -> list[dict]:
    batch = collator(samples)
    pixels = batch["pixel_values"].to(device)
    targets = stack_targets(batch, device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        features = vision.forward_stroke_resolution(pixels)
        atom_output = atom_head(features)
        rescue_atom_output = (
            rescue_atom_head(features) if rescue_atom_head is not None else None
        )
        bond_output = bond_head(features)
        atom_proposals = (
            fuse_predicted_atom_proposals(
                atom_output,
                rescue_atom_output,
                proposal_config,
                primary_capacity=primary_atom_capacity,
                rescue_capacity=rescue_atom_capacity,
                dedup_radius_px=atom_fusion_radius_px,
            )
            if rescue_atom_output is not None
            else decode_predicted_atom_proposals(atom_output, proposal_config)
        )
        proposals = associate_predicted_proposals(
            atom_proposals,
            decode_predicted_bond_proposals(bond_output, proposal_config),
            targets,
            proposal_config,
        )
    return [
        _cache_record(sample, proposals, targets, batch_index)
        for batch_index, sample in enumerate(samples)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--expected-split", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--maximum-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cache-shard-size", type=int, default=500)
    parser.add_argument("--grounded-graph-checkpoint", default=DEFAULT_GRAPH)
    parser.add_argument("--atom-checkpoint", default=DEFAULT_ATOM)
    parser.add_argument("--rescue-atom-checkpoint")
    parser.add_argument("--bond-checkpoint", default=DEFAULT_BOND)
    parser.add_argument("--maximum-atoms", type=int, default=128)
    parser.add_argument("--primary-atom-capacity", type=int, default=128)
    parser.add_argument("--rescue-atom-capacity", type=int, default=16)
    parser.add_argument("--atom-fusion-radius-px", type=float, default=6.0)
    parser.add_argument("--maximum-bonds", type=int, default=128)
    parser.add_argument("--atom-threshold", type=float, default=0.01)
    parser.add_argument("--bond-threshold", type=float, default=0.01)
    parser.add_argument("--atom-match-radius-px", type=float, default=6.0)
    parser.add_argument("--bond-match-radius-px", type=float, default=12.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Proposal cache generation requires CUDA")
    if min(args.batch_size, args.cache_shard_size) < 1:
        raise ValueError("Batch and cache shard sizes must be positive")
    device = torch.device("cuda")
    input_shards = _expand_shards(args.shards)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    vision, atom_head, bond_head, sources = load_frozen_components(
        Path(args.grounded_graph_checkpoint),
        Path(args.atom_checkpoint),
        Path(args.bond_checkpoint),
        device,
    )
    rescue_atom_head = None
    if args.rescue_atom_checkpoint:
        rescue_payload = torch.load(
            args.rescue_atom_checkpoint, map_location="cpu", weights_only=False
        )
        rescue_atom_head = SpatialAtomObjectDetector(
            SpatialAtomObjectConfig(**rescue_payload["config"])
        ).to(device)
        rescue_atom_head.load_state_dict(rescue_payload["head"], strict=True)
        rescue_atom_head.eval().requires_grad_(False)
        if rescue_atom_head.config.input_channels != atom_head.config.input_channels:
            raise ValueError("Primary and rescue Atom heads use different channels")
        if args.maximum_atoms < args.primary_atom_capacity + args.rescue_atom_capacity:
            raise ValueError("maximum-atoms is too small for fused Atom proposals")
        sources["rescue_atom_checkpoint"] = args.rescue_atom_checkpoint
    proposal_config = PredictedProposalConfig(
        maximum_atoms=args.maximum_atoms,
        maximum_bonds=args.maximum_bonds,
        atom_threshold=args.atom_threshold,
        bond_threshold=args.bond_threshold,
        atom_match_radius_px=args.atom_match_radius_px,
        bond_match_radius_px=args.bond_match_radius_px,
    )
    collator = AtomBondSceneCollator(AtomBondSceneConfig())
    dataset = ChemVectorIterableDataset(input_shards, sequence_version="0.4")
    pending_samples = []
    pending_records = []
    cache_files = []
    rejected = []
    raw_seen = valid = 0
    started = time.perf_counter()

    def flush() -> None:
        nonlocal pending_records
        if not pending_records:
            return
        path = output_dir / f"cache-{len(cache_files):05d}.pt"
        torch.save(
            {
                "schema_version": "proposal-object-cache-shard-0.1",
                "records": pending_records,
            },
            path,
        )
        cache_files.append(
            {
                "path": path.name,
                "records": len(pending_records),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
        pending_records = []

    def process(samples: list[dict]) -> None:
        nonlocal valid
        try:
            records = _process_batch(
                samples,
                collator,
                vision,
                atom_head,
                bond_head,
                rescue_atom_head,
                proposal_config,
                device,
                args.primary_atom_capacity,
                args.rescue_atom_capacity,
                args.atom_fusion_radius_px,
            )
        except Exception as error:
            if len(samples) > 1:
                for sample in samples:
                    process([sample])
                return
            rejected.append(
                {
                    "key": samples[0].get("key"),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            return
        pending_records.extend(records)
        valid += len(records)
        if len(pending_records) >= args.cache_shard_size:
            overflow = pending_records[args.cache_shard_size :]
            del pending_records[args.cache_shard_size :]
            flush()
            pending_records.extend(overflow)

    for sample in dataset:
        if args.maximum_samples and raw_seen >= args.maximum_samples:
            break
        raw_seen += 1
        if _sample_split(sample) != args.expected_split:
            raise ValueError(
                f"Sample {sample.get('key')} is outside split {args.expected_split!r}"
            )
        pending_samples.append(sample)
        if len(pending_samples) == args.batch_size:
            process(pending_samples)
            pending_samples = []
        if raw_seen % 500 == 0:
            print(
                json.dumps(
                    {
                        "raw_seen": raw_seen,
                        "valid": valid,
                        "rejected": len(rejected),
                        "elapsed_seconds": time.perf_counter() - started,
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
    if pending_samples:
        process(pending_samples)
    flush()
    manifest = {
        "schema_version": "proposal-object-cache-manifest-0.1",
        "expected_split": args.expected_split,
        "input_shards": [
            {"path": path, "bytes": Path(path).stat().st_size}
            for path in input_shards
        ],
        "raw_seen": raw_seen,
        "valid_records": valid,
        "rejected_count": len(rejected),
        "rejected": rejected,
        "proposal_config": asdict(proposal_config),
        "sources": sources,
        "cache_files": cache_files,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
