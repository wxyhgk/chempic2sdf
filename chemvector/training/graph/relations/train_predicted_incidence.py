#!/usr/bin/env python3
"""Train incidence on target-independent predicted Atom/Bond proposals."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from itertools import islice
import json
from pathlib import Path
import random
import time

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.audit_predicted_incidence import evaluate_records
from chemvector.data import ChemVectorIterableDataset
from chemvector.predicted_incidence import (
    PredictedProposalConfig,
    associate_predicted_proposals,
    decode_predicted_atom_proposals,
    decode_predicted_bond_proposals,
)
from chemvector.spatial_incidence import (
    SpatialIncidenceConfig,
    SpatialIncidenceHead,
    spatial_incidence_loss,
)
from chemvector.train_atom_bond_scene import AtomBondSceneCollator, stack_targets
from chemvector.train_spatial_incidence import (
    DEFAULT_ATOM,
    DEFAULT_BOND,
    DEFAULT_GRAPH,
    load_frozen_components,
)


DEFAULT_TRAIN_SHARD = (
    "Data/sdf/visual_svg_v0_1/capacity_100/shards/capacity-000000.tar"
)
DEFAULT_VALIDATION_SHARD = (
    "Data/sdf/chemvision_layers_v0_1/pilot_10k/shards/validation-000000.tar"
)
DEFAULT_WARMSTART = (
    "outputs/checkpoints/chemvector-v0945-spatial-incidence-capacity100-step1000/"
    "best.pt"
)
DEFAULT_OUTPUT = (
    "outputs/checkpoints/chemvector-v0946-predicted-incidence-train100-val100"
)


def _sample_split(sample: dict) -> str | None:
    meta = sample.get("meta", {})
    return meta.get("source_split", meta.get("split"))


def _load_records(path: str, count: int, expected_split: str) -> list[dict]:
    records = list(
        islice(
            iter(ChemVectorIterableDataset([path], sequence_version="0.4")),
            count,
        )
    )
    if len(records) != count:
        raise ValueError(f"Shard contains {len(records)} samples, expected {count}")
    if any(_sample_split(sample) != expected_split for sample in records):
        raise ValueError(f"Shard contains samples outside split {expected_split!r}")
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-shard", default=DEFAULT_TRAIN_SHARD)
    parser.add_argument("--validation-shard", default=DEFAULT_VALIDATION_SHARD)
    parser.add_argument("--train-samples", type=int, default=100)
    parser.add_argument("--validation-samples", type=int, default=100)
    parser.add_argument("--grounded-graph-checkpoint", default=DEFAULT_GRAPH)
    parser.add_argument("--atom-checkpoint", default=DEFAULT_ATOM)
    parser.add_argument("--bond-checkpoint", default=DEFAULT_BOND)
    parser.add_argument("--warmstart-incidence-checkpoint", default=DEFAULT_WARMSTART)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--maximum-atoms", type=int, default=128)
    parser.add_argument("--maximum-bonds", type=int, default=128)
    parser.add_argument("--atom-threshold", type=float, default=0.01)
    parser.add_argument("--bond-threshold", type=float, default=0.01)
    parser.add_argument("--atom-match-radius-px", type=float, default=6.0)
    parser.add_argument("--bond-match-radius-px", type=float, default=12.0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Predicted incidence training requires CUDA")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    vision, atom_head, bond_head, sources = load_frozen_components(
        Path(args.grounded_graph_checkpoint),
        Path(args.atom_checkpoint),
        Path(args.bond_checkpoint),
        device,
    )
    warmstart = torch.load(
        args.warmstart_incidence_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    incidence_config = SpatialIncidenceConfig(**warmstart["config"])
    incidence_head = SpatialIncidenceHead(incidence_config).to(device)
    incidence_head.load_state_dict(warmstart["head"], strict=True)
    proposal_config = PredictedProposalConfig(
        maximum_atoms=args.maximum_atoms,
        maximum_bonds=args.maximum_bonds,
        atom_threshold=args.atom_threshold,
        bond_threshold=args.bond_threshold,
        atom_match_radius_px=args.atom_match_radius_px,
        bond_match_radius_px=args.bond_match_radius_px,
    )
    optimizer = torch.optim.AdamW(
        incidence_head.parameters(), lr=args.learning_rate, weight_decay=0.0
    )
    train_records = _load_records(args.train_shard, args.train_samples, "train")
    validation_records = _load_records(
        args.validation_shard, args.validation_samples, "validation"
    )
    collator = AtomBondSceneCollator(AtomBondSceneConfig())
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_args.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    rng = random.Random(args.seed)
    best: tuple[float, ...] | None = None
    running_loss = 0.0
    started = time.perf_counter()
    for step in range(1, args.max_steps + 1):
        incidence_head.train()
        selected = [
            train_records[rng.randrange(len(train_records))]
            for _ in range(args.batch_size)
        ]
        batch = collator(selected)
        pixels = batch["pixel_values"].to(device)
        targets = stack_targets(batch, device)
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            features = vision.forward_stroke_resolution(pixels)
            atom_output = atom_head(features)
            bond_output = bond_head(features)
            proposals = associate_predicted_proposals(
                decode_predicted_atom_proposals(atom_output, proposal_config),
                decode_predicted_bond_proposals(bond_output, proposal_config),
                targets,
                proposal_config,
            )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = incidence_head(
                proposals["atom_features"],
                proposals["atom_geometry"],
                proposals["atom_mask"],
                proposals["bond_features"],
                proposals["bond_endpoints"],
                proposals["bond_types"],
                proposals["bond_mask"],
            )
            losses = spatial_incidence_loss(
                output,
                proposals["target_atom_refs"],
                proposals["bond_supervision_mask"],
            )
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(incidence_head.parameters(), 1.0)
        optimizer.step()
        running_loss += float(losses["loss"].detach())
        if step % args.log_every == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": running_loss / args.log_every,
                        "endpoint_accuracy": float(losses["endpoint_accuracy"]),
                        "bond_accuracy": float(losses["bond_accuracy"]),
                        "supervised_bonds": int(
                            proposals["bond_supervision_mask"].sum()
                        ),
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
            running_loss = 0.0
        if step % args.validate_every == 0 or step == args.max_steps:
            train_metrics = evaluate_records(
                vision,
                atom_head,
                bond_head,
                incidence_head,
                train_records,
                collator,
                proposal_config,
                device,
                args.batch_size,
            )
            validation_metrics = evaluate_records(
                vision,
                atom_head,
                bond_head,
                incidence_head,
                validation_records,
                collator,
                proposal_config,
                device,
                args.batch_size,
            )
            elapsed = time.perf_counter() - started
            report = {
                "step": step,
                "elapsed_seconds": elapsed,
                "train": train_metrics,
                "validation": validation_metrics,
            }
            print(json.dumps({"evaluation": report}, separators=(",", ":")), flush=True)
            checkpoint = {
                "checkpoint_schema_version": "predicted-incidence-checkpoint-0.1",
                "head": incidence_head.state_dict(),
                "config": asdict(incidence_config),
                "proposal_config": asdict(proposal_config),
                "optimizer": optimizer.state_dict(),
                "global_step": step,
                "evaluation": report,
                "sources": sources,
                "warmstart_incidence_checkpoint": args.warmstart_incidence_checkpoint,
                "training_args": vars(args),
            }
            torch.save(checkpoint, output_dir / f"step-{step:06d}.pt")
            torch.save(checkpoint, output_dir / "last.pt")
            score = (
                validation_metrics["end_to_end_incidence_image_exact"],
                validation_metrics["eligible_bond_accuracy"],
                validation_metrics["eligible_endpoint_accuracy"],
            )
            if best is None or score > best:
                best = score
                torch.save(checkpoint, output_dir / "best.pt")
            (output_dir / "metrics.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
