#!/usr/bin/env python3
"""Calibrate final Bond presence on a frozen joint proposal-object checkpoint."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.dense_bond_object import (
    DenseBondObjectConfig,
    DenseBondObjectDetector,
    MultiScaleDenseBondObjectDetector,
)
from chemvector.predicted_incidence import (
    PredictedProposalConfig,
    associate_predicted_proposals,
    decode_predicted_atom_proposals,
    decode_predicted_bond_proposals,
)
from chemvector.proposal_object_set import (
    ProposalObjectSetConfig,
    ProposalObjectSetHead,
    proposal_object_set_loss,
)
from chemvector.spatial_atom_object import (
    SpatialAtomObjectConfig,
    SpatialAtomObjectDetector,
)
from chemvector.spatial_incidence import SpatialIncidenceConfig
from chemvector.train_atom_bond_scene import AtomBondSceneCollator, stack_targets
from chemvector.train_joint_proposal_objects_full import (
    DEFAULT_TRAIN_SHARDS,
    DEFAULT_VALIDATION_SHARDS,
    load_validation_records,
    training_batches,
)
from chemvector.train_proposal_object_set import evaluate
from chemvector.train_spatial_atom_object import load_grounded_vision
from chemvector.train_spatial_incidence import DEFAULT_GRAPH, forward_frozen_bond_head


DEFAULT_SOURCE = (
    "outputs/checkpoints/chemvector-v0971-joint-proposal-objects-full50k/"
    "samples-030000.pt"
)
DEFAULT_OUTPUT = (
    "outputs/checkpoints/chemvector-v0972-bond-presence-calibration-10k"
)


def load_components(path: str, graph_checkpoint: str, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    vision, _, _ = load_grounded_vision(Path(graph_checkpoint), device)
    vision.eval().requires_grad_(False)

    atom = SpatialAtomObjectDetector(
        SpatialAtomObjectConfig(**payload["atom_config"])
    ).to(device)
    atom.load_state_dict(payload["atom_head"], strict=True)
    atom.eval().requires_grad_(False)

    bond_config = DenseBondObjectConfig(**payload["bond_config"])
    bond = (
        MultiScaleDenseBondObjectDetector(bond_config).to(device)
        if payload["bond_multiscale"]
        else DenseBondObjectDetector(bond_config).to(device)
    )
    bond.load_state_dict(payload["bond_head"], strict=True)
    bond.eval().requires_grad_(False)

    set_config_payload = dict(payload["set_config"])
    set_config_payload["incidence"] = SpatialIncidenceConfig(
        **set_config_payload["incidence"]
    )
    set_config = ProposalObjectSetConfig(**set_config_payload)
    set_head = ProposalObjectSetHead(set_config).to(device)
    set_head.load_state_dict(payload["set_head"], strict=True)
    set_head.requires_grad_(False)
    set_head.bond_presence_residual.requires_grad_(True)
    return vision, atom, bond, set_head, payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", default=DEFAULT_SOURCE)
    parser.add_argument("--grounded-graph-checkpoint", default=DEFAULT_GRAPH)
    parser.add_argument("--train-shards", nargs="+", default=DEFAULT_TRAIN_SHARDS)
    parser.add_argument(
        "--validation-shards", nargs="+", default=DEFAULT_VALIDATION_SHARDS
    )
    parser.add_argument("--train-samples", type=int, default=10_000)
    parser.add_argument("--validation-samples-per-shard", type=int, default=498)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--validate-every-samples", type=int, default=2_500)
    parser.add_argument("--log-every-samples", type=int, default=1_000)
    parser.add_argument("--shuffle-buffer", type=int, default=2048)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Bond presence calibration requires CUDA")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_args.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    vision, atom, bond, set_head, source = load_components(
        args.source_checkpoint, args.grounded_graph_checkpoint, device
    )
    proposal_config = PredictedProposalConfig(**source["proposal_config"])
    scene_config = AtomBondSceneConfig(
        atom_queries=proposal_config.maximum_atoms,
        bond_queries=proposal_config.maximum_bonds,
    )
    collator = AtomBondSceneCollator(scene_config)
    validation, validation_audit = load_validation_records(
        args.validation_shards,
        per_shard=args.validation_samples_per_shard,
        config=scene_config,
    )
    (output_dir / "validation_audit.json").write_text(
        json.dumps(validation_audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    atom_threshold = float(source["training_args"]["atom_decision_threshold"])
    bond_threshold = float(source["training_args"]["bond_decision_threshold"])
    initial = evaluate(
        vision,
        atom,
        bond,
        set_head,
        validation,
        collator,
        proposal_config,
        device,
        args.batch_size,
        atom_threshold,
        bond_threshold,
    )
    (output_dir / "initial_metrics.json").write_text(
        json.dumps(initial, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"initial_validation": initial}, separators=(",", ":")), flush=True)

    trainable = list(set_head.bond_presence_residual.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.0)
    audit = {}
    batches = training_batches(
        args.train_shards,
        samples=args.train_samples,
        batch_size=args.batch_size,
        seed=args.seed,
        shuffle_buffer=args.shuffle_buffer,
        config=scene_config,
        audit=audit,
    )
    samples_seen = 0
    running_loss = 0.0
    running_samples = 0
    next_log = args.log_every_samples
    next_validation = args.validate_every_samples
    best_score = None
    best_metrics = None
    started = time.perf_counter()

    for samples in batches:
        set_head.train()
        batch = collator(samples)
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        targets = stack_targets(batch, device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            features = vision.forward_stroke_resolution(pixels)
            atom_output = atom(features)
            bond_output = forward_frozen_bond_head(vision, bond, features)
            proposals = associate_predicted_proposals(
                decode_predicted_atom_proposals(atom_output, proposal_config),
                decode_predicted_bond_proposals(bond_output, proposal_config),
                targets,
                proposal_config,
            )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = set_head(
                proposals["atom_features"],
                proposals["atom_geometry"],
                proposals["atom_scores"],
                proposals["atom_mask"],
                proposals["bond_features"],
                proposals["bond_endpoints"],
                proposals["bond_types"],
                proposals["bond_scores"],
                proposals["bond_mask"],
            )
            losses = proposal_object_set_loss(
                output,
                proposals["atom_presence_targets"],
                proposals["bond_presence_targets"],
                proposals["atom_mask"],
                proposals["bond_mask"],
                proposals["target_atom_refs"],
                proposals["bond_supervision_mask"],
                atom_decision_threshold=atom_threshold,
                bond_decision_threshold=bond_threshold,
            )
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        count = len(samples)
        samples_seen += count
        running_samples += count
        running_loss += float(losses["loss"].detach()) * count
        if samples_seen >= next_log:
            print(
                json.dumps(
                    {
                        "samples_seen": samples_seen,
                        "loss": running_loss / running_samples,
                        "elapsed_seconds": time.perf_counter() - started,
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
            running_loss = 0.0
            running_samples = 0
            next_log += args.log_every_samples
        if samples_seen >= next_validation or samples_seen >= args.train_samples:
            metrics = evaluate(
                vision,
                atom,
                bond,
                set_head,
                validation,
                collator,
                proposal_config,
                device,
                args.batch_size,
                atom_threshold,
                bond_threshold,
            )
            metrics.update(
                {
                    "samples_seen": samples_seen,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            print(json.dumps({"validation": metrics}, separators=(",", ":")), flush=True)
            checkpoint = {
                **{
                    key: value
                    for key, value in source.items()
                    if key not in {"optimizer", "validation", "samples_seen"}
                },
                "checkpoint_schema_version": "joint-proposal-objects-checkpoint-0.2",
                "set_head": set_head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "samples_seen": source["samples_seen"] + samples_seen,
                "calibration_samples": samples_seen,
                "validation": metrics,
                "source_joint_checkpoint": args.source_checkpoint,
                "calibration_args": vars(args),
            }
            torch.save(checkpoint, output_dir / f"samples-{samples_seen:06d}.pt")
            torch.save(checkpoint, output_dir / "last.pt")
            score = (
                metrics["full_object_exact_rate"],
                metrics["bond_deployed_exact_rate"],
            )
            if best_score is None or score > best_score:
                best_score = score
                best_metrics = metrics
                torch.save(checkpoint, output_dir / "best.pt")
            (output_dir / "metrics.json").write_text(
                json.dumps(
                    {"latest": metrics, "best": best_metrics},
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            next_validation += args.validate_every_samples
    (output_dir / "training_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
