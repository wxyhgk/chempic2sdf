#!/usr/bin/env python3
"""Train Atom/Bond proposal heads on three balanced rendering domains."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.dense_bond_object import dense_bond_object_loss
from chemvector.predicted_incidence import PredictedProposalConfig
from chemvector.spatial_atom_object import spatial_atom_object_loss
from chemvector.train_atom_bond_scene import AtomBondSceneCollator, stack_targets
from chemvector.train_bond_presence_calibration import load_components
from chemvector.train_joint_proposal_objects_full import bond_targets, load_validation_records
from chemvector.train_multidomain_set_head import (
    TRAIN_DOMAINS,
    VALIDATION_DOMAINS,
    balanced_batches,
)
from chemvector.train_spatial_incidence import DEFAULT_GRAPH, forward_frozen_bond_head
from chemvector.train_stable_bond_presence import domain_validation


DEFAULT_SOURCE = (
    "outputs/checkpoints/chemvector-v0972-bond-presence-calibration-10k/best.pt"
)
DEFAULT_OUTPUT = "outputs/checkpoints/chemvector-v0976-multidomain-proposals-balanced18k"


def freeze_proposal_object_features(
    atom: torch.nn.Module,
    bond: torch.nn.Module,
) -> None:
    """Keep the feature tensors consumed by the downstream set head stable."""

    for module in (atom.input_projection, atom.context):
        module.requires_grad_(False)
    bond_base = bond.base if hasattr(bond, "base") else bond
    for module in (bond_base.input_projection, bond_base.context):
        module.requires_grad_(False)
    for name in ("stride8_adapter", "stride16_adapter"):
        adapter = getattr(bond, name, None)
        if adapter is not None:
            adapter.requires_grad_(False)


def add_proposal_summary(metrics: dict) -> dict:
    """Add domain-balanced proposal coverage metrics used for model selection."""

    domains = metrics["domains"]
    atom_recall = [item["atom_proposal_recall"] for item in domains.values()]
    bond_recall = [item["bond_proposal_recall"] for item in domains.values()]
    joint_recall = [
        min(item["atom_proposal_recall"], item["bond_proposal_recall"])
        for item in domains.values()
    ]
    metrics["proposal_summary"] = {
        "minimum_domain_atom_proposal_recall": min(atom_recall),
        "minimum_domain_bond_proposal_recall": min(bond_recall),
        "minimum_domain_joint_proposal_recall": min(joint_recall),
        "mean_domain_atom_proposal_recall": sum(atom_recall) / len(atom_recall),
        "mean_domain_bond_proposal_recall": sum(bond_recall) / len(bond_recall),
    }
    return metrics


def proposal_checkpoint_score(metrics: dict) -> tuple[float, ...]:
    summary = metrics["proposal_summary"]
    return (
        summary["minimum_domain_joint_proposal_recall"],
        summary["minimum_domain_bond_proposal_recall"],
        summary["minimum_domain_atom_proposal_recall"],
        summary["mean_domain_bond_proposal_recall"],
        summary["mean_domain_atom_proposal_recall"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", default=DEFAULT_SOURCE)
    parser.add_argument("--grounded-graph-checkpoint", default=DEFAULT_GRAPH)
    parser.add_argument("--samples-per-domain", type=int, default=6_000)
    parser.add_argument("--validation-samples-per-domain", type=int, default=498)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--atom-learning-rate", type=float, default=1e-5)
    parser.add_argument("--bond-learning-rate", type=float, default=1e-5)
    parser.add_argument("--atom-loss-weight", type=float, default=1.0)
    parser.add_argument("--bond-loss-weight", type=float, default=1.0)
    parser.add_argument("--freeze-object-features", action="store_true")
    parser.add_argument("--feature-distillation-weight", type=float, default=0.0)
    parser.add_argument("--validate-every-total-samples", type=int, default=4_500)
    parser.add_argument("--log-every-total-samples", type=int, default=900)
    parser.add_argument("--shuffle-buffer", type=int, default=1024)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Multidomain proposal training requires CUDA")
    if args.batch_size % len(TRAIN_DOMAINS):
        raise ValueError("Batch size must be divisible by the number of domains")
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
    teacher_atom = teacher_bond = None
    if args.feature_distillation_weight > 0.0:
        teacher_atom = copy.deepcopy(atom).eval().requires_grad_(False)
        teacher_bond = copy.deepcopy(bond).eval().requires_grad_(False)
    atom.requires_grad_(True).train()
    bond.requires_grad_(True).train()
    if args.freeze_object_features:
        freeze_proposal_object_features(atom, bond)
    set_head.requires_grad_(False).eval()
    proposal_config = PredictedProposalConfig(**source["proposal_config"])
    scene_config = AtomBondSceneConfig(
        atom_queries=proposal_config.maximum_atoms,
        bond_queries=proposal_config.maximum_bonds,
    )
    collator = AtomBondSceneCollator(scene_config)
    validation_records = {}
    validation_audits = {}
    for name, shard in VALIDATION_DOMAINS.items():
        records, audit = load_validation_records(
            [shard],
            per_shard=args.validation_samples_per_domain,
            config=scene_config,
        )
        validation_records[name] = records
        validation_audits[name] = audit
    (output_dir / "validation_audit.json").write_text(
        json.dumps(validation_audits, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    atom_threshold = float(source["training_args"]["atom_decision_threshold"])
    bond_threshold = float(source["training_args"]["bond_decision_threshold"])

    def validate() -> dict:
        return add_proposal_summary(
            domain_validation(
                vision,
                atom,
                bond,
                set_head,
                validation_records,
                collator,
                proposal_config,
                device,
                args.batch_size,
                atom_threshold,
                bond_threshold,
            )
        )

    initial = validate()
    (output_dir / "initial_metrics.json").write_text(
        json.dumps(initial, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"initial_validation": initial}, separators=(",", ":")), flush=True)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": [parameter for parameter in atom.parameters() if parameter.requires_grad],
                "lr": args.atom_learning_rate,
            },
            {
                "params": [parameter for parameter in bond.parameters() if parameter.requires_grad],
                "lr": args.bond_learning_rate,
            },
        ],
        weight_decay=0.0,
    )
    trainable = [
        parameter
        for module in (atom, bond)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    total_seen = 0
    next_log = args.log_every_total_samples
    next_validation = args.validate_every_total_samples
    running = Counter()
    running_samples = 0
    best_score = None
    best_metrics = None
    started = time.perf_counter()

    for samples, domain_seen in balanced_batches(
        samples_per_domain=args.samples_per_domain,
        batch_size=args.batch_size,
        config=scene_config,
        seed=args.seed,
        shuffle_buffer=args.shuffle_buffer,
    ):
        atom.train()
        bond.train()
        batch = collator(samples)
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        targets = stack_targets(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            features = vision.forward_stroke_resolution(pixels)
            teacher_atom_output = teacher_atom(features) if teacher_atom is not None else None
            teacher_bond_output = (
                forward_frozen_bond_head(vision, teacher_bond, features)
                if teacher_bond is not None
                else None
            )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            atom_output = atom(features)
            bond_output = forward_frozen_bond_head(vision, bond, features)
            atom_losses = spatial_atom_object_loss(atom_output, targets)
            bond_losses = dense_bond_object_loss(
                bond_output,
                bond_targets(targets),
                heatmap_set_weight=0.1,
                slot_matching="hungarian",
                cell_count_weight=0.05,
                duplicate_weight=0.02,
            )
            atom_feature_loss = (
                F.mse_loss(
                    atom_output["object_features"].float(),
                    teacher_atom_output["object_features"].float(),
                )
                if teacher_atom_output is not None
                else atom_losses["loss"] * 0.0
            )
            bond_feature_loss = (
                F.mse_loss(
                    bond_output["object_features"].float(),
                    teacher_bond_output["object_features"].float(),
                )
                if teacher_bond_output is not None
                else bond_losses["loss"] * 0.0
            )
            loss = (
                args.atom_loss_weight * atom_losses["loss"]
                + args.bond_loss_weight * bond_losses["loss"]
                + args.feature_distillation_weight
                * (atom_feature_loss + bond_feature_loss)
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        count = len(samples)
        total_seen += count
        running_samples += count
        for name, value in (
            ("loss", loss),
            ("atom_loss", atom_losses["loss"]),
            ("bond_loss", bond_losses["loss"]),
            ("atom_feature_loss", atom_feature_loss),
            ("bond_feature_loss", bond_feature_loss),
        ):
            running[name] += float(value.detach()) * count

        if total_seen >= next_log:
            print(
                json.dumps(
                    {
                        "total_seen": total_seen,
                        "domain_seen": domain_seen,
                        **{name: value / running_samples for name, value in running.items()},
                        "elapsed_seconds": time.perf_counter() - started,
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
            running.clear()
            running_samples = 0
            next_log += args.log_every_total_samples

        if total_seen >= next_validation or min(domain_seen.values()) >= args.samples_per_domain:
            metrics = validate()
            metrics.update(
                {
                    "total_seen": total_seen,
                    "domain_seen": domain_seen,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            print(json.dumps({"validation": metrics}, separators=(",", ":")), flush=True)
            checkpoint = {
                **{
                    key: value
                    for key, value in source.items()
                    if key not in {"optimizer", "validation", "atom_head", "bond_head"}
                },
                "checkpoint_schema_version": "joint-proposal-objects-checkpoint-0.5",
                "atom_head": atom.state_dict(),
                "atom_config": asdict(atom.config),
                "bond_head": bond.state_dict(),
                "bond_config": asdict(bond.config),
                "optimizer": optimizer.state_dict(),
                "samples_seen": int(source["samples_seen"]) + total_seen,
                "multidomain_proposal_samples": total_seen,
                "validation": metrics,
                "source_joint_checkpoint": args.source_checkpoint,
                "multidomain_proposal_args": vars(args),
            }
            torch.save(checkpoint, output_dir / f"samples-{total_seen:06d}.pt")
            torch.save(checkpoint, output_dir / "last.pt")
            score = proposal_checkpoint_score(metrics)
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
            next_validation += args.validate_every_total_samples


if __name__ == "__main__":
    main()
