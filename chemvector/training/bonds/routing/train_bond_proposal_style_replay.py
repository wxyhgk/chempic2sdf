#!/usr/bin/env python3
"""Fine-tune only Bond proposals on paired styles with two-domain replay."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.audit_style_proposal_bottlenecks import (
    collect_rows,
    summarize_stage_rows,
)
from chemvector.dense_bond_object import (
    BondProposalResidualAdapter,
    BondProposalResidualConfig,
    MultiScaleDenseBondObjectDetector,
    dense_bond_object_loss,
)
from chemvector.evaluate_style_sensitivity import load_style_records
from chemvector.predicted_incidence import PredictedProposalConfig
from chemvector.train_atom_bond_scene import AtomBondSceneCollator, stack_targets
from chemvector.train_atom_pair_bond_group import TRAIN_DOMAINS, VALIDATION_DOMAINS
from chemvector.train_atom_pair_style_consistency import (
    load_paired_style_training_records,
    load_replay_records,
)
from chemvector.train_bond_presence_calibration import load_components
from chemvector.train_joint_proposal_objects_full import (
    bond_targets,
    load_validation_records,
)
from chemvector.train_spatial_incidence import DEFAULT_GRAPH, forward_frozen_bond_head
from chemvector.train_stable_bond_presence import domain_validation


DEFAULT_SOURCE = (
    "outputs/checkpoints/chemvector-v0972-bond-presence-calibration-10k/best.pt"
)
DEFAULT_STYLE_ROOT = Path("Data/sdf/chemvision_layers_v0_2/font_pilot_10k")
DEFAULT_OUTPUT = (
    "outputs/checkpoints/chemvector-v0995-bond-proposal-style-replay-pilot250"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", default=DEFAULT_SOURCE)
    parser.add_argument("--grounded-graph-checkpoint", default=DEFAULT_GRAPH)
    parser.add_argument("--style-root", default=str(DEFAULT_STYLE_ROOT))
    parser.add_argument("--style-molecules", type=int, default=250)
    parser.add_argument("--style-candidate-reserve", type=int, default=64)
    parser.add_argument("--replay-samples-per-domain", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--style-molecules-per-batch", type=int, default=2)
    parser.add_argument("--replay-batch-size", type=int, default=8)
    parser.add_argument("--validation-samples-per-domain", type=int, default=498)
    parser.add_argument("--style-validation-samples", type=int, default=0)
    parser.add_argument("--validation-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--residual-hidden-channels", type=int, default=32)
    parser.add_argument("--maximum-endpoint-delta-pixels", type=float, default=16.0)
    parser.add_argument("--train-full-bond-head", action="store_true")
    parser.add_argument("--replay-weight", type=float, default=1.0)
    parser.add_argument("--heatmap-set-weight", type=float, default=0.1)
    parser.add_argument("--cell-count-weight", type=float, default=0.05)
    parser.add_argument("--duplicate-weight", type=float, default=0.02)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260725)
    return parser.parse_args()


def selection_key(
    official: dict,
    style: dict,
    baseline_official: dict,
) -> tuple[float, ...]:
    """Prefer style coverage only when every official domain is non-regressing."""

    deltas = []
    for domain, baseline in baseline_official["domains"].items():
        candidate = official["domains"][domain]
        deltas.extend(
            (
                candidate["bond_proposal_recall"]
                - baseline["bond_proposal_recall"],
                float(
                    candidate["bond_deployed_exact"]
                    - baseline["bond_deployed_exact"]
                ),
                float(
                    candidate["full_object_exact"]
                    - baseline["full_object_exact"]
                ),
            )
        )
    paired = style["paired"]["bond_coverage"]
    overall = style["overall"]
    safe = min(deltas) >= -1e-12
    return (
        float(safe),
        float(paired["all_renderings"]) if safe else min(deltas),
        -float(paired["mixed"]),
        float(overall["bond_coverage"]),
        float(overall["representable"]),
        min(
            metrics["bond_proposal_recall"]
            for metrics in official["domains"].values()
        ),
    )


@torch.no_grad()
def validate_style(
    *,
    vision,
    atom,
    bond,
    set_head,
    records: list[dict],
    collator,
    proposal_config,
    device: torch.device,
    batch_size: int,
) -> dict:
    rows = collect_rows(
        vision=vision,
        atom=atom,
        bond=bond,
        set_head=set_head,
        records=records,
        collator=collator,
        proposal_config=proposal_config,
        device=device,
        batch_size=batch_size,
    )
    return summarize_stage_rows(rows)


def _bond_loss(output: dict, targets: dict, args: argparse.Namespace) -> dict:
    return dense_bond_object_loss(
        output,
        bond_targets(targets),
        heatmap_set_weight=args.heatmap_set_weight,
        slot_matching="hungarian",
        cell_count_weight=args.cell_count_weight,
        duplicate_weight=args.duplicate_weight,
    )


def _training_loss(
    samples: list[dict],
    *,
    vision,
    bond,
    collator,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    batch = collator(samples)
    pixels = batch["pixel_values"].to(device, non_blocking=True)
    targets = stack_targets(batch, device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        features = vision.forward_stroke_resolution(pixels)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = forward_frozen_bond_head(vision, bond, features)
        return _bond_loss(output, targets, args)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Bond proposal style/replay training requires CUDA")
    if args.replay_batch_size % len(TRAIN_DOMAINS):
        raise ValueError("Replay batch size must be divisible by the domain count")
    if min(args.style_molecules_per_batch, args.style_molecules, args.epochs) < 1:
        raise ValueError("Training sizes must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
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
    for module in (vision, atom, set_head):
        module.eval().requires_grad_(False)
    if args.train_full_bond_head:
        bond.train().requires_grad_(True)
    else:
        bond = BondProposalResidualAdapter(
            bond,
            BondProposalResidualConfig(
                hidden_channels=args.residual_hidden_channels,
                maximum_endpoint_delta_pixels=args.maximum_endpoint_delta_pixels,
            ),
        ).to(device)
        bond.train()
    proposal_config = PredictedProposalConfig(**source["proposal_config"])
    scene_config = AtomBondSceneConfig(
        atom_queries=proposal_config.maximum_atoms,
        bond_queries=proposal_config.maximum_bonds,
    )
    collator = AtomBondSceneCollator(scene_config)

    style_training, style_training_audit = load_paired_style_training_records(
        Path(args.style_root),
        molecule_count=args.style_molecules,
        reserve=args.style_candidate_reserve,
        seed=args.seed,
        config=scene_config,
    )
    replay_records, replay_audit = load_replay_records(
        samples_per_domain=args.replay_samples_per_domain,
        seed=args.seed,
        config=scene_config,
    )
    style_validation, style_validation_audit = load_style_records(
        str(Path(args.style_root) / "shards" / "validation-000000.tar"),
        split="validation",
        samples=args.style_validation_samples,
        config=scene_config,
    )
    official_records = {}
    official_audit = {}
    for domain, shard in VALIDATION_DOMAINS.items():
        rows, audit = load_validation_records(
            [shard],
            per_shard=args.validation_samples_per_domain,
            config=scene_config,
        )
        official_records[domain] = rows
        official_audit[domain] = audit
    (output_dir / "data_audit.json").write_text(
        json.dumps(
            {
                "style_training": style_training_audit,
                "replay": replay_audit,
                "style_validation": style_validation_audit,
                "official_validation": official_audit,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    atom_threshold = float(source["training_args"]["atom_decision_threshold"])
    bond_threshold = float(source["training_args"]["bond_decision_threshold"])

    def validate() -> dict:
        bond.eval()
        official = domain_validation(
            vision,
            atom,
            bond,
            set_head,
            official_records,
            collator,
            proposal_config,
            device,
            args.validation_batch_size,
            atom_threshold,
            bond_threshold,
        )
        style = validate_style(
            vision=vision,
            atom=atom,
            bond=bond,
            set_head=set_head,
            records=style_validation,
            collator=collator,
            proposal_config=proposal_config,
            device=device,
            batch_size=args.validation_batch_size,
        )
        return {"official": official, "style": style}

    initial = validate()
    (output_dir / "initial_metrics.json").write_text(
        json.dumps(initial, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"initial": initial}, separators=(",", ":")), flush=True)

    trainable = (
        list(bond.parameters())
        if args.train_full_bond_head
        else bond.adapter_parameters()
    )
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=0.0
    )
    baseline_official = initial["official"]
    best_key = selection_key(initial["official"], initial["style"], baseline_official)
    best_metrics = initial
    history = []
    total_style_seen = 0
    total_replay_seen = 0
    started = time.perf_counter()

    def checkpoint_payload(epoch: int, metrics: dict) -> dict:
        return {
            **{
                key: value
                for key, value in source.items()
                if key not in {"optimizer", "validation", "samples_seen", "bond_head"}
            },
            "checkpoint_schema_version": "joint-proposal-objects-checkpoint-0.6",
            "bond_head": bond.state_dict(),
            "bond_config": asdict(bond.config),
            "bond_multiscale": (
                isinstance(bond, MultiScaleDenseBondObjectDetector)
                or (
                    isinstance(bond, BondProposalResidualAdapter)
                    and isinstance(
                        bond.base_detector, MultiScaleDenseBondObjectDetector
                    )
                )
            ),
            **(
                {
                    "bond_proposal_residual_config": asdict(bond.residual_config),
                }
                if isinstance(bond, BondProposalResidualAdapter)
                else {}
            ),
            "optimizer": optimizer.state_dict(),
            "samples_seen": int(source["samples_seen"])
            + total_style_seen
            + total_replay_seen,
            "style_samples_seen": total_style_seen,
            "replay_samples_seen": total_replay_seen,
            "validation": metrics["official"],
            "style_validation": metrics["style"],
            "source_joint_checkpoint": args.source_checkpoint,
            "bond_style_replay_args": vars(args),
            "epoch": epoch,
        }

    torch.save(checkpoint_payload(0, initial), output_dir / "initial.pt")
    torch.save(checkpoint_payload(0, initial), output_dir / "best.pt")
    per_domain_replay = args.replay_batch_size // len(TRAIN_DOMAINS)

    for epoch in range(1, args.epochs + 1):
        style_order = list(range(len(style_training)))
        random.Random(args.seed + epoch).shuffle(style_order)
        replay_orders = {}
        for offset, (domain, rows) in enumerate(replay_records.items()):
            order = list(range(len(rows)))
            random.Random(args.seed + epoch * 17 + offset).shuffle(order)
            replay_orders[domain] = order
        replay_offsets = {domain: 0 for domain in replay_records}
        running = defaultdict(float)
        steps = 0
        bond.train()

        for start in range(0, len(style_order), args.style_molecules_per_batch):
            molecule_indices = style_order[
                start : start + args.style_molecules_per_batch
            ]
            if not molecule_indices:
                continue
            style_samples = [
                sample
                for molecule_index in molecule_indices
                for sample in style_training[molecule_index]
            ]
            replay_samples = []
            for domain, rows in replay_records.items():
                offset = replay_offsets[domain]
                order = replay_orders[domain]
                indices = [
                    order[(offset + index) % len(order)]
                    for index in range(per_domain_replay)
                ]
                replay_offsets[domain] = (offset + per_domain_replay) % len(order)
                replay_samples.extend(rows[index] for index in indices)

            style_losses = _training_loss(
                style_samples,
                vision=vision,
                bond=bond,
                collator=collator,
                device=device,
                args=args,
            )
            replay_losses = _training_loss(
                replay_samples,
                vision=vision,
                bond=bond,
                collator=collator,
                device=device,
                args=args,
            )
            loss = style_losses["loss"] + args.replay_weight * replay_losses["loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            total_style_seen += len(style_samples)
            total_replay_seen += len(replay_samples)
            steps += 1
            running["loss"] += float(loss.detach())
            for prefix, losses in (("style", style_losses), ("replay", replay_losses)):
                for name, value in losses.items():
                    running[f"{prefix}_{name}"] += float(value.detach())

        metrics = validate()
        key = selection_key(metrics["official"], metrics["style"], baseline_official)
        selected = key > best_key
        if selected:
            best_key = key
            best_metrics = metrics
            torch.save(checkpoint_payload(epoch, metrics), output_dir / "best.pt")
        torch.save(checkpoint_payload(epoch, metrics), output_dir / f"epoch-{epoch}.pt")
        torch.save(checkpoint_payload(epoch, metrics), output_dir / "last.pt")
        row = {
            "epoch": epoch,
            "selected_best": selected,
            "selection_key": list(key),
            "style_seen": total_style_seen,
            "replay_seen": total_replay_seen,
            "mean_losses": {
                name: value / max(steps, 1) for name, value in running.items()
            },
            **metrics,
            "elapsed_seconds": time.perf_counter() - started,
        }
        history.append(row)
        (output_dir / "metrics.json").write_text(
            json.dumps(
                {"initial": initial, "best": best_metrics, "history": history},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"epoch_result": row}, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
