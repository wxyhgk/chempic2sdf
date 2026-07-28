#!/usr/bin/env python3
"""Jointly train Bond proposals and set/incidence on balanced rendering domains."""

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
from chemvector.dense_bond_object import dense_bond_object_loss
from chemvector.predicted_incidence import (
    PredictedProposalConfig,
    associate_predicted_proposals,
    decode_predicted_atom_proposals,
    decode_predicted_bond_proposals,
)
from chemvector.proposal_object_set import proposal_object_set_loss
from chemvector.train_atom_bond_scene import AtomBondSceneCollator, stack_targets
from chemvector.train_bond_presence_calibration import load_components
from chemvector.train_joint_proposal_objects_full import (
    DEFAULT_TRAIN_SHARDS,
    bond_targets,
    load_validation_records,
)
from chemvector.train_spatial_incidence import DEFAULT_GRAPH, forward_frozen_bond_head
from chemvector.train_stable_bond_presence import (
    CHEMVISION_TRAIN_SHARDS,
    domain_validation,
    valid_stream,
)


DEFAULT_SOURCE = (
    "outputs/checkpoints/chemvector-v0972-bond-presence-calibration-10k/best.pt"
)
DEFAULT_OUTPUT = (
    "outputs/checkpoints/chemvector-v0999-joint-bond-set-multidomain-engineering-smoke"
)
FONT_TRAIN_SHARDS = [
    f"Data/sdf/chemvision_layers_v0_2/font_pilot_10k/shards/train-{index:06d}.tar"
    for index in range(3)
]
TRAIN_DOMAINS = {
    "chemvision": CHEMVISION_TRAIN_SHARDS,
    "font_v2": FONT_TRAIN_SHARDS,
    "full_v0": DEFAULT_TRAIN_SHARDS,
}
VALIDATION_DOMAINS = {
    "chemvision": (
        "Data/sdf/chemvision_layers_v0_1/pilot_10k/shards/validation-000000.tar"
    ),
    "font_v2": (
        "Data/sdf/chemvision_layers_v0_2/font_pilot_10k/shards/validation-000000.tar"
    ),
    "full_v0": "Data/sdf/webdataset_full_v0/shards/validation-000000.tar",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", default=DEFAULT_SOURCE)
    parser.add_argument("--grounded-graph-checkpoint", default=DEFAULT_GRAPH)
    parser.add_argument("--samples-per-domain", type=int, default=2)
    parser.add_argument("--validation-samples-per-domain", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--bond-learning-rate", type=float, default=1e-6)
    parser.add_argument("--presence-learning-rate", type=float, default=2e-6)
    parser.add_argument("--incidence-learning-rate", type=float, default=2e-7)
    parser.add_argument("--bond-loss-weight", type=float, default=1.0)
    parser.add_argument("--set-loss-weight", type=float, default=1.0)
    parser.add_argument("--graph-exact-surrogate-weight", type=float, default=0.1)
    parser.add_argument("--shuffle-buffer", type=int, default=1024)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260728)
    return parser.parse_args()


def balanced_batches(
    *,
    samples_per_domain: int,
    batch_size: int,
    config: AtomBondSceneConfig,
    seed: int,
    shuffle_buffer: int,
):
    if batch_size % len(TRAIN_DOMAINS):
        raise ValueError("Batch size must be divisible by the domain count")
    per_batch = batch_size // len(TRAIN_DOMAINS)
    streams = {
        domain: iter(
            valid_stream(
                shards,
                config=config,
                seed=seed + offset,
                shuffle_buffer=shuffle_buffer,
            )
        )
        for offset, (domain, shards) in enumerate(TRAIN_DOMAINS.items())
    }
    seen = {domain: 0 for domain in streams}
    while min(seen.values()) < samples_per_domain:
        rows = []
        for domain, stream in streams.items():
            take = min(per_batch, samples_per_domain - seen[domain])
            for _ in range(take):
                try:
                    rows.append(next(stream))
                except StopIteration as error:
                    raise RuntimeError(
                        f"Joint train domain {domain} exhausted at {seen[domain]}"
                    ) from error
            seen[domain] += take
        if rows:
            yield rows, dict(seen)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Joint Bond/set engineering training requires CUDA")
    if min(args.samples_per_domain, args.batch_size) < 1:
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
        json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    vision, atom, bond, set_head, source = load_components(
        args.source_checkpoint, args.grounded_graph_checkpoint, device
    )
    vision.eval().requires_grad_(False)
    atom.eval().requires_grad_(False)
    bond.train().requires_grad_(True)
    set_head.train().requires_grad_(True)
    proposal_config = PredictedProposalConfig(**source["proposal_config"])
    scene_config = AtomBondSceneConfig(
        atom_queries=proposal_config.maximum_atoms,
        bond_queries=proposal_config.maximum_bonds,
    )
    collator = AtomBondSceneCollator(scene_config)
    validation_records = {}
    validation_audit = {}
    for domain, shard in VALIDATION_DOMAINS.items():
        validation_records[domain], validation_audit[domain] = load_validation_records(
            [shard],
            per_shard=args.validation_samples_per_domain,
            config=scene_config,
        )
    (output_dir / "validation_audit.json").write_text(
        json.dumps(validation_audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    atom_threshold = float(source["training_args"]["atom_decision_threshold"])
    bond_threshold = float(source["training_args"]["bond_decision_threshold"])

    def validate() -> dict:
        return domain_validation(
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

    initial = validate()
    (output_dir / "initial_metrics.json").write_text(
        json.dumps(initial, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    incidence_parameters = list(set_head.incidence.parameters())
    incidence_ids = {id(parameter) for parameter in incidence_parameters}
    presence_parameters = [
        parameter for parameter in set_head.parameters()
        if id(parameter) not in incidence_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": bond.parameters(), "lr": args.bond_learning_rate},
            {"params": presence_parameters, "lr": args.presence_learning_rate},
            {"params": incidence_parameters, "lr": args.incidence_learning_rate},
        ],
        weight_decay=0.0,
    )
    trainable = list(bond.parameters()) + list(set_head.parameters())
    totals = defaultdict(float)
    total_seen = 0
    steps = 0
    started = time.perf_counter()

    for samples, domain_seen in balanced_batches(
        samples_per_domain=args.samples_per_domain,
        batch_size=args.batch_size,
        config=scene_config,
        seed=args.seed,
        shuffle_buffer=args.shuffle_buffer,
    ):
        bond.train()
        set_head.train()
        batch = collator(samples)
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        targets = stack_targets(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            features = vision.forward_stroke_resolution(pixels)
            atom_output = atom(features)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            bond_output = forward_frozen_bond_head(vision, bond, features)
            proposals = associate_predicted_proposals(
                decode_predicted_atom_proposals(atom_output, proposal_config),
                decode_predicted_bond_proposals(bond_output, proposal_config),
                targets,
                proposal_config,
            )
            set_output = set_head(
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
            bond_losses = dense_bond_object_loss(
                bond_output,
                bond_targets(targets),
                heatmap_set_weight=0.1,
                slot_matching="hungarian",
                cell_count_weight=0.05,
                duplicate_weight=0.02,
            )
            set_losses = proposal_object_set_loss(
                set_output,
                proposals["atom_presence_targets"],
                proposals["bond_presence_targets"],
                proposals["atom_mask"],
                proposals["bond_mask"],
                proposals["target_atom_refs"],
                proposals["bond_supervision_mask"],
                graph_exact_surrogate_weight=args.graph_exact_surrogate_weight,
                atom_decision_threshold=atom_threshold,
                bond_decision_threshold=bond_threshold,
            )
            loss = (
                args.bond_loss_weight * bond_losses["loss"]
                + args.set_loss_weight * set_losses["loss"]
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        count = len(samples)
        total_seen += count
        steps += 1
        for name, value in (
            ("loss", loss),
            ("bond_loss", bond_losses["loss"]),
            ("set_loss", set_losses["loss"]),
            ("graph_exact_surrogate_loss", set_losses["graph_exact_surrogate_loss"]),
        ):
            totals[name] += float(value.detach())

    metrics = validate()
    checkpoint = {
        **{
            key: value
            for key, value in source.items()
            if key not in {"optimizer", "validation", "samples_seen", "bond_head", "set_head"}
        },
        "checkpoint_schema_version": "joint-proposal-objects-checkpoint-0.9",
        "bond_head": bond.state_dict(),
        "bond_config": asdict(bond.config),
        "bond_multiscale": hasattr(bond, "base"),
        "set_head": set_head.state_dict(),
        "set_config": {
            "incidence": asdict(set_head.config.incidence),
            "hidden_dim": set_head.config.hidden_dim,
            "stable_bond_hidden_dim": set_head.config.stable_bond_hidden_dim,
        },
        "optimizer": optimizer.state_dict(),
        "samples_seen": int(source["samples_seen"]) + total_seen,
        "joint_multidomain_samples": total_seen,
        "validation": metrics,
        "initial_validation": initial,
        "joint_multidomain_args": vars(args),
    }
    torch.save(checkpoint, output_dir / "last.pt")
    report = {
        "engineering_only": True,
        "fresh_dev_used": False,
        "total_seen": total_seen,
        "domain_seen": domain_seen,
        "mean_losses": {name: value / max(steps, 1) for name, value in totals.items()},
        "initial": initial,
        "final": metrics,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
