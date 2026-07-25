#!/usr/bin/env python3
"""Train a geometry/incidence Bond presence adapter on balanced render domains."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import random
import time

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.data import ChemVectorIterableDataset
from chemvector.predicted_incidence import (
    PredictedProposalConfig,
    associate_predicted_proposals,
    decode_predicted_atom_proposals,
    decode_predicted_bond_proposals,
)
from chemvector.proposal_object_set import (
    ProposalObjectSetHead,
    proposal_object_set_loss,
)
from chemvector.train_atom_bond_scene import AtomBondSceneCollator, stack_targets
from chemvector.train_bond_presence_calibration import load_components
from chemvector.train_joint_proposal_objects_full import (
    DEFAULT_TRAIN_SHARDS,
    load_validation_records,
    rejection_reason,
    sample_split,
)
from chemvector.train_proposal_object_set import evaluate
from chemvector.train_spatial_incidence import DEFAULT_GRAPH, forward_frozen_bond_head


CHEMVISION_TRAIN_SHARDS = [
    f"Data/sdf/chemvision_layers_v0_1/pilot_10k/shards/train-{index:06d}.tar"
    for index in range(3)
]
VALIDATION_DOMAINS = {
    "chemvision": (
        "Data/sdf/chemvision_layers_v0_1/pilot_10k/shards/validation-000000.tar"
    ),
    "full_v0": "Data/sdf/webdataset_full_v0/shards/validation-000000.tar",
}
DEFAULT_SOURCE = (
    "outputs/checkpoints/chemvector-v0972-bond-presence-calibration-10k/best.pt"
)
DEFAULT_OUTPUT = "outputs/checkpoints/chemvector-v0973-stable-bond-presence-balanced16k"


def valid_stream(
    shards: list[str],
    *,
    config: AtomBondSceneConfig,
    seed: int,
    shuffle_buffer: int,
):
    ordered = list(shards)
    random.Random(seed).shuffle(ordered)
    dataset = ChemVectorIterableDataset(
        ordered,
        sequence_version="0.4",
        sample_shuffle_buffer=shuffle_buffer,
        sample_shuffle_seed=seed,
    )
    for sample in dataset:
        if sample_split(sample) != "train":
            raise ValueError("Balanced stream contains a non-train sample")
        if rejection_reason(sample, config) is None:
            yield sample


def balanced_batches(
    *,
    samples_per_domain: int,
    batch_size: int,
    config: AtomBondSceneConfig,
    seed: int,
    shuffle_buffer: int,
):
    if batch_size % 2:
        raise ValueError("Balanced batch size must be even")
    half = batch_size // 2
    streams = {
        "chemvision": iter(
            valid_stream(
                CHEMVISION_TRAIN_SHARDS,
                config=config,
                seed=seed,
                shuffle_buffer=shuffle_buffer,
            )
        ),
        "full_v0": iter(
            valid_stream(
                DEFAULT_TRAIN_SHARDS,
                config=config,
                seed=seed + 1,
                shuffle_buffer=shuffle_buffer,
            )
        ),
    }
    seen = {name: 0 for name in streams}
    while min(seen.values()) < samples_per_domain:
        batch = []
        for name, stream in streams.items():
            remaining = samples_per_domain - seen[name]
            take = min(half, remaining)
            for _ in range(take):
                try:
                    batch.append(next(stream))
                except StopIteration as error:
                    raise ValueError(
                        f"Balanced domain {name} exhausted at {seen[name]} samples"
                    ) from error
            seen[name] += take
        if batch:
            yield batch, dict(seen)


def domain_validation(
    vision,
    atom,
    bond,
    set_head,
    records: dict[str, list[dict]],
    collator,
    proposal_config,
    device,
    batch_size,
    atom_threshold,
    bond_threshold,
) -> dict:
    domains = {
        name: evaluate(
            vision,
            atom,
            bond,
            set_head,
            rows,
            collator,
            proposal_config,
            device,
            batch_size,
            atom_threshold,
            bond_threshold,
        )
        for name, rows in records.items()
    }
    samples = sum(item["samples"] for item in domains.values())
    pooled = {}
    for count_name, rate_name in (
        ("atom_deployed_exact", "atom_deployed_exact_rate"),
        ("bond_deployed_exact", "bond_deployed_exact_rate"),
        ("incidence_exact", "incidence_exact_rate"),
        ("full_object_exact", "full_object_exact_rate"),
    ):
        count = sum(item[count_name] for item in domains.values())
        pooled[count_name] = count
        pooled[rate_name] = count / samples
    pooled["samples"] = samples
    return {
        "domains": domains,
        "pooled": pooled,
        "minimum_domain_full_object_exact_rate": min(
            item["full_object_exact_rate"] for item in domains.values()
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", default=DEFAULT_SOURCE)
    parser.add_argument("--grounded-graph-checkpoint", default=DEFAULT_GRAPH)
    parser.add_argument("--samples-per-domain", type=int, default=8_000)
    parser.add_argument("--validation-samples-per-domain", type=int, default=498)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--stable-hidden-dim", type=int, default=64)
    parser.add_argument("--validate-every-total-samples", type=int, default=4_000)
    parser.add_argument("--log-every-total-samples", type=int, default=1_000)
    parser.add_argument("--shuffle-buffer", type=int, default=1024)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Stable Bond presence training requires CUDA")
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

    vision, atom, bond, base_set_head, source = load_components(
        args.source_checkpoint, args.grounded_graph_checkpoint, device
    )
    stable_config = replace(
        base_set_head.config,
        stable_bond_hidden_dim=args.stable_hidden_dim,
    )
    set_head = ProposalObjectSetHead(stable_config).to(device)
    missing, unexpected = set_head.load_state_dict(base_set_head.state_dict(), strict=False)
    if unexpected or any(not name.startswith("stable_bond_presence.") for name in missing):
        raise ValueError(f"Stable adapter warmstart differs: missing={missing} unexpected={unexpected}")
    set_head.requires_grad_(False)
    set_head.stable_bond_presence.requires_grad_(True)
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
    initial = domain_validation(
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
    (output_dir / "initial_metrics.json").write_text(
        json.dumps(initial, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"initial_validation": initial}, separators=(",", ":")), flush=True)

    trainable = list(set_head.stable_bond_presence.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.0)
    total_seen = 0
    next_log = args.log_every_total_samples
    next_validation = args.validate_every_total_samples
    running_loss = 0.0
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
        set_head.train()
        batch = collator(samples)
        pixels = batch["pixel_values"].to(device)
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
        total_seen += count
        running_samples += count
        running_loss += float(losses["loss"].detach()) * count
        if total_seen >= next_log:
            print(
                json.dumps(
                    {
                        "total_seen": total_seen,
                        "domain_seen": domain_seen,
                        "loss": running_loss / running_samples,
                        "elapsed_seconds": time.perf_counter() - started,
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
            running_loss = 0.0
            running_samples = 0
            next_log += args.log_every_total_samples
        if total_seen >= next_validation or min(domain_seen.values()) >= args.samples_per_domain:
            metrics = domain_validation(
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
                    if key not in {"optimizer", "validation", "samples_seen", "set_head", "set_config"}
                },
                "checkpoint_schema_version": "joint-proposal-objects-checkpoint-0.3",
                "set_head": set_head.state_dict(),
                "set_config": {
                    "incidence": asdict(stable_config.incidence),
                    "hidden_dim": stable_config.hidden_dim,
                    "stable_bond_hidden_dim": stable_config.stable_bond_hidden_dim,
                },
                "optimizer": optimizer.state_dict(),
                "samples_seen": int(source["samples_seen"]) + total_seen,
                "stable_adapter_samples": total_seen,
                "validation": metrics,
                "source_joint_checkpoint": args.source_checkpoint,
                "stable_adapter_args": vars(args),
            }
            torch.save(checkpoint, output_dir / f"samples-{total_seen:06d}.pt")
            torch.save(checkpoint, output_dir / "last.pt")
            score = (
                metrics["minimum_domain_full_object_exact_rate"],
                metrics["pooled"]["full_object_exact_rate"],
                metrics["pooled"]["bond_deployed_exact_rate"],
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
            next_validation += args.validate_every_total_samples


if __name__ == "__main__":
    main()
