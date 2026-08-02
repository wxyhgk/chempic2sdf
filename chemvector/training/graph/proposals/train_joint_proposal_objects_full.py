#!/usr/bin/env python3
"""Jointly train spatial Atom/Bond proposals and proposal-set incidence at scale."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import torch

from Data.sdf.chemscene_v0_2 import build_chemscene_v0_2
from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.data import ChemVectorIterableDataset
from chemvector.dense_bond_object import dense_bond_object_loss
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
from chemvector.spatial_atom_object import spatial_atom_object_loss
from chemvector.spatial_incidence import SpatialIncidenceConfig
from chemvector.train_atom_bond_scene import AtomBondSceneCollator, stack_targets
from chemvector.train_proposal_object_set import evaluate
from chemvector.train_spatial_atom_object_full import spatial_cell_collision_count
from chemvector.train_spatial_incidence import (
    DEFAULT_GRAPH,
    forward_frozen_bond_head,
    load_frozen_components,
)


DEFAULT_TRAIN_SHARDS = [
    str(path)
    for path in sorted(
        Path("Data/sdf/webdataset_full_v0/shards").glob("train-[0-9]*.tar")
    )[:50]
]
DEFAULT_VALIDATION_SHARDS = [
    "Data/sdf/chemvision_layers_v0_1/pilot_10k/shards/validation-000000.tar",
    "Data/sdf/webdataset_full_v0/shards/validation-000000.tar",
]
DEFAULT_ATOM = (
    "outputs/checkpoints/chemvector-v0967-atom-object-full8967-step2000-rerun/"
    "step-000250.pt"
)
DEFAULT_BOND = (
    "outputs/checkpoints/chemvector-v082-multiscale-joint-9k-1epoch/best.pt"
)
DEFAULT_SET = (
    "outputs/checkpoints/chemvector-v0962-representable-hard-replay-step1500/best.pt"
)
DEFAULT_OUTPUT = "outputs/checkpoints/chemvector-v0971-joint-proposal-objects-full50k"


def sample_split(sample: dict) -> str | None:
    meta = sample.get("meta", {})
    return meta.get("source_split", meta.get("split"))


def rejection_reason(sample: dict, config: AtomBondSceneConfig) -> str | None:
    if len(sample["chemmol"]["atoms"]) > config.atom_queries:
        return "atom_capacity"
    if len(sample["chemmol"]["bonds"]) > config.bond_queries:
        return "bond_capacity"
    if spatial_cell_collision_count(sample):
        return "atom_cell_collision"
    try:
        build_chemscene_v0_2(
            sample["chemmol"], sample["render"], sample.get("layers")
        )
    except Exception:
        return "chemscene_protocol"
    return None


def load_validation_records(
    shards: list[str],
    *,
    per_shard: int,
    config: AtomBondSceneConfig,
) -> tuple[list[dict], dict]:
    records = []
    rejected = Counter()
    source_counts = Counter()
    for shard in shards:
        accepted = 0
        for sample in ChemVectorIterableDataset([shard], sequence_version="0.4"):
            if sample_split(sample) != "validation":
                raise ValueError(f"Validation shard contains non-validation sample: {shard}")
            reason = rejection_reason(sample, config)
            if reason:
                rejected[reason] += 1
                continue
            records.append(sample)
            source_counts[Path(shard).name] += 1
            accepted += 1
            if accepted >= per_shard:
                break
        if accepted < per_shard:
            raise ValueError(
                f"Validation shard {shard} yielded {accepted}, expected {per_shard}"
            )
    return records, {
        "accepted": len(records),
        "per_shard": per_shard,
        "source_counts": dict(source_counts),
        "rejected": dict(rejected),
    }


def training_batches(
    shards: list[str],
    *,
    samples: int,
    batch_size: int,
    seed: int,
    shuffle_buffer: int,
    config: AtomBondSceneConfig,
    audit: dict,
):
    rng = random.Random(seed)
    ordered = list(shards)
    rng.shuffle(ordered)
    dataset = ChemVectorIterableDataset(
        ordered,
        sequence_version="0.4",
        sample_shuffle_buffer=shuffle_buffer,
    )
    batch = []
    accepted = 0
    rejected = Counter()
    seen_keys = set()
    for sample in dataset:
        if sample_split(sample) != "train":
            raise ValueError("Training stream contains a non-train sample")
        if sample["key"] in seen_keys:
            raise ValueError(f"Training stream contains duplicate key: {sample['key']}")
        seen_keys.add(sample["key"])
        reason = rejection_reason(sample, config)
        if reason:
            rejected[reason] += 1
            continue
        batch.append(sample)
        accepted += 1
        if len(batch) == batch_size:
            yield batch
            batch = []
        if accepted >= samples:
            break
    if batch:
        yield batch
    audit.update(
        {
            "requested": samples,
            "accepted": accepted,
            "unique_keys": len(seen_keys),
            "rejected": dict(rejected),
        }
    )
    if accepted != samples:
        raise ValueError(f"Training stream yielded {accepted}, expected {samples}")


def bond_targets(targets: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "endpoints": targets["bond_ports"],
        "bbox": targets["bond_ports"].new_zeros(
            (*targets["bond_ports"].shape[:2], 4)
        ),
        "display": targets["bond_display"],
        "chemical": targets["bond_chemical"],
        "direction": targets["bond_direction"],
        "mask": targets["bond_mask"],
    }


def load_set_head(path: str, device: torch.device) -> tuple[ProposalObjectSetHead, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    incidence = SpatialIncidenceConfig(**payload["config"]["incidence"])
    config = ProposalObjectSetConfig(
        incidence=incidence,
        hidden_dim=payload["config"]["hidden_dim"],
    )
    head = ProposalObjectSetHead(config).to(device)
    head.load_state_dict(payload["head"], strict=True)
    return head, payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-shards", nargs="+", default=DEFAULT_TRAIN_SHARDS)
    parser.add_argument(
        "--validation-shards", nargs="+", default=DEFAULT_VALIDATION_SHARDS
    )
    parser.add_argument("--train-samples", type=int, default=50_000)
    parser.add_argument("--validation-samples-per-shard", type=int, default=498)
    parser.add_argument("--grounded-graph-checkpoint", default=DEFAULT_GRAPH)
    parser.add_argument("--atom-checkpoint", default=DEFAULT_ATOM)
    parser.add_argument("--bond-checkpoint", default=DEFAULT_BOND)
    parser.add_argument("--set-checkpoint", default=DEFAULT_SET)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--maximum-atoms", type=int, default=128)
    parser.add_argument("--maximum-bonds", type=int, default=192)
    parser.add_argument("--atom-threshold", type=float, default=0.01)
    parser.add_argument("--bond-threshold", type=float, default=0.01)
    parser.add_argument("--atom-match-radius-px", type=float, default=6.0)
    parser.add_argument("--bond-match-radius-px", type=float, default=12.0)
    parser.add_argument("--atom-decision-threshold", type=float, default=-0.25)
    parser.add_argument("--bond-decision-threshold", type=float, default=-0.5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--atom-learning-rate", type=float, default=2e-5)
    parser.add_argument("--bond-learning-rate", type=float, default=1e-5)
    parser.add_argument("--presence-learning-rate", type=float, default=5e-5)
    parser.add_argument("--incidence-learning-rate", type=float, default=5e-6)
    parser.add_argument("--atom-loss-weight", type=float, default=1.0)
    parser.add_argument("--bond-loss-weight", type=float, default=1.0)
    parser.add_argument("--set-loss-weight", type=float, default=1.0)
    parser.add_argument("--validate-every-samples", type=int, default=10_000)
    parser.add_argument("--log-every-samples", type=int, default=1_000)
    parser.add_argument("--shuffle-buffer", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Joint proposal-object training requires CUDA")
    if min(args.train_samples, args.batch_size) < 1:
        raise ValueError("Training sample and batch counts must be positive")
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

    scene_config = AtomBondSceneConfig(
        atom_queries=args.maximum_atoms,
        bond_queries=args.maximum_bonds,
    )
    collator = AtomBondSceneCollator(scene_config)
    validation_records, validation_audit = load_validation_records(
        args.validation_shards,
        per_shard=args.validation_samples_per_shard,
        config=scene_config,
    )
    (output_dir / "validation_audit.json").write_text(
        json.dumps(validation_audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    vision, atom_head, bond_head, sources = load_frozen_components(
        Path(args.grounded_graph_checkpoint),
        Path(args.atom_checkpoint),
        Path(args.bond_checkpoint),
        device,
    )
    atom_head.requires_grad_(True).train()
    bond_head.requires_grad_(True).train()
    set_head, set_payload = load_set_head(args.set_checkpoint, device)
    set_head.train()
    proposal_config = PredictedProposalConfig(
        maximum_atoms=args.maximum_atoms,
        maximum_bonds=args.maximum_bonds,
        atom_threshold=args.atom_threshold,
        bond_threshold=args.bond_threshold,
        atom_match_radius_px=args.atom_match_radius_px,
        bond_match_radius_px=args.bond_match_radius_px,
    )
    incidence_parameters = list(set_head.incidence.parameters())
    incidence_ids = {id(parameter) for parameter in incidence_parameters}
    presence_parameters = [
        parameter
        for parameter in set_head.parameters()
        if id(parameter) not in incidence_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": atom_head.parameters(), "lr": args.atom_learning_rate},
            {"params": bond_head.parameters(), "lr": args.bond_learning_rate},
            {"params": presence_parameters, "lr": args.presence_learning_rate},
            {"params": incidence_parameters, "lr": args.incidence_learning_rate},
        ],
        weight_decay=0.0,
    )
    trainable = [
        parameter
        for module in (atom_head, bond_head, set_head)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]

    initial = evaluate(
        vision,
        atom_head,
        bond_head,
        set_head,
        validation_records,
        collator,
        proposal_config,
        device,
        args.batch_size,
        args.atom_decision_threshold,
        args.bond_decision_threshold,
    )
    (output_dir / "initial_metrics.json").write_text(
        json.dumps(initial, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps({"initial_validation": initial}, separators=(",", ":")),
        flush=True,
    )

    training_audit: dict = {}
    batches = training_batches(
        args.train_shards,
        samples=args.train_samples,
        batch_size=args.batch_size,
        seed=args.seed,
        shuffle_buffer=args.shuffle_buffer,
        config=scene_config,
        audit=training_audit,
    )
    running = Counter()
    running_samples = 0
    samples_seen = 0
    best_score = None
    best_metrics = None
    started = time.perf_counter()
    next_log = args.log_every_samples
    next_validation = args.validate_every_samples

    for samples in batches:
        atom_head.train()
        bond_head.train()
        set_head.train()
        batch = collator(samples)
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        targets = stack_targets(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            features = vision.forward_stroke_resolution(pixels)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            atom_output = atom_head(features)
            bond_output = forward_frozen_bond_head(vision, bond_head, features)
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
            atom_losses = spatial_atom_object_loss(atom_output, targets)
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
                atom_decision_threshold=args.atom_decision_threshold,
                bond_decision_threshold=args.bond_decision_threshold,
            )
            loss = (
                args.atom_loss_weight * atom_losses["loss"]
                + args.bond_loss_weight * bond_losses["loss"]
                + args.set_loss_weight * set_losses["loss"]
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        count = len(samples)
        samples_seen += count
        running_samples += count
        for name, value in (
            ("loss", loss),
            ("atom_loss", atom_losses["loss"]),
            ("bond_loss", bond_losses["loss"]),
            ("set_loss", set_losses["loss"]),
            ("endpoint_accuracy", set_losses["endpoint_accuracy"]),
        ):
            running[name] += float(value.detach()) * count

        if samples_seen >= next_log:
            report = {
                "samples_seen": samples_seen,
                **{name: total / running_samples for name, total in running.items()},
                "elapsed_seconds": time.perf_counter() - started,
            }
            print(json.dumps(report, separators=(",", ":")), flush=True)
            running.clear()
            running_samples = 0
            next_log += args.log_every_samples

        if samples_seen >= next_validation or samples_seen >= args.train_samples:
            metrics = evaluate(
                vision,
                atom_head,
                bond_head,
                set_head,
                validation_records,
                collator,
                proposal_config,
                device,
                args.batch_size,
                args.atom_decision_threshold,
                args.bond_decision_threshold,
            )
            metrics.update(
                {
                    "samples_seen": samples_seen,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            print(
                json.dumps({"validation": metrics}, separators=(",", ":")),
                flush=True,
            )
            checkpoint = {
                "checkpoint_schema_version": "joint-proposal-objects-checkpoint-0.1",
                "atom_head": atom_head.state_dict(),
                "atom_config": asdict(atom_head.config),
                "bond_head": bond_head.state_dict(),
                "bond_config": asdict(bond_head.config),
                "bond_multiscale": hasattr(bond_head, "base"),
                "set_head": set_head.state_dict(),
                "set_config": {
                    "incidence": asdict(set_head.config.incidence),
                    "hidden_dim": set_head.config.hidden_dim,
                },
                "proposal_config": asdict(proposal_config),
                "optimizer": optimizer.state_dict(),
                "samples_seen": samples_seen,
                "validation": metrics,
                "initial_validation": initial,
                "source_components": sources,
                "source_set_checkpoint": args.set_checkpoint,
                "source_set_evaluation": set_payload.get("evaluation"),
                "training_args": vars(args),
            }
            torch.save(checkpoint, output_dir / f"samples-{samples_seen:06d}.pt")
            torch.save(checkpoint, output_dir / "last.pt")
            score = (
                metrics["full_object_exact_rate"],
                metrics["atom_deployed_exact_rate"],
                metrics["bond_deployed_exact_rate"],
                metrics["incidence_exact_rate"],
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
        json.dumps(training_audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
