#!/usr/bin/env python3
"""Train proposal-invariant AtomPair Bond presence on balanced domains."""

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
from chemvector.atom_pair_bond_group import (
    AtomPairBondGroupConfig,
    AtomPairBondGroupHead,
    atom_pair_group_loss,
    build_atom_pair_groups,
)
from chemvector.predicted_incidence import (
    PredictedProposalConfig,
    associate_predicted_proposals,
    decode_predicted_atom_proposals,
    decode_predicted_bond_proposals,
)
from chemvector.train_atom_bond_scene import AtomBondSceneCollator, stack_targets
from chemvector.train_bond_presence_calibration import load_components
from chemvector.train_joint_proposal_objects_full import (
    DEFAULT_TRAIN_SHARDS,
    load_validation_records,
)
from chemvector.train_spatial_incidence import DEFAULT_GRAPH, forward_frozen_bond_head
from chemvector.train_stable_bond_presence import (
    CHEMVISION_TRAIN_SHARDS,
    valid_stream,
)


DEFAULT_SOURCE = (
    "outputs/checkpoints/chemvector-v0972-bond-presence-calibration-10k/best.pt"
)
DEFAULT_OUTPUT = "outputs/checkpoints/chemvector-v0982-atom-pair-bond-group-two-domain18k"
TRAIN_DOMAINS = {
    "chemvision": CHEMVISION_TRAIN_SHARDS,
    "full_v0": DEFAULT_TRAIN_SHARDS,
}
VALIDATION_DOMAINS = {
    "chemvision": (
        "Data/sdf/chemvision_layers_v0_1/pilot_10k/shards/validation-000000.tar"
    ),
    "full_v0": "Data/sdf/webdataset_full_v0/shards/validation-000000.tar",
}


def balanced_two_domain_batches(
    *,
    samples_per_domain: int,
    batch_size: int,
    config: AtomBondSceneConfig,
    seed: int,
    shuffle_buffer: int,
    start_per_domain: int = 0,
):
    if batch_size % len(TRAIN_DOMAINS):
        raise ValueError("Batch size must be divisible by the number of domains")
    per_batch = batch_size // len(TRAIN_DOMAINS)
    streams = {
        name: iter(
            valid_stream(
                shards,
                config=config,
                seed=seed + offset,
                shuffle_buffer=shuffle_buffer,
            )
        )
        for offset, (name, shards) in enumerate(TRAIN_DOMAINS.items())
    }
    if not 0 <= start_per_domain <= samples_per_domain:
        raise ValueError("start_per_domain is outside the training range")
    for name, stream in streams.items():
        for _ in range(start_per_domain):
            try:
                next(stream)
            except StopIteration as error:
                raise ValueError(
                    f"Domain {name} exhausted while resuming at {start_per_domain}"
                ) from error
    seen = {name: start_per_domain for name in streams}
    while min(seen.values()) < samples_per_domain:
        batch = []
        for name, stream in streams.items():
            take = min(per_batch, samples_per_domain - seen[name])
            for _ in range(take):
                try:
                    batch.append(next(stream))
                except StopIteration as error:
                    raise ValueError(
                        f"Domain {name} exhausted at {seen[name]} samples"
                    ) from error
            seen[name] += take
        if batch:
            yield batch, dict(seen)


@torch.no_grad()
def frozen_groups(
    vision,
    atom,
    bond,
    set_head,
    pixels,
    targets,
    proposal_config,
):
    features = vision.forward_stroke_resolution(pixels)
    atom_output = atom(features)
    bond_output = forward_frozen_bond_head(vision, bond, features)
    proposals = associate_predicted_proposals(
        decode_predicted_atom_proposals(atom_output, proposal_config),
        decode_predicted_bond_proposals(bond_output, proposal_config),
        targets,
        proposal_config,
    )
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
    return proposals, build_atom_pair_groups(output, proposals, targets)


@torch.no_grad()
def validate(
    vision,
    atom,
    bond,
    set_head,
    group_head,
    records,
    collator,
    proposal_config,
    device,
    batch_size,
) -> dict:
    domains = {}
    for domain, rows in records.items():
        totals = defaultdict(int)
        for start in range(0, len(rows), batch_size):
            batch = collator(rows[start : start + batch_size])
            pixels = batch["pixel_values"].to(device)
            targets = stack_targets(batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                proposals, groups = frozen_groups(
                    vision, atom, bond, set_head, pixels, targets, proposal_config
                )
                group_output = group_head(groups["features"])
            predicted = group_output["presence_logits"] >= 0.0
            for batch_index in range(len(pixels)):
                sample = groups["sample_indices"] == batch_index
                expected = groups["targets"][sample]
                selected = predicted[sample]
                target_count = int(targets["bond_mask"][batch_index].sum())
                matched_count = int(
                    proposals["bond_target_matched"][batch_index].sum()
                )
                group_positive = int(expected.sum())
                candidate_exact = bool((selected == expected).all())
                deployed_exact = (
                    candidate_exact
                    and matched_count == target_count
                    and group_positive == target_count
                )
                totals["samples"] += 1
                totals["targets"] += target_count
                totals["matched_targets"] += matched_count
                totals["groups"] += int(sample.sum())
                totals["positive_groups"] += group_positive
                totals["selected_groups"] += int(selected.sum())
                totals["candidate_exact"] += int(candidate_exact)
                totals["deployed_exact"] += int(deployed_exact)
        samples = max(totals["samples"], 1)
        domains[domain] = {
            **dict(totals),
            "proposal_recall": totals["matched_targets"] / max(totals["targets"], 1),
            "candidate_exact_rate": totals["candidate_exact"] / samples,
            "deployed_exact_rate": totals["deployed_exact"] / samples,
        }
    return {
        "domains": domains,
        "minimum_domain_deployed_exact_rate": min(
            item["deployed_exact_rate"] for item in domains.values()
        ),
        "pooled_deployed_exact_rate": sum(
            item["deployed_exact"] for item in domains.values()
        )
        / sum(item["samples"] for item in domains.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", default=DEFAULT_SOURCE)
    parser.add_argument("--grounded-graph-checkpoint", default=DEFAULT_GRAPH)
    parser.add_argument("--samples-per-domain", type=int, default=8_967)
    parser.add_argument("--validation-samples-per-domain", type=int, default=498)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--validate-every-total-samples", type=int, default=4_500)
    parser.add_argument("--log-every-total-samples", type=int, default=900)
    parser.add_argument("--shuffle-buffer", type=int, default=1024)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("AtomPair Bond group training requires CUDA")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume_checkpoint:
        raise RuntimeError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_args.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    vision, atom, bond, set_head, source = load_components(
        args.source_checkpoint, args.grounded_graph_checkpoint, device
    )
    for module in (vision, atom, bond, set_head):
        module.eval().requires_grad_(False)
    group_config = AtomPairBondGroupConfig()
    group_head = AtomPairBondGroupHead(group_config).to(device)
    resume = None
    if args.resume_checkpoint:
        resume = torch.load(
            args.resume_checkpoint, map_location="cpu", weights_only=False
        )
        if resume["source_checkpoint"] != args.source_checkpoint:
            raise ValueError("Resume source checkpoint differs from current source")
        if resume["config"] != asdict(group_config):
            raise ValueError("Resume group-head config differs")
        group_head.load_state_dict(resume["head"], strict=True)
    proposal_config = PredictedProposalConfig(**source["proposal_config"])
    scene_config = AtomBondSceneConfig(
        atom_queries=proposal_config.maximum_atoms,
        bond_queries=proposal_config.maximum_bonds,
    )
    collator = AtomBondSceneCollator(scene_config)
    validation_records = {}
    for name, shard in VALIDATION_DOMAINS.items():
        rows, _ = load_validation_records(
            [shard], per_shard=args.validation_samples_per_domain, config=scene_config
        )
        validation_records[name] = rows
    if resume is None:
        initial = validate(
            vision, atom, bond, set_head, group_head, validation_records, collator,
            proposal_config, device, args.batch_size,
        )
        (output_dir / "initial_metrics.json").write_text(
            json.dumps(initial, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"initial_validation": initial}, separators=(",", ":")), flush=True)

    optimizer = torch.optim.AdamW(
        group_head.parameters(), lr=args.learning_rate, weight_decay=0.0
    )
    if resume is not None:
        optimizer.load_state_dict(resume["optimizer"])
    total_seen = int(resume["samples_seen"]) if resume is not None else 0
    if total_seen % len(TRAIN_DOMAINS):
        raise ValueError("Resume samples_seen is not balanced across domains")
    start_per_domain = total_seen // len(TRAIN_DOMAINS)
    next_log = (
        total_seen // args.log_every_total_samples + 1
    ) * args.log_every_total_samples
    next_validation = (
        total_seen // args.validate_every_total_samples + 1
    ) * args.validate_every_total_samples
    running_loss = 0.0
    running_samples = 0
    best_score = None
    best_metrics = None
    metrics_path = output_dir / "metrics.json"
    if resume is not None and metrics_path.exists():
        previous = json.loads(metrics_path.read_text(encoding="utf-8"))
        best_metrics = previous["best"]
        best_score = (
            best_metrics["minimum_domain_deployed_exact_rate"],
            best_metrics["pooled_deployed_exact_rate"],
        )
    started = time.perf_counter()
    for samples, domain_seen in balanced_two_domain_batches(
        samples_per_domain=args.samples_per_domain,
        batch_size=args.batch_size,
        config=scene_config,
        seed=args.seed,
        shuffle_buffer=args.shuffle_buffer,
        start_per_domain=start_per_domain,
    ):
        group_head.train()
        batch = collator(samples)
        pixels = batch["pixel_values"].to(device)
        targets = stack_targets(batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, groups = frozen_groups(
                vision, atom, bond, set_head, pixels, targets, proposal_config
            )
            group_output = group_head(groups["features"])
            loss = atom_pair_group_loss(
                group_output["presence_logits"],
                groups["targets"],
                groups["sample_indices"],
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(group_head.parameters(), 1.0)
        optimizer.step()
        count = len(samples)
        total_seen += count
        running_samples += count
        running_loss += float(loss.detach()) * count
        if total_seen >= next_log:
            print(json.dumps({
                "total_seen": total_seen,
                "domain_seen": domain_seen,
                "loss": running_loss / running_samples,
                "elapsed_seconds": time.perf_counter() - started,
            }, separators=(",", ":")), flush=True)
            running_loss = 0.0
            running_samples = 0
            next_log += args.log_every_total_samples
        if total_seen >= next_validation or min(domain_seen.values()) >= args.samples_per_domain:
            metrics = validate(
                vision, atom, bond, set_head, group_head, validation_records, collator,
                proposal_config, device, args.batch_size,
            )
            metrics.update({
                "total_seen": total_seen,
                "domain_seen": domain_seen,
                "elapsed_seconds": time.perf_counter() - started,
            })
            print(json.dumps({"validation": metrics}, separators=(",", ":")), flush=True)
            checkpoint = {
                "checkpoint_schema_version": "atom-pair-bond-group-0.1",
                "head": group_head.state_dict(),
                "config": asdict(group_config),
                "source_checkpoint": args.source_checkpoint,
                "optimizer": optimizer.state_dict(),
                "samples_seen": total_seen,
                "validation": metrics,
                "training_args": vars(args),
            }
            torch.save(checkpoint, output_dir / f"samples-{total_seen:06d}.pt")
            torch.save(checkpoint, output_dir / "last.pt")
            score = (
                metrics["minimum_domain_deployed_exact_rate"],
                metrics["pooled_deployed_exact_rate"],
            )
            if best_score is None or score > best_score:
                best_score = score
                best_metrics = metrics
                torch.save(checkpoint, output_dir / "best.pt")
            (output_dir / "metrics.json").write_text(
                json.dumps({"latest": metrics, "best": best_metrics}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            next_validation += args.validate_every_total_samples


if __name__ == "__main__":
    main()
