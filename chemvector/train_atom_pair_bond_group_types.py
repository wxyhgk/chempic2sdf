#!/usr/bin/env python3
"""Train AtomPair group display/chemical/direction heads on two domains."""

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
    atom_pair_group_type_loss,
)
from chemvector.predicted_incidence import PredictedProposalConfig
from chemvector.train_atom_bond_scene import AtomBondSceneCollator, stack_targets
from chemvector.train_atom_pair_bond_group import (
    VALIDATION_DOMAINS,
    balanced_two_domain_batches,
    frozen_groups,
)
from chemvector.train_bond_presence_calibration import load_components
from chemvector.train_joint_proposal_objects_full import load_validation_records
from chemvector.train_spatial_incidence import DEFAULT_GRAPH


DEFAULT_SOURCE = (
    "outputs/checkpoints/chemvector-v0972-bond-presence-calibration-10k/best.pt"
)
DEFAULT_PRESENCE = (
    "outputs/checkpoints/chemvector-v0982-atom-pair-bond-group-two-domain18k/best.pt"
)
DEFAULT_OUTPUT = (
    "outputs/checkpoints/chemvector-v0983-atom-pair-bond-group-types-two-domain18k"
)


def load_group_head(path: str, device: torch.device) -> tuple[AtomPairBondGroupHead, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = AtomPairBondGroupConfig(**payload["config"])
    head = AtomPairBondGroupHead(config).to(device)
    missing, unexpected = head.load_state_dict(payload["head"], strict=False)
    expected_missing = {
        "display_head.weight",
        "display_head.bias",
        "chemical_head.weight",
        "chemical_head.bias",
        "direction_head.weight",
        "direction_head.bias",
    }
    if set(missing) != expected_missing or unexpected:
        raise ValueError(
            f"Presence warmstart differs: missing={missing} unexpected={unexpected}"
        )
    head.network.requires_grad_(False)
    return head, payload


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
                output = group_head(groups["features"])
            presence = output["presence_logits"] >= 0.0
            display = output["display_logits"].argmax(-1)
            chemical = output["chemical_logits"].argmax(-1)
            direction = output["direction_logits"].argmax(-1)
            type_mask = groups["type_mask"]
            display_correct = display == groups["display_targets"]
            chemical_correct = chemical == groups["chemical_targets"]
            direction_correct = direction == groups["direction_targets"]
            totals["typed_groups"] += int(type_mask.sum())
            totals["display_correct"] += int((type_mask & display_correct).sum())
            totals["chemical_correct"] += int((type_mask & chemical_correct).sum())
            totals["direction_correct"] += int((type_mask & direction_correct).sum())
            for batch_index in range(len(pixels)):
                sample = groups["sample_indices"] == batch_index
                expected_presence = groups["targets"][sample]
                predicted_presence = presence[sample]
                target_count = int(targets["bond_mask"][batch_index].sum())
                matched_count = int(
                    proposals["bond_target_matched"][batch_index].sum()
                )
                group_positive = int(expected_presence.sum())
                presence_exact = bool((predicted_presence == expected_presence).all())
                deployed_presence = (
                    presence_exact
                    and matched_count == target_count
                    and group_positive == target_count
                )
                positive_types = sample & groups["type_mask"]
                display_exact = bool(display_correct[positive_types].all())
                chemical_exact = bool(chemical_correct[positive_types].all())
                direction_exact = bool(direction_correct[positive_types].all())
                type_coverage = int(positive_types.sum()) == target_count
                typed_deployed = (
                    deployed_presence
                    and type_coverage
                    and display_exact
                    and chemical_exact
                )
                chemical_deployed = (
                    deployed_presence and type_coverage and chemical_exact
                )
                display_deployed = (
                    deployed_presence and type_coverage and display_exact
                )
                direction_deployed = (
                    deployed_presence and type_coverage and direction_exact
                )
                totals["samples"] += 1
                totals["presence_deployed_exact"] += int(deployed_presence)
                totals["display_image_exact"] += int(type_coverage and display_exact)
                totals["chemical_image_exact"] += int(type_coverage and chemical_exact)
                totals["direction_image_exact"] += int(type_coverage and direction_exact)
                totals["typed_deployed_exact"] += int(typed_deployed)
                totals["chemical_deployed_exact"] += int(chemical_deployed)
                totals["display_deployed_exact"] += int(display_deployed)
                totals["direction_deployed_exact"] += int(direction_deployed)
        samples = max(totals["samples"], 1)
        typed_groups = max(totals["typed_groups"], 1)
        domains[domain] = {
            **dict(totals),
            "presence_deployed_exact_rate": totals["presence_deployed_exact"] / samples,
            "display_accuracy": totals["display_correct"] / typed_groups,
            "chemical_accuracy": totals["chemical_correct"] / typed_groups,
            "direction_accuracy": totals["direction_correct"] / typed_groups,
            "display_image_exact_rate": totals["display_image_exact"] / samples,
            "chemical_image_exact_rate": totals["chemical_image_exact"] / samples,
            "direction_image_exact_rate": totals["direction_image_exact"] / samples,
            "typed_deployed_exact_rate": totals["typed_deployed_exact"] / samples,
            "chemical_deployed_exact_rate": totals["chemical_deployed_exact"] / samples,
            "display_deployed_exact_rate": totals["display_deployed_exact"] / samples,
            "direction_deployed_exact_rate": totals["direction_deployed_exact"] / samples,
        }
    return {
        "domains": domains,
        "minimum_domain_chemical_deployed_exact_rate": min(
            item["chemical_deployed_exact_rate"] for item in domains.values()
        ),
        "pooled_chemical_deployed_exact_rate": sum(
            item["chemical_deployed_exact"] for item in domains.values()
        )
        / sum(item["samples"] for item in domains.values()),
        "minimum_domain_typed_deployed_exact_rate": min(
            item["typed_deployed_exact_rate"] for item in domains.values()
        ),
        "pooled_typed_deployed_exact_rate": sum(
            item["typed_deployed_exact"] for item in domains.values()
        )
        / sum(item["samples"] for item in domains.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", default=DEFAULT_SOURCE)
    parser.add_argument("--presence-checkpoint", default=DEFAULT_PRESENCE)
    parser.add_argument("--grounded-graph-checkpoint", default=DEFAULT_GRAPH)
    parser.add_argument("--samples-per-domain", type=int, default=8_967)
    parser.add_argument("--validation-samples-per-domain", type=int, default=498)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--direction-weight", type=float, default=0.25)
    parser.add_argument("--validate-every-total-samples", type=int, default=4_500)
    parser.add_argument("--log-every-total-samples", type=int, default=900)
    parser.add_argument("--shuffle-buffer", type=int, default=1024)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("AtomPair Bond type training requires CUDA")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
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
    for module in (vision, atom, bond, set_head):
        module.eval().requires_grad_(False)
    group_head, presence_payload = load_group_head(args.presence_checkpoint, device)
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
    initial = validate(
        vision, atom, bond, set_head, group_head, validation_records, collator,
        proposal_config, device, args.batch_size,
    )
    (output_dir / "initial_metrics.json").write_text(
        json.dumps(initial, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"initial_validation": initial}, separators=(",", ":")), flush=True)

    trainable = [
        parameter
        for module in (
            group_head.display_head,
            group_head.chemical_head,
            group_head.direction_head,
        )
        for parameter in module.parameters()
    ]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.0)
    total_seen = 0
    next_log = args.log_every_total_samples
    next_validation = args.validate_every_total_samples
    running = defaultdict(float)
    running_samples = 0
    best_score = None
    best_metrics = None
    started = time.perf_counter()
    for samples, domain_seen in balanced_two_domain_batches(
        samples_per_domain=args.samples_per_domain,
        batch_size=args.batch_size,
        config=scene_config,
        seed=args.seed,
        shuffle_buffer=args.shuffle_buffer,
    ):
        group_head.train()
        batch = collator(samples)
        pixels = batch["pixel_values"].to(device)
        targets = stack_targets(batch, device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, groups = frozen_groups(
                vision, atom, bond, set_head, pixels, targets, proposal_config
            )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = group_head(groups["features"])
            losses = atom_pair_group_type_loss(
                output, groups, direction_weight=args.direction_weight
            )
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        count = len(samples)
        total_seen += count
        running_samples += count
        for name, value in losses.items():
            running[name] += float(value.detach()) * count
        if total_seen >= next_log:
            print(json.dumps({
                "total_seen": total_seen,
                "domain_seen": domain_seen,
                **{name: value / running_samples for name, value in running.items()},
                "elapsed_seconds": time.perf_counter() - started,
            }, separators=(",", ":")), flush=True)
            running.clear()
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
                "checkpoint_schema_version": "atom-pair-bond-group-types-0.1",
                "head": group_head.state_dict(),
                "config": asdict(group_head.config),
                "source_checkpoint": args.source_checkpoint,
                "presence_checkpoint": args.presence_checkpoint,
                "presence_samples_seen": presence_payload["samples_seen"],
                "optimizer": optimizer.state_dict(),
                "samples_seen": total_seen,
                "validation": metrics,
                "training_args": vars(args),
            }
            torch.save(checkpoint, output_dir / f"samples-{total_seen:06d}.pt")
            torch.save(checkpoint, output_dir / "last.pt")
            score = (
                metrics["minimum_domain_chemical_deployed_exact_rate"],
                metrics["pooled_chemical_deployed_exact_rate"],
                min(item["chemical_accuracy"] for item in metrics["domains"].values()),
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
