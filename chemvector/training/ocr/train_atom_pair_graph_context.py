#!/usr/bin/env python3
"""Train whole-graph context over frozen AtomPair Bond groups."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.atom_pair_bond_group import AtomPairBondGroupConfig, AtomPairBondGroupHead
from chemvector.atom_pair_graph_context import (
    AtomPairGraphContextConfig,
    AtomPairGraphContextHead,
    atom_pair_graph_context_loss,
    flatten_atom_pairs,
)
from chemvector.predicted_incidence import PredictedProposalConfig
from chemvector.train_atom_bond_scene import AtomBondSceneCollator, stack_targets
from chemvector.train_atom_pair_bond_group import (
    TRAIN_DOMAINS,
    VALIDATION_DOMAINS,
    balanced_two_domain_batches,
    frozen_groups,
)
from chemvector.train_bond_presence_calibration import load_components
from chemvector.train_joint_proposal_objects_full import load_validation_records
from chemvector.train_spatial_incidence import DEFAULT_GRAPH


DEFAULT_GROUP = (
    "outputs/checkpoints/chemvector-v0983-atom-pair-bond-group-types-two-domain18k/"
    "best-chemical.pt"
)
DEFAULT_OUTPUT = "outputs/checkpoints/chemvector-v0986-atom-pair-graph-context-4k5"


def load_group_head(
    path: str, device: torch.device
) -> tuple[AtomPairBondGroupHead, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = AtomPairBondGroupHead(AtomPairBondGroupConfig(**payload["config"])).to(device)
    model.load_state_dict(payload["head"], strict=True)
    model.eval().requires_grad_(False)
    return model, payload


@torch.no_grad()
def validate(
    vision,
    atom,
    bond,
    set_head,
    group_head,
    context_head,
    records,
    collator,
    proposal_config,
    device,
    batch_size,
    residual_scale: float = 1.0,
    base_positive_protection_threshold: float | None = None,
) -> dict:
    context_head.eval()
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
                base = group_head(groups["features"])
                pairs = flatten_atom_pairs(groups["atom_pairs"], device=device)
                output = context_head(
                    groups["features"],
                    base["presence_logits"],
                    groups["sample_indices"],
                    pairs,
                    batch_size=len(pixels),
                )
            scaled_logits = (
                base["presence_logits"]
                + float(residual_scale) * output["residual_logits"]
            )
            if base_positive_protection_threshold is not None:
                protected = base["presence_logits"] >= float(
                    base_positive_protection_threshold
                )
                scaled_logits = torch.where(
                    protected, base["presence_logits"], scaled_logits
                )
            selected_all = scaled_logits >= 0.0
            chemical = base["chemical_logits"].argmax(-1)
            chemical_correct = chemical == groups["chemical_targets"]
            for batch_index in range(len(pixels)):
                sample = groups["sample_indices"] == batch_index
                expected = groups["targets"][sample]
                selected = selected_all[sample]
                target_count = int(targets["bond_mask"][batch_index].sum())
                matched_count = int(
                    proposals["bond_target_matched"][batch_index].sum()
                )
                positive_count = int(expected.sum())
                candidate_exact = bool((selected == expected).all())
                deployed_exact = (
                    candidate_exact
                    and matched_count == target_count
                    and positive_count == target_count
                )
                positive_types = sample & groups["type_mask"]
                chemical_exact = bool(chemical_correct[positive_types].all())
                type_coverage = int(positive_types.sum()) == target_count
                chemical_deployed_exact = (
                    deployed_exact and type_coverage and chemical_exact
                )
                totals["samples"] += 1
                totals["targets"] += target_count
                totals["matched_targets"] += matched_count
                totals["groups"] += int(sample.sum())
                totals["positive_groups"] += positive_count
                totals["selected_groups"] += int(selected.sum())
                totals["true_positive"] += int((selected & expected).sum())
                totals["false_positive"] += int((selected & ~expected).sum())
                totals["false_negative"] += int((~selected & expected).sum())
                totals["count_exact"] += int(int(selected.sum()) == positive_count)
                totals["candidate_exact"] += int(candidate_exact)
                totals["deployed_exact"] += int(deployed_exact)
                totals["chemical_deployed_exact"] += int(chemical_deployed_exact)
        samples = max(totals["samples"], 1)
        domains[domain] = {
            **dict(totals),
            "proposal_recall": totals["matched_targets"] / max(totals["targets"], 1),
            "precision": totals["true_positive"]
            / max(totals["true_positive"] + totals["false_positive"], 1),
            "recall": totals["true_positive"]
            / max(totals["true_positive"] + totals["false_negative"], 1),
            "count_exact_rate": totals["count_exact"] / samples,
            "candidate_exact_rate": totals["candidate_exact"] / samples,
            "deployed_exact_rate": totals["deployed_exact"] / samples,
            "chemical_deployed_exact_rate": totals["chemical_deployed_exact"]
            / samples,
        }
    return {
        "domains": domains,
        "minimum_domain_deployed_exact_rate": min(
            values["deployed_exact_rate"] for values in domains.values()
        ),
        "pooled_deployed_exact_rate": sum(
            values["deployed_exact"] for values in domains.values()
        )
        / sum(values["samples"] for values in domains.values()),
        "minimum_domain_chemical_deployed_exact_rate": min(
            values["chemical_deployed_exact_rate"] for values in domains.values()
        ),
        "pooled_chemical_deployed_exact_rate": sum(
            values["chemical_deployed_exact"] for values in domains.values()
        )
        / sum(values["samples"] for values in domains.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-checkpoint", default=DEFAULT_GROUP)
    parser.add_argument("--grounded-graph-checkpoint", default=DEFAULT_GRAPH)
    parser.add_argument("--samples-per-domain", type=int, default=2_250)
    parser.add_argument("--validation-samples-per-domain", type=int, default=498)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--count-weight", type=float, default=0.25)
    parser.add_argument("--degree-weight", type=float, default=0.25)
    parser.add_argument("--residual-weight", type=float, default=1e-4)
    parser.add_argument("--positive-margin", type=float, default=0.5)
    parser.add_argument("--positive-margin-weight", type=float, default=0.0)
    parser.add_argument("--validate-every-total-samples", type=int, default=4_500)
    parser.add_argument("--log-every-total-samples", type=int, default=900)
    parser.add_argument("--shuffle-buffer", type=int, default=1024)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("AtomPair graph-context training requires CUDA")
    if args.batch_size % len(TRAIN_DOMAINS):
        raise ValueError("Batch size must be divisible by the domain count")
    if min(args.samples_per_domain, args.batch_size, args.hidden_dim) < 1:
        raise ValueError("Training dimensions must be positive")
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

    group_head, group_payload = load_group_head(args.group_checkpoint, device)
    vision, atom, bond, set_head, source = load_components(
        group_payload["source_checkpoint"], args.grounded_graph_checkpoint, device
    )
    for module in (vision, atom, bond, set_head, group_head):
        module.eval().requires_grad_(False)
    proposal_config = PredictedProposalConfig(**source["proposal_config"])
    scene_config = AtomBondSceneConfig(
        atom_queries=proposal_config.maximum_atoms,
        bond_queries=proposal_config.maximum_bonds,
    )
    collator = AtomBondSceneCollator(scene_config)
    validation_records = {}
    validation_audit = {}
    for name, shard in VALIDATION_DOMAINS.items():
        rows, audit = load_validation_records(
            [shard], per_shard=args.validation_samples_per_domain, config=scene_config
        )
        validation_records[name] = rows
        validation_audit[name] = audit
    (output_dir / "validation_audit.json").write_text(
        json.dumps(validation_audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    config = AtomPairGraphContextConfig(
        input_dim=group_head.config.input_dim,
        hidden_dim=args.hidden_dim,
        maximum_atoms=proposal_config.maximum_atoms,
    )
    context_head = AtomPairGraphContextHead(config).to(device)
    initial = validate(
        vision,
        atom,
        bond,
        set_head,
        group_head,
        context_head,
        validation_records,
        collator,
        proposal_config,
        device,
        args.batch_size,
    )
    (output_dir / "initial_metrics.json").write_text(
        json.dumps(initial, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"initial_validation": initial}, separators=(",", ":")), flush=True)

    optimizer = torch.optim.AdamW(
        context_head.parameters(), lr=args.learning_rate, weight_decay=0.0
    )
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
        context_head.train()
        batch = collator(samples)
        pixels = batch["pixel_values"].to(device)
        targets = stack_targets(batch, device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            proposals, groups = frozen_groups(
                vision, atom, bond, set_head, pixels, targets, proposal_config
            )
            base = group_head(groups["features"])
        pairs = flatten_atom_pairs(groups["atom_pairs"], device=device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = context_head(
                groups["features"],
                base["presence_logits"],
                groups["sample_indices"],
                pairs,
                batch_size=len(pixels),
            )
            losses = atom_pair_graph_context_loss(
                output,
                groups["targets"],
                groups["sample_indices"],
                pairs,
                batch_size=len(pixels),
                maximum_atoms=config.maximum_atoms,
                count_weight=args.count_weight,
                degree_weight=args.degree_weight,
                residual_weight=args.residual_weight,
                positive_margin=args.positive_margin,
                positive_margin_weight=args.positive_margin_weight,
            )
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(context_head.parameters(), 1.0)
        optimizer.step()
        count = len(samples)
        total_seen += count
        running_samples += count
        for name, value in losses.items():
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
            metrics = validate(
                vision,
                atom,
                bond,
                set_head,
                group_head,
                context_head,
                validation_records,
                collator,
                proposal_config,
                device,
                args.batch_size,
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
                "checkpoint_schema_version": "atom-pair-graph-context-0.1",
                "head": context_head.state_dict(),
                "config": asdict(config),
                "group_checkpoint": args.group_checkpoint,
                "source_checkpoint": group_payload["source_checkpoint"],
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
                json.dumps(
                    {"latest": metrics, "best": best_metrics},
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            next_validation += args.validate_every_total_samples
    if best_metrics is None:
        raise RuntimeError("Graph-context training produced no validation checkpoint")


if __name__ == "__main__":
    main()
