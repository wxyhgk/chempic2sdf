#!/usr/bin/env python3
"""Train an edge-centric veto over frozen v0.9.87 AtomPair predictions."""

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
from chemvector.atom_pair_bond_group import build_atom_pair_groups
from chemvector.atom_pair_deletion_veto import (
    AtomPairDeletionVetoConfig,
    AtomPairDeletionVetoHead,
    apply_atom_pair_deletion_veto,
    atom_pair_deletion_veto_loss,
    build_atom_pair_relation_features,
)
from chemvector.atom_pair_graph_context import (
    flatten_atom_pairs,
    load_atom_pair_graph_context_head,
    refine_atom_pair_group_output,
)
from chemvector.predicted_incidence import (
    PredictedProposalConfig,
    associate_predicted_proposals,
    decode_predicted_atom_proposals,
    decode_predicted_bond_proposals,
)
from chemvector.train_atom_bond_scene import AtomBondSceneCollator, stack_targets
from chemvector.train_atom_pair_bond_group import (
    TRAIN_DOMAINS,
    VALIDATION_DOMAINS,
    balanced_two_domain_batches,
)
from chemvector.train_atom_pair_graph_context import load_group_head
from chemvector.train_bond_presence_calibration import load_components
from chemvector.train_joint_proposal_objects_full import load_validation_records
from chemvector.train_spatial_incidence import DEFAULT_GRAPH, forward_frozen_bond_head


DEFAULT_BASE_CONTEXT = (
    "outputs/checkpoints/"
    "chemvector-v0987-atom-pair-graph-context-positive-margin-calibrated/best.pt"
)
DEFAULT_OUTPUT = "outputs/checkpoints/chemvector-v0988-atom-pair-deletion-veto-4k5"
DEFAULT_THRESHOLDS = (
    -4.0,
    -3.5,
    -3.0,
    -2.5,
    -2.0,
    -1.5,
    -1.0,
    -0.5,
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
)


@torch.no_grad()
def frozen_relation_groups(
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
    groups = build_atom_pair_groups(set_output, proposals, targets)
    return proposals, set_output, groups


def frozen_base_output(
    group_head,
    context_head,
    context_payload,
    groups,
    *,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    local = group_head(groups["features"])
    refined, _ = refine_atom_pair_group_output(
        context_head,
        local,
        groups,
        batch_size=batch_size,
        residual_scale=float(context_payload.get("residual_scale", 1.0)),
        base_positive_protection_threshold=context_payload.get(
            "base_positive_protection_threshold"
        ),
    )
    return refined


def _new_totals() -> defaultdict[str, int]:
    return defaultdict(int)


def _update_totals(
    totals: defaultdict[str, int],
    *,
    selected: torch.Tensor,
    expected: torch.Tensor,
    chemical_correct: torch.Tensor,
    type_mask: torch.Tensor,
    target_count: int,
    matched_count: int,
) -> None:
    positive_count = int(expected.sum())
    candidate_exact = bool((selected == expected).all())
    deployed_exact = (
        candidate_exact
        and matched_count == target_count
        and positive_count == target_count
    )
    chemical_exact = bool(chemical_correct[type_mask].all())
    type_coverage = int(type_mask.sum()) == target_count
    chemical_deployed_exact = deployed_exact and type_coverage and chemical_exact
    totals["samples"] += 1
    totals["targets"] += target_count
    totals["matched_targets"] += matched_count
    totals["groups"] += int(len(expected))
    totals["positive_groups"] += positive_count
    totals["selected_groups"] += int(selected.sum())
    totals["true_positive"] += int((selected & expected).sum())
    totals["false_positive"] += int((selected & ~expected).sum())
    totals["false_negative"] += int((~selected & expected).sum())
    totals["count_exact"] += int(int(selected.sum()) == positive_count)
    totals["candidate_exact"] += int(candidate_exact)
    totals["deployed_exact"] += int(deployed_exact)
    totals["chemical_deployed_exact"] += int(chemical_deployed_exact)


def _finalize_domains(raw: dict[str, defaultdict[str, int]]) -> dict:
    domains = {}
    for domain, totals in raw.items():
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
    return domains


def summarize_domains(domains: dict) -> dict:
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
        "pooled_deployed_exact": sum(
            values["deployed_exact"] for values in domains.values()
        ),
        "pooled_chemical_deployed_exact": sum(
            values["chemical_deployed_exact"] for values in domains.values()
        ),
        "pooled_false_positive": sum(
            values["false_positive"] for values in domains.values()
        ),
        "pooled_false_negative": sum(
            values["false_negative"] for values in domains.values()
        ),
    }


def selection_key(metrics: dict) -> tuple:
    return (
        metrics["minimum_domain_chemical_deployed_exact_rate"],
        metrics["pooled_chemical_deployed_exact_rate"],
        metrics["minimum_domain_deployed_exact_rate"],
        metrics["pooled_deployed_exact_rate"],
        -metrics["pooled_false_negative"],
        -metrics["pooled_false_positive"],
    )


@torch.no_grad()
def validate(
    vision,
    atom,
    bond,
    set_head,
    group_head,
    context_head,
    context_payload,
    veto_head,
    records,
    collator,
    proposal_config,
    device,
    batch_size,
    *,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
) -> dict:
    veto_head.eval()
    baseline_raw = {domain: _new_totals() for domain in records}
    threshold_raw = {
        threshold: {domain: _new_totals() for domain in records}
        for threshold in thresholds
    }
    deletion_counts = {threshold: 0 for threshold in thresholds}
    for domain, rows in records.items():
        for start in range(0, len(rows), batch_size):
            batch = collator(rows[start : start + batch_size])
            pixels = batch["pixel_values"].to(device)
            targets = stack_targets(batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                proposals, set_output, groups = frozen_relation_groups(
                    vision, atom, bond, set_head, pixels, targets, proposal_config
                )
                base = frozen_base_output(
                    group_head,
                    context_head,
                    context_payload,
                    groups,
                    batch_size=len(pixels),
                )
            pairs = flatten_atom_pairs(groups["atom_pairs"], device=device)
            relation = build_atom_pair_relation_features(
                pixels,
                proposals["atom_geometry"][..., :2],
                proposals["atom_mask"],
                set_output["atom_presence_logits"],
                groups["sample_indices"],
                pairs,
                base["presence_logits"],
                batch_size=len(pixels),
                config=veto_head.config,
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                veto = veto_head(
                    groups["features"],
                    relation,
                    base["presence_logits"],
                    groups["sample_indices"],
                    pairs,
                    batch_size=len(pixels),
                )
            baseline_selected = base["presence_logits"] >= 0.0
            threshold_selected = {}
            for threshold in thresholds:
                refined, deleted = apply_atom_pair_deletion_veto(
                    base["presence_logits"],
                    veto["deletion_logits"],
                    threshold=threshold,
                )
                threshold_selected[threshold] = refined >= 0.0
                deletion_counts[threshold] += int(deleted.sum())
            chemical_correct = (
                base["chemical_logits"].argmax(-1) == groups["chemical_targets"]
            )
            for batch_index in range(len(pixels)):
                sample = groups["sample_indices"] == batch_index
                expected = groups["targets"][sample]
                type_mask = groups["type_mask"][sample]
                target_count = int(targets["bond_mask"][batch_index].sum())
                matched_count = int(
                    proposals["bond_target_matched"][batch_index].sum()
                )
                _update_totals(
                    baseline_raw[domain],
                    selected=baseline_selected[sample],
                    expected=expected,
                    chemical_correct=chemical_correct[sample],
                    type_mask=type_mask,
                    target_count=target_count,
                    matched_count=matched_count,
                )
                for threshold in thresholds:
                    _update_totals(
                        threshold_raw[threshold][domain],
                        selected=threshold_selected[threshold][sample],
                        expected=expected,
                        chemical_correct=chemical_correct[sample],
                        type_mask=type_mask,
                        target_count=target_count,
                        matched_count=matched_count,
                    )
    baseline = summarize_domains(_finalize_domains(baseline_raw))
    candidates = {}
    for threshold in thresholds:
        metrics = summarize_domains(_finalize_domains(threshold_raw[threshold]))
        metrics["deletions"] = deletion_counts[threshold]
        candidates[str(threshold)] = metrics
    best_threshold = max(
        thresholds,
        key=lambda value: selection_key(candidates[str(value)]),
    )
    return {
        "baseline": baseline,
        "thresholds": candidates,
        "best_threshold": best_threshold,
        "best": candidates[str(best_threshold)],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-context-checkpoint", default=DEFAULT_BASE_CONTEXT)
    parser.add_argument("--grounded-graph-checkpoint", default=DEFAULT_GRAPH)
    parser.add_argument("--samples-per-domain", type=int, default=2_250)
    parser.add_argument("--validation-samples-per-domain", type=int, default=498)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--relation-only", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--ranking-margin", type=float, default=1.0)
    parser.add_argument("--deletion-positive-weight", type=float)
    parser.add_argument("--ranking-weight", type=float, default=1.0)
    parser.add_argument("--validate-every-total-samples", type=int, default=4_500)
    parser.add_argument("--log-every-total-samples", type=int, default=900)
    parser.add_argument("--shuffle-buffer", type=int, default=1024)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260725)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("AtomPair deletion-veto training requires CUDA")
    if args.batch_size % len(TRAIN_DOMAINS):
        raise ValueError("Batch size must be divisible by the domain count")
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

    context_head, context_payload = load_atom_pair_graph_context_head(
        args.base_context_checkpoint, device
    )
    group_head, group_payload = load_group_head(
        context_payload["group_checkpoint"], device
    )
    vision, atom, bond, set_head, source = load_components(
        group_payload["source_checkpoint"], args.grounded_graph_checkpoint, device
    )
    for module in (vision, atom, bond, set_head, group_head, context_head):
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

    config = AtomPairDeletionVetoConfig(
        input_dim=group_head.config.input_dim,
        hidden_dim=args.hidden_dim,
        maximum_atoms=proposal_config.maximum_atoms,
        image_size=proposal_config.image_size,
        relation_only=args.relation_only,
    )
    veto_head = AtomPairDeletionVetoHead(config).to(device)
    initial = validate(
        vision,
        atom,
        bond,
        set_head,
        group_head,
        context_head,
        context_payload,
        veto_head,
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

    optimizer = torch.optim.AdamW(veto_head.parameters(), lr=args.learning_rate, weight_decay=0.0)
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
        veto_head.train()
        batch = collator(samples)
        pixels = batch["pixel_values"].to(device)
        targets = stack_targets(batch, device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            proposals, set_output, groups = frozen_relation_groups(
                vision, atom, bond, set_head, pixels, targets, proposal_config
            )
            base = frozen_base_output(
                group_head,
                context_head,
                context_payload,
                groups,
                batch_size=len(pixels),
            )
        pairs = flatten_atom_pairs(groups["atom_pairs"], device=device)
        relation = build_atom_pair_relation_features(
            pixels,
            proposals["atom_geometry"][..., :2],
            proposals["atom_mask"],
            set_output["atom_presence_logits"],
            groups["sample_indices"],
            pairs,
            base["presence_logits"],
            batch_size=len(pixels),
            config=config,
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = veto_head(
                groups["features"],
                relation,
                base["presence_logits"],
                groups["sample_indices"],
                pairs,
                batch_size=len(pixels),
            )
            losses = atom_pair_deletion_veto_loss(
                output["deletion_logits"],
                groups["targets"],
                base["presence_logits"],
                groups["sample_indices"],
                ranking_margin=args.ranking_margin,
                deletion_positive_weight=args.deletion_positive_weight,
                ranking_weight=args.ranking_weight,
            )
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(veto_head.parameters(), 1.0)
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
                context_payload,
                veto_head,
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
                "checkpoint_schema_version": "atom-pair-deletion-veto-0.1",
                "head": veto_head.state_dict(),
                "config": asdict(config),
                "base_context_checkpoint": args.base_context_checkpoint,
                "group_checkpoint": context_payload["group_checkpoint"],
                "source_checkpoint": group_payload["source_checkpoint"],
                "samples_seen": total_seen,
                "deletion_threshold": metrics["best_threshold"],
                "validation": metrics,
                "training_args": vars(args),
            }
            torch.save(checkpoint, output_dir / f"samples-{total_seen:06d}.pt")
            torch.save(checkpoint, output_dir / "last.pt")
            score = selection_key(metrics["best"])
            if best_score is None or score > best_score:
                best_score = score
                best_metrics = metrics
                torch.save(checkpoint, output_dir / "best.pt")
            (output_dir / "metrics.json").write_text(
                json.dumps({"latest": metrics, "best": best_metrics}, ensure_ascii=False, indent=2)
                + "\n",
                encoding="utf-8",
            )
            next_validation += args.validate_every_total_samples
    if best_metrics is None:
        raise RuntimeError("Deletion-veto training produced no validation checkpoint")


if __name__ == "__main__":
    main()
