#!/usr/bin/env python3
"""Train set-level keep/add/delete/swap selection over frozen v0.9.87 groups."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.atom_pair_deletion_veto import (
    AtomPairDeletionVetoConfig,
    build_atom_pair_relation_features,
)
from chemvector.atom_pair_graph_context import (
    flatten_atom_pairs,
    load_atom_pair_graph_context_head,
)
from chemvector.atom_pair_one_move import (
    AtomPairOneMoveConfig,
    AtomPairOneMoveHead,
    apply_atom_pair_action,
    choose_atom_pair_action,
    one_move_loss,
    target_action_index,
)
from chemvector.predicted_incidence import PredictedProposalConfig
from chemvector.train_atom_bond_scene import AtomBondSceneCollator, stack_targets
from chemvector.train_atom_pair_bond_group import (
    TRAIN_DOMAINS,
    VALIDATION_DOMAINS,
    balanced_two_domain_batches,
)
from chemvector.train_atom_pair_deletion_veto import (
    _finalize_domains,
    _new_totals,
    _update_totals,
    frozen_base_output,
    frozen_relation_groups,
    selection_key,
    summarize_domains,
)
from chemvector.train_atom_pair_graph_context import load_group_head
from chemvector.train_bond_presence_calibration import load_components
from chemvector.train_joint_proposal_objects_full import load_validation_records
from chemvector.train_spatial_incidence import DEFAULT_GRAPH


DEFAULT_BASE_CONTEXT = (
    "outputs/checkpoints/"
    "chemvector-v0987-atom-pair-graph-context-positive-margin-calibrated/best.pt"
)
DEFAULT_OUTPUT = "outputs/checkpoints/chemvector-v0989-atom-pair-one-move-4k5"
DEFAULT_MARGINS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)


@torch.no_grad()
def validate(
    vision,
    atom,
    bond,
    set_head,
    group_head,
    context_head,
    context_payload,
    action_head,
    relation_config,
    records,
    collator,
    proposal_config,
    device,
    batch_size,
    *,
    margins: tuple[float, ...] = DEFAULT_MARGINS,
) -> dict:
    action_head.eval()
    baseline_raw = {domain: _new_totals() for domain in records}
    margin_raw = {
        margin: {domain: _new_totals() for domain in records} for margin in margins
    }
    action_counts = {margin: Counter() for margin in margins}
    action_margins = {margin: [] for margin in margins}
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
                config=relation_config,
            )
            edge_features = torch.cat(
                (relation, base["presence_logits"].float()[:, None]), dim=-1
            )
            chemical_correct = (
                base["chemical_logits"].argmax(-1) == groups["chemical_targets"]
            )
            for batch_index in range(len(pixels)):
                sample = groups["sample_indices"] == batch_index
                sample_base = base["presence_logits"][sample]
                baseline_selected = sample_base >= 0.0
                expected = groups["targets"][sample]
                type_mask = groups["type_mask"][sample]
                target_count = int(targets["bond_mask"][batch_index].sum())
                matched_count = int(
                    proposals["bond_target_matched"][batch_index].sum()
                )
                _update_totals(
                    baseline_raw[domain],
                    selected=baseline_selected,
                    expected=expected,
                    chemical_correct=chemical_correct[sample],
                    type_mask=type_mask,
                    target_count=target_count,
                    matched_count=matched_count,
                )
                scores, actions = action_head.score_sample(
                    edge_features[sample],
                    relation[sample],
                    sample_base,
                    pairs[sample],
                )
                for margin in margins:
                    action, score_margin = choose_atom_pair_action(
                        scores, actions, minimum_margin=margin
                    )
                    revised = apply_atom_pair_action(baseline_selected, action)
                    action_counts[margin][action.kind] += 1
                    action_margins[margin].append(score_margin)
                    _update_totals(
                        margin_raw[margin][domain],
                        selected=revised,
                        expected=expected,
                        chemical_correct=chemical_correct[sample],
                        type_mask=type_mask,
                        target_count=target_count,
                        matched_count=matched_count,
                    )
    baseline = summarize_domains(_finalize_domains(baseline_raw))
    candidates = {}
    for margin in margins:
        metrics = summarize_domains(_finalize_domains(margin_raw[margin]))
        metrics["actions"] = dict(action_counts[margin])
        values = action_margins[margin]
        metrics["mean_best_action_margin"] = sum(values) / max(len(values), 1)
        candidates[str(margin)] = metrics
    best_margin = max(margins, key=lambda value: selection_key(candidates[str(value)]))
    return {
        "baseline": baseline,
        "margins": candidates,
        "best_margin": best_margin,
        "best": candidates[str(best_margin)],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-context-checkpoint", default=DEFAULT_BASE_CONTEXT)
    parser.add_argument("--grounded-graph-checkpoint", default=DEFAULT_GRAPH)
    parser.add_argument("--samples-per-domain", type=int, default=2_250)
    parser.add_argument("--validation-samples-per-domain", type=int, default=498)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--delete-candidates", type=int, default=4)
    parser.add_argument("--add-candidates", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--edit-weight", type=float, default=16.0)
    parser.add_argument("--validate-every-total-samples", type=int, default=4_500)
    parser.add_argument("--log-every-total-samples", type=int, default=900)
    parser.add_argument("--shuffle-buffer", type=int, default=1024)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--start-per-domain", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260725)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("AtomPair one-move training requires CUDA")
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
    group_head, group_payload = load_group_head(context_payload["group_checkpoint"], device)
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
    relation_config = AtomPairDeletionVetoConfig(
        maximum_atoms=proposal_config.maximum_atoms,
        image_size=proposal_config.image_size,
    )
    action_config = AtomPairOneMoveConfig(
        hidden_dim=args.hidden_dim,
        delete_candidates=args.delete_candidates,
        add_candidates=args.add_candidates,
    )
    action_head = AtomPairOneMoveHead(action_config).to(device)
    if args.resume_checkpoint:
        resume_payload = torch.load(
            args.resume_checkpoint, map_location="cpu", weights_only=False
        )
        if resume_payload.get("checkpoint_schema_version") != "atom-pair-one-move-0.1":
            raise ValueError("Unsupported one-move resume checkpoint")
        if resume_payload["config"] != asdict(action_config):
            raise ValueError("One-move resume config differs from requested config")
        action_head.load_state_dict(resume_payload["head"], strict=True)
        if resume_payload["samples_seen"] != args.start_per_domain * len(TRAIN_DOMAINS):
            raise ValueError("Resume sample count differs from start-per-domain")
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
    initial = validate(
        vision,
        atom,
        bond,
        set_head,
        group_head,
        context_head,
        context_payload,
        action_head,
        relation_config,
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

    optimizer = torch.optim.AdamW(action_head.parameters(), lr=args.learning_rate, weight_decay=0.0)
    total_seen = args.start_per_domain * len(TRAIN_DOMAINS)
    next_log = total_seen + args.log_every_total_samples
    next_validation = total_seen + args.validate_every_total_samples
    running = defaultdict(float)
    running_counts = Counter()
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
        start_per_domain=args.start_per_domain,
    ):
        action_head.train()
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
            config=relation_config,
        )
        edge_features = torch.cat(
            (relation, base["presence_logits"].float()[:, None]), dim=-1
        )
        losses = []
        batch_counts = Counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for batch_index in range(len(pixels)):
                sample = groups["sample_indices"] == batch_index
                expected = groups["targets"][sample]
                target_count = int(targets["bond_mask"][batch_index].sum())
                matched_count = int(
                    proposals["bond_target_matched"][batch_index].sum()
                )
                if matched_count != target_count or int(expected.sum()) != target_count:
                    batch_counts["unrepresentable"] += 1
                    continue
                sample_base = base["presence_logits"][sample]
                scores, actions = action_head.score_sample(
                    edge_features[sample],
                    relation[sample],
                    sample_base,
                    pairs[sample],
                )
                target_index = target_action_index(actions, sample_base, expected)
                if target_index is None:
                    batch_counts["not_one_move"] += 1
                    continue
                batch_counts[actions[target_index].kind] += 1
                losses.append(
                    one_move_loss(
                        scores,
                        target_index,
                        edit_weight=args.edit_weight,
                    )
                )
        if losses:
            loss = torch.stack(losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(action_head.parameters(), 1.0)
            optimizer.step()
            running["loss"] += float(loss.detach()) * len(losses)
            running_samples += len(losses)
        running_counts.update(batch_counts)
        count = len(samples)
        total_seen += count
        if total_seen >= next_log:
            print(
                json.dumps(
                    {
                        "total_seen": total_seen,
                        "domain_seen": domain_seen,
                        "eligible_samples": running_samples,
                        "loss": running["loss"] / max(running_samples, 1),
                        "action_targets": dict(running_counts),
                        "elapsed_seconds": time.perf_counter() - started,
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
            running.clear()
            running_counts.clear()
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
                action_head,
                relation_config,
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
                "checkpoint_schema_version": "atom-pair-one-move-0.1",
                "head": action_head.state_dict(),
                "config": asdict(action_config),
                "relation_config": asdict(relation_config),
                "base_context_checkpoint": args.base_context_checkpoint,
                "group_checkpoint": context_payload["group_checkpoint"],
                "source_checkpoint": group_payload["source_checkpoint"],
                "samples_seen": total_seen,
                "minimum_margin": metrics["best_margin"],
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
        raise RuntimeError("One-move training produced no validation checkpoint")


if __name__ == "__main__":
    main()
