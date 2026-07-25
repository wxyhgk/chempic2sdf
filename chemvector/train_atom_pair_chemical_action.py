#!/usr/bin/env python3
"""Train a chemical-aware one-edit action head on train-only partitions."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.atom_pair_chemical_action import (
    AtomPairChemicalActionConfig,
    AtomPairChemicalActionHead,
    apply_chemical_action,
    build_chemical_action_features,
    chemical_action_loss,
    choose_chemical_action,
    initialize_chemical_action_from_one_move,
    target_chemical_action_index,
)
from chemvector.atom_pair_chemical_data import (
    balanced_fit_batches,
    load_partition_records,
    sample_molecule_id,
)
from chemvector.atom_pair_deletion_veto import (
    AtomPairDeletionVetoConfig,
    build_atom_pair_relation_features,
)
from chemvector.atom_pair_graph_context import (
    flatten_atom_pairs,
    load_atom_pair_graph_context_head,
)
from chemvector.atom_pair_one_move import (
    apply_atom_pair_action,
    choose_atom_pair_action,
    load_atom_pair_one_move_head,
)
from chemvector.audit_atom_pair_chemical_reachability import (
    ABSENT_STATE,
    target_typed_states,
    typed_states,
)
from chemvector.predicted_incidence import PredictedProposalConfig
from chemvector.train_atom_bond_scene import AtomBondSceneCollator, stack_targets
from chemvector.train_atom_pair_bond_group import TRAIN_DOMAINS
from chemvector.train_atom_pair_deletion_veto import (
    frozen_base_output,
    frozen_relation_groups,
)
from chemvector.train_atom_pair_graph_context import load_group_head
from chemvector.train_bond_presence_calibration import load_components
from chemvector.train_spatial_incidence import DEFAULT_GRAPH


DEFAULT_FIRST_MOVE = (
    "outputs/checkpoints/chemvector-v0989-atom-pair-one-move-final/best.pt"
)
DEFAULT_OUTPUT = "outputs/checkpoints/chemvector-v0991-chemical-action-pilot"
DEFAULT_MARGINS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)


def _new_totals() -> Counter:
    return Counter()


def _update_totals(
    totals: Counter,
    *,
    states: torch.Tensor,
    expected_states: torch.Tensor,
    representable: bool,
) -> tuple[bool, bool]:
    selected = states != ABSENT_STATE
    expected_selected = expected_states != ABSENT_STATE
    presence_exact = representable and bool((selected == expected_selected).all())
    chemical_exact = representable and bool((states == expected_states).all())
    totals["samples"] += 1
    totals["representable"] += int(representable)
    totals["presence_exact"] += int(presence_exact)
    totals["chemical_exact"] += int(chemical_exact)
    totals["false_positive"] += int((selected & ~expected_selected).sum())
    totals["false_negative"] += int((~selected & expected_selected).sum())
    totals["wrong_type"] += int(
        (selected & expected_selected & (states != expected_states)).sum()
    )
    return presence_exact, chemical_exact


def _finalize_domains(raw: dict[str, Counter]) -> dict:
    domains = {}
    for domain, totals in raw.items():
        samples = max(totals["samples"], 1)
        domains[domain] = {
            **dict(totals),
            "presence_exact_rate": totals["presence_exact"] / samples,
            "chemical_exact_rate": totals["chemical_exact"] / samples,
        }
    return domains


def summarize_domains(domains: dict) -> dict:
    samples = sum(values["samples"] for values in domains.values())
    return {
        "domains": domains,
        "minimum_domain_presence_exact_rate": min(
            values["presence_exact_rate"] for values in domains.values()
        ),
        "pooled_presence_exact_rate": sum(
            values["presence_exact"] for values in domains.values()
        )
        / samples,
        "minimum_domain_chemical_exact_rate": min(
            values["chemical_exact_rate"] for values in domains.values()
        ),
        "pooled_chemical_exact_rate": sum(
            values["chemical_exact"] for values in domains.values()
        )
        / samples,
        "pooled_presence_exact": sum(
            values["presence_exact"] for values in domains.values()
        ),
        "pooled_chemical_exact": sum(
            values["chemical_exact"] for values in domains.values()
        ),
        "pooled_false_positive": sum(
            values["false_positive"] for values in domains.values()
        ),
        "pooled_false_negative": sum(
            values["false_negative"] for values in domains.values()
        ),
        "pooled_wrong_type": sum(values["wrong_type"] for values in domains.values()),
    }


def selection_key(metrics: dict) -> tuple:
    return (
        metrics["minimum_domain_chemical_exact_rate"],
        metrics["pooled_chemical_exact_rate"],
        -metrics.get("breaks", 0),
        metrics["minimum_domain_presence_exact_rate"],
        metrics["pooled_presence_exact_rate"],
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
    first_head,
    first_margin,
    chemical_head,
    relation_config,
    records,
    collator,
    proposal_config,
    device,
    batch_size,
    *,
    margins: tuple[float, ...] = DEFAULT_MARGINS,
) -> dict:
    chemical_head.eval()
    baseline_raw = {domain: _new_totals() for domain in records}
    margin_raw = {
        margin: {domain: _new_totals() for domain in records} for margin in margins
    }
    action_counts = {margin: Counter() for margin in margins}
    rescues = Counter()
    breaks = Counter()
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
            first_features = torch.cat(
                (relation, base["presence_logits"].float()[:, None]), dim=-1
            )
            for batch_index in range(len(pixels)):
                sample = groups["sample_indices"] == batch_index
                sample_logits = base["presence_logits"][sample]
                chemical_logits = base["chemical_logits"][sample]
                expected_presence = groups["targets"][sample]
                type_mask = groups["type_mask"][sample]
                target_count = int(targets["bond_mask"][batch_index].sum())
                matched_count = int(
                    proposals["bond_target_matched"][batch_index].sum()
                )
                representable = (
                    matched_count == target_count
                    and int(expected_presence.sum()) == target_count
                    and int((expected_presence & type_mask).sum()) == target_count
                )
                first_scores, first_actions = first_head.score_sample(
                    first_features[sample],
                    relation[sample],
                    sample_logits,
                    pairs[sample],
                )
                first_action, _ = choose_atom_pair_action(
                    first_scores,
                    first_actions,
                    minimum_margin=first_margin,
                )
                selected = apply_atom_pair_action(
                    sample_logits >= 0.0, first_action
                )
                current_states = typed_states(selected, chemical_logits)
                expected_states = target_typed_states(
                    expected_presence, groups["chemical_targets"][sample]
                )
                _, baseline_exact = _update_totals(
                    baseline_raw[domain],
                    states=current_states,
                    expected_states=expected_states,
                    representable=representable,
                )
                features = build_chemical_action_features(
                    relation[sample], sample_logits, chemical_logits, selected
                )
                scores, actions = chemical_head.score_sample(
                    features, sample_logits, chemical_logits, selected
                )
                for margin in margins:
                    action, _ = choose_chemical_action(
                        scores, actions, minimum_margin=margin
                    )
                    revised_states = apply_chemical_action(current_states, action)
                    _, revised_exact = _update_totals(
                        margin_raw[margin][domain],
                        states=revised_states,
                        expected_states=expected_states,
                        representable=representable,
                    )
                    action_counts[margin][action.kind] += 1
                    rescues[margin] += int(not baseline_exact and revised_exact)
                    breaks[margin] += int(baseline_exact and not revised_exact)
    baseline = summarize_domains(_finalize_domains(baseline_raw))
    candidates = {}
    for margin in margins:
        metrics = summarize_domains(_finalize_domains(margin_raw[margin]))
        metrics["actions"] = dict(action_counts[margin])
        metrics["rescues"] = rescues[margin]
        metrics["breaks"] = breaks[margin]
        candidates[str(margin)] = metrics
    best_margin = max(margins, key=lambda value: selection_key(candidates[str(value)]))
    return {
        "baseline": baseline,
        "margins": candidates,
        "best_margin": best_margin,
        "best": candidates[str(best_margin)],
    }


def _identity_sha256(values: set[str]) -> str:
    payload = "\n".join(sorted(values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-move-checkpoint", default=DEFAULT_FIRST_MOVE)
    parser.add_argument("--grounded-graph-checkpoint", default=DEFAULT_GRAPH)
    parser.add_argument("--samples-per-domain", type=int, default=2_250)
    parser.add_argument("--dev-samples-per-domain", type=int, default=498)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--type-embedding-dim", type=int, default=16)
    parser.add_argument("--delete-candidates", type=int, default=4)
    parser.add_argument("--add-candidates", type=int, default=16)
    parser.add_argument("--retype-candidates", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--edit-weight", type=float, default=16.0)
    parser.add_argument("--shuffle-buffer", type=int, default=1024)
    parser.add_argument("--data-seed", type=int, default=20260725)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Chemical action training requires CUDA")
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

    first_head, first_payload = load_atom_pair_one_move_head(
        args.first_move_checkpoint, device
    )
    context_head, context_payload = load_atom_pair_graph_context_head(
        first_payload["base_context_checkpoint"], device
    )
    group_head, group_payload = load_group_head(
        context_payload["group_checkpoint"], device
    )
    vision, atom, bond, set_head, source = load_components(
        group_payload["source_checkpoint"], args.grounded_graph_checkpoint, device
    )
    for module in (
        vision,
        atom,
        bond,
        set_head,
        group_head,
        context_head,
        first_head,
    ):
        module.eval().requires_grad_(False)
    proposal_config = PredictedProposalConfig(**source["proposal_config"])
    scene_config = AtomBondSceneConfig(
        atom_queries=proposal_config.maximum_atoms,
        bond_queries=proposal_config.maximum_bonds,
    )
    collator = AtomBondSceneCollator(scene_config)
    relation_config = AtomPairDeletionVetoConfig(**first_payload["relation_config"])
    chemical_config = AtomPairChemicalActionConfig(
        hidden_dim=args.hidden_dim,
        type_embedding_dim=args.type_embedding_dim,
        delete_candidates=args.delete_candidates,
        add_candidates=args.add_candidates,
        retype_candidates=args.retype_candidates,
    )
    chemical_head = AtomPairChemicalActionHead(chemical_config).to(device)
    initialize_chemical_action_from_one_move(chemical_head, first_head)

    dev_records = {
        domain: load_partition_records(
            domain,
            partition="train_dev",
            count=args.dev_samples_per_domain,
            config=scene_config,
            data_seed=args.data_seed,
            shuffle_buffer=args.shuffle_buffer,
        )
        for domain in TRAIN_DOMAINS
    }
    dev_ids = {
        sample_molecule_id(sample)
        for rows in dev_records.values()
        for sample in rows
    }
    initial = validate(
        vision,
        atom,
        bond,
        set_head,
        group_head,
        context_head,
        context_payload,
        first_head,
        float(first_payload["minimum_margin"]),
        chemical_head,
        relation_config,
        dev_records,
        collator,
        proposal_config,
        device,
        args.batch_size,
    )
    (output_dir / "initial_metrics.json").write_text(
        json.dumps(initial, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"initial_train_dev": initial}, separators=(",", ":")), flush=True)

    optimizer = torch.optim.AdamW(
        chemical_head.parameters(), lr=args.learning_rate, weight_decay=0.0
    )
    running = defaultdict(float)
    running_counts = Counter()
    running_eligible = 0
    total_seen = 0
    fit_ids: set[str] = set()
    started = time.perf_counter()
    for samples, domain_seen in balanced_fit_batches(
        samples_per_domain=args.samples_per_domain,
        batch_size=args.batch_size,
        config=scene_config,
        data_seed=args.data_seed,
        shuffle_buffer=args.shuffle_buffer,
    ):
        fit_ids.update(sample_molecule_id(sample) for sample in samples)
        chemical_head.train()
        batch = collator(samples)
        pixels = batch["pixel_values"].to(device)
        targets = stack_targets(batch, device)
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
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
        first_features = torch.cat(
            (relation, base["presence_logits"].float()[:, None]), dim=-1
        )
        losses = []
        batch_counts = Counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for batch_index in range(len(pixels)):
                sample = groups["sample_indices"] == batch_index
                sample_logits = base["presence_logits"][sample]
                chemical_logits = base["chemical_logits"][sample]
                expected_presence = groups["targets"][sample]
                type_mask = groups["type_mask"][sample]
                target_count = int(targets["bond_mask"][batch_index].sum())
                matched_count = int(
                    proposals["bond_target_matched"][batch_index].sum()
                )
                representable = (
                    matched_count == target_count
                    and int(expected_presence.sum()) == target_count
                    and int((expected_presence & type_mask).sum()) == target_count
                )
                if not representable:
                    batch_counts["unrepresentable"] += 1
                    continue
                with torch.no_grad():
                    first_scores, first_actions = first_head.score_sample(
                        first_features[sample],
                        relation[sample],
                        sample_logits,
                        pairs[sample],
                    )
                    first_action, _ = choose_atom_pair_action(
                        first_scores,
                        first_actions,
                        minimum_margin=float(first_payload["minimum_margin"]),
                    )
                    selected = apply_atom_pair_action(
                        sample_logits >= 0.0, first_action
                    )
                    current_states = typed_states(selected, chemical_logits)
                    expected_states = target_typed_states(
                        expected_presence, groups["chemical_targets"][sample]
                    )
                features = build_chemical_action_features(
                    relation[sample], sample_logits, chemical_logits, selected
                )
                scores, actions = chemical_head.score_sample(
                    features, sample_logits, chemical_logits, selected
                )
                target_index = target_chemical_action_index(
                    actions, current_states, expected_states
                )
                if target_index is None:
                    batch_counts["not_one_typed_edit"] += 1
                    continue
                action_kind = actions[target_index].kind
                batch_counts[action_kind] += 1
                losses.append(
                    chemical_action_loss(
                        scores,
                        target_index,
                        edit_weight=args.edit_weight,
                    )
                )
        if losses:
            loss = torch.stack(losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(chemical_head.parameters(), 1.0)
            optimizer.step()
            running["loss"] += float(loss.detach()) * len(losses)
            running_eligible += len(losses)
        running_counts.update(batch_counts)
        total_seen += len(samples)
        if total_seen % 900 < len(samples):
            print(
                json.dumps(
                    {
                        "total_seen": total_seen,
                        "domain_seen": domain_seen,
                        "eligible_samples": running_eligible,
                        "loss": running["loss"] / max(running_eligible, 1),
                        "targets": dict(running_counts),
                        "elapsed_seconds": time.perf_counter() - started,
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
            running.clear()
            running_counts.clear()
            running_eligible = 0

    overlap = fit_ids & dev_ids
    if overlap:
        raise RuntimeError(f"train_fit/train_dev molecule overlap: {len(overlap)}")
    partition_audit = {
        "protocol": "sha256(molecule_id) modulo 5; bucket 0 is train_dev",
        "data_seed": args.data_seed,
        "fit_unique_molecules": len(fit_ids),
        "dev_unique_molecules": len(dev_ids),
        "overlap": 0,
        "fit_sha256": _identity_sha256(fit_ids),
        "dev_sha256": _identity_sha256(dev_ids),
    }
    (output_dir / "partition_audit.json").write_text(
        json.dumps(partition_audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    metrics = validate(
        vision,
        atom,
        bond,
        set_head,
        group_head,
        context_head,
        context_payload,
        first_head,
        float(first_payload["minimum_margin"]),
        chemical_head,
        relation_config,
        dev_records,
        collator,
        proposal_config,
        device,
        args.batch_size,
    )
    metrics.update(
        {
            "total_seen": total_seen,
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    checkpoint = {
        "checkpoint_schema_version": "atom-pair-chemical-action-0.1",
        "head": chemical_head.state_dict(),
        "config": asdict(chemical_config),
        "relation_config": asdict(relation_config),
        "first_move_checkpoint": args.first_move_checkpoint,
        "base_context_checkpoint": first_payload["base_context_checkpoint"],
        "group_checkpoint": context_payload["group_checkpoint"],
        "source_checkpoint": group_payload["source_checkpoint"],
        "samples_seen": total_seen,
        "minimum_margin": metrics["best_margin"],
        "train_dev": metrics,
        "partition_audit": partition_audit,
        "training_args": vars(args),
    }
    torch.save(checkpoint, output_dir / "best.pt")
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"train_dev": metrics}, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
