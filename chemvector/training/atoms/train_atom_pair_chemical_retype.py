#!/usr/bin/env python3
"""Train a two-stage chemical retype gate/action on train-only partitions."""

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
from chemvector.atom_pair_chemical_data import (
    balanced_fit_batches,
    count_partition_records,
    load_partition_records,
    sample_molecule_id,
)
from chemvector.atom_pair_chemical_retype import (
    AtomPairChemicalRetypeConfig,
    AtomPairChemicalRetypeHead,
    apply_retype_action,
    build_retype_features,
    chemical_retype_joint_loss,
    chemical_retype_loss,
    chemical_retype_safety_loss,
    choose_joint_retype_action,
    choose_retype_action,
    initialize_retype_encoder_from_one_move,
    load_atom_pair_chemical_retype_head,
    target_retype_action_index,
)
from chemvector.atom_pair_chemical_retype_verifier import (
    AtomPairChemicalRetypeVerifier,
    AtomPairChemicalRetypeVerifierConfig,
    candidate_exact_labels,
    chemical_retype_verifier_loss,
    choose_verified_retype_action,
    initialize_verifier_from_retype,
    load_atom_pair_chemical_retype_verifier,
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
from chemvector.bond_object import BOND_TYPES
from chemvector.predicted_incidence import PredictedProposalConfig
from chemvector.train_atom_bond_scene import AtomBondSceneCollator, stack_targets
from chemvector.train_atom_pair_bond_group import TRAIN_DOMAINS
from chemvector.train_atom_pair_chemical_action import (
    _finalize_domains,
    _identity_sha256,
    _new_totals,
    _update_totals,
    summarize_domains,
)
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
DEFAULT_OUTPUT = "outputs/checkpoints/chemvector-v0991-chemical-retype-pilot"
DEFAULT_THRESHOLDS = (
    -2.0,
    -1.0,
    *(round(-0.5 + 0.05 * index, 2) for index in range(21)),
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
)
DEFAULT_VERIFIER_THRESHOLDS = (
    *(round(-5.0 + 0.25 * index, 2) for index in range(25)),
    1.5,
    2.0,
    3.0,
    4.0,
)


def selection_key(metrics: dict) -> tuple:
    return (
        metrics["minimum_domain_chemical_exact_rate"],
        metrics["pooled_chemical_exact_rate"],
        -metrics.get("breaks", 0),
        metrics.get("rescues", 0),
    )


def validation_checkpoint_due(
    total_seen: int,
    last_validated_seen: int,
    interval: int,
    *,
    final: bool = False,
) -> bool:
    if min(total_seen, last_validated_seen, interval) < 0:
        raise ValueError("Validation checkpoint counts must be non-negative")
    if final:
        return total_seen != last_validated_seen
    return interval > 0 and total_seen // interval > last_validated_seen // interval


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
    retype_head,
    relation_config,
    records,
    collator,
    proposal_config,
    device,
    batch_size,
    *,
    thresholds: tuple[float, ...] | None = None,
    objective: str = "separate",
    verifier_head=None,
    collect_sample_details: bool = False,
) -> dict:
    if objective not in {"separate", "joint", "safety", "verifier"}:
        raise ValueError(f"Unknown retype objective: {objective}")
    if objective == "verifier" and verifier_head is None:
        raise ValueError("Verifier objective requires a candidate verifier")
    if thresholds is None:
        thresholds = (
            DEFAULT_VERIFIER_THRESHOLDS
            if objective == "verifier"
            else DEFAULT_THRESHOLDS
        )
    retype_head.eval()
    if verifier_head is not None:
        verifier_head.eval()
    baseline_raw = {domain: _new_totals() for domain in records}
    threshold_raw = {
        threshold: {domain: _new_totals() for domain in records}
        for threshold in thresholds
    }
    action_counts = {threshold: Counter() for threshold in thresholds}
    rescues = Counter()
    breaks = Counter()
    gate_values = []
    gate_values_by_target = {"keep": [], "retype": []}
    diagnostic_counts = Counter()
    action_target_ranks = []
    verifier_target_ranks = []
    sample_details = {threshold: [] for threshold in thresholds}
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
                baseline_presence_exact, baseline_exact = _update_totals(
                    baseline_raw[domain],
                    states=current_states,
                    expected_states=expected_states,
                    representable=representable,
                )
                retype_features = build_retype_features(
                    relation[sample], sample_logits, chemical_logits, selected
                )
                gate, action_scores, actions = retype_head.score_sample(
                    retype_features,
                    chemical_logits,
                    selected,
                )
                verifier_scores = None
                if objective == "verifier":
                    verifier_scores = verifier_head.score_sample(
                        retype_features,
                        chemical_logits,
                        selected,
                        gate,
                        action_scores,
                        actions,
                    )
                gate_values.append(float(gate))
                target = target_retype_action_index(
                    actions, current_states, expected_states
                )
                if target is None:
                    diagnostic_counts["not_keep_or_one_retype"] += 1
                else:
                    gate_target, action_target = target
                    target_name = "retype" if gate_target else "keep"
                    diagnostic_counts[target_name] += 1
                    gate_values_by_target[target_name].append(float(gate))
                    if action_target is not None:
                        ranking = torch.argsort(
                            action_scores, descending=True, stable=True
                        )
                        rank = int(torch.where(ranking == action_target)[0][0]) + 1
                        action_target_ranks.append(rank)
                        diagnostic_counts["action_top1_correct"] += int(rank == 1)
                        if verifier_scores is not None:
                            verifier_ranking = torch.argsort(
                                verifier_scores, descending=True, stable=True
                            )
                            verifier_rank = int(
                                torch.where(verifier_ranking == action_target)[0][0]
                            ) + 1
                            verifier_target_ranks.append(verifier_rank)
                            diagnostic_counts["verifier_top1_correct"] += int(
                                verifier_rank == 1
                            )
                for threshold in thresholds:
                    if objective == "verifier":
                        action = choose_verified_retype_action(
                            verifier_scores,
                            actions,
                            minimum_margin=threshold,
                        )
                    elif objective in {"joint", "safety"}:
                        action = choose_joint_retype_action(
                            gate,
                            action_scores,
                            actions,
                            minimum_margin=threshold,
                        )
                    else:
                        action = choose_retype_action(
                            gate,
                            action_scores,
                            actions,
                            gate_threshold=threshold,
                        )
                    revised = apply_retype_action(current_states, action)
                    revised_presence_exact, revised_exact = _update_totals(
                        threshold_raw[threshold][domain],
                        states=revised,
                        expected_states=expected_states,
                        representable=representable,
                    )
                    action_counts[threshold]["keep" if action is None else "retype"] += 1
                    rescues[threshold] += int(not baseline_exact and revised_exact)
                    breaks[threshold] += int(baseline_exact and not revised_exact)
                    if collect_sample_details:
                        row = rows[start + batch_index]
                        render_parameters = row["render"].get("render_parameters", {})
                        state_counts = {
                            name: int((revised == state).sum())
                            for state, name in enumerate(BOND_TYPES)
                        }
                        sample_details[threshold].append(
                            {
                                "domain": domain,
                                "sample_key": str(row["key"]),
                                "molecule_id": str(row["chemmol"]["molecule_id"]),
                                "variant": str(row["render"]["variant"]),
                                "style": {
                                    "font_family_id": render_parameters.get(
                                        "font_family_id"
                                    ),
                                    "font_size": render_parameters.get("font_size"),
                                    "bond_line_width": render_parameters.get(
                                        "bond_line_width"
                                    ),
                                    "rotation_degrees": render_parameters.get(
                                        "rotation_degrees"
                                    ),
                                    "multiple_bond_offset": render_parameters.get(
                                        "multiple_bond_offset"
                                    ),
                                    "layout": render_parameters.get("layout"),
                                    "kekulize": render_parameters.get("kekulize"),
                                    "stress_operation": render_parameters.get(
                                        "stress_operation"
                                    ),
                                },
                                "representable": bool(representable),
                                "baseline_presence_exact": bool(
                                    baseline_presence_exact
                                ),
                                "baseline_chemical_exact": bool(baseline_exact),
                                "baseline_false_positive": int(
                                    (
                                        (current_states != ABSENT_STATE)
                                        & (expected_states == ABSENT_STATE)
                                    ).sum()
                                ),
                                "baseline_false_negative": int(
                                    (
                                        (current_states == ABSENT_STATE)
                                        & (expected_states != ABSENT_STATE)
                                    ).sum()
                                ),
                                "baseline_wrong_type": int(
                                    (
                                        (current_states != ABSENT_STATE)
                                        & (expected_states != ABSENT_STATE)
                                        & (current_states != expected_states)
                                    ).sum()
                                ),
                                "presence_exact": bool(revised_presence_exact),
                                "chemical_exact": bool(revised_exact),
                                "false_positive": int(
                                    (
                                        (revised != ABSENT_STATE)
                                        & (expected_states == ABSENT_STATE)
                                    ).sum()
                                ),
                                "false_negative": int(
                                    (
                                        (revised == ABSENT_STATE)
                                        & (expected_states != ABSENT_STATE)
                                    ).sum()
                                ),
                                "wrong_type": int(
                                    (
                                        (revised != ABSENT_STATE)
                                        & (expected_states != ABSENT_STATE)
                                        & (revised != expected_states)
                                    ).sum()
                                ),
                                "predicted_type_counts": state_counts,
                                "action": (
                                    None
                                    if action is None
                                    else {
                                        "group_index": int(action.group_index),
                                        "target_type": BOND_TYPES[action.target_state],
                                    }
                                ),
                            }
                        )
    baseline = summarize_domains(_finalize_domains(baseline_raw))
    candidates = {}
    for threshold in thresholds:
        metrics = summarize_domains(_finalize_domains(threshold_raw[threshold]))
        metrics["actions"] = dict(action_counts[threshold])
        metrics["rescues"] = rescues[threshold]
        metrics["breaks"] = breaks[threshold]
        candidates[str(threshold)] = metrics
    best_threshold = max(
        thresholds, key=lambda value: selection_key(candidates[str(value)])
    )
    retype_targets = diagnostic_counts["retype"]
    action_top1 = diagnostic_counts["action_top1_correct"]
    result = {
        "baseline": baseline,
        "thresholds": candidates,
        "best_threshold": best_threshold,
        "best": candidates[str(best_threshold)],
        "mean_gate_logit": sum(gate_values) / max(len(gate_values), 1),
        "objective": objective,
        "diagnostics": {
            "target_counts": dict(diagnostic_counts),
            "mean_gate_logit_by_target": {
                name: sum(values) / max(len(values), 1)
                for name, values in gate_values_by_target.items()
            },
            "retype_action_top1_correct": action_top1,
            "retype_action_top1_rate": action_top1 / max(retype_targets, 1),
            "mean_retype_action_target_rank": (
                sum(action_target_ranks) / max(len(action_target_ranks), 1)
            ),
            "oracle_gate_with_predicted_action_chemical_exact": (
                baseline["pooled_chemical_exact"] + action_top1
            ),
            "verifier_retype_action_top1_correct": diagnostic_counts[
                "verifier_top1_correct"
            ],
            "verifier_retype_action_top1_rate": diagnostic_counts[
                "verifier_top1_correct"
            ]
            / max(retype_targets, 1),
            "mean_verifier_action_target_rank": (
                sum(verifier_target_ranks) / max(len(verifier_target_ranks), 1)
            ),
        },
    }
    if collect_sample_details:
        result["samples"] = sample_details[best_threshold]
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-move-checkpoint", default=DEFAULT_FIRST_MOVE)
    parser.add_argument("--grounded-graph-checkpoint", default=DEFAULT_GRAPH)
    parser.add_argument("--samples-per-domain", type=int, default=2_250)
    parser.add_argument("--dev-samples-per-domain", type=int, default=498)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--type-embedding-dim", type=int, default=16)
    parser.add_argument("--retype-candidates", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--gate-positive-weight", type=float, default=6.0)
    parser.add_argument("--action-weight", type=float, default=1.0)
    parser.add_argument(
        "--objective", choices=("separate", "joint", "safety", "verifier")
    )
    parser.add_argument("--shuffle-buffer", type=int, default=1024)
    parser.add_argument("--data-seed", type=int, default=20260725)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--evaluate-checkpoint")
    parser.add_argument("--evaluate-verifier-checkpoint")
    parser.add_argument("--retype-init-checkpoint")
    parser.add_argument("--validation-interval-total", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Chemical retype training requires CUDA")
    if args.batch_size % len(TRAIN_DOMAINS):
        raise ValueError("Batch size must be divisible by the domain count")
    if args.validation_interval_total < 0:
        raise ValueError("Validation interval must be non-negative")
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
    verifier_head = None
    verifier_config = None
    evaluation_modes = sum(
        bool(value)
        for value in (args.evaluate_checkpoint, args.evaluate_verifier_checkpoint)
    )
    if evaluation_modes > 1:
        raise ValueError("Only one evaluation checkpoint may be supplied")
    if evaluation_modes and args.retype_init_checkpoint:
        raise ValueError("Evaluation and training initialization are mutually exclusive")
    fit_capacity = {}
    if not evaluation_modes:
        for domain in TRAIN_DOMAINS:
            fit_capacity[domain] = count_partition_records(
                domain,
                partition="train_fit",
                config=scene_config,
                data_seed=args.data_seed,
                limit=args.samples_per_domain,
            )
            if fit_capacity[domain] < args.samples_per_domain:
                raise RuntimeError(
                    "Requested samples-per-domain exceeds strict train_fit capacity: "
                    f"requested={args.samples_per_domain}, available={fit_capacity}"
                )
        (output_dir / "fit_capacity.json").write_text(
            json.dumps(
                {
                    "requested_samples_per_domain": args.samples_per_domain,
                    "probed_train_fit_samples": fit_capacity,
                    "probe_limit": args.samples_per_domain,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    if args.evaluate_verifier_checkpoint:
        verifier_head, verifier_payload = load_atom_pair_chemical_retype_verifier(
            args.evaluate_verifier_checkpoint, device
        )
        verifier_config = verifier_head.config
        retype_path = verifier_payload["retype_init_checkpoint"]
        retype_head, initialization_payload = load_atom_pair_chemical_retype_head(
            retype_path, device
        )
        retype_config = retype_head.config
        if initialization_payload["first_move_checkpoint"] != args.first_move_checkpoint:
            raise ValueError("Verifier checkpoint uses a different first-move model")
        objective = "verifier"
    elif args.evaluate_checkpoint:
        retype_head, evaluation_payload = load_atom_pair_chemical_retype_head(
            args.evaluate_checkpoint, device
        )
        retype_config = retype_head.config
        if evaluation_payload["first_move_checkpoint"] != args.first_move_checkpoint:
            raise ValueError("Evaluation checkpoint uses a different first-move model")
        objective = evaluation_payload.get(
            "objective",
            evaluation_payload.get("training_args", {}).get("objective") or "separate",
        )
    elif args.retype_init_checkpoint:
        retype_head, initialization_payload = load_atom_pair_chemical_retype_head(
            args.retype_init_checkpoint, device
        )
        retype_head.train().requires_grad_(True)
        retype_config = retype_head.config
        if initialization_payload["first_move_checkpoint"] != args.first_move_checkpoint:
            raise ValueError("Initialization checkpoint uses a different first-move model")
        objective = args.objective or initialization_payload.get(
            "objective",
            initialization_payload.get("training_args", {}).get("objective")
            or "separate",
        )
        if objective == "safety":
            retype_head.requires_grad_(False)
            retype_head.gate_head.requires_grad_(True)
        elif objective == "verifier":
            retype_head.eval().requires_grad_(False)
            verifier_config = AtomPairChemicalRetypeVerifierConfig(
                hidden_dim=args.hidden_dim,
                type_embedding_dim=args.type_embedding_dim,
            )
            verifier_head = AtomPairChemicalRetypeVerifier(verifier_config).to(device)
            initialize_verifier_from_retype(verifier_head, retype_head)
    else:
        retype_config = AtomPairChemicalRetypeConfig(
            hidden_dim=args.hidden_dim,
            type_embedding_dim=args.type_embedding_dim,
            retype_candidates=args.retype_candidates,
        )
        retype_head = AtomPairChemicalRetypeHead(retype_config).to(device)
        initialize_retype_encoder_from_one_move(retype_head, first_head)
        objective = args.objective or "joint"
        if objective == "verifier":
            raise ValueError("Verifier training requires --retype-init-checkpoint")

    trainable_head = verifier_head if objective == "verifier" else retype_head

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
        retype_head,
        relation_config,
        dev_records,
        collator,
        proposal_config,
        device,
        args.batch_size,
        objective=objective,
        verifier_head=verifier_head,
    )
    if evaluation_modes:
        result = {
            "evaluation_checkpoint": (
                args.evaluate_verifier_checkpoint or args.evaluate_checkpoint
            ),
            "metrics": initial,
        }
        (output_dir / "metrics.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, separators=(",", ":")), flush=True)
        return
    (output_dir / "initial_metrics.json").write_text(
        json.dumps(initial, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"initial_train_dev": initial}, separators=(",", ":")), flush=True)

    optimizer = torch.optim.AdamW(
        trainable_head.parameters(), lr=args.learning_rate, weight_decay=0.0
    )
    running = defaultdict(float)
    running_counts = Counter()
    running_eligible = 0
    total_seen = 0
    fit_ids: set[str] = set()
    started = time.perf_counter()
    last_validated_seen = 0
    best_checkpoint_key = None
    best_metrics = None
    validation_history = []

    def partition_audit() -> dict:
        overlap = fit_ids & dev_ids
        if overlap:
            raise RuntimeError(f"train_fit/train_dev molecule overlap: {len(overlap)}")
        return {
            "protocol": "sha256(molecule_id) modulo 5; bucket 0 is train_dev",
            "data_seed": args.data_seed,
            "fit_unique_molecules": len(fit_ids),
            "dev_unique_molecules": len(dev_ids),
            "overlap": 0,
            "fit_sha256": _identity_sha256(fit_ids),
            "dev_sha256": _identity_sha256(dev_ids),
        }

    def validate_and_checkpoint(stage: str) -> dict:
        nonlocal last_validated_seen, best_checkpoint_key, best_metrics
        audit = partition_audit()
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
            retype_head,
            relation_config,
            dev_records,
            collator,
            proposal_config,
            device,
            args.batch_size,
            objective=objective,
            verifier_head=verifier_head,
        )
        metrics.update(
            {
                "total_seen": total_seen,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        checkpoint = {
            "checkpoint_schema_version": (
                "atom-pair-chemical-retype-verifier-0.1"
                if objective == "verifier"
                else "atom-pair-chemical-retype-0.1"
            ),
            "head": trainable_head.state_dict(),
            "config": asdict(
                verifier_config if objective == "verifier" else retype_config
            ),
            "relation_config": asdict(relation_config),
            "first_move_checkpoint": args.first_move_checkpoint,
            "base_context_checkpoint": first_payload["base_context_checkpoint"],
            "group_checkpoint": context_payload["group_checkpoint"],
            "source_checkpoint": group_payload["source_checkpoint"],
            "retype_init_checkpoint": args.retype_init_checkpoint,
            "samples_seen": total_seen,
            "objective": objective,
            "gate_threshold": metrics["best_threshold"],
            "train_dev": metrics,
            "partition_audit": audit,
            "fit_capacity": fit_capacity,
            "training_args": vars(args),
            "selection_stage": stage,
        }
        step_name = f"step-{total_seen:06d}"
        torch.save(checkpoint, output_dir / f"{step_name}.pt")
        (output_dir / f"{step_name}-metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        current_key = selection_key(metrics["best"])
        selected = best_checkpoint_key is None or current_key > best_checkpoint_key
        if selected:
            best_checkpoint_key = current_key
            best_metrics = metrics
            torch.save(checkpoint, output_dir / "best.pt")
            (output_dir / "metrics.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        validation_history.append(
            {
                "stage": stage,
                "total_seen": total_seen,
                "selected_best": selected,
                "best_threshold": metrics["best_threshold"],
                "minimum_domain_chemical_exact_rate": metrics["best"][
                    "minimum_domain_chemical_exact_rate"
                ],
                "pooled_chemical_exact": metrics["best"]["pooled_chemical_exact"],
                "rescues": metrics["best"]["rescues"],
                "breaks": metrics["best"]["breaks"],
            }
        )
        (output_dir / "validation_history.json").write_text(
            json.dumps(validation_history, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "validation_stage": stage,
                    "selected_best": selected,
                    "train_dev": metrics,
                },
                separators=(",", ":"),
            ),
            flush=True,
        )
        last_validated_seen = total_seen
        return metrics

    for samples, domain_seen in balanced_fit_batches(
        samples_per_domain=args.samples_per_domain,
        batch_size=args.batch_size,
        config=scene_config,
        data_seed=args.data_seed,
        shuffle_buffer=args.shuffle_buffer,
    ):
        fit_ids.update(sample_molecule_id(sample) for sample in samples)
        trainable_head.train()
        if objective == "verifier":
            retype_head.eval()
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
        sample_losses = []
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
                retype_features = build_retype_features(
                    relation[sample], sample_logits, chemical_logits, selected
                )
                if objective == "verifier":
                    with torch.no_grad():
                        gate, action_scores, actions = retype_head.score_sample(
                            retype_features,
                            chemical_logits,
                            selected,
                        )
                    if not actions:
                        batch_counts["no_retype_action"] += 1
                        continue
                    verifier_scores = verifier_head.score_sample(
                        retype_features,
                        chemical_logits,
                        selected,
                        gate,
                        action_scores,
                        actions,
                    )
                    labels = candidate_exact_labels(
                        actions, current_states, expected_states
                    )
                    has_exact_action = bool((labels > 0.5).any())
                    batch_counts[
                        "exact_candidate" if has_exact_action else "keep_only"
                    ] += 1
                    losses = chemical_retype_verifier_loss(
                        verifier_scores,
                        labels,
                        ranking_margin=verifier_config.ranking_margin,
                    )
                else:
                    gate, action_scores, actions = retype_head.score_sample(
                        retype_features,
                        chemical_logits,
                        selected,
                    )
                if objective == "safety":
                    if not actions:
                        batch_counts["no_retype_action"] += 1
                        continue
                    proposed = actions[int(action_scores.argmax())]
                    revised = apply_retype_action(current_states, proposed)
                    safe_target = int(
                        not bool((current_states == expected_states).all())
                        and bool((revised == expected_states).all())
                    )
                    batch_counts["safe_retype" if safe_target else "unsafe_keep"] += 1
                    losses = chemical_retype_safety_loss(
                        gate,
                        action_scores,
                        safe_target,
                        positive_weight=args.gate_positive_weight,
                    )
                elif objective != "verifier":
                    target = target_retype_action_index(
                        actions, current_states, expected_states
                    )
                    if target is None:
                        batch_counts["not_keep_or_one_retype"] += 1
                        continue
                    gate_target, action_target = target
                    batch_counts["retype" if gate_target else "keep"] += 1
                    if objective == "joint":
                        losses = chemical_retype_joint_loss(
                            gate,
                            gate_target,
                            action_scores,
                            action_target,
                            retype_weight=args.gate_positive_weight,
                        )
                    else:
                        losses = chemical_retype_loss(
                            gate,
                            gate_target,
                            action_scores,
                            action_target,
                            gate_positive_weight=args.gate_positive_weight,
                            action_weight=args.action_weight,
                        )
                sample_losses.append(losses)
        if sample_losses:
            loss = torch.stack([values["loss"] for values in sample_losses]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_head.parameters(), 1.0)
            optimizer.step()
            loss_names = (
                ("loss", "positive_loss", "negative_loss", "ranking_loss")
                if objective == "verifier"
                else (
                    ("loss", "safety_loss")
                    if objective == "safety"
                    else (
                        ("loss", "joint_loss")
                        if objective == "joint"
                        else ("loss", "gate_loss", "action_loss")
                    )
                )
            )
            for name in loss_names:
                running[name] += sum(
                    float(values[name].detach()) for values in sample_losses
                )
            running_eligible += len(sample_losses)
        running_counts.update(batch_counts)
        total_seen += len(samples)
        if total_seen % 900 < len(samples):
            print(
                json.dumps(
                    {
                        "total_seen": total_seen,
                        "domain_seen": domain_seen,
                        "eligible_samples": running_eligible,
                        **{
                            name: value / max(running_eligible, 1)
                            for name, value in running.items()
                        },
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
        if validation_checkpoint_due(
            total_seen,
            last_validated_seen,
            args.validation_interval_total,
        ):
            validate_and_checkpoint("interval")

    if validation_checkpoint_due(
        total_seen,
        last_validated_seen,
        args.validation_interval_total,
        final=True,
    ):
        validate_and_checkpoint("final")
    final_audit = partition_audit()
    (output_dir / "partition_audit.json").write_text(
        json.dumps(final_audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if best_metrics is None:
        raise RuntimeError("Training completed without a validation checkpoint")


if __name__ == "__main__":
    main()
