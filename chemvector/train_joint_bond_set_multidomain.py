#!/usr/bin/env python3
"""Jointly train Bond proposals and set/incidence with a committed fresh dev."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import random
import stat
import subprocess
import time

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.data import ChemVectorIterableDataset
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
    bond_targets,
    load_validation_records,
    rejection_reason,
    sample_split,
)
from chemvector.train_proposal_object_set import evaluate
from chemvector.train_spatial_incidence import DEFAULT_GRAPH, forward_frozen_bond_head
from chemvector.train_stable_bond_presence import domain_validation


DEFAULT_SOURCE = (
    "outputs/checkpoints/chemvector-v0972-bond-presence-calibration-10k/best.pt"
)
DEFAULT_OUTPUT = "outputs/checkpoints/chemvector-v0100-joint-bond-set-style50k"
JOINT_ROOT = Path("Data/sdf/chemvector_joint_v1")
STYLE_COMMITMENT = "2082372d253fa309715becfd5cba9e5cdb78eb8d84a5ece4ff1eb73bd67390eb"
FRESH_COMMITMENT = "13a456afb2c475a73fc4dca82df5fe62d4a31277d6b584637d9cd92736024512"

STYLE_TRAIN_SHARDS = [
    str(path)
    for path in sorted((JOINT_ROOT / "style_train_100k/shards").glob("train-*.tar"))
]
CHEMVISION_TRAIN_SHARDS = [
    f"Data/sdf/chemvision_layers_v0_1/pilot_10k/shards/train-{index:06d}.tar"
    for index in range(3)
]
FONT_TRAIN_SHARDS = [
    f"Data/sdf/chemvision_layers_v0_2/font_pilot_10k/shards/train-{index:06d}.tar"
    for index in range(3)
]
TRAIN_DOMAINS = {
    "style_train": STYLE_TRAIN_SHARDS,
    "chemvision_replay": CHEMVISION_TRAIN_SHARDS,
    "font_v2_replay": FONT_TRAIN_SHARDS,
}
DEFAULT_DOMAIN_SAMPLE_COUNTS = {
    "style_train": 33_334,
    "chemvision_replay": 8_333,
    "font_v2_replay": 8_333,
}
DEFAULT_DOMAIN_BATCH_COUNTS = {
    "style_train": 4,
    "chemvision_replay": 1,
    "font_v2_replay": 1,
}
FRESH_DEV_SHARDS = [
    str(path)
    for path in sorted(
        (JOINT_ROOT / "fresh_eval_12k/shards").glob("fresh_dev-*.tar")
    )
]
REGRESSION_DOMAINS = {
    "chemvision": (
        "Data/sdf/chemvision_layers_v0_1/pilot_10k/shards/validation-000000.tar"
    ),
    "font_v2": (
        "Data/sdf/chemvision_layers_v0_2/font_pilot_10k/shards/validation-000000.tar"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", default=DEFAULT_SOURCE)
    parser.add_argument("--grounded-graph-checkpoint", default=DEFAULT_GRAPH)
    parser.add_argument("--style-train-samples", type=int, default=33_334)
    parser.add_argument("--chemvision-replay-samples", type=int, default=8_333)
    parser.add_argument("--font-v2-replay-samples", type=int, default=8_333)
    parser.add_argument("--fresh-dev-samples", type=int, default=8_000)
    parser.add_argument("--fresh-probe-samples", type=int, default=1_000)
    parser.add_argument("--regression-samples-per-domain", type=int, default=498)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--bond-learning-rate", type=float, default=1e-6)
    parser.add_argument("--presence-learning-rate", type=float, default=2e-6)
    parser.add_argument("--incidence-learning-rate", type=float, default=2e-7)
    parser.add_argument("--bond-loss-weight", type=float, default=1.0)
    parser.add_argument("--set-loss-weight", type=float, default=1.0)
    parser.add_argument("--graph-exact-surrogate-weight", type=float, default=0.1)
    parser.add_argument("--validate-every-total-samples", type=int, default=10_000)
    parser.add_argument("--log-every-total-samples", type=int, default=1_000)
    parser.add_argument("--shuffle-buffer", type=int, default=1024)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260729)
    return parser.parse_args()


def verify_data_gate() -> dict:
    paths = {
        "root_validation": JOINT_ROOT / "validation.json",
        "style_validation": JOINT_ROOT / "style_train_100k/validation.json",
        "fresh_validation": JOINT_ROOT / "fresh_eval_12k/validation.json",
        "style_commitment": JOINT_ROOT / "style_train_100k/commitment.json",
        "fresh_commitment": JOINT_ROOT / "fresh_eval_12k/commitment.json",
    }
    payloads = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in paths.items()
    }
    for name in ("root_validation", "style_validation", "fresh_validation"):
        if payloads[name].get("status") != "passed":
            raise ValueError(f"Data gate is not passed: {paths[name]}")
    observed = {
        "style_train_100k": payloads["style_commitment"].get("commitment_sha256"),
        "fresh_eval_12k": payloads["fresh_commitment"].get("commitment_sha256"),
    }
    expected = {
        "style_train_100k": STYLE_COMMITMENT,
        "fresh_eval_12k": FRESH_COMMITMENT,
    }
    if observed != expected:
        raise ValueError(f"Data commitment mismatch: observed={observed} expected={expected}")
    sealed = JOINT_ROOT / "fresh_eval_12k/sealed_test"
    sealed_mode = stat.S_IMODE(sealed.stat().st_mode)
    if sealed_mode != 0:
        raise ValueError(f"sealed_test must remain mode 000, found {sealed_mode:o}")
    return {
        "status": "passed",
        "commitments": observed,
        "sealed_test_mode": f"{sealed_mode:03o}",
        "sealed_test_read": False,
    }


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
        serialize_target=False,
        load_raster_layers=False,
    )
    for sample in dataset:
        if sample_split(sample) != "train":
            raise ValueError("Joint training stream contains a non-train sample")
        if rejection_reason(sample, config) is None:
            yield sample


def mixed_batches(
    *,
    domain_sample_counts: dict[str, int],
    domain_batch_counts: dict[str, int],
    batch_size: int,
    config: AtomBondSceneConfig,
    seed: int,
    shuffle_buffer: int,
):
    if set(domain_sample_counts) != set(TRAIN_DOMAINS):
        raise ValueError("Domain sample counts must cover every training domain")
    if set(domain_batch_counts) != set(TRAIN_DOMAINS):
        raise ValueError("Domain batch counts must cover every training domain")
    if any(value < 1 for value in domain_sample_counts.values()):
        raise ValueError("Domain sample counts must be positive")
    if sum(domain_batch_counts.values()) != batch_size:
        raise ValueError("Domain batch counts must sum to batch size")
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
    while any(seen[name] < domain_sample_counts[name] for name in streams):
        rows = []
        for domain, stream in streams.items():
            remaining = domain_sample_counts[domain] - seen[domain]
            take = min(domain_batch_counts[domain], remaining)
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


def load_fresh_records(
    *,
    expected_samples: int,
    config: AtomBondSceneConfig,
) -> tuple[list[dict], dict]:
    records = []
    collision_samples = 0
    for sample in ChemVectorIterableDataset(
        FRESH_DEV_SHARDS,
        sequence_version="0.4",
        serialize_target=False,
        load_raster_layers=False,
    ):
        if sample_split(sample) != "fresh_dev":
            raise ValueError("Fresh-dev shard contains another split")
        reason = rejection_reason(sample, config)
        if reason == "atom_cell_collision":
            collision_samples += 1
        elif reason is not None:
            raise ValueError(f"Fresh-dev sample is outside model capacity: {reason}")
        records.append(sample)
        if len(records) >= expected_samples:
            break
    if len(records) != expected_samples:
        raise ValueError(
            f"Fresh-dev loader yielded {len(records)}, expected {expected_samples}"
        )
    return records, {
        "expected": expected_samples,
        "loaded": len(records),
        "atom_cell_collision_samples_included": collision_samples,
        "coordinate_collision_samples_included": (
            26 if expected_samples == 8_000 else None
        ),
        "target_serialization_disabled": True,
    }


def checkpoint_payload(
    *,
    source: dict,
    bond,
    set_head,
    optimizer,
    args: argparse.Namespace,
    total_seen: int,
    domain_seen: dict[str, int],
    validation: dict,
    data_gate: dict,
) -> dict:
    return {
        **{
            key: value
            for key, value in source.items()
            if key
            not in {
                "optimizer",
                "validation",
                "samples_seen",
                "bond_head",
                "set_head",
            }
        },
        "checkpoint_schema_version": "joint-proposal-objects-checkpoint-1.0",
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
        "joint_multidomain_domain_seen": domain_seen,
        "validation": validation,
        "joint_multidomain_args": vars(args),
        "data_gate": data_gate,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Joint Bond/set training requires CUDA")
    domain_sample_counts = {
        "style_train": args.style_train_samples,
        "chemvision_replay": args.chemvision_replay_samples,
        "font_v2_replay": args.font_v2_replay_samples,
    }
    domain_batch_counts = dict(DEFAULT_DOMAIN_BATCH_COUNTS)
    if min(*domain_sample_counts.values(), args.batch_size) < 1:
        raise ValueError("Training sizes must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    data_gate = verify_data_gate()
    code_git_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_args.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "selection_protocol.json").write_text(
        json.dumps(
            {
                "schema_version": "chemvector-fresh-selection-1.0",
                "fixed_before_fresh_dev_read": True,
                "primary_selection": [
                    "fresh_probe.full_object_exact_rate",
                    "fresh_probe.bond_deployed_exact_rate",
                    "fresh_probe.bond_recall",
                    "fresh_probe.incidence_exact_rate",
                ],
                "candidate_gate": {
                    "fresh_bond_deployed_exact_delta_min": 0.02,
                    "fresh_full_object_exact_delta_min": 0.0,
                    "old_minimum_full_object_exact_delta_min": -0.02,
                },
                "data_gate": data_gate,
                "code_git_revision": code_git_revision,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
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
    fresh_records, fresh_audit = load_fresh_records(
        expected_samples=args.fresh_dev_samples,
        config=scene_config,
    )
    if not 1 <= args.fresh_probe_samples <= len(fresh_records):
        raise ValueError("Fresh probe size must be within the fresh-dev set")
    fresh_probe_records = fresh_records[: args.fresh_probe_samples]
    regression_records = {}
    regression_audits = {}
    for domain, shard in REGRESSION_DOMAINS.items():
        regression_records[domain], regression_audits[domain] = load_validation_records(
            [shard],
            per_shard=args.regression_samples_per_domain,
            config=scene_config,
        )
    (output_dir / "input_audit.json").write_text(
        json.dumps(
            {
                "data_gate": data_gate,
                "fresh_dev": fresh_audit,
                "regression": regression_audits,
                "train_domain_sample_counts": domain_sample_counts,
                "train_domain_batch_counts": domain_batch_counts,
                "code_git_revision": code_git_revision,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    atom_threshold = float(source["training_args"]["atom_decision_threshold"])
    bond_threshold = float(source["training_args"]["bond_decision_threshold"])

    def evaluate_fresh(rows: list[dict]) -> dict:
        return evaluate(
            vision,
            atom,
            bond,
            set_head,
            rows,
            collator,
            proposal_config,
            device,
            args.batch_size,
            atom_threshold,
            bond_threshold,
        )

    def evaluate_regression() -> dict:
        return domain_validation(
            vision,
            atom,
            bond,
            set_head,
            regression_records,
            collator,
            proposal_config,
            device,
            args.batch_size,
            atom_threshold,
            bond_threshold,
        )

    initial_fresh = evaluate_fresh(fresh_records)
    initial_regression = evaluate_regression()
    initial = {"fresh_dev": initial_fresh, "regression": initial_regression}
    (output_dir / "initial_metrics.json").write_text(
        json.dumps(initial, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"initial_validation": initial}, separators=(",", ":")), flush=True)

    incidence_parameters = list(set_head.incidence.parameters())
    incidence_ids = {id(parameter) for parameter in incidence_parameters}
    presence_parameters = [
        parameter
        for parameter in set_head.parameters()
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
    window = defaultdict(float)
    total_seen = 0
    window_seen = 0
    steps = 0
    next_log = args.log_every_total_samples
    next_validation = args.validate_every_total_samples
    best_score = None
    best_validation = None
    started = time.perf_counter()

    for samples, domain_seen in mixed_batches(
        domain_sample_counts=domain_sample_counts,
        domain_batch_counts=domain_batch_counts,
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
        window_seen += count
        steps += 1
        for name, value in (
            ("loss", loss),
            ("bond_loss", bond_losses["loss"]),
            ("set_loss", set_losses["loss"]),
            ("graph_exact_surrogate_loss", set_losses["graph_exact_surrogate_loss"]),
        ):
            scalar = float(value.detach()) * count
            totals[name] += scalar
            window[name] += scalar

        if total_seen >= next_log:
            print(
                json.dumps(
                    {
                        "samples_seen": total_seen,
                        "domain_seen": domain_seen,
                        "mean_losses": {
                            name: value / max(window_seen, 1)
                            for name, value in window.items()
                        },
                        "elapsed_seconds": time.perf_counter() - started,
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
            window.clear()
            window_seen = 0
            next_log += args.log_every_total_samples

        if total_seen >= next_validation or total_seen == sum(domain_sample_counts.values()):
            validation = {
                "fresh_probe": evaluate_fresh(fresh_probe_records),
                "regression": evaluate_regression(),
                "samples_seen": total_seen,
                "domain_seen": domain_seen,
                "elapsed_seconds": time.perf_counter() - started,
            }
            payload = checkpoint_payload(
                source=source,
                bond=bond,
                set_head=set_head,
                optimizer=optimizer,
                args=args,
                total_seen=total_seen,
                domain_seen=domain_seen,
                validation=validation,
                data_gate=data_gate,
            )
            payload["code_git_revision"] = code_git_revision
            torch.save(payload, output_dir / f"samples-{total_seen:06d}.pt")
            torch.save(payload, output_dir / "last.pt")
            fresh_probe = validation["fresh_probe"]
            score = (
                fresh_probe["full_object_exact_rate"],
                fresh_probe["bond_deployed_exact_rate"],
                fresh_probe["bond_recall"],
                fresh_probe["incidence_exact_rate"],
            )
            if best_score is None or score > best_score:
                best_score = score
                best_validation = validation
                torch.save(payload, output_dir / "best.pt")
            (output_dir / "metrics.json").write_text(
                json.dumps(
                    {"latest": validation, "best": best_validation},
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(json.dumps({"validation": validation}, separators=(",", ":")), flush=True)
            next_validation += args.validate_every_total_samples

    selected = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=False)
    bond.load_state_dict(selected["bond_head"], strict=True)
    set_head.load_state_dict(selected["set_head"], strict=True)
    selected_fresh = evaluate_fresh(fresh_records)
    selected_regression = evaluate_regression()
    fresh_bond_delta = (
        selected_fresh["bond_deployed_exact_rate"]
        - initial_fresh["bond_deployed_exact_rate"]
    )
    fresh_full_delta = (
        selected_fresh["full_object_exact_rate"]
        - initial_fresh["full_object_exact_rate"]
    )
    regression_delta = (
        selected_regression["minimum_domain_full_object_exact_rate"]
        - initial_regression["minimum_domain_full_object_exact_rate"]
    )
    candidate_gate = {
        "fresh_bond_deployed_exact_delta": fresh_bond_delta,
        "fresh_full_object_exact_delta": fresh_full_delta,
        "old_minimum_full_object_exact_delta": regression_delta,
        "passed": (
            fresh_bond_delta >= 0.02
            and fresh_full_delta >= 0.0
            and regression_delta >= -0.02
        ),
    }
    final_report = {
        "initial": initial,
        "selected_checkpoint_samples": selected["joint_multidomain_samples"],
        "selected": {
            "fresh_dev": selected_fresh,
            "regression": selected_regression,
        },
        "candidate_gate": candidate_gate,
        "mean_train_losses": {
            name: value / max(total_seen, 1) for name, value in totals.items()
        },
        "total_seen": total_seen,
        "domain_seen": domain_seen,
        "elapsed_seconds": time.perf_counter() - started,
        "sealed_test_read": False,
    }
    (output_dir / "fresh_dev_metrics.json").write_text(
        json.dumps(final_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    selected["fresh_dev_full"] = selected_fresh
    selected["regression_full"] = selected_regression
    selected["candidate_gate"] = candidate_gate
    torch.save(selected, output_dir / "selected.pt")
    print(json.dumps({"final": final_report}, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
