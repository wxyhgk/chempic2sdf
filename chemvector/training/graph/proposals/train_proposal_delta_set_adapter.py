#!/usr/bin/env python3
"""Train a zero-init set/incidence adapter from residual Bond proposal deltas."""

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
from chemvector.dense_bond_object import BondProposalResidualAdapter
from chemvector.evaluate_style_sensitivity import load_style_records
from chemvector.predicted_incidence import (
    PredictedProposalConfig,
    associate_predicted_proposals,
    decode_predicted_atom_proposals,
    decode_predicted_bond_proposals,
)
from chemvector.proposal_delta_set_adapter import (
    ProposalDeltaSetAdapter,
    ProposalDeltaSetConfig,
    proposal_deltas,
)
from chemvector.proposal_object_set import proposal_object_set_loss
from chemvector.train_atom_bond_scene import AtomBondSceneCollator, stack_targets
from chemvector.train_atom_pair_bond_group import VALIDATION_DOMAINS
from chemvector.train_atom_pair_style_consistency import (
    load_paired_style_training_records,
    load_replay_records,
)
from chemvector.train_bond_presence_calibration import load_components
from chemvector.train_joint_proposal_objects_full import load_validation_records
from chemvector.train_proposal_object_set import _sample_set_metrics
from chemvector.train_set_incidence_dual_proposal import (
    DEFAULT_SOURCE,
    DEFAULT_STYLE_ROOT,
)
from chemvector.train_spatial_incidence import DEFAULT_GRAPH, forward_frozen_bond_head


DEFAULT_OUTPUT = (
    "outputs/checkpoints/chemvector-v0997-proposal-delta-set-adapter-pilot250"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", default=DEFAULT_SOURCE)
    parser.add_argument("--grounded-graph-checkpoint", default=DEFAULT_GRAPH)
    parser.add_argument("--style-root", default=str(DEFAULT_STYLE_ROOT))
    parser.add_argument("--style-molecules", type=int, default=250)
    parser.add_argument("--style-candidate-reserve", type=int, default=64)
    parser.add_argument("--replay-samples-per-domain", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--style-molecules-per-batch", type=int, default=2)
    parser.add_argument("--replay-batch-size", type=int, default=8)
    parser.add_argument("--validation-samples-per-domain", type=int, default=498)
    parser.add_argument("--style-validation-samples", type=int, default=0)
    parser.add_argument("--validation-batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--replay-weight", type=float, default=1.0)
    parser.add_argument("--residual-weight", type=float, default=1e-4)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260726)
    return parser.parse_args()


def summarize_set_totals(totals: dict[str, int]) -> dict:
    samples = max(totals["samples"], 1)

    def rate(numerator: str, denominator: str) -> float:
        return totals[numerator] / max(totals[denominator], 1)

    atom_precision_denominator = totals["atom_true_positive"] + totals["atom_false_positive"]
    bond_precision_denominator = totals["bond_true_positive"] + totals["bond_false_positive"]
    return {
        **totals,
        "atom_proposal_recall": rate("atom_matched", "atom_targets"),
        "atom_precision": totals["atom_true_positive"] / max(atom_precision_denominator, 1),
        "atom_recall": rate("atom_true_positive", "atom_targets"),
        "atom_candidate_exact_rate": totals["atom_candidate_exact"] / samples,
        "atom_deployed_exact_rate": totals["atom_deployed_exact"] / samples,
        "bond_proposal_recall": rate("bond_matched", "bond_targets"),
        "bond_precision": totals["bond_true_positive"] / max(bond_precision_denominator, 1),
        "bond_recall": rate("bond_true_positive", "bond_targets"),
        "bond_candidate_exact_rate": totals["bond_candidate_exact"] / samples,
        "bond_deployed_exact_rate": totals["bond_deployed_exact"] / samples,
        "incidence_endpoint_accuracy": rate("incidence_endpoint_correct", "incidence_endpoints"),
        "incidence_bond_accuracy": rate("incidence_bond_correct", "incidence_bonds"),
        "incidence_exact_rate": totals["incidence_exact"] / samples,
        "full_object_exact_rate": totals["full_object_exact"] / samples,
        "mean_atom_selected": totals["atom_selected"] / samples,
        "mean_bond_selected": totals["bond_selected"] / samples,
    }


def _candidate_batch(
    samples: list[dict],
    *,
    vision,
    atom,
    candidate_bond,
    original_bond,
    set_head,
    adapter,
    collator,
    proposal_config,
    device: torch.device,
) -> tuple[dict, dict, dict]:
    batch = collator(samples)
    pixels = batch["pixel_values"].to(device, non_blocking=True)
    targets = stack_targets(batch, device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        features = vision.forward_stroke_resolution(pixels)
        atom_output = atom(features)
        candidate_dense = forward_frozen_bond_head(vision, candidate_bond, features)
        original_dense = forward_frozen_bond_head(vision, original_bond, features)
        proposals = associate_predicted_proposals(
            decode_predicted_atom_proposals(atom_output, proposal_config),
            decode_predicted_bond_proposals(candidate_dense, proposal_config),
            targets,
            proposal_config,
        )
        base = set_head(
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
        deltas = proposal_deltas(proposals, original_dense)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        adjusted = adapter(base, proposals, deltas)
        adjusted["proposal_delta_score"] = deltas["score"]
        adjusted["proposal_delta_endpoints"] = deltas["endpoints"]
    return adjusted, proposals, targets


@torch.no_grad()
def evaluate_delta(
    records: list[dict],
    *,
    vision,
    atom,
    candidate_bond,
    original_bond,
    set_head,
    adapter,
    collator,
    proposal_config,
    device: torch.device,
    batch_size: int,
    atom_threshold: float,
    bond_threshold: float,
) -> dict:
    totals: defaultdict[str, int] = defaultdict(int)
    adapter.eval()
    for start in range(0, len(records), batch_size):
        rows = records[start : start + batch_size]
        output, proposals, targets = _candidate_batch(
            rows,
            vision=vision,
            atom=atom,
            candidate_bond=candidate_bond,
            original_bond=original_bond,
            set_head=set_head,
            adapter=adapter,
            collator=collator,
            proposal_config=proposal_config,
            device=device,
        )
        for batch_index in range(len(rows)):
            metrics = _sample_set_metrics(
                output,
                proposals,
                targets,
                batch_index,
                atom_threshold=atom_threshold,
                bond_threshold=bond_threshold,
            )
            totals["samples"] += 1
            for name, value in metrics.items():
                totals[name] += int(value)
    return summarize_set_totals(dict(totals))


def domain_delta_validation(records: dict[str, list[dict]], **kwargs) -> dict:
    domains = {
        name: evaluate_delta(rows, **kwargs) for name, rows in records.items()
    }
    samples = sum(item["samples"] for item in domains.values())
    pooled = {"samples": samples}
    for count_name, rate_name in (
        ("atom_deployed_exact", "atom_deployed_exact_rate"),
        ("bond_deployed_exact", "bond_deployed_exact_rate"),
        ("incidence_exact", "incidence_exact_rate"),
        ("full_object_exact", "full_object_exact_rate"),
    ):
        pooled[count_name] = sum(item[count_name] for item in domains.values())
        pooled[rate_name] = pooled[count_name] / max(samples, 1)
    return {"domains": domains, "pooled": pooled}


def selection_key(metrics: dict, baseline: dict) -> tuple[float, ...]:
    deltas = []
    for domain, expected in baseline["official"]["domains"].items():
        actual = metrics["official"]["domains"][domain]
        deltas.extend(
            (
                actual["bond_deployed_exact"] - expected["bond_deployed_exact"],
                actual["full_object_exact"] - expected["full_object_exact"],
            )
        )
    safe = min(deltas) >= 0
    style = metrics["style"]["domains"]["style"]
    return (
        float(safe),
        float(style["full_object_exact"]) if safe else float(min(deltas)),
        float(style["bond_deployed_exact"]),
        float(metrics["official"]["pooled"]["full_object_exact"]),
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Proposal-delta adapter training requires CUDA")
    if args.replay_batch_size % 2:
        raise ValueError("Replay batch size must be even")
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
        json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    vision, atom, candidate_bond, set_head, source = load_components(
        args.source_checkpoint, args.grounded_graph_checkpoint, device
    )
    if not isinstance(candidate_bond, BondProposalResidualAdapter):
        raise ValueError("Source checkpoint does not contain a Bond proposal residual")
    original_bond = candidate_bond.base_detector
    for module in (vision, atom, candidate_bond, original_bond, set_head):
        module.eval().requires_grad_(False)
    adapter = ProposalDeltaSetAdapter(
        ProposalDeltaSetConfig(hidden_dim=args.hidden_dim)
    ).to(device)
    proposal_config = PredictedProposalConfig(**source["proposal_config"])
    scene_config = AtomBondSceneConfig(
        atom_queries=proposal_config.maximum_atoms,
        bond_queries=proposal_config.maximum_bonds,
    )
    collator = AtomBondSceneCollator(scene_config)
    style_training, style_training_audit = load_paired_style_training_records(
        Path(args.style_root),
        molecule_count=args.style_molecules,
        reserve=args.style_candidate_reserve,
        seed=args.seed,
        config=scene_config,
    )
    replay_records, replay_audit = load_replay_records(
        samples_per_domain=args.replay_samples_per_domain,
        seed=args.seed,
        config=scene_config,
    )
    style_validation, style_validation_audit = load_style_records(
        str(Path(args.style_root) / "shards" / "validation-000000.tar"),
        split="validation",
        samples=args.style_validation_samples,
        config=scene_config,
    )
    official_records = {}
    official_audit = {}
    for domain, shard in VALIDATION_DOMAINS.items():
        official_records[domain], official_audit[domain] = load_validation_records(
            [shard],
            per_shard=args.validation_samples_per_domain,
            config=scene_config,
        )
    (output_dir / "data_audit.json").write_text(
        json.dumps(
            {
                "style_training": style_training_audit,
                "replay": replay_audit,
                "style_validation": style_validation_audit,
                "official_validation": official_audit,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    atom_threshold = float(source["training_args"]["atom_decision_threshold"])
    bond_threshold = float(source["training_args"]["bond_decision_threshold"])
    validation_kwargs = {
        "vision": vision,
        "atom": atom,
        "candidate_bond": candidate_bond,
        "original_bond": original_bond,
        "set_head": set_head,
        "adapter": adapter,
        "collator": collator,
        "proposal_config": proposal_config,
        "device": device,
        "batch_size": args.validation_batch_size,
        "atom_threshold": atom_threshold,
        "bond_threshold": bond_threshold,
    }

    def validate() -> dict:
        return {
            "official": domain_delta_validation(official_records, **validation_kwargs),
            "style": domain_delta_validation({"style": style_validation}, **validation_kwargs),
        }

    initial = validate()
    (output_dir / "initial_metrics.json").write_text(
        json.dumps(initial, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"initial": initial}, separators=(",", ":")), flush=True)
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=args.learning_rate, weight_decay=0.0
    )
    trainable = list(adapter.parameters())
    best_key = selection_key(initial, initial)
    best_metrics = initial
    history = []
    style_seen = replay_seen = 0
    started = time.perf_counter()

    def checkpoint_payload(epoch: int, metrics: dict) -> dict:
        return {
            "checkpoint_schema_version": "proposal-delta-set-adapter-0.1",
            "head": adapter.state_dict(),
            "config": asdict(adapter.config),
            "source_checkpoint": args.source_checkpoint,
            "samples_seen": style_seen + replay_seen,
            "style_samples_seen": style_seen,
            "replay_samples_seen": replay_seen,
            "validation": metrics,
            "training_args": vars(args),
            "epoch": epoch,
        }

    torch.save(checkpoint_payload(0, initial), output_dir / "initial.pt")
    torch.save(checkpoint_payload(0, initial), output_dir / "best.pt")
    per_domain_replay = args.replay_batch_size // len(replay_records)

    for epoch in range(1, args.epochs + 1):
        style_order = list(range(len(style_training)))
        random.Random(args.seed + epoch).shuffle(style_order)
        replay_orders = {}
        for offset, (domain, rows) in enumerate(replay_records.items()):
            order = list(range(len(rows)))
            random.Random(args.seed + epoch * 17 + offset).shuffle(order)
            replay_orders[domain] = order
        replay_offsets = {domain: 0 for domain in replay_records}
        totals = defaultdict(float)
        steps = 0
        adapter.train()
        for start in range(0, len(style_order), args.style_molecules_per_batch):
            molecule_indices = style_order[start : start + args.style_molecules_per_batch]
            if not molecule_indices:
                continue
            style_samples = [
                sample
                for molecule_index in molecule_indices
                for sample in style_training[molecule_index]
            ]
            replay_samples = []
            for domain, rows in replay_records.items():
                offset = replay_offsets[domain]
                order = replay_orders[domain]
                indices = [
                    order[(offset + index) % len(order)]
                    for index in range(per_domain_replay)
                ]
                replay_offsets[domain] = (offset + per_domain_replay) % len(order)
                replay_samples.extend(rows[index] for index in indices)

            style_output, style_proposals, _ = _candidate_batch(
                style_samples,
                vision=vision,
                atom=atom,
                candidate_bond=candidate_bond,
                original_bond=original_bond,
                set_head=set_head,
                adapter=adapter,
                collator=collator,
                proposal_config=proposal_config,
                device=device,
            )
            replay_output, replay_proposals, _ = _candidate_batch(
                replay_samples,
                vision=vision,
                atom=atom,
                candidate_bond=candidate_bond,
                original_bond=original_bond,
                set_head=set_head,
                adapter=adapter,
                collator=collator,
                proposal_config=proposal_config,
                device=device,
            )
            style_loss = proposal_object_set_loss(
                style_output,
                style_proposals["atom_presence_targets"],
                style_proposals["bond_presence_targets"],
                style_proposals["atom_mask"],
                style_proposals["bond_mask"],
                style_proposals["target_atom_refs"],
                style_proposals["bond_supervision_mask"],
                atom_decision_threshold=atom_threshold,
                bond_decision_threshold=bond_threshold,
            )["loss"]
            replay_loss = proposal_object_set_loss(
                replay_output,
                replay_proposals["atom_presence_targets"],
                replay_proposals["bond_presence_targets"],
                replay_proposals["atom_mask"],
                replay_proposals["bond_mask"],
                replay_proposals["target_atom_refs"],
                replay_proposals["bond_supervision_mask"],
                atom_decision_threshold=atom_threshold,
                bond_decision_threshold=bond_threshold,
            )["loss"]
            residual_loss = (
                style_output["proposal_delta_bond_presence_residual"].float().square().mean()
                + style_output["proposal_delta_incidence_residual"].float().square().mean()
                + replay_output["proposal_delta_bond_presence_residual"].float().square().mean()
                + replay_output["proposal_delta_incidence_residual"].float().square().mean()
            )
            loss = style_loss + args.replay_weight * replay_loss + args.residual_weight * residual_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            style_seen += len(style_samples)
            replay_seen += len(replay_samples)
            steps += 1
            for name, value in (
                ("loss", loss),
                ("style_loss", style_loss),
                ("replay_loss", replay_loss),
                ("residual_loss", residual_loss),
            ):
                totals[name] += float(value.detach())

        metrics = validate()
        key = selection_key(metrics, initial)
        selected = key > best_key
        if selected:
            best_key = key
            best_metrics = metrics
            torch.save(checkpoint_payload(epoch, metrics), output_dir / "best.pt")
        torch.save(checkpoint_payload(epoch, metrics), output_dir / f"epoch-{epoch}.pt")
        torch.save(checkpoint_payload(epoch, metrics), output_dir / "last.pt")
        row = {
            "epoch": epoch,
            "selected_best": selected,
            "selection_key": list(key),
            "style_seen": style_seen,
            "replay_seen": replay_seen,
            "mean_losses": {name: value / max(steps, 1) for name, value in totals.items()},
            **metrics,
            "elapsed_seconds": time.perf_counter() - started,
        }
        history.append(row)
        (output_dir / "metrics.json").write_text(
            json.dumps(
                {"initial": initial, "best": best_metrics, "history": history},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"epoch_result": row}, separators=(",", ":")), flush=True)


def load_proposal_delta_set_adapter(
    path: str, device: torch.device
) -> tuple[ProposalDeltaSetAdapter, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    adapter = ProposalDeltaSetAdapter(
        ProposalDeltaSetConfig(**payload["config"])
    ).to(device)
    adapter.load_state_dict(payload["head"], strict=True)
    return adapter, payload


if __name__ == "__main__":
    main()
