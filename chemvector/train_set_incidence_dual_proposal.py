#!/usr/bin/env python3
"""Realign set/incidence on residual and original Bond proposal distributions."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from pathlib import Path
import json
import random
import time

import torch
from torch.nn import functional as F

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.dense_bond_object import BondProposalResidualAdapter
from chemvector.evaluate_style_sensitivity import load_style_records
from chemvector.predicted_incidence import (
    PredictedProposalConfig,
    associate_predicted_proposals,
    decode_predicted_atom_proposals,
    decode_predicted_bond_proposals,
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
from chemvector.train_spatial_incidence import DEFAULT_GRAPH, forward_frozen_bond_head
from chemvector.train_stable_bond_presence import domain_validation


DEFAULT_SOURCE = (
    "outputs/checkpoints/chemvector-v0995-bond-proposal-style-residual-pilot250/"
    "candidate.pt"
)
DEFAULT_STYLE_ROOT = Path("Data/sdf/chemvision_layers_v0_2/font_pilot_10k")
DEFAULT_OUTPUT = (
    "outputs/checkpoints/chemvector-v0996-set-incidence-dual-proposal-pilot250"
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
    parser.add_argument("--presence-learning-rate", type=float, default=2e-6)
    parser.add_argument("--incidence-learning-rate", type=float, default=2e-7)
    parser.add_argument("--candidate-replay-weight", type=float, default=1.0)
    parser.add_argument("--original-replay-weight", type=float, default=1.0)
    parser.add_argument("--teacher-weight", type=float, default=0.1)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260726)
    return parser.parse_args()


def _domain_safety_deltas(candidate: dict, baseline: dict) -> list[float]:
    deltas = []
    for domain, expected in baseline["domains"].items():
        actual = candidate["domains"][domain]
        deltas.extend(
            (
                float(actual["bond_deployed_exact"] - expected["bond_deployed_exact"]),
                float(actual["full_object_exact"] - expected["full_object_exact"]),
            )
        )
    return deltas


def selection_key(
    candidate_official: dict,
    original_official: dict,
    style: dict,
    baseline_candidate: dict,
    baseline_original: dict,
) -> tuple[float, ...]:
    """Require both proposal paths to preserve every official domain."""

    deltas = _domain_safety_deltas(candidate_official, baseline_candidate)
    deltas.extend(_domain_safety_deltas(original_official, baseline_original))
    safe = min(deltas) >= 0.0
    style_metrics = style["domains"]["style"]
    return (
        float(safe),
        float(style_metrics["full_object_exact"]) if safe else min(deltas),
        float(style_metrics["bond_deployed_exact"]),
        float(candidate_official["pooled"]["full_object_exact"]),
        float(original_official["pooled"]["full_object_exact"]),
    )


@torch.no_grad()
def _frozen_proposals(
    *,
    atom_output: dict,
    bond_output: dict,
    targets: dict,
    proposal_config,
) -> dict:
    return associate_predicted_proposals(
        decode_predicted_atom_proposals(atom_output, proposal_config),
        decode_predicted_bond_proposals(bond_output, proposal_config),
        targets,
        proposal_config,
    )


def _set_forward(set_head, proposals: dict) -> dict:
    return set_head(
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


def _supervised_loss(
    output: dict,
    proposals: dict,
    *,
    atom_threshold: float,
    bond_threshold: float,
) -> dict[str, torch.Tensor]:
    return proposal_object_set_loss(
        output,
        proposals["atom_presence_targets"],
        proposals["bond_presence_targets"],
        proposals["atom_mask"],
        proposals["bond_mask"],
        proposals["target_atom_refs"],
        proposals["bond_supervision_mask"],
        atom_decision_threshold=atom_threshold,
        bond_decision_threshold=bond_threshold,
    )


def _teacher_loss(student: dict, teacher: dict, proposals: dict) -> torch.Tensor:
    atom_mask = proposals["atom_mask"]
    bond_mask = proposals["bond_mask"]
    bond_supervision = proposals["bond_supervision_mask"]
    losses = [
        F.mse_loss(
            student["atom_presence_logits"][atom_mask].float(),
            teacher["atom_presence_logits"][atom_mask].float(),
        ),
        F.mse_loss(
            student["bond_presence_logits"][bond_mask].float(),
            teacher["bond_presence_logits"][bond_mask].float(),
        ),
    ]
    if bool(bond_supervision.any()):
        losses.append(
            F.kl_div(
                student["incidence_logits"][bond_supervision].float().log_softmax(-1),
                teacher["incidence_logits"][bond_supervision].float().softmax(-1),
                reduction="batchmean",
            )
        )
    return sum(losses)


def _batch_outputs(
    samples: list[dict],
    *,
    vision,
    atom,
    candidate_bond,
    original_bond,
    collator,
    proposal_config,
    device: torch.device,
) -> tuple[dict, dict]:
    batch = collator(samples)
    pixels = batch["pixel_values"].to(device, non_blocking=True)
    targets = stack_targets(batch, device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        features = vision.forward_stroke_resolution(pixels)
        atom_output = atom(features)
        candidate_output = forward_frozen_bond_head(vision, candidate_bond, features)
        original_output = forward_frozen_bond_head(vision, original_bond, features)
        candidate = _frozen_proposals(
            atom_output=atom_output,
            bond_output=candidate_output,
            targets=targets,
            proposal_config=proposal_config,
        )
        original = _frozen_proposals(
            atom_output=atom_output,
            bond_output=original_output,
            targets=targets,
            proposal_config=proposal_config,
        )
    return candidate, original


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Dual-proposal set/incidence training requires CUDA")
    if args.replay_batch_size % 2:
        raise ValueError("Replay batch size must be even")
    if min(args.style_molecules, args.style_molecules_per_batch, args.epochs) < 1:
        raise ValueError("Training sizes must be positive")
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

    vision, atom, candidate_bond, set_head, source = load_components(
        args.source_checkpoint, args.grounded_graph_checkpoint, device
    )
    if not isinstance(candidate_bond, BondProposalResidualAdapter):
        raise ValueError("Source checkpoint does not contain a Bond proposal residual")
    original_bond = candidate_bond.base_detector
    for module in (vision, atom, candidate_bond, original_bond):
        module.eval().requires_grad_(False)
    teacher_set = copy.deepcopy(set_head).eval().requires_grad_(False)
    set_head.train().requires_grad_(True)

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
        rows, audit = load_validation_records(
            [shard],
            per_shard=args.validation_samples_per_domain,
            config=scene_config,
        )
        official_records[domain] = rows
        official_audit[domain] = audit
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

    def validate() -> dict:
        set_head.eval()
        arguments = (
            official_records,
            collator,
            proposal_config,
            device,
            args.validation_batch_size,
            atom_threshold,
            bond_threshold,
        )
        return {
            "candidate_official": domain_validation(
                vision, atom, candidate_bond, set_head, *arguments
            ),
            "original_official": domain_validation(
                vision, atom, original_bond, set_head, *arguments
            ),
            "style": domain_validation(
                vision,
                atom,
                candidate_bond,
                set_head,
                {"style": style_validation},
                collator,
                proposal_config,
                device,
                args.validation_batch_size,
                atom_threshold,
                bond_threshold,
            ),
        }

    initial = validate()
    (output_dir / "initial_metrics.json").write_text(
        json.dumps(initial, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"initial": initial}, separators=(",", ":")), flush=True)

    incidence_parameters = list(set_head.incidence.parameters())
    incidence_ids = {id(parameter) for parameter in incidence_parameters}
    presence_parameters = [
        parameter for parameter in set_head.parameters()
        if id(parameter) not in incidence_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": presence_parameters, "lr": args.presence_learning_rate},
            {"params": incidence_parameters, "lr": args.incidence_learning_rate},
        ],
        weight_decay=0.0,
    )
    trainable = list(set_head.parameters())
    baseline_candidate = initial["candidate_official"]
    baseline_original = initial["original_official"]
    best_key = selection_key(
        baseline_candidate,
        baseline_original,
        initial["style"],
        baseline_candidate,
        baseline_original,
    )
    best_metrics = initial
    history = []
    style_seen = replay_seen = 0
    started = time.perf_counter()

    def checkpoint_payload(epoch: int, metrics: dict) -> dict:
        return {
            **{
                key: value
                for key, value in source.items()
                if key not in {"optimizer", "validation", "samples_seen", "set_head"}
            },
            "checkpoint_schema_version": "joint-proposal-objects-checkpoint-0.7",
            "set_head": set_head.state_dict(),
            "set_config": {
                "incidence": asdict(set_head.config.incidence),
                "hidden_dim": set_head.config.hidden_dim,
                "stable_bond_hidden_dim": set_head.config.stable_bond_hidden_dim,
            },
            "optimizer": optimizer.state_dict(),
            "samples_seen": int(source["samples_seen"]) + style_seen + replay_seen,
            "set_realign_style_samples": style_seen,
            "set_realign_replay_samples": replay_seen,
            "validation": metrics["candidate_official"],
            "original_proposal_validation": metrics["original_official"],
            "style_validation": metrics["style"],
            "source_joint_checkpoint": args.source_checkpoint,
            "dual_proposal_set_args": vars(args),
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
        totals = {"loss": 0.0, "style": 0.0, "candidate": 0.0, "original": 0.0, "teacher": 0.0}
        steps = 0
        set_head.train()

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

            style_candidate, _ = _batch_outputs(
                style_samples,
                vision=vision,
                atom=atom,
                candidate_bond=candidate_bond,
                original_bond=original_bond,
                collator=collator,
                proposal_config=proposal_config,
                device=device,
            )
            replay_candidate, replay_original = _batch_outputs(
                replay_samples,
                vision=vision,
                atom=atom,
                candidate_bond=candidate_bond,
                original_bond=original_bond,
                collator=collator,
                proposal_config=proposal_config,
                device=device,
            )
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                teacher_output = _set_forward(teacher_set, replay_original)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                style_output = _set_forward(set_head, style_candidate)
                candidate_output = _set_forward(set_head, replay_candidate)
                original_output = _set_forward(set_head, replay_original)
                style_loss = _supervised_loss(
                    style_output,
                    style_candidate,
                    atom_threshold=atom_threshold,
                    bond_threshold=bond_threshold,
                )["loss"]
                candidate_loss = _supervised_loss(
                    candidate_output,
                    replay_candidate,
                    atom_threshold=atom_threshold,
                    bond_threshold=bond_threshold,
                )["loss"]
                original_loss = _supervised_loss(
                    original_output,
                    replay_original,
                    atom_threshold=atom_threshold,
                    bond_threshold=bond_threshold,
                )["loss"]
                teacher_loss = _teacher_loss(
                    original_output, teacher_output, replay_original
                )
                loss = (
                    style_loss
                    + args.candidate_replay_weight * candidate_loss
                    + args.original_replay_weight * original_loss
                    + args.teacher_weight * teacher_loss
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            style_seen += len(style_samples)
            replay_seen += len(replay_samples)
            steps += 1
            for name, value in (
                ("loss", loss),
                ("style", style_loss),
                ("candidate", candidate_loss),
                ("original", original_loss),
                ("teacher", teacher_loss),
            ):
                totals[name] += float(value.detach())

        metrics = validate()
        key = selection_key(
            metrics["candidate_official"],
            metrics["original_official"],
            metrics["style"],
            baseline_candidate,
            baseline_original,
        )
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


if __name__ == "__main__":
    main()
