#!/usr/bin/env python3
"""Train set/incidence on student discovery with teacher score calibration."""

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
from chemvector.audit_hybrid_proposal_representation import rank_match_scores
from chemvector.predicted_incidence import (
    PredictedProposalConfig,
    associate_predicted_proposals,
    decode_predicted_atom_proposals,
    decode_predicted_bond_proposals,
)
from chemvector.proposal_object_set import proposal_object_set_loss
from chemvector.train_atom_bond_scene import AtomBondSceneCollator, stack_targets
from chemvector.train_bond_presence_calibration import load_components
from chemvector.train_joint_proposal_objects_full import load_validation_records
from chemvector.train_multidomain_set_head import (
    VALIDATION_DOMAINS,
    balanced_batches,
)
from chemvector.train_proposal_object_set import _sample_set_metrics
from chemvector.train_spatial_incidence import DEFAULT_GRAPH, forward_frozen_bond_head


DEFAULT_TEACHER = (
    "outputs/checkpoints/chemvector-v0972-bond-presence-calibration-10k/best.pt"
)
DEFAULT_STUDENT = (
    "outputs/checkpoints/chemvector-v0979-feature-distilled-multidomain-proposals-"
    "balanced18k/samples-004500.pt"
)
DEFAULT_OUTPUT = (
    "outputs/checkpoints/chemvector-v0980-hybrid-multidomain-set-balanced18k"
)


def build_hybrid_proposals(
    vision,
    student_atom,
    student_bond,
    teacher_atom,
    teacher_bond,
    pixels: torch.Tensor,
    targets: dict[str, torch.Tensor],
    proposal_config: PredictedProposalConfig,
) -> dict[str, torch.Tensor]:
    features = vision.forward_stroke_resolution(pixels)
    student_atom_output = student_atom(features)
    student_bond_output = forward_frozen_bond_head(vision, student_bond, features)
    teacher_atom_output = teacher_atom(features)
    teacher_bond_output = forward_frozen_bond_head(vision, teacher_bond, features)
    student_atom_output["object_features"] = teacher_atom_output["object_features"]
    student_bond_output["object_features"] = teacher_bond_output["object_features"]
    atom_proposals = decode_predicted_atom_proposals(
        student_atom_output, proposal_config
    )
    bond_proposals = decode_predicted_bond_proposals(
        student_bond_output, proposal_config
    )
    teacher_atom_proposals = decode_predicted_atom_proposals(
        teacher_atom_output, proposal_config
    )
    teacher_bond_proposals = decode_predicted_bond_proposals(
        teacher_bond_output, proposal_config
    )
    atom_proposals["atom_scores"] = rank_match_scores(
        atom_proposals["atom_scores"],
        atom_proposals["atom_mask"],
        teacher_atom_proposals["atom_scores"],
        teacher_atom_proposals["atom_mask"],
    )
    bond_proposals["bond_scores"] = rank_match_scores(
        bond_proposals["bond_scores"],
        bond_proposals["bond_mask"],
        teacher_bond_proposals["bond_scores"],
        teacher_bond_proposals["bond_mask"],
    )
    return associate_predicted_proposals(
        atom_proposals, bond_proposals, targets, proposal_config
    )


def summarize_totals(totals: dict[str, int]) -> dict:
    samples = max(totals["samples"], 1)
    result = dict(totals)
    for name in (
        "atom_deployed_exact",
        "bond_deployed_exact",
        "incidence_exact",
        "full_object_exact",
    ):
        result[name + "_rate"] = totals[name] / samples
    result["atom_proposal_recall"] = totals["atom_matched"] / max(
        totals["atom_targets"], 1
    )
    result["bond_proposal_recall"] = totals["bond_matched"] / max(
        totals["bond_targets"], 1
    )
    return result


@torch.no_grad()
def validate(
    vision,
    student_atom,
    student_bond,
    teacher_atom,
    teacher_bond,
    set_head,
    records,
    collator,
    proposal_config,
    device,
    batch_size,
    atom_threshold,
    bond_threshold,
) -> dict:
    domains = {}
    pooled = defaultdict(int)
    for domain, rows in records.items():
        totals = defaultdict(int)
        for start in range(0, len(rows), batch_size):
            batch = collator(rows[start : start + batch_size])
            pixels = batch["pixel_values"].to(device)
            targets = stack_targets(batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                proposals = build_hybrid_proposals(
                    vision,
                    student_atom,
                    student_bond,
                    teacher_atom,
                    teacher_bond,
                    pixels,
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
            for batch_index in range(len(pixels)):
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
        domains[domain] = summarize_totals(dict(totals))
        for name, value in totals.items():
            pooled[name] += value
    pooled_metrics = summarize_totals(dict(pooled))
    return {
        "domains": domains,
        "pooled": pooled_metrics,
        "minimum_domain_full_object_exact_rate": min(
            item["full_object_exact_rate"] for item in domains.values()
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-checkpoint", default=DEFAULT_STUDENT)
    parser.add_argument("--teacher-checkpoint", default=DEFAULT_TEACHER)
    parser.add_argument("--grounded-graph-checkpoint", default=DEFAULT_GRAPH)
    parser.add_argument("--samples-per-domain", type=int, default=6_000)
    parser.add_argument("--validation-samples-per-domain", type=int, default=498)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--presence-learning-rate", type=float, default=5e-5)
    parser.add_argument("--incidence-learning-rate", type=float, default=5e-6)
    parser.add_argument("--validate-every-total-samples", type=int, default=4_500)
    parser.add_argument("--log-every-total-samples", type=int, default=900)
    parser.add_argument("--shuffle-buffer", type=int, default=1024)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Hybrid multidomain set training requires CUDA")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_args.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    vision, student_atom, student_bond, _, student_payload = load_components(
        args.student_checkpoint, args.grounded_graph_checkpoint, device
    )
    _, teacher_atom, teacher_bond, set_head, teacher_payload = load_components(
        args.teacher_checkpoint, args.grounded_graph_checkpoint, device
    )
    for module in (vision, student_atom, student_bond, teacher_atom, teacher_bond):
        module.eval().requires_grad_(False)
    set_head.requires_grad_(True)
    proposal_config = PredictedProposalConfig(**student_payload["proposal_config"])
    if student_payload["proposal_config"] != teacher_payload["proposal_config"]:
        raise ValueError("Student and teacher proposal protocols differ")
    scene_config = AtomBondSceneConfig(
        atom_queries=proposal_config.maximum_atoms,
        bond_queries=proposal_config.maximum_bonds,
    )
    collator = AtomBondSceneCollator(scene_config)
    validation_records = {}
    validation_audits = {}
    for name, shard in VALIDATION_DOMAINS.items():
        rows, audit = load_validation_records(
            [shard],
            per_shard=args.validation_samples_per_domain,
            config=scene_config,
        )
        validation_records[name] = rows
        validation_audits[name] = audit
    (output_dir / "validation_audit.json").write_text(
        json.dumps(validation_audits, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    atom_threshold = float(teacher_payload["training_args"]["atom_decision_threshold"])
    bond_threshold = float(teacher_payload["training_args"]["bond_decision_threshold"])
    initial = validate(
        vision,
        student_atom,
        student_bond,
        teacher_atom,
        teacher_bond,
        set_head,
        validation_records,
        collator,
        proposal_config,
        device,
        args.batch_size,
        atom_threshold,
        bond_threshold,
    )
    (output_dir / "initial_metrics.json").write_text(
        json.dumps(initial, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"initial_validation": initial}, separators=(",", ":")), flush=True)

    incidence_parameters = list(set_head.incidence.parameters())
    incidence_ids = {id(parameter) for parameter in incidence_parameters}
    presence_parameters = [
        parameter for parameter in set_head.parameters() if id(parameter) not in incidence_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": presence_parameters, "lr": args.presence_learning_rate},
            {"params": incidence_parameters, "lr": args.incidence_learning_rate},
        ],
        weight_decay=0.0,
    )
    trainable = list(set_head.parameters())
    total_seen = 0
    next_log = args.log_every_total_samples
    next_validation = args.validate_every_total_samples
    running_loss = 0.0
    running_samples = 0
    best_score = None
    best_metrics = None
    started = time.perf_counter()

    for samples, domain_seen in balanced_batches(
        samples_per_domain=args.samples_per_domain,
        batch_size=args.batch_size,
        config=scene_config,
        seed=args.seed,
        shuffle_buffer=args.shuffle_buffer,
    ):
        set_head.train()
        batch = collator(samples)
        pixels = batch["pixel_values"].to(device)
        targets = stack_targets(batch, device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            proposals = build_hybrid_proposals(
                vision,
                student_atom,
                student_bond,
                teacher_atom,
                teacher_bond,
                pixels,
                targets,
                proposal_config,
            )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
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
            losses = proposal_object_set_loss(
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
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        count = len(samples)
        total_seen += count
        running_samples += count
        running_loss += float(losses["loss"].detach()) * count
        if total_seen >= next_log:
            print(
                json.dumps(
                    {
                        "total_seen": total_seen,
                        "domain_seen": domain_seen,
                        "loss": running_loss / running_samples,
                        "elapsed_seconds": time.perf_counter() - started,
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
            running_loss = 0.0
            running_samples = 0
            next_log += args.log_every_total_samples
        if total_seen >= next_validation or min(domain_seen.values()) >= args.samples_per_domain:
            metrics = validate(
                vision,
                student_atom,
                student_bond,
                teacher_atom,
                teacher_bond,
                set_head,
                validation_records,
                collator,
                proposal_config,
                device,
                args.batch_size,
                atom_threshold,
                bond_threshold,
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
                "checkpoint_schema_version": "hybrid-joint-proposal-objects-0.1",
                "student_checkpoint": args.student_checkpoint,
                "teacher_checkpoint": args.teacher_checkpoint,
                "set_head": set_head.state_dict(),
                "set_config": {
                    "incidence": asdict(set_head.config.incidence),
                    "hidden_dim": set_head.config.hidden_dim,
                    "stable_bond_hidden_dim": set_head.config.stable_bond_hidden_dim,
                },
                "proposal_config": asdict(proposal_config),
                "optimizer": optimizer.state_dict(),
                "samples_seen": total_seen,
                "validation": metrics,
                "training_args": vars(args),
            }
            torch.save(checkpoint, output_dir / f"samples-{total_seen:06d}.pt")
            torch.save(checkpoint, output_dir / "last.pt")
            score = (
                metrics["minimum_domain_full_object_exact_rate"],
                metrics["pooled"]["full_object_exact_rate"],
                metrics["pooled"]["bond_deployed_exact_rate"],
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


if __name__ == "__main__":
    main()
