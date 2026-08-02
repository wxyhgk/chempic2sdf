#!/usr/bin/env python3
"""Audit student proposal discovery with stable teacher representations."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.predicted_incidence import (
    PredictedProposalConfig,
    associate_predicted_proposals,
    decode_predicted_atom_proposals,
    decode_predicted_bond_proposals,
)
from chemvector.train_atom_bond_scene import AtomBondSceneCollator, stack_targets
from chemvector.train_bond_presence_calibration import load_components
from chemvector.train_joint_proposal_objects_full import load_validation_records
from chemvector.train_multidomain_set_head import VALIDATION_DOMAINS
from chemvector.train_proposal_object_set import _sample_set_metrics
from chemvector.train_spatial_incidence import DEFAULT_GRAPH, forward_frozen_bond_head


DEFAULT_TEACHER = (
    "outputs/checkpoints/chemvector-v0972-bond-presence-calibration-10k/best.pt"
)
DEFAULT_STUDENT = (
    "outputs/checkpoints/chemvector-v0979-feature-distilled-multidomain-proposals-"
    "balanced18k/samples-004500.pt"
)


def rank_match_scores(
    student_scores: torch.Tensor,
    student_mask: torch.Tensor,
    teacher_scores: torch.Tensor,
    teacher_mask: torch.Tensor,
) -> torch.Tensor:
    """Keep student ranking while restoring the teacher score marginal."""

    result = student_scores.clone()
    for batch_index in range(student_scores.size(0)):
        student_indices = torch.where(student_mask[batch_index])[0]
        teacher_values = teacher_scores[batch_index, teacher_mask[batch_index]]
        if not len(student_indices) or not len(teacher_values):
            continue
        student_order = student_scores[batch_index, student_indices].argsort(descending=True)
        teacher_values = teacher_values.sort(descending=True).values
        if len(teacher_values) != len(student_indices):
            positions = torch.linspace(
                0,
                len(teacher_values) - 1,
                len(student_indices),
                device=teacher_values.device,
            ).round().long()
            teacher_values = teacher_values[positions]
        result[batch_index, student_indices[student_order]] = teacher_values
    return result


def summarize(totals: dict[str, int]) -> dict:
    samples = max(totals["samples"], 1)
    return {
        **totals,
        "atom_proposal_recall": totals["atom_matched"] / max(totals["atom_targets"], 1),
        "bond_proposal_recall": totals["bond_matched"] / max(totals["bond_targets"], 1),
        "atom_deployed_exact_rate": totals["atom_deployed_exact"] / samples,
        "bond_deployed_exact_rate": totals["bond_deployed_exact"] / samples,
        "incidence_exact_rate": totals["incidence_exact"] / samples,
        "full_object_exact_rate": totals["full_object_exact"] / samples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-checkpoint", default=DEFAULT_STUDENT)
    parser.add_argument("--teacher-checkpoint", default=DEFAULT_TEACHER)
    parser.add_argument("--grounded-graph-checkpoint", default=DEFAULT_GRAPH)
    parser.add_argument("--samples-per-domain", type=int, default=498)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument(
        "--output",
        default=(
            "outputs/audits/chemvector-v0979-hybrid-proposal-representation/"
            "summary.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Hybrid proposal audit requires CUDA")
    device = torch.device("cuda")
    vision, student_atom, student_bond, set_head, student_payload = load_components(
        args.student_checkpoint, args.grounded_graph_checkpoint, device
    )
    _, teacher_atom, teacher_bond, _, teacher_payload = load_components(
        args.teacher_checkpoint, args.grounded_graph_checkpoint, device
    )
    proposal_config = PredictedProposalConfig(**student_payload["proposal_config"])
    if student_payload["proposal_config"] != teacher_payload["proposal_config"]:
        raise ValueError("Student and teacher proposal protocols differ")
    scene_config = AtomBondSceneConfig(
        atom_queries=proposal_config.maximum_atoms,
        bond_queries=proposal_config.maximum_bonds,
    )
    collator = AtomBondSceneCollator(scene_config)
    atom_threshold = float(student_payload["training_args"]["atom_decision_threshold"])
    bond_threshold = float(student_payload["training_args"]["bond_decision_threshold"])
    modes = (
        "student",
        "teacher_features",
        "teacher_features_scores",
        "teacher_features_types_scores",
    )
    result = {"domains": {}}

    for domain, shard in VALIDATION_DOMAINS.items():
        records, audit = load_validation_records(
            [shard], per_shard=args.samples_per_domain, config=scene_config
        )
        totals = {mode: defaultdict(int) for mode in modes}
        for start in range(0, len(records), args.batch_size):
            batch = collator(records[start : start + args.batch_size])
            pixels = batch["pixel_values"].to(device)
            targets = stack_targets(batch, device)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                features = vision.forward_stroke_resolution(pixels)
                student_atom_output = student_atom(features)
                student_bond_output = forward_frozen_bond_head(
                    vision, student_bond, features
                )
                teacher_atom_output = teacher_atom(features)
                teacher_bond_output = forward_frozen_bond_head(
                    vision, teacher_bond, features
                )
                teacher_atom_proposals = decode_predicted_atom_proposals(
                    teacher_atom_output, proposal_config
                )
                teacher_bond_proposals = decode_predicted_bond_proposals(
                    teacher_bond_output, proposal_config
                )

                for mode in modes:
                    atom_output = dict(student_atom_output)
                    bond_output = dict(student_bond_output)
                    if mode != "student":
                        atom_output["object_features"] = teacher_atom_output[
                            "object_features"
                        ]
                        bond_output["object_features"] = teacher_bond_output[
                            "object_features"
                        ]
                    if mode == "teacher_features_types_scores":
                        for name in (
                            "display_logits",
                            "chemical_logits",
                            "direction_logits",
                        ):
                            bond_output[name] = teacher_bond_output[name]
                    atom_proposals = decode_predicted_atom_proposals(
                        atom_output, proposal_config
                    )
                    bond_proposals = decode_predicted_bond_proposals(
                        bond_output, proposal_config
                    )
                    if mode in {
                        "teacher_features_scores",
                        "teacher_features_types_scores",
                    }:
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
                    proposals = associate_predicted_proposals(
                        atom_proposals,
                        bond_proposals,
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
                        totals[mode]["samples"] += 1
                        for name, value in metrics.items():
                            totals[mode][name] += int(value)
        result["domains"][domain] = {
            "audit": audit,
            "modes": {mode: summarize(dict(values)) for mode, values in totals.items()},
        }

    result.update(
        {
            "student_checkpoint": args.student_checkpoint,
            "teacher_checkpoint": args.teacher_checkpoint,
            "atom_threshold": atom_threshold,
            "bond_threshold": bond_threshold,
        }
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    compact = {
        domain: {
            mode: {
                "atom_proposal_recall": values["atom_proposal_recall"],
                "bond_proposal_recall": values["bond_proposal_recall"],
                "bond_deployed_exact_rate": values["bond_deployed_exact_rate"],
                "full_object_exact_rate": values["full_object_exact_rate"],
            }
            for mode, values in payload["modes"].items()
        }
        for domain, payload in result["domains"].items()
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
