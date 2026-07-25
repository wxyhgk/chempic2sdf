#!/usr/bin/env python3
"""Audit MR-TADF failures across proposal, selection, and incidence stages."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

from PIL import Image
import torch

from chemvector.benchmark_joint_mr_tadf import (
    DEFAULT_FALLBACK,
    DEFAULT_MANIFEST,
    DEFAULT_PRIMARY,
    _load_set_head,
)
from chemvector.joint_graph_milp import JointGraphMilpConfig, apply_joint_graph_milp
from chemvector.predicted_incidence import (
    PredictedProposalConfig,
    associate_predicted_proposals,
    decode_predicted_atom_proposals,
    decode_predicted_bond_proposals,
)
from chemvector.train_atom_bond_scene import _image_tensor
from chemvector.train_spatial_incidence import (
    DEFAULT_ATOM,
    DEFAULT_BOND,
    DEFAULT_GRAPH,
    forward_frozen_bond_head,
    load_frozen_components,
)


DEFAULT_BENCHMARK_RECORDS = (
    "outputs/evaluations/chemvector-v0964-mr-tadf-v1-main1100/records.jsonl"
)
DEFAULT_OUTPUT = "outputs/audits/chemvector-v0965-mr-tadf-component-audit1100"


def targets_from_annotations(
    annotations: list[dict], device: torch.device
) -> dict[str, torch.Tensor]:
    batch = len(annotations)
    maximum_atoms = max(len(item["atoms"]) for item in annotations)
    maximum_bonds = max(len(item["bonds"]) for item in annotations)
    atom_anchors = torch.zeros((batch, maximum_atoms, 2), device=device)
    atom_mask = torch.zeros((batch, maximum_atoms), dtype=torch.bool, device=device)
    bond_atom_refs = torch.zeros(
        (batch, maximum_bonds, 2), dtype=torch.long, device=device
    )
    bond_mask = torch.zeros((batch, maximum_bonds), dtype=torch.bool, device=device)
    for batch_index, annotation in enumerate(annotations):
        atoms = sorted(annotation["atoms"], key=lambda item: int(item["atom_id"]))
        for expected_id, atom in enumerate(atoms):
            if int(atom["atom_id"]) != expected_id:
                raise ValueError("MR-TADF atom ids must be contiguous")
            atom_anchors[batch_index, expected_id] = torch.tensor(
                [float(atom["image_x"]) / 768.0, float(atom["image_y"]) / 768.0],
                device=device,
            )
            atom_mask[batch_index, expected_id] = True
        for bond_index, bond in enumerate(annotation["bonds"]):
            bond_atom_refs[batch_index, bond_index] = torch.tensor(
                [int(bond["begin_atom"]), int(bond["end_atom"])], device=device
            )
            bond_mask[batch_index, bond_index] = True
    return {
        "atom_anchors": atom_anchors,
        "atom_mask": atom_mask,
        "bond_atom_refs": bond_atom_refs,
        "bond_mask": bond_mask,
    }


def decision_metrics(
    *,
    atom_logits: torch.Tensor,
    bond_logits: torch.Tensor,
    incidence_logits: torch.Tensor,
    proposals: dict[str, torch.Tensor],
    batch_index: int,
    atom_threshold: float,
    bond_threshold: float,
    target_atoms: int,
    target_bonds: int,
) -> dict:
    atom_mask = proposals["atom_mask"][batch_index].cpu()
    bond_mask = proposals["bond_mask"][batch_index].cpu()
    atom_positive = proposals["atom_presence_targets"][batch_index].cpu()
    bond_positive = proposals["bond_presence_targets"][batch_index].cpu()
    selected_atoms = atom_mask & (atom_logits >= atom_threshold)
    selected_bonds = bond_mask & (bond_logits >= bond_threshold)
    supervised = proposals["bond_supervision_mask"][batch_index].cpu()
    expected_refs = proposals["target_atom_refs"][batch_index].cpu()
    predicted_refs = incidence_logits.argmax(-1)
    endpoint_correct = predicted_refs[supervised] == expected_refs[supervised]
    incidence_bond_correct = (
        int(endpoint_correct.all(-1).sum()) if len(endpoint_correct) else 0
    )
    atom_true = int((selected_atoms & atom_positive).sum())
    atom_false = int((selected_atoms & ~atom_positive).sum())
    bond_true = int((selected_bonds & bond_positive).sum())
    bond_false = int((selected_bonds & ~bond_positive).sum())
    atom_exact = atom_true == target_atoms and atom_false == 0
    bond_exact = bond_true == target_bonds and bond_false == 0
    incidence_exact = incidence_bond_correct == target_bonds
    return {
        "selected_atoms": int(selected_atoms.sum()),
        "selected_atom_targets": atom_true,
        "selected_false_atoms": atom_false,
        "atom_selection_exact": atom_exact,
        "selected_bonds": int(selected_bonds.sum()),
        "selected_bond_targets": bond_true,
        "selected_false_bonds": bond_false,
        "bond_selection_exact": bond_exact,
        "incidence_supervised_bonds": int(supervised.sum()),
        "incidence_correct_bonds": incidence_bond_correct,
        "incidence_correct_endpoints": int(endpoint_correct.sum()),
        "incidence_endpoints": int(endpoint_correct.numel()),
        "incidence_exact": incidence_exact,
        "object_incidence_exact": atom_exact and bond_exact and incidence_exact,
    }


def summarize(records: list[dict]) -> dict:
    count = max(len(records), 1)
    result = {"samples": len(records)}
    for key in (
        "atom_proposal_complete",
        "bond_proposal_complete",
        "incidence_eligible_complete",
    ):
        total = sum(bool(record[key]) for record in records)
        result[key] = total
        result[f"{key}_rate"] = total / count
    result["proposal_failure_first_stage"] = dict(
        Counter(record["proposal_failure_first_stage"] for record in records)
    )
    for mode in ("independent", "joint"):
        mode_records = [record[mode] for record in records]
        for key in (
            "atom_selection_exact",
            "bond_selection_exact",
            "incidence_exact",
            "object_incidence_exact",
        ):
            total = sum(bool(record[key]) for record in mode_records)
            result[f"{mode}_{key}"] = total
            result[f"{mode}_{key}_rate"] = total / count
        for key in (
            "selected_atom_targets",
            "selected_false_atoms",
            "selected_bond_targets",
            "selected_false_bonds",
            "incidence_correct_bonds",
            "incidence_supervised_bonds",
        ):
            result[f"{mode}_{key}"] = sum(int(record[key]) for record in mode_records)
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[record["difficulty"]].append(record)
    result["difficulty"] = {
        name: summarize(group) | {"difficulty": {}}
        for name, group in sorted(groups.items())
    } if len(groups) > 1 else {}
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--benchmark-records", default=DEFAULT_BENCHMARK_RECORDS)
    parser.add_argument("--primary-checkpoint", default=DEFAULT_PRIMARY)
    parser.add_argument("--fallback-checkpoint", default=DEFAULT_FALLBACK)
    parser.add_argument("--grounded-graph-checkpoint", default=DEFAULT_GRAPH)
    parser.add_argument("--atom-checkpoint", default=DEFAULT_ATOM)
    parser.add_argument("--bond-checkpoint", default=DEFAULT_BOND)
    parser.add_argument(
        "--proposal-bond-checkpoint",
        help="Optional single- or multi-scale Bond proposal checkpoint for A/B audit",
    )
    parser.add_argument("--samples", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--maximum-atoms", type=int, default=128)
    parser.add_argument("--maximum-bonds", type=int, default=192)
    parser.add_argument("--atom-threshold", type=float, default=-0.25)
    parser.add_argument("--bond-threshold", type=float, default=-0.5)
    parser.add_argument("--endpoint-topk", type=int, default=2)
    parser.add_argument("--incidence-weight", type=float, default=0.5)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("MR-TADF component audit requires CUDA")
    manifest = Path(args.benchmark_manifest)
    root = manifest.parent
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if args.samples:
        rows = rows[: args.samples]
    benchmark_records = {
        record["molecule_id"]: record
        for line in Path(args.benchmark_records).read_text().splitlines()
        if (record := json.loads(line))
    }
    device = torch.device("cuda")
    vision, atom_head, bond_head, _ = load_frozen_components(
        Path(args.grounded_graph_checkpoint),
        Path(args.atom_checkpoint),
        Path(args.proposal_bond_checkpoint or args.bond_checkpoint),
        device,
    )
    primary, _ = _load_set_head(args.primary_checkpoint, device)
    proposal_config = PredictedProposalConfig(
        maximum_atoms=args.maximum_atoms,
        maximum_bonds=args.maximum_bonds,
        atom_threshold=0.01,
        bond_threshold=0.01,
    )
    joint_config = JointGraphMilpConfig(
        endpoint_topk=args.endpoint_topk,
        incidence_weight=args.incidence_weight,
        time_limit_seconds=2.0,
    )
    records = []
    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        images = [Image.open(root / row["image_file"]).convert("RGB") for row in batch_rows]
        annotations = [
            json.loads((root / row["annotation_file"]).read_text()) for row in batch_rows
        ]
        targets = targets_from_annotations(annotations, device)
        pixels = torch.stack([_image_tensor(image) for image in images]).to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            features = vision.forward_stroke_resolution(pixels)
            atom_output = atom_head(features)
            bond_output = forward_frozen_bond_head(vision, bond_head, features)
            atom_proposals = decode_predicted_atom_proposals(atom_output, proposal_config)
            bond_proposals = decode_predicted_bond_proposals(bond_output, proposal_config)
            proposals = associate_predicted_proposals(
                atom_proposals, bond_proposals, targets, proposal_config
            )
            output = primary(
                proposals["atom_features"], proposals["atom_geometry"], proposals["atom_scores"], proposals["atom_mask"],
                proposals["bond_features"], proposals["bond_endpoints"], proposals["bond_types"], proposals["bond_scores"], proposals["bond_mask"],
            )
        for batch_index, row in enumerate(batch_rows):
            target_atoms = int(targets["atom_mask"][batch_index].sum())
            target_bonds = int(targets["bond_mask"][batch_index].sum())
            atom_matched = int(proposals["atom_target_matched"][batch_index].sum())
            bond_matched = int(proposals["bond_target_matched"][batch_index].sum())
            eligible = int(proposals["incidence_target_eligible"][batch_index].sum())
            atom_logits = output["atom_presence_logits"][batch_index].float().cpu()
            bond_logits = output["bond_presence_logits"][batch_index].float().cpu()
            incidence_logits = output["incidence_logits"][batch_index].float().cpu()
            independent = decision_metrics(
                atom_logits=atom_logits,
                bond_logits=bond_logits,
                incidence_logits=incidence_logits,
                proposals=proposals,
                batch_index=batch_index,
                atom_threshold=args.atom_threshold,
                bond_threshold=args.bond_threshold,
                target_atoms=target_atoms,
                target_bonds=target_bonds,
            )
            joint_atom, joint_bond, joint_incidence, milp = apply_joint_graph_milp(
                atom_presence_logits=atom_logits,
                atom_mask=proposals["atom_mask"][batch_index].cpu(),
                bond_presence_logits=bond_logits,
                bond_mask=proposals["bond_mask"][batch_index].cpu(),
                incidence_logits=incidence_logits,
                atom_threshold=args.atom_threshold,
                bond_threshold=args.bond_threshold,
                config=joint_config,
            )
            joint = decision_metrics(
                atom_logits=joint_atom,
                bond_logits=joint_bond,
                incidence_logits=joint_incidence,
                proposals=proposals,
                batch_index=batch_index,
                atom_threshold=args.atom_threshold,
                bond_threshold=args.bond_threshold,
                target_atoms=target_atoms,
                target_bonds=target_bonds,
            )
            if atom_matched < target_atoms:
                first_failure = "atom_proposal"
            elif bond_matched < target_bonds:
                first_failure = "bond_proposal"
            elif eligible < target_bonds:
                first_failure = "incidence_eligibility"
            else:
                first_failure = "fully_representable"
            prior = benchmark_records.get(row["molecule_id"], {})
            records.append(
                {
                    "molecule_id": row["molecule_id"],
                    "difficulty": row["difficulty"],
                    "benchmark_sdf_valid": bool(prior.get("sdf_valid")),
                    "benchmark_exact": bool(prior.get("chemical_match")),
                    "target_atoms": target_atoms,
                    "target_bonds": target_bonds,
                    "atom_proposals_matched": atom_matched,
                    "bond_proposals_matched": bond_matched,
                    "incidence_eligible_bonds": eligible,
                    "atom_proposal_complete": atom_matched == target_atoms,
                    "bond_proposal_complete": bond_matched == target_bonds,
                    "incidence_eligible_complete": eligible == target_bonds,
                    "proposal_failure_first_stage": first_failure,
                    "independent": independent,
                    "joint": joint,
                    "milp_success": bool(milp["solver_success"]),
                }
            )
    summary = summarize(records)
    summary["protocol"] = {
        "model_version": "v0.9.63",
        "audit_version": "v0.9.65",
        "mr_tadf_used_for_training": False,
        "mr_tadf_used_for_threshold_tuning": False,
        "proposal_config": vars(args),
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "records.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    )
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
