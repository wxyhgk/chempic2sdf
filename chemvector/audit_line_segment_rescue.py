#!/usr/bin/env python3
"""Audit length-independent line-to-AtomPair rescue on fixed image domains."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import asdict
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.benchmark_joint_mr_tadf import DEFAULT_PRIMARY, _load_set_head
from chemvector.line_segment_rescue import (
    LineSegmentRescueConfig,
    detect_atom_pair_line_candidates,
)
from chemvector.predicted_incidence import (
    PredictedProposalConfig,
    associate_predicted_proposals,
    decode_predicted_atom_proposals,
    decode_predicted_bond_proposals,
)
from chemvector.train_atom_bond_scene import (
    AtomBondSceneCollator,
    _image_tensor,
    stack_targets,
)
from chemvector.train_joint_proposal_objects_full import load_validation_records
from chemvector.train_spatial_incidence import (
    DEFAULT_ATOM,
    DEFAULT_BOND,
    DEFAULT_GRAPH,
    forward_frozen_bond_head,
    load_frozen_components,
)


DEFAULT_INTERNAL_SHARD = (
    "Data/sdf/chemvision_layers_v0_1/pilot_10k/shards/validation-000000.tar"
)
DEFAULT_MR_MANIFEST = "Data/benchmarks/mr_tadf_v1/manifest.csv"
DEFAULT_OUTPUT = "outputs/audits/chemvector-v0984-line-segment-rescue"


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
    bond_mask = torch.zeros(
        (batch, maximum_bonds), dtype=torch.bool, device=device
    )
    for batch_index, annotation in enumerate(annotations):
        atoms = sorted(annotation["atoms"], key=lambda item: int(item["atom_id"]))
        for expected_id, atom in enumerate(atoms):
            if int(atom["atom_id"]) != expected_id:
                raise ValueError("Atom ids must be contiguous")
            atom_anchors[batch_index, expected_id] = torch.tensor(
                (float(atom["image_x"]) / 768.0, float(atom["image_y"]) / 768.0),
                device=device,
            )
            atom_mask[batch_index, expected_id] = True
        for bond_index, bond in enumerate(annotation["bonds"]):
            bond_atom_refs[batch_index, bond_index] = torch.tensor(
                (int(bond["begin_atom"]), int(bond["end_atom"])), device=device
            )
            bond_mask[batch_index, bond_index] = True
    return {
        "atom_anchors": atom_anchors,
        "atom_mask": atom_mask,
        "bond_atom_refs": bond_atom_refs,
        "bond_mask": bond_mask,
    }


def _proposal_to_target_atoms(
    predicted_pixels: np.ndarray,
    proposal_mask: np.ndarray,
    target_pixels: np.ndarray,
    *,
    radius_pixels: float,
) -> dict[int, int]:
    proposal_indices = np.flatnonzero(proposal_mask)
    if not len(proposal_indices) or not len(target_pixels):
        return {}
    distances = np.linalg.norm(
        predicted_pixels[proposal_indices, None, :] - target_pixels[None, :, :],
        axis=-1,
    )
    rows, columns = linear_sum_assignment(distances)
    return {
        int(proposal_indices[row]): int(column)
        for row, column in zip(rows, columns)
        if float(distances[row, column]) <= radius_pixels
    }


def _evaluate_mode(
    *,
    image: Image.Image,
    predicted_pixels: np.ndarray,
    candidate_atom_mask: np.ndarray,
    proposal_to_target: dict[int, int],
    expected_pairs: set[tuple[int, int]],
    dense_pairs: set[tuple[int, int]],
    existing_atom_pairs: set[tuple[int, int]],
    rescue_config: LineSegmentRescueConfig,
    minimum_support_segments: int,
) -> dict:
    candidates = tuple(
        candidate
        for candidate in detect_atom_pair_line_candidates(
            image,
            predicted_pixels,
            atom_mask=candidate_atom_mask,
            config=rescue_config,
        )
        if len(candidate.support_segment_indices) >= minimum_support_segments
    )
    mapped_pairs = set()
    unknown_pairs = 0
    support_segments = 0
    maximum_line_length = 0.0
    for candidate in candidates:
        support_segments += len(candidate.support_segment_indices)
        maximum_line_length = max(
            maximum_line_length, candidate.maximum_support_length_pixels
        )
        begin, end = candidate.atom_pair
        if begin not in proposal_to_target or end not in proposal_to_target:
            unknown_pairs += 1
            continue
        mapped_pair = tuple(
            sorted((proposal_to_target[begin], proposal_to_target[end]))
        )
        mapped_pairs.add(mapped_pair)
    true_pairs = mapped_pairs & expected_pairs
    false_pairs = mapped_pairs - expected_pairs
    rescued_pairs = true_pairs - dense_pairs
    union_pairs = dense_pairs | true_pairs
    line_only = tuple(
        candidate
        for candidate in candidates
        if candidate.atom_pair not in existing_atom_pairs
    )
    line_only_mapped = set()
    line_only_unknown = 0
    for candidate in line_only:
        begin, end = candidate.atom_pair
        if begin not in proposal_to_target or end not in proposal_to_target:
            line_only_unknown += 1
            continue
        line_only_mapped.add(
            tuple(sorted((proposal_to_target[begin], proposal_to_target[end])))
        )
    line_only_true = line_only_mapped & expected_pairs
    line_only_false = line_only_mapped - expected_pairs
    line_only_rescued = line_only_true - dense_pairs
    line_only_union = dense_pairs | line_only_true
    return {
        "line_candidate_pairs": len(candidates),
        "line_support_segments": support_segments,
        "line_mapped_true_pairs": len(true_pairs),
        "line_mapped_false_pairs": len(false_pairs),
        "line_unknown_pairs": unknown_pairs,
        "line_rescued_target_pairs": len(rescued_pairs),
        "union_covered_pairs": len(union_pairs),
        "union_complete": union_pairs == expected_pairs,
        "maximum_line_length_pixels": maximum_line_length,
        "rescued_pairs": [list(pair) for pair in sorted(rescued_pairs)],
        "false_pairs": [list(pair) for pair in sorted(false_pairs)],
        "line_only_candidate_pairs": len(line_only),
        "line_only_mapped_true_pairs": len(line_only_true),
        "line_only_mapped_false_pairs": len(line_only_false),
        "line_only_unknown_pairs": line_only_unknown,
        "line_only_rescued_target_pairs": len(line_only_rescued),
        "line_only_union_covered_pairs": len(line_only_union),
        "line_only_union_complete": line_only_union == expected_pairs,
        "line_only_rescued_pairs": [
            list(pair) for pair in sorted(line_only_rescued)
        ],
    }


def _summarize(records: list[dict]) -> dict:
    samples = max(len(records), 1)
    target_bonds = sum(int(record["target_bonds"]) for record in records)
    result = {
        "samples": len(records),
        "target_bonds": target_bonds,
        "dense_covered_pairs": sum(
            int(record["dense_covered_pairs"]) for record in records
        ),
        "dense_complete": sum(bool(record["dense_complete"]) for record in records),
    }
    result["dense_bond_recall"] = result["dense_covered_pairs"] / max(target_bonds, 1)
    result["dense_complete_rate"] = result["dense_complete"] / samples
    for mode in ("proposal_atoms", "selected_atoms"):
        rows = [record[mode] for record in records]
        prefix = mode
        for key in (
            "line_candidate_pairs",
            "line_support_segments",
            "line_mapped_true_pairs",
            "line_mapped_false_pairs",
            "line_unknown_pairs",
            "line_rescued_target_pairs",
            "union_covered_pairs",
            "line_only_candidate_pairs",
            "line_only_mapped_true_pairs",
            "line_only_mapped_false_pairs",
            "line_only_unknown_pairs",
            "line_only_rescued_target_pairs",
            "line_only_union_covered_pairs",
        ):
            result[f"{prefix}_{key}"] = sum(int(row[key]) for row in rows)
        result[f"{prefix}_union_complete"] = sum(
            bool(row["union_complete"]) for row in rows
        )
        result[f"{prefix}_line_only_union_complete"] = sum(
            bool(row["line_only_union_complete"]) for row in rows
        )
        result[f"{prefix}_line_pair_precision"] = result[
            f"{prefix}_line_mapped_true_pairs"
        ] / max(
            result[f"{prefix}_line_mapped_true_pairs"]
            + result[f"{prefix}_line_mapped_false_pairs"],
            1,
        )
        result[f"{prefix}_line_bond_recall"] = result[
            f"{prefix}_line_mapped_true_pairs"
        ] / max(target_bonds, 1)
        result[f"{prefix}_union_bond_recall"] = result[
            f"{prefix}_union_covered_pairs"
        ] / max(target_bonds, 1)
        result[f"{prefix}_union_complete_rate"] = result[
            f"{prefix}_union_complete"
        ] / samples
        result[f"{prefix}_line_only_pair_precision"] = result[
            f"{prefix}_line_only_mapped_true_pairs"
        ] / max(
            result[f"{prefix}_line_only_mapped_true_pairs"]
            + result[f"{prefix}_line_only_mapped_false_pairs"],
            1,
        )
        result[f"{prefix}_line_only_union_bond_recall"] = result[
            f"{prefix}_line_only_union_covered_pairs"
        ] / max(target_bonds, 1)
        result[f"{prefix}_line_only_union_complete_rate"] = result[
            f"{prefix}_line_only_union_complete"
        ] / samples
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[str(record["difficulty"])].append(record)
    if len(grouped) > 1:
        result["difficulty"] = {
            name: _summarize(rows) | {"difficulty": {}}
            for name, rows in sorted(grouped.items())
        }
    return result


@torch.no_grad()
def _process_batch(
    *,
    identifiers: list[str],
    difficulties: list[str],
    images: list[Image.Image],
    targets: dict[str, torch.Tensor],
    vision,
    atom_head,
    bond_head,
    set_head,
    proposal_config: PredictedProposalConfig,
    rescue_config: LineSegmentRescueConfig,
    atom_selection_threshold: float,
    minimum_support_segments: int,
) -> list[dict]:
    device = targets["atom_anchors"].device
    normalized_images = [
        image.convert("RGB").resize((768, 768), Image.Resampling.BICUBIC)
        if image.size != (768, 768)
        else image.convert("RGB")
        for image in images
    ]
    pixels = torch.stack([_image_tensor(image) for image in normalized_images]).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        features = vision.forward_stroke_resolution(pixels)
        atom_output = atom_head(features)
        bond_output = forward_frozen_bond_head(vision, bond_head, features)
        atom_proposals = decode_predicted_atom_proposals(atom_output, proposal_config)
        bond_proposals = decode_predicted_bond_proposals(bond_output, proposal_config)
        proposals = associate_predicted_proposals(
            atom_proposals, bond_proposals, targets, proposal_config
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

    records = []
    for batch_index, (identifier, difficulty, image) in enumerate(
        zip(identifiers, difficulties, normalized_images)
    ):
        atom_count = int(targets["atom_mask"][batch_index].sum())
        bond_count = int(targets["bond_mask"][batch_index].sum())
        target_pixels = (
            targets["atom_anchors"][batch_index, :atom_count].float().cpu().numpy()
            * 768.0
        )
        predicted_pixels = (
            proposals["atom_geometry"][batch_index, :, :2].float().cpu().numpy()
            * 768.0
        )
        proposal_mask = proposals["atom_mask"][batch_index].cpu().numpy().astype(bool)
        selected_mask = proposal_mask & (
            set_output["atom_presence_logits"][batch_index].float().cpu().numpy()
            >= atom_selection_threshold
        )
        proposal_to_target = _proposal_to_target_atoms(
            predicted_pixels,
            proposal_mask,
            target_pixels,
            radius_pixels=proposal_config.atom_match_radius_px,
        )
        references = targets["bond_atom_refs"][batch_index, :bond_count].cpu().tolist()
        expected_pairs = {
            tuple(sorted((int(begin), int(end)))) for begin, end in references
        }
        matched = proposals["bond_target_matched"][batch_index, :bond_count].cpu()
        dense_pairs = {
            tuple(sorted((int(references[index][0]), int(references[index][1]))))
            for index in range(bond_count)
            if bool(matched[index])
        }
        predicted_refs = (
            set_output["incidence_logits"][batch_index].argmax(-1).cpu().tolist()
        )
        existing_atom_pairs = {
            tuple(sorted((int(begin), int(end))))
            for proposal_index, (begin, end) in enumerate(predicted_refs)
            if bool(proposals["bond_mask"][batch_index, proposal_index])
            and int(begin) != int(end)
        }
        records.append(
            {
                "sample_id": identifier,
                "difficulty": difficulty,
                "target_atoms": atom_count,
                "target_bonds": bond_count,
                "matched_atom_proposals": len(set(proposal_to_target.values())),
                "selected_atom_proposals": int(selected_mask.sum()),
                "dense_covered_pairs": len(dense_pairs),
                "dense_complete": dense_pairs == expected_pairs,
                "proposal_atoms": _evaluate_mode(
                    image=image,
                    predicted_pixels=predicted_pixels,
                    candidate_atom_mask=proposal_mask,
                    proposal_to_target=proposal_to_target,
                    expected_pairs=expected_pairs,
                    dense_pairs=dense_pairs,
                    existing_atom_pairs=existing_atom_pairs,
                    rescue_config=rescue_config,
                    minimum_support_segments=minimum_support_segments,
                ),
                "selected_atoms": _evaluate_mode(
                    image=image,
                    predicted_pixels=predicted_pixels,
                    candidate_atom_mask=selected_mask,
                    proposal_to_target=proposal_to_target,
                    expected_pairs=expected_pairs,
                    dense_pairs=dense_pairs,
                    existing_atom_pairs=existing_atom_pairs,
                    rescue_config=rescue_config,
                    minimum_support_segments=minimum_support_segments,
                ),
            }
        )
    return records


def _internal_batches(
    *, samples: int, batch_size: int, device: torch.device
) -> Iterable[tuple[list[str], list[str], list[Image.Image], dict[str, torch.Tensor]]]:
    config = AtomBondSceneConfig(atom_queries=128, bond_queries=192)
    records, _ = load_validation_records(
        [DEFAULT_INTERNAL_SHARD], per_shard=samples, config=config
    )
    collator = AtomBondSceneCollator(config)
    for start in range(0, len(records), batch_size):
        rows = records[start : start + batch_size]
        batch = collator(rows)
        yield (
            [f"chemvision:{row['key']}" for row in rows],
            ["internal_validation"] * len(rows),
            [row["image"] for row in rows],
            stack_targets(batch, device),
        )


def _mr_batches(
    *, samples: int, batch_size: int, device: torch.device
) -> Iterable[tuple[list[str], list[str], list[Image.Image], dict[str, torch.Tensor]]]:
    manifest = Path(DEFAULT_MR_MANIFEST)
    root = manifest.parent
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))[:samples]
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        images = [Image.open(root / row["image_file"]).convert("RGB") for row in batch_rows]
        annotations = [
            json.loads((root / row["annotation_file"]).read_text())
            for row in batch_rows
        ]
        yield (
            [row["molecule_id"] for row in batch_rows],
            [row["difficulty"] for row in batch_rows],
            images,
            targets_from_annotations(annotations, device),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domain", choices=("internal", "mr-tadf", "all"), default="all"
    )
    parser.add_argument("--internal-samples", type=int, default=498)
    parser.add_argument("--mr-samples", type=int, default=1100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grounded-graph-checkpoint", default=DEFAULT_GRAPH)
    parser.add_argument("--atom-checkpoint", default=DEFAULT_ATOM)
    parser.add_argument("--bond-checkpoint", default=DEFAULT_BOND)
    parser.add_argument("--set-checkpoint", default=DEFAULT_PRIMARY)
    parser.add_argument("--atom-selection-threshold", type=float, default=-0.25)
    parser.add_argument("--endpoint-distance-pixels", type=float, default=16.0)
    parser.add_argument("--maximum-angle-degrees", type=float, default=12.0)
    parser.add_argument("--minimum-support-segments", type=int, default=2)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Line segment rescue audit requires CUDA")
    if min(args.internal_samples, args.mr_samples, args.batch_size) < 1:
        raise ValueError("Sample counts and batch size must be positive")
    if args.minimum_support_segments < 1:
        raise ValueError("minimum-support-segments must be positive")
    device = torch.device("cuda")
    vision, atom_head, bond_head, _ = load_frozen_components(
        Path(args.grounded_graph_checkpoint),
        Path(args.atom_checkpoint),
        Path(args.bond_checkpoint),
        device,
    )
    set_head, _ = _load_set_head(args.set_checkpoint, device)
    proposal_config = PredictedProposalConfig(
        maximum_atoms=128,
        maximum_bonds=192,
        atom_threshold=0.01,
        bond_threshold=0.01,
    )
    rescue_config = LineSegmentRescueConfig(
        maximum_endpoint_distance_pixels=args.endpoint_distance_pixels,
        maximum_angle_degrees=args.maximum_angle_degrees,
    )
    domains = []
    if args.domain in {"internal", "all"}:
        domains.append(
            (
                "internal",
                _internal_batches(
                    samples=args.internal_samples,
                    batch_size=args.batch_size,
                    device=device,
                ),
            )
        )
    if args.domain in {"mr-tadf", "all"}:
        domains.append(
            (
                "mr_tadf",
                _mr_batches(
                    samples=args.mr_samples,
                    batch_size=args.batch_size,
                    device=device,
                ),
            )
        )

    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": "line-segment-rescue-audit-0.1",
        "model_version": "v0.9.84",
        "protocol": {
            "maximum_bond_length": None,
            "line_detector": "OpenCV LSD_REFINE_STD",
            "semantic_unit": "unordered predicted AtomPair",
            "multiple_strokes_same_pair": "one semantic bond candidate",
            "threshold_selection": "fixed before full MR-TADF audit",
            "mr_tadf_used_for_training": False,
            "mr_tadf_used_for_threshold_tuning": False,
            "proposal_config": asdict(proposal_config),
            "rescue_config": asdict(rescue_config),
            "atom_selection_threshold": args.atom_selection_threshold,
            "minimum_support_segments": args.minimum_support_segments,
        },
        "domains": {},
    }
    for domain_name, batches in domains:
        records = []
        for identifiers, difficulties, images, targets in batches:
            records.extend(
                _process_batch(
                    identifiers=identifiers,
                    difficulties=difficulties,
                    images=images,
                    targets=targets,
                    vision=vision,
                    atom_head=atom_head,
                    bond_head=bond_head,
                    set_head=set_head,
                    proposal_config=proposal_config,
                    rescue_config=rescue_config,
                    atom_selection_threshold=args.atom_selection_threshold,
                    minimum_support_segments=args.minimum_support_segments,
                )
            )
        domain_output = output / domain_name
        domain_output.mkdir(parents=True, exist_ok=True)
        (domain_output / "records.jsonl").write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )
        domain_summary = _summarize(records)
        (domain_output / "summary.json").write_text(
            json.dumps(domain_summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        summary["domains"][domain_name] = domain_summary
        print(json.dumps({domain_name: domain_summary}, ensure_ascii=False, indent=2))
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
