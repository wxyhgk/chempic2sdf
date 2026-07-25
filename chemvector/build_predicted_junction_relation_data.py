#!/usr/bin/env python3
"""Build Bond-relation labels on frozen predicted Junctions."""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path

from PIL import Image
from scipy.optimize import linear_sum_assignment
import torch

from chemvector.data import ChemVectorIterableDataset, structure_labels
from chemvector.evaluate_semantic_bond_existence import file_sha256
from chemvector.image_io import image_tensor
from chemvector.infer_chemscene_hybrid import ChemSceneHybridPipeline
from chemvector.semantic_bond_relation import (
    SemanticBondExistence,
    SemanticBondRelationConfig,
)
from chemvector.train import expand_shards


def match_predicted_junctions(
    predicted_points: torch.Tensor,
    target_points: dict[int, tuple[float, float]],
    *,
    image_size: int,
    maximum_distance_pixels: float,
) -> tuple[dict[int, int], list[float]]:
    if predicted_points.ndim != 2 or predicted_points.size(-1) != 2:
        raise ValueError("predicted points must have shape [points, 2]")
    if maximum_distance_pixels <= 0.0:
        raise ValueError("matching distance must be positive")
    if not len(predicted_points) or not target_points:
        return {}, []
    atom_ids = sorted(target_points)
    targets = torch.tensor(
        [target_points[atom_id] for atom_id in atom_ids], dtype=torch.float32
    )
    distances = torch.cdist(predicted_points.float().cpu(), targets) * image_size
    predicted_indices, target_indices = linear_sum_assignment(distances.numpy())
    mapping = {}
    accepted_distances = []
    for predicted_index, target_index in zip(predicted_indices, target_indices):
        distance = float(distances[predicted_index, target_index])
        if distance > maximum_distance_pixels:
            continue
        mapping[int(predicted_index)] = atom_ids[int(target_index)]
        accepted_distances.append(distance)
    return mapping, accepted_distances


@torch.inference_mode()
def score_all_pairs(
    model: SemanticBondExistence,
    pixels: torch.Tensor,
    points: torch.Tensor,
    *,
    chunk_size: int,
) -> tuple[list[tuple[int, int]], torch.Tensor]:
    pairs = list(combinations(range(len(points)), 2))
    scores = []
    for start in range(0, len(pairs), chunk_size):
        current = pairs[start : start + chunk_size]
        pair_indices = torch.tensor(
            [current], device=pixels.device, dtype=torch.long
        )
        pair_mask = torch.ones(
            pair_indices.shape[:2], device=pixels.device, dtype=torch.bool
        )
        with torch.autocast(
            device_type=pixels.device.type,
            dtype=torch.bfloat16,
            enabled=pixels.device.type == "cuda",
        ):
            output = model(
                pixels,
                points.to(pixels.device)[None],
                pair_indices,
                pair_mask,
            )
        scores.append(output["edge_logits"].float().cpu().sigmoid())
    return pairs, torch.cat(scores) if scores else torch.empty(0)


def target_geometry(labels: dict) -> dict[int, tuple[float, float]]:
    width = float(labels["width"])
    height = float(labels["height"])
    return {
        int(atom["atom_id"]): (
            float(atom["center"][0]) / width,
            float(atom["center"][1]) / height,
        )
        for atom in labels["atoms"]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--existence-checkpoint", required=True)
    parser.add_argument(
        "--selector-checkpoint",
        default="outputs/checkpoints/chemvector-v0739-boundary-group-selector-9k-15k/best.pt",
    )
    parser.add_argument(
        "--bond-checkpoint",
        default="outputs/checkpoints/chemvector-v082-multiscale-joint-9k-1epoch/best.pt",
    )
    parser.add_argument(
        "--bond-set-selector-checkpoint",
        default="outputs/checkpoints/chemvector-v0821-junction-aware-bond-selector-9k/best.pt",
    )
    parser.add_argument(
        "--endpoint-grouper-checkpoint",
        default="outputs/checkpoints/chemvector-v0820-visual-endpoint-junction-grouper-capacity500/best.pt",
    )
    parser.add_argument(
        "--line-band-checkpoint",
        default="outputs/checkpoints/chemvector-v0738-node-conditioned-bond-jitter-9k-15k/end-to-end-best.pt",
    )
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--matching-radius-pixels", type=float, default=12.0)
    parser.add_argument("--hard-negative-ratio", type=float, default=1.5)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if min(args.samples, args.image_size, args.chunk_size) < 1:
        raise ValueError("sample, image, and chunk counts must be positive")
    if args.hard_negative_ratio <= 0.0:
        raise ValueError("hard-negative ratio must be positive")

    device = torch.device(args.device)
    pipeline = ChemSceneHybridPipeline(
        Path(args.selector_checkpoint),
        Path(args.bond_checkpoint),
        device,
        bond_set_selector_checkpoint=Path(args.bond_set_selector_checkpoint),
        endpoint_grouper_checkpoint=Path(args.endpoint_grouper_checkpoint),
        line_band_checkpoint=Path(args.line_band_checkpoint),
    )
    existence_checkpoint = Path(args.existence_checkpoint)
    existence_payload = torch.load(
        existence_checkpoint, map_location="cpu", weights_only=False
    )
    existence = SemanticBondExistence(
        SemanticBondRelationConfig(**existence_payload["config"])
    ).to(device)
    existence.load_state_dict(existence_payload["semantic_bond_existence"])
    existence.eval().requires_grad_(False)
    records = iter(
        ChemVectorIterableDataset(
            expand_shards(args.shards), sequence_version="0.4"
        )
    )

    output_samples = []
    failures = []
    totals = Counter()
    accepted_distances = []
    for attempted, record in enumerate(records, 1):
        if len(output_samples) >= args.samples:
            break
        image = record["image"].convert("RGB")
        if image.size != (args.image_size, args.image_size):
            image = image.resize(
                (args.image_size, args.image_size), Image.Resampling.BICUBIC
            )
        pixels = image_tensor(image).unsqueeze(0).to(device)
        try:
            points, _, _, _ = pipeline.predict_components(
                pixels,
                include_ocr=False,
                junction_source="junction_aware_bonds",
            )
        except Exception as error:
            failures.append({"key": record["key"], "error": str(error)})
            totals["failed_candidates"] += 1
            continue
        points = points.detach().float().cpu()
        labels = structure_labels(record)
        targets = target_geometry(labels)
        mapping, distances = match_predicted_junctions(
            points,
            targets,
            image_size=args.image_size,
            maximum_distance_pixels=args.matching_radius_pixels,
        )
        accepted_distances.extend(distances)
        target_bonds = {
            tuple(sorted((int(bond["begin_atom"]), int(bond["end_atom"])))): bond
            for bond in labels["bonds"]
        }
        positive_bonds = []
        positive_pairs = set()
        for first, second in combinations(range(len(points)), 2):
            if first not in mapping or second not in mapping:
                continue
            target_pair = tuple(sorted((mapping[first], mapping[second])))
            bond = target_bonds.get(target_pair)
            if bond is None:
                continue
            pair = (first, second)
            positive_pairs.add(pair)
            positive_bonds.append(
                {
                    "pair": list(pair),
                    "display_type": str(bond["display_type"]),
                    "chemical_type": str(bond["chemical_type"]),
                }
            )
        all_pairs, scores = score_all_pairs(
            existence, pixels, points, chunk_size=args.chunk_size
        )
        negative_offsets = [
            offset for offset, pair in enumerate(all_pairs) if pair not in positive_pairs
        ]
        hard_count = min(
            len(negative_offsets),
            max(1, round(len(positive_pairs) * args.hard_negative_ratio)),
        )
        ranked_negative_offsets = sorted(
            negative_offsets, key=lambda offset: float(scores[offset]), reverse=True
        )[:hard_count]
        output_samples.append(
            {
                "key": record["key"],
                "points": points.tolist(),
                "point_to_atom": [mapping.get(point_index, -1) for point_index in range(len(points))],
                "positive_bonds": positive_bonds,
                "hard_negative_pairs": [
                    list(all_pairs[offset]) for offset in ranked_negative_offsets
                ],
                "hard_negative_scores": [
                    float(scores[offset]) for offset in ranked_negative_offsets
                ],
                "target_atom_count": len(targets),
                "target_bond_count": len(target_bonds),
            }
        )
        totals["samples"] += 1
        totals["predicted_points"] += len(points)
        totals["target_atoms"] += len(targets)
        totals["matched_points"] += len(mapping)
        totals["unmatched_points"] += len(points) - len(mapping)
        totals["target_bonds"] += len(target_bonds)
        totals["recoverable_bonds"] += len(positive_pairs)
        totals["point_count_exact"] += len(points) == len(targets)
        if len(output_samples) % 25 == 0:
            print(
                f"built={len(output_samples)} attempted={attempted}", flush=True
            )

    sample_count = max(totals["samples"], 1)
    result = {
        "schema_version": "predicted-junction-relations-0.1",
        "protocol": "frozen v0.9.28 Junctions matched one-to-one to GT atoms; all relation labels are defined on predicted points",
        "uses_length_for_relation_selection": False,
        "matching_radius_pixels": args.matching_radius_pixels,
        "existence_checkpoint": str(existence_checkpoint),
        "existence_checkpoint_sha256": file_sha256(existence_checkpoint),
        "samples": output_samples,
        "failures": failures,
        "summary": {
            **dict(totals),
            "point_count_exact_rate": totals["point_count_exact"] / sample_count,
            "point_match_rate": totals["matched_points"]
            / max(totals["predicted_points"], 1),
            "gt_bond_recoverable_rate": totals["recoverable_bonds"]
            / max(totals["target_bonds"], 1),
            "matching_distance_mean_pixels": sum(accepted_distances)
            / max(len(accepted_distances), 1),
            "matching_distance_max_pixels": max(accepted_distances, default=0.0),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["summary"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()


__all__ = ["match_predicted_junctions"]
