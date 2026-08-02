#!/usr/bin/env python3
"""Evaluate unlabeled molecular skeletons without OCR, repair, or SDF rules."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
import torch
from torch.utils.data import DataLoader

from Data.sdf.chemsvg_sequence_v4 import (
    ParsedChemSVGSequenceV4,
    VertexRecordV4,
    parse_chemsvg_v4,
)
from chemvector.data import ChemVectorIterableDataset
from chemvector.evaluate import load_model
from chemvector.parallel_v6 import score_vertex_pairs_parallel
from chemvector.stroke import STROKE_CLASSES
from chemvector.structure_gate import (
    expert_route_probability,
    load_packed_structure_expert,
    load_structure_gate,
)
from chemvector.train import expand_shards
from chemvector.vertex import decode_vertices


@dataclass(frozen=True, slots=True)
class SkeletonComparison:
    expected_vertices: int
    predicted_vertices: int
    matched_vertices: int
    expected_edges: int
    predicted_edges: int
    correct_edges: int
    expected_visible_bond_types: int
    correct_bond_types: int
    false_edges: int
    missed_edges: int
    vertex_exact: bool
    edge_set_exact: bool
    skeleton_exact: bool


def skeleton_collate(samples: list[dict], image_size: int) -> dict:
    pixels = []
    for sample in samples:
        image = sample["image"]
        if image.size != (image_size, image_size):
            image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        pixels.append((torch.from_numpy(array).permute(2, 0, 1) - 0.5) / 0.5)
    return {
        "pixel_values": torch.stack(pixels),
        "keys": [sample["key"] for sample in samples],
        "target_sequences": [sample["target_sequence"] for sample in samples],
    }


def parse_thresholds(value: str) -> tuple[float, ...]:
    thresholds = tuple(float(item) for item in value.split(","))
    if not thresholds or any(not 0.0 <= item <= 1.0 for item in thresholds):
        raise argparse.ArgumentTypeError("Thresholds must be comma-separated values in [0, 1]")
    return thresholds


def match_vertices(
    expected: ParsedChemSVGSequenceV4,
    predicted_vertices: list[VertexRecordV4] | tuple[VertexRecordV4, ...],
    *,
    tolerance_pixels: float,
    image_size: int,
) -> dict[int, int]:
    if not expected.vertices or not predicted_vertices:
        return {}
    expected_points = np.asarray([(v.x, v.y) for v in expected.vertices], dtype=np.float32)
    predicted_points = np.asarray([(v.x, v.y) for v in predicted_vertices], dtype=np.float32)
    distances = np.linalg.norm(
        predicted_points[:, None, :] - expected_points[None, :, :],
        axis=-1,
    )
    predicted_indices, expected_indices = linear_sum_assignment(distances)
    tolerance = tolerance_pixels / image_size * 1023.0
    return {
        int(predicted_index): int(expected_index)
        for predicted_index, expected_index in zip(predicted_indices, expected_indices)
        if distances[predicted_index, expected_index] <= tolerance
    }


def compare_skeleton(
    expected: ParsedChemSVGSequenceV4,
    predicted_vertices: list[VertexRecordV4] | tuple[VertexRecordV4, ...],
    pairs: list[tuple[int, int]],
    probabilities: torch.Tensor,
    *,
    edge_threshold: float,
    tolerance_pixels: float = 6.0,
    image_size: int = 768,
) -> SkeletonComparison:
    mapping = match_vertices(
        expected,
        predicted_vertices,
        tolerance_pixels=tolerance_pixels,
        image_size=image_size,
    )
    expected_coordinate_to_index = {
        (vertex.x, vertex.y): index for index, vertex in enumerate(expected.vertices)
    }
    expected_edges = {
        tuple(
            sorted(
                (
                    expected_coordinate_to_index[stroke.begin],
                    expected_coordinate_to_index[stroke.end],
                )
            )
        ): stroke.display_type
        for stroke in expected.strokes
    }
    predicted_edges: dict[tuple[int, int], str] = {}
    unmatched_edges = 0
    for pair, row in zip(pairs, probabilities.cpu()):
        if 1.0 - float(row[0]) < edge_threshold:
            continue
        class_id = int(row[1:].argmax()) + 1
        if pair[0] not in mapping or pair[1] not in mapping:
            unmatched_edges += 1
            continue
        mapped_pair = tuple(sorted((mapping[pair[0]], mapping[pair[1]])))
        predicted_edges[mapped_pair] = STROKE_CLASSES[class_id]
    common = set(expected_edges) & set(predicted_edges)
    visible_expected = {
        pair: bond_type
        for pair, bond_type in expected_edges.items()
        if bond_type != "AROMATIC"
    }
    correct_bond_types = sum(
        pair in predicted_edges and visible_expected[pair] == predicted_edges[pair]
        for pair in visible_expected
    )
    false_edges = unmatched_edges + len(set(predicted_edges) - set(expected_edges))
    missed_edges = len(set(expected_edges) - set(predicted_edges))
    vertex_exact = (
        len(mapping) == len(expected.vertices) == len(predicted_vertices)
    )
    edge_set_exact = vertex_exact and not false_edges and not missed_edges
    skeleton_exact = edge_set_exact and correct_bond_types == len(visible_expected)
    return SkeletonComparison(
        expected_vertices=len(expected.vertices),
        predicted_vertices=len(predicted_vertices),
        matched_vertices=len(mapping),
        expected_edges=len(expected_edges),
        predicted_edges=len(predicted_edges) + unmatched_edges,
        correct_edges=len(common),
        expected_visible_bond_types=len(visible_expected),
        correct_bond_types=correct_bond_types,
        false_edges=false_edges,
        missed_edges=missed_edges,
        vertex_exact=vertex_exact,
        edge_set_exact=edge_set_exact,
        skeleton_exact=skeleton_exact,
    )


def detected_vertices(
    model,
    stroke_features: torch.Tensor,
    *,
    threshold: float,
    maximum_vertices: int,
) -> list[VertexRecordV4]:
    detections = decode_vertices(
        model.vertex_head(stroke_features),
        threshold=threshold,
        maximum_vertices=maximum_vertices,
    )[0]
    unique = {}
    for item in detections:
        coordinate = (
            max(0, min(1023, round(float(item["x"]) * 1023))),
            max(0, min(1023, round(float(item["y"]) * 1023))),
        )
        if coordinate not in unique or float(item["score"]) > float(
            unique[coordinate]["score"]
        ):
            unique[coordinate] = item
    return [
        VertexRecordV4(
            x=x,
            y=y,
            element="C",
            formal_charge=0,
            isotope=0,
            aromatic=bool(unique[(x, y)]["aromatic"]),
            explicit_h_count=0,
            no_implicit=False,
            radical_electrons=0,
            canonical_chiral_tag="CHI_UNSPECIFIED",
        )
        for x, y in sorted(unique, key=lambda point: (point[1], point[0]))
    ]


def summarize(records: list[dict], modes: tuple[str, ...], thresholds: tuple[float, ...]) -> dict:
    result = {"samples": len(records), "modes": {}}
    for mode in modes:
        mode_result = {}
        for threshold in thresholds:
            key = f"{threshold:.4f}"
            values = [record["modes"][mode][key] for record in records]
            expected_vertices = sum(value["expected_vertices"] for value in values)
            predicted_vertices = sum(value["predicted_vertices"] for value in values)
            expected_edges = sum(value["expected_edges"] for value in values)
            expected_visible_bond_types = sum(
                value["expected_visible_bond_types"] for value in values
            )
            predicted_edges = sum(value["predicted_edges"] for value in values)
            correct_edges = sum(value["correct_edges"] for value in values)
            mode_result[key] = {
                "vertex_precision": sum(value["matched_vertices"] for value in values)
                / max(predicted_vertices, 1),
                "vertex_recall": sum(value["matched_vertices"] for value in values)
                / max(expected_vertices, 1),
                "vertex_exact_rate": sum(value["vertex_exact"] for value in values)
                / max(len(values), 1),
                "edge_precision": correct_edges / max(predicted_edges, 1),
                "edge_recall": correct_edges / max(expected_edges, 1),
                "bond_type_accuracy": sum(value["correct_bond_types"] for value in values)
                / max(expected_visible_bond_types, 1),
                "edge_set_exact_rate": sum(value["edge_set_exact"] for value in values)
                / max(len(values), 1),
                "skeleton_exact_rate": sum(value["skeleton_exact"] for value in values)
                / max(len(values), 1),
                "false_edges": sum(value["false_edges"] for value in values),
                "missed_edges": sum(value["missed_edges"] for value in values),
            }
        result["modes"][mode] = mode_result
    return result


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gate-checkpoint")
    parser.add_argument("--gate-threshold", type=float, default=0.99)
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--vertex-threshold", type=float, default=0.3)
    parser.add_argument("--maximum-vertices", type=int, default=128)
    parser.add_argument("--vertex-tolerance-pixels", type=float, default=6.0)
    parser.add_argument(
        "--edge-thresholds",
        type=parse_thresholds,
        default=parse_thresholds("0.30,0.40,0.50,0.60,0.70"),
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("oracle_vertices", "detected_vertices"),
        default=("oracle_vertices", "detected_vertices"),
    )
    args = parser.parse_args()
    if args.samples < 1 or args.maximum_vertices < 1:
        raise ValueError("Sample and vertex limits must be positive")
    if not 0.0 <= args.gate_threshold <= 1.0:
        raise ValueError("gate-threshold must be in [0, 1]")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    device = torch.device(args.device)
    model, tokenizer = load_model(Path(args.checkpoint), device)
    model.eval()
    gate = None
    expert_model = None
    if args.gate_checkpoint:
        gate, gate_payload = load_structure_gate(args.gate_checkpoint, device)
        if Path(gate_payload["base_checkpoint"]).resolve() != Path(
            args.checkpoint
        ).resolve():
            raise ValueError("Gate base checkpoint does not match --checkpoint")
        if "expert_structure_tensors" in gate_payload:
            expert_model = load_packed_structure_expert(model, gate_payload, device)
        else:
            expert_model, _ = load_model(
                Path(gate_payload["expert_checkpoint"]), device
            )
            expert_model.eval()
    sequence_version = tokenizer.sequence_schema_version.removeprefix(
        "chemmolseq-"
    ).removeprefix("chemsvg-")
    loader = DataLoader(
        ChemVectorIterableDataset(
            expand_shards(args.shards),
            sequence_version=sequence_version,
        ),
        batch_size=1,
        num_workers=0,
        pin_memory=True,
        collate_fn=lambda samples: skeleton_collate(samples, args.image_size),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for batch in loader:
        if len(records) >= args.samples:
            break
        expected = parse_chemsvg_v4(batch["target_sequences"][0])
        pixels = batch["pixel_values"].to(device)
        inference_model = model
        route_probability = None
        if gate is not None:
            route_probability = expert_route_probability(model, gate, pixels)
            if route_probability >= args.gate_threshold:
                inference_model = expert_model
        stroke_features = inference_model.vision_encoder.forward_stroke_resolution(pixels)
        mode_vertices = {}
        if "oracle_vertices" in args.modes:
            mode_vertices["oracle_vertices"] = list(expected.vertices)
        if "detected_vertices" in args.modes:
            mode_vertices["detected_vertices"] = detected_vertices(
                inference_model,
                stroke_features,
                threshold=args.vertex_threshold,
                maximum_vertices=args.maximum_vertices,
            )
        record = {"key": batch["keys"][0], "modes": {}}
        if route_probability is not None:
            record["expert_route_probability"] = route_probability
            record["selected_expert"] = route_probability >= args.gate_threshold
        for mode, vertices in mode_vertices.items():
            pairs, probabilities = score_vertex_pairs_parallel(
                inference_model,
                stroke_features,
                vertices,
                pixels.shape[-2:],
            )
            record["modes"][mode] = {
                f"{threshold:.4f}": asdict(
                    compare_skeleton(
                        expected,
                        vertices,
                        pairs,
                        probabilities,
                        edge_threshold=threshold,
                        tolerance_pixels=args.vertex_tolerance_pixels,
                        image_size=args.image_size,
                    )
                )
                for threshold in args.edge_thresholds
            }
        records.append(record)
        if len(records) % 25 == 0:
            print(
                json.dumps(
                    summarize(records, tuple(args.modes), args.edge_thresholds),
                    ensure_ascii=False,
                ),
                flush=True,
            )
    with (output_dir / "records.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = summarize(records, tuple(args.modes), args.edge_thresholds)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
