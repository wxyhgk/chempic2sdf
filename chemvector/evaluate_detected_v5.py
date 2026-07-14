#!/usr/bin/env python3
"""End-to-end scaffold validation for detected ChemSVG v0.5 inference."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
from rdkit import Chem
from scipy.optimize import linear_sum_assignment
import torch

from Data.sdf.chemsvg_sequence_v4 import (
    ParsedChemSVGSequenceV4,
    mol_from_chemsvg_v4,
    parse_chemsvg_v4,
)
from Data.sdf.chemsvg_tokenizer_v4 import ChemSVGTokenizerV4
from chemvector.evaluate import load_model, normalized_smiles
from chemvector.generation_v4 import generate_detected_chemsvg_v4
from chemvector.parallel_v6 import (
    generate_parallel_chemsvg_v6,
    strip_autoregressive_modules,
)
from chemvector.repair_v4 import (
    InkGraphRepairConfig,
    repair_chemsvg_v4_from_image,
)
from chemvector.train import expand_shards, make_loader


@dataclass(frozen=True, slots=True)
class GraphComparison:
    expected_vertices: int
    predicted_vertices: int
    matched_vertices: int
    correct_elements: int
    expected_edges: int
    predicted_edges: int
    correct_edges: int
    correct_bond_types: int
    exact_topology: bool


def tensor_to_image(pixel_values: torch.Tensor) -> Image.Image:
    array = (
        (pixel_values.detach().cpu().float().clamp(-1, 1) * 0.5 + 0.5)
        .permute(1, 2, 0)
        .numpy()
    )
    return Image.fromarray(np.rint(array * 255).astype(np.uint8), mode="RGB")


def _coordinate_array(parsed: ParsedChemSVGSequenceV4) -> np.ndarray:
    return np.array([(vertex.x, vertex.y) for vertex in parsed.vertices], dtype=np.float64)


def compare_graphs(
    expected: ParsedChemSVGSequenceV4,
    predicted: ParsedChemSVGSequenceV4,
    *,
    tolerance_pixels: float = 6.0,
    image_size: int = 768,
) -> GraphComparison:
    expected_points = _coordinate_array(expected)
    predicted_points = _coordinate_array(predicted)
    distances = np.linalg.norm(
        predicted_points[:, None, :] - expected_points[None, :, :],
        axis=-1,
    )
    predicted_indices, expected_indices = linear_sum_assignment(distances)
    tolerance = tolerance_pixels / image_size * 1023.0
    mapping = {
        int(predicted_index): int(expected_index)
        for predicted_index, expected_index in zip(predicted_indices, expected_indices)
        if distances[predicted_index, expected_index] <= tolerance
    }
    correct_elements = sum(
        predicted.vertices[predicted_index].element
        == expected.vertices[expected_index].element
        and predicted.vertices[predicted_index].formal_charge
        == expected.vertices[expected_index].formal_charge
        and predicted.vertices[predicted_index].isotope
        == expected.vertices[expected_index].isotope
        for predicted_index, expected_index in mapping.items()
    )

    expected_coordinate_to_index = {
        (vertex.x, vertex.y): index for index, vertex in enumerate(expected.vertices)
    }
    predicted_coordinate_to_index = {
        (vertex.x, vertex.y): index for index, vertex in enumerate(predicted.vertices)
    }
    expected_edges = {
        tuple(
            sorted(
                (
                    expected_coordinate_to_index[stroke.begin],
                    expected_coordinate_to_index[stroke.end],
                )
            )
        ): stroke.bond_type
        for stroke in expected.strokes
    }
    predicted_edges = {}
    unmapped_edges = 0
    for stroke in predicted.strokes:
        begin = predicted_coordinate_to_index[stroke.begin]
        end = predicted_coordinate_to_index[stroke.end]
        if begin not in mapping or end not in mapping:
            unmapped_edges += 1
            continue
        predicted_edges[tuple(sorted((mapping[begin], mapping[end])))] = stroke.bond_type
    common = set(expected_edges) & set(predicted_edges)
    correct_bond_types = sum(
        expected_edges[pair] == predicted_edges[pair] for pair in common
    )
    predicted_edge_count = len(predicted_edges) + unmapped_edges
    exact_topology = (
        len(mapping) == len(expected.vertices) == len(predicted.vertices)
        and correct_elements == len(expected.vertices)
        and set(expected_edges) == set(predicted_edges)
        and correct_bond_types == len(expected_edges)
        and unmapped_edges == 0
    )
    return GraphComparison(
        expected_vertices=len(expected.vertices),
        predicted_vertices=len(predicted.vertices),
        matched_vertices=len(mapping),
        correct_elements=correct_elements,
        expected_edges=len(expected_edges),
        predicted_edges=predicted_edge_count,
        correct_edges=len(common),
        correct_bond_types=correct_bond_types,
        exact_topology=exact_topology,
    )


def draw_overlay(
    image: Image.Image,
    expected: ParsedChemSVGSequenceV4,
    predicted: ParsedChemSVGSequenceV4 | None,
) -> Image.Image:
    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    width, height = overlay.size

    def point(vertex):
        return vertex.x / 1023.0 * width, vertex.y / 1023.0 * height

    for stroke in expected.strokes:
        coordinates = {(vertex.x, vertex.y): vertex for vertex in expected.vertices}
        draw.line((*point(coordinates[stroke.begin]), *point(coordinates[stroke.end])), fill=(0, 180, 0), width=2)
    if predicted is not None:
        coordinates = {(vertex.x, vertex.y): vertex for vertex in predicted.vertices}
        for stroke in predicted.strokes:
            draw.line((*point(coordinates[stroke.begin]), *point(coordinates[stroke.end])), fill=(220, 0, 0), width=1)
        for vertex in predicted.vertices:
            x, y = point(vertex)
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline=(220, 0, 0), width=1)
    return overlay


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / max(float(denominator), 1.0)


def summarize(records: list[dict]) -> dict:
    expected_vertices = sum(record.get("expected_vertices", 0) for record in records)
    predicted_vertices = sum(record.get("predicted_vertices", 0) for record in records)
    matched_vertices = sum(record.get("matched_vertices", 0) for record in records)
    expected_edges = sum(record.get("expected_edges", 0) for record in records)
    predicted_edges = sum(record.get("predicted_edges", 0) for record in records)
    correct_edges = sum(record.get("correct_edges", 0) for record in records)
    valid = sum(bool(record.get("sdf_valid")) for record in records)
    repaired = [record for record in records if record.get("repair_success")]
    repaired_expected_vertices = sum(record["expected_vertices"] for record in repaired)
    repaired_predicted_vertices = sum(record["predicted_vertices"] for record in repaired)
    repaired_matched_vertices = sum(record["matched_vertices"] for record in repaired)
    repaired_expected_edges = sum(record["expected_edges"] for record in repaired)
    repaired_predicted_edges = sum(record["predicted_edges"] for record in repaired)
    repaired_correct_edges = sum(record["correct_edges"] for record in repaired)
    failure_categories = {
        "disconnected": 0,
        "valence": 0,
        "kekulize": 0,
        "other": 0,
    }
    for record in records:
        if record.get("repair_success"):
            continue
        error = record.get("error", "")
        if "not connected" in error:
            failure_categories["disconnected"] += 1
        elif "Explicit valence" in error:
            failure_categories["valence"] += 1
        elif "kekulize" in error.lower() or "non-ring atom" in error:
            failure_categories["kekulize"] += 1
        else:
            failure_categories["other"] += 1
    molecules: dict[str, list[dict]] = {}
    for record in records:
        molecule_key = record["key"].rsplit("_", 1)[0]
        molecules.setdefault(molecule_key, []).append(record)
    exact_topology = [record for record in records if record.get("exact_topology")]
    layouts: dict[str, list[dict]] = {}
    for record in records:
        layout = record["key"].rsplit("_", 1)[-1]
        layouts.setdefault(layout, []).append(record)
    result = {
        "samples": len(records),
        "molecules": len(molecules),
        "completed_generation_rate": _safe_ratio(
            sum(bool(record.get("generation_completed")) for record in records),
            len(records),
        ),
        "repair_success_rate": _safe_ratio(
            sum(bool(record.get("repair_success")) for record in records),
            len(records),
        ),
        "sdf_valid_rate": _safe_ratio(valid, len(records)),
        "chemical_match_rate": _safe_ratio(
            sum(bool(record.get("chemical_match")) for record in records),
            len(records),
        ),
        "exact_topology_rate": _safe_ratio(
            sum(bool(record.get("exact_topology")) for record in records),
            len(records),
        ),
        "chemical_match_given_exact_topology": _safe_ratio(
            sum(bool(record.get("chemical_match")) for record in exact_topology),
            len(exact_topology),
        ),
        "vertex_precision": _safe_ratio(matched_vertices, predicted_vertices),
        "vertex_recall": _safe_ratio(matched_vertices, expected_vertices),
        "element_accuracy_on_matched": _safe_ratio(
            sum(record.get("correct_elements", 0) for record in records),
            matched_vertices,
        ),
        "edge_precision": _safe_ratio(correct_edges, predicted_edges),
        "edge_recall": _safe_ratio(correct_edges, expected_edges),
        "bond_type_accuracy_on_correct_edges": _safe_ratio(
            sum(record.get("correct_bond_types", 0) for record in records),
            correct_edges,
        ),
        "average_seconds": _safe_ratio(
            sum(record.get("seconds", 0.0) for record in records),
            len(records),
        ),
        "conditional_vertex_precision": _safe_ratio(
            repaired_matched_vertices,
            repaired_predicted_vertices,
        ),
        "conditional_vertex_recall": _safe_ratio(
            repaired_matched_vertices,
            repaired_expected_vertices,
        ),
        "conditional_edge_precision": _safe_ratio(
            repaired_correct_edges,
            repaired_predicted_edges,
        ),
        "conditional_edge_recall": _safe_ratio(
            repaired_correct_edges,
            repaired_expected_edges,
        ),
        "molecule_any_repair_rate": _safe_ratio(
            sum(any(record.get("repair_success") for record in group) for group in molecules.values()),
            len(molecules),
        ),
        "molecule_any_chemical_match_rate": _safe_ratio(
            sum(any(record.get("chemical_match") for record in group) for group in molecules.values()),
            len(molecules),
        ),
        "failure_categories": failure_categories,
        "layout_rates": {
            layout: {
                "samples": len(group),
                "repair_success_rate": _safe_ratio(
                    sum(bool(record.get("repair_success")) for record in group),
                    len(group),
                ),
                "exact_topology_rate": _safe_ratio(
                    sum(bool(record.get("exact_topology")) for record in group),
                    len(group),
                ),
                "chemical_match_rate": _safe_ratio(
                    sum(bool(record.get("chemical_match")) for record in group),
                    len(group),
                ),
            }
            for layout, group in sorted(layouts.items())
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--inference-mode",
        choices=("autoregressive", "parallel"),
        default="autoregressive",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--vertex-threshold", type=float, default=0.3)
    parser.add_argument("--stroke-threshold", type=float, default=0.5)
    parser.add_argument("--ocr-roi-radius", type=float, default=20.0)
    parser.add_argument("--vertex-tolerance-pixels", type=float, default=6.0)
    parser.add_argument("--failure-overlays", type=int, default=20)
    parser.add_argument("--reject-keys-file")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if args.inference_mode == "autoregressive" and args.device != "cuda":
        raise ValueError("Autoregressive evaluation currently requires CUDA")
    if args.samples < 1:
        raise ValueError("samples must be positive")
    if args.cpu_threads < 1:
        raise ValueError("cpu_threads must be positive")
    torch.set_num_threads(args.cpu_threads)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    records = []
    completed_keys = set()
    if records_path.is_file():
        records = [
            json.loads(line)
            for line in records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        completed_keys = {record["key"] for record in records}

    device = torch.device(args.device)
    model, tokenizer = load_model(Path(args.checkpoint), device)
    if not isinstance(tokenizer, ChemSVGTokenizerV4):
        raise ValueError("Detected evaluation requires ChemSVG v0.4")
    if args.inference_mode == "parallel":
        strip_autoregressive_modules(model)
    reject_keys = set()
    if args.reject_keys_file:
        reject_keys = {
            line.strip()
            for line in Path(args.reject_keys_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
    loader = make_loader(
        expand_shards(args.shards),
        tokenizer,
        model.config,
        batch_size=1,
        image_size=args.image_size,
        workers=0,
        shuffle=False,
        reject_keys=reject_keys,
    )
    overlays = len(list((output_dir / "failures").glob("*.png")))
    for batch in loader:
        if len(records) >= args.samples:
            break
        key = batch["keys"][0]
        if key in completed_keys:
            continue
        expected_sequence = batch["target_sequences"][0]
        expected = parse_chemsvg_v4(expected_sequence)
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        image = tensor_to_image(batch["pixel_values"][0])
        started = time.perf_counter()
        predicted = None
        record = {
            "key": key,
            "generation_completed": False,
            "repair_success": False,
            "sdf_valid": False,
            "chemical_match": False,
            "exact_topology": False,
            "expected_vertices": len(expected.vertices),
            "predicted_vertices": 0,
            "matched_vertices": 0,
            "correct_elements": 0,
            "expected_edges": len(expected.strokes),
            "predicted_edges": 0,
            "correct_edges": 0,
            "correct_bond_types": 0,
        }
        try:
            if args.inference_mode == "parallel":
                generated = generate_parallel_chemsvg_v6(
                    model,
                    pixels,
                    vertex_threshold=args.vertex_threshold,
                    edge_threshold=args.stroke_threshold,
                    ocr_roi_radius=args.ocr_roi_radius,
                )
                generated_sequence = generated.sequence
                record["generation_completed"] = True
                record["generation_chemically_valid"] = False
                record["generation_error"] = None
                repair_config = InkGraphRepairConfig(
                    aromatic_vote_source="vertices"
                )
            else:
                generated = generate_detected_chemsvg_v4(
                    model,
                    pixels,
                    tokenizer,
                    vertex_threshold=args.vertex_threshold,
                    stroke_threshold=args.stroke_threshold,
                    ocr_roi_radius=args.ocr_roi_radius,
                )
                record["generation_completed"] = generated.completed
                record["generation_chemically_valid"] = generated.chemically_valid
                record["generation_error"] = generated.error
                if generated.sequence is None:
                    raise ValueError(
                        generated.error or "generation returned no sequence"
                    )
                generated_sequence = generated.sequence
                repair_config = None
            repaired = repair_chemsvg_v4_from_image(
                generated_sequence,
                image,
                config=repair_config,
            )
            predicted = parse_chemsvg_v4(repaired.sequence)
            record["repair_success"] = True
            comparison = compare_graphs(
                expected,
                predicted,
                tolerance_pixels=args.vertex_tolerance_pixels,
                image_size=args.image_size,
            )
            record.update(asdict(comparison))
            actual_mol = mol_from_chemsvg_v4(repaired.sequence)
            record["sdf_valid"] = True
            actual_smiles = normalized_smiles(actual_mol)
            actual_inchikey = Chem.MolToInchiKey(actual_mol)
            expected_mol = mol_from_chemsvg_v4(expected_sequence)
            record["chemical_match"] = (
                actual_smiles == normalized_smiles(expected_mol)
                and actual_inchikey == Chem.MolToInchiKey(expected_mol)
            )
            record["actual_smiles"] = actual_smiles
            record["actual_inchikey"] = actual_inchikey
        except Exception as error:
            record["error"] = str(error)
        record["seconds"] = time.perf_counter() - started
        records.append(record)
        completed_keys.add(key)
        with records_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if not record["chemical_match"] and overlays < args.failure_overlays:
            failure_dir = output_dir / "failures"
            failure_dir.mkdir(parents=True, exist_ok=True)
            draw_overlay(image, expected, predicted).save(failure_dir / f"{key}.png")
            (failure_dir / f"{key}.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            overlays += 1
        if len(records) % 10 == 0 or len(records) == args.samples:
            summary = summarize(records)
            (output_dir / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(summary, ensure_ascii=False), flush=True)
    summary = summarize(records)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
