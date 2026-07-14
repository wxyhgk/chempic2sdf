"""Pure graph comparison, visualization, and aggregate evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment

from Data.sdf.chemsvg_sequence_v4 import ParsedChemSVGSequenceV4


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


def compare_graphs(
    expected: ParsedChemSVGSequenceV4,
    predicted: ParsedChemSVGSequenceV4,
    *,
    tolerance_pixels: float = 6.0,
    image_size: int = 768,
) -> GraphComparison:
    expected_points = np.array(
        [(vertex.x, vertex.y) for vertex in expected.vertices],
        dtype=np.float64,
    )
    predicted_points = np.array(
        [(vertex.x, vertex.y) for vertex in predicted.vertices],
        dtype=np.float64,
    )
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

    expected_coordinates = {
        (vertex.x, vertex.y): vertex for vertex in expected.vertices
    }
    for stroke in expected.strokes:
        draw.line(
            (
                *point(expected_coordinates[stroke.begin]),
                *point(expected_coordinates[stroke.end]),
            ),
            fill=(0, 180, 0),
            width=2,
        )
    if predicted is not None:
        predicted_coordinates = {
            (vertex.x, vertex.y): vertex for vertex in predicted.vertices
        }
        for stroke in predicted.strokes:
            draw.line(
                (
                    *point(predicted_coordinates[stroke.begin]),
                    *point(predicted_coordinates[stroke.end]),
                ),
                fill=(220, 0, 0),
                width=1,
            )
        for vertex in predicted.vertices:
            x, y = point(vertex)
            draw.ellipse(
                (x - 3, y - 3, x + 3, y + 3),
                outline=(220, 0, 0),
                width=1,
            )
    return overlay


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / max(float(denominator), 1.0)


def summarize(records: list[dict]) -> dict:
    totals = {
        name: sum(record.get(name, 0) for record in records)
        for name in (
            "expected_vertices",
            "predicted_vertices",
            "matched_vertices",
            "correct_elements",
            "expected_edges",
            "predicted_edges",
            "correct_edges",
            "correct_bond_types",
        )
    }
    repaired = [record for record in records if record.get("repair_success")]
    repaired_totals = {
        name: sum(record.get(name, 0) for record in repaired)
        for name in (
            "expected_vertices",
            "predicted_vertices",
            "matched_vertices",
            "expected_edges",
            "predicted_edges",
            "correct_edges",
        )
    }
    failure_categories = dict.fromkeys(
        ("disconnected", "valence", "kekulize", "other"),
        0,
    )
    for record in records:
        if record.get("repair_success"):
            continue
        error = record.get("error", "")
        if "not connected" in error:
            category = "disconnected"
        elif "Explicit valence" in error:
            category = "valence"
        elif "kekulize" in error.lower() or "non-ring atom" in error:
            category = "kekulize"
        else:
            category = "other"
        failure_categories[category] += 1
    molecules: dict[str, list[dict]] = {}
    layouts: dict[str, list[dict]] = {}
    for record in records:
        molecules.setdefault(record["key"].rsplit("_", 1)[0], []).append(record)
        layouts.setdefault(record["key"].rsplit("_", 1)[-1], []).append(record)
    exact = [record for record in records if record.get("exact_topology")]
    return {
        "samples": len(records),
        "molecules": len(molecules),
        "completed_generation_rate": _safe_ratio(
            sum(bool(record.get("generation_completed")) for record in records),
            len(records),
        ),
        "repair_success_rate": _safe_ratio(len(repaired), len(records)),
        "sdf_valid_rate": _safe_ratio(
            sum(bool(record.get("sdf_valid")) for record in records),
            len(records),
        ),
        "chemical_match_rate": _safe_ratio(
            sum(bool(record.get("chemical_match")) for record in records),
            len(records),
        ),
        "exact_topology_rate": _safe_ratio(len(exact), len(records)),
        "chemical_match_given_exact_topology": _safe_ratio(
            sum(bool(record.get("chemical_match")) for record in exact),
            len(exact),
        ),
        "vertex_precision": _safe_ratio(
            totals["matched_vertices"], totals["predicted_vertices"]
        ),
        "vertex_recall": _safe_ratio(
            totals["matched_vertices"], totals["expected_vertices"]
        ),
        "element_accuracy_on_matched": _safe_ratio(
            totals["correct_elements"], totals["matched_vertices"]
        ),
        "edge_precision": _safe_ratio(
            totals["correct_edges"], totals["predicted_edges"]
        ),
        "edge_recall": _safe_ratio(
            totals["correct_edges"], totals["expected_edges"]
        ),
        "bond_type_accuracy_on_correct_edges": _safe_ratio(
            totals["correct_bond_types"], totals["correct_edges"]
        ),
        "average_seconds": _safe_ratio(
            sum(record.get("seconds", 0.0) for record in records),
            len(records),
        ),
        "conditional_vertex_precision": _safe_ratio(
            repaired_totals["matched_vertices"],
            repaired_totals["predicted_vertices"],
        ),
        "conditional_vertex_recall": _safe_ratio(
            repaired_totals["matched_vertices"],
            repaired_totals["expected_vertices"],
        ),
        "conditional_edge_precision": _safe_ratio(
            repaired_totals["correct_edges"],
            repaired_totals["predicted_edges"],
        ),
        "conditional_edge_recall": _safe_ratio(
            repaired_totals["correct_edges"],
            repaired_totals["expected_edges"],
        ),
        "molecule_any_repair_rate": _safe_ratio(
            sum(any(item.get("repair_success") for item in group) for group in molecules.values()),
            len(molecules),
        ),
        "molecule_any_chemical_match_rate": _safe_ratio(
            sum(any(item.get("chemical_match") for item in group) for group in molecules.values()),
            len(molecules),
        ),
        "failure_categories": failure_categories,
        "layout_rates": {
            layout: {
                "samples": len(group),
                "repair_success_rate": _safe_ratio(
                    sum(bool(item.get("repair_success")) for item in group),
                    len(group),
                ),
                "exact_topology_rate": _safe_ratio(
                    sum(bool(item.get("exact_topology")) for item in group),
                    len(group),
                ),
                "chemical_match_rate": _safe_ratio(
                    sum(bool(item.get("chemical_match")) for item in group),
                    len(group),
                ),
            }
            for layout, group in sorted(layouts.items())
        },
    }
