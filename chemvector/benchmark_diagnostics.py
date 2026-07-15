#!/usr/bin/env python3
"""Graph-level error diagnostics for ChemVector benchmark artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from Data.sdf.chemsvg_sequence_v4 import ParsedChemSVGSequenceV4, parse_chemsvg_v4


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _point_segment_distance(
    point: tuple[float, float],
    begin: tuple[float, float],
    end: tuple[float, float],
) -> float:
    point_array = np.asarray(point, dtype=np.float64)
    begin_array = np.asarray(begin, dtype=np.float64)
    delta = np.asarray(end, dtype=np.float64) - begin_array
    denominator = float(np.dot(delta, delta))
    if denominator == 0.0:
        return float(np.linalg.norm(point_array - begin_array))
    fraction = float(np.dot(point_array - begin_array, delta) / denominator)
    fraction = min(max(fraction, 0.0), 1.0)
    projection = begin_array + fraction * delta
    return float(np.linalg.norm(point_array - projection))


def _orientation(
    first: tuple[float, float],
    second: tuple[float, float],
    third: tuple[float, float],
) -> float:
    return (second[0] - first[0]) * (third[1] - first[1]) - (
        second[1] - first[1]
    ) * (third[0] - first[0])


def _segments_cross_interior(
    first_begin: tuple[float, float],
    first_end: tuple[float, float],
    second_begin: tuple[float, float],
    second_end: tuple[float, float],
    *,
    epsilon: float = 1e-6,
) -> bool:
    orientations = (
        _orientation(first_begin, first_end, second_begin),
        _orientation(first_begin, first_end, second_end),
        _orientation(second_begin, second_end, first_begin),
        _orientation(second_begin, second_end, first_end),
    )
    return (
        orientations[0] * orientations[1] < -epsilon
        and orientations[2] * orientations[3] < -epsilon
    )


def _predicted_points(
    parsed: ParsedChemSVGSequenceV4,
    width: int,
    height: int,
) -> np.ndarray:
    return np.asarray(
        [
            (vertex.x / 1023.0 * width, vertex.y / 1023.0 * height)
            for vertex in parsed.vertices
        ],
        dtype=np.float64,
    )


def _match_vertices(
    expected_points: np.ndarray,
    predicted_points: np.ndarray,
    tolerance_pixels: float,
) -> dict[int, int]:
    if not len(expected_points) or not len(predicted_points):
        return {}
    distances = np.linalg.norm(
        predicted_points[:, None, :] - expected_points[None, :, :],
        axis=-1,
    )
    predicted_indices, expected_indices = linear_sum_assignment(distances)
    return {
        int(predicted_index): int(expected_index)
        for predicted_index, expected_index in zip(predicted_indices, expected_indices)
        if distances[predicted_index, expected_index] <= tolerance_pixels
    }


def _primary_error(diagnostics: dict[str, Any]) -> str:
    if diagnostics["graph_exact"]:
        if diagnostics.get("chemical_match") is False:
            if diagnostics.get("nonstereo_structure_match"):
                return "stereo"
            if not diagnostics.get("sdf_valid"):
                return f"invalid_{diagnostics.get('failure_category') or 'other'}"
            return "other"
        return "exact"
    if diagnostics["missing_atoms"] or diagnostics["extra_atoms"]:
        return "atom_detection"
    if diagnostics["element_mismatches"]:
        return "ocr"
    if diagnostics["extra_bonds"] and diagnostics["missing_bonds"]:
        return "bond_rewire"
    if diagnostics["extra_bonds"]:
        if diagnostics["crossing_extra_bonds"] or diagnostics["extra_bonds_through_atom"]:
            return "crossing_extra_bond"
        return "extra_bond"
    if diagnostics["missing_bonds"]:
        return "missing_bond"
    if diagnostics["bond_type_mismatches"]:
        return "bond_type"
    if diagnostics["aromatic_mismatches"]:
        return "aromaticity"
    if diagnostics["atom_property_mismatches"]:
        return "atom_property"
    failure_category = diagnostics.get("failure_category")
    if failure_category:
        return f"invalid_{failure_category}"
    return "other"


def _bond_types_equivalent(
    expected_type: str,
    predicted_type: str,
    expected_begin: dict[str, Any],
    expected_end: dict[str, Any],
) -> bool:
    if expected_type == predicted_type:
        return True
    return (
        expected_type == "AROMATIC"
        and predicted_type in {"SINGLE", "DOUBLE"}
        and bool(expected_begin.get("aromatic"))
        and bool(expected_end.get("aromatic"))
    )


def diagnose_chemsvg(
    annotation: dict[str, Any],
    sequence: str,
    *,
    sdf_valid: bool,
    chemical_match: bool | None = None,
    nonstereo_structure_match: bool | None = None,
    failure_category: str | None = None,
    tolerance_pixels: float = 12.0,
    through_atom_tolerance_pixels: float = 5.0,
) -> dict[str, Any]:
    """Compare ChemSVG geometry with render annotation without using SMILES."""

    parsed = parse_chemsvg_v4(sequence)
    width = int(annotation["width"])
    height = int(annotation["height"])
    expected_atoms = sorted(annotation["atoms"], key=lambda atom: atom["atom_id"])
    expected_points = np.asarray(
        [(atom["image_x"], atom["image_y"]) for atom in expected_atoms],
        dtype=np.float64,
    )
    predicted_points = _predicted_points(parsed, width, height)
    predicted_to_expected = _match_vertices(
        expected_points,
        predicted_points,
        tolerance_pixels,
    )
    expected_to_predicted = {
        expected: predicted for predicted, expected in predicted_to_expected.items()
    }

    element_mismatches = 0
    charge_mismatches = 0
    isotope_mismatches = 0
    aromatic_mismatches = 0
    visible_element_mismatches = 0
    for predicted_index, expected_index in predicted_to_expected.items():
        predicted = parsed.vertices[predicted_index]
        expected = expected_atoms[expected_index]
        element_differs = predicted.element != expected["element"]
        element_mismatches += int(element_differs)
        charge_mismatches += int(predicted.formal_charge != expected["formal_charge"])
        isotope_mismatches += int(predicted.isotope != expected["isotope"])
        aromatic_mismatches += int(
            predicted.aromatic != bool(expected.get("aromatic", False))
        )
        visible_element_mismatches += int(
            element_differs
            and (predicted.element != "C" or expected["element"] != "C")
        )

    coordinate_to_predicted = {
        (vertex.x, vertex.y): index for index, vertex in enumerate(parsed.vertices)
    }
    predicted_edges: dict[tuple[int, int], str] = {}
    unmatched_predicted_edges = 0
    for stroke in parsed.strokes:
        predicted_begin = coordinate_to_predicted[stroke.begin]
        predicted_end = coordinate_to_predicted[stroke.end]
        if (
            predicted_begin not in predicted_to_expected
            or predicted_end not in predicted_to_expected
        ):
            unmatched_predicted_edges += 1
            continue
        pair = tuple(
            sorted(
                (
                    predicted_to_expected[predicted_begin],
                    predicted_to_expected[predicted_end],
                )
            )
        )
        predicted_edges[pair] = stroke.bond_type

    expected_edges = {
        tuple(sorted((int(bond["begin_atom"]), int(bond["end_atom"])))): bond[
            "bond_type"
        ]
        for bond in annotation["bonds"]
    }
    comparable_expected_edges = {
        pair: bond_type
        for pair, bond_type in expected_edges.items()
        if pair[0] in expected_to_predicted and pair[1] in expected_to_predicted
    }
    common_pairs = set(comparable_expected_edges) & set(predicted_edges)
    missing_pairs = set(comparable_expected_edges) - set(predicted_edges)
    extra_pairs = set(predicted_edges) - set(expected_edges)
    missing_edges = len(missing_pairs) + sum(
        pair[0] not in expected_to_predicted or pair[1] not in expected_to_predicted
        for pair in expected_edges
    )
    extra_edges = len(extra_pairs) + unmatched_predicted_edges
    bond_type_mismatches = sum(
        not _bond_types_equivalent(
            comparable_expected_edges[pair],
            predicted_edges[pair],
            expected_atoms[pair[0]],
            expected_atoms[pair[1]],
        )
        for pair in common_pairs
    )
    if chemical_match:
        # SDF round-trip equality is authoritative when aromatic/Kekule encodings differ.
        bond_type_mismatches = 0
        aromatic_mismatches = 0

    extra_bonds_through_atom = 0
    crossing_extra_pairs: set[tuple[int, int]] = set()
    expected_point_tuples = [tuple(point) for point in expected_points.tolist()]
    for pair in extra_pairs:
        begin = expected_point_tuples[pair[0]]
        end = expected_point_tuples[pair[1]]
        if any(
            atom_id not in pair
            and _point_segment_distance(point, begin, end)
            <= through_atom_tolerance_pixels
            for atom_id, point in enumerate(expected_point_tuples)
        ):
            extra_bonds_through_atom += 1
        for expected_pair in expected_edges:
            if set(pair) & set(expected_pair):
                continue
            if _segments_cross_interior(
                begin,
                end,
                expected_point_tuples[expected_pair[0]],
                expected_point_tuples[expected_pair[1]],
            ):
                crossing_extra_pairs.add(pair)
                break
    extra_pair_list = sorted(extra_pairs)
    for index, pair in enumerate(extra_pair_list):
        for other in extra_pair_list[index + 1 :]:
            if set(pair) & set(other):
                continue
            if _segments_cross_interior(
                expected_point_tuples[pair[0]],
                expected_point_tuples[pair[1]],
                expected_point_tuples[other[0]],
                expected_point_tuples[other[1]],
            ):
                crossing_extra_pairs.update((pair, other))

    atom_property_mismatches = charge_mismatches + isotope_mismatches
    graph_exact = (
        len(predicted_to_expected) == len(expected_atoms) == len(parsed.vertices)
        and element_mismatches == 0
        and atom_property_mismatches == 0
        and aromatic_mismatches == 0
        and missing_edges == 0
        and extra_edges == 0
        and bond_type_mismatches == 0
    )
    error_flags = []
    isolated_aromatic_mismatches = aromatic_mismatches * int(
        element_mismatches == 0
        and atom_property_mismatches == 0
        and missing_edges == 0
        and extra_edges == 0
        and bond_type_mismatches == 0
        and len(expected_atoms) == len(predicted_to_expected) == len(parsed.vertices)
    )
    for flag, value in (
        ("missing_atom", len(expected_atoms) - len(predicted_to_expected)),
        ("extra_atom", len(parsed.vertices) - len(predicted_to_expected)),
        ("ocr", element_mismatches),
        ("atom_property", atom_property_mismatches),
        ("aromaticity", isolated_aromatic_mismatches),
        ("missing_bond", missing_edges),
        ("extra_bond", extra_edges),
        ("crossing_extra_bond", len(crossing_extra_pairs)),
        ("extra_bond_through_atom", extra_bonds_through_atom),
        ("bond_type", bond_type_mismatches),
    ):
        if value:
            error_flags.append(flag)
    if not sdf_valid:
        error_flags.append(f"invalid_{failure_category or 'other'}")

    diagnostics = {
        "matched_atoms": len(predicted_to_expected),
        "missing_atoms": len(expected_atoms) - len(predicted_to_expected),
        "extra_atoms": len(parsed.vertices) - len(predicted_to_expected),
        "element_mismatches": element_mismatches,
        "visible_element_mismatches": visible_element_mismatches,
        "charge_mismatches": charge_mismatches,
        "isotope_mismatches": isotope_mismatches,
        "aromatic_mismatches": aromatic_mismatches,
        "isolated_aromatic_mismatches": isolated_aromatic_mismatches,
        "atom_property_mismatches": atom_property_mismatches,
        "missing_bonds": missing_edges,
        "extra_bonds": extra_edges,
        "crossing_extra_bonds": len(crossing_extra_pairs),
        "extra_bonds_through_atom": extra_bonds_through_atom,
        "bond_type_mismatches": bond_type_mismatches,
        "graph_exact": graph_exact,
        "sdf_valid": bool(sdf_valid),
        "chemical_match": chemical_match,
        "nonstereo_structure_match": nonstereo_structure_match,
        "failure_category": failure_category,
        "error_flags": error_flags,
        "coordinate_tolerance_pixels": tolerance_pixels,
    }
    diagnostics["primary_error"] = _primary_error(diagnostics)
    return diagnostics


def summarize_diagnostics(records: list[dict[str, Any]]) -> dict[str, Any]:
    diagnosed = [record for record in records if record.get("diagnostics")]
    incorrect = [record for record in diagnosed if not record.get("chemical_match")]
    primary = Counter(
        record["diagnostics"]["primary_error"] for record in incorrect
    )
    flags = Counter(
        flag
        for record in incorrect
        for flag in record["diagnostics"].get("error_flags", [])
    )
    totals = Counter()
    for record in incorrect:
        diagnostics = record["diagnostics"]
        for key in (
            "missing_atoms",
            "extra_atoms",
            "element_mismatches",
            "visible_element_mismatches",
            "isolated_aromatic_mismatches",
            "missing_bonds",
            "extra_bonds",
            "crossing_extra_bonds",
            "extra_bonds_through_atom",
            "bond_type_mismatches",
        ):
            totals[key] += int(diagnostics.get(key, 0))
    return {
        "diagnosed_samples": len(diagnosed),
        "incorrect_samples": len(incorrect),
        "primary_error_counts": dict(sorted(primary.items())),
        "error_flag_sample_counts": dict(sorted(flags.items())),
        "error_event_counts": dict(sorted(totals.items())),
    }


def _resolve_project_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def backfill_evaluation(output_dir: Path) -> dict[str, Any]:
    config = json.loads((output_dir / "benchmark_config.json").read_text(encoding="utf-8"))
    manifest_path = _resolve_project_path(config["benchmark_manifest"])
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        manifest = {row["molecule_id"]: row for row in csv.DictReader(handle)}
    records_path = output_dir / "records.jsonl"
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for record in records:
        molecule_id = record["molecule_id"]
        row = manifest[molecule_id]
        annotation = json.loads(
            (manifest_path.parent / row["annotation_file"]).read_text(encoding="utf-8")
        )
        repaired_path = output_dir / "repaired_chemsvg" / f"{molecule_id}.chemsvg.txt"
        raw_path = output_dir / "raw_chemsvg" / f"{molecule_id}.chemsvg.txt"
        sequence_path = repaired_path if repaired_path.is_file() else raw_path
        if not sequence_path.is_file():
            continue
        diagnostics = diagnose_chemsvg(
            annotation,
            sequence_path.read_text(encoding="utf-8"),
            sdf_valid=bool(record.get("sdf_valid")),
            chemical_match=bool(record.get("chemical_match")),
            nonstereo_structure_match=bool(
                record.get("nonstereo_structure_match")
            ),
            failure_category=record.get("failure_category"),
        )
        record["diagnostics"] = diagnostics
        record["primary_error"] = diagnostics["primary_error"]
        record["error_flags"] = diagnostics["error_flags"]

    temporary = records_path.with_suffix(".jsonl.tmp")
    temporary.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    temporary.replace(records_path)
    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["error_diagnostics"] = summarize_diagnostics(records)
    temporary_summary = summary_path.with_suffix(".json.tmp")
    temporary_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_summary.replace(summary_path)
    return summary["error_diagnostics"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluation_dir")
    args = parser.parse_args()
    result = backfill_evaluation(Path(args.evaluation_dir))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
