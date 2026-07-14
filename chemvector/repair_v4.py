#!/usr/bin/env python3
"""Image-grounded graph repair for detected ChemSVG v0.4 vertices."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from Data.sdf.chemsvg_sequence_v4 import (
    ParsedChemSVGSequenceV4,
    StrokeRecordV4,
    VertexRecordV4,
    mol_from_chemsvg_v4,
    parse_chemsvg_v4,
    serialize_parsed_chemsvg_v4,
)
from chemvector.generation_v4 import write_sdf_v4


Coordinate = tuple[int, int]
IndexPair = tuple[int, int]


@dataclass(frozen=True, slots=True)
class InkGraphRepairConfig:
    ink_threshold: float = 0.72
    endpoint_trim: float = 0.14
    maximum_mean_gap_pixels: float = 1.2
    maximum_gap_pixels: float = 4.0
    maximum_gap_fraction: float = 0.18
    gap_pixel_threshold: float = 2.5
    intermediate_vertex_radius: float = 12.0
    duplicate_label_radius: float = 16.0
    maximum_aromatic_cycle_size: int = 6
    aromatic_vote_fraction: float = 0.5
    aromatic_vote_source: str = "auto"


@dataclass(frozen=True, slots=True)
class InkGraphRepairResult:
    sequence: str
    removed_coordinates: tuple[Coordinate, ...]
    vertices: int
    strokes: int
    components: int
    aromatic_strokes: int


def _distance(left: VertexRecordV4, right: VertexRecordV4) -> float:
    return math.hypot(left.x - right.x, left.y - right.y)


def _duplicate_label_indices(
    vertices: tuple[VertexRecordV4, ...],
    radius: float,
) -> set[int]:
    """Remove later near-identical non-carbon peaks produced inside one glyph."""

    removed: set[int] = set()
    for right, vertex in enumerate(vertices):
        if vertex.element == "C":
            continue
        for left in range(right):
            other = vertices[left]
            if left in removed or other.element != vertex.element:
                continue
            if _distance(other, vertex) <= radius:
                removed.add(right)
                break
    return removed


def _ink_distance_field(image: Image.Image, threshold: float) -> np.ndarray:
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    return distance_transform_edt(gray > threshold)


def _line_gap_statistics(
    distance_field: np.ndarray,
    begin: VertexRecordV4,
    end: VertexRecordV4,
    config: InkGraphRepairConfig,
) -> tuple[float, float, float]:
    height, width = distance_field.shape
    length = _distance(begin, end)
    samples = max(32, int(length * 0.7))
    trim = min(samples // 3, int(samples * config.endpoint_trim))
    x = np.linspace(begin.x / 1023.0 * (width - 1), end.x / 1023.0 * (width - 1), samples)
    y = np.linspace(begin.y / 1023.0 * (height - 1), end.y / 1023.0 * (height - 1), samples)
    if trim:
        x = x[trim:-trim]
        y = y[trim:-trim]
    x_indices = np.rint(x).astype(np.int64).clip(0, width - 1)
    y_indices = np.rint(y).astype(np.int64).clip(0, height - 1)
    gaps = distance_field[y_indices, x_indices]
    return (
        float(gaps.mean()),
        float(gaps.max(initial=0.0)),
        float((gaps > config.gap_pixel_threshold).mean()),
    )


def _has_intermediate_vertex(
    vertices: tuple[VertexRecordV4, ...],
    retained: tuple[int, ...],
    begin_index: int,
    end_index: int,
    radius: float,
) -> bool:
    begin = np.array((vertices[begin_index].x, vertices[begin_index].y), dtype=np.float64)
    end = np.array((vertices[end_index].x, vertices[end_index].y), dtype=np.float64)
    delta = end - begin
    denominator = float(delta @ delta)
    for index in retained:
        if index in {begin_index, end_index}:
            continue
        point = np.array((vertices[index].x, vertices[index].y), dtype=np.float64)
        position = float((point - begin) @ delta / denominator)
        if not 0.12 < position < 0.88:
            continue
        perpendicular = np.linalg.norm(point - (begin + position * delta))
        if perpendicular < radius:
            return True
    return False


def _connected_components(adjacency: dict[int, set[int]]) -> int:
    remaining = set(adjacency)
    components = 0
    while remaining:
        components += 1
        pending = [remaining.pop()]
        while pending:
            node = pending.pop()
            discovered = adjacency[node] & remaining
            remaining.difference_update(discovered)
            pending.extend(discovered)
    return components


def _component_labels(adjacency: dict[int, set[int]]) -> dict[int, int]:
    labels = {}
    component = 0
    for start in adjacency:
        if start in labels:
            continue
        labels[start] = component
        pending = [start]
        while pending:
            node = pending.pop()
            for neighbor in adjacency[node]:
                if neighbor not in labels:
                    labels[neighbor] = component
                    pending.append(neighbor)
        component += 1
    return labels


def _shortest_path_without_edge(
    adjacency: dict[int, set[int]],
    begin: int,
    end: int,
    maximum_edges: int,
) -> tuple[int, ...] | None:
    queue = deque([(begin, (begin,))])
    visited = {begin}
    while queue:
        node, path = queue.popleft()
        if len(path) - 1 >= maximum_edges:
            continue
        for neighbor in adjacency[node]:
            if {node, neighbor} == {begin, end}:
                continue
            if neighbor == end:
                return (*path, end)
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, (*path, neighbor)))
    return None


def _aromatic_pairs(
    vertices: tuple[VertexRecordV4, ...],
    pairs: set[IndexPair],
    adjacency: dict[int, set[int]],
    original_strokes: dict[tuple[Coordinate, Coordinate], StrokeRecordV4],
    config: InkGraphRepairConfig,
) -> set[IndexPair]:
    aromatic: set[IndexPair] = set()
    for begin, end in pairs:
        path = _shortest_path_without_edge(
            adjacency,
            begin,
            end,
            config.maximum_aromatic_cycle_size - 1,
        )
        if path is None:
            continue
        cycle = set(path)
        if len(cycle) > config.maximum_aromatic_cycle_size:
            continue
        eligible = [
            index
            for index in cycle
            if vertices[index].element not in {"B", "F", "Cl", "Br", "I"}
            and len(adjacency[index]) <= 3
        ]
        if len(eligible) != len(cycle):
            continue
        vertex_votes = sum(vertices[index].aromatic for index in cycle)
        cycle_pairs = {
            tuple(sorted(pair))
            for pair in zip(path, path[1:])
        }
        cycle_pairs.add(tuple(sorted((begin, end))))
        covered_strokes = 0
        stroke_votes = 0
        for left, right in cycle_pairs:
            coordinate_pair = tuple(
                sorted(
                    (
                        (vertices[left].x, vertices[left].y),
                        (vertices[right].x, vertices[right].y),
                    )
                )
            )
            stroke = original_strokes.get(coordinate_pair)
            covered_strokes += stroke is not None
            stroke_votes += bool(stroke is not None and stroke.bond_type == "AROMATIC")
        coverage = covered_strokes / max(len(cycle_pairs), 1)
        if config.aromatic_vote_source == "vertices":
            votes = vertex_votes / len(cycle)
        elif config.aromatic_vote_source == "strokes":
            votes = stroke_votes / max(len(cycle_pairs), 1)
        elif config.aromatic_vote_source == "auto":
            votes = (
                stroke_votes / max(len(cycle_pairs), 1)
                if coverage >= 0.5
                else vertex_votes / len(cycle)
            )
        else:
            raise ValueError(
                f"Unknown aromatic_vote_source: {config.aromatic_vote_source!r}"
            )
        if votes >= config.aromatic_vote_fraction:
            aromatic.update(cycle_pairs)
    return aromatic


def repair_chemsvg_v4_from_image(
    sequence: str,
    image: Image.Image,
    *,
    element_overrides: dict[Coordinate, str] | None = None,
    config: InkGraphRepairConfig | None = None,
) -> InkGraphRepairResult:
    """Rebuild strokes from visible continuous ink without a bond-length cutoff."""

    config = config or InkGraphRepairConfig()
    parsed = parse_chemsvg_v4(sequence)
    source_vertices = tuple(
        replace(vertex, element=element_overrides[(vertex.x, vertex.y)])
        if element_overrides and (vertex.x, vertex.y) in element_overrides
        else vertex
        for vertex in parsed.vertices
    )
    removed = _duplicate_label_indices(source_vertices, config.duplicate_label_radius)
    retained_source = tuple(
        index for index in range(len(source_vertices)) if index not in removed
    )
    source_to_repaired = {
        source_index: repaired_index
        for repaired_index, source_index in enumerate(retained_source)
    }
    retained_vertices = tuple(source_vertices[index] for index in retained_source)
    distance_field = _ink_distance_field(image, config.ink_threshold)

    source_pairs: set[IndexPair] = set()
    for offset, begin in enumerate(retained_source):
        for end in retained_source[offset + 1 :]:
            mean_gap, maximum_gap, gap_fraction = _line_gap_statistics(
                distance_field,
                source_vertices[begin],
                source_vertices[end],
                config,
            )
            if (
                mean_gap <= config.maximum_mean_gap_pixels
                and maximum_gap <= config.maximum_gap_pixels
                and gap_fraction <= config.maximum_gap_fraction
                and not _has_intermediate_vertex(
                    source_vertices,
                    retained_source,
                    begin,
                    end,
                    config.intermediate_vertex_radius,
                )
            ):
                source_pairs.add((begin, end))

    pairs = {
        tuple(sorted((source_to_repaired[begin], source_to_repaired[end])))
        for begin, end in source_pairs
    }
    adjacency = {index: set() for index in range(len(retained_vertices))}
    for begin, end in pairs:
        adjacency[begin].add(end)
        adjacency[end].add(begin)
    original_strokes = {
        tuple(sorted((stroke.begin, stroke.end))): stroke for stroke in parsed.strokes
    }
    components = _connected_components(adjacency)
    if components != 1:
        coordinate_to_repaired = {
            (vertex.x, vertex.y): index
            for index, vertex in enumerate(retained_vertices)
        }
        bridge_candidates = []
        for coordinate_pair, stroke in original_strokes.items():
            if not all(coordinate in coordinate_to_repaired for coordinate in coordinate_pair):
                continue
            pair = tuple(
                sorted(coordinate_to_repaired[coordinate] for coordinate in coordinate_pair)
            )
            if pair in pairs:
                continue
            bridge_candidates.append(pair)
        for begin, end in bridge_candidates:
            labels = _component_labels(adjacency)
            if labels[begin] == labels[end]:
                continue
            pairs.add((begin, end))
            adjacency[begin].add(end)
            adjacency[end].add(begin)
            if _connected_components(adjacency) == 1:
                break
        components = _connected_components(adjacency)
    if components != 1:
        raise ValueError(
            f"Ink graph is not connected: {components} components for "
            f"{len(retained_vertices)} vertices"
        )

    aromatic_pairs = _aromatic_pairs(
        retained_vertices,
        pairs,
        adjacency,
        original_strokes,
        config,
    )
    aromatic_vertices = {index for pair in aromatic_pairs for index in pair}
    repaired_vertices = tuple(
        replace(vertex, aromatic=index in aromatic_vertices)
        for index, vertex in enumerate(retained_vertices)
    )
    strokes = []
    for begin, end in sorted(pairs):
        left = repaired_vertices[begin]
        right = repaired_vertices[end]
        coordinate_pair = tuple(sorted(((left.x, left.y), (right.x, right.y))))
        original = original_strokes.get(coordinate_pair)
        if (begin, end) in aromatic_pairs:
            display_type = bond_type = "AROMATIC"
            direction = "NONE"
            stereo = "STEREONONE"
            stereo_coordinates = (-1, -1, -1, -1)
        elif original is not None and original.bond_type in {"DOUBLE", "TRIPLE"}:
            display_type = original.display_type
            bond_type = original.bond_type
            direction = original.direction
            stereo = original.stereo
            stereo_coordinates = (
                original.stereo_atom_0_x,
                original.stereo_atom_0_y,
                original.stereo_atom_1_x,
                original.stereo_atom_1_y,
            )
            retained_coordinates = {(vertex.x, vertex.y) for vertex in repaired_vertices}
            if original.stereo_atom_0 is not None and not {
                original.stereo_atom_0,
                original.stereo_atom_1,
            } <= retained_coordinates:
                stereo = "STEREONONE"
                stereo_coordinates = (-1, -1, -1, -1)
        else:
            display_type = bond_type = "SINGLE"
            direction = original.direction if original is not None else "NONE"
            stereo = "STEREONONE"
            stereo_coordinates = (-1, -1, -1, -1)
        strokes.append(
            StrokeRecordV4(
                begin_x=left.x,
                begin_y=left.y,
                end_x=right.x,
                end_y=right.y,
                display_type=display_type,
                bond_type=bond_type,
                direction=direction,
                stereo=stereo,
                stereo_atom_0_x=stereo_coordinates[0],
                stereo_atom_0_y=stereo_coordinates[1],
                stereo_atom_1_x=stereo_coordinates[2],
                stereo_atom_1_y=stereo_coordinates[3],
            )
        )

    repaired = serialize_parsed_chemsvg_v4(
        ParsedChemSVGSequenceV4(repaired_vertices, tuple(strokes))
    )
    mol_from_chemsvg_v4(repaired)
    return InkGraphRepairResult(
        sequence=repaired,
        removed_coordinates=tuple(
            (source_vertices[index].x, source_vertices[index].y)
            for index in sorted(removed)
        ),
        vertices=len(repaired_vertices),
        strokes=len(strokes),
        components=components,
        aromatic_strokes=len(aromatic_pairs),
    )


def _parse_element_override(value: str) -> tuple[Coordinate, str]:
    try:
        coordinate, element = value.split("=", 1)
        x, y = (int(item) for item in coordinate.split(",", 1))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected X,Y=ELEMENT") from error
    return (x, y), element


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-sequence", required=True)
    parser.add_argument("--output-sdf", required=True)
    parser.add_argument("--metrics")
    parser.add_argument(
        "--element-override",
        action="append",
        default=[],
        type=_parse_element_override,
        metavar="X,Y=ELEMENT",
    )
    args = parser.parse_args()
    result = repair_chemsvg_v4_from_image(
        Path(args.sequence).read_text(encoding="utf-8"),
        Image.open(args.image),
        element_overrides=dict(args.element_override),
    )
    output_sequence = Path(args.output_sequence)
    output_sequence.parent.mkdir(parents=True, exist_ok=True)
    output_sequence.write_text(result.sequence, encoding="utf-8")
    write_sdf_v4(result.sequence, args.output_sdf)
    metrics = {
        "vertices": result.vertices,
        "strokes": result.strokes,
        "components": result.components,
        "aromatic_strokes": result.aromatic_strokes,
        "removed_coordinates": result.removed_coordinates,
    }
    if args.metrics:
        Path(args.metrics).write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
