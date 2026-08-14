"""Finite-domain candidate enumeration from complete neutral foreground geometry.

This is deliberately separate from the PNG extractor.  It forward-renders the
entire public v0.1 finite grammar at runtime and compares complete foreground
row-runs.  It never selects candidates by an observation digest.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Iterable

from PIL import Image, ImageDraw

from .finite_geometry_grammar_v0_1 import DOMAIN_LAYERS, classify_geometry
from .finite_micro_scene_v0_1 import (
    CANVAS_SIZE,
    Scene,
    canonical_graph_signature,
    fixture_scenes,
    iter_core_scenes,
)


SCHEMA_VERSION = "raster-candidate-enumerator-v0.3"
LINE_WIDTH = 3


class CandidateEnumerationError(ValueError):
    """The evidence is malformed or outside the enumerated finite domain."""


def render_public_scene(scene: Scene) -> Image.Image:
    """Frozen v0.3 public raster profile, independent of production preprocessing."""
    image = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), (255, 255, 255, 255))
    drawing = ImageDraw.Draw(image)
    for bond in scene.bonds:
        for member in bond.members:
            drawing.line(
                (*member.begin, *member.end), fill=(0, 0, 0, 255), width=LINE_WIDTH
            )
    return image


@dataclass(frozen=True, slots=True)
class ForwardCandidateIndex:
    domain: str
    scene_count: int
    observation_count: int
    candidates_by_foreground: dict[tuple[tuple[int, int, int], ...], frozenset[str]]
    png_by_foreground: dict[tuple[tuple[int, int, int], ...], bytes]
    scene_ids_by_foreground: dict[tuple[tuple[int, int, int], ...], tuple[str, ...]]


def foreground_key_from_image(image) -> tuple[tuple[int, int, int], ...]:
    """Independently derive black-foreground runs from a rendered RGBA image."""
    rgba = image.convert("RGBA")
    rows: list[tuple[int, int, int]] = []
    pixels = rgba.load()
    for y in range(rgba.height):
        active: list[int] = []
        for x in range(rgba.width):
            red, green, blue, alpha = pixels[x, y]
            red = (red * alpha + 255 * (255 - alpha) + 127) // 255
            green = (green * alpha + 255 * (255 - alpha) + 127) // 255
            blue = (blue * alpha + 255 * (255 - alpha) + 127) // 255
            luma = (77 * red + 150 * green + 29 * blue + 128) // 256
            if luma < 128:
                active.append(x)
        if not active:
            continue
        begin = previous = active[0]
        for x in active[1:]:
            if x != previous + 1:
                rows.append((y, begin, previous))
                begin = x
            previous = x
        rows.append((y, begin, previous))
    return tuple(rows)


def foreground_key_from_evidence(evidence: dict[str, Any]) -> tuple[tuple[int, int, int], ...]:
    if evidence.get("schema_version") != "raster-neutral-evidence-v0.3":
        raise CandidateEnumerationError("unexpected_evidence_schema")
    if evidence.get("status") != "ok":
        raise CandidateEnumerationError("evidence_not_ok")
    foreground = evidence.get("foreground")
    if not isinstance(foreground, dict) or not isinstance(foreground.get("row_runs"), list):
        raise CandidateEnumerationError("missing_foreground_row_runs")
    result: list[tuple[int, int, int]] = []
    previous: tuple[int, int, int] | None = None
    for raw in foreground["row_runs"]:
        if (
            not isinstance(raw, list)
            or len(raw) != 3
            or any(not isinstance(value, int) for value in raw)
        ):
            raise CandidateEnumerationError("invalid_foreground_row_run")
        row = tuple(raw)
        if row[1] > row[2] or (previous is not None and row <= previous):
            raise CandidateEnumerationError("noncanonical_foreground_row_runs")
        result.append(row)
        previous = row
    return tuple(result)


def iter_domain_scenes(domain: str) -> Iterable[Scene]:
    layers = DOMAIN_LAYERS.get(domain)
    if layers is None:
        raise CandidateEnumerationError(f"unsupported_domain:{domain}")
    for scene in iter_core_scenes():
        if classify_geometry(scene).layer in layers:
            yield scene
    for scene in fixture_scenes():
        if classify_geometry(scene).layer in layers:
            yield scene


def build_forward_candidate_index(domain: str) -> ForwardCandidateIndex:
    graph_sets: dict[tuple[tuple[int, int, int], ...], set[str]] = defaultdict(set)
    png_by_foreground: dict[tuple[tuple[int, int, int], ...], bytes] = {}
    scene_ids: dict[tuple[tuple[int, int, int], ...], list[str]] = defaultdict(list)
    scene_count = 0
    for scene in iter_domain_scenes(domain):
        scene_count += 1
        image = render_public_scene(scene)
        key = foreground_key_from_image(image)
        graph_sets[key].add(canonical_graph_signature(scene))
        scene_ids[key].append(scene.scene_id)
        if key not in png_by_foreground:
            output = BytesIO()
            image.save(output, format="PNG", optimize=False)
            png_by_foreground[key] = output.getvalue()
    return ForwardCandidateIndex(
        domain=domain,
        scene_count=scene_count,
        observation_count=len(graph_sets),
        candidates_by_foreground={
            key: frozenset(values) for key, values in graph_sets.items()
        },
        png_by_foreground=png_by_foreground,
        scene_ids_by_foreground={
            key: tuple(sorted(values)) for key, values in scene_ids.items()
        },
    )


def enumerate_candidate_graphs(
    evidence: dict[str, Any], index: ForwardCandidateIndex
) -> frozenset[str]:
    return index.candidates_by_foreground.get(
        foreground_key_from_evidence(evidence), frozenset()
    )


__all__ = [
    "CandidateEnumerationError",
    "ForwardCandidateIndex",
    "SCHEMA_VERSION",
    "build_forward_candidate_index",
    "enumerate_candidate_graphs",
    "foreground_key_from_evidence",
    "foreground_key_from_image",
    "iter_domain_scenes",
    "render_public_scene",
]
