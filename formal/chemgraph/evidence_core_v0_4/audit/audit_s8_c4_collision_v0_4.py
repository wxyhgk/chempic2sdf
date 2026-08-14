#!/usr/bin/env python3
"""Audit an exact S8/C4 raster collision in the public v0.4 grammar.

This audit may import the frozen generator/rasterizer.  It is not part of the
extractor or candidate dependency closure and does not use a hidden seed.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from formal.chemgraph.challenge_v0_4.generate_pack_v0_4 import (  # noqa: E402
    Transform,
    px,
    separated_support_certificate,
)
from formal.chemgraph.challenge_v0_4.rasterizer_v0_4 import (  # noqa: E402
    Q,
    LineStyle,
    Segment,
    rasterize_gray,
    rasterize_png,
)


SCHEMA_VERSION = "chemgraph-s8-c4-collision-audit-v0.4"
CANVAS_TRANSLATION = 384
S8_HALF = 62
S8_GAP = 15
C4_HALF = 93


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")


def _segments(
    raw: tuple[tuple[tuple[int, int], tuple[int, int]], ...],
    transform: Transform,
    prefix: str,
) -> list[Segment]:
    result = []
    for index, (begin, end) in enumerate(raw):
        transformed_begin = transform.point(px(*begin))
        transformed_end = transform.point(px(*end))
        result.append(
            Segment(
                f"{prefix}-{index}",
                transformed_begin[0],
                transformed_begin[1],
                transformed_end[0],
                transformed_end[1],
            )
        )
    return result


def collision_segments() -> tuple[list[Segment], list[Segment]]:
    s8 = _segments(
        (
            ((-S8_HALF, 0), (-S8_GAP, 0)),
            ((S8_GAP, -S8_HALF), (S8_GAP, S8_HALF)),
        ),
        Transform(
            "none", (1, 1), (0, 1), (3, 4),
            CANVAS_TRANSLATION * Q, CANVAS_TRANSLATION * Q,
        ),
        "s8",
    )
    c4 = _segments(
        (
            ((-C4_HALF, -80), (C4_HALF, -80)),
            ((0, -35), (0, 35)),
        ),
        Transform(
            "none", (0, 1), (1, 1), (1, 2),
            (CANVAS_TRANSLATION - 29) * Q, CANVAS_TRANSLATION * Q,
        ),
        "c4",
    )
    return s8, c4


def collision_styles() -> tuple[LineStyle, ...]:
    return tuple(
        LineStyle(width * Q, cap, "binary_1x", "round")
        for width in (2, 4, 6, 8)
        for cap in ("butt", "square")
    )


def _coordinates(segments: list[Segment]) -> list[list[list[int]]]:
    return [
        [[segment.x0_q16, segment.y0_q16], [segment.x1_q16, segment.y1_q16]]
        for segment in segments
    ]


def _canonical_graph(
    labels: tuple[str, ...], edges: tuple[tuple[int, int, int], ...]
) -> str:
    """Canonicalize the tiny anonymous graph independently of generator IDs."""
    encodings = []
    for permutation in itertools.permutations(range(len(labels))):
        inverse = {old: new for new, old in enumerate(permutation)}
        ordered_labels = tuple(labels[old] for old in permutation)
        ordered_edges = tuple(
            sorted(
                (min(inverse[left], inverse[right]),
                 max(inverse[left], inverse[right]), order)
                for left, right, order in edges
            )
        )
        encodings.append((ordered_labels, ordered_edges))
    labels_value, edges_value = min(encodings)
    return json.dumps(
        {"edges": edges_value, "labels": labels_value},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def semantic_graph_sets() -> tuple[frozenset[str], frozenset[str]]:
    two_lanes = _canonical_graph(
        ("C", "C", "C", "C"), ((0, 1, 1), (2, 3, 1))
    )
    text = _canonical_graph(
        ("C", "C", "TEXT_STROKE"), ((0, 1, 1),)
    )
    return frozenset((two_lanes,)), frozenset((two_lanes, text))


def audit_collision() -> dict[str, Any]:
    s8, c4 = collision_segments()
    s8_graphs, c4_graphs = semantic_graph_sets()
    style_results = []
    for style in collision_styles():
        s8_gray = rasterize_gray(s8, style)
        c4_gray = rasterize_gray(c4, style)
        s8_png, s8_gray_sha = rasterize_png(s8, style)
        c4_png, c4_gray_sha = rasterize_png(c4, style)
        s8_support, _ = separated_support_certificate(s8, style)
        c4_support, _ = separated_support_certificate(c4, style)
        style_results.append(
            {
                "antialias_profile": style.antialias_profile,
                "c4_support_gate_passed": c4_support,
                "cap": style.cap,
                "gray_bytes_equal": s8_gray == c4_gray,
                "gray_sha256": s8_gray_sha,
                "later_c4_hits_seen_png": hashlib.sha256(c4_png).digest()
                in {hashlib.sha256(s8_png).digest()},
                "png_bytes_equal": s8_png == c4_png,
                "png_sha256": hashlib.sha256(s8_png).hexdigest(),
                "s8_support_gate_passed": s8_support,
                "width_pixels": style.width_q16 // Q,
            }
        )
    all_exact = all(
        row["gray_bytes_equal"]
        and row["png_bytes_equal"]
        and row["s8_support_gate_passed"]
        and row["c4_support_gate_passed"]
        for row in style_results
    )
    duplicate_gate_would_retry = all(
        row["later_c4_hits_seen_png"] for row in style_results
    )
    shared_graphs = sorted(s8_graphs & c4_graphs)
    semantic_conflict = (
        len(s8_graphs) == 1
        and len(c4_graphs) == 2
        and len(shared_graphs) == 1
        and s8_graphs != c4_graphs
    )
    return {
        "collision": {
            "c4_coordinates_q16": _coordinates(c4),
            "c4_half": C4_HALF,
            "c4_rotation": [[0, 1], [1, 1]],
            "c4_scale": [1, 2],
            "c4_translation_pixels": [CANVAS_TRANSLATION - 29, CANVAS_TRANSLATION],
            "kind": "exact_gray_and_canonical_png_bytes",
            "s8_coordinates_q16": _coordinates(s8),
            "s8_gap": S8_GAP,
            "s8_half": S8_HALF,
            "s8_rotation": [[1, 1], [0, 1]],
            "s8_scale": [3, 4],
            "s8_translation_pixels": [CANVAS_TRANSLATION, CANVAS_TRANSLATION],
            "style_results": style_results,
        },
        "conclusion": {
            "accepted_same_pack_pair": not duplicate_gate_would_retry,
            "evidence_collision_for_deterministic_png_extractor": all_exact,
            "generator_duplicate_observation_gate_would_retry_later_c4":
                duplicate_gate_would_retry,
            "is_only_near_neighbor": False,
            "semantic_demand_conflict": semantic_conflict,
        },
        "scope": {
            "formal_hidden_seed_used": False,
            "imports_frozen_generator_and_rasterizer_for_audit_only": True,
            "lean_proves_png_equality": False,
            "parameter_space_witness_only": True,
            "prng_joint_reachability_verified": False,
            "training_authorized": False,
        },
        "semantics": {
            "c4_decode_requirement": "Ambiguous(text_or_lane)",
            "c4_graph_set": sorted(c4_graphs),
            "s8_decode_requirement": "Graph(two_disconnected_single_bonds)",
            "s8_graph_set": sorted(s8_graphs),
            "shared_graphs": shared_graphs,
        },
        "schema_version": SCHEMA_VERSION,
        "status": "no_go_cross_family_exact_png_collision" if all_exact and semantic_conflict else "failed",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_collision()
    encoded = canonical_bytes(result)
    if args.output is not None:
        args.output.write_bytes(encoded)
    sys.stdout.buffer.write(encoded)
    return 0 if result["status"] == "no_go_cross_family_exact_png_collision" else 1


if __name__ == "__main__":
    raise SystemExit(main())
