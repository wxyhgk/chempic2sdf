"""Build the annotation-free R0 observation from the frozen renderer fixture."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from formal.chemgraph.audit.fragment_root_constructor_v0_4.fragment_root import (
    SCHEMA_VERSION as GEOMETRY_SCHEMA,
)
from formal.chemgraph.audit.fragment_support_partition_v0_4.support_partition import (
    SCHEMA_VERSION as SUPPORT_SCHEMA,
    _rle_sha256,
)
from formal.chemgraph.challenge_v0_4.rasterizer_v0_4 import (
    Q,
    LineStyle,
    Segment,
    rasterize_gray,
    rasterize_support,
)

from .common import canonical_bytes, encode_runs, fail, sha256


ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
RENDERER_PATH = REPOSITORY_ROOT / "formal/chemgraph/challenge_v0_4/rasterizer_v0_4.py"
PNG_PATH = (
    REPOSITORY_ROOT
    / "formal/chemgraph/audit/png_ownership_realization_v0_5/representative_edge_w5_round_box4.png"
)
EXPECTED_PNG_SHA256 = "0152efc7791f03515ef554739b545088b36d654aa081da9718d0c31eef3ecb4e"
EXPECTED_RENDERER_SHA256 = "5d673ae1c33648b5498b4ebacfc1b91136fee5abdbae4fa1503a144a59816943"
LOWER_HEX = frozenset("0123456789abcdef")


def _is_sha256_hex(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in LOWER_HEX for character in value)
    )


def _fixture() -> tuple[list[Segment], LineStyle]:
    return [
        Segment("H", 10 * Q, 20 * Q, 30 * Q, 20 * Q),
        Segment("V1", 18 * Q, 10 * Q, 18 * Q, 20 * Q),
        Segment("V2", 22 * Q, 20 * Q, 22 * Q, 30 * Q),
    ], LineStyle(width_q16=5 * Q, cap="round", antialias_profile="box_4x4")


def build_observation() -> dict[str, Any]:
    renderer_digest = sha256(RENDERER_PATH.read_bytes())
    png_digest = sha256(PNG_PATH.read_bytes())
    if renderer_digest != EXPECTED_RENDERER_SHA256:
        fail("RENDERER_IDENTITY_MISMATCH")
    if png_digest != EXPECTED_PNG_SHA256:
        fail("PNG_IDENTITY_MISMATCH")

    segments, style = _fixture()
    supports = [rasterize_support([segment], style) for segment in segments]
    gray = rasterize_gray(segments, style)
    foreground = {index for index, value in enumerate(gray) if value != 255}
    if set().union(*supports) != foreground:
        fail("SUPPORT_UNION_GRAY_MISMATCH")

    q4_segments = [
        [[10 * 16, 20 * 16], [30 * 16, 20 * 16]],
        [[18 * 16, 10 * 16], [18 * 16, 20 * 16]],
        [[22 * 16, 20 * 16], [22 * 16, 30 * 16]],
    ]
    geometry_root = {
        "base_strokes": [
            {"boundary_member_ids": [index], "id": index, "segment_q4": segment}
            for index, segment in enumerate(q4_segments)
        ],
        "schema_version": GEOMETRY_SCHEMA,
        "training_authorized": False,
    }
    parent_supports = []
    for index, support in enumerate(supports):
        pixels = {divmod(pixel, 768) for pixel in support}
        runs = encode_runs(pixels)
        parent_supports.append({
            "base_stroke_id": index,
            "support_pixel_count": len(pixels),
            "support_rle_sha256": _rle_sha256(runs),
            "support_runs": runs,
        })
    support_input = {
        "geometry_root": geometry_root,
        "parent_supports": parent_supports,
        "schema_version": SUPPORT_SCHEMA,
        "training_authorized": False,
    }
    return {
        "geometry_root": geometry_root,
        "parent_supports": parent_supports,
        "png_sha256": png_digest,
        "renderer_sha256": renderer_digest,
        "schema_version": "owner-candidate-r0-observation-v0.5-research",
        "support_input_sha256": sha256(canonical_bytes(support_input)),
        "training_authorized": False,
    }


def support_input_from_observation(observation: Any) -> dict[str, Any]:
    if not isinstance(observation, dict) or set(observation) != {
        "geometry_root", "parent_supports", "png_sha256", "renderer_sha256",
        "schema_version", "support_input_sha256", "training_authorized"
    }:
        fail("OBSERVATION_NOT_CLOSED")
    if observation["schema_version"] != "owner-candidate-r0-observation-v0.5-research":
        fail("OBSERVATION_SCHEMA_MISMATCH")
    if observation["training_authorized"] is not False:
        fail("TRAINING_AUTHORIZED")
    if not _is_sha256_hex(observation["png_sha256"]):
        fail("PNG_SHA256_INVALID")
    if not _is_sha256_hex(observation["renderer_sha256"]):
        fail("RENDERER_SHA256_INVALID")
    support_input = {
        "geometry_root": observation["geometry_root"],
        "parent_supports": observation["parent_supports"],
        "schema_version": SUPPORT_SCHEMA,
        "training_authorized": False,
    }
    if observation["support_input_sha256"] != sha256(canonical_bytes(support_input)):
        fail("SUPPORT_INPUT_HASH_MISMATCH")
    return support_input
