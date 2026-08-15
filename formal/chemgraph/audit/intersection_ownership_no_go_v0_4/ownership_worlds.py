"""Executable same-observation/different-ownership witness."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from fractions import Fraction
from typing import Any

from formal.chemgraph.audit.fragment_root_constructor_v0_4.fragment_root import (
    SCHEMA_VERSION as GEOMETRY_SCHEMA,
    compile_fragment_root,
)
from formal.chemgraph.audit.fragment_support_partition_v0_4.support_partition import (
    SCHEMA_VERSION as SUPPORT_SCHEMA,
    _encode_runs,
    _rle_sha256,
    compile_support_partition,
)


class OwnershipWorldError(ValueError):
    """Closed validation failure for an ownership witness."""


FROZEN_ARTIFACT_SHA256 = {
    "support_input": "a51d941903447556a4b1168947a5935f45b0c3c76e71f427640bb38d1f3d23d2",
    "observation": "4a1d9a42f6be3f78b537fdfbc777e5cb6f1b7a63ff153bcfece9796a628e7a25",
    "world_a": "dfb5ea46278e3b0792a5af51e066a2ce22847d28727661da1b29da1ef9e8b083",
    "world_b": "2c77b77aeb1d8f04a7eef055a4e296f20d1db5853d2ef0fe63f35e0394436018",
    "manifest": "cc4b53d9f35275b3cfb09b10e6bcefbee2aa2777510b5c5e9dd7bd5137821147",
}


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def observation_sha256(observation: Any) -> str:
    return hashlib.sha256(canonical_bytes(observation)).hexdigest()


def _source_sha256(source: Any) -> str:
    return hashlib.sha256(canonical_bytes(source)).hexdigest()


def _anchor_from_exact(point_q4_exact: list[list[int]]) -> list[int]:
    x = Fraction(point_q4_exact[0][0], point_q4_exact[0][1])
    y = Fraction(point_q4_exact[1][0], point_q4_exact[1][1])
    return [y.numerator // (16 * y.denominator), x.numerator // (16 * x.denominator)]


def project_observation(support_input: Any) -> dict[str, Any]:
    geometry = compile_fragment_root(support_input["geometry_root"])
    support = compile_support_partition(support_input)
    return {
        "schema_version": "intersection-ownership-observation-v0.4-research",
        "training_authorized": False,
        "source_support_input_sha256": _source_sha256(support_input),
        "intersections": [
            {
                "id": item["id"],
                "anchor_pixel": _anchor_from_exact(item["point_q4_exact"]),
                "incident_carrier_ids": item["base_stroke_ids"],
            }
            for item in geometry["intersections"]
        ],
        "overlap_pixels": support["cross_carrier_overlaps"],
    }


def build_bridge_support_input() -> dict[str, Any]:
    segments = [
        [[168, 328], [488, 328]],
        [[296, 168], [296, 488]],
        [[360, 168], [360, 488]],
    ]
    supports = [
        {(20, column) for column in range(10, 31)},
        {(10, 18), (20, 18), (20, 19), (20, 20), (30, 18)},
        {(10, 22), (20, 20), (20, 21), (20, 22), (30, 22)},
    ]
    geometry_root = {
        "schema_version": GEOMETRY_SCHEMA,
        "training_authorized": False,
        "base_strokes": [
            {"id": index, "boundary_member_ids": [index], "segment_q4": segment}
            for index, segment in enumerate(segments)
        ],
    }
    parent_supports = []
    for index, pixels in enumerate(supports):
        runs = _encode_runs(pixels)
        parent_supports.append({
            "base_stroke_id": index,
            "support_pixel_count": len(pixels),
            "support_rle_sha256": _rle_sha256(runs),
            "support_runs": runs,
        })
    return {
        "schema_version": SUPPORT_SCHEMA,
        "training_authorized": False,
        "geometry_root": geometry_root,
        "parent_supports": parent_supports,
    }


def _connected(pixels: set[tuple[int, int]]) -> bool:
    if not pixels:
        return False
    unseen = set(pixels)
    start = min(unseen)
    unseen.remove(start)
    queue = deque([start])
    while queue:
        row, column = queue.popleft()
        for row_delta in (-1, 0, 1):
            for column_delta in (-1, 0, 1):
                if row_delta == 0 and column_delta == 0:
                    continue
                neighbor = row + row_delta, column + column_delta
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
    return not unseen


def validate_observation(observation: Any) -> dict[str, Any]:
    if not isinstance(observation, dict) or set(observation) != {
        "intersections", "overlap_pixels", "schema_version",
        "source_support_input_sha256", "training_authorized"
    }:
        raise OwnershipWorldError("OBSERVATION_NOT_CLOSED")
    if observation["schema_version"] != "intersection-ownership-observation-v0.4-research":
        raise OwnershipWorldError("OBSERVATION_SCHEMA_MISMATCH")
    if observation["training_authorized"] is not False:
        raise OwnershipWorldError("TRAINING_NOT_AUTHORIZED")
    source_digest = observation["source_support_input_sha256"]
    if not (
        isinstance(source_digest, str)
        and len(source_digest) == 64
        and all(character in "0123456789abcdef" for character in source_digest)
    ):
        raise OwnershipWorldError("SOURCE_SUPPORT_INPUT_SHA256_INVALID")
    intersections = observation["intersections"]
    overlaps = observation["overlap_pixels"]
    if not isinstance(intersections, list) or not isinstance(overlaps, list):
        raise OwnershipWorldError("OBSERVATION_ARRAY_INVALID")
    if [item.get("id") for item in intersections if isinstance(item, dict)] != list(range(len(intersections))):
        raise OwnershipWorldError("INTERSECTION_IDS_NONCANONICAL")
    seen_anchors: set[tuple[int, int]] = set()
    for index, intersection in enumerate(intersections):
        if not isinstance(intersection, dict) or set(intersection) != {
            "anchor_pixel", "id", "incident_carrier_ids"
        }:
            raise OwnershipWorldError("INTERSECTION_NOT_CLOSED")
        anchor = intersection["anchor_pixel"]
        incident = intersection["incident_carrier_ids"]
        if type(intersection["id"]) is not int or intersection["id"] != index:
            raise OwnershipWorldError("INTERSECTION_IDS_NONCANONICAL")
        if not (
            isinstance(anchor, list) and len(anchor) == 2
            and all(type(value) is int and 0 <= value < 768 for value in anchor)
            and isinstance(incident, list) and len(incident) >= 2
            and all(type(value) is int and value >= 0 for value in incident)
            and incident == sorted(set(incident))
        ):
            raise OwnershipWorldError("INTERSECTION_INVALID")
        anchor_key = tuple(anchor)
        if anchor_key in seen_anchors:
            raise OwnershipWorldError("INTERSECTION_ANCHOR_DUPLICATE")
        seen_anchors.add(anchor_key)
    overlap_pixels: dict[tuple[int, int], list[int]] = {}
    for record in overlaps:
        if not isinstance(record, dict) or set(record) != {"base_stroke_ids", "pixel"}:
            raise OwnershipWorldError("OVERLAP_RECORD_NOT_CLOSED")
        pixel = record["pixel"]
        carriers = record["base_stroke_ids"]
        if not (
            isinstance(pixel, list) and len(pixel) == 2
            and all(type(value) is int and 0 <= value < 768 for value in pixel)
            and isinstance(carriers, list) and len(carriers) >= 2
            and all(type(value) is int and value >= 0 for value in carriers)
            and carriers == sorted(set(carriers))
        ):
            raise OwnershipWorldError("OVERLAP_RECORD_INVALID")
        key = tuple(pixel)
        if key in overlap_pixels:
            raise OwnershipWorldError("OVERLAP_PIXEL_DUPLICATE")
        overlap_pixels[key] = carriers
    if list(overlap_pixels) != sorted(overlap_pixels):
        raise OwnershipWorldError("OVERLAP_ORDER_NONCANONICAL")
    return {"intersections": intersections, "overlap_pixels": overlap_pixels}


def validate_world(observation: Any, world: Any) -> bool:
    try:
        root = validate_observation(observation)
        if not isinstance(world, dict) or set(world) != {
            "observation_sha256", "regions", "schema_version", "training_authorized", "world_id"
        }:
            return False
        if world["schema_version"] != "intersection-ownership-world-v0.4-research":
            return False
        if world["training_authorized"] is not False:
            return False
        if not isinstance(world["world_id"], str):
            return False
        if world["observation_sha256"] != observation_sha256(observation):
            return False
        regions = world["regions"]
        intersections = root["intersections"]
        if not isinstance(regions, list) or len(regions) != len(intersections):
            return False
        union: set[tuple[int, int]] = set()
        for index, region in enumerate(regions):
            if not isinstance(region, dict) or set(region) != {
                "intersection_id", "owner_id", "support_pixels"
            }:
                return False
            if (
                type(region["owner_id"]) is not int
                or type(region["intersection_id"]) is not int
                or region["owner_id"] != index
                or region["intersection_id"] != index
            ):
                return False
            support_raw = region["support_pixels"]
            if not isinstance(support_raw, list):
                return False
            if any(
                not isinstance(pixel, list)
                or len(pixel) != 2
                or any(type(value) is not int or not 0 <= value < 768 for value in pixel)
                for pixel in support_raw
            ):
                return False
            support = {tuple(pixel) for pixel in support_raw}
            if len(support) != len(support_raw) or sorted(support) != [tuple(pixel) for pixel in support_raw]:
                return False
            intersection = intersections[index]
            seed = tuple(intersection["anchor_pixel"])
            incident = set(intersection["incident_carrier_ids"])
            if seed not in support or not _connected(support):
                return False
            if not support <= set(root["overlap_pixels"]):
                return False
            if any(not incident <= set(root["overlap_pixels"][pixel]) for pixel in support):
                return False
            if union & support:
                return False
            union.update(support)
        return union == set(root["overlap_pixels"])
    except (OwnershipWorldError, KeyError, TypeError, ValueError):
        return False


def _pixel_owner(world: dict[str, Any], pixel: list[int]) -> int | None:
    owners = [
        region["owner_id"]
        for region in world["regions"]
        if pixel in region["support_pixels"]
    ]
    return owners[0] if len(owners) == 1 else None


def build_manifest(
    support_input: Any,
    observation: Any,
    world_a: Any,
    world_b: Any,
) -> dict[str, Any]:
    source_digest = _source_sha256(support_input)
    observation_digest = observation_sha256(observation)
    bridge = [20, 20]
    return {
        "schema_version": "intersection-ownership-no-go-manifest-v0.4-research",
        "training_authorized": False,
        "source_support_input_sha256": source_digest,
        "observation_sha256": observation_digest,
        "projection_byte_exact": canonical_bytes(project_observation(support_input)) == canonical_bytes(observation),
        "world_a_legal": validate_world(observation, world_a),
        "world_b_legal": validate_world(observation, world_b),
        "same_observation": (
            isinstance(world_a, dict)
            and isinstance(world_b, dict)
            and world_a.get("observation_sha256") == world_b.get("observation_sha256") == observation_digest
        ),
        "different_owner_partitions": (
            isinstance(world_a, dict)
            and isinstance(world_b, dict)
            and world_a.get("regions") != world_b.get("regions")
        ),
        "bridge_pixel": bridge,
        "world_a_bridge_owner": _pixel_owner(world_a, bridge) if isinstance(world_a, dict) else None,
        "world_b_bridge_owner": _pixel_owner(world_b, bridge) if isinstance(world_b, dict) else None,
        "unique_ownership_authorized": False,
    }


def validate_self_consistent_bundle(
    support_input: Any,
    observation: Any,
    world_a: Any,
    world_b: Any,
    manifest: Any,
) -> bool:
    try:
        expected = build_manifest(support_input, observation, world_a, world_b)
        return (
            canonical_bytes(project_observation(support_input)) == canonical_bytes(observation)
            and validate_world(observation, world_a)
            and validate_world(observation, world_b)
            and expected["same_observation"] is True
            and expected["different_owner_partitions"] is True
            and expected["world_a_bridge_owner"] == 0
            and expected["world_b_bridge_owner"] == 1
            and canonical_bytes(manifest) == canonical_bytes(expected)
        )
    except (KeyError, TypeError, ValueError, OwnershipWorldError):
        return False


def validate_bundle(
    support_input: Any,
    observation: Any,
    world_a: Any,
    world_b: Any,
    manifest: Any,
) -> bool:
    """Validate the preregistered witness, not merely an internally consistent bundle."""
    artifacts = {
        "support_input": support_input,
        "observation": observation,
        "world_a": world_a,
        "world_b": world_b,
        "manifest": manifest,
    }
    return (
        validate_self_consistent_bundle(
            support_input, observation, world_a, world_b, manifest
        )
        and all(
            hashlib.sha256(canonical_bytes(artifacts[name])).hexdigest() == digest
            for name, digest in FROZEN_ARTIFACT_SHA256.items()
        )
    )


def build_bridge_witness() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    observation = project_observation(build_bridge_support_input())
    digest = observation_sha256(observation)
    common = {
        "schema_version": "intersection-ownership-world-v0.4-research",
        "training_authorized": False,
        "observation_sha256": digest,
    }
    world_a = {
        **common,
        "world_id": "bridge_owned_by_left",
        "regions": [
            {"owner_id": 0, "intersection_id": 0, "support_pixels": [[20, 18], [20, 19], [20, 20]]},
            {"owner_id": 1, "intersection_id": 1, "support_pixels": [[20, 21], [20, 22]]},
        ],
    }
    world_b = {
        **common,
        "world_id": "bridge_owned_by_right",
        "regions": [
            {"owner_id": 0, "intersection_id": 0, "support_pixels": [[20, 18], [20, 19]]},
            {"owner_id": 1, "intersection_id": 1, "support_pixels": [[20, 20], [20, 21], [20, 22]]},
        ],
    }
    return observation, world_a, world_b
