from __future__ import annotations

import hashlib
import copy
import json
import unittest
from pathlib import Path

from formal.chemgraph.audit.fragment_root_constructor_v0_4.fragment_root import (
    SCHEMA_VERSION as GEOMETRY_SCHEMA,
)
from formal.chemgraph.audit.fragment_support_partition_v0_4.support_partition import (
    SCHEMA_VERSION as SUPPORT_SCHEMA,
    _rle_sha256,
)

from .component_probe import _components, analyze_overlap_components
from .build_counterexamples import build_values
from .ownership_worlds import (
    FROZEN_ARTIFACT_SHA256,
    build_bridge_support_input,
    build_bridge_witness,
    build_manifest,
    canonical_bytes,
    project_observation,
    validate_bundle,
    validate_self_consistent_bundle,
    validate_world,
)


def _runs(pixels: set[tuple[int, int]]) -> list[list[int]]:
    ordered = sorted(pixels)
    if not ordered:
        return []
    result: list[list[int]] = []
    row, start = ordered[0]
    end = start + 1
    for next_row, column in ordered[1:]:
        if next_row == row and column == end:
            end += 1
        else:
            result.append([row, start, end - start])
            row, start, end = next_row, column, column + 1
    result.append([row, start, end - start])
    return result


def scene(
    segments: list[list[list[int]]], supports: list[set[tuple[int, int]]],
) -> dict:
    ordered = sorted(zip(segments, supports, strict=True), key=lambda item: item[0])
    root = {
        "schema_version": GEOMETRY_SCHEMA,
        "training_authorized": False,
        "base_strokes": [
            {"id": index, "boundary_member_ids": [index], "segment_q4": segment}
            for index, (segment, _) in enumerate(ordered)
        ],
    }
    parent_supports = []
    for index, (_, pixels) in enumerate(ordered):
        runs = _runs(pixels)
        parent_supports.append({
            "base_stroke_id": index,
            "support_pixel_count": len(pixels),
            "support_rle_sha256": _rle_sha256(runs),
            "support_runs": runs,
        })
    return {
        "schema_version": SUPPORT_SCHEMA,
        "training_authorized": False,
        "geometry_root": root,
        "parent_supports": parent_supports,
    }


X_SEGMENTS = [
    [[168, 328], [488, 328]],
    [[328, 168], [328, 488]],
]


class ComponentProbeTests(unittest.TestCase):
    def test_clean_x_is_structurally_exact_only(self) -> None:
        result = analyze_overlap_components(scene(X_SEGMENTS, [
            {(20, column) for column in range(10, 31)},
            {(row, 20) for row in range(10, 31)},
        ]))
        self.assertEqual(result["status"], "STRUCTURALLY_EXACT_NOT_SEMANTICALLY_IDENTIFIED")
        self.assertFalse(result["semantic_identifiability_proven"])
        self.assertEqual(result["regions"][0]["support_pixels"], [[20, 20]])

    def test_disconnected_far_overlap_is_rejected(self) -> None:
        result = analyze_overlap_components(scene(X_SEGMENTS, [
            {(20, column) for column in range(10, 31)} | {(0, 0)},
            {(row, 20) for row in range(10, 31)} | {(0, 0)},
        ]))
        self.assertEqual(result["status"], "UNEXPLAINED_OVERLAP_COMPONENTS")
        self.assertEqual(result["unexplained_component_count"], 1)

    def test_connected_far_snake_is_structurally_accepted_counterexample(self) -> None:
        snake = {(value, value) for value in range(21)}
        result = analyze_overlap_components(scene(X_SEGMENTS, [
            {(20, column) for column in range(10, 31)} | snake,
            {(row, 20) for row in range(10, 31)} | snake,
        ]))
        self.assertEqual(result["status"], "STRUCTURALLY_EXACT_NOT_SEMANTICALLY_IDENTIFIED")
        self.assertEqual(result["regions"][0]["support_pixel_count"], 21)
        self.assertIn([0, 0], result["regions"][0]["support_pixels"])

    def test_two_intersections_joined_by_overlap_are_rejected(self) -> None:
        horizontal = {(20, column) for column in range(10, 31)}
        bridge = {(20, column) for column in range(15, 26)}
        result = analyze_overlap_components(scene([
            [[168, 328], [488, 328]],
            [[248, 168], [248, 488]],
            [[408, 168], [408, 488]],
        ], [
            horizontal,
            {(row, 15) for row in range(10, 31)} | bridge,
            {(row, 25) for row in range(10, 31)},
        ]))
        self.assertEqual(result["status"], "MERGED_INTERSECTION_COMPONENTS")
        self.assertEqual(result["merged_component_count"], 1)

    def test_three_carrier_anchor_preserves_complete_fiber(self) -> None:
        result = analyze_overlap_components(scene([
            [[168, 168], [488, 488]],
            [[168, 328], [488, 328]],
            [[328, 168], [328, 488]],
        ], [
            {(value, value) for value in range(10, 31)},
            {(20, value) for value in range(10, 31)},
            {(value, 20) for value in range(10, 31)},
        ]))
        self.assertEqual(result["status"], "STRUCTURALLY_EXACT_NOT_SEMANTICALLY_IDENTIFIED")
        self.assertEqual(result["regions"][0]["base_stroke_ids"], [0, 1, 2])

    def test_region_union_can_hide_partial_three_carrier_fibers(self) -> None:
        result = analyze_overlap_components(scene([
            [[168, 168], [488, 488]],
            [[168, 328], [488, 328]],
            [[328, 168], [328, 488]],
        ], [
            {(10, 10), (20, 20), (21, 20), (30, 30)},
            {(20, 10), (20, 20), (20, 21), (20, 30)},
            {(10, 20), (20, 21), (21, 20), (30, 20)},
        ]))
        self.assertEqual(result["status"], "STRUCTURALLY_EXACT_NOT_SEMANTICALLY_IDENTIFIED")
        fibers = result["regions"][0]["pixel_carrier_fibers"]
        self.assertEqual([item["base_stroke_ids"] for item in fibers], [
            [0, 1], [1, 2], [0, 2]
        ])

    def test_eight_connectivity_merges_corner_touch(self) -> None:
        self.assertEqual(_components({(2, 2), (3, 3)}), [[(2, 2), (3, 3)]])

    def test_same_observation_has_two_legal_owner_partitions(self) -> None:
        observation, world_a, world_b = build_bridge_witness()
        self.assertTrue(validate_world(observation, world_a))
        self.assertTrue(validate_world(observation, world_b))
        self.assertEqual(world_a["observation_sha256"], world_b["observation_sha256"])
        self.assertNotEqual(world_a["regions"], world_b["regions"])
        self.assertIn([20, 20], world_a["regions"][0]["support_pixels"])
        self.assertIn([20, 20], world_b["regions"][1]["support_pixels"])

    def test_incomplete_or_double_owned_bridge_is_illegal(self) -> None:
        observation, world_a, _ = build_bridge_witness()
        incomplete = {
            **world_a,
            "regions": [
                {**world_a["regions"][0], "support_pixels": [[20, 18], [20, 19]]},
                world_a["regions"][1],
            ],
        }
        double_owned = {
            **world_a,
            "regions": [
                world_a["regions"][0],
                {**world_a["regions"][1], "support_pixels": [[20, 20], [20, 21], [20, 22]]},
            ],
        }
        self.assertFalse(validate_world(observation, incomplete))
        self.assertFalse(validate_world(observation, double_owned))

    def test_boolean_cannot_impersonate_owner_id(self) -> None:
        observation, world_a, _ = build_bridge_witness()
        world_a["regions"][0]["owner_id"] = False
        self.assertFalse(validate_world(observation, world_a))

    def test_counterexamples_replay_byte_exact(self) -> None:
        root = Path(__file__).resolve().parent / "counterexamples"
        for filename, value in build_values().items():
            with self.subTest(filename=filename):
                self.assertEqual((root / filename).read_bytes(), canonical_bytes(value))

    def test_full_support_input_projects_to_frozen_observation(self) -> None:
        root = Path(__file__).resolve().parent / "counterexamples"
        support_input = build_bridge_support_input()
        projected = project_observation(support_input)
        self.assertEqual(root.joinpath("support_input.json").read_bytes(), canonical_bytes(support_input))
        self.assertEqual(root.joinpath("observation.json").read_bytes(), canonical_bytes(projected))
        self.assertEqual(
            projected["source_support_input_sha256"],
            hashlib.sha256(canonical_bytes(support_input)).hexdigest(),
        )

    def test_single_bundle_validator_binds_the_complete_chain(self) -> None:
        root = Path(__file__).resolve().parent / "counterexamples"
        values = {
            name: json.loads(root.joinpath(name).read_bytes())
            for name in (
                "support_input.json", "observation.json", "world_a.json",
                "world_b.json", "manifest.json",
            )
        }
        self.assertTrue(validate_bundle(
            values["support_input.json"], values["observation.json"],
            values["world_a.json"], values["world_b.json"], values["manifest.json"],
        ))

        for name, value in values.items():
            key = name.removesuffix(".json")
            self.assertEqual(
                hashlib.sha256(canonical_bytes(value)).hexdigest(),
                FROZEN_ARTIFACT_SHA256[key],
            )

    def test_bundle_rejects_synchronized_observation_forgery(self) -> None:
        support_input = build_bridge_support_input()
        observation, world_a, world_b = build_bridge_witness()
        forged_observation = copy.deepcopy(observation)
        forged_observation["source_support_input_sha256"] = "0" * 64
        forged_digest = hashlib.sha256(canonical_bytes(forged_observation)).hexdigest()
        forged_a = {**world_a, "observation_sha256": forged_digest}
        forged_b = {**world_b, "observation_sha256": forged_digest}
        forged_manifest = build_manifest(
            support_input, forged_observation, forged_a, forged_b
        )
        self.assertFalse(validate_bundle(
            support_input, forged_observation, forged_a, forged_b, forged_manifest
        ))

    def test_frozen_bundle_rejects_synchronized_complete_chain_replacement(self) -> None:
        support_input = copy.deepcopy(build_bridge_support_input())
        parent = support_input["parent_supports"][0]
        parent["support_runs"] = sorted(parent["support_runs"] + [[0, 767, 1]])
        parent["support_pixel_count"] += 1
        parent["support_rle_sha256"] = _rle_sha256(parent["support_runs"])

        observation = project_observation(support_input)
        _, world_a, world_b = build_bridge_witness()
        digest = hashlib.sha256(canonical_bytes(observation)).hexdigest()
        world_a["observation_sha256"] = digest
        world_b["observation_sha256"] = digest
        manifest = build_manifest(support_input, observation, world_a, world_b)

        self.assertTrue(validate_self_consistent_bundle(
            support_input, observation, world_a, world_b, manifest
        ))
        self.assertFalse(validate_bundle(
            support_input, observation, world_a, world_b, manifest
        ))


if __name__ == "__main__":
    unittest.main()
