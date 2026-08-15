from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from .build_observation import build_observation
from .candidate_domain import build_candidate_domain, ensure_constructor_has_no_oracle_literal
from .common import (
    LIMITS,
    CandidateDomainError,
    canonical_bytes,
    capacity_gate,
    framed_items_sha256,
    sha256,
)


ROOT = Path(__file__).resolve().parent


def _walk_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        keys.update(value)
        for child in value.values():
            keys.update(_walk_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_walk_keys(child))
    return keys


class CandidateDomainTests(unittest.TestCase):
    def test_frozen_observation_and_candidate_replay_byte_exact(self) -> None:
        observation = build_observation()
        candidate = build_candidate_domain(observation)
        self.assertEqual(
            ROOT.joinpath("r0_observation.json").read_bytes(),
            canonical_bytes(observation),
        )
        self.assertEqual(
            ROOT.joinpath("r0_candidate_domain.json").read_bytes(),
            canonical_bytes(candidate),
        )

    def test_observation_is_annotation_free_and_closed(self) -> None:
        observation = build_observation()
        self.assertEqual(set(observation), {
            "geometry_root", "parent_supports", "png_sha256", "renderer_sha256",
            "schema_version", "support_input_sha256", "training_authorized",
        })
        ensure_constructor_has_no_oracle_literal()
        forged = {**observation, "oracle_annotation": {"bridge_owner": 0}}
        with self.assertRaisesRegex(CandidateDomainError, "OBSERVATION_NOT_CLOSED"):
            build_candidate_domain(forged)

    def test_factorized_counts_and_nontriviality_metrics(self) -> None:
        candidate = build_candidate_domain(build_observation())
        self.assertEqual(candidate["counts"], {
            "adjacency_edge_count": 188,
            "ambiguous_variable_count": 10,
            "atomic_fragment_count": 5,
            "carrier_count": 3,
            "constraint_count": 7,
            "domain_class_count": 3,
            "evidence_membership_count": 58,
            "owner_option_count": 2,
            "pixel_variable_count": 58,
            "residual_reason_count": 0,
        })
        self.assertEqual(candidate["domain_classes"], [
            {"id": 0, "owner_option_ids": [0]},
            {"id": 1, "owner_option_ids": [0, 1]},
            {"id": 2, "owner_option_ids": [1]},
        ])
        metrics = candidate["metrics"]
        self.assertEqual(metrics["raw_cartesian_assignment_count"], 1024)
        self.assertEqual(metrics["universal_option_membership_count"], 116)
        self.assertEqual(metrics["candidate_option_membership_count"], 68)
        self.assertEqual(metrics["eligible_option_count"], 68)
        self.assertEqual(metrics["retained_eligible_option_count"], 68)
        self.assertEqual(metrics["ineligible_option_count"], 48)
        self.assertEqual(metrics["eliminated_ineligible_option_count"], 48)
        self.assertEqual(metrics["eligible_option_recall"], [1, 1])
        self.assertEqual(metrics["ineligible_option_elimination_recall"], [1, 1])
        self.assertEqual(metrics["eliminated_option_ratio"], [12, 29])
        self.assertEqual(metrics["mean_domain_size"], [34, 29])
        self.assertEqual(metrics["singleton_ratio"], [24, 29])
        self.assertEqual(metrics["multi_intersection_ratio"], [5, 29])
        self.assertEqual(metrics["residual_containing_ratio"], [0, 1])

    def test_evidence_is_multicarrier_but_unique_within_each_carrier(self) -> None:
        candidate = build_candidate_domain(build_observation())
        saw_three_carriers = False
        for evidence in candidate["evidence_memberships"]:
            carriers = evidence["carrier_ids"]
            fragments = evidence["fragment_memberships"]
            self.assertEqual(len(fragments), len(carriers))
            self.assertEqual(
                [item["carrier_id"] for item in fragments], carriers
            )
            self.assertEqual(len({item["carrier_id"] for item in fragments}), len(carriers))
            saw_three_carriers |= len(carriers) == 3
        self.assertTrue(saw_three_carriers)

    def test_domains_use_per_pixel_fiber_not_region_union(self) -> None:
        candidate = build_candidate_domain(build_observation())
        classes = {
            item["id"]: item["owner_option_ids"]
            for item in candidate["domain_classes"]
        }
        evidence = {
            item["id"]: item for item in candidate["evidence_memberships"]
        }
        for variable in candidate["pixel_variables"]:
            carriers = evidence[variable["evidence_membership_id"]]["carrier_ids"]
            domain = classes[variable["domain_class_id"]]
            if carriers == [0, 1]:
                self.assertEqual(domain, [0])
            elif carriers == [0, 2]:
                self.assertEqual(domain, [1])
            elif carriers == [0, 1, 2]:
                self.assertEqual(domain, [0, 1])
            else:
                self.fail(f"unexpected carrier fiber: {carriers}")

    def test_section_hashes_and_stage_chain_rebuild(self) -> None:
        candidate = build_candidate_domain(build_observation())
        section_items = {
            "source_binding": [candidate["source_binding"]],
            "limits": [candidate["limits"]],
            "counts": [candidate["counts"]],
            "owner_options": candidate["owner_options"],
            "evidence_memberships": candidate["evidence_memberships"],
            "domain_classes": candidate["domain_classes"],
            "pixel_variables": candidate["pixel_variables"],
            "adjacency_edges": candidate["adjacency_edges"],
            "constraints": candidate["constraints"],
            "residual_reasons": candidate["residual_reasons"],
            "metrics": [candidate["metrics"]],
        }
        declared = {item["section"]: item for item in candidate["section_hashes"]}
        self.assertEqual(set(declared), set(section_items))
        for name, items in section_items.items():
            self.assertEqual(declared[name]["item_count"], len(items))
            self.assertEqual(declared[name]["sha256"], framed_items_sha256(items))
        previous = candidate["source_binding"]["observation_sha256"]
        for index, stage in enumerate(candidate["stage_trace"]):
            self.assertEqual(stage["index"], index)
            self.assertEqual(stage["input_sha256"], previous)
            self.assertEqual(stage["output_sha256"], declared[stage["stage"]]["sha256"])
            previous = stage["output_sha256"]

    def test_prohibited_output_fields_are_absent(self) -> None:
        keys = _walk_keys(build_candidate_domain(build_observation()))
        self.assertTrue({
            "winner", "unique", "accepted", "preferred_option", "top_k",
            "graph", "sdf",
        }.isdisjoint(keys))

    def test_every_capacity_axis_accepts_n_and_rejects_n_plus_one(self) -> None:
        for name, limit in LIMITS.items():
            with self.subTest(name=name, side="exact"):
                capacity_gate(name, limit)
            with self.subTest(name=name, side="plus_one"):
                with self.assertRaisesRegex(CandidateDomainError, "CAPACITY_EXCEEDED"):
                    capacity_gate(name, limit + 1)

    def test_constructor_routes_every_derived_axis_through_capacity_gate(self) -> None:
        source = ROOT.joinpath("candidate_domain.py").read_text(encoding="utf-8")
        for name in LIMITS:
            with self.subTest(name=name):
                self.assertIn(f'capacity_gate("{name}"', source)
        self.assertNotIn("top_k", source)
        self.assertNotIn("truncate", source.lower())

    def test_renamed_reversed_and_shuffled_source_rebuilds_same_semantic_csp(self) -> None:
        observation = build_observation()
        transformed = copy.deepcopy(observation)
        id_map = {0: 7, 1: 3, 2: 5}
        for stroke in transformed["geometry_root"]["base_strokes"]:
            old_id = stroke["id"]
            stroke["id"] = id_map[old_id]
            stroke["boundary_member_ids"] = [id_map[old_id]]
            stroke["segment_q4"] = list(reversed(stroke["segment_q4"]))
        transformed["geometry_root"]["base_strokes"].reverse()
        for support in transformed["parent_supports"]:
            support["base_stroke_id"] = id_map[support["base_stroke_id"]]
        transformed["parent_supports"].reverse()
        raw_support_input = {
            "geometry_root": transformed["geometry_root"],
            "parent_supports": transformed["parent_supports"],
            "schema_version": "chemgraph-fragment-support-partition-v0.4-research",
            "training_authorized": False,
        }
        transformed["support_input_sha256"] = sha256(canonical_bytes(raw_support_input))

        baseline = build_candidate_domain(observation)
        rebuilt = build_candidate_domain(transformed)
        semantic_sections = (
            "owner_options", "evidence_memberships", "domain_classes",
            "pixel_variables", "adjacency_edges", "constraints", "residual_reasons",
            "counts", "metrics",
        )
        for name in semantic_sections:
            with self.subTest(section=name):
                self.assertEqual(rebuilt[name], baseline[name])
        self.assertNotEqual(
            rebuilt["source_binding"]["observation_sha256"],
            baseline["source_binding"]["observation_sha256"],
        )

    def test_duplicate_source_boundary_member_id_is_closed_rejected(self) -> None:
        observation = build_observation()
        mutated = copy.deepcopy(observation)
        strokes = mutated["geometry_root"]["base_strokes"]
        strokes[1]["boundary_member_ids"] = strokes[0]["boundary_member_ids"]
        raw_support_input = {
            "geometry_root": mutated["geometry_root"],
            "parent_supports": mutated["parent_supports"],
            "schema_version": "chemgraph-fragment-support-partition-v0.4-research",
            "training_authorized": False,
        }
        mutated["support_input_sha256"] = sha256(canonical_bytes(raw_support_input))
        with self.assertRaisesRegex(
            CandidateDomainError, "R0_BOUNDARY_MEMBER_ID_NOT_UNIQUE"
        ):
            build_candidate_domain(mutated)

    def test_declared_count_and_hash_mutations_do_not_change_rebuild(self) -> None:
        observation = build_observation()
        expected = build_candidate_domain(observation)
        mutated = copy.deepcopy(expected)
        mutated["counts"]["pixel_variable_count"] -= 1
        mutated["section_hashes"][0]["sha256"] = "0" * 64
        self.assertNotEqual(canonical_bytes(mutated), canonical_bytes(expected))
        self.assertEqual(
            canonical_bytes(build_candidate_domain(observation)),
            canonical_bytes(expected),
        )

    def test_frozen_json_is_canonical(self) -> None:
        for name in (
            "r0_observation.json", "r0_candidate_domain.json",
            "r0_verification.json", "r0_replay.json",
        ):
            raw = ROOT.joinpath(name).read_bytes()
            self.assertEqual(raw, canonical_bytes(json.loads(raw)))

    def test_replay_manifest_records_full_determinism_run(self) -> None:
        replay = json.loads(ROOT.joinpath("r0_replay.json").read_bytes())
        self.assertEqual(replay["status"], "passed")
        self.assertEqual(replay["sequential_replay_count"], 1000)
        self.assertEqual(replay["parallel_replay_count"], 32)
        self.assertEqual(replay["worker_count"], 8)
        self.assertEqual(replay["unique_digest_count"], 1)
        self.assertEqual(
            replay["artifact_sha256"],
            sha256(ROOT.joinpath("r0_candidate_domain.json").read_bytes()),
        )


if __name__ == "__main__":
    unittest.main()
