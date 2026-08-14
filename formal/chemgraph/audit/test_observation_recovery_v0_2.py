from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from formal.chemgraph.audit.finite_micro_scene_v0_1 import (
    PhysicalMember,
    Scene,
    canonical_graph_signature,
    crossing_junction_fixtures,
    double_parallel_fixtures,
)
from formal.chemgraph.audit.observation_recovery_v0_2 import (
    AMBIGUITY_CROSSING,
    AMBIGUITY_PARALLEL,
    CROSSING_PRIOR_ID,
    FROZEN_CERTIFICATES_SHA256,
    FROZEN_MANIFEST_SHA256,
    PARALLEL_PRIOR_ID,
    Candidate,
    ObservationClass,
    RecoveryAuditError,
    _generic_graph_signature,
    decode_oracle_field,
    load_frozen_inputs,
    merge_visible_lanes,
    oracle_field_for_scene,
    oracle_field_from_observable_evidence,
    strategy_a_conservative,
    strategy_b_resolved,
)


def _members(scene: Scene):
    return (member for bond in scene.bonds for member in bond.members)


def _observation_class(*scenes: Scene) -> ObservationClass:
    evidence = [merge_visible_lanes(_members(scene)) for scene in scenes]
    if any(item != evidence[0] for item in evidence[1:]):
        raise AssertionError("Test scenes do not have one canonical lane evidence")
    candidates = tuple(
        Candidate(
            scene_id=scene.scene_id,
            graph_signature=canonical_graph_signature(scene),
            grammar_layer="test",
            field=oracle_field_for_scene(scene, evidence[0]),
        )
        for scene in scenes
    )
    return ObservationClass("not-used-by-forward-chooser", evidence[0], candidates)


class ObservationRecoveryV02Tests(unittest.TestCase):
    def test_frozen_inputs_are_authenticated(self) -> None:
        root = (
            Path(__file__).resolve().parent
            / "results"
            / "finite_geometry_grammar_v0_1"
            / "final_a"
        )
        frozen = load_frozen_inputs(
            root / "scene_manifest.jsonl", root / "collision_certificates.json"
        )
        self.assertEqual(frozen.manifest_sha256, FROZEN_MANIFEST_SHA256)
        self.assertEqual(frozen.certificates_sha256, FROZEN_CERTIFICATES_SHA256)
        self.assertEqual(len(frozen.manifest_rows), 66_716)

        with tempfile.TemporaryDirectory() as directory:
            bad_manifest = Path(directory) / "manifest.jsonl"
            bad_manifest.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(RecoveryAuditError, "SHA256 mismatch"):
                load_frozen_inputs(
                    bad_manifest, root / "collision_certificates.json"
                )

    def test_strategy_a_abstains_on_crossing_and_parallel_collisions(self) -> None:
        crossing = _observation_class(*crossing_junction_fixtures())
        crossing_decision = strategy_a_conservative(crossing)
        self.assertEqual(crossing_decision.result_type, "Ambiguous")
        self.assertEqual(crossing_decision.ambiguity_kind, AMBIGUITY_CROSSING)
        self.assertIsNone(crossing_decision.graph_signature)
        self.assertIsNone(crossing_decision.protocol_choice_graph_signature)
        self.assertIsNone(
            crossing_decision.oracle_field["protocol_interpretation"]
        )

        parallel = _observation_class(*double_parallel_fixtures())
        parallel_decision = strategy_a_conservative(parallel)
        self.assertEqual(parallel_decision.result_type, "Ambiguous")
        self.assertEqual(parallel_decision.ambiguity_kind, AMBIGUITY_PARALLEL)

    def test_strategy_b_is_a_deterministic_evidence_forward_chooser(self) -> None:
        crossing_scene, junction_scene = crossing_junction_fixtures()
        crossing = _observation_class(crossing_scene, junction_scene)
        decision = strategy_b_resolved(crossing)
        self.assertEqual(decision.result_type, "Graph")
        self.assertEqual(
            decision.graph_signature, canonical_graph_signature(crossing_scene)
        )
        self.assertEqual(
            decision.protocol_choice_graph_signature, decision.graph_signature
        )
        self.assertEqual(
            decision.active_prior_ids,
            (PARALLEL_PRIOR_ID, CROSSING_PRIOR_ID),
        )
        self.assertTrue(decision.decode_left_inverse)
        self.assertFalse(decision.deployable_from_opaque_observation)
        self.assertFalse(decision.observation_extractor_implemented)

        double_scene, parallel_scene = double_parallel_fixtures()
        parallel = _observation_class(double_scene, parallel_scene)
        decision = strategy_b_resolved(parallel)
        self.assertEqual(decision.result_type, "Graph")
        self.assertEqual(
            decision.graph_signature, canonical_graph_signature(double_scene)
        )

    def test_strategy_b_forward_choice_never_reads_observation_hash_or_meaning(self) -> None:
        double_scene, parallel_scene = double_parallel_fixtures()
        original = _observation_class(double_scene, parallel_scene)
        renamed = replace(original, observation_sha256="completely-different-hash")
        original_field = oracle_field_from_observable_evidence(original.visible_lanes)
        renamed_field = oracle_field_from_observable_evidence(renamed.visible_lanes)
        self.assertEqual(original_field, renamed_field)
        self.assertEqual(
            _generic_graph_signature(original_field),
            _generic_graph_signature(renamed_field),
        )
        serialized = json.dumps(original_field, sort_keys=True)
        self.assertNotIn(original.observation_sha256, serialized)
        self.assertNotIn(double_scene.scene_id, serialized)
        self.assertNotIn(canonical_graph_signature(double_scene), serialized)

    def test_global_parallel_prior_has_a_real_coverage_counterexample(self) -> None:
        _, parallel_scene = double_parallel_fixtures()
        only_parallel_singles = _observation_class(parallel_scene)
        decision = strategy_b_resolved(only_parallel_singles)
        self.assertEqual(decision.result_type, "Reject")
        self.assertIsNone(decision.graph_signature)
        self.assertEqual(
            decision.protocol_choice_graph_signature,
            canonical_graph_signature(double_parallel_fixtures()[0]),
        )
        self.assertEqual(decision.survivor_count, 0)
        self.assertTrue(decision.semantic_field_factorizes_through_evidence)
        self.assertFalse(decision.deployable_from_opaque_observation)

    def test_oracle_left_inverse_is_only_semantic_serialization(self) -> None:
        for scene in (*crossing_junction_fixtures(), *double_parallel_fixtures()):
            with self.subTest(scene=scene.scene_id):
                field = oracle_field_for_scene(scene)
                self.assertEqual(
                    decode_oracle_field(field), canonical_graph_signature(scene)
                )
                self.assertTrue(field["audit_oracle_only"])
                self.assertFalse(field["deployable_from_opaque_observation"])
                self.assertEqual(
                    field["evidence_source"],
                    "finite_scene_physical_members_not_opaque_tensor",
                )

    def test_lane_evidence_and_prior_graph_are_rigid_motion_invariant(self) -> None:
        scene, _ = crossing_junction_fixtures()

        def transform_scene(name: str, transform) -> Scene:
            return replace(
                scene,
                scene_id=name,
                nodes=tuple(
                    replace(
                        node,
                        x=transform((node.x, node.y))[0],
                        y=transform((node.x, node.y))[1],
                    )
                    for node in scene.nodes
                ),
                bonds=tuple(
                    replace(
                        bond,
                        members=tuple(
                            PhysicalMember(
                                transform(member.begin), transform(member.end)
                            )
                            for member in bond.members
                        ),
                    )
                    for bond in scene.bonds
                ),
            )

        expected_graph = _generic_graph_signature(
            oracle_field_from_observable_evidence(
                merge_visible_lanes(_members(scene))
            )
        )
        transforms = {
            "translated": lambda point: (point[0] + 4, point[1] - 3),
            "rotated90": lambda point: (-point[1], point[0]),
            "reflected": lambda point: (-point[0], point[1]),
            "scaled": lambda point: (3 * point[0], 3 * point[1]),
        }
        for name, transform in transforms.items():
            with self.subTest(name=name):
                transformed = transform_scene(name, transform)
                graph = _generic_graph_signature(
                    oracle_field_from_observable_evidence(
                        merge_visible_lanes(_members(transformed))
                    )
                )
                self.assertEqual(graph, expected_graph)


if __name__ == "__main__":
    unittest.main()
