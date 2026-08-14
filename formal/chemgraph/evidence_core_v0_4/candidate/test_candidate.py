from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

from formal.chemgraph.evidence_core_v0_4.candidate.candidate import (
    CandidateError,
    build_candidate_set,
    canonical_json_bytes,
    graphs_isomorphic,
)


ROOT = Path(__file__).resolve().parent
FIXTURES = ROOT / "fixtures"
CANDIDATE = ROOT / "candidate.py"


def load_fixture(name: str) -> dict:
    path = FIXTURES / f"{name}.evidence.json"
    raw = path.read_bytes()
    value = json.loads(raw)
    if raw != canonical_json_bytes(value):
        raise AssertionError(f"fixture is not canonical: {path.name}")
    return value


def degree_sequence(graph: dict) -> list[int]:
    degrees = [0] * len(graph["nodes"])
    for edge in graph["edges"]:
        degrees[edge["begin"]] += 1
        degrees[edge["end"]] += 1
    return sorted(degrees)


class GeometryInterpretationTests(unittest.TestCase):
    def test_chain_fan_ring_and_event_tolerant_junction_are_unique(self) -> None:
        expected = {
            "chain": ([1, 1, 2], 2),
            "fan": ([1, 1, 1, 3], 3),
            "ring": ([2, 2, 2], 3),
            "tolerant_junction": ([1, 1, 2], 2),
        }
        for name, (degrees, edge_count) in expected.items():
            with self.subTest(name=name):
                result = build_candidate_set(load_fixture(name))
                self.assertEqual(result["decode_kind"], "Graph")
                self.assertEqual(len(result["candidates"]), 1)
                graph = result["candidates"][0]
                self.assertEqual(len(graph["edges"]), edge_count)
                self.assertEqual(degree_sequence(graph), degrees)

    def test_proper_crossing_preserves_crossing_and_junction(self) -> None:
        result = build_candidate_set(load_fixture("crossing"))
        self.assertEqual(result["decode_kind"], "Ambiguous")
        self.assertEqual(len(result["candidates"]), 2)
        summaries = sorted(
            (len(graph["nodes"]), len(graph["edges"]), degree_sequence(graph))
            for graph in result["candidates"]
        )
        self.assertEqual(summaries, [
            (4, 2, [1, 1, 1, 1]),
            (5, 4, [1, 1, 1, 1, 4]),
        ])

    def test_aligned_double_and_triple_have_bundle_and_singles(self) -> None:
        for name, order in (
            ("double_parallel", 2),
            ("triple_parallel", 3),
            ("noisy_triple_parallel", 3),
        ):
            with self.subTest(name=name):
                result = build_candidate_set(load_fixture(name))
                self.assertEqual(result["decode_kind"], "Ambiguous")
                self.assertEqual(len(result["candidates"]), 2)
                orders = sorted(tuple(edge["order"] for edge in graph["edges"])
                                for graph in result["candidates"])
                self.assertIn((order,), orders)
                self.assertIn(tuple(1 for _ in range(order)), orders)

    def test_nonaligned_parallel_remains_disconnected_singles(self) -> None:
        result = build_candidate_set(load_fixture("nonaligned_parallel"))
        self.assertEqual(result["decode_kind"], "Graph")
        self.assertEqual(len(result["candidates"]), 1)
        graph = result["candidates"][0]
        self.assertEqual(degree_sequence(graph), [1, 1, 1, 1])
        self.assertEqual([edge["order"] for edge in graph["edges"]], [1, 1])

    def test_glyph_like_has_text_stroke_and_bond_lane_roles(self) -> None:
        result = build_candidate_set(load_fixture("glyph_like"))
        self.assertEqual(result["decode_kind"], "Ambiguous")
        self.assertEqual(len(result["candidates"]), 2)
        roles = [tuple(node["role"] for node in graph["nodes"])
                 for graph in result["candidates"]]
        self.assertEqual(sum("text_stroke" in item for item in roles), 1)
        hypotheses = {
            tuple((item["primitive_id"], item["role"])
                  for item in graph["role_hypotheses"])
            for graph in result["candidates"]
        }
        self.assertEqual(hypotheses, {((1, "bond_lane"),), ((1, "text_stroke"),)})
        self.assertNotIn("I", canonical_json_bytes(result).decode("ascii"))

    def test_empty_evidence_rejects_without_guessing(self) -> None:
        result = build_candidate_set(load_fixture("empty"))
        self.assertEqual(result, {
            "candidates": [],
            "decode_kind": "Reject",
            "schema_version": "anonymous-candidate-set-v0.4",
        })

    def test_upstream_reject_evidence_remains_reject(self) -> None:
        result = build_candidate_set(load_fixture("reject"))
        self.assertEqual(result, {
            "candidates": [],
            "decode_kind": "Reject",
            "schema_version": "anonymous-candidate-set-v0.4",
        })


class AnonymousGraphTests(unittest.TestCase):
    def test_isomorphism_ignores_node_numbering_and_edge_order(self) -> None:
        left = {
            "nodes": [{"id": 0, "role": "implicit"}, {"id": 1, "role": "implicit"},
                      {"id": 2, "role": "implicit"}],
            "edges": [{"begin": 0, "end": 1, "order": 1},
                      {"begin": 1, "end": 2, "order": 2}],
            "role_hypotheses": [],
        }
        right = {
            "nodes": [{"id": 0, "role": "implicit"}, {"id": 1, "role": "implicit"},
                      {"id": 2, "role": "implicit"}],
            "edges": [{"begin": 2, "end": 0, "order": 2},
                      {"begin": 1, "end": 2, "order": 1}],
            "role_hypotheses": [],
        }
        self.assertTrue(graphs_isomorphic(left, right))

    def test_isomorphism_respects_bond_order_and_node_role(self) -> None:
        base = {
            "nodes": [{"id": 0, "role": "implicit"}, {"id": 1, "role": "implicit"}],
            "edges": [{"begin": 0, "end": 1, "order": 1}],
            "role_hypotheses": [],
        }
        order_changed = copy.deepcopy(base)
        order_changed["edges"][0]["order"] = 2
        role_changed = copy.deepcopy(base)
        role_changed["nodes"][0]["role"] = "text_stroke"
        self.assertFalse(graphs_isomorphic(base, order_changed))
        self.assertFalse(graphs_isomorphic(base, role_changed))


class BoundaryAndClosureTests(unittest.TestCase):
    def test_all_fixtures_and_outputs_are_canonical_and_bounded(self) -> None:
        for path in sorted(FIXTURES.glob("*.evidence.json")):
            with self.subTest(path=path.name):
                evidence = load_fixture(path.name.removesuffix(".evidence.json"))
                output = canonical_json_bytes(build_candidate_set(evidence))
                self.assertLessEqual(len(path.read_bytes()), 65_536)
                self.assertLessEqual(len(output), 65_536)
                decoded = json.loads(output)
                self.assertEqual(output, canonical_json_bytes(decoded))
                self.assertLessEqual(len(decoded["candidates"]), 64)
                for graph in decoded["candidates"]:
                    self.assertEqual([node["id"] for node in graph["nodes"]],
                                     list(range(len(graph["nodes"]))))
                    self.assertLessEqual(len(graph["nodes"]), 128)
                    self.assertLessEqual(len(graph["edges"]), 128)
                    self.assertTrue(all(edge["begin"] < edge["end"] for edge in graph["edges"]))

    def test_closed_input_rejects_extra_field_float_and_capacity_overflow(self) -> None:
        evidence = load_fixture("chain")
        extra = copy.deepcopy(evidence)
        extra["png_sha256"] = "0" * 64
        with self.assertRaisesRegex(CandidateError, "closed fields mismatch"):
            build_candidate_set(extra)
        floating = copy.deepcopy(evidence)
        floating["stroke_primitives"][0]["support_score"] = 1.5
        with self.assertRaisesRegex(CandidateError, "expected integer"):
            build_candidate_set(floating)
        overflow = load_fixture("empty")
        overflow["stroke_primitives"] = [{
            "begin_q4": [index * 32, 100],
            "closed": False,
            "end_q4": [index * 32 + 16, 100],
            "id": index,
            "local_residual_count": 0,
            "orientation_bucket": 0,
            "straight": True,
            "support_score": 10000,
            "width_q4": 16,
        } for index in range(65)]
        with self.assertRaisesRegex(CandidateError, "capacity exceeded"):
            build_candidate_set(overflow)

    def test_valid_but_unsupported_geometry_returns_reject(self) -> None:
        closed = load_fixture("chain")
        closed["stroke_primitives"][0]["closed"] = True
        self.assertEqual(build_candidate_set(closed)["decode_kind"], "Reject")

        too_many_axes = load_fixture("empty")
        too_many_axes["stroke_primitives"] = [{
            "begin_q4": [1000 + index * 100, 2000],
            "closed": False,
            "end_q4": [1000 + index * 100, 2400],
            "id": index,
            "local_residual_count": 0,
            "orientation_bucket": 180,
            "straight": True,
            "support_score": 10000,
            "width_q4": 16,
        } for index in range(7)]
        too_many_axes["event_regions"] = [{
            "bbox_q4": [990 + index * 100, 1990, 1010 + index * 100, 2410],
            "centroid_q4": [1000 + index * 100, 2200],
            "id": index,
            "incident_primitive_ids": [index],
            "kind_hypothesis": "glyph_like",
            "score": 9000,
        } for index in range(7)]
        self.assertEqual(build_candidate_set(too_many_axes)["decode_kind"], "Reject")

    def test_public_evidence_schema_copy_has_frozen_interface_hash(self) -> None:
        payload = (ROOT / "evidence_core.schema.json").read_bytes()
        self.assertEqual(
            hashlib.sha256(payload).hexdigest(),
            "f4624ef3971373cad5ceb78562ab99fd549bacc7920472c93177da0a3ecd12ed",
        )
        schema = json.loads(payload)
        primitive = schema["oneOf"][0]["properties"]["stroke_primitives"]["items"]["properties"]
        self.assertEqual(primitive["orientation_bucket"]["maximum"], 719)
        self.assertIn("begin_q4", primitive)
        self.assertIn("end_q4", primitive)

    def test_schema_valid_array_order_endpoint_direction_and_incident_order_are_neutral(self) -> None:
        chain = load_fixture("chain")
        expected_chain = build_candidate_set(chain)
        chain["stroke_primitives"].reverse()
        item = chain["stroke_primitives"][1]
        item["begin_q4"], item["end_q4"] = item["end_q4"], item["begin_q4"]
        item["orientation_bucket"] = 719
        actual_chain = build_candidate_set(chain)
        self.assertEqual(actual_chain, expected_chain)

        crossing = load_fixture("crossing")
        expected_crossing = build_candidate_set(crossing)
        crossing["event_regions"][0]["incident_primitive_ids"] = [1, 0]
        crossing["stroke_primitives"].reverse()
        self.assertEqual(build_candidate_set(crossing), expected_crossing)

    def test_cli_reads_only_canonical_stdin_and_is_byte_deterministic(self) -> None:
        raw = (FIXTURES / "crossing.evidence.json").read_bytes()
        first = subprocess.run(
            [sys.executable, str(CANDIDATE)], input=raw, capture_output=True, check=True,
        )
        second = subprocess.run(
            [sys.executable, str(CANDIDATE)], input=raw, capture_output=True, check=True,
        )
        self.assertEqual(first.stdout, second.stdout)
        self.assertEqual(first.stdout, canonical_json_bytes(json.loads(first.stdout)))
        noncanonical = subprocess.run(
            [sys.executable, str(CANDIDATE)], input=raw.rstrip(), capture_output=True,
        )
        self.assertNotEqual(noncanonical.returncode, 0)
        self.assertEqual(noncanonical.stdout, b"")

    def test_cli_maps_legal_upstream_reject_to_candidate_reject(self) -> None:
        raw = (FIXTURES / "reject.evidence.json").read_bytes()
        result = subprocess.run(
            [sys.executable, str(CANDIDATE)], input=raw, capture_output=True, check=True,
        )
        self.assertEqual(result.stdout, canonical_json_bytes({
            "candidates": [],
            "decode_kind": "Reject",
            "schema_version": "anonymous-candidate-set-v0.4",
        }))

    def test_candidate_dependency_closure_is_stdlib_and_has_no_file_reads(self) -> None:
        source = CANDIDATE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        allowed = {"__future__", "collections", "dataclasses", "json", "sys", "typing"}
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add((node.module or "").split(".", 1)[0])
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self.assertNotIn(node.func.id, {"open", "exec", "eval", "compile", "__import__"})
        self.assertLessEqual(imported, allowed)
        forbidden = (
            "generate_pack_v0_4", "rasterizer_v0_4", "sealed_gt", "support_certificate",
            "opencv", "cv2", "numpy", "png", "canonicalizer", "pack auditor",
        )
        lowered = source.lower()
        for token in forbidden:
            self.assertNotIn(token, lowered)

    def test_schemas_are_closed_for_every_instance_object(self) -> None:
        for name in ("evidence_core.schema.json", "candidate_set.schema.json"):
            schema = json.loads((ROOT / name).read_text(encoding="utf-8"))

            def walk(value: object) -> None:
                if isinstance(value, dict):
                    if value.get("type") == "object":
                        self.assertIs(value.get("additionalProperties"), False, name)
                    for child in value.values():
                        walk(child)
                elif isinstance(value, list):
                    for child in value:
                        walk(child)

            walk(schema)


if __name__ == "__main__":
    unittest.main(verbosity=2)
