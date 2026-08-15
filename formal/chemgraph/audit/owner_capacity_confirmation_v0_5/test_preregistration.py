from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from .build_axis_plan import (
    AXIS_PLAN_BUILDER_PATH,
    CAPACITY_RENDERER_PATH,
    CERTIFICATE_CHECKER_PATH,
    CONSTRUCTOR_PATH,
    LIMITS,
    SOURCE_BUILDER_PATH,
    _canonical_bytes,
)
from .capacity_png import render_support_png
from .certificate_checker import check_certificates
from .source_builder import build_source


ROOT = Path(__file__).resolve().parent


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class CapacityPreregistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan_raw = ROOT.joinpath("axis_plan.json").read_bytes()
        cls.manifest_raw = ROOT.joinpath("source_manifest.json").read_bytes()
        cls.plan = json.loads(cls.plan_raw)
        cls.manifest = json.loads(cls.manifest_raw)

    def test_frozen_documents_are_canonical_and_bound(self) -> None:
        self.assertEqual(self.plan_raw, _canonical_bytes(self.plan))
        self.assertEqual(self.manifest_raw, _canonical_bytes(self.manifest))
        self.assertEqual(
            self.plan["source_manifest_sha256"], _sha256(self.manifest_raw)
        )
        self.assertFalse(self.manifest["constructor_called_by_plan_builder"])
        self.assertTrue(self.manifest["global_preregistration_contaminated"])
        exclusion = self.plan["exploratory_exclusion"]
        self.assertEqual(
            exclusion["sha256"], _sha256((ROOT / "EXPLORATORY_EXCLUSION.md").read_bytes())
        )

    def test_axis_key_set_is_exact_and_categories_are_honest(self) -> None:
        axes = {item["axis"]: item for item in self.plan["axes"]}
        self.assertEqual(set(axes), set(LIMITS))
        self.assertEqual(self.plan["axis_key_set"], sorted(LIMITS))
        self.assertEqual(
            {axis for axis, item in axes.items() if item["boundary"]["tag"] == "reachable"},
            {"max_carriers", "max_intersections", "max_pixel_variables", "max_total_rle_runs"},
        )
        self.assertNotIn("unreachable", {
            item["boundary"]["tag"] for item in axes.values()
        })
        self.assertEqual(
            sum(
                item["boundary"]["tag"] == "conditional_or_unproven_no_go"
                for item in axes.values()
            ),
            8,
        )
        self.assertEqual(
            axes["max_intersections"]["category"], "reachable_rejection_only"
        )
        self.assertEqual(
            self.plan["verdict_if_all_reachable_pass"], "PARTIAL/NO_GO_GENERIC"
        )
        self.assertEqual(
            self.plan["unreachable_certificate_status"],
            "conditional_or_unproven_no_go",
        )

    def test_planned_implementation_hashes_match_current_bytes(self) -> None:
        bindings = (
            ("axis_plan_builder", AXIS_PLAN_BUILDER_PATH),
            ("constructor", CONSTRUCTOR_PATH),
            ("source_builder", SOURCE_BUILDER_PATH),
            ("certificate_checker", CERTIFICATE_CHECKER_PATH),
            ("capacity_support_renderer", CAPACITY_RENDERER_PATH),
        )
        for name, path in bindings:
            with self.subTest(name=name):
                self.assertEqual(self.plan[name]["sha256"], _sha256(path.read_bytes()))

    def test_witnesses_rebuild_byte_exact_without_constructor(self) -> None:
        self.assertEqual(self.manifest["record_count"], 8)
        for record in self.manifest["records"]:
            with self.subTest(axis=record["axis"], side=record["side"]):
                source_path = ROOT / record["source_path"]
                source_raw = source_path.read_bytes()
                self.assertEqual(source_raw, _canonical_bytes(json.loads(source_raw)))
                self.assertEqual(record["canonical_source_sha256"], _sha256(source_raw))
                rebuilt = _canonical_bytes(build_source(record["axis"], record["side"]))
                self.assertEqual(source_raw, rebuilt)

    def test_support_pngs_bind_exact_parent_union(self) -> None:
        for record in self.manifest["records"]:
            with self.subTest(axis=record["axis"], side=record["side"]):
                source = json.loads((ROOT / record["source_path"]).read_bytes())
                expected_png = render_support_png(source["parent_supports"])
                actual_png = (ROOT / record["png_path"]).read_bytes()
                self.assertEqual(actual_png, expected_png)
                self.assertEqual(_sha256(actual_png), source["png_sha256"])
                self.assertTrue(record["png_binding"])
                self.assertTrue(record["renderer_binding"])
                self.assertFalse(record["geometry_renderer_binding"])

    def test_every_high_source_is_isolated_by_frozen_count_vector(self) -> None:
        for record in self.manifest["records"]:
            if record["side"] != "H":
                continue
            target = record["axis"]
            counts = record["count_vector"]
            self.assertGreater(counts[target], LIMITS[target])
            for axis, value in counts.items():
                if axis != target:
                    self.assertLessEqual(value, LIMITS[axis], (target, axis, value))

    def test_pixel_variable_left_has_one_connected_singleton_world(self) -> None:
        record = next(
            item for item in self.manifest["records"]
            if item["axis"] == "max_pixel_variables" and item["side"] == "L"
        )
        source = json.loads((ROOT / record["source_path"]).read_bytes())
        carrier_pixels: dict[int, set[tuple[int, int]]] = {}
        for support in source["parent_supports"]:
            pixels = {
                (row, column)
                for row, start, length in support["support_runs"]
                for column in range(start, start + length)
            }
            carrier_pixels[support["base_stroke_id"]] = pixels
        owner_fibers = (
            (carrier_pixels[0] & carrier_pixels[1], (20, 18)),
            (carrier_pixels[0] & carrier_pixels[2], (20, 22)),
        )
        self.assertEqual(sum(len(fiber) for fiber, _ in owner_fibers), 4096)
        for fiber, seed in owner_fibers:
            self.assertIn(seed, fiber)
            unseen = set(fiber)
            frontier = [seed]
            unseen.remove(seed)
            while frontier:
                row, column = frontier.pop()
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        neighbor = (row + dr, column + dc)
                        if (dr or dc) and neighbor in unseen:
                            unseen.remove(neighbor)
                            frontier.append(neighbor)
            self.assertFalse(unseen)
        self.assertEqual(record["count_vector"]["max_ambiguous_variables"], 0)
        self.assertEqual(record["count_vector"]["max_raw_assignments"], 1)

    def test_conditional_certificates_are_separately_reported(self) -> None:
        certificate = check_certificates()
        self.assertEqual(certificate["status"], "conditional_certificates_computed")
        self.assertFalse(certificate["premises_recomputed"])
        self.assertTrue(certificate["lean_module_hash_bound"])
        self.assertFalse(certificate["implementation_correctness_proved"])
        self.assertFalse(certificate["universal_byte_bound_verified"])
        self.assertLess(
            certificate["axes"]["max_output_bytes"]["upper_bound"],
            LIMITS["max_output_bytes"],
        )

    def test_preregistration_tools_do_not_call_candidate_constructor(self) -> None:
        for name in ("source_builder.py", "build_axis_plan.py", "certificate_checker.py"):
            source = ROOT.joinpath(name).read_text(encoding="utf-8")
            self.assertNotIn("build_" + "candidate_domain", source)


if __name__ == "__main__":
    unittest.main()
