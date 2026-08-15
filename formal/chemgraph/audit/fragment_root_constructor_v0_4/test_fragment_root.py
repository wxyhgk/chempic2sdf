from __future__ import annotations

import copy
import unittest

from .fragment_root import (
    FragmentRootError,
    SCHEMA_VERSION,
    compare_declared_root,
    compile_fragment_root,
)


def root(*segments: list[list[int]]) -> dict:
    ordered = sorted(segments)
    return {
        "schema_version": SCHEMA_VERSION,
        "training_authorized": False,
        "base_strokes": [
            {"id": index, "boundary_member_ids": [index], "segment_q4": segment}
            for index, segment in enumerate(ordered)
        ],
    }


class FragmentRootTests(unittest.TestCase):
    def test_empty_base_stroke_root_is_rejected(self) -> None:
        with self.assertRaisesRegex(FragmentRootError, "BASE_STROKES_EMPTY_OR_INVALID"):
            compile_fragment_root(root())

    def test_exact_x_produces_four_fragments(self) -> None:
        value = compile_fragment_root(root(
            [[160, 320], [480, 320]],
            [[320, 160], [320, 480]],
        ))
        self.assertEqual(value["intersection_count"], 1)
        self.assertEqual(value["split_site_count"], 2)
        self.assertEqual(value["atomic_fragment_count"], 4)
        self.assertEqual([site["t"] for site in value["split_sites"]], [[1, 2], [1, 2]])

    def test_t_touch_does_not_make_zero_length_child(self) -> None:
        value = compile_fragment_root(root(
            [[160, 320], [480, 320]],
            [[320, 160], [320, 320]],
        ))
        self.assertEqual(value["intersection_count"], 1)
        self.assertEqual(value["split_site_count"], 1)
        self.assertEqual(value["atomic_fragment_count"], 3)
        self.assertTrue(all(item["begin_t"] != item["end_t"] for item in value["atomic_fragments"]))
        touching_vertical = next(
            item for item in value["atomic_fragments"]
            if item["base_stroke_id"] == 1
        )
        self.assertEqual(touching_vertical["end_intersection_id"], 0)

    def test_endpoint_to_endpoint_touch_is_retained_on_both_fragments(self) -> None:
        value = compile_fragment_root(root(
            [[160, 320], [320, 320]],
            [[320, 320], [480, 480]],
        ))
        self.assertEqual(value["intersection_count"], 1)
        self.assertEqual(value["split_site_count"], 0)
        self.assertEqual(value["atomic_fragment_count"], 2)
        self.assertEqual(value["atomic_fragments"][0]["end_intersection_id"], 0)
        self.assertEqual(value["atomic_fragments"][1]["begin_intersection_id"], 0)

    def test_collinear_endpoint_touch_is_retained_on_both_fragments(self) -> None:
        value = compile_fragment_root(root(
            [[160, 320], [320, 320]],
            [[320, 320], [480, 320]],
        ))
        self.assertEqual(value["intersection_count"], 1)
        self.assertEqual(value["split_site_count"], 0)
        self.assertEqual(value["atomic_fragments"][0]["end_intersection_id"], 0)
        self.assertEqual(value["atomic_fragments"][1]["begin_intersection_id"], 0)

    def test_double_x_produces_seven_fragments(self) -> None:
        value = compile_fragment_root(root(
            [[160, 320], [640, 320]],
            [[320, 160], [320, 480]],
            [[480, 160], [480, 480]],
        ))
        self.assertEqual(value["intersection_count"], 2)
        self.assertEqual(value["split_site_count"], 4)
        self.assertEqual(value["atomic_fragment_count"], 7)

    def test_three_lines_at_one_point_deduplicate_cuts(self) -> None:
        value = compile_fragment_root(root(
            [[160, 160], [480, 480]],
            [[160, 320], [480, 320]],
            [[160, 480], [480, 160]],
        ))
        self.assertEqual(value["intersection_count"], 1)
        self.assertEqual(value["split_site_count"], 3)
        self.assertEqual(value["atomic_fragment_count"], 6)
        self.assertEqual(value["intersections"][0]["base_stroke_ids"], [0, 1, 2])

    def test_rational_intersection_is_not_rounded(self) -> None:
        value = compile_fragment_root(root(
            [[0, 0], [12, 12]],
            [[0, 8], [12, 0]],
        ))
        self.assertEqual(value["split_sites"][0]["t"], [2, 5])
        self.assertEqual(value["intersections"][0]["point_q4_exact"], [[24, 5], [24, 5]])

    def test_collinear_positive_overlap_is_closed_failure(self) -> None:
        with self.assertRaisesRegex(FragmentRootError, "COLLINEAR_OVERLAP_UNSUPPORTED"):
            compile_fragment_root(root(
                [[160, 320], [480, 320]],
                [[320, 320], [640, 320]],
            ))

    def test_endpoint_reversal_is_rejected(self) -> None:
        document = root([[160, 320], [480, 320]])
        document["base_strokes"][0]["segment_q4"].reverse()
        with self.assertRaisesRegex(FragmentRootError, "SEGMENT_ENDPOINTS_NONCANONICAL"):
            compile_fragment_root(document)

    def test_boolean_cannot_impersonate_id(self) -> None:
        document = root([[160, 320], [480, 320]])
        document["base_strokes"][0]["id"] = False
        with self.assertRaisesRegex(FragmentRootError, "BASE_STROKE_ID_NONCANONICAL"):
            compile_fragment_root(document)

    def test_boolean_cannot_impersonate_coordinate(self) -> None:
        document = root([[160, 320], [480, 320]])
        document["base_strokes"][0]["segment_q4"][0][0] = False
        with self.assertRaisesRegex(FragmentRootError, "INVALID_POINT_Q4"):
            compile_fragment_root(document)

    def test_base_stroke_capacity_is_closed_failure(self) -> None:
        document = {
            "schema_version": SCHEMA_VERSION,
            "training_authorized": False,
            "base_strokes": [{} for _ in range(129)],
        }
        with self.assertRaisesRegex(FragmentRootError, "BASE_STROKES_OUTSIDE_CAPACITY"):
            compile_fragment_root(document)

    def test_input_order_must_be_canonical(self) -> None:
        document = root(
            [[160, 320], [480, 320]],
            [[320, 160], [320, 480]],
        )
        document["base_strokes"].reverse()
        for index, stroke in enumerate(document["base_strokes"]):
            stroke["id"] = index
        with self.assertRaisesRegex(FragmentRootError, "BASE_STROKE_ORDER_NONCANONICAL"):
            compile_fragment_root(document)

    def test_boundary_member_cannot_belong_to_two_base_strokes(self) -> None:
        document = root(
            [[160, 320], [480, 320]],
            [[320, 160], [320, 480]],
        )
        document["base_strokes"][1]["boundary_member_ids"] = [0]
        with self.assertRaisesRegex(FragmentRootError, "BOUNDARY_MEMBER_REUSED"):
            compile_fragment_root(document)

    def test_deleted_fragment_is_detected_after_self_signed_fields_change(self) -> None:
        document = root(
            [[160, 320], [480, 320]],
            [[320, 160], [320, 480]],
        )
        declared = compile_fragment_root(document)
        declared["atomic_fragments"].pop()
        declared["atomic_fragment_count"] -= 1
        result = compare_declared_root(document, declared)
        self.assertEqual(result["status"], "FRAGMENT_UNIVERSE_INCOMPLETE")
        self.assertEqual(result["missing_count"], 1)

    def test_correct_set_in_wrong_order_is_noncanonical(self) -> None:
        document = root(
            [[160, 320], [480, 320]],
            [[320, 160], [320, 480]],
        )
        declared = compile_fragment_root(document)
        declared["atomic_fragments"].reverse()
        result = compare_declared_root(document, declared)
        self.assertEqual(result["status"], "FRAGMENT_UNIVERSE_NONCANONICAL")
        self.assertEqual(result["missing_count"], 0)

    def test_duplicate_fragment_body_is_rejected_explicitly(self) -> None:
        document = root(
            [[160, 320], [480, 320]],
            [[320, 160], [320, 480]],
        )
        declared = compile_fragment_root(document)
        duplicate = copy.deepcopy(declared["atomic_fragments"][0])
        duplicate["id"] = len(declared["atomic_fragments"])
        declared["atomic_fragments"].append(duplicate)
        declared["atomic_fragment_count"] += 1
        result = compare_declared_root(document, declared)
        self.assertEqual(result["status"], "DUPLICATE_FRAGMENT_BODY")

    def test_duplicate_fragment_id_is_rejected_explicitly(self) -> None:
        document = root(
            [[160, 320], [480, 320]],
            [[320, 160], [320, 480]],
        )
        declared = compile_fragment_root(document)
        declared["atomic_fragments"][1]["id"] = 0
        result = compare_declared_root(document, declared)
        self.assertEqual(result["status"], "FRAGMENT_ID_CONFLICT")

    def test_non_json_declared_body_is_closed_invalid(self) -> None:
        document = root([[160, 320], [480, 320]])
        declared = compile_fragment_root(document)
        declared["atomic_fragments"][0]["unexpected"] = object()
        self.assertEqual(compare_declared_root(document, declared)["status"], "INVALID")

    def test_declared_numeric_type_impersonation_is_invalid(self) -> None:
        document = root(
            [[160, 320], [480, 320]],
            [[320, 160], [320, 480]],
        )
        mutations = [
            lambda value: value.__setitem__("atomic_fragment_count", 4.0),
            lambda value: value.__setitem__("training_authorized", 0),
            lambda value: value["intersections"][0].__setitem__("id", False),
            lambda value: value["intersections"][0]["base_stroke_ids"].__setitem__(0, 0.0),
            lambda value: value["intersections"][0]["point_q4_exact"][0].__setitem__(0, 320.0),
            lambda value: value["split_sites"][0].__setitem__("id", False),
            lambda value: value["split_sites"][0].__setitem__("base_stroke_id", 0.0),
            lambda value: value["split_sites"][0]["t"].__setitem__(0, 1.0),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                declared = compile_fragment_root(document)
                mutate(declared)
                result = compare_declared_root(document, declared)
                self.assertEqual(result["status"], "INVALID")
                self.assertFalse(result["exact"])

    def test_output_is_deterministic(self) -> None:
        document = root(
            [[160, 320], [480, 320]],
            [[320, 160], [320, 480]],
        )
        self.assertEqual(compile_fragment_root(document), compile_fragment_root(copy.deepcopy(document)))


if __name__ == "__main__":
    unittest.main()
