from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

from formal.chemgraph.audit.fragment_root_constructor_v0_4.fragment_root import SCHEMA_VERSION as GEOMETRY_SCHEMA

from .support_partition import (
    MAX_SUPPORT_PIXELS,
    SCHEMA_VERSION,
    SupportPartitionError,
    _decode_runs,
    _rle_sha256,
    compare_declared_partition,
    compile_support_partition,
)
from . import support_partition as support_partition_module


def geometry(*segments: list[list[int]]) -> dict:
    ordered = sorted(segments)
    return {
        "schema_version": GEOMETRY_SCHEMA,
        "training_authorized": False,
        "base_strokes": [
            {"id": index, "boundary_member_ids": [index], "segment_q4": segment}
            for index, segment in enumerate(ordered)
        ],
    }


def pixels_to_runs(pixels: set[tuple[int, int]]) -> list[list[int]]:
    ordered = sorted(pixels)
    runs: list[list[int]] = []
    if not ordered:
        return runs
    row, start = ordered[0]
    end = start + 1
    for next_row, column in ordered[1:]:
        if next_row == row and column == end:
            end += 1
        else:
            runs.append([row, start, end - start])
            row, start, end = next_row, column, column + 1
    runs.append([row, start, end - start])
    return runs


def document(root: dict, supports: list[set[tuple[int, int]]]) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "training_authorized": False,
        "geometry_root": root,
        "parent_supports": [
            {
                "base_stroke_id": index,
                "support_pixel_count": len(pixels),
                "support_runs": (runs := pixels_to_runs(pixels)),
                "support_rle_sha256": _rle_sha256(runs),
            }
            for index, pixels in enumerate(supports)
        ],
    }


def child_sets_by_stroke(value: dict) -> dict[int, list[set[tuple[int, int]]]]:
    result: dict[int, list[set[tuple[int, int]]]] = {}
    for support in value["child_supports"]:
        result.setdefault(support["base_stroke_id"], []).append(
            _decode_runs(support["support_runs"], support["support_pixel_count"])
        )
    return result


def transform_point_q4(point: list[int], operation: str) -> list[int]:
    x, y = point
    if operation == "translate":
        return [x + 16 * 11, y + 16 * 7]
    if operation == "rotate90":
        return [12288 - y, x]
    if operation == "reflect_x":
        return [12288 - x, y]
    raise AssertionError(operation)


def transform_pixel(pixel: tuple[int, int], operation: str) -> tuple[int, int]:
    row, column = pixel
    if operation == "translate":
        return row + 7, column + 11
    if operation == "rotate90":
        return column, 767 - row
    if operation == "reflect_x":
        return row, 767 - column
    raise AssertionError(operation)


def transform_scene(
    segments: list[list[list[int]]],
    supports: list[set[tuple[int, int]]],
    operation: str,
) -> tuple[list[list[list[int]]], list[set[tuple[int, int]]]]:
    transformed = []
    for segment, support in zip(segments, supports, strict=True):
        raw_begin = transform_point_q4(segment[0], operation)
        raw_end = transform_point_q4(segment[1], operation)
        transformed.append((
            sorted([raw_begin, raw_end]),
            {transform_pixel(pixel, operation) for pixel in support},
        ))
    transformed.sort(key=lambda item: item[0])
    return (
        [segment for segment, _ in transformed],
        [support for _, support in transformed],
    )


def scalar_paths(value, prefix=()):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from scalar_paths(child, prefix + (key,))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from scalar_paths(child, prefix + (index,))
    else:
        yield prefix, value


def set_path(value, path, replacement):
    target = value
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = replacement


class SupportPartitionTests(unittest.TestCase):
    def test_uncut_parent_support_is_preserved_exactly(self) -> None:
        pixels = {(20, column) for column in range(10, 31)}
        value = compile_support_partition(document(
            geometry([[168, 328], [488, 328]]), [pixels]
        ))
        self.assertEqual(value["atomic_fragment_count"], 1)
        support = value["child_supports"][0]
        self.assertEqual(_decode_runs(support["support_runs"], support["support_pixel_count"]), pixels)

    def test_x_partitions_each_parent_with_exact_cover(self) -> None:
        horizontal = {(20, column) for column in range(10, 31)}
        vertical = {(row, 20) for row in range(10, 31)}
        value = compile_support_partition(document(geometry(
            [[168, 328], [488, 328]],
            [[328, 168], [328, 488]],
        ), [horizontal, vertical]))
        self.assertEqual(value["atomic_fragment_count"], 4)
        self.assertEqual(value["parent_membership_count"], 42)
        self.assertEqual(value["child_membership_count"], 42)
        self.assertEqual(value["unique_parent_pixel_count"], 41)
        child_sets = [
            _decode_runs(item["support_runs"], item["support_pixel_count"])
            for item in value["child_supports"]
        ]
        self.assertEqual(child_sets[0] | child_sets[1], horizontal)
        self.assertFalse(child_sets[0] & child_sets[1])
        self.assertEqual(child_sets[2] | child_sets[3], vertical)
        self.assertFalse(child_sets[2] & child_sets[3])

    def test_cut_tie_is_assigned_to_right_child(self) -> None:
        horizontal = {(20, column) for column in range(10, 31)}
        vertical = {(row, 20) for row in range(10, 31)}
        value = compile_support_partition(document(geometry(
            [[168, 328], [488, 328]],
            [[328, 168], [328, 488]],
        ), [horizontal, vertical]))
        left, right = value["child_supports"][:2]
        self.assertNotIn((20, 20), _decode_runs(left["support_runs"], left["support_pixel_count"]))
        self.assertIn((20, 20), _decode_runs(right["support_runs"], right["support_pixel_count"]))

    def test_horizontal_vertical_and_diagonal_cut_ties_choose_right(self) -> None:
        segments = [
            [[168, 168], [488, 488]],
            [[168, 328], [488, 328]],
            [[168, 488], [488, 168]],
            [[328, 168], [328, 488]],
        ]
        supports = [
            {(value, value) for value in range(10, 31)},
            {(20, value) for value in range(10, 31)},
            {(40 - value, value) for value in range(10, 31)},
            {(value, 20) for value in range(10, 31)},
        ]
        value = compile_support_partition(document(geometry(*segments), supports))
        by_stroke = child_sets_by_stroke(value)
        self.assertEqual(len(by_stroke), 4)
        for children in by_stroke.values():
            self.assertEqual(len(children), 2)
            self.assertNotIn((20, 20), children[0])
            self.assertIn((20, 20), children[1])

    def test_two_cuts_make_three_half_open_children(self) -> None:
        segments = [
            [[168, 328], [488, 328]],
            [[248, 168], [248, 488]],
            [[408, 168], [408, 488]],
        ]
        supports = [
            {(20, column) for column in range(10, 31)},
            {(row, 15) for row in range(10, 31)},
            {(row, 25) for row in range(10, 31)},
        ]
        root = geometry(*segments)
        value = compile_support_partition(document(root, supports))
        horizontal_id = next(
            stroke["id"]
            for stroke in root["base_strokes"]
            if stroke["segment_q4"] == segments[0]
        )
        children = child_sets_by_stroke(value)[horizontal_id]
        self.assertEqual(len(children), 3)
        self.assertEqual(children[0], {(20, column) for column in range(10, 15)})
        self.assertEqual(children[1], {(20, column) for column in range(15, 25)})
        self.assertEqual(children[2], {(20, column) for column in range(25, 31)})

    def test_endcap_pixels_clamp_to_first_and_last_child(self) -> None:
        horizontal = {(20, column) for column in range(5, 36)}
        vertical = {(row, 20) for row in range(10, 31)}
        value = compile_support_partition(document(geometry(
            [[168, 328], [488, 328]],
            [[328, 168], [328, 488]],
        ), [horizontal, vertical]))
        left, right = value["child_supports"][:2]
        self.assertIn((20, 5), _decode_runs(left["support_runs"], left["support_pixel_count"]))
        self.assertIn((20, 35), _decode_runs(right["support_runs"], right["support_pixel_count"]))

    def test_partition_is_equivariant_under_grid_symmetries(self) -> None:
        segments = [
            [[168, 328], [488, 328]],
            [[328, 168], [328, 488]],
        ]
        supports = [
            {(20, column) for column in range(10, 31)},
            {(row, 20) for row in range(10, 31)},
        ]
        baseline = compile_support_partition(document(geometry(*segments), supports))
        baseline_sizes = sorted(
            len(child)
            for children in child_sets_by_stroke(baseline).values()
            for child in children
        )
        for operation in ("translate", "rotate90", "reflect_x"):
            with self.subTest(operation=operation):
                moved_segments, moved_supports = transform_scene(
                    segments, supports, operation
                )
                moved = compile_support_partition(document(
                    geometry(*moved_segments), moved_supports
                ))
                moved_sizes = sorted(
                    len(child)
                    for children in child_sets_by_stroke(moved).values()
                    for child in children
                )
                self.assertEqual(moved_sizes, baseline_sizes)

    def test_cross_carrier_overlap_is_visible_as_unresolved_scope(self) -> None:
        horizontal = {(20, column) for column in range(10, 31)} | {(0, 0)}
        vertical = {(row, 20) for row in range(10, 31)} | {(0, 0)}
        value = compile_support_partition(document(geometry(
            [[168, 328], [488, 328]],
            [[328, 168], [328, 488]],
        ), [horizontal, vertical]))
        by_stroke = child_sets_by_stroke(value)
        containing_center = sum(
            (20, 20) in child
            for children in by_stroke.values()
            for child in children
        )
        self.assertEqual(containing_center, 2)
        self.assertNotIn("intersection_supports", value)
        self.assertEqual(value["cross_carrier_overlap_pixel_count"], 2)
        self.assertEqual(value["cross_carrier_overlaps"], [
            {"pixel": [0, 0], "base_stroke_ids": [0, 1]},
            {"pixel": [20, 20], "base_stroke_ids": [0, 1]},
        ])

    def test_empty_child_is_closed_failure(self) -> None:
        root = geometry(
            [[168, 328], [488, 328]],
            [[248, 168], [248, 488]],
            [[264, 168], [264, 488]],
        )
        supports = [
            {(20, 10), (20, 30)},
            {(row, 15) for row in range(10, 31)},
            {(row, 16) for row in range(10, 31)},
        ]
        with self.assertRaisesRegex(SupportPartitionError, "EMPTY_FRAGMENT_SUPPORT"):
            compile_support_partition(document(root, supports))

    def test_parent_support_fiber_must_be_exact(self) -> None:
        value = document(geometry([[168, 328], [488, 328]]), [{(20, 10)}])
        value["parent_supports"] = []
        with self.assertRaisesRegex(SupportPartitionError, "PARENT_SUPPORT_FIBER_INCOMPLETE"):
            compile_support_partition(value)

    def test_adjacent_rle_runs_are_noncanonical(self) -> None:
        value = document(geometry([[168, 328], [488, 328]]), [{(20, 10), (20, 11)}])
        value["parent_supports"][0]["support_runs"] = [[20, 10, 1], [20, 11, 1]]
        with self.assertRaisesRegex(SupportPartitionError, "SUPPORT_RUN_NOT_MAXIMAL"):
            compile_support_partition(value)

    def test_parent_rle_hash_is_recomputed(self) -> None:
        value = document(geometry([[168, 328], [488, 328]]), [{(20, 10)}])
        value["parent_supports"][0]["support_rle_sha256"] = "0" * 64
        with self.assertRaisesRegex(SupportPartitionError, "PARENT_SUPPORT_RLE_HASH_MISMATCH"):
            compile_support_partition(value)

    def test_rle_hash_uses_frozen_v03_binary_framing(self) -> None:
        self.assertEqual(
            _rle_sha256([[20, 10, 2]]),
            "2e6345e272dd67336498b6889390318d3f49b6d6afb50735ce0f61c15e272650",
        )

    def test_child_total_run_capacity_is_independent(self) -> None:
        horizontal = {(20, column) for column in range(10, 31)}
        source = document(geometry(
            [[168, 328], [488, 328]],
            [[248, 168], [248, 488]],
            [[408, 168], [408, 488]],
        ), [
            horizontal,
            {(19, 15), (20, 15)},
            {(19, 25), (20, 25)},
        ])
        with patch.object(support_partition_module, "MAX_TOTAL_SUPPORT_RUNS", 6):
            with self.assertRaisesRegex(SupportPartitionError, "CHILD_SUPPORT_RUNS_OUTSIDE_CAPACITY"):
                compile_support_partition(source)

    def test_overlapping_and_reordered_rle_runs_are_noncanonical(self) -> None:
        for runs, error in (
            ([[20, 10, 3], [20, 12, 2]], "SUPPORT_RUN_NOT_MAXIMAL"),
            ([[21, 10, 1], [20, 10, 1]], "SUPPORT_RUN_ORDER_NONCANONICAL"),
        ):
            with self.subTest(runs=runs):
                value = document(geometry([[168, 328], [488, 328]]), [{(20, 10)}])
                value["parent_supports"][0]["support_runs"] = runs
                value["parent_supports"][0]["support_pixel_count"] = sum(
                    run[2] for run in runs
                )
                with self.assertRaisesRegex(SupportPartitionError, error):
                    compile_support_partition(value)

    def test_boolean_cannot_impersonate_rle_integer(self) -> None:
        value = document(geometry([[168, 328], [488, 328]]), [{(20, 10)}])
        value["parent_supports"][0]["support_runs"][0][0] = False
        with self.assertRaisesRegex(SupportPartitionError, "INVALID_SUPPORT_RUN"):
            compile_support_partition(value)

    def test_float_and_numeric_string_cannot_impersonate_integer(self) -> None:
        for replacement in (20.0, "20"):
            with self.subTest(replacement=replacement):
                value = document(geometry([[168, 328], [488, 328]]), [{(20, 10)}])
                value["parent_supports"][0]["support_runs"][0][0] = replacement
                with self.assertRaisesRegex(SupportPartitionError, "INVALID_SUPPORT_RUN"):
                    compile_support_partition(value)

    def test_support_capacity_exact_and_plus_one(self) -> None:
        all_pixels = {
            (row, column)
            for row in range(768)
            for column in range(768)
        }
        source = document(
            geometry([[8, 8], [12280, 12280]]),
            [all_pixels],
        )
        compiled = compile_support_partition(source)
        self.assertEqual(
            compiled["child_supports"][0]["support_pixel_count"],
            MAX_SUPPORT_PIXELS,
        )

        source["parent_supports"][0]["support_pixel_count"] = MAX_SUPPORT_PIXELS + 1
        with self.assertRaisesRegex(SupportPartitionError, "INVALID_SUPPORT_PIXEL_COUNT"):
            compile_support_partition(source)

    def test_deleted_pixel_is_detected_after_count_refresh(self) -> None:
        source = document(geometry([[168, 328], [488, 328]]), [{(20, 10), (20, 11)}])
        declared = compile_support_partition(source)
        declared["child_supports"][0]["support_runs"] = [[20, 10, 1]]
        declared["child_supports"][0]["support_pixel_count"] = 1
        result = compare_declared_partition(source, declared)
        self.assertEqual(result["status"], "SUPPORT_PARTITION_MISMATCH")
        self.assertEqual(result["missing_pixel_count"], 1)

    def test_pixel_moved_between_children_is_detected_both_ways(self) -> None:
        horizontal = {(20, column) for column in range(10, 31)}
        vertical = {(row, 20) for row in range(10, 31)}
        source = document(geometry(
            [[168, 328], [488, 328]],
            [[328, 168], [328, 488]],
        ), [horizontal, vertical])
        declared = compile_support_partition(source)
        left, right = declared["child_supports"][:2]
        left_pixels = _decode_runs(left["support_runs"], left["support_pixel_count"])
        right_pixels = _decode_runs(right["support_runs"], right["support_pixel_count"])
        moved = min(left_pixels)
        left_pixels.remove(moved)
        right_pixels.add(moved)
        left["support_runs"] = pixels_to_runs(left_pixels)
        left["support_pixel_count"] = len(left_pixels)
        right["support_runs"] = pixels_to_runs(right_pixels)
        right["support_pixel_count"] = len(right_pixels)
        result = compare_declared_partition(source, declared)
        self.assertEqual(result["status"], "SUPPORT_PARTITION_MISMATCH")
        self.assertEqual(result["missing_pixel_count"], 1)
        self.assertEqual(result["extra_pixel_count"], 1)

    def test_declared_bool_and_float_impersonation_is_invalid(self) -> None:
        source = document(geometry([[168, 328], [488, 328]]), [{(20, 10)}])
        mutations = [
            lambda value: value.__setitem__("training_authorized", 0),
            lambda value: value.__setitem__("atomic_fragment_count", 1.0),
            lambda value: value["child_supports"][0].__setitem__("atomic_fragment_id", False),
            lambda value: value["child_supports"][0]["support_runs"][0].__setitem__(0, 20.0),
        ]
        for mutate in mutations:
            declared = compile_support_partition(source)
            mutate(declared)
            self.assertEqual(compare_declared_partition(source, declared)["status"], "INVALID")

    def test_every_declared_numeric_leaf_has_strict_native_type(self) -> None:
        source = document(geometry([[168, 328], [488, 328]]), [{(20, 10), (20, 11)}])
        expected = compile_support_partition(source)
        checked = 0
        for path, value in scalar_paths(expected):
            if type(value) is int:
                replacement = float(value)
            elif type(value) is bool:
                replacement = int(value)
            else:
                continue
            declared = copy.deepcopy(expected)
            set_path(declared, path, replacement)
            with self.subTest(path=path):
                self.assertEqual(compare_declared_partition(source, declared)["status"], "INVALID")
            checked += 1
        self.assertGreaterEqual(checked, 9)

    def test_output_is_deterministic(self) -> None:
        source = document(geometry([[168, 328], [488, 328]]), [{(20, 10), (20, 11)}])
        self.assertEqual(compile_support_partition(source), compile_support_partition(copy.deepcopy(source)))

    def test_translation_preserves_child_support_counts(self) -> None:
        horizontal = {(20, column) for column in range(10, 31)}
        vertical = {(row, 20) for row in range(10, 31)}
        source = document(geometry(
            [[168, 328], [488, 328]],
            [[328, 168], [328, 488]],
        ), [horizontal, vertical])
        translated = document(geometry(
            [[184, 344], [504, 344]],
            [[344, 184], [344, 504]],
        ), [
            {(row + 1, column + 1) for row, column in horizontal},
            {(row + 1, column + 1) for row, column in vertical},
        ])
        original_counts = [item["support_pixel_count"] for item in compile_support_partition(source)["child_supports"]]
        translated_counts = [item["support_pixel_count"] for item in compile_support_partition(translated)["child_supports"]]
        self.assertEqual(translated_counts, original_counts)

    def test_axis_swap_preserves_child_support_counts(self) -> None:
        horizontal = {(20, column) for column in range(10, 31)}
        vertical = {(row, 20) for row in range(10, 31)}
        source = document(geometry(
            [[168, 328], [488, 328]],
            [[328, 168], [328, 488]],
        ), [horizontal, vertical])
        swapped = document(geometry(
            [[328, 168], [328, 488]],
            [[168, 328], [488, 328]],
        ), [
            {(column, row) for row, column in vertical},
            {(column, row) for row, column in horizontal},
        ])
        original_counts = [item["support_pixel_count"] for item in compile_support_partition(source)["child_supports"]]
        swapped_counts = [item["support_pixel_count"] for item in compile_support_partition(swapped)["child_supports"]]
        self.assertEqual(swapped_counts, original_counts)


if __name__ == "__main__":
    unittest.main()
