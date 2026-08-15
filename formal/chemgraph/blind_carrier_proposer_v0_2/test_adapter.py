from __future__ import annotations

import base64
import json
import unittest

from formal.chemgraph.audit.blind_carrier_v0_2._common import (
    canonical_grayscale_png,
    canonical_json_bytes,
)
from formal.chemgraph.audit.blind_carrier_v0_2.validate_predictions import (
    validate_prediction_bytes,
)
from formal.chemgraph.blind_carrier_proposer_v0_2 import (
    SemanticBondGroup,
    TerminalHint,
    VisibleSegment,
    propose_prediction,
)


def horizontal_segment(segment_id: int, row: int, start: int = 100, end: int = 500) -> VisibleSegment:
    return VisibleSegment(
        id=segment_id,
        begin_q4=(16 * start + 8, 16 * row + 8),
        end_q4=(16 * end + 8, 16 * row + 8),
        width_q4=16,
        support_runs=((row, start, end - start + 1),),
    )


def vertical_segment(segment_id: int, column: int, start: int, end: int) -> VisibleSegment:
    return VisibleSegment(
        id=segment_id,
        begin_q4=(16 * column + 8, 16 * start + 8),
        end_q4=(16 * column + 8, 16 * end + 8),
        width_q4=16,
        support_runs=tuple((row, column, 1) for row in range(start, end + 1)),
    )


def segment_pixels(segment: VisibleSegment) -> set[tuple[int, int]]:
    return {
        (column, row)
        for row, start, length in segment.support_runs
        for column in range(start, start + length)
    }


def inbox_for(segments: tuple[VisibleSegment, ...]) -> bytes:
    decoded = bytearray([255]) * (768 * 768)
    for segment in segments:
        for column, row in segment_pixels(segment):
            decoded[row * 768 + column] = 0
    png = canonical_grayscale_png(bytes(decoded))
    return canonical_json_bytes({
        "image": {
            "bytes_base64": base64.b64encode(png).decode("ascii"),
            "format": "png",
            "height": 768,
            "width": 768,
        },
        "schema_version": "chemgraph-blind-carrier-input-v0.2",
        "training_authorized": False,
    })


def hint(hint_id: int, point: tuple[int, int]) -> TerminalHint:
    return TerminalHint(hint_id, point)


def status(raw: bytes) -> str:
    return json.loads(raw)["status"]


class ProposerPositiveTests(unittest.TestCase):
    def assert_valid_ok(
        self,
        segments: tuple[VisibleSegment, ...],
        groups: tuple[SemanticBondGroup, ...],
    ) -> dict:
        prediction = propose_prediction(segments, groups)
        self.assertEqual(status(prediction), "ok", prediction.decode("utf-8"))
        self.assertEqual(prediction, propose_prediction(reversed(segments), reversed(groups)))
        report = validate_prediction_bytes(prediction, inbox_for(segments))
        self.assertEqual(report["status"], "STRUCTURALLY_VALID_NOT_SCORED", report)
        return json.loads(prediction)

    def test_single(self) -> None:
        segments = (horizontal_segment(7, 384),)
        groups = (SemanticBondGroup(
            (7,), hint(10, segments[0].begin_q4), hint(20, segments[0].end_q4),
        ),)
        value = self.assert_valid_ok(segments, groups)
        self.assertEqual(value["bond_hypotheses"][0]["order"], 1)

    def test_parallel_double(self) -> None:
        segments = (horizontal_segment(9, 383), horizontal_segment(4, 385))
        groups = (SemanticBondGroup(
            (9, 4), hint(10, (1608, 6152)), hint(20, (8008, 6152)),
        ),)
        value = self.assert_valid_ok(segments, groups)
        self.assertEqual(value["bond_hypotheses"][0]["order"], 2)

    def test_standard_parallel_triple(self) -> None:
        segments = tuple(horizontal_segment(index, row) for index, row in enumerate((381, 384, 387)))
        groups = (SemanticBondGroup(
            (2, 0, 1), hint(10, (1608, 6152)), hint(20, (8008, 6152)),
        ),)
        value = self.assert_valid_ok(segments, groups)
        self.assertEqual(value["terminals"][0]["center_q4"], [1608, 6152])
        self.assertEqual(value["bond_hypotheses"][0]["order"], 3)

    def test_chain_shares_one_terminal(self) -> None:
        left = VisibleSegment(3, (1608, 6152), (4808, 6152), 16, ((384, 100, 201),))
        right = VisibleSegment(8, (8008, 6152), (4808, 6152), 16, ((384, 300, 201),))
        segments = (left, right)
        shared = hint(20, (4808, 6152))
        groups = (
            SemanticBondGroup((3,), hint(10, left.begin_q4), shared),
            SemanticBondGroup((8,), hint(30, right.begin_q4), shared),
        )
        value = self.assert_valid_ok(segments, groups)
        shared_terminal = next(item for item in value["terminals"] if item["id"] == 1)
        self.assertEqual(len(shared_terminal["endpoint_alignment"]), 2)

    def test_fan_shares_one_terminal(self) -> None:
        left = VisibleSegment(0, (4808, 6152), (1608, 6152), 16, ((384, 100, 201),))
        up = vertical_segment(1, 300, 200, 384)
        right = VisibleSegment(2, (4808, 6152), (8008, 6152), 16, ((384, 300, 201),))
        segments = (left, up, right)
        center = hint(10, (4808, 6152))
        groups = (
            SemanticBondGroup((0,), center, hint(20, left.end_q4)),
            SemanticBondGroup((1,), center, hint(30, up.begin_q4)),
            SemanticBondGroup((2,), center, hint(40, right.end_q4)),
        )
        value = self.assert_valid_ok(segments, groups)
        center_terminal = next(
            item for item in value["terminals"] if len(item["endpoint_alignment"]) == 3
        )
        self.assertEqual(center_terminal["center_q4"], [4808, 6152])

    def test_different_bonds_may_cross(self) -> None:
        horizontal = horizontal_segment(5, 384)
        vertical = vertical_segment(2, 300, 200, 568)
        segments = (horizontal, vertical)
        groups = (
            SemanticBondGroup((5,), hint(10, horizontal.begin_q4), hint(20, horizontal.end_q4)),
            SemanticBondGroup((2,), hint(30, vertical.begin_q4), hint(40, vertical.end_q4)),
        )
        value = self.assert_valid_ok(segments, groups)
        self.assertEqual(len(value["bond_hypotheses"]), 2)
        self.assertEqual(segment_pixels(horizontal) & segment_pixels(vertical), {(300, 384)})

    def test_reversed_physical_segment_is_oriented_by_hints(self) -> None:
        segment = VisibleSegment(
            11, (8008, 6152), (1608, 6152), 16, ((384, 100, 401),),
        )
        groups = (SemanticBondGroup(
            (11,), hint(10, (1608, 6152)), hint(20, (8008, 6152)),
        ),)
        value = self.assert_valid_ok((segment,), groups)
        self.assertEqual(value["terminals"][0]["endpoint_alignment"], [
            {"endpoint_index": 1, "segment_id": 0},
        ])

    def test_support_runs_are_canonicalized(self) -> None:
        segment = VisibleSegment(
            0, (1608, 6152), (8008, 6152), 16,
            ((384, 301, 200), (384, 100, 201)),
        )
        groups = (SemanticBondGroup(
            (0,), hint(10, segment.begin_q4), hint(20, segment.end_q4),
        ),)
        value = self.assert_valid_ok((segment,), groups)
        self.assertEqual(value["segments"][0]["support_runs"], [[384, 100, 401]])

    def test_swapping_unordered_group_hints_is_byte_identical(self) -> None:
        segment = horizontal_segment(0, 384)
        left = hint(90, segment.begin_q4)
        right = hint(10, segment.end_q4)
        forward = propose_prediction(
            (segment,), (SemanticBondGroup((0,), left, right),),
        )
        reverse = propose_prediction(
            (segment,), (SemanticBondGroup((0,), right, left),),
        )
        self.assertEqual(forward, reverse)
        self.assertEqual(status(forward), "ok")
        self.assertEqual(
            validate_prediction_bytes(forward, inbox_for((segment,)))["status"],
            "STRUCTURALLY_VALID_NOT_SCORED",
        )

    def test_renaming_all_opaque_hint_ids_is_byte_identical(self) -> None:
        left = VisibleSegment(3, (1608, 6152), (4808, 6152), 16, ((384, 100, 201),))
        right = VisibleSegment(8, (4808, 6152), (8008, 6152), 16, ((384, 300, 201),))
        original = (
            SemanticBondGroup((3,), hint(10, left.begin_q4), hint(20, left.end_q4)),
            SemanticBondGroup((8,), hint(20, right.begin_q4), hint(30, right.end_q4)),
        )
        renamed = (
            SemanticBondGroup((3,), hint(501, left.begin_q4), hint(7, left.end_q4)),
            SemanticBondGroup((8,), hint(7, right.begin_q4), hint(111, right.end_q4)),
        )
        original_prediction = propose_prediction((left, right), original)
        self.assertEqual(original_prediction, propose_prediction((left, right), renamed))
        self.assertEqual(
            validate_prediction_bytes(
                original_prediction, inbox_for((left, right)),
            )["status"],
            "STRUCTURALLY_VALID_NOT_SCORED",
        )

    def test_512_one_segment_lanes_are_valid_at_capacity(self) -> None:
        points = (
            [(64 + 5 * index, 64) for index in range(128)]
            + [(704, 64 + 5 * index) for index in range(128)]
            + [(704 - 5 * index, 704) for index in range(128)]
            + [(64, 704 - 5 * index) for index in range(128)]
        )
        segments = []
        groups = []
        for index, (begin, end) in enumerate(zip(points, points[1:] + points[:1])):
            begin_x, begin_y = begin
            end_x, end_y = end
            if begin_y == end_y:
                support_runs = ((begin_y, min(begin_x, end_x), abs(end_x - begin_x) + 1),)
            else:
                support_runs = tuple(
                    (row, begin_x, 1)
                    for row in range(min(begin_y, end_y), max(begin_y, end_y) + 1)
                )
            segment = VisibleSegment(
                index,
                (16 * begin_x + 8, 16 * begin_y + 8),
                (16 * end_x + 8, 16 * end_y + 8),
                16,
                support_runs,
            )
            segments.append(segment)
            groups.append(SemanticBondGroup(
                (index,),
                hint(index, segment.begin_q4),
                hint((index + 1) % 512, segment.end_q4),
            ))

        prediction = propose_prediction(tuple(segments), tuple(groups))
        value = json.loads(prediction)
        self.assertEqual(value["status"], "ok", value)
        self.assertEqual(len(value["lane_hypotheses"]), 512)
        self.assertEqual(
            validate_prediction_bytes(prediction, inbox_for(tuple(segments)))["status"],
            "STRUCTURALLY_VALID_NOT_SCORED",
        )


class ProposerRejectTests(unittest.TestCase):
    def assert_reject(
        self,
        segments: tuple[VisibleSegment, ...],
        groups: tuple[SemanticBondGroup, ...],
        reason: str,
    ) -> None:
        prediction = propose_prediction(segments, groups)
        value = json.loads(prediction)
        self.assertEqual(value, {
            "reason": reason,
            "schema_version": "chemgraph-blind-carrier-prediction-v0.2",
            "status": "reject",
            "training_authorized": False,
        })
        report = validate_prediction_bytes(prediction, inbox_for(segments))
        self.assertEqual(report["status"], "STRUCTURALLY_VALID_NOT_SCORED", report)

    def test_bad_support_rejects(self) -> None:
        segment = VisibleSegment(0, (1608, 6152), (8008, 6152), 16, ((100, 100, 401),))
        groups = (SemanticBondGroup(
            (0,), hint(10, segment.begin_q4), hint(20, segment.end_q4),
        ),)
        self.assert_reject((segment,), groups, "no_supported_carrier")

    def test_duplicate_segment_membership_rejects(self) -> None:
        segment = horizontal_segment(0, 384)
        groups = (
            SemanticBondGroup((0,), hint(10, segment.begin_q4), hint(20, segment.end_q4)),
            SemanticBondGroup((0,), hint(30, segment.begin_q4), hint(40, segment.end_q4)),
        )
        self.assert_reject((segment,), groups, "unsupported_geometry")

    def test_near_overlay_fake_double_rejects(self) -> None:
        even = tuple((384, column, 1) for column in range(100, 501, 2))
        odd = tuple((384, column, 1) for column in range(101, 501, 2))
        left = VisibleSegment(0, (1608, 6152), (8008, 6152), 16, even)
        right = VisibleSegment(1, (1608, 6153), (8008, 6153), 16, odd)
        groups = (SemanticBondGroup(
            (0, 1), hint(10, (1608, 6152)), hint(20, (8008, 6152)),
        ),)
        self.assert_reject((left, right), groups, "unsupported_geometry")

    def test_disconnected_bundle_endpoint_disks_reject(self) -> None:
        top = horizontal_segment(0, 378)
        bottom = horizontal_segment(1, 390)
        groups = (SemanticBondGroup(
            (0, 1), hint(10, (1608, 6152)), hint(20, (8008, 6152)),
        ),)
        self.assert_reject((top, bottom), groups, "unsupported_geometry")

    def test_same_bundle_support_overlap_rejects(self) -> None:
        first = horizontal_segment(0, 384)
        second = horizontal_segment(1, 384)
        groups = (SemanticBondGroup(
            (0, 1), hint(10, first.begin_q4), hint(20, first.end_q4),
        ),)
        self.assert_reject((first, second), groups, "unsupported_geometry")

    def test_unassigned_segment_rejects(self) -> None:
        first = horizontal_segment(0, 383)
        second = horizontal_segment(1, 385)
        groups = (SemanticBondGroup(
            (0,), hint(10, first.begin_q4), hint(20, first.end_q4),
        ),)
        self.assert_reject((first, second), groups, "unsupported_geometry")

    def test_empty_evidence_is_closed_reject(self) -> None:
        prediction = propose_prediction((), ())
        self.assertEqual(json.loads(prediction)["reason"], "no_supported_carrier")

    def test_orientation_tie_rejects(self) -> None:
        segment = horizontal_segment(0, 384)
        groups = (SemanticBondGroup(
            (0,), hint(10, (4808, 6144)), hint(20, (4808, 6160)),
        ),)
        self.assert_reject((segment,), groups, "unsupported_geometry")

    def test_conflicting_coordinates_for_same_hint_reject(self) -> None:
        left = VisibleSegment(0, (1608, 6152), (4808, 6152), 16, ((384, 100, 201),))
        right = VisibleSegment(1, (4808, 6152), (8008, 6152), 16, ((384, 300, 201),))
        groups = (
            SemanticBondGroup((0,), hint(10, left.begin_q4), hint(20, left.end_q4)),
            SemanticBondGroup((1,), hint(20, (4824, 6152)), hint(30, right.end_q4)),
        )
        self.assert_reject((left, right), groups, "unsupported_geometry")

    def test_bool_is_rejected_in_every_integer_input_role(self) -> None:
        base = horizontal_segment(1, 384)
        cases = (
            (
                (VisibleSegment(True, base.begin_q4, base.end_q4, 16, base.support_runs),),
                (SemanticBondGroup((True,), hint(10, base.begin_q4), hint(20, base.end_q4)),),
                "unsupported_geometry",
            ),
            (
                (VisibleSegment(1, (True, 6152), base.end_q4, 16, base.support_runs),),
                (SemanticBondGroup((1,), hint(10, base.begin_q4), hint(20, base.end_q4)),),
                "unsupported_geometry",
            ),
            (
                (VisibleSegment(1, base.begin_q4, base.end_q4, True, base.support_runs),),
                (SemanticBondGroup((1,), hint(10, base.begin_q4), hint(20, base.end_q4)),),
                "outside_capacity",
            ),
            (
                (VisibleSegment(1, base.begin_q4, base.end_q4, 16, ((True, 100, 401),)),),
                (SemanticBondGroup((1,), hint(10, base.begin_q4), hint(20, base.end_q4)),),
                "unsupported_geometry",
            ),
            (
                (base,),
                (SemanticBondGroup((1,), hint(True, base.begin_q4), hint(20, base.end_q4)),),
                "outside_capacity",
            ),
            (
                (base,),
                (SemanticBondGroup(
                    (1,), hint(10, (True, 6152)), hint(20, base.end_q4),
                ),),
                "outside_capacity",
            ),
            (
                (base,),
                (SemanticBondGroup((True,), hint(10, base.begin_q4), hint(20, base.end_q4)),),
                "unsupported_geometry",
            ),
        )
        for segments, groups, reason in cases:
            with self.subTest(segments=segments, groups=groups):
                self.assert_reject(segments, groups, reason)

    def test_cross_bond_near_overlay_is_rejected_globally(self) -> None:
        even = tuple((384, column, 1) for column in range(100, 501, 2))
        odd = tuple((384, column, 1) for column in range(101, 501, 2))
        first = VisibleSegment(0, (1608, 6152), (8008, 6152), 16, even)
        second = VisibleSegment(1, (1608, 6153), (8008, 6153), 16, odd)
        groups = (
            SemanticBondGroup((0,), hint(10, (1608, 6152)), hint(20, (8008, 6152))),
            SemanticBondGroup((1,), hint(30, (1608, 6153)), hint(40, (8008, 6153))),
        )
        self.assert_reject((first, second), groups, "unsupported_geometry")

    def test_distinct_hint_ids_cannot_name_the_same_point(self) -> None:
        left = VisibleSegment(0, (1608, 6152), (4808, 6152), 16, ((384, 100, 201),))
        right = VisibleSegment(1, (4808, 6152), (8008, 6152), 16, ((384, 300, 201),))
        groups = (
            SemanticBondGroup((0,), hint(10, left.begin_q4), hint(20, left.end_q4)),
            SemanticBondGroup((1,), hint(30, right.begin_q4), hint(40, right.end_q4)),
        )
        self.assert_reject((left, right), groups, "unsupported_geometry")

    def test_terminal_alignment_over_64_is_closed_reject(self) -> None:
        segments = tuple(horizontal_segment(index, 384) for index in range(65))
        shared = hint(0, (1608, 6152))
        groups = tuple(
            SemanticBondGroup(
                (index,), shared, hint(index + 1, (8008, 100 + index)),
            )
            for index in range(65)
        )
        self.assert_reject(segments, groups, "outside_capacity")

    def test_513_one_segment_lanes_are_closed_reject(self) -> None:
        segment = horizontal_segment(0, 384)
        segments = tuple(
            VisibleSegment(
                index,
                segment.begin_q4,
                segment.end_q4,
                segment.width_q4,
                segment.support_runs,
            )
            for index in range(513)
        )
        groups = (SemanticBondGroup(
            (0,), hint(10, segment.begin_q4), hint(20, segment.end_q4),
        ),)
        self.assert_reject(segments, groups, "outside_capacity")

    def test_overlapping_distinct_terminal_regions_are_rejected(self) -> None:
        left = VisibleSegment(
            0, (1608, 6152), (4808, 6152), 16, ((384, 100, 201),),
        )
        up = vertical_segment(1, 300, 184, 384)
        groups = (
            SemanticBondGroup(
                (0,), hint(10, left.begin_q4), hint(20, left.end_q4),
            ),
            SemanticBondGroup(
                (1,), hint(30, (4809, 6152)), hint(40, up.begin_q4),
            ),
        )
        self.assert_reject((left, up), groups, "unsupported_geometry")

    def test_unexpected_iterable_failure_propagates_to_runner(self) -> None:
        class BrokenIterable:
            def __iter__(self):
                raise RuntimeError("broken evidence stream")

        with self.assertRaisesRegex(RuntimeError, "broken evidence stream"):
            propose_prediction(BrokenIterable(), ())


if __name__ == "__main__":
    unittest.main()
