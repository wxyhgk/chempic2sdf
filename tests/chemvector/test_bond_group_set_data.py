from __future__ import annotations

import json
from pathlib import Path
import unittest

from PIL import Image

from Data.sdf.visual_svg_v0_2 import (
    BondGroupRecordV02,
    LineMemberV02,
    ParsedVisualSVGSequenceV02,
    visual_svg_v0_2_to_dict,
)
from chemvector.bond_group_set import HASHED_WEDGE, TEXT_BOX
from chemvector.bond_group_set_data import (
    BondGroupSetCollator,
    build_bond_group_set_target,
    target_to_visual_svg_v0_2_json,
)


FIXTURE = (
    Path(__file__).parents[1]
    / "data_sdf/fixtures/visual_svg_v0_2/all_group_kinds.json"
)


class BondGroupSetDataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_encodes_all_group_kinds_and_text_box(self) -> None:
        target = build_bond_group_set_target(
            self.payload,
            query_count=8,
            max_members=4,
        )

        self.assertEqual(int(target["object_count"]), 7)
        self.assertEqual(tuple(target["types"].shape), (8,))
        self.assertEqual(tuple(target["member_geometry"].shape), (8, 4, 8))
        self.assertEqual(tuple(target["member_stroke"].shape), (8, 4))
        self.assertEqual(int((target["types"] == HASHED_WEDGE).sum()), 1)
        hashed_index = int((target["types"] == HASHED_WEDGE).nonzero()[0])
        self.assertEqual(int(target["member_counts"][hashed_index]), 3)
        self.assertEqual(int(target["member_mask"][hashed_index].sum()), 3)
        text_index = int((target["types"] == TEXT_BOX).nonzero()[0])
        self.assertEqual(int(target["member_counts"][text_index]), 0)
        self.assertGreater(float(target["text_geometry"][text_index, 2]), 0.0)
        self.assertEqual(int(target["target_mask"].sum()), 7)

    def test_member_overflow_is_a_hard_failure(self) -> None:
        with self.assertRaisesRegex(ValueError, "never truncated"):
            build_bond_group_set_target(
                self.payload,
                query_count=8,
                max_members=2,
            )

    def test_object_overflow_is_a_hard_failure(self) -> None:
        with self.assertRaisesRegex(ValueError, "query capacity"):
            build_bond_group_set_target(
                self.payload,
                query_count=6,
                max_members=32,
            )

    def test_target_contract_supports_143_members(self) -> None:
        members = tuple(
            LineMemberV02(100, 10 * index, 200, 10 * index, 11)
            for index in range(1, 144)
        )
        payload = visual_svg_v0_2_to_dict(
            ParsedVisualSVGSequenceV02(
                (BondGroupRecordV02("hashed_wedge", members),),
                (),
            )
        )

        target = build_bond_group_set_target(
            payload,
            query_count=2,
            max_members=143,
        )

        self.assertEqual(int(target["member_counts"][0]), 143)
        self.assertEqual(int(target["member_mask"][0].sum()), 143)
        self.assertEqual(tuple(target["member_geometry"].shape), (2, 143, 8))

    def test_collator_rejects_internal_test_from_sample_metadata(self) -> None:
        sample = {
            "key": "sealed",
            "image": Image.new("RGB", (16, 16), "white"),
            "bond_layer": Image.new("L", (16, 16), "white"),
            "primitives": self.payload,
            "meta": {"source_split": "internal_test"},
        }
        collator = BondGroupSetCollator(
            image_size=16,
            query_count=8,
            max_members=4,
        )

        with self.assertRaisesRegex(ValueError, "internal_test"):
            collator([sample])

    def test_collator_uses_the_protocol_conversion_byte_for_byte(self) -> None:
        sequence = target_to_visual_svg_v0_2_json(self.payload)

        self.assertEqual(sequence, FIXTURE.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
