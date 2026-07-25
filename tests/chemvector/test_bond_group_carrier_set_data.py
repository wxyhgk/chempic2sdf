from __future__ import annotations

import io
import json
from pathlib import Path
import tarfile
import unittest

from PIL import Image
import torch

from Data.sdf.visual_svg_v0_2 import (
    visual_primitives_v0_1_to_visual_svg_v0_2,
    visual_svg_v0_2_to_dict,
)
from chemvector.bond_group_carrier_geometry import CARRIER_KIND_TO_ID
from chemvector.bond_group_carrier_set_data import (
    BondGroupCarrierSetCollator,
    build_bond_group_carrier_set_target,
)
from chemvector.bond_group_set import HASHED_WEDGE, NO_OBJECT, TEXT_BOX


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    ROOT / "tests/data_sdf/fixtures/visual_svg_v0_2/all_group_kinds.json"
)
CAPACITY100_SHARD = (
    ROOT / "Data/sdf/visual_svg_v0_1/capacity_100/shards/capacity-000000.tar"
)
PILOT_SHARDS = {
    split: ROOT
    / f"Data/sdf/visual_svg_v0_1/pilot_10k/shards/{split}-000000.tar"
    for split in ("train", "validation")
}


class BondGroupCarrierSetDataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_encodes_six_group_kinds_text_and_padding(self) -> None:
        target = build_bond_group_carrier_set_target(
            self.payload,
            query_count=8,
            max_members=4,
        )

        self.assertEqual(int(target["object_count"]), 7)
        self.assertEqual(tuple(target["types"].shape), (8,))
        self.assertEqual(tuple(target["carrier_points"].shape), (8, 8))
        self.assertEqual(tuple(target["point_mask"].shape), (8, 8))
        self.assertEqual(tuple(target["positive_params"].shape), (8, 2))
        self.assertEqual(tuple(target["signed_params"].shape), (8, 2))
        self.assertEqual(int(target["target_mask"].sum()), 7)
        self.assertEqual(int(target["types"][-1]), NO_OBJECT)
        self.assertFalse(bool(target["point_mask"][-1].any()))
        self.assertTrue(torch.equal(target["carrier_points"][-1], torch.zeros(8)))

        group_types = target["types"][:6].tolist()
        self.assertEqual(group_types, list(CARRIER_KIND_TO_ID.values()))
        self.assertEqual(
            target["point_mask"][:6].sum(dim=-1).tolist(),
            [4, 8, 4, 4, 6, 8],
        )
        self.assertEqual(
            target["positive_param_mask"][:6].sum(dim=-1).tolist(),
            [0, 0, 1, 2, 0, 0],
        )
        self.assertEqual(
            target["signed_param_mask"][:6].sum(dim=-1).tolist(),
            [0, 0, 0, 0, 0, 0],
        )
        hashed_index = int((target["types"] == HASHED_WEDGE).nonzero()[0])
        self.assertEqual(int(target["member_counts"][hashed_index]), 3)
        text_index = int((target["types"] == TEXT_BOX).nonzero()[0])
        self.assertEqual(int(target["member_counts"][text_index]), 0)
        self.assertGreater(float(target["text_geometry"][text_index, 2]), 0.0)
        self.assertFalse(bool(target["point_mask"][text_index].any()))

    def test_object_member_and_protocol_overflow_are_hard_failures(self) -> None:
        with self.assertRaisesRegex(ValueError, "never truncated"):
            build_bond_group_carrier_set_target(
                self.payload,
                query_count=6,
                max_members=4,
            )
        with self.assertRaisesRegex(ValueError, "never truncated"):
            build_bond_group_carrier_set_target(
                self.payload,
                query_count=8,
                max_members=2,
            )
        with self.assertRaisesRegex(ValueError, "Unsupported carrier target schema"):
            build_bond_group_carrier_set_target({"schema_version": "unknown"})

    def test_v0_1_and_direct_v0_2_sidecars_encode_identically(self) -> None:
        with tarfile.open(CAPACITY100_SHARD, "r") as archive:
            member = next(
                item for item in archive if item.name.endswith(".primitives.json")
            )
            stream = archive.extractfile(member)
            self.assertIsNotNone(stream)
            v0_1 = json.load(stream)
        v0_2 = visual_svg_v0_2_to_dict(
            visual_primitives_v0_1_to_visual_svg_v0_2(v0_1)
        )

        source_target = build_bond_group_carrier_set_target(v0_1)
        direct_target = build_bond_group_carrier_set_target(v0_2)

        self.assertEqual(source_target.keys(), direct_target.keys())
        for field in source_target:
            torch.testing.assert_close(source_target[field], direct_target[field])

    def test_collator_preserves_metadata_and_rejects_internal_test(self) -> None:
        meta = {"source_split": "train"}
        bond_layer = Image.new("L", (16, 16), "white")
        sample = {
            "key": "carrier-sample",
            "image": Image.new("RGB", (16, 16), "black"),
            "bond_layer": bond_layer,
            "visual_svg_v0_2": self.payload,
            "meta": meta,
        }
        collator = BondGroupCarrierSetCollator(
            image_size=16,
            query_count=8,
            max_members=4,
        )

        batch = collator([sample])

        self.assertEqual(tuple(batch["pixel_values"].shape), (1, 3, 16, 16))
        self.assertEqual(tuple(batch["carrier_points"].shape), (1, 8, 8))
        self.assertIs(batch["meta"][0], meta)
        self.assertIs(batch["bond_layers"][0], bond_layer)

        sealed = dict(sample)
        sealed["meta"] = {"source_split": "internal_test"}
        with self.assertRaisesRegex(ValueError, "internal_test"):
            collator([sealed])

    def test_capacity100_all_samples_collate_without_truncation(self) -> None:
        self.assertTrue(CAPACITY100_SHARD.is_file())
        collator = BondGroupCarrierSetCollator(image_size=64)
        pending = []
        seen = 0
        seen_types: set[int] = set()
        with tarfile.open(CAPACITY100_SHARD, "r") as archive:
            primitive_names = [
                member.name
                for member in archive
                if member.name.endswith(".primitives.json")
            ]
            for primitive_name in primitive_names:
                key = primitive_name.removesuffix(".primitives.json")
                self.assertNotIn("internal_test", key.lower())

                def read_bytes(suffix: str) -> bytes:
                    stream = archive.extractfile(f"{key}.{suffix}")
                    self.assertIsNotNone(stream)
                    return stream.read()

                image = Image.open(io.BytesIO(read_bytes("png"))).convert("RGB")
                image.load()
                bond_layer = Image.open(
                    io.BytesIO(read_bytes("bond.png"))
                ).convert("L")
                bond_layer.load()
                sample = {
                    "key": key,
                    "image": image,
                    "bond_layer": bond_layer,
                    "primitives": json.loads(read_bytes("primitives.json")),
                    "meta": json.loads(read_bytes("meta.json")),
                }
                pending.append(sample)
                if len(pending) < 8:
                    continue
                batch = collator(pending)
                seen += len(pending)
                seen_types.update(batch["types"][batch["target_mask"]].tolist())
                self.assertTrue(bool((batch["object_count"] <= 192).all()))
                self.assertTrue(bool((batch["member_counts"] <= 143).all()))
                pending.clear()
        if pending:
            batch = collator(pending)
            seen += len(pending)
            seen_types.update(batch["types"][batch["target_mask"]].tolist())

        self.assertEqual(seen, 100)
        self.assertEqual(
            seen_types,
            {*CARRIER_KIND_TO_ID.values(), TEXT_BOX},
        )

    def test_pilot_train_validation_max_hashed_groups_collate_without_overflow(
        self,
    ) -> None:
        expected_maxima = {"train": 143, "validation": 90}
        collator = BondGroupCarrierSetCollator(image_size=64)
        for split, shard in PILOT_SHARDS.items():
            self.assertTrue(shard.is_file())
            maximum = -1
            maximum_key = None
            with tarfile.open(shard, "r") as archive:
                for member in archive:
                    if not member.name.endswith(".primitives.json"):
                        continue
                    stream = archive.extractfile(member)
                    self.assertIsNotNone(stream)
                    parsed = visual_primitives_v0_1_to_visual_svg_v0_2(
                        json.load(stream)
                    )
                    local_maximum = max(
                        (
                            len(group.members)
                            for group in parsed.bond_groups
                            if group.kind == "hashed_wedge"
                        ),
                        default=-1,
                    )
                    if local_maximum > maximum:
                        maximum = local_maximum
                        maximum_key = member.name.removesuffix(".primitives.json")
                self.assertEqual(maximum, expected_maxima[split])
                self.assertIsNotNone(maximum_key)
                self.assertNotIn("internal_test", maximum_key.lower())

                def read_bytes(suffix: str) -> bytes:
                    stream = archive.extractfile(f"{maximum_key}.{suffix}")
                    self.assertIsNotNone(stream)
                    return stream.read()

                image = Image.open(io.BytesIO(read_bytes("png"))).convert("RGB")
                image.load()
                bond_layer = Image.open(
                    io.BytesIO(read_bytes("bond.png"))
                ).convert("L")
                bond_layer.load()
                sample = {
                    "key": maximum_key,
                    "image": image,
                    "bond_layer": bond_layer,
                    "primitives": json.loads(read_bytes("primitives.json")),
                    "meta": json.loads(read_bytes("meta.json")),
                }
            batch = collator([sample])
            active_counts = batch["member_counts"][batch["target_mask"]]
            self.assertEqual(int(active_counts.max()), maximum)
            self.assertTrue(bool((active_counts <= 143).all()))


if __name__ == "__main__":
    unittest.main()
