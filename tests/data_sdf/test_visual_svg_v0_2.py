import copy
import json
from pathlib import Path
import tarfile
import unittest

try:
    import fastjsonschema
except ImportError:  # The training-only environment intentionally omits it.
    fastjsonschema = None

from Data.sdf.visual_svg_sequence_v0_1 import parsed_visual_svg_from_primitives
from Data.sdf.visual_svg_v0_2 import (
    GROUP_KINDS,
    BondGroupRecordV02,
    LineMemberV02,
    ParsedVisualSVGSequenceV02,
    VisualSVGV02Error,
    flatten_visual_svg_v0_2_to_v0_1,
    parse_visual_svg_v0_2,
    serialize_parsed_visual_svg_v0_2,
    visual_primitives_v0_1_to_visual_svg_v0_2,
    visual_svg_v0_2_to_dict,
)


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).with_name("fixtures") / "visual_svg_v0_2"
SCHEMA = (
    ROOT
    / "Data/sdf/visual_svg_v0_2/schemas/visual-svg-0.2.schema.json"
)
CAPACITY_TAR = (
    ROOT
    / "Data/sdf/visual_svg_v0_1/capacity_100/shards/capacity-000000.tar"
)
FORBIDDEN_MODEL_KEYS = {
    "atom_id",
    "begin_atom",
    "bond_id",
    "chemical_type",
    "direction",
    "display_type",
    "end_atom",
    "molecule_id",
    "owner_id",
    "owner_kind",
    "primitive_id",
    "sample_key",
}


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def recursive_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from recursive_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from recursive_keys(child)


class VisualSVGV02ProtocolTest(unittest.TestCase):
    def test_all_group_kinds_fixture_is_canonical(self):
        sequence = fixture("all_group_kinds.json")
        parsed = parse_visual_svg_v0_2(sequence)
        self.assertEqual(serialize_parsed_visual_svg_v0_2(parsed), sequence)
        self.assertEqual(
            tuple(group.kind for group in parsed.bond_groups),
            GROUP_KINDS,
        )
        self.assertEqual(parsed.group_count, 6)
        self.assertEqual(parsed.primitive_count, 12)

    @unittest.skipIf(fastjsonschema is None, "fastjsonschema is not installed")
    def test_fixture_passes_json_schema(self):
        schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
        validate = fastjsonschema.compile(schema)
        validate(json.loads(fixture("all_group_kinds.json")))

    def test_same_flat_primitives_with_different_partition_stay_distinct(self):
        sequence_a = fixture("partition_a.json")
        sequence_b = fixture("partition_b.json")
        parsed_a = parse_visual_svg_v0_2(sequence_a)
        parsed_b = parse_visual_svg_v0_2(sequence_b)

        self.assertNotEqual(sequence_a, sequence_b)
        self.assertNotEqual(parsed_a, parsed_b)
        self.assertEqual(
            flatten_visual_svg_v0_2_to_v0_1(parsed_a),
            flatten_visual_svg_v0_2_to_v0_1(parsed_b),
        )

    def test_noncanonical_group_order_is_rejected(self):
        payload = json.loads(fixture("all_group_kinds.json"))
        payload["bond_groups"].reverse()
        with self.assertRaisesRegex(VisualSVGV02Error, "canonical order"):
            parse_visual_svg_v0_2(json.dumps(payload))

    def test_duplicate_json_fields_are_rejected(self):
        with self.assertRaisesRegex(VisualSVGV02Error, "Duplicate JSON field"):
            parse_visual_svg_v0_2(
                '{"schema_version":"visual-svg-0.2",'
                '"schema_version":"visual-svg-0.2","units":"q2047",'
                '"bond_groups":[],"text_boxes":[]}'
            )

    def test_group_cardinality_is_enforced(self):
        parsed = ParsedVisualSVGSequenceV02(
            (
                BondGroupRecordV02(
                    "hashed_wedge",
                    (LineMemberV02(10, 10, 20, 20, 10),),
                ),
            ),
            (),
        )
        with self.assertRaisesRegex(VisualSVGV02Error, "at least two"):
            serialize_parsed_visual_svg_v0_2(parsed)

    def test_model_target_has_no_source_or_chemical_fields(self):
        payload = json.loads(fixture("all_group_kinds.json"))
        self.assertTrue(FORBIDDEN_MODEL_KEYS.isdisjoint(recursive_keys(payload)))

    def test_capacity_100_structural_roundtrip(self):
        self.assertTrue(CAPACITY_TAR.is_file())
        seen_group_kinds = set()
        samples = 0
        schema_validator = None
        if fastjsonschema is not None:
            schema_validator = fastjsonschema.compile(
                json.loads(SCHEMA.read_text(encoding="utf-8"))
            )
        with tarfile.open(CAPACITY_TAR, "r") as archive:
            for member in archive:
                if not member.name.endswith(".primitives.json"):
                    continue
                self.assertNotIn("internal_test", member.name)
                stream = archive.extractfile(member)
                self.assertIsNotNone(stream)
                source = json.load(stream)
                expected_v0_1 = parsed_visual_svg_from_primitives(source)

                parsed_v0_2 = visual_primitives_v0_1_to_visual_svg_v0_2(source)
                sequence = serialize_parsed_visual_svg_v0_2(parsed_v0_2)
                reparsed = parse_visual_svg_v0_2(sequence)
                flattened = flatten_visual_svg_v0_2_to_v0_1(reparsed)

                self.assertEqual(flattened, expected_v0_1, member.name)
                self.assertEqual(
                    len(parsed_v0_2.bond_groups),
                    len(source["bond_groups"]),
                    member.name,
                )
                self.assertEqual(
                    parsed_v0_2.primitive_count,
                    expected_v0_1.primitive_count,
                    member.name,
                )
                target = visual_svg_v0_2_to_dict(reparsed)
                if schema_validator is not None:
                    schema_validator(target)
                self.assertTrue(
                    FORBIDDEN_MODEL_KEYS.isdisjoint(recursive_keys(target)),
                    member.name,
                )
                seen_group_kinds.update(group.kind for group in reparsed.bond_groups)
                samples += 1

        self.assertEqual(samples, 100)
        self.assertEqual(seen_group_kinds, set(GROUP_KINDS))

    def test_source_bond_ids_and_group_order_do_not_change_target(self):
        with tarfile.open(CAPACITY_TAR, "r") as archive:
            member = next(
                item for item in archive if item.name.endswith(".primitives.json")
            )
            stream = archive.extractfile(member)
            self.assertIsNotNone(stream)
            source = json.load(stream)

        remapped = copy.deepcopy(source)
        source_ids = [int(group["bond_id"]) for group in remapped["bond_groups"]]
        mapping = {
            bond_id: replacement
            for bond_id, replacement in zip(source_ids, reversed(source_ids))
        }
        for primitive in remapped["primitives"]:
            if primitive["owner_kind"] == "bond":
                primitive["owner_id"] = mapping[int(primitive["owner_id"])]
        for group in remapped["bond_groups"]:
            group["bond_id"] = mapping[int(group["bond_id"])]
        remapped["bond_groups"].reverse()

        self.assertEqual(
            serialize_parsed_visual_svg_v0_2(
                visual_primitives_v0_1_to_visual_svg_v0_2(source)
            ),
            serialize_parsed_visual_svg_v0_2(
                visual_primitives_v0_1_to_visual_svg_v0_2(remapped)
            ),
        )


if __name__ == "__main__":
    unittest.main()
