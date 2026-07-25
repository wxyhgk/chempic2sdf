import json
import math
from pathlib import Path
import tarfile
import unittest

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from Data.sdf.visual_svg_v0_2 import (
    BondGroupRecordV02,
    LineMemberV02,
    QuadMemberV02,
    TriangleMemberV02,
    visual_primitives_v0_1_to_visual_svg_v0_2,
)
from chemvector.bond_group_carrier_geometry import (
    CARRIER_KIND_TO_ID,
    CARRIER_POINT_WIDTH,
    CARRIER_POSITIVE_WIDTH,
    CARRIER_SIGNED_WIDTH,
    CARRIER_VARIANT_COUNT,
    BondGroupCarrierEncoding,
    carrier_parameter_masks,
    carrier_parameter_variants,
    decode_bond_group_carriers,
    encode_bond_group_carrier,
)


ROOT = Path(__file__).resolve().parents[2]
CAPACITY100_SHARD = (
    ROOT
    / "Data/sdf/visual_svg_v0_1/capacity_100/shards/capacity-000000.tar"
)
PILOT10K_TRAIN_SHARDS = tuple(
    sorted(
        (
            ROOT
            / "Data/sdf/visual_svg_v0_1/pilot_10k/shards"
        ).glob("train-*.tar")
    )
)
PILOT10K_CROSSING_KEYS = {
    "000001591",
    "000003870",
    "000004152",
    "000004735",
    "000005834",
    "000007727",
    "000008400",
}


def _line_cost(left: np.ndarray, right: np.ndarray) -> float:
    direct = max(
        np.linalg.norm(left[0] - right[0]),
        np.linalg.norm(left[1] - right[1]),
    )
    reverse = max(
        np.linalg.norm(left[0] - right[1]),
        np.linalg.norm(left[1] - right[0]),
    )
    return float(min(direct, reverse))


def _set_line_errors(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    costs = np.asarray(
        [[_line_cost(a, b) for b in right] for a in left],
        dtype=np.float64,
    )
    rows, columns = linear_sum_assignment(costs)
    return costs[rows, columns]


def _decoded_lines(encoding: BondGroupCarrierEncoding) -> np.ndarray:
    decoded = decode_bond_group_carriers(
        encoding.carrier_points.unsqueeze(0),
        encoding.positive_params.unsqueeze(0),
        encoding.signed_params.unsqueeze(0),
        encoding.kind_id.unsqueeze(0),
        encoding.member_count.unsqueeze(0),
        encoding.group_stroke.unsqueeze(0),
        max_members=32,
    )
    count = int(encoding.member_count)
    return (
        decoded.member_geometry[0, :count, :4]
        .detach()
        .cpu()
        .numpy()
        .reshape(count, 2, 2)
        * 2047.0
    )


def _rotate_group(
    group: BondGroupRecordV02,
    angle_degrees: float,
) -> BondGroupRecordV02:
    points = []
    for member in group.members:
        if not isinstance(member, LineMemberV02):
            raise TypeError("Rotation helper only supports line members")
        points.extend(((member.x0, member.y0), (member.x1, member.y1)))
    center = np.asarray(points, dtype=np.float64).mean(axis=0)
    angle = math.radians(angle_degrees)
    rotation = np.asarray(
        ((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle))),
        dtype=np.float64,
    )
    members = []
    for member in group.members:
        endpoints = np.asarray(
            ((member.x0, member.y0), (member.x1, member.y1)),
            dtype=np.float64,
        )
        rotated = (endpoints - center) @ rotation.T + center
        members.append(
            LineMemberV02(
                int(round(rotated[0, 0])),
                int(round(rotated[0, 1])),
                int(round(rotated[1, 0])),
                int(round(rotated[1, 1])),
                member.stroke_width,
            )
        )
    return BondGroupRecordV02(group.kind, tuple(members))


class BondGroupCarrierGeometryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.groups = {
            "single_line": BondGroupRecordV02(
                "single_line",
                (LineMemberV02(100, 200, 500, 350, 21),),
            ),
            "parallel_2": BondGroupRecordV02(
                "parallel_2",
                (
                    LineMemberV02(104, 190, 494, 340, 21),
                    LineMemberV02(96, 210, 506, 360, 21),
                ),
            ),
            "parallel_3": BondGroupRecordV02(
                "parallel_3",
                (
                    LineMemberV02(104, 190, 504, 340, 21),
                    LineMemberV02(100, 200, 500, 350, 21),
                    LineMemberV02(96, 210, 496, 360, 21),
                ),
            ),
            "hashed_wedge": BondGroupRecordV02(
                "hashed_wedge",
                tuple(
                    LineMemberV02(
                        200 + index * 40 - (3 + index * 3),
                        180 + index * 30 + (4 + index * 4),
                        200 + index * 40 + (3 + index * 3),
                        180 + index * 30 - (4 + index * 4),
                        11,
                    )
                    for index in range(8)
                ),
            ),
            "solid_triangle": BondGroupRecordV02(
                "solid_triangle",
                (TriangleMemberV02(100, 100, 180, 220, 80, 250, 7),),
            ),
            "solid_quad": BondGroupRecordV02(
                "solid_quad",
                (QuadMemberV02(100, 100, 180, 120, 170, 240, 80, 220, 7),),
            ),
        }

    def test_encoding_shapes_and_active_masks(self) -> None:
        expected = {
            "single_line": (4, 0, 0, 1),
            "parallel_2": (8, 0, 0, 2),
            "parallel_3": (4, 1, 0, 3),
            "hashed_wedge": (4, 2, 0, 8),
            "solid_triangle": (6, 0, 0, 1),
            "solid_quad": (8, 0, 0, 1),
        }
        for kind, group in self.groups.items():
            with self.subTest(kind=kind):
                encoded = encode_bond_group_carrier(group)
                self.assertEqual(encoded.carrier_points.shape, (CARRIER_POINT_WIDTH,))
                self.assertEqual(
                    encoded.positive_params.shape,
                    (CARRIER_POSITIVE_WIDTH,),
                )
                self.assertEqual(encoded.signed_params.shape, (CARRIER_SIGNED_WIDTH,))
                counts = (
                    int(encoded.carrier_point_mask.sum()),
                    int(encoded.positive_param_mask.sum()),
                    int(encoded.signed_param_mask.sum()),
                    int(encoded.member_count),
                )
                self.assertEqual(counts, expected[kind])
                self.assertEqual(int(encoded.kind_id), CARRIER_KIND_TO_ID[kind])
                self.assertAlmostEqual(
                    float(encoded.group_stroke),
                    group.members[0].stroke_width / 63.0,
                )

    def test_masks_can_be_built_for_a_tensor_batch(self) -> None:
        kind_ids = torch.tensor(
            [CARRIER_KIND_TO_ID[kind] for kind in self.groups],
            dtype=torch.long,
        )
        point, positive, signed = carrier_parameter_masks(kind_ids)
        self.assertEqual(point.shape, (6, CARRIER_POINT_WIDTH))
        self.assertEqual(positive.shape, (6, CARRIER_POSITIVE_WIDTH))
        self.assertEqual(signed.shape, (6, CARRIER_SIGNED_WIDTH))
        self.assertEqual(point.sum(dim=-1).tolist(), [4, 8, 4, 4, 6, 8])
        self.assertEqual(positive.sum(dim=-1).tolist(), [0, 0, 1, 2, 0, 0])
        self.assertEqual(signed.sum(dim=-1).tolist(), [0, 0, 0, 0, 0, 0])

    def test_encode_decode_roundtrip_for_synthetic_groups(self) -> None:
        for kind, group in self.groups.items():
            with self.subTest(kind=kind):
                encoded = encode_bond_group_carrier(group)
                decoded = decode_bond_group_carriers(
                    encoded.carrier_points.unsqueeze(0),
                    encoded.positive_params.unsqueeze(0),
                    encoded.signed_params.unsqueeze(0),
                    encoded.kind_id.unsqueeze(0),
                    encoded.member_count.unsqueeze(0),
                    encoded.group_stroke.unsqueeze(0),
                    max_members=32,
                )
                self.assertEqual(int(decoded.member_mask.sum()), len(group.members))
                if isinstance(group.members[0], LineMemberV02):
                    expected = np.asarray(
                        [
                            ((m.x0, m.y0), (m.x1, m.y1))
                            for m in group.members
                        ],
                        dtype=np.float64,
                    )
                    self.assertLessEqual(
                        float(_set_line_errors(_decoded_lines(encoded), expected).max()),
                        2.0,
                    )
                else:
                    coordinate_count = 6 if kind == "solid_triangle" else 8
                    source = group.members[0]
                    coordinates = (
                        (
                            source.x0,
                            source.y0,
                            source.x1,
                            source.y1,
                            source.x2,
                            source.y2,
                        )
                        if isinstance(source, TriangleMemberV02)
                        else (
                            source.x0,
                            source.y0,
                            source.x1,
                            source.y1,
                            source.x2,
                            source.y2,
                            source.x3,
                            source.y3,
                        )
                    )
                    expected = torch.tensor(
                        [value / 2047.0 for value in coordinates[:coordinate_count]],
                        dtype=torch.float32,
                    )
                    self.assertTrue(
                        torch.allclose(
                            decoded.member_geometry[0, 0, :coordinate_count],
                            expected,
                        )
                    )

    def test_encode_is_invariant_to_member_and_endpoint_reversal(self) -> None:
        for kind in ("parallel_2", "parallel_3", "hashed_wedge"):
            group = self.groups[kind]
            reversed_group = BondGroupRecordV02(
                kind,
                tuple(
                    LineMemberV02(
                        member.x1,
                        member.y1,
                        member.x0,
                        member.y0,
                        member.stroke_width,
                    )
                    for member in reversed(group.members)
                ),
            )
            direct = encode_bond_group_carrier(group)
            reversed_encoding = encode_bond_group_carrier(reversed_group)
            with self.subTest(kind=kind):
                self.assertTrue(
                    torch.allclose(
                        direct.carrier_points,
                        reversed_encoding.carrier_points,
                        atol=1e-7,
                    )
                )
                self.assertTrue(
                    torch.allclose(
                        direct.positive_params,
                        reversed_encoding.positive_params,
                        atol=1e-7,
                    )
                )
                self.assertTrue(
                    torch.allclose(
                        direct.signed_params,
                        reversed_encoding.signed_params,
                        atol=1e-7,
                    )
                )

    def test_direct_and_reversed_carrier_variants_decode_to_same_sets(self) -> None:
        for kind in ("single_line", "parallel_2", "parallel_3", "hashed_wedge"):
            encoded = encode_bond_group_carrier(self.groups[kind])
            variants = carrier_parameter_variants(
                encoded.carrier_points.unsqueeze(0),
                encoded.positive_params.unsqueeze(0),
                encoded.signed_params.unsqueeze(0),
                encoded.kind_id.unsqueeze(0),
            )
            self.assertEqual(variants.carrier_points.shape, (1, CARRIER_VARIANT_COUNT, 8))
            expected_variants = 8 if kind == "parallel_2" else 2
            self.assertEqual(int(variants.variant_mask.sum()), expected_variants)
            decoded = decode_bond_group_carriers(
                variants.carrier_points[0, :expected_variants],
                variants.positive_params[0, :expected_variants],
                variants.signed_params[0, :expected_variants],
                encoded.kind_id.expand(expected_variants),
                encoded.member_count.expand(expected_variants),
                encoded.group_stroke.expand(expected_variants),
                max_members=32,
            )
            count = int(encoded.member_count)
            left = (
                decoded.member_geometry[0, :count, :4]
                .detach()
                .numpy()
                .reshape(count, 2, 2)
            )
            with self.subTest(kind=kind):
                for variant_index in range(1, expected_variants):
                    right = (
                        decoded.member_geometry[variant_index, :count, :4]
                        .detach()
                        .numpy()
                        .reshape(count, 2, 2)
                    )
                    self.assertLessEqual(
                        float(_set_line_errors(left, right).max()),
                        1e-6,
                    )

    def test_polygon_variants_cover_all_cycles_and_reversals(self) -> None:
        for kind, expected_variants in (("solid_triangle", 6), ("solid_quad", 8)):
            encoded = encode_bond_group_carrier(self.groups[kind])
            variants = carrier_parameter_variants(
                encoded.carrier_points.unsqueeze(0),
                encoded.positive_params.unsqueeze(0),
                encoded.signed_params.unsqueeze(0),
                encoded.kind_id.unsqueeze(0),
            )
            with self.subTest(kind=kind):
                self.assertEqual(int(variants.variant_mask.sum()), expected_variants)
                active = variants.carrier_points[0, :expected_variants]
                self.assertEqual(
                    len({tuple(row.tolist()) for row in active}),
                    expected_variants,
                )

    def test_decode_is_differentiable_for_active_parameters(self) -> None:
        encodings = [
            encode_bond_group_carrier(self.groups["parallel_2"]),
            encode_bond_group_carrier(self.groups["hashed_wedge"]),
        ]
        points = torch.stack([encoding.carrier_points for encoding in encodings])
        positive = torch.stack([encoding.positive_params for encoding in encodings])
        signed = torch.stack([encoding.signed_params for encoding in encodings])
        points.requires_grad_()
        positive.requires_grad_()
        signed.requires_grad_()
        decoded = decode_bond_group_carriers(
            points,
            positive,
            signed,
            torch.stack([encoding.kind_id for encoding in encodings]),
            torch.stack([encoding.member_count for encoding in encodings]),
            torch.stack([encoding.group_stroke for encoding in encodings]),
            max_members=32,
        )
        loss = decoded.member_geometry[decoded.member_geometry_mask].square().sum()
        loss.backward()
        self.assertTrue(torch.isfinite(points.grad).all())
        self.assertTrue(torch.isfinite(positive.grad).all())
        self.assertGreater(float(points.grad.abs().sum()), 0.0)
        self.assertGreater(float(positive.grad.abs().sum()), 0.0)
        self.assertIsNone(signed.grad)

    def test_rotation_preserves_decoded_physical_sets(self) -> None:
        for kind in ("parallel_2", "parallel_3", "hashed_wedge"):
            original = self.groups[kind]
            rotated = _rotate_group(original, 37.0)
            decoded_rotated = _decoded_lines(encode_bond_group_carrier(rotated))
            expected = np.asarray(
                [
                    ((member.x0, member.y0), (member.x1, member.y1))
                    for member in rotated.members
                ],
                dtype=np.float64,
            )
            with self.subTest(kind=kind):
                self.assertLessEqual(
                    float(_set_line_errors(decoded_rotated, expected).max()),
                    2.0,
                )

    def test_mixed_group_strokes_are_rejected(self) -> None:
        group = BondGroupRecordV02(
            "parallel_2",
            (
                LineMemberV02(100, 100, 300, 100, 10),
                LineMemberV02(100, 120, 300, 120, 11),
            ),
        )
        with self.assertRaisesRegex(ValueError, "stroke"):
            encode_bond_group_carrier(group)

    def test_capacity100_full_roundtrip_stays_below_two_q(self) -> None:
        self.assertTrue(CAPACITY100_SHARD.is_file())
        maximum_error = 0.0
        line_groups = 0
        with tarfile.open(CAPACITY100_SHARD) as archive:
            for member in archive:
                if not member.name.endswith(".primitives.json"):
                    continue
                payload = json.load(archive.extractfile(member))
                parsed = visual_primitives_v0_1_to_visual_svg_v0_2(payload)
                for group in parsed.bond_groups:
                    encoding = encode_bond_group_carrier(group)
                    if not isinstance(group.members[0], LineMemberV02):
                        continue
                    expected = np.asarray(
                        [
                            ((item.x0, item.y0), (item.x1, item.y1))
                            for item in group.members
                        ],
                        dtype=np.float64,
                    )
                    errors = _set_line_errors(_decoded_lines(encoding), expected)
                    maximum_error = max(maximum_error, float(errors.max()))
                    line_groups += 1
        self.assertEqual(line_groups, 3412)
        self.assertLessEqual(maximum_error, 2.0)

    def test_capacity100_parallel_2_is_lossless(self) -> None:
        maximum_error = 0.0
        groups = 0
        with tarfile.open(CAPACITY100_SHARD) as archive:
            for member in archive:
                if not member.name.endswith(".primitives.json"):
                    continue
                payload = json.load(archive.extractfile(member))
                parsed = visual_primitives_v0_1_to_visual_svg_v0_2(payload)
                for group in parsed.bond_groups:
                    if group.kind != "parallel_2":
                        continue
                    encoding = encode_bond_group_carrier(group)
                    expected = np.asarray(
                        [
                            ((item.x0, item.y0), (item.x1, item.y1))
                            for item in group.members
                        ],
                        dtype=np.float64,
                    )
                    maximum_error = max(
                        maximum_error,
                        float(
                            _set_line_errors(_decoded_lines(encoding), expected).max()
                        ),
                    )
                    groups += 1
        self.assertEqual(groups, 868)
        self.assertLessEqual(maximum_error, 1e-3)

    def test_pilot10k_train_parallel_2_codec_is_lossless(self) -> None:
        self.assertTrue(PILOT10K_TRAIN_SHARDS)
        self.assertTrue(
            all("internal_test" not in str(path).lower() for path in PILOT10K_TRAIN_SHARDS)
        )
        seen_keys = set()
        parallel_groups = 0
        maximum_error = 0.0
        for shard in PILOT10K_TRAIN_SHARDS:
            with tarfile.open(shard) as archive:
                for member in archive:
                    if not member.name.endswith(".primitives.json"):
                        continue
                    sample_key = member.name.split(".", 1)[0]
                    payload = json.load(archive.extractfile(member))
                    parsed = visual_primitives_v0_1_to_visual_svg_v0_2(payload)
                    sample_parallel_groups = 0
                    for group in parsed.bond_groups:
                        if group.kind != "parallel_2":
                            continue
                        encoding = encode_bond_group_carrier(group)
                        expected = np.asarray(
                            [
                                ((item.x0, item.y0), (item.x1, item.y1))
                                for item in group.members
                            ],
                            dtype=np.float64,
                        )
                        maximum_error = max(
                            maximum_error,
                            float(
                                _set_line_errors(
                                    _decoded_lines(encoding),
                                    expected,
                                ).max()
                            ),
                        )
                        parallel_groups += 1
                        sample_parallel_groups += 1
                    if sample_key in PILOT10K_CROSSING_KEYS:
                        self.assertGreater(sample_parallel_groups, 0, sample_key)
                        seen_keys.add(sample_key)
        self.assertEqual(seen_keys, PILOT10K_CROSSING_KEYS)
        self.assertGreater(parallel_groups, 0)
        self.assertLessEqual(maximum_error, 1e-3)


if __name__ == "__main__":
    unittest.main()
