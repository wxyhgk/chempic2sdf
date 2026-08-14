#!/usr/bin/env python3
"""Tests for the independent ChemGraph v0.5 domain reference."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

try:
    from .audit_domain_reference_v0_5 import audit
    from .enumerate_domain_reference_v0_5 import (
        AXIS_NAMES,
        ReferenceDomain,
        ReferenceEnumerationError,
        canonical_json_line,
        census_order,
        enumerate_to_directory,
        load_contract,
        merged_canonical_shard_ids,
        pilot_audit,
        root_and_optionally_write_records,
    )
except ImportError:
    from audit_domain_reference_v0_5 import audit  # type: ignore
    from enumerate_domain_reference_v0_5 import (  # type: ignore
        AXIS_NAMES,
        ReferenceDomain,
        ReferenceEnumerationError,
        canonical_json_line,
        census_order,
        enumerate_to_directory,
        load_contract,
        merged_canonical_shard_ids,
        pilot_audit,
        root_and_optionally_write_records,
    )


HERE = Path(__file__).resolve().parent
CONTRACT_PATH = HERE / "prereg" / "p0_contract_v0_5.json"
CONTRACT_SHA256 = "55ddcaacdf851166554ccd586b2a924c0e793f86cc7a6319f98132ef0ffb6b65"


def load_domain() -> ReferenceDomain:
    contract, digest = load_contract(CONTRACT_PATH, CONTRACT_SHA256)
    return ReferenceDomain.from_contract(contract, digest)


class RankUnrankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.domain = load_domain()

    def test_exact_census_and_bounds(self) -> None:
        self.assertEqual(self.domain.radices, (51, 52, 9, 3, 3, 3, 8, 7))
        self.assertEqual(self.domain.declared_count, 36_088_416)
        self.assertEqual(self.domain.tuple_id_minimum, 0)
        self.assertEqual(self.domain.tuple_id_maximum, 36_088_415)

    def test_known_first_and_last_records(self) -> None:
        first = self.domain.record(0)
        self.assertEqual(
            [first[name] for name in AXIS_NAMES],
            [60, 0, 1, "butt", "binary_1x", "none", 0, 0],
        )
        last = self.domain.record(self.domain.declared_count - 1)
        self.assertEqual(
            [last[name] for name in AXIS_NAMES],
            [110, 51, 9, "square", "box_4x4", "y", 7, 6],
        )

    def test_mixed_radix_boundaries_are_bijective(self) -> None:
        witnesses = {0, 1, self.domain.declared_count - 2, self.domain.declared_count - 1}
        stride = 1
        for radix in reversed(self.domain.radices):
            witnesses.update((stride - 1, stride, radix * stride - 1))
            stride *= radix
        for tuple_id in sorted(value for value in witnesses
                               if 0 <= value < self.domain.declared_count):
            self.assertEqual(self.domain.rank(self.domain.unrank(tuple_id)), tuple_id)

    def test_out_of_range_ids_fail_closed(self) -> None:
        for value in (-1, self.domain.declared_count, True):
            with self.assertRaises(ReferenceEnumerationError):
                self.domain.unrank(value)

    def test_record_is_canonical_json_with_one_lf(self) -> None:
        payload = canonical_json_line(self.domain.record(123456))
        self.assertTrue(payload.endswith(b"\n"))
        self.assertFalse(payload.endswith(b"\n\n"))
        self.assertEqual(canonical_json_line(json.loads(payload)), payload)


class EnumerationOrderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.domain = load_domain()
        # A small immutable view exercises enumeration mechanics without
        # changing the production contract parsed above.
        # Use a product of 210 while preserving the production eight-axis type.
        self.small = ReferenceDomain(
            self.domain.contract_sha256,
            self.domain.domain_separator,
            self.domain.coverage_separator,
            ((0, 1), (0, 1, 2), (0, 1, 2, 3, 4), (0,), (0,), (0,),
             (0, 1, 2, 3, 4, 5, 6), (0,)),
            210,
            0,
            209,
        )

    def test_each_order_is_a_bijection_for_every_shard(self) -> None:
        for shard_count in (1, 2, 7, 211):
            for shard_index in range(shard_count):
                expected = list(self.small.canonical_ids(shard_index, shard_count))
                for order in ("forward", "reverse", "shuffled"):
                    actual = list(self.small.ordered_ids(order, shard_index, shard_count))
                    self.assertEqual(sorted(actual), expected)
                    self.assertEqual(len(actual), len(set(actual)))

    def test_root_is_order_invariant_after_canonical_reduction(self) -> None:
        expected, _, _, _ = root_and_optionally_write_records(
            self.small, self.small.canonical_ids(), None
        )
        for order in ("forward", "reverse", "shuffled"):
            visited = list(self.small.ordered_ids(order))
            actual, _, _, _ = root_and_optionally_write_records(
                self.small, sorted(visited), None
            )
            self.assertEqual(actual, expected)

    def test_shards_partition_the_domain(self) -> None:
        for shard_count in (2, 7, 211):
            union = []
            for shard_index in range(shard_count):
                union.extend(self.small.canonical_ids(shard_index, shard_count))
            self.assertEqual(sorted(union), list(range(self.small.declared_count)))
            self.assertEqual(len(union), len(set(union)))

    def test_shard_merge_restores_global_root(self) -> None:
        expected, _, _, _ = root_and_optionally_write_records(
            self.small, self.small.canonical_ids(), None
        )
        for shard_count in (1, 2, 7, 211):
            merged_ids = merged_canonical_shard_ids(self.small, shard_count)
            actual, count, minimum, maximum = root_and_optionally_write_records(
                self.small, merged_ids, None
            )
            self.assertEqual(actual, expected)
            self.assertEqual((count, minimum, maximum), (210, 0, 209))

    def test_census_checks_rank_unrank_and_coverage(self) -> None:
        for order in ("forward", "reverse", "shuffled"):
            count, coverage = census_order(self.small, order, 0, 1)
            self.assertEqual(count, 210)
            for name, axis in zip(AXIS_NAMES, self.small.axes, strict=True):
                self.assertEqual(coverage[name], list(axis))


class FrozenContractAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.domain = load_domain()

    def test_contract_hash_is_checked(self) -> None:
        with self.assertRaises(ReferenceEnumerationError):
            load_contract(CONTRACT_PATH, "0" * 64)

    def test_pilot_count_and_all_axis_values(self) -> None:
        result = pilot_audit(self.domain)
        self.assertEqual(result["record_count"], 360_885)
        self.assertEqual(result["status"], "passed")
        self.assertTrue(all(not missing for missing in result["missing"].values()))
        for name, axis in zip(AXIS_NAMES, self.domain.axes, strict=True):
            self.assertEqual(result["coverage"][name], list(axis))

    def test_reference_audit_passes(self) -> None:
        result = audit(CONTRACT_PATH, CONTRACT_SHA256)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["calculated_tuple_count"], 36_088_416)

    def test_audit_only_manifests_are_order_invariant(self) -> None:
        # Use the production domain with many shards so each exact census is
        # small while still checking real contract records and roots.
        shard_count = 100_000
        shard_index = 99_999
        manifests = []
        with tempfile.TemporaryDirectory(prefix="chemgraph-v05-reference-") as temporary:
            root = Path(temporary)
            for order in ("forward", "reverse", "shuffled"):
                manifests.append(enumerate_to_directory(
                    self.domain, root / order, order, shard_index, shard_count,
                    write_records=False,
                ))
        self.assertEqual({item["records_root"] for item in manifests},
                         {manifests[0]["records_root"]})
        self.assertEqual({item["record_count"] for item in manifests},
                         {manifests[0]["record_count"]})


if __name__ == "__main__":
    unittest.main(verbosity=2)
