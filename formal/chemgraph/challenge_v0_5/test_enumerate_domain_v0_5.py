from __future__ import annotations

import ast
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from formal.chemgraph.challenge_v0_4.validate_schema_bundle import validate_instance
from formal.chemgraph.challenge_v0_5.enumerate_domain_v0_5 import (
    AXES,
    FIXED_WORKERS,
    FROZEN_CONTRACT_SHA256,
    TOTAL_TUPLES,
    canonical_json_bytes,
    census_shard,
    decode_tuple_id,
    encode_tuple_record,
    shard_record_count,
    traversal_ids,
    tuple_record_bytes,
)


ROOT = Path(__file__).resolve().parent
CONTRACT = ROOT / "prereg" / "p0_contract_v0_5.json"
SCHEMAS = ROOT / "schemas"
ENUMERATOR = ROOT / "enumerate_domain_v0_5.py"


class TupleBijectionTests(unittest.TestCase):
    def test_count_bounds_and_mixed_radix_roundtrip(self) -> None:
        self.assertEqual(TOTAL_TUPLES, 36_088_416)
        probes = [0, 1, 6, 7, 100, 1_000_003, 35_408_029, TOTAL_TUPLES - 1]
        for tuple_id in probes:
            with self.subTest(tuple_id=tuple_id):
                record = decode_tuple_id(tuple_id)
                self.assertEqual(encode_tuple_record(record), tuple_id)
                self.assertEqual(tuple_record_bytes(tuple_id), canonical_json_bytes(record))

    def test_frozen_endpoints(self) -> None:
        first = decode_tuple_id(0)
        self.assertEqual(
            {name: first[name] for name, _ in AXES},
            {
                "half_length": 60, "motif_ordinal": 0, "width_px": 1,
                "cap": "butt", "antialias_profile": "binary_1x",
                "reflection": "none", "rotation_index": 0, "scale_index": 0,
            },
        )
        last = decode_tuple_id(TOTAL_TUPLES - 1)
        self.assertEqual(
            {name: last[name] for name, _ in AXES},
            {
                "half_length": 110, "motif_ordinal": 51, "width_px": 9,
                "cap": "square", "antialias_profile": "box_4x4",
                "reflection": "y", "rotation_index": 7, "scale_index": 6,
            },
        )

    def test_shard_partition_counts_sum_to_domain(self) -> None:
        for shard_count in (1, 2, 7, 101, 1_000):
            actual = sum(shard_record_count(index, shard_count)
                         for index in range(shard_count))
            self.assertEqual(actual, TOTAL_TUPLES)


class OrderAndRootTests(unittest.TestCase):
    def test_all_orders_are_permutations_with_identical_canonical_roots(self) -> None:
        shard_count = 3_000_000
        shard_index = 17
        expected_ids = list(range(shard_index, TOTAL_TUPLES, shard_count))
        results = {}
        streams = {}
        for order in ("forward", "reverse", "shuffled"):
            stream = BytesIO()
            result = census_shard(
                order, shard_index, shard_count, FROZEN_CONTRACT_SHA256, stream,
            )
            results[order] = result
            streams[order] = [json.loads(line) for line in stream.getvalue().splitlines()]
            self.assertEqual(
                sorted(item["tuple_id"] for item in streams[order]), expected_ids,
            )
        roots = {value["records_root"] for value in results.values()}
        coverages = {canonical_json_bytes(value["coverage"]) for value in results.values()}
        self.assertEqual(len(roots), 1)
        self.assertEqual(len(coverages), 1)
        self.assertNotEqual(
            [item["tuple_id"] for item in streams["forward"]],
            [item["tuple_id"] for item in streams["reverse"]],
        )

    def test_pilot_selection_covers_every_axis_value(self) -> None:
        coverage = {name: set() for name, _ in AXES}
        count = 0
        for tuple_id in range(0, TOTAL_TUPLES, 100):
            record = decode_tuple_id(tuple_id)
            for name in coverage:
                coverage[name].add(record[name])
            count += 1
        self.assertEqual(count, 360_885)
        for name, allowed in AXES:
            self.assertEqual(coverage[name], set(allowed), name)


class SchemaAndCliTests(unittest.TestCase):
    def test_contract_tuple_and_manifest_validate(self) -> None:
        contract = json.loads(CONTRACT.read_bytes())
        validate_instance(contract, "p0_contract.schema.json", SCHEMAS)
        validate_instance(decode_tuple_id(0), "domain_tuple.schema.json", SCHEMAS)
        result = census_shard(
            "forward", 0, TOTAL_TUPLES, FROZEN_CONTRACT_SHA256,
        )
        manifest = {
            "contract_sha256": FROZEN_CONTRACT_SHA256,
            "coverage": result["coverage"],
            "coverage_root": result["coverage_root"],
            "expected_record_count": 1,
            "mode": "census",
            "order": "forward",
            "record_count": 1,
            "records_artifact": None,
            "records_root": result["records_root"],
            "schema_version": "chemgraph-domain-census-manifest-v0.5",
            "shard_count": TOTAL_TUPLES,
            "shard_index": 0,
            "status": "passed",
            "tuple_id_max": 0,
            "tuple_id_min": 0,
            "workers": FIXED_WORKERS,
        }
        validate_instance(manifest, "closure_manifest.schema.json", SCHEMAS)

    def test_every_instance_object_schema_is_closed(self) -> None:
        for path in sorted(SCHEMAS.glob("*.schema.json")):
            schema = json.loads(path.read_bytes())

            def walk(value: object) -> None:
                if isinstance(value, dict):
                    if value.get("type") == "object":
                        self.assertIs(value.get("additionalProperties"), False, path.name)
                    choices = value.get("type")
                    if isinstance(choices, list) and "object" in choices:
                        self.assertIs(value.get("additionalProperties"), False, path.name)
                    for child in value.values():
                        walk(child)
                elif isinstance(value, list):
                    for child in value:
                        walk(child)

            walk(schema)

    def test_cli_writes_canonical_records_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "enumeration"
            command = [
                sys.executable, str(ENUMERATOR),
                "--contract", str(CONTRACT),
                "--expected-contract-sha256", FROZEN_CONTRACT_SHA256,
                "--output", str(output),
                "--workers", "16",
                "--order", "shuffled",
                "--shard-index", "0",
                "--shard-count", str(TOTAL_TUPLES),
                "--mode", "records",
            ]
            completed = subprocess.run(command, capture_output=True, check=True)
            manifest_raw = (output / "manifest.json").read_bytes()
            manifest = json.loads(manifest_raw)
            self.assertEqual(manifest_raw, canonical_json_bytes(manifest))
            self.assertEqual(completed.stdout, manifest_raw)
            record_raw = (output / "domain_tuples.jsonl").read_bytes()
            self.assertEqual(record_raw, tuple_record_bytes(0))
            validate_instance(manifest, "closure_manifest.schema.json", SCHEMAS)
            validate_instance(json.loads(record_raw), "domain_tuple.schema.json", SCHEMAS)

    def test_cli_rejects_wrong_frozen_hash_and_worker_count(self) -> None:
        for extra in (
            ["--expected-contract-sha256", "0" * 64, "--workers", "16"],
            ["--expected-contract-sha256", FROZEN_CONTRACT_SHA256, "--workers", "1"],
        ):
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as temporary:
                command = [
                    sys.executable, str(ENUMERATOR), "--contract", str(CONTRACT),
                    "--output", str(Path(temporary) / "out"),
                    "--shard-count", str(TOTAL_TUPLES),
                    *extra,
                ]
                completed = subprocess.run(command, capture_output=True)
                self.assertNotEqual(completed.returncode, 0)
                self.assertEqual(completed.stdout, b"")

    def test_enumerator_dependency_closure_is_stdlib_only(self) -> None:
        tree = ast.parse(ENUMERATOR.read_text(encoding="utf-8"))
        allowed = {
            "__future__", "argparse", "collections", "hashlib", "json", "math", "os",
            "pathlib", "shutil", "sys", "typing",
        }
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add((node.module or "").split(".", 1)[0])
        self.assertLessEqual(imported, allowed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
