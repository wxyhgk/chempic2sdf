#!/usr/bin/env python3
"""Enumerate the frozen v0.5 quotient tuple domain without rendering.

This process is deliberately limited to integer tuple decoding, coverage,
canonical record framing, and optional JSONL emission.  It does not import the
generator, rasterizer, extractor, candidate, or any training dependency.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable, Iterator


FROZEN_CONTRACT_SHA256 = "55ddcaacdf851166554ccd586b2a924c0e793f86cc7a6319f98132ef0ffb6b65"
TUPLE_SCHEMA_VERSION = "chemgraph-domain-tuple-v0.5"
MANIFEST_SCHEMA_VERSION = "chemgraph-domain-census-manifest-v0.5"
DOMAIN_SEPARATOR = b"chemgraph-v0.5/domain-tuple"
COVERAGE_SEPARATOR = b"chemgraph-v0.5/coverage-bitset"
TOTAL_TUPLES = 36_088_416
FIXED_WORKERS = 16
AXES = (
    ("half_length", tuple(range(60, 111))),
    ("motif_ordinal", tuple(range(52))),
    ("width_px", tuple(range(1, 10))),
    ("cap", ("butt", "round", "square")),
    ("antialias_profile", ("binary_1x", "box_2x2", "box_4x4")),
    ("reflection", ("none", "x", "y")),
    ("rotation_index", tuple(range(8))),
    ("scale_index", tuple(range(7))),
)
RADICES = tuple(len(values) for _, values in AXES)


class DomainEnumerationError(ValueError):
    """The contract or enumeration request violates the frozen P0 domain."""


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DomainEnumerationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")


def framed_record(domain: bytes, record: bytes) -> bytes:
    return (
        len(domain).to_bytes(4, "big")
        + domain
        + len(record).to_bytes(8, "big")
        + record
    )


def _require_keys(value: Any, expected: set[str], path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DomainEnumerationError(f"{path}: expected object")
    actual = set(value)
    if actual != expected:
        raise DomainEnumerationError(
            f"{path}: field mismatch missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _load_and_validate_contract(path: Path, expected_sha256: str) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if expected_sha256 != FROZEN_CONTRACT_SHA256:
        raise DomainEnumerationError("expected contract SHA256 is not the frozen P0 digest")
    if actual_sha256 != expected_sha256:
        raise DomainEnumerationError(
            f"contract SHA256 mismatch expected={expected_sha256} actual={actual_sha256}"
        )
    try:
        contract = json.loads(raw, object_pairs_hook=_reject_duplicate_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DomainEnumerationError(f"invalid contract JSON: {error}") from error
    root = _require_keys(
        contract,
        {
            "contract_format", "schema_version", "training_authorized", "source_baseline",
            "domain", "motif_rules", "graph", "raster", "roots", "pilot", "pack",
            "commitment",
        },
        "$",
    )
    if root["schema_version"] != "chemgraph-observation-closure-p0-v0.5":
        raise DomainEnumerationError("unsupported contract schema_version")
    if root["training_authorized"] is not False:
        raise DomainEnumerationError("P0 contract must not authorize training")
    domain = _require_keys(
        root["domain"],
        {
            "kind", "quotient_tuple_count", "delivery_translation_count",
            "delivery_lift_count", "axes_fastest_last", "half_length", "motif_ordinal",
            "width_px", "cap", "join", "antialias_profile", "reflection", "rotation",
            "scale", "delivery_translation_pixels", "absolute_translation_pixels",
            "tuple_id_formula", "tuple_id_minimum", "tuple_id_maximum", "validity",
            "official_delivery_render", "translation_reduction", "failure_policy",
        },
        "$/domain",
    )
    expected_axis_names = [name for name, _ in AXES]
    contract_axis_names = [
        "half_length", "motif_ordinal", "width_px", "cap", "antialias_profile",
        "reflection", "rotation", "scale",
    ]
    if domain["axes_fastest_last"] != contract_axis_names:
        raise DomainEnumerationError("domain axis order differs from P0")
    if domain["quotient_tuple_count"] != TOTAL_TUPLES:
        raise DomainEnumerationError("quotient tuple count differs from P0")
    if domain["tuple_id_minimum"] != 0 or domain["tuple_id_maximum"] != TOTAL_TUPLES - 1:
        raise DomainEnumerationError("tuple ID bounds differ from P0")
    if domain["half_length"] != {"minimum": 60, "maximum": 110, "step": 1}:
        raise DomainEnumerationError("half_length axis differs from P0")
    if domain["width_px"] != list(AXES[2][1]):
        raise DomainEnumerationError("width axis differs from P0")
    if domain["cap"] != list(AXES[3][1]):
        raise DomainEnumerationError("cap axis differs from P0")
    if domain["antialias_profile"] != list(AXES[4][1]):
        raise DomainEnumerationError("antialias axis differs from P0")
    if domain["reflection"] != list(AXES[5][1]):
        raise DomainEnumerationError("reflection axis differs from P0")
    if len(domain["motif_ordinal"]) != 12 or len(domain["rotation"]) != 8:
        raise DomainEnumerationError("motif or rotation axis cardinality differs from P0")
    if len(domain["scale"]) != 7:
        raise DomainEnumerationError("scale axis cardinality differs from P0")
    motif_values = []
    for index, item in enumerate(domain["motif_ordinal"]):
        motif = _require_keys(
            item, {"family", "ordinal_begin", "ordinal_end", "variant"},
            f"$/domain/motif_ordinal/{index}",
        )
        begin, end = motif["ordinal_begin"], motif["ordinal_end"]
        if not isinstance(begin, int) or not isinstance(end, int) or begin > end:
            raise DomainEnumerationError("invalid motif ordinal interval")
        motif_values.extend(range(begin, end + 1))
    if motif_values != list(range(52)):
        raise DomainEnumerationError("motif ordinal intervals are not exactly 0..51")
    computed = math.prod(RADICES)
    if computed != TOTAL_TUPLES:
        raise DomainEnumerationError("internal radix product is inconsistent")
    if domain["delivery_translation_count"] != 4_225:
        raise DomainEnumerationError("delivery translation count differs from P0")
    if domain["delivery_lift_count"] != TOTAL_TUPLES * 4_225:
        raise DomainEnumerationError("delivery lift count differs from P0")
    if expected_axis_names != [
        "half_length", "motif_ordinal", "width_px", "cap", "antialias_profile",
        "reflection", "rotation_index", "scale_index",
    ]:
        raise AssertionError("internal axis declaration changed unexpectedly")
    roots = root["roots"]
    if roots.get("domain_separators", {}).get("domain_tuple") != DOMAIN_SEPARATOR.decode("ascii"):
        raise DomainEnumerationError("domain tuple separator differs from P0")
    if roots.get("domain_separators", {}).get("coverage_bitset") != COVERAGE_SEPARATOR.decode("ascii"):
        raise DomainEnumerationError("coverage separator differs from P0")
    pilot = root["pilot"]
    if pilot.get("workers") != FIXED_WORKERS:
        raise DomainEnumerationError("worker count differs from P0")
    return contract, actual_sha256


def decode_tuple_id(tuple_id: int) -> dict[str, Any]:
    if isinstance(tuple_id, bool) or not isinstance(tuple_id, int) or not 0 <= tuple_id < TOTAL_TUPLES:
        raise DomainEnumerationError(f"tuple_id outside 0..{TOTAL_TUPLES - 1}")
    remainder = tuple_id
    indices = [0] * len(AXES)
    for index in range(len(AXES) - 1, -1, -1):
        remainder, indices[index] = divmod(remainder, RADICES[index])
    if remainder:
        raise AssertionError("mixed-radix decode left a quotient")
    values = [axis_values[axis_index]
              for (_, axis_values), axis_index in zip(AXES, indices)]
    return {
        "schema_version": TUPLE_SCHEMA_VERSION,
        "tuple_id": tuple_id,
        "half_length": values[0],
        "motif_ordinal": values[1],
        "width_px": values[2],
        "cap": values[3],
        "antialias_profile": values[4],
        "reflection": values[5],
        "rotation_index": values[6],
        "scale_index": values[7],
    }


def encode_tuple_record(record: dict[str, Any]) -> int:
    expected = {
        "schema_version", "tuple_id", "half_length", "motif_ordinal", "width_px",
        "cap", "antialias_profile", "reflection", "rotation_index", "scale_index",
    }
    _require_keys(record, expected, "tuple")
    if record["schema_version"] != TUPLE_SCHEMA_VERSION:
        raise DomainEnumerationError("tuple schema_version mismatch")
    values = [record[name] for name, _ in AXES]
    encoded = 0
    for value, (_, allowed) in zip(values, AXES):
        try:
            axis_index = allowed.index(value)
        except ValueError as error:
            raise DomainEnumerationError(f"tuple axis value outside contract: {value!r}") from error
        encoded = encoded * len(allowed) + axis_index
    if encoded != record["tuple_id"]:
        raise DomainEnumerationError("tuple_id does not match tuple axis values")
    return encoded


def tuple_record_bytes(tuple_id: int) -> bytes:
    item = decode_tuple_id(tuple_id)
    # Keys are deliberately emitted in canonical sort order without generic JSON overhead.
    return (
        f'{{"antialias_profile":"{item["antialias_profile"]}",'
        f'"cap":"{item["cap"]}",'
        f'"half_length":{item["half_length"]},'
        f'"motif_ordinal":{item["motif_ordinal"]},'
        f'"reflection":"{item["reflection"]}",'
        f'"rotation_index":{item["rotation_index"]},'
        f'"scale_index":{item["scale_index"]},'
        f'"schema_version":"{TUPLE_SCHEMA_VERSION}",'
        f'"tuple_id":{tuple_id},'
        f'"width_px":{item["width_px"]}}}\n'
    ).encode("ascii")


def shard_record_count(shard_index: int, shard_count: int) -> int:
    if shard_count <= 0 or shard_index < 0 or shard_index >= shard_count:
        raise DomainEnumerationError("shard requires 0 <= index < count and count > 0")
    if shard_index >= TOTAL_TUPLES:
        return 0
    return (TOTAL_TUPLES - 1 - shard_index) // shard_count + 1


def canonical_shard_ids(shard_index: int, shard_count: int) -> range:
    return range(shard_index, TOTAL_TUPLES, shard_count)


def _shuffle_parameters(count: int, contract_sha256: str,
                        shard_index: int, shard_count: int) -> tuple[int, int]:
    if count <= 1:
        return 1, 0
    seed = hashlib.sha256(
        bytes.fromhex(contract_sha256)
        + shard_index.to_bytes(8, "big")
        + shard_count.to_bytes(8, "big")
        + b"chemgraph-v0.5/enumerator-a-shuffle"
    ).digest()
    multiplier = int.from_bytes(seed[:8], "big") % count
    if multiplier == 0:
        multiplier = 1
    while math.gcd(multiplier, count) != 1:
        multiplier = (multiplier + 1) % count or 1
    offset = int.from_bytes(seed[8:16], "big") % count
    return multiplier, offset


def traversal_ids(order: str, shard_index: int, shard_count: int,
                  contract_sha256: str) -> Iterator[int]:
    count = shard_record_count(shard_index, shard_count)
    if order == "forward":
        yield from canonical_shard_ids(shard_index, shard_count)
        return
    if order == "reverse":
        for local in range(count - 1, -1, -1):
            yield shard_index + local * shard_count
        return
    if order != "shuffled":
        raise DomainEnumerationError(f"unsupported order: {order}")
    multiplier, offset = _shuffle_parameters(
        count, contract_sha256, shard_index, shard_count,
    )
    for cursor in range(count):
        local = (multiplier * cursor + offset) % count
        yield shard_index + local * shard_count


def empty_coverage() -> OrderedDict[str, set[Any]]:
    return OrderedDict((name, set()) for name, _ in AXES)


def update_coverage(coverage: OrderedDict[str, set[Any]], record: dict[str, Any]) -> None:
    for name in coverage:
        coverage[name].add(record[name])


def coverage_payload(coverage: OrderedDict[str, set[Any]]) -> dict[str, list[Any]]:
    return {name: sorted(values) for name, values in coverage.items()}


def census_shard(order: str, shard_index: int, shard_count: int,
                 contract_sha256: str, records_file: Any | None = None) -> dict[str, Any]:
    coverage = empty_coverage()
    record_count = 0
    tuple_id_min: int | None = None
    tuple_id_max: int | None = None
    for tuple_id in traversal_ids(order, shard_index, shard_count, contract_sha256):
        record = decode_tuple_id(tuple_id)
        update_coverage(coverage, record)
        record_count += 1
        tuple_id_min = tuple_id if tuple_id_min is None else min(tuple_id_min, tuple_id)
        tuple_id_max = tuple_id if tuple_id_max is None else max(tuple_id_max, tuple_id)
        if records_file is not None:
            records_file.write(tuple_record_bytes(tuple_id))
    expected_count = shard_record_count(shard_index, shard_count)
    if record_count != expected_count:
        raise DomainEnumerationError("traversal did not cover the shard exactly once")

    records_hasher = hashlib.sha256()
    for tuple_id in canonical_shard_ids(shard_index, shard_count):
        records_hasher.update(framed_record(DOMAIN_SEPARATOR, tuple_record_bytes(tuple_id)))
    coverage_value = coverage_payload(coverage)
    coverage_bytes = canonical_json_bytes(coverage_value)
    coverage_root = hashlib.sha256(
        framed_record(COVERAGE_SEPARATOR, coverage_bytes)
    ).hexdigest()
    return {
        "record_count": record_count,
        "expected_record_count": expected_count,
        "tuple_id_min": tuple_id_min,
        "tuple_id_max": tuple_id_max,
        "coverage": coverage_value,
        "records_root": records_hasher.hexdigest(),
        "coverage_root": coverage_root,
    }


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "byte_length": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.workers != FIXED_WORKERS:
        raise DomainEnumerationError(f"--workers must equal frozen value {FIXED_WORKERS}")
    contract_path = args.contract.resolve()
    _contract, contract_sha256 = _load_and_validate_contract(
        contract_path, args.expected_contract_sha256,
    )
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise DomainEnumerationError(f"refusing to overwrite non-empty output: {output}")
    if output.exists():
        output.rmdir()
    stage = output.with_name(f".{output.name}.stage-{os.getpid()}")
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    records_path = stage / "domain_tuples.jsonl"
    records_file = records_path.open("wb") if args.mode == "records" else None
    try:
        census = census_shard(
            args.order, args.shard_index, args.shard_count, contract_sha256, records_file,
        )
        if records_file is not None:
            records_file.flush()
            os.fsync(records_file.fileno())
    finally:
        if records_file is not None:
            records_file.close()
    manifest = {
        "contract_sha256": contract_sha256,
        "coverage": census["coverage"],
        "coverage_root": census["coverage_root"],
        "expected_record_count": census["expected_record_count"],
        "mode": args.mode,
        "order": args.order,
        "record_count": census["record_count"],
        "records_artifact": (_artifact(records_path, stage)
                             if args.mode == "records" else None),
        "records_root": census["records_root"],
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "status": "passed",
        "tuple_id_max": census["tuple_id_max"],
        "tuple_id_min": census["tuple_id_min"],
        "workers": args.workers,
    }
    manifest_path = stage / "manifest.json"
    with manifest_path.open("wb") as handle:
        handle.write(canonical_json_bytes(manifest))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(stage, output)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--expected-contract-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=FIXED_WORKERS)
    parser.add_argument("--order", choices=("forward", "reverse", "shuffled"), default="forward")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--mode", choices=("census", "records"), default="census")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        manifest = run(parse_args(argv))
        sys.stdout.buffer.write(canonical_json_bytes(manifest))
        return 0
    except (DomainEnumerationError, OSError) as error:
        print(f"domain enumeration failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
