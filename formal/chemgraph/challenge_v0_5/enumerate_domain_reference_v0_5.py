#!/usr/bin/env python3
"""Independent reference enumerator for the ChemGraph v0.5 quotient domain.

This file intentionally does not import the primary enumerator.  It uses a
mixed-radix rank/unrank implementation and an affine permutation for shuffled
enumeration.  Enumeration order is audited first; records and their root are
then emitted in canonical tuple-id order.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import heapq
import json
import math
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


TUPLE_SCHEMA_VERSION = "chemgraph-domain-tuple-v0.5"
MANIFEST_SCHEMA_VERSION = "chemgraph-domain-census-manifest-v0.5"
FROZEN_CONTRACT_SHA256 = "55ddcaacdf851166554ccd586b2a924c0e793f86cc7a6319f98132ef0ffb6b65"
RECORDS_FILENAME = "domain_tuples.jsonl"
MANIFEST_FILENAME = "manifest.json"
ORDERS = ("forward", "reverse", "shuffled")
AXIS_NAMES = (
    "half_length",
    "motif_ordinal",
    "width_px",
    "cap",
    "antialias_profile",
    "reflection",
    "rotation_index",
    "scale_index",
)


class ReferenceEnumerationError(ValueError):
    """The frozen contract or requested enumeration is inconsistent."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReferenceEnumerationError(f"duplicate_json_key:{key}")
        result[key] = value
    return result


def canonical_json_line(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            + "\n").encode("utf-8")


def load_contract(path: Path, expected_sha256: str) -> tuple[dict[str, Any], str]:
    if expected_sha256 != FROZEN_CONTRACT_SHA256:
        raise ReferenceEnumerationError("expected_contract_sha256_is_not_frozen_p0")
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise ReferenceEnumerationError(
            f"contract_sha256_mismatch:expected={expected_sha256}:actual={actual}"
        )
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReferenceEnumerationError(f"invalid_contract_json:{error}") from error
    if not isinstance(value, dict):
        raise ReferenceEnumerationError("contract_root_not_object")
    return value, actual


def _integer_range(spec: Any, name: str) -> tuple[int, ...]:
    if not isinstance(spec, dict):
        raise ReferenceEnumerationError(f"{name}_range_not_object")
    try:
        minimum = spec["minimum"]
        maximum = spec["maximum"]
        step = spec["step"]
    except KeyError as error:
        raise ReferenceEnumerationError(f"{name}_range_missing:{error.args[0]}") from error
    if not all(isinstance(item, int) and not isinstance(item, bool)
               for item in (minimum, maximum, step)):
        raise ReferenceEnumerationError(f"{name}_range_not_integer")
    if step <= 0 or minimum > maximum or (maximum - minimum) % step:
        raise ReferenceEnumerationError(f"{name}_range_invalid")
    return tuple(range(minimum, maximum + 1, step))


def _motif_ordinals(specs: Any) -> tuple[int, ...]:
    if not isinstance(specs, list) or not specs:
        raise ReferenceEnumerationError("motif_ordinal_not_nonempty_array")
    values: list[int] = []
    for index, spec in enumerate(specs):
        if not isinstance(spec, dict):
            raise ReferenceEnumerationError(f"motif_range_not_object:{index}")
        begin = spec.get("ordinal_begin")
        end = spec.get("ordinal_end")
        if (not isinstance(begin, int) or isinstance(begin, bool) or
                not isinstance(end, int) or isinstance(end, bool) or begin > end):
            raise ReferenceEnumerationError(f"motif_range_invalid:{index}")
        values.extend(range(begin, end + 1))
    if values != list(range(len(values))):
        raise ReferenceEnumerationError("motif_ordinals_not_contiguous_zero_based")
    return tuple(values)


def _index_axis(length: int, name: str) -> tuple[int, ...]:
    if length <= 0:
        raise ReferenceEnumerationError(f"{name}_empty")
    return tuple(range(length))


@dataclass(frozen=True)
class ReferenceDomain:
    """Closed mixed-radix domain decoded from the P0 contract."""

    contract_sha256: str
    domain_separator: str
    coverage_separator: str
    axes: tuple[tuple[Any, ...], ...]
    declared_count: int
    tuple_id_minimum: int
    tuple_id_maximum: int

    @classmethod
    def from_contract(cls, contract: dict[str, Any], contract_sha256: str) -> "ReferenceDomain":
        try:
            domain = contract["domain"]
            roots = contract["roots"]
        except KeyError as error:
            raise ReferenceEnumerationError(f"contract_missing:{error.args[0]}") from error
        if domain.get("axes_fastest_last") != [
            "half_length", "motif_ordinal", "width_px", "cap",
            "antialias_profile", "reflection", "rotation", "scale",
        ]:
            raise ReferenceEnumerationError("unexpected_axis_order")

        axes: tuple[tuple[Any, ...], ...] = (
            _integer_range(domain.get("half_length"), "half_length"),
            _motif_ordinals(domain.get("motif_ordinal")),
            _literal_axis(domain.get("width_px"), "width_px", int),
            _literal_axis(domain.get("cap"), "cap", str),
            _literal_axis(domain.get("antialias_profile"), "antialias_profile", str),
            _literal_axis(domain.get("reflection"), "reflection", str),
            _index_axis(_list_length(domain.get("rotation"), "rotation"), "rotation_index"),
            _index_axis(_list_length(domain.get("scale"), "scale"), "scale_index"),
        )
        calculated = math.prod(len(axis) for axis in axes)
        declared = _strict_int(domain.get("quotient_tuple_count"), "quotient_tuple_count")
        minimum = _strict_int(domain.get("tuple_id_minimum"), "tuple_id_minimum")
        maximum = _strict_int(domain.get("tuple_id_maximum"), "tuple_id_maximum")
        if calculated != declared:
            raise ReferenceEnumerationError(
                f"tuple_count_product_mismatch:declared={declared}:calculated={calculated}"
            )
        if minimum != 0 or maximum != declared - 1:
            raise ReferenceEnumerationError("tuple_id_bounds_not_contiguous_zero_based")
        separators = roots.get("domain_separators", {})
        separator = separators.get("domain_tuple")
        coverage_separator = separators.get("coverage_bitset")
        if not isinstance(separator, str) or not separator.isascii() or not separator:
            raise ReferenceEnumerationError("invalid_domain_tuple_separator")
        if (not isinstance(coverage_separator, str) or
                not coverage_separator.isascii() or not coverage_separator):
            raise ReferenceEnumerationError("invalid_coverage_bitset_separator")
        return cls(contract_sha256, separator, coverage_separator, axes,
                   declared, minimum, maximum)

    @property
    def radices(self) -> tuple[int, ...]:
        return tuple(len(axis) for axis in self.axes)

    def unrank(self, tuple_id: int) -> tuple[Any, ...]:
        if not isinstance(tuple_id, int) or isinstance(tuple_id, bool):
            raise ReferenceEnumerationError("tuple_id_not_integer")
        if tuple_id < 0 or tuple_id >= self.declared_count:
            raise ReferenceEnumerationError(f"tuple_id_out_of_range:{tuple_id}")
        remainder = tuple_id
        indices = [0] * len(self.axes)
        for position in range(len(self.axes) - 1, -1, -1):
            remainder, indices[position] = divmod(remainder, len(self.axes[position]))
        if remainder:
            raise AssertionError("mixed-radix unrank left a quotient")
        return tuple(axis[index] for axis, index in zip(self.axes, indices, strict=True))

    def rank(self, values: Sequence[Any]) -> int:
        if len(values) != len(self.axes):
            raise ReferenceEnumerationError("tuple_axis_count_mismatch")
        result = 0
        for name, axis, value in zip(AXIS_NAMES, self.axes, values, strict=True):
            try:
                index = axis.index(value)
            except ValueError as error:
                raise ReferenceEnumerationError(f"axis_value_out_of_domain:{name}:{value!r}") from error
            result = result * len(axis) + index
        return result

    def record(self, tuple_id: int) -> dict[str, Any]:
        values = self.unrank(tuple_id)
        return {
            "antialias_profile": values[4],
            "cap": values[3],
            "half_length": values[0],
            "motif_ordinal": values[1],
            "reflection": values[5],
            "rotation_index": values[6],
            "scale_index": values[7],
            "schema_version": TUPLE_SCHEMA_VERSION,
            "tuple_id": tuple_id,
            "width_px": values[2],
        }

    def framed_record(self, tuple_id: int) -> bytes:
        record = canonical_json_line(self.record(tuple_id))
        domain = self.domain_separator.encode("ascii")
        return (len(domain).to_bytes(4, "big") + domain +
                len(record).to_bytes(8, "big") + record)

    def canonical_ids(self, shard_index: int = 0, shard_count: int = 1) -> range:
        _validate_shard(shard_index, shard_count)
        return range(shard_index, self.declared_count, shard_count)

    def ordered_ids(self, order: str, shard_index: int = 0,
                    shard_count: int = 1) -> Iterator[int]:
        ids = self.canonical_ids(shard_index, shard_count)
        count = len(ids)
        if order == "forward":
            yield from ids
            return
        if order == "reverse":
            yield from reversed(ids)
            return
        if order != "shuffled":
            raise ReferenceEnumerationError(f"unsupported_order:{order}")
        if count == 0:
            return
        multiplier, offset = _affine_permutation_parameters(
            count, self.contract_sha256, shard_index, shard_count
        )
        for position in range(count):
            local_index = (multiplier * position + offset) % count
            yield shard_index + local_index * shard_count


def _strict_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ReferenceEnumerationError(f"{name}_not_integer")
    return value


def _list_length(value: Any, name: str) -> int:
    if not isinstance(value, list) or not value:
        raise ReferenceEnumerationError(f"{name}_not_nonempty_array")
    return len(value)


def _literal_axis(value: Any, name: str, expected_type: type) -> tuple[Any, ...]:
    if not isinstance(value, list) or not value:
        raise ReferenceEnumerationError(f"{name}_not_nonempty_array")
    if any(not isinstance(item, expected_type) or
           (expected_type is int and isinstance(item, bool)) for item in value):
        raise ReferenceEnumerationError(f"{name}_wrong_value_type")
    if len(set(value)) != len(value):
        raise ReferenceEnumerationError(f"{name}_duplicate_value")
    return tuple(value)


def _validate_shard(shard_index: int, shard_count: int) -> None:
    if shard_count <= 0:
        raise ReferenceEnumerationError("shard_count_not_positive")
    if shard_index < 0 or shard_index >= shard_count:
        raise ReferenceEnumerationError("shard_index_out_of_range")


def _affine_permutation_parameters(count: int, contract_sha256: str,
                                   shard_index: int, shard_count: int) -> tuple[int, int]:
    seed = hashlib.sha256(
        bytes.fromhex(contract_sha256) +
        shard_index.to_bytes(8, "big") + shard_count.to_bytes(8, "big")
    ).digest()
    if count == 1:
        return 1, 0
    multiplier = int.from_bytes(seed[:8], "big") % count
    if multiplier == 0:
        multiplier = 1
    while math.gcd(multiplier, count) != 1:
        multiplier += 1
        if multiplier == count:
            multiplier = 1
    offset = int.from_bytes(seed[8:16], "big") % count
    return multiplier, offset


def _coverage_template() -> dict[str, set[Any]]:
    return {name: set() for name in AXIS_NAMES}


def _add_coverage(coverage: dict[str, set[Any]], values: Sequence[Any]) -> None:
    for name, value in zip(AXIS_NAMES, values, strict=True):
        coverage[name].add(value)


def _ordered_coverage(coverage: dict[str, set[Any]], domain: ReferenceDomain) -> dict[str, list[Any]]:
    result: dict[str, list[Any]] = {}
    for name, axis in zip(AXIS_NAMES, domain.axes, strict=True):
        seen = coverage[name]
        result[name] = [value for value in axis if value in seen]
    return result


def census_order(domain: ReferenceDomain, order: str, shard_index: int,
                 shard_count: int) -> tuple[int, dict[str, list[Any]]]:
    """Exhaustively prove that an order visits its shard exactly once."""
    canonical = domain.canonical_ids(shard_index, shard_count)
    expected_count = len(canonical)
    seen = bytearray((expected_count + 7) // 8)
    coverage = _coverage_template()
    count = 0
    for tuple_id in domain.ordered_ids(order, shard_index, shard_count):
        if (tuple_id - shard_index) % shard_count:
            raise ReferenceEnumerationError(f"tuple_in_wrong_shard:{tuple_id}")
        local = (tuple_id - shard_index) // shard_count
        if local < 0 or local >= expected_count:
            raise ReferenceEnumerationError(f"tuple_outside_shard:{tuple_id}")
        byte_index, bit_index = divmod(local, 8)
        mask = 1 << bit_index
        if seen[byte_index] & mask:
            raise ReferenceEnumerationError(f"duplicate_tuple_id:{tuple_id}")
        seen[byte_index] |= mask
        values = domain.unrank(tuple_id)
        if domain.rank(values) != tuple_id:
            raise ReferenceEnumerationError(f"rank_unrank_mismatch:{tuple_id}")
        _add_coverage(coverage, values)
        count += 1
    if count != expected_count:
        raise ReferenceEnumerationError(
            f"shard_count_mismatch:expected={expected_count}:actual={count}"
        )
    full_bytes, trailing_bits = divmod(expected_count, 8)
    if any(value != 0xFF for value in seen[:full_bytes]):
        raise ReferenceEnumerationError("shard_seen_bitset_has_gap")
    if trailing_bits and seen[-1] != (1 << trailing_bits) - 1:
        raise ReferenceEnumerationError("shard_seen_bitset_has_trailing_gap")
    return count, _ordered_coverage(coverage, domain)


def root_and_optionally_write_records(domain: ReferenceDomain, ids: Iterable[int],
                                      output_path: Path | None) -> tuple[str, int, int | None, int | None]:
    digest = hashlib.sha256()
    count = 0
    minimum: int | None = None
    maximum: int | None = None
    handle = output_path.open("xb") if output_path is not None else None
    try:
        for tuple_id in ids:
            record = canonical_json_line(domain.record(tuple_id))
            domain_bytes = domain.domain_separator.encode("ascii")
            digest.update(len(domain_bytes).to_bytes(4, "big"))
            digest.update(domain_bytes)
            digest.update(len(record).to_bytes(8, "big"))
            digest.update(record)
            if handle is not None:
                handle.write(record)
            minimum = tuple_id if minimum is None else min(minimum, tuple_id)
            maximum = tuple_id if maximum is None else max(maximum, tuple_id)
            count += 1
    finally:
        if handle is not None:
            handle.flush()
            handle.close()
    return digest.hexdigest(), count, minimum, maximum


def merged_canonical_shard_ids(domain: ReferenceDomain, shard_count: int) -> Iterator[int]:
    """K-way merge canonical shards back into the global tuple-id order."""
    if shard_count <= 0:
        raise ReferenceEnumerationError("shard_count_not_positive")
    yield from heapq.merge(
        *(domain.canonical_ids(shard_index, shard_count)
          for shard_index in range(shard_count))
    )


def pilot_audit(domain: ReferenceDomain) -> dict[str, Any]:
    coverage = _coverage_template()
    count = 0
    for tuple_id in range(domain.tuple_id_minimum, domain.tuple_id_maximum + 1, 100):
        _add_coverage(coverage, domain.unrank(tuple_id))
        count += 1
    ordered = _ordered_coverage(coverage, domain)
    missing = {
        name: [value for value in axis if value not in coverage[name]]
        for name, axis in zip(AXIS_NAMES, domain.axes, strict=True)
    }
    return {
        "coverage": ordered,
        "missing": missing,
        "record_count": count,
        "selection": "tuple_id mod 100 == 0",
        "status": "passed" if all(not values for values in missing.values()) else "failed",
    }


def _prepare_output(path: Path) -> None:
    if path.exists():
        if not path.is_dir() or any(path.iterdir()):
            raise ReferenceEnumerationError(f"output_not_empty:{path}")
    else:
        path.mkdir(parents=True)


def _coverage_root(domain: ReferenceDomain, coverage: dict[str, list[Any]]) -> str:
    separator = domain.coverage_separator.encode("ascii")
    payload = canonical_json_line(coverage)
    framed = (len(separator).to_bytes(4, "big") + separator +
              len(payload).to_bytes(8, "big") + payload)
    return hashlib.sha256(framed).hexdigest()


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "byte_length": len(payload),
        "relative_path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def enumerate_to_directory(domain: ReferenceDomain, output: Path, order: str,
                           shard_index: int, shard_count: int,
                           write_records: bool) -> dict[str, Any]:
    _prepare_output(output)
    count, coverage = census_order(domain, order, shard_index, shard_count)
    records_path = output / RECORDS_FILENAME if write_records else None
    root, canonical_count, minimum, maximum = root_and_optionally_write_records(
        domain, domain.canonical_ids(shard_index, shard_count), records_path
    )
    if canonical_count != count:
        raise ReferenceEnumerationError("census_and_canonical_pass_count_mismatch")
    manifest = {
        "contract_sha256": domain.contract_sha256,
        "coverage": coverage,
        "coverage_root": _coverage_root(domain, coverage),
        "expected_record_count": len(domain.canonical_ids(shard_index, shard_count)),
        "mode": "records" if write_records else "census",
        "order": order,
        "record_count": count,
        "records_artifact": _artifact(records_path, output) if records_path else None,
        "records_root": root,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "shard_count": shard_count,
        "shard_index": shard_index,
        "status": "passed",
        "tuple_id_max": maximum,
        "tuple_id_min": minimum,
        "workers": 16,
    }
    (output / MANIFEST_FILENAME).write_bytes(canonical_json_line(manifest))
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--expected-contract-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--order", choices=ORDERS, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--mode", choices=("census", "records"), default="census")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.workers != 16:
        raise ReferenceEnumerationError("workers_must_equal_frozen_value_16")
    contract, digest = load_contract(args.contract, args.expected_contract_sha256)
    domain = ReferenceDomain.from_contract(contract, digest)
    manifest = enumerate_to_directory(
        domain,
        args.output,
        args.order,
        args.shard_index,
        args.shard_count,
        write_records=args.mode == "records",
    )
    print(json.dumps(manifest, ensure_ascii=True, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
