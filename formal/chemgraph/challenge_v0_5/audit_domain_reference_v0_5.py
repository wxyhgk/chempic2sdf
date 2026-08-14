#!/usr/bin/env python3
"""Independent P1 audit for the ChemGraph v0.5 reference enumerator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

try:
    from .enumerate_domain_reference_v0_5 import (
        AXIS_NAMES,
        ReferenceDomain,
        ReferenceEnumerationError,
        canonical_json_line,
        load_contract,
        pilot_audit,
        root_and_optionally_write_records,
    )
except ImportError:
    from enumerate_domain_reference_v0_5 import (  # type: ignore
        AXIS_NAMES,
        ReferenceDomain,
        ReferenceEnumerationError,
        canonical_json_line,
        load_contract,
        pilot_audit,
        root_and_optionally_write_records,
    )


VALIDATION_SCHEMA_VERSION = "chemgraph-domain-reference-audit-v0.5"


def _cartesian_count_without_product_helper(domain: ReferenceDomain) -> int:
    count = 1
    for axis in domain.axes:
        count *= len(axis)
    return count


def _boundary_bijection_witnesses(domain: ReferenceDomain) -> list[int]:
    witnesses = {0, 1, domain.declared_count - 2, domain.declared_count - 1}
    stride = 1
    for radix in reversed(domain.radices):
        witnesses.add(stride - 1)
        witnesses.add(stride)
        witnesses.add(radix * stride - 1)
        stride *= radix
    return sorted(value for value in witnesses if 0 <= value < domain.declared_count)


def audit(contract_path: Path, expected_sha256: str) -> dict[str, Any]:
    contract, actual = load_contract(contract_path, expected_sha256)
    domain = ReferenceDomain.from_contract(contract, actual)
    calculated_count = _cartesian_count_without_product_helper(domain)
    witnesses = _boundary_bijection_witnesses(domain)
    for tuple_id in witnesses:
        values = domain.unrank(tuple_id)
        if domain.rank(values) != tuple_id:
            raise ReferenceEnumerationError(f"boundary_bijection_failed:{tuple_id}")

    pilot = pilot_audit(domain)
    if pilot["status"] != "passed":
        raise ReferenceEnumerationError("pilot_axis_coverage_failed")
    expected_pilot_count = (domain.declared_count + 99) // 100
    if pilot["record_count"] != expected_pilot_count:
        raise ReferenceEnumerationError("pilot_count_formula_mismatch")

    # A real production-domain shard proves that three independently traversed
    # orders reduce to the same canonical record stream without a full 36M
    # artifact write.
    shard_index, shard_count = 97, 100_003
    canonical_ids = list(domain.canonical_ids(shard_index, shard_count))
    order_roots: dict[str, str] = {}
    for order in ("forward", "reverse", "shuffled"):
        visited = list(domain.ordered_ids(order, shard_index, shard_count))
        if len(visited) != len(set(visited)) or sorted(visited) != canonical_ids:
            raise ReferenceEnumerationError(f"sample_order_not_bijective:{order}")
        order_roots[order], sample_count, _, _ = root_and_optionally_write_records(
            domain, sorted(visited), None
        )
    if len(set(order_roots.values())) != 1:
        raise ReferenceEnumerationError("sample_order_roots_differ")
    canonical_root = order_roots["forward"]

    return {
        "axis_cardinalities": dict(zip(AXIS_NAMES, domain.radices, strict=True)),
        "bijection_boundary_witnesses": witnesses,
        "calculated_tuple_count": calculated_count,
        "contract_sha256": actual,
        "declared_tuple_count": domain.declared_count,
        "pilot": pilot,
        "sample_canonical_root": canonical_root,
        "sample_order_roots": order_roots,
        "sample_record_count": sample_count,
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "status": "passed",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--expected-contract-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = audit(args.contract, args.expected_contract_sha256)
    if args.output.exists():
        raise ReferenceEnumerationError(f"output_exists:{args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_line(result))
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
