#!/usr/bin/env python3
"""Independently verify all eight frozen v0.6 E2 producer witnesses."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import queue
import stat
import sys
from typing import Any

try:
    from .independent_e2_core_v0_6 import (
        E2_CAPS,
        PARENT_CARDINALITY,
        PARENT_CONTRACT_SHA256,
        TRANSLATION_CHECKS,
        V0_6_CONTRACT_SHA256,
        WIDTHS,
        IndependentVerificationError,
        PairComputation,
        SideComputation,
        all_expected_pairs,
        canonical_json_bytes,
        sha256,
    )
except ImportError:
    from independent_e2_core_v0_6 import (  # type: ignore
        E2_CAPS,
        PARENT_CARDINALITY,
        PARENT_CONTRACT_SHA256,
        TRANSLATION_CHECKS,
        V0_6_CONTRACT_SHA256,
        WIDTHS,
        IndependentVerificationError,
        PairComputation,
        SideComputation,
        all_expected_pairs,
        canonical_json_bytes,
        sha256,
    )


MANIFEST_SCHEMA_VERSION = "chemgraph-e2-producer-manifest-v0.6"
VALIDATION_SCHEMA_VERSION = "chemgraph-e2-verifier-a-validation-v0.6"
ARTIFACT_DOMAIN = b"chemgraph-v0.6/e2-producer-artifacts"
FIXED_WORKERS = 4
FIXED_ASSIGNMENTS = ((0, 1), (2, 3), (4, 5), (6, 7))
EXPECTED_PLAN = {
    "antialias_profile": "binary_1x",
    "caps": ["butt", "square"],
    "left_fixed": {
        "half_length": 62,
        "interpretation_index": 0,
        "motif_ordinal": 11,
        "reflection": "none",
        "rotation_index": 0,
        "scale_index": 2,
    },
    "right_fixed": {
        "half_length": 93,
        "interpretation_index": 0,
        "motif_ordinal": 48,
        "reflection": "none",
        "rotation_index": 4,
        "scale_index": 0,
    },
    "width_px": [2, 4, 6, 8],
    "witness_pair_count": 8,
}


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise IndependentVerificationError(f"duplicate_json_key:{key}")
        result[key] = value
    return result


def _load_json(
    path: Path, label: str, *, require_canonical: bool = True
) -> dict[str, Any]:
    _require_regular_file(path, label)
    payload = path.read_bytes()
    try:
        value = json.loads(payload, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IndependentVerificationError(f"invalid_json:{label}:{error}") from error
    if not isinstance(value, dict):
        raise IndependentVerificationError(f"json_root_not_object:{label}")
    if require_canonical and canonical_json_bytes(value) != payload:
        raise IndependentVerificationError(f"noncanonical_json:{label}")
    return value


def _require_regular_file(path: Path, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise IndependentVerificationError(f"missing_file:{label}") from error
    if not stat.S_ISREG(mode):
        raise IndependentVerificationError(f"not_regular_file:{label}")


def _require_keys(value: object, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise IndependentVerificationError(f"expected_object:{label}")
    actual = set(value)
    if actual != expected:
        raise IndependentVerificationError(
            f"field_mismatch:{label}:missing={sorted(expected - actual)}:"
            f"extra={sorted(actual - expected)}"
        )
    return value


def _verify_contracts(v0_6_contract: Path, parent_contract: Path) -> None:
    v0_6_payload = v0_6_contract.read_bytes()
    if sha256(v0_6_payload) != V0_6_CONTRACT_SHA256:
        raise IndependentVerificationError("v0_6_contract_sha256_mismatch")
    v0_6 = _load_json(v0_6_contract, "v0_6_contract")
    if v0_6.get("schema_version") != "chemgraph-observation-closure-p0-v0.6":
        raise IndependentVerificationError("v0_6_contract_schema_mismatch")
    if v0_6.get("training_authorized") is not False:
        raise IndependentVerificationError("v0_6_contract_authorizes_training")
    if v0_6.get("allow_exclusions") is not False:
        raise IndependentVerificationError("v0_6_contract_allows_exclusions")
    domain = _require_keys(
        v0_6.get("domain_binding"),
        {
            "allow_exclusions", "exclusion_count", "frozen_quotient_tuple_count",
            "interpretation_scope", "parent_contract_path", "parent_contract_sha256",
            "reuse_mode", "scene_domain", "tuple_id_maximum", "tuple_id_minimum",
        },
        "v0_6_contract.domain_binding",
    )
    if domain["parent_contract_sha256"] != PARENT_CONTRACT_SHA256:
        raise IndependentVerificationError("v0_6_parent_binding_mismatch")
    if domain["frozen_quotient_tuple_count"] != PARENT_CARDINALITY:
        raise IndependentVerificationError("v0_6_parent_cardinality_mismatch")
    if domain["allow_exclusions"] is not False or domain["exclusion_count"] != 0:
        raise IndependentVerificationError("v0_6_domain_exclusion_detected")
    if domain["reuse_mode"] != "reference_only_no_copy_or_redefinition":
        raise IndependentVerificationError("v0_6_parent_domain_redefined")
    lanes = v0_6.get("claim_lanes")
    if not isinstance(lanes, dict) or lanes.get("E2", {}).get("witness_plan") != EXPECTED_PLAN:
        raise IndependentVerificationError("v0_6_e2_plan_mismatch")

    parent_payload = parent_contract.read_bytes()
    if sha256(parent_payload) != PARENT_CONTRACT_SHA256:
        raise IndependentVerificationError("parent_contract_sha256_mismatch")
    parent = _load_json(
        parent_contract, "parent_contract", require_canonical=False
    )
    if parent.get("schema_version") != "chemgraph-observation-closure-p0-v0.5":
        raise IndependentVerificationError("parent_contract_schema_mismatch")
    parent_domain = parent.get("domain")
    if not isinstance(parent_domain, dict):
        raise IndependentVerificationError("parent_domain_missing")
    if parent_domain.get("quotient_tuple_count") != PARENT_CARDINALITY:
        raise IndependentVerificationError("parent_domain_cardinality_mismatch")
    if parent_domain.get("tuple_id_minimum") != 0 or parent_domain.get("tuple_id_maximum") != PARENT_CARDINALITY - 1:
        raise IndependentVerificationError("parent_tuple_bounds_mismatch")


def _descriptor(path: str, payload: bytes, *, width: int, height: int, format_name: str) -> dict[str, object]:
    return {
        "byte_length": len(payload),
        "format": format_name,
        "height": height,
        "path": path,
        "sha256": sha256(payload),
        "width": width,
    }


def _side_payloads(side: SideComputation) -> dict[str, bytes]:
    return {
        "raw_gray": side.raw_gray,
        "padded_gray": side.padded_gray,
        "canonical_gray": side.canonical_gray,
        "canonical_png": side.canonical_png,
    }


def _expected_side(
    pair_id: str,
    role: str,
    side: SideComputation,
    artifacts: dict[str, bytes],
) -> dict[str, object]:
    payloads = _side_payloads(side)
    extensions = {
        "raw_gray": "raw.gray",
        "padded_gray": "padded.gray",
        "canonical_gray": "canonical.gray",
        "canonical_png": "canonical.png",
    }
    dimensions = {
        "raw_gray": (768, 768, "gray8-row-major"),
        "padded_gray": (1024, 1024, "gray8-row-major"),
        "canonical_gray": (768, 768, "gray8-row-major"),
        "canonical_png": (768, 768, "canonical-gray8-png"),
    }
    descriptors: dict[str, object] = {}
    for field, payload in payloads.items():
        relative = f"artifacts/{pair_id}/{role}.{extensions[field]}"
        artifacts[relative] = payload
        width, height, format_name = dimensions[field]
        descriptors[field] = _descriptor(
            relative, payload, width=width, height=height, format_name=format_name
        )
    return {
        **descriptors,
        "family": side.family,
        "graph": side.graph,
        "interpretation_index": side.interpretation_index,
        "interpretation_membership": True,
        "padded_clipping_certificate": side.padded_certificate,
        "tuple": side.tuple_record,
        "tuple_membership": True,
    }


def _artifact_tree_hash(artifacts: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    digest.update(len(ARTIFACT_DOMAIN).to_bytes(4, "big"))
    digest.update(ARTIFACT_DOMAIN)
    for path in sorted(artifacts):
        path_bytes = path.encode("ascii")
        payload = artifacts[path]
        digest.update(len(path_bytes).to_bytes(4, "big"))
        digest.update(path_bytes)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _expected_pair(
    pair_index: int,
    computation: PairComputation,
    artifacts: dict[str, bytes],
) -> dict[str, object]:
    pair_id = f"e2-{pair_index:02d}-{computation.cap}-w{computation.width_px}"
    left = _expected_side(pair_id, "left", computation.left, artifacts)
    right = _expected_side(pair_id, "right", computation.right, artifacts)
    return {
        "checks": {
            "canonical_gray_full_bytes_equal": True,
            "canonical_png_full_bytes_equal": True,
            "graph_signatures_distinct": True,
            "padded_clipping_certificates_passed": True,
            "raw_gray_full_bytes_distinct": True,
            "tuple_ids_distinct": True,
        },
        "left": left,
        "pair_id": pair_id,
        "right": right,
        "style": {
            "antialias_profile": "binary_1x",
            "cap": computation.cap,
            "width_px": computation.width_px,
        },
    }


def _expected_manifest() -> tuple[dict[str, object], dict[str, bytes]]:
    artifacts: dict[str, bytes] = {}
    pairs = [
        _expected_pair(index, computation, artifacts)
        for index, computation in enumerate(all_expected_pairs())
    ]
    total = sum(len(payload) for payload in artifacts.values())
    return {
        "artifact_count": len(artifacts),
        "artifact_total_bytes": total,
        "artifact_tree_sha256": _artifact_tree_hash(artifacts),
        "contract_binding": {
            "parent_contract_sha256": PARENT_CONTRACT_SHA256,
            "parent_domain_cardinality": PARENT_CARDINALITY,
            "v0_6_contract_sha256": V0_6_CONTRACT_SHA256,
        },
        "pair_count": len(pairs),
        "pairs": pairs,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "scope": {
            "claim": "exact_full_domain_existential_cross_tuple_full_bytes",
            "full_domain_collision_rate_claimed": False,
            "sampled_to_full_extrapolation_used": False,
            "training_authorized": False,
        },
        "status": "passed",
        "witness_plan": EXPECTED_PLAN,
    }, artifacts


def _safe_artifact_files(delivery: Path) -> set[str]:
    artifact_root = delivery / "artifacts"
    if not artifact_root.exists() or artifact_root.is_symlink() or not artifact_root.is_dir():
        raise IndependentVerificationError("artifact_root_unsafe_or_missing")
    files: set[str] = set()
    for path in artifact_root.rglob("*"):
        relative = path.relative_to(delivery).as_posix()
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise IndependentVerificationError(f"artifact_symlink:{relative}")
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise IndependentVerificationError(f"artifact_not_regular:{relative}")
        try:
            relative.encode("ascii")
        except UnicodeEncodeError as error:
            raise IndependentVerificationError(f"artifact_path_not_ascii:{relative}") from error
        files.add(relative)
    return files


def _planned_artifact_paths() -> set[str]:
    paths = set()
    extensions = ("raw.gray", "padded.gray", "canonical.gray", "canonical.png")
    for pair_index in range(8):
        cap = E2_CAPS[pair_index // len(WIDTHS)]
        width = WIDTHS[pair_index % len(WIDTHS)]
        pair_id = f"e2-{pair_index:02d}-{cap}-w{width}"
        for role in ("left", "right"):
            for extension in extensions:
                paths.add(f"artifacts/{pair_id}/{role}.{extension}")
    return paths


def _verify_artifact_payloads(
    delivery: Path, expected_artifacts: dict[str, bytes]
) -> tuple[int, int]:
    byte_count = 0
    for relative, expected_payload in sorted(expected_artifacts.items()):
        path = delivery / relative
        _require_regular_file(path, relative)
        payload = path.read_bytes()
        if payload != expected_payload:
            raise IndependentVerificationError(f"artifact_full_bytes_mismatch:{relative}")
        byte_count += len(payload)
    return len(expected_artifacts), byte_count


def _verify_worker_pairs(
    manifest_path: Path, worker_index: int, pair_indices: tuple[int, int]
) -> dict[str, object]:
    manifest = _load_json(manifest_path, f"worker_{worker_index}_producer_manifest")
    pairs = manifest.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != 8:
        raise IndependentVerificationError("worker_manifest_pair_count_mismatch")
    delivery = manifest_path.parent
    verified_ids = []
    sidecar_count = 0
    sidecar_bytes = 0
    for pair_index in pair_indices:
        cap = E2_CAPS[pair_index // len(WIDTHS)]
        width = WIDTHS[pair_index % len(WIDTHS)]
        computation = all_expected_pairs_for_indices(cap, width)
        expected_artifacts: dict[str, bytes] = {}
        expected_pair = _expected_pair(pair_index, computation, expected_artifacts)
        count, byte_count = _verify_artifact_payloads(delivery, expected_artifacts)
        if pairs[pair_index] != expected_pair:
            raise IndependentVerificationError(
                f"worker_pair_recomputation_mismatch:{pair_index}"
            )
        verified_ids.append(expected_pair["pair_id"])
        sidecar_count += count
        sidecar_bytes += byte_count
    return {
        "pair_ids": verified_ids,
        "pair_indices": list(pair_indices),
        "pid": os.getpid(),
        "recomputed_pair_count": len(pair_indices),
        "sidecar_bytes": sidecar_bytes,
        "sidecar_count": sidecar_count,
        "status": "passed",
        "worker_index": worker_index,
    }


def all_expected_pairs_for_indices(cap: str, width: int) -> PairComputation:
    # Importing the one-pair function here keeps each spawned process scoped to
    # exactly its two assigned pairs rather than materializing all eight.
    try:
        from .independent_e2_core_v0_6 import compute_pair
    except ImportError:
        from independent_e2_core_v0_6 import compute_pair  # type: ignore
    return compute_pair(cap, width)


def _worker_entry(
    result_queue: Any,
    manifest_path: str,
    worker_index: int,
    pair_indices: tuple[int, int],
) -> None:
    try:
        result = _verify_worker_pairs(
            Path(manifest_path), worker_index, pair_indices
        )
    except BaseException as error:
        result_queue.put({
            "error": f"{type(error).__name__}:{error}",
            "pair_indices": list(pair_indices),
            "pid": os.getpid(),
            "status": "failed",
            "worker_index": worker_index,
        })
        raise SystemExit(1) from error
    result_queue.put(result)


def _run_fixed_workers(manifest_path: Path) -> list[dict[str, object]]:
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_worker_entry,
            args=(result_queue, manifest_path.as_posix(), index, assignment),
            name=f"chemgraph-e2-verifier-a-{index}",
        )
        for index, assignment in enumerate(FIXED_ASSIGNMENTS)
    ]
    if len(processes) != FIXED_WORKERS:
        raise IndependentVerificationError("fixed_worker_construction_mismatch")
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    results = []
    for _ in processes:
        try:
            results.append(result_queue.get(timeout=2))
        except queue.Empty as error:
            raise IndependentVerificationError("worker_result_missing") from error
    result_queue.close()
    result_queue.join_thread()
    exit_codes = [process.exitcode for process in processes]
    failed = [result for result in results if result.get("status") != "passed"]
    if exit_codes != [0] * FIXED_WORKERS or failed:
        raise IndependentVerificationError(
            f"worker_failure:exit_codes={exit_codes}:results={failed}"
        )
    ordered = sorted(results, key=lambda result: int(result["worker_index"]))
    if [tuple(result["pair_indices"]) for result in ordered] != list(FIXED_ASSIGNMENTS):
        raise IndependentVerificationError("worker_assignment_mismatch")
    pids = {int(result["pid"]) for result in ordered}
    if len(pids) != FIXED_WORKERS or os.getpid() in pids:
        raise IndependentVerificationError("workers_not_four_distinct_child_processes")
    if any(result["recomputed_pair_count"] != 2 for result in ordered):
        raise IndependentVerificationError("worker_pair_count_not_two")
    return ordered


def verify_manifest(
    manifest_path: Path,
    v0_6_contract: Path,
    parent_contract: Path,
) -> dict[str, object]:
    _verify_contracts(v0_6_contract, parent_contract)
    manifest = _load_json(manifest_path, "producer_manifest")
    _require_keys(
        manifest,
        {
            "artifact_count", "artifact_total_bytes", "artifact_tree_sha256",
            "contract_binding", "pair_count", "pairs", "schema_version", "scope",
            "status", "witness_plan",
        },
        "producer_manifest",
    )
    if manifest["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise IndependentVerificationError("producer_manifest_schema_mismatch")
    if manifest["status"] != "passed":
        raise IndependentVerificationError("producer_manifest_not_passed")
    if manifest["contract_binding"] != {
        "parent_contract_sha256": PARENT_CONTRACT_SHA256,
        "parent_domain_cardinality": PARENT_CARDINALITY,
        "v0_6_contract_sha256": V0_6_CONTRACT_SHA256,
    }:
        raise IndependentVerificationError("producer_contract_binding_mismatch")
    if manifest["witness_plan"] != EXPECTED_PLAN:
        raise IndependentVerificationError("producer_witness_plan_mismatch")
    if manifest["scope"] != {
        "claim": "exact_full_domain_existential_cross_tuple_full_bytes",
        "full_domain_collision_rate_claimed": False,
        "sampled_to_full_extrapolation_used": False,
        "training_authorized": False,
    }:
        raise IndependentVerificationError("producer_scope_mismatch")
    if manifest["pair_count"] != 8:
        raise IndependentVerificationError("producer_pair_count_mismatch")
    pairs = manifest["pairs"]
    if not isinstance(pairs, list) or len(pairs) != 8:
        raise IndependentVerificationError("producer_pairs_length_mismatch")
    delivery = manifest_path.parent
    actual_paths = _safe_artifact_files(delivery)
    expected_paths = _planned_artifact_paths()
    if actual_paths != expected_paths:
        raise IndependentVerificationError(
            f"artifact_path_set_mismatch:missing={sorted(expected_paths - actual_paths)}:"
            f"extra={sorted(actual_paths - expected_paths)}"
        )
    worker_results = _run_fixed_workers(manifest_path)
    if sum(int(result["recomputed_pair_count"]) for result in worker_results) != 8:
        raise IndependentVerificationError("worker_total_pair_count_mismatch")
    if sum(int(result["sidecar_count"]) for result in worker_results) != 64:
        raise IndependentVerificationError("worker_total_sidecar_count_mismatch")
    public_worker_results = [
        {key: value for key, value in result.items() if key != "pid"}
        for result in worker_results
    ]
    verified_payloads: dict[str, bytes] = {}
    for relative in sorted(expected_paths):
        path = delivery / relative
        _require_regular_file(path, relative)
        payload = path.read_bytes()
        verified_payloads[relative] = payload
    artifact_total = sum(len(payload) for payload in verified_payloads.values())
    artifact_tree = _artifact_tree_hash(verified_payloads)
    if manifest["artifact_count"] != 64:
        raise IndependentVerificationError("producer_artifact_count_mismatch")
    if manifest["artifact_total_bytes"] != artifact_total:
        raise IndependentVerificationError("producer_artifact_total_bytes_mismatch")
    if manifest["artifact_tree_sha256"] != artifact_tree:
        raise IndependentVerificationError("producer_artifact_tree_mismatch")
    return {
        "artifact_count": len(verified_payloads),
        "artifact_total_bytes": artifact_total,
        "artifact_tree_sha256": artifact_tree,
        "canonical_gray_full_byte_equal_pair_count": 8,
        "canonical_png_full_byte_equal_pair_count": 8,
        "full_byte_sidecar_count": len(verified_payloads),
        "graph_inequivalent_pair_count": 8,
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": sha256(manifest_path.read_bytes()),
        "pair_count": 8,
        "padded_clipping_certificate_count": 16,
        "raw_gray_full_byte_distinct_pair_count": 8,
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "status": "passed",
        "training_authorized": False,
        "translation_preprocess_replay_count": 16 * len(TRANSLATION_CHECKS),
        "tuple_membership_count": 16,
        "verifier_id": "python-stdlib-independent-a",
        "v0_6_contract_sha256": V0_6_CONTRACT_SHA256,
        "worker_recomputed_pair_count": 8,
        "worker_process_count": FIXED_WORKERS,
        "worker_results": public_worker_results,
        "worker_start_method": "spawn",
        "workers": FIXED_WORKERS,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--v0-6-contract", type=Path, required=True)
    parser.add_argument("--parent-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify_manifest(args.manifest, args.v0_6_contract, args.parent_contract)
    encoded = canonical_json_bytes(report)
    if args.output is not None:
        args.output.write_bytes(encoded)
    sys.stdout.buffer.write(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
