#!/usr/bin/env python3
"""Produce the eight preregistered v0.6 E2 cross-tuple witnesses."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import shutil
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from formal.chemgraph.challenge_v0_4.rasterizer_v0_4 import (  # noqa: E402
    encode_canonical_png,
)
from formal.chemgraph.challenge_v0_5.canonical_raster_v0_5 import (  # noqa: E402
    foreground_bbox,
    preprocess_gray,
    require_white_border,
    verify_contract as verify_parent_contract,
)
from formal.chemgraph.challenge_v0_5.domain_scene_v0_5 import (  # noqa: E402
    decode_motif_ordinal,
    render_centered_gray,
    verify_padded_clipping_certificate,
)
from formal.chemgraph.challenge_v0_5.enumerate_domain_v0_5 import (  # noqa: E402
    AXES,
    TOTAL_TUPLES,
    decode_tuple_id,
)


SCHEMA_VERSION = "chemgraph-e2-producer-manifest-v0.6"
VALIDATION_SCHEMA_VERSION = "chemgraph-e2-producer-validation-v0.6"
V0_6_CONTRACT_SHA256 = (
    "e326fc744903b998793746f299232c4225ce3b8648e1abdda7b4e065f06a9402"
)
PARENT_CONTRACT_SHA256 = (
    "55ddcaacdf851166554ccd586b2a924c0e793f86cc7a6319f98132ef0ffb6b65"
)
PARENT_DOMAIN_CARDINALITY = 36_088_416
ARTIFACT_DOMAIN = b"chemgraph-v0.6/e2-producer-artifacts"
FIXED_WORKERS = 4
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


class E2ProducerError(ValueError):
    """Raised when a frozen E2 producer obligation fails."""


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise E2ProducerError(f"duplicate_json_key:{key}")
        value[key] = item
    return value


def _load_v0_6_contract(path: Path, expected_sha256: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if expected_sha256 != V0_6_CONTRACT_SHA256:
        raise E2ProducerError("expected_v0_6_contract_sha256_not_frozen")
    if sha256(raw) != expected_sha256:
        raise E2ProducerError("v0_6_contract_sha256_mismatch")
    try:
        contract = json.loads(raw, object_pairs_hook=_reject_duplicate_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise E2ProducerError(f"invalid_v0_6_contract:{error}") from error
    if contract.get("schema_version") != "chemgraph-observation-closure-p0-v0.6":
        raise E2ProducerError("v0_6_contract_schema_changed")
    if contract.get("training_authorized") is not False:
        raise E2ProducerError("v0_6_contract_authorizes_training")
    plan = contract.get("claim_lanes", {}).get("E2", {}).get("witness_plan")
    if plan != EXPECTED_PLAN:
        raise E2ProducerError("e2_witness_plan_changed")
    binding = contract.get("domain_binding", {})
    if binding.get("parent_contract_sha256") != PARENT_CONTRACT_SHA256:
        raise E2ProducerError("parent_contract_binding_changed")
    if binding.get("frozen_quotient_tuple_count") != PARENT_DOMAIN_CARDINALITY:
        raise E2ProducerError("parent_domain_cardinality_changed")
    return contract


def _tuple_id(axis_values: dict[str, Any]) -> int:
    encoded = 0
    for name, allowed in AXES:
        try:
            index = allowed.index(axis_values[name])
        except (KeyError, ValueError) as error:
            raise E2ProducerError(f"tuple_axis_out_of_domain:{name}") from error
        encoded = encoded * len(allowed) + index
    expected = {
        "schema_version": "chemgraph-domain-tuple-v0.5",
        "tuple_id": encoded,
        **axis_values,
    }
    if decode_tuple_id(encoded) != expected:
        raise E2ProducerError("tuple_membership_roundtrip_failed")
    return encoded


def _render_axes(tuple_record: dict[str, Any]) -> dict[str, int]:
    axis_map = {name: values for name, values in AXES}
    return {
        "half_length": tuple_record["half_length"],
        "motif_ordinal": tuple_record["motif_ordinal"],
        "width_px": tuple_record["width_px"],
        "cap_index": axis_map["cap"].index(tuple_record["cap"]),
        "antialias_index": axis_map["antialias_profile"].index(
            tuple_record["antialias_profile"]
        ),
        "reflection_index": axis_map["reflection"].index(
            tuple_record["reflection"]
        ),
        "rotation_index": tuple_record["rotation_index"],
        "scale_index": tuple_record["scale_index"],
    }


def _foreground_bbox(gray: bytes, width: int, height: int) -> list[int]:
    if len(gray) != width * height:
        raise E2ProducerError("gray_shape_mismatch")
    min_x, min_y, max_x, max_y = width, height, -1, -1
    for y in range(height):
        row = gray[y * width:(y + 1) * width]
        left_trimmed = row.lstrip(b"\xff")
        if not left_trimmed:
            continue
        left = width - len(left_trimmed)
        right = len(row.rstrip(b"\xff")) - 1
        min_x = min(min_x, left)
        min_y = min(min_y, y)
        max_x = max(max_x, right)
        max_y = y
    if max_x < 0:
        raise E2ProducerError("empty_foreground")
    return [min_x, min_y, max_x, max_y]


def _graph_signature(interpretation: Any) -> str:
    nodes = tuple((atom.label, 0) for atom in interpretation.atoms)
    bonds = tuple(
        (bond.begin, bond.end, bond.order) for bond in interpretation.bonds
    )
    best: str | None = None
    for order in itertools.permutations(range(len(nodes))):
        inverse = {old: new for new, old in enumerate(order)}
        value = {
            "edges": sorted(
                [
                    min(inverse[left], inverse[right]),
                    max(inverse[left], inverse[right]),
                    bond_order,
                ]
                for left, right, bond_order in bonds
            ),
            "nodes": [[nodes[old][0], nodes[old][1]] for old in order],
        }
        encoded = canonical_bytes(value).decode("ascii").rstrip("\n")
        if best is None or encoded < best:
            best = encoded
    if best is None:
        raise E2ProducerError("empty_graph_not_supported")
    return best


def _descriptor(relative_path: Path, payload: bytes, *, width: int, height: int,
                format_name: str) -> dict[str, Any]:
    return {
        "byte_length": len(payload),
        "format": format_name,
        "height": height,
        "path": relative_path.as_posix(),
        "sha256": sha256(payload),
        "width": width,
    }


def _write_sidecar(stage: Path, relative_path: Path, payload: bytes, *,
                   width: int, height: int, format_name: str,
                   artifact_payloads: dict[str, bytes]) -> dict[str, Any]:
    destination = stage / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    artifact_payloads[relative_path.as_posix()] = payload
    return _descriptor(
        relative_path, payload, width=width, height=height, format_name=format_name
    )


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


def _tuple_record(fixed: dict[str, Any], cap: str, width_px: int) -> dict[str, Any]:
    axes = {
        "half_length": fixed["half_length"],
        "motif_ordinal": fixed["motif_ordinal"],
        "width_px": width_px,
        "cap": cap,
        "antialias_profile": EXPECTED_PLAN["antialias_profile"],
        "reflection": fixed["reflection"],
        "rotation_index": fixed["rotation_index"],
        "scale_index": fixed["scale_index"],
    }
    return {
        "tuple_id": _tuple_id(axes),
        **axes,
    }


def _compute_side(
    pair_path: Path,
    role: str,
    tuple_record: dict[str, Any],
    interpretation_index: int,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    axes = _render_axes(tuple_record)
    raw_gray, motif = render_centered_gray(**axes)
    padded_gray, padded_motif = render_centered_gray(
        **axes, canvas_size=1024, center_px=512
    )
    verify_padded_clipping_certificate(raw_gray, padded_gray)
    if motif != padded_motif:
        raise E2ProducerError(f"padded_motif_changed:{role}")
    if not 0 <= interpretation_index < len(motif.interpretations):
        raise E2ProducerError(f"interpretation_not_in_Iq:{role}")
    require_white_border(raw_gray, 33)
    canonical_gray = preprocess_gray(raw_gray)
    canonical_png = encode_canonical_png(canonical_gray)
    interpretation = motif.interpretations[interpretation_index]
    signature = _graph_signature(interpretation)
    payloads = {
        "raw_gray": raw_gray,
        "padded_gray": padded_gray,
        "canonical_gray": canonical_gray,
        "canonical_png": canonical_png,
    }
    descriptors = {
        "raw_gray": _descriptor(
            pair_path / f"{role}.raw.gray", raw_gray,
            width=768, height=768, format_name="gray8-row-major",
        ),
        "padded_gray": _descriptor(
            pair_path / f"{role}.padded.gray", padded_gray,
            width=1024, height=1024, format_name="gray8-row-major",
        ),
        "canonical_gray": _descriptor(
            pair_path / f"{role}.canonical.gray", canonical_gray,
            width=768, height=768, format_name="gray8-row-major",
        ),
        "canonical_png": _descriptor(
            pair_path / f"{role}.canonical.png", canonical_png,
            width=768, height=768, format_name="canonical-gray8-png",
        ),
    }
    side = {
        **descriptors,
        "family": decode_motif_ordinal(tuple_record["motif_ordinal"]).family,
        "graph": {
            "byte_length": len(signature.encode("ascii")),
            "canonical_bytes_ascii": signature,
            "decode_kind": interpretation.decode_kind,
            "edge_count": len(interpretation.bonds),
            "node_count": len(interpretation.atoms),
            "sha256": sha256(signature.encode("ascii")),
        },
        "interpretation_index": interpretation_index,
        "interpretation_membership": True,
        "padded_clipping_certificate": {
            "embedded_raw_full_bytes_equal": True,
            "foreground_bbox": _foreground_bbox(padded_gray, 1024, 1024),
            "offset": [128, 128],
            "outer_white": True,
            "passed": True,
        },
        "tuple": tuple_record,
        "tuple_membership": True,
    }
    return side, payloads


def _pair_spec(pair_index: int) -> tuple[str, int]:
    width_count = len(EXPECTED_PLAN["width_px"])
    cap_index, width_index = divmod(pair_index, width_count)
    try:
        return EXPECTED_PLAN["caps"][cap_index], EXPECTED_PLAN["width_px"][width_index]
    except IndexError as error:
        raise E2ProducerError(f"pair_index_out_of_range:{pair_index}") from error


def _compute_pair(pair_index: int, worker_id: int) -> dict[str, Any]:
    cap, width_px = _pair_spec(pair_index)
    pair_id = f"e2-{pair_index:02d}-{cap}-w{width_px}"
    pair_path = Path("artifacts") / pair_id
    left_tuple = _tuple_record(EXPECTED_PLAN["left_fixed"], cap, width_px)
    right_tuple = _tuple_record(EXPECTED_PLAN["right_fixed"], cap, width_px)
    left, left_payloads = _compute_side(
        pair_path, "left", left_tuple,
        EXPECTED_PLAN["left_fixed"]["interpretation_index"],
    )
    right, right_payloads = _compute_side(
        pair_path, "right", right_tuple,
        EXPECTED_PLAN["right_fixed"]["interpretation_index"],
    )
    checks = {
        "canonical_gray_full_bytes_equal":
            left_payloads["canonical_gray"] == right_payloads["canonical_gray"],
        "canonical_png_full_bytes_equal":
            left_payloads["canonical_png"] == right_payloads["canonical_png"],
        "graph_signatures_distinct":
            left["graph"]["canonical_bytes_ascii"]
            != right["graph"]["canonical_bytes_ascii"],
        "padded_clipping_certificates_passed":
            left["padded_clipping_certificate"]["passed"]
            and right["padded_clipping_certificate"]["passed"],
        "raw_gray_full_bytes_distinct":
            left_payloads["raw_gray"] != right_payloads["raw_gray"],
        "tuple_ids_distinct": left_tuple["tuple_id"] != right_tuple["tuple_id"],
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise E2ProducerError(f"pair_obligation_failed:{pair_id}:{failed}")
    return {
        "left_payloads": left_payloads,
        "pair": {
            "checks": checks,
            "left": left,
            "pair_id": pair_id,
            "right": right,
            "style": {
                "antialias_profile": EXPECTED_PLAN["antialias_profile"],
                "cap": cap,
                "width_px": width_px,
            },
        },
        "pair_index": pair_index,
        "right_payloads": right_payloads,
        "worker_id": worker_id,
        "worker_pid": os.getpid(),
    }


def _worker_main(worker_id: int, output: Any) -> None:
    try:
        for pair_index in range(worker_id, EXPECTED_PLAN["witness_pair_count"], FIXED_WORKERS):
            output.put(("pair", worker_id, os.getpid(), _compute_pair(pair_index, worker_id)))
        output.put(("done", worker_id, os.getpid(), None))
    except BaseException as error:
        output.put(("error", worker_id, os.getpid(), f"{type(error).__name__}:{error}"))


def _run_fixed_workers(runtime_audit: dict[str, Any] | None) -> list[dict[str, Any]]:
    context = mp.get_context("spawn")
    output = context.Queue()
    processes = [
        context.Process(target=_worker_main, args=(worker_id, output))
        for worker_id in range(FIXED_WORKERS)
    ]
    for process in processes:
        process.start()
    results: list[dict[str, Any]] = []
    done: dict[int, int] = {}
    try:
        while len(done) < FIXED_WORKERS:
            try:
                kind, worker_id, worker_pid, payload = output.get(timeout=60)
            except queue.Empty as error:
                raise E2ProducerError("worker_result_timeout") from error
            if kind == "error":
                raise E2ProducerError(f"worker_failed:{worker_id}:{payload}")
            if kind == "pair":
                results.append(payload)
                continue
            if kind != "done" or worker_id in done:
                raise E2ProducerError(f"invalid_worker_message:{kind}:{worker_id}")
            done[worker_id] = worker_pid
    finally:
        for process in processes:
            process.join(timeout=10)
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()
    if any(process.exitcode != 0 for process in processes):
        raise E2ProducerError("worker_nonzero_exit")
    if set(done) != set(range(FIXED_WORKERS)) or len(set(done.values())) != FIXED_WORKERS:
        raise E2ProducerError("four_distinct_workers_not_observed")
    if os.getpid() in done.values():
        raise E2ProducerError("worker_ran_in_parent_process")
    expected_assignments = {
        worker_id: list(range(worker_id, EXPECTED_PLAN["witness_pair_count"], FIXED_WORKERS))
        for worker_id in range(FIXED_WORKERS)
    }
    actual_assignments = {worker_id: [] for worker_id in range(FIXED_WORKERS)}
    for result in results:
        actual_assignments[result["worker_id"]].append(result["pair_index"])
    for pair_indices in actual_assignments.values():
        pair_indices.sort()
    if actual_assignments != expected_assignments:
        raise E2ProducerError("worker_assignment_mismatch")
    if len(results) != EXPECTED_PLAN["witness_pair_count"]:
        raise E2ProducerError("worker_pair_count_mismatch")
    if runtime_audit is not None:
        runtime_audit.clear()
        runtime_audit.update({
            "parent_pid": os.getpid(),
            "worker_assignments": actual_assignments,
            "worker_pids": [done[index] for index in range(FIXED_WORKERS)],
        })
    return sorted(results, key=lambda item: item["pair_index"])


def produce(
    output_dir: Path,
    v0_6_contract: Path,
    expected_v0_6_sha256: str,
    parent_contract: Path,
    expected_parent_sha256: str,
    runtime_audit: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _load_v0_6_contract(v0_6_contract, expected_v0_6_sha256)
    if expected_parent_sha256 != PARENT_CONTRACT_SHA256:
        raise E2ProducerError("expected_parent_contract_sha256_not_frozen")
    verify_parent_contract(parent_contract, expected_parent_sha256)
    if TOTAL_TUPLES != PARENT_DOMAIN_CARDINALITY:
        raise E2ProducerError("parent_enumerator_cardinality_changed")
    if output_dir.exists():
        raise E2ProducerError("output_directory_already_exists")

    stage = output_dir.with_name(f".{output_dir.name}.tmp-{os.getpid()}")
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    artifact_payloads: dict[str, bytes] = {}
    pairs: list[dict[str, Any]] = []
    try:
        worker_results = _run_fixed_workers(runtime_audit)
        for result in worker_results:
            pair = result["pair"]
            pairs.append(pair)
            for role, payloads in (
                ("left", result["left_payloads"]),
                ("right", result["right_payloads"]),
            ):
                for field, payload in payloads.items():
                    descriptor = pair[role][field]
                    written = _write_sidecar(
                        stage,
                        Path(descriptor["path"]),
                        payload,
                        width=descriptor["width"],
                        height=descriptor["height"],
                        format_name=descriptor["format"],
                        artifact_payloads=artifact_payloads,
                    )
                    if written != descriptor:
                        raise E2ProducerError(
                            f"worker_descriptor_mismatch:{pair['pair_id']}:{role}:{field}"
                        )

        if len(pairs) != EXPECTED_PLAN["witness_pair_count"]:
            raise E2ProducerError("witness_pair_count_mismatch")
        artifact_total_bytes = sum(len(payload) for payload in artifact_payloads.values())
        manifest = {
            "artifact_count": len(artifact_payloads),
            "artifact_total_bytes": artifact_total_bytes,
            "artifact_tree_sha256": _artifact_tree_hash(artifact_payloads),
            "contract_binding": {
                "parent_contract_sha256": expected_parent_sha256,
                "parent_domain_cardinality": PARENT_DOMAIN_CARDINALITY,
                "v0_6_contract_sha256": expected_v0_6_sha256,
            },
            "pair_count": len(pairs),
            "pairs": pairs,
            "schema_version": SCHEMA_VERSION,
            "scope": {
                "claim": "exact_full_domain_existential_cross_tuple_full_bytes",
                "full_domain_collision_rate_claimed": False,
                "sampled_to_full_extrapolation_used": False,
                "training_authorized": False,
            },
            "status": "passed",
            "witness_plan": EXPECTED_PLAN,
        }
        manifest_bytes = canonical_bytes(manifest)
        (stage / "manifest.json").write_bytes(manifest_bytes)
        validation = {
            "artifact_count": len(artifact_payloads),
            "artifact_total_bytes": artifact_total_bytes,
            "canonical_gray_full_byte_equal_pair_count": len(pairs),
            "canonical_png_full_byte_equal_pair_count": len(pairs),
            "graph_signature_distinct_pair_count": len(pairs),
            "interpretation_membership_count": 2 * len(pairs),
            "manifest_path": "manifest.json",
            "manifest_sha256": sha256(manifest_bytes),
            "pair_count": len(pairs),
            "padded_clipping_certificate_count": 2 * len(pairs),
            "raw_gray_full_byte_distinct_pair_count": len(pairs),
            "schema_version": VALIDATION_SCHEMA_VERSION,
            "status": "passed",
            "training_authorized": False,
            "tuple_membership_count": 2 * len(pairs),
            "v0_6_contract_sha256": expected_v0_6_sha256,
            "worker_assignments": [
                {
                    "pair_ordinals": list(range(
                        worker_id,
                        EXPECTED_PLAN["witness_pair_count"],
                        FIXED_WORKERS,
                    )),
                    "worker_id": worker_id,
                }
                for worker_id in range(FIXED_WORKERS)
            ],
            "worker_count": FIXED_WORKERS,
            "worker_process_distinct_count": FIXED_WORKERS,
        }
        (stage / "validation.json").write_bytes(canonical_bytes(validation))
        stage.rename(output_dir)
        return manifest, validation
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--v0-6-contract", type=Path, required=True)
    parser.add_argument(
        "--expected-v0-6-sha256", default=V0_6_CONTRACT_SHA256
    )
    parser.add_argument("--parent-contract", type=Path, required=True)
    parser.add_argument(
        "--expected-parent-sha256", default=PARENT_CONTRACT_SHA256
    )
    args = parser.parse_args()
    _, validation = produce(
        args.output_dir,
        args.v0_6_contract,
        args.expected_v0_6_sha256,
        args.parent_contract,
        args.expected_parent_sha256,
    )
    sys.stdout.buffer.write(canonical_bytes(validation))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
