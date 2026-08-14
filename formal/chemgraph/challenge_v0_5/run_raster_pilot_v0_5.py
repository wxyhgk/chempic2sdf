#!/usr/bin/env python3
"""Run the frozen 1% v0.5 raster/preprocessor resource pilot."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import struct
import subprocess
import time
from typing import Any, Iterable
import zlib

try:
    from .canonical_raster_v0_5 import (
        CONTRACT_SHA256, foreground_bbox, preprocess_gray, require_white_border,
        verify_contract,
    )
    from .domain_scene_v0_5 import (
        render_and_verify_padded_certificate, render_centered_gray,
    )
    from .enumerate_domain_v0_5 import (
        FIXED_WORKERS, TOTAL_TUPLES, canonical_json_bytes, decode_tuple_id,
    )
except ImportError:
    from canonical_raster_v0_5 import (  # type: ignore
        CONTRACT_SHA256, foreground_bbox, preprocess_gray, require_white_border,
        verify_contract,
    )
    from domain_scene_v0_5 import (  # type: ignore
        render_and_verify_padded_certificate, render_centered_gray,
    )
    from enumerate_domain_v0_5 import (  # type: ignore
        FIXED_WORKERS, TOTAL_TUPLES, canonical_json_bytes, decode_tuple_id,
    )


SCHEMA_VERSION = "chemgraph-raster-pilot-v0.5"
RECORD_DOMAIN = b"chemgraph-v0.5/tuple-raster-membership"
PILOT_COUNT = (TOTAL_TUPLES - 1) // 100 + 1
CAP_INDEX = {"butt": 0, "round": 1, "square": 2}
AA_INDEX = {"binary_1x": 0, "box_2x2": 1, "box_4x4": 2}
REFLECTION_INDEX = {"none": 0, "x": 1, "y": 2}
SORT_BINARY = "/usr/bin/sort"


class RasterPilotError(RuntimeError):
    """The resource pilot violated a frozen v0.5 condition."""


def verify_source_freeze(path: Path, expected_sha256: str, workspace: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise RasterPilotError(f"source_freeze_sha256_mismatch:{actual}")
    try:
        record = json.loads(payload)
    except json.JSONDecodeError as error:
        raise RasterPilotError(f"source_freeze_json_error:{error}") from error
    if record.get("schema_version") != "chemgraph-observation-closure-p1-source-freeze-v0.5":
        raise RasterPilotError("source_freeze_schema_mismatch")
    if record.get("status") != "p1_source_frozen" or record.get("training_authorized") is not False:
        raise RasterPilotError("source_freeze_status_mismatch")
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise RasterPilotError("source_freeze_artifacts_missing")
    for artifact in artifacts:
        if set(artifact) != {"path", "sha256"}:
            raise RasterPilotError("source_freeze_artifact_fields")
        artifact_path = workspace / artifact["path"]
        if not artifact_path.is_file():
            raise RasterPilotError(f"source_freeze_artifact_missing:{artifact['path']}")
        digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        if digest != artifact["sha256"]:
            raise RasterPilotError(f"source_freeze_artifact_changed:{artifact['path']}")
    return record


def pilot_tuple_ids() -> range:
    return range(0, TOTAL_TUPLES, 100)


def _worker_init() -> None:
    identity = mp.current_process()._identity  # Stable Pool worker ordinal.
    worker_index = (identity[0] - 1) % FIXED_WORKERS if identity else 0
    os.sched_setaffinity(0, {worker_index})


def _axes(record: dict[str, Any]) -> dict[str, int]:
    return {
        "half_length": record["half_length"],
        "motif_ordinal": record["motif_ordinal"],
        "width_px": record["width_px"],
        "cap_index": CAP_INDEX[record["cap"]],
        "antialias_index": AA_INDEX[record["antialias_profile"]],
        "reflection_index": REFLECTION_INDEX[record["reflection"]],
        "rotation_index": record["rotation_index"],
        "scale_index": record["scale_index"],
    }


def _render(tuple_id: int, verify_padded: bool = True) -> tuple[dict[str, Any], bytes]:
    record = decode_tuple_id(tuple_id)
    axes = _axes(record)
    if verify_padded:
        gray, motif = render_and_verify_padded_certificate(**axes)
    else:
        gray, motif = render_centered_gray(**axes)
    raw_bbox = require_white_border(gray, 33)
    canonical = preprocess_gray(gray)
    canonical_bbox = foreground_bbox(canonical)
    crop_width = raw_bbox[2] - raw_bbox[0] + 1
    crop_height = raw_bbox[3] - raw_bbox[1] + 1
    result = {
        "schema_version": "chemgraph-raster-pilot-record-v0.5",
        "tuple_id": tuple_id,
        "raw_gray_sha256": hashlib.sha256(gray).hexdigest(),
        "canonical_gray_sha256": hashlib.sha256(canonical).hexdigest(),
        "crop_width": crop_width,
        "crop_height": crop_height,
        "raw_bbox": list(raw_bbox),
        "canonical_bbox": list(canonical_bbox),
        "interpretation_count": len(motif.interpretations),
    }
    return result, canonical


def _render_artifact(tuple_id: int) -> tuple[dict[str, Any], bytes]:
    record, canonical = _render(tuple_id)
    return record, zlib.compress(canonical, 1)


def _render_compressed(tuple_id: int) -> tuple[int, str, bytes]:
    record, canonical = _render(tuple_id, verify_padded=False)
    return tuple_id, record["canonical_gray_sha256"], zlib.compress(canonical, 1)


def _read_frame(handle: Any, offset: int | None = None) -> tuple[int, bytes]:
    if offset is not None:
        handle.seek(offset)
    header = handle.read(12)
    if len(header) != 12:
        raise RasterPilotError("canonical_frame_truncated_header")
    tuple_id, length = struct.unpack(">QI", header)
    payload = handle.read(length)
    if len(payload) != length:
        raise RasterPilotError("canonical_frame_truncated_payload")
    return tuple_id, zlib.decompress(payload)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def run_pilot(
    contract_path: Path,
    source_freeze_path: Path,
    expected_source_freeze_sha256: str,
    workspace: Path,
    output: Path,
) -> dict[str, Any]:
    verify_contract(contract_path, CONTRACT_SHA256)
    source_freeze = verify_source_freeze(
        source_freeze_path, expected_source_freeze_sha256, workspace,
    )
    if output.exists():
        if any(output.iterdir()):
            raise RasterPilotError(f"refusing_nonempty_output:{output}")
        output.rmdir()
    stage = output.with_name(f".{output.name}.stage-{os.getpid()}")
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    records_path = stage / "records.jsonl"
    frames_path = stage / "canonical_frames.zlib"
    offsets_path = stage / "canonical_frame_offsets.u64be"
    keys_path = stage / "class_keys.tsv"
    sorted_keys_path = stage / "class_keys.sorted.tsv"
    sort_tmp = stage / "sort_tmp"
    sort_tmp.mkdir()
    crop_widths: dict[int, int] = defaultdict(int)
    crop_heights: dict[int, int] = defaultdict(int)
    interpretation_counts: dict[int, int] = defaultdict(int)
    records_hasher = hashlib.sha256()
    records_raw_hasher = hashlib.sha256()

    context = mp.get_context("fork")
    with context.Pool(FIXED_WORKERS, initializer=_worker_init) as pool:
        render_started = time.perf_counter()
        frame_offsets: list[int] = []
        with (records_path.open("xb") as handle,
              keys_path.open("xb") as key_handle,
              frames_path.open("xb") as frame_handle,
              offsets_path.open("xb") as offset_handle):
            for index, (record, compressed) in enumerate(
                pool.imap(_render_artifact, pilot_tuple_ids(), chunksize=16), start=1
            ):
                payload = canonical_json_bytes(record)
                handle.write(payload)
                records_raw_hasher.update(payload)
                records_hasher.update(len(RECORD_DOMAIN).to_bytes(4, "big"))
                records_hasher.update(RECORD_DOMAIN)
                records_hasher.update(len(payload).to_bytes(8, "big"))
                records_hasher.update(payload)
                key_handle.write(
                    f'{record["canonical_gray_sha256"]}\t{record["tuple_id"]:08d}\n'.encode("ascii")
                )
                offset = frame_handle.tell()
                frame_offsets.append(offset)
                offset_handle.write(struct.pack(">Q", offset))
                frame_handle.write(struct.pack(">QI", record["tuple_id"], len(compressed)))
                frame_handle.write(compressed)
                crop_widths[record["crop_width"]] += 1
                crop_heights[record["crop_height"]] += 1
                interpretation_counts[record["interpretation_count"]] += 1
            handle.flush()
            os.fsync(handle.fileno())
            key_handle.flush()
            os.fsync(key_handle.fileno())
            frame_handle.flush()
            os.fsync(frame_handle.fileno())
            offset_handle.flush()
            os.fsync(offset_handle.fileno())
        render_wall_nanoseconds = int((time.perf_counter() - render_started) * 1_000_000_000)

        if not Path(SORT_BINARY).is_file():
            raise RasterPilotError("gnu_sort_not_found")
        subprocess.run(
            [SORT_BINARY, "--parallel=16", f"--temporary-directory={sort_tmp}",
             "-o", str(sorted_keys_path), str(keys_path)],
            check=True,
            env={"LC_ALL": "C", "PATH": os.environ.get("PATH", "")},
        )
        with sorted_keys_path.open("rb") as sorted_handle:
            os.fsync(sorted_handle.fileno())

        digest_members: dict[str, list[int]] = defaultdict(list)
        with sorted_keys_path.open("rt", encoding="ascii") as sorted_handle:
            for line in sorted_handle:
                digest, tuple_text = line.rstrip("\n").split("\t")
                digest_members[digest].append(int(tuple_text))

        expected_digest = {
            tuple_id: digest for digest, members in digest_members.items() for tuple_id in members
        }
        duplicate_digests = {
            digest for digest, members in digest_members.items() if len(members) > 1
        }
        verified_duplicates = 0
        with frames_path.open("rb") as frame_handle:
            for digest in sorted(duplicate_digests):
                members = digest_members[digest]
                first_id = members[0]
                first_index = first_id // 100
                stored_id, representative = _read_frame(
                    frame_handle, frame_offsets[first_index],
                )
                if stored_id != first_id:
                    raise RasterPilotError("canonical_frame_tuple_mismatch")
                for tuple_id in members:
                    member_index = tuple_id // 100
                    stored_id, canonical = _read_frame(
                        frame_handle, frame_offsets[member_index],
                    )
                    if stored_id != tuple_id or canonical != representative:
                        raise RasterPilotError(
                            f"first_pass_digest_collision_different_bytes:{digest}:{tuple_id}"
                        )
                    verified_duplicates += 1

        replay_started = time.perf_counter()
        with frames_path.open("rb") as frame_handle:
            for index, (tuple_id, digest, compressed) in enumerate(
                pool.imap(_render_compressed, pilot_tuple_ids(), chunksize=8), start=1
            ):
                if digest != expected_digest[tuple_id]:
                    raise RasterPilotError(
                        f"replay_digest_changed:{tuple_id}:{expected_digest[tuple_id]}:{digest}"
                    )
                stored_id, first_canonical = _read_frame(frame_handle)
                replay_canonical = zlib.decompress(compressed)
                if stored_id != tuple_id or first_canonical != replay_canonical:
                    raise RasterPilotError(f"replay_full_bytes_changed:{tuple_id}")
        if index != PILOT_COUNT:
            raise RasterPilotError(f"replay_count_mismatch:{index}:{PILOT_COUNT}")
        replay_wall_nanoseconds = int((time.perf_counter() - replay_started) * 1_000_000_000)

    records_bytes = records_path.stat().st_size
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed_pending_resource_gate",
        "contract_sha256": CONTRACT_SHA256,
        "p1_source_commit": source_freeze["p1_source_commit"],
        "p1_source_freeze_sha256": expected_source_freeze_sha256,
        "selection": "tuple_id mod 100 == 0",
        "tuple_count": PILOT_COUNT,
        "workers": FIXED_WORKERS,
        "worker_render_wall_nanoseconds": render_wall_nanoseconds,
        "worker_replay_wall_nanoseconds": replay_wall_nanoseconds,
        "padded_clipping_certificates": PILOT_COUNT,
        "records_byte_length": records_bytes,
        "records_sha256": records_raw_hasher.hexdigest(),
        "records_framed_root": records_hasher.hexdigest(),
        "canonical_frame_count": PILOT_COUNT,
        "distinct_raster_classes": len(digest_members),
        "duplicate_member_count": sum(len(items) for items in digest_members.values()
                                      if len(items) > 1),
        "duplicate_full_bytes_verified": verified_duplicates,
        "deterministic_replay_count": PILOT_COUNT,
        "maximum_raster_class_members": max(map(len, digest_members.values())),
        "crop_width_histogram": [[key, crop_widths[key]] for key in sorted(crop_widths)],
        "crop_height_histogram": [[key, crop_heights[key]] for key in sorted(crop_heights)],
        "interpretation_count_histogram": [
            [key, interpretation_counts[key]] for key in sorted(interpretation_counts)
        ],
        "training_authorized": False,
    }
    validation = canonical_json_bytes(result)
    validation_path = stage / "validation.json"
    with validation_path.open("xb") as handle:
        handle.write(validation)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(stage)
    os.replace(stage, output)
    _fsync_directory(output.parent)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--expected-contract-sha256", required=True)
    parser.add_argument("--source-freeze", type=Path, required=True)
    parser.add_argument("--expected-source-freeze-sha256", required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    if args.expected_contract_sha256 != CONTRACT_SHA256:
        raise RasterPilotError("expected_contract_sha256_not_frozen")
    if args.workers != FIXED_WORKERS:
        raise RasterPilotError("workers_must_equal_16")
    result = run_pilot(
        args.contract, args.source_freeze, args.expected_source_freeze_sha256,
        args.workspace, args.output,
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
