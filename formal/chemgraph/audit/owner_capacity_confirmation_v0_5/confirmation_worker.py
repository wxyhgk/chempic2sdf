"""Sandbox worker that invokes the frozen constructor on one committed source."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
from pathlib import Path
from typing import Any


RESOURCE_LIMITS = {
    "address_space_bytes": 2_147_483_648,
    "core_bytes": 0,
    "cpu_seconds": 120,
    "file_bytes": 20_971_520,
    "open_files": 64,
    "processes": 1,
}


def _apply_resource_limits() -> None:
    pairs = (
        (resource.RLIMIT_AS, RESOURCE_LIMITS["address_space_bytes"]),
        (resource.RLIMIT_CORE, RESOURCE_LIMITS["core_bytes"]),
        (resource.RLIMIT_CPU, RESOURCE_LIMITS["cpu_seconds"]),
        (resource.RLIMIT_FSIZE, RESOURCE_LIMITS["file_bytes"]),
        (resource.RLIMIT_NOFILE, RESOURCE_LIMITS["open_files"]),
        (resource.RLIMIT_NPROC, RESOURCE_LIMITS["processes"]),
    )
    for resource_id, value in pairs:
        resource.setrlimit(resource_id, (value, value))

APP_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(APP_ROOT))


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_write(path: Path, value: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    temporary.unlink()
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def main() -> None:
    _apply_resource_limits()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expected-kind", choices=("candidate", "rejection"), required=True
    )
    args = parser.parse_args()
    expected_output = Path(
        "/out/candidate.json" if args.expected_kind == "candidate" else "/out/rejection.json"
    )
    if args.output != expected_output:
        print("CAPACITY_WORKER_FAILURE:OUTPUT_PATH_INVALID", file=sys.stderr)
        raise SystemExit(40)
    raw = args.source.read_bytes()
    if _sha256(raw) != args.source_sha256:
        print("CAPACITY_WORKER_FAILURE:SOURCE_SHA_MISMATCH", file=sys.stderr)
        raise SystemExit(41)
    try:
        source = json.loads(raw)
    except json.JSONDecodeError:
        print("CAPACITY_WORKER_FAILURE:SOURCE_JSON_INVALID", file=sys.stderr)
        raise SystemExit(42)
    if raw != _canonical_bytes(source):
        print("CAPACITY_WORKER_FAILURE:SOURCE_NONCANONICAL", file=sys.stderr)
        raise SystemExit(43)
    from formal.chemgraph.audit.owner_candidate_domain_v0_5.candidate_domain import (
        build_candidate_domain,
    )
    from formal.chemgraph.audit.owner_candidate_domain_v0_5.common import (
        CandidateDomainError,
    )

    try:
        artifact = build_candidate_domain(source)
    except CandidateDomainError as error:
        if args.expected_kind != "rejection":
            print(
                f"CAPACITY_WORKER_FAILURE:UNEXPECTED_REJECTION:{error}",
                file=sys.stderr,
            )
            raise SystemExit(44)
        _atomic_write(args.output, _canonical_bytes({
            "error": str(error),
            "schema_version": "owner-capacity-worker-rejection-v0.5",
            "status": "rejected",
            "training_authorized": False,
        }))
        raise SystemExit(20)
    if args.expected_kind != "candidate":
        print("CAPACITY_WORKER_FAILURE:UNEXPECTED_CANDIDATE", file=sys.stderr)
        raise SystemExit(45)
    _atomic_write(args.output, _canonical_bytes(artifact))


if __name__ == "__main__":
    main()
