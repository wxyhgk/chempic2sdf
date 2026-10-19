"""Execute and audit the frozen non-constructor sandbox probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any


PROBE_IN_BUNDLE = (
    "/app/formal/chemgraph/audit/owner_capacity_confirmation_v0_5/sandbox_probe.py"
)


class SandboxProbeError(ValueError):
    """The non-constructor isolation probe did not satisfy its closed contract."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SandboxProbeError("PROBE_DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def _load_probe(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    value = json.loads(raw, object_pairs_hook=_strict_object)
    if raw != _canonical(value):
        raise SandboxProbeError("PROBE_NONCANONICAL")
    expected_keys = {
        "constructor_imported", "fork_errno", "network_errno", "outside_write_errno",
        "schema_version", "training_authorized",
    }
    if set(value) != expected_keys:
        raise SandboxProbeError("PROBE_NOT_CLOSED")
    if not (
        value["constructor_imported"] is False
        and value["training_authorized"] is False
        and value["schema_version"] == "owner-capacity-sandbox-probe-v0.5"
        and all(type(value[key]) is int and value[key] > 0 for key in (
            "fork_errno", "network_errno", "outside_write_errno"
        ))
    ):
        raise SandboxProbeError("PROBE_RESULT_INVALID")
    return value, raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--bundle-sha256", required=True)
    parser.add_argument("--executor-commit", required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{64}", args.bundle_sha256):
        raise SandboxProbeError("BUNDLE_SHA_INVALID")
    if not re.fullmatch(r"[0-9a-f]{40}", args.executor_commit):
        raise SandboxProbeError("EXECUTOR_COMMIT_INVALID")
    bundle = args.bundle.resolve()
    run = args.run_directory.resolve()
    if args.run_directory.exists():
        raise SandboxProbeError("RUN_DIRECTORY_ALREADY_EXISTS")
    output = run / "out"
    trace = run / "trace"
    output.mkdir(parents=True, mode=0o700)
    trace.mkdir(mode=0o700)
    command = [
        "/usr/bin/strace", "-ff", "-ttt", "-T", "-v", "-yy", "-s", "65535",
        "-e", "trace=all", "-o", str(trace / "syscalls"),
        "/usr/bin/bwrap", "--die-with-parent", "--new-session", "--unshare-user",
        "--unshare-pid", "--unshare-net", "--unshare-ipc", "--unshare-uts",
        "--unshare-cgroup", "--cap-drop", "ALL", "--clearenv",
        "--ro-bind", str(bundle), "/app", "--ro-bind", "/usr", "/usr",
        "--ro-bind", "/lib", "/lib", "--ro-bind", "/lib64", "/lib64",
        "--bind", str(output), "/out", "--dir", "/dev",
        "--dev-bind", "/dev/null", "/dev/null", "--remount-ro", "/",
        "--setenv", "PATH", "/usr/bin", "--setenv", "HOME", "/nonexistent",
        "--setenv", "LANG", "C", "--setenv", "LC_ALL", "C", "--setenv", "TZ", "UTC",
        "/usr/bin/python3", "-I", "-S", "-B", PROBE_IN_BUNDLE,
    ]
    completed = subprocess.run(command, capture_output=True, check=False, timeout=30)
    if completed.returncode != 0 or completed.stdout or completed.stderr:
        raise SandboxProbeError("PROBE_PROCESS_FAILED")
    if sorted(path.name for path in output.iterdir()) != ["probe.json"]:
        raise SandboxProbeError("PROBE_OUTPUT_SET_NOT_CLOSED")
    result, raw = _load_probe(output / "probe.json")
    trace_files = sorted(trace.glob("syscalls.*"))
    if not trace_files:
        raise SandboxProbeError("PROBE_TRACE_MISSING")
    trace_text = "\n".join(path.read_text(encoding="utf-8") for path in trace_files)
    required = (
        'execve("/usr/bin/python3"',
        '"/outside-probe"',
        "socket(",
    )
    if any(token not in trace_text for token in required):
        raise SandboxProbeError("PROBE_TRACE_EVIDENCE_MISSING")
    if not ("clone(" in trace_text or "clone3(" in trace_text or "fork(" in trace_text):
        raise SandboxProbeError("PROBE_FORK_TRACE_MISSING")
    receipt = {
        "bundle_sha256": args.bundle_sha256,
        "constructor_called": False,
        "executor_commit": args.executor_commit,
        "probe_output_bytes": len(raw),
        "probe_output_sha256": _sha256(raw),
        "result": result,
        "schema_version": "owner-capacity-sandbox-probe-receipt-v0.5",
        "status": "passed",
        "trace_files": [
            {"bytes": path.stat().st_size, "name": path.name, "sha256": _sha256(path.read_bytes())}
            for path in trace_files
        ],
        "training_authorized": False,
    }
    receipt_path = run / "probe_receipt.json"
    descriptor = os.open(
        receipt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400
    )
    try:
        os.write(descriptor, _canonical(receipt))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


if __name__ == "__main__":
    main()
