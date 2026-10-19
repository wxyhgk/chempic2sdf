"""Closed bwrap/strace executor for preregistered owner-capacity cases.

Importing this module never invokes the candidate constructor. Formal cases may only
be executed from a committed read-only bundle after the execution protocol is frozen.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import resource
import signal
import stat
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
PLAN_COMMIT = "1fcd10c490755041a6afbff0d3ae3a48ca9aeab7"
PLAN_SHA256 = "06366e83731f1ca9dee116f995da14dd5a7e64c68f45e98f2a847c0e423bcd89"
MANIFEST_SHA256 = "a06489212869743c414d8f87729c4ff6d3836f7bec3b7c5ef8af908ad7410118"
EXCLUSION_SHA256 = "48e7918c9f940fc38e924764ea8e60d8d1b987cd638bdc931ee6acbc32b55a7d"
CASE_MATRIX_SHA256 = "f1c0905af213592293ce259cfe00d3398b3e36e4075fbb7b232403c6d8403206"
WORKER_IN_BUNDLE = (
    "/app/formal/chemgraph/audit/owner_capacity_confirmation_v0_5/"
    "confirmation_worker.py"
)
SOURCE_PREFIX = Path("formal/chemgraph/audit/owner_capacity_confirmation_v0_5")
WALL_TIMEOUT_SECONDS = 180
TRACE_FILE_LIMIT_BYTES = 268_435_456
TRACE_NETWORK_CALLS = {
    "accept", "accept4", "bind", "connect", "getpeername", "getsockname",
    "getsockopt", "listen", "recvfrom", "recvmmsg", "recvmsg", "sendmmsg",
    "sendmsg", "sendto", "setsockopt", "shutdown", "socket", "socketpair",
}
TRACE_PROCESS_CALLS = {"clone", "clone3", "execve", "execveat", "fork", "vfork"}
TRACE_MUTATING_PATH_CALLS = {
    "chmod", "chown", "creat", "fchmodat", "fchownat", "link", "linkat",
    "mkdir", "mkdirat", "mknod", "mknodat", "mount", "open", "openat",
    "openat2", "rename", "renameat", "renameat2", "rmdir", "symlink",
    "symlinkat", "truncate", "unlink", "unlinkat",
}


class IsolatedRunnerError(ValueError):
    """Frozen contract, sandbox execution, or closed outcome is invalid."""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _load_canonical(path: Path) -> tuple[Any, bytes]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw, object_pairs_hook=_strict_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IsolatedRunnerError(f"JSON_INVALID:{path.name}") from error
    if raw != _canonical_bytes(value):
        raise IsolatedRunnerError(f"NONCANONICAL:{path.name}")
    return value, raw


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise IsolatedRunnerError("JSON_DUPLICATE_KEY")
        result[key] = value
    return result


def frozen_contract() -> tuple[dict[str, Any], dict[str, Any]]:
    plan, plan_raw = _load_canonical(ROOT / "axis_plan.json")
    manifest, manifest_raw = _load_canonical(ROOT / "source_manifest.json")
    if _sha256(plan_raw) != PLAN_SHA256:
        raise IsolatedRunnerError("PLAN_SHA_MISMATCH")
    if _sha256(manifest_raw) != MANIFEST_SHA256:
        raise IsolatedRunnerError("MANIFEST_SHA_MISMATCH")
    if _sha256((ROOT / "EXPLORATORY_EXCLUSION.md").read_bytes()) != EXCLUSION_SHA256:
        raise IsolatedRunnerError("EXCLUSION_SHA_MISMATCH")
    return plan, manifest


def _case_matrix() -> list[dict[str, Any]]:
    value, raw = _load_canonical(ROOT / "case_matrix.json")
    if _sha256(raw) != CASE_MATRIX_SHA256:
        raise IsolatedRunnerError("CASE_MATRIX_SHA_MISMATCH")
    if set(value) != {"cases", "schema_version", "training_authorized"}:
        raise IsolatedRunnerError("CASE_MATRIX_NOT_CLOSED")
    if value["training_authorized"] is not False or len(value["cases"]) != 8:
        raise IsolatedRunnerError("CASE_MATRIX_INVALID")
    return value["cases"]


def case_record(case_id: str) -> dict[str, Any]:
    matches = [item for item in _case_matrix() if item["case_id"] == case_id]
    if len(matches) != 1:
        raise IsolatedRunnerError("CASE_NOT_ALLOWLISTED")
    record = matches[0]
    source = source_record(record["axis"], record["side"])
    if (
        source["source_path"] != record["source_path"]
        or source["raw_source_sha256"] != record["source_sha256"]
    ):
        raise IsolatedRunnerError("CASE_SOURCE_MANIFEST_MISMATCH")
    return record


def source_record(axis: str, side: str) -> dict[str, Any]:
    _, manifest = frozen_contract()
    matches = [
        item for item in manifest["records"]
        if item["axis"] == axis and item["side"] == side
    ]
    if len(matches) != 1:
        raise IsolatedRunnerError("SOURCE_NOT_ALLOWLISTED")
    return matches[0]


def _require_absolute_nonrepo(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise IsolatedRunnerError(f"{label}_NOT_ABSOLUTE")
    resolved = path.resolve()
    repository = REPOSITORY_ROOT.resolve()
    if resolved == repository or resolved.is_relative_to(repository):
        raise IsolatedRunnerError(f"{label}_INSIDE_LIVE_REPOSITORY")
    return resolved


def _require_empty_private_directory(path: Path, label: str) -> Path:
    resolved = _require_absolute_nonrepo(path, label)
    if path.is_symlink() or not resolved.is_dir():
        raise IsolatedRunnerError(f"{label}_NOT_REAL_DIRECTORY")
    if stat.S_IMODE(resolved.stat().st_mode) != 0o700:
        raise IsolatedRunnerError(f"{label}_MODE_INVALID")
    if any(resolved.iterdir()):
        raise IsolatedRunnerError(f"{label}_NOT_EMPTY")
    return resolved


def sandbox_command(
    *, bundle_root: Path, output_directory: Path, trace_prefix: Path, case_id: str,
) -> list[str]:
    record = case_record(case_id)
    bundle = _require_absolute_nonrepo(bundle_root, "BUNDLE")
    output = _require_absolute_nonrepo(output_directory, "OUTPUT")
    trace = _require_absolute_nonrepo(trace_prefix.parent, "TRACE") / trace_prefix.name
    input_file = bundle / SOURCE_PREFIX / record["source_path"]
    if input_file.is_symlink() or not input_file.is_file():
        raise IsolatedRunnerError("BUNDLE_SOURCE_NOT_REGULAR")
    if _sha256(input_file.read_bytes()) != record["source_sha256"]:
        raise IsolatedRunnerError("BUNDLE_SOURCE_SHA_MISMATCH")
    output_name = "candidate.json" if record["expected_kind"] == "candidate" else "rejection.json"
    return [
        "/usr/bin/strace", "-ff", "-ttt", "-T", "-v", "-yy", "-s", "65535",
        "-e", "trace=all", "-o", str(trace),
        "/usr/bin/bwrap", "--die-with-parent", "--new-session", "--unshare-user",
        "--unshare-pid", "--unshare-net", "--unshare-ipc", "--unshare-uts",
        "--unshare-cgroup", "--cap-drop", "ALL", "--clearenv",
        "--ro-bind", str(bundle), "/app",
        "--ro-bind", "/usr", "/usr", "--ro-bind", "/lib", "/lib",
        "--ro-bind", "/lib64", "/lib64", "--dir", "/input",
        "--ro-bind", str(input_file), "/input/observation.json",
        "--bind", str(output), "/out", "--dir", "/dev",
        "--dev-bind", "/dev/null", "/dev/null", "--remount-ro", "/",
        "--setenv", "PATH", "/usr/bin", "--setenv", "HOME", "/nonexistent",
        "--setenv", "LANG", "C", "--setenv", "LC_ALL", "C", "--setenv", "TZ", "UTC",
        "/usr/bin/python3", "-I", "-S", "-B", WORKER_IN_BUNDLE,
        "--source", "/input/observation.json",
        "--source-sha256", record["source_sha256"],
        "--output", f"/out/{output_name}", "--expected-kind", record["expected_kind"],
    ]


def _trace_files(trace_prefix: Path) -> list[Path]:
    return sorted(trace_prefix.parent.glob(f"{trace_prefix.name}.*"))


def validate_trace(trace_prefix: Path) -> dict[str, Any]:
    files = _trace_files(trace_prefix)
    if not files:
        raise IsolatedRunnerError("TRACE_MISSING")
    python_seen = 0
    network_attempts: list[str] = []
    post_python_process_calls: list[str] = []
    outside_output_mutations: list[str] = []
    output_mutation_count = 0
    malformed: list[str] = []
    for path in files:
        raw = path.read_bytes()
        if len(raw) > TRACE_FILE_LIMIT_BYTES:
            raise IsolatedRunnerError("TRACE_FILE_LIMIT_EXCEEDED")
        text = raw.decode("utf-8", errors="strict")
        if "<unfinished ...>" in text or "resumed>" in text:
            malformed.append(path.name)
        in_python = False
        for line in text.splitlines():
            match = re.search(r"\b([a-zA-Z0-9_]+)\(", line)
            if not match:
                continue
            syscall = match.group(1)
            if syscall in {"execve", "execveat"} and '"/usr/bin/python3"' in line:
                python_seen += 1
                in_python = True
                continue
            if not in_python:
                continue
            if syscall in TRACE_NETWORK_CALLS:
                network_attempts.append(f"{path.name}:{syscall}")
            if syscall in TRACE_PROCESS_CALLS:
                post_python_process_calls.append(f"{path.name}:{syscall}")
            mutating = syscall in TRACE_MUTATING_PATH_CALLS
            if syscall in {"open", "openat", "openat2"}:
                mutating = any(
                    flag in line
                    for flag in (
                        "O_WRONLY", "O_RDWR", "O_CREAT", "O_TRUNC",
                        "O_APPEND", "O_TMPFILE",
                    )
                )
            if mutating:
                quoted_paths = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', line)
                relevant = [value for value in quoted_paths if value.startswith("/")]
                if relevant and all(value.startswith("/out/") for value in relevant):
                    output_mutation_count += 1
                else:
                    outside_output_mutations.append(f"{path.name}:{syscall}")
        if not re.search(r"\+\+\+ (exited with|killed by)", text):
            malformed.append(path.name)
    if python_seen != 1:
        raise IsolatedRunnerError("TRACE_PYTHON_EXEC_COUNT_INVALID")
    if malformed:
        raise IsolatedRunnerError("TRACE_INCOMPLETE_OR_UNPARSEABLE")
    if network_attempts:
        raise IsolatedRunnerError("TRACE_NETWORK_ATTEMPT")
    if post_python_process_calls:
        raise IsolatedRunnerError("TRACE_POST_PYTHON_PROCESS_ATTEMPT")
    if outside_output_mutations:
        raise IsolatedRunnerError("TRACE_OUTSIDE_OUTPUT_MUTATION_ATTEMPT")
    return {
        "network_attempt_count": 0,
        "output_mutation_count": output_mutation_count,
        "outside_output_mutation_count": 0,
        "post_python_process_attempt_count": 0,
        "python_exec_count": 1,
        "trace_files": [
            {"bytes": path.stat().st_size, "name": path.name, "sha256": _sha256(path.read_bytes())}
            for path in files
        ],
    }


def validate_closed_outcome(
    *, case: dict[str, Any], output_directory: Path, exit_code: int,
    stdout: bytes, stderr: bytes, timed_out: bool, source_pre_sha256: str,
    source_post_sha256: str, trace_validation: dict[str, Any],
) -> dict[str, Any]:
    if timed_out:
        raise IsolatedRunnerError("CASE_TIMEOUT")
    if source_pre_sha256 != case["source_sha256"] or source_post_sha256 != case["source_sha256"]:
        raise IsolatedRunnerError("CASE_SOURCE_TOCTOU")
    if exit_code != case["expected_exit_code"]:
        raise IsolatedRunnerError("CASE_EXIT_CODE_MISMATCH")
    if stdout or stderr:
        raise IsolatedRunnerError("CASE_STDIO_NOT_EMPTY")
    expected_name = "candidate.json" if case["expected_kind"] == "candidate" else "rejection.json"
    names = sorted(path.name for path in output_directory.iterdir())
    if names != [expected_name]:
        raise IsolatedRunnerError("CASE_OUTPUT_SET_NOT_CLOSED")
    output = output_directory / expected_name
    if output.is_symlink() or not output.is_file():
        raise IsolatedRunnerError("CASE_OUTPUT_NOT_REGULAR")
    value, raw = _load_canonical(output)
    if case["expected_kind"] == "candidate":
        if not isinstance(value, dict) or value.get("status") != "candidate_domain":
            raise IsolatedRunnerError("CASE_CANDIDATE_KIND_INVALID")
    else:
        expected = {
            "error": case["expected_error"],
            "schema_version": "owner-capacity-worker-rejection-v0.5",
            "status": "rejected",
            "training_authorized": False,
        }
        if value != expected:
            raise IsolatedRunnerError("CASE_REJECTION_NOT_EXACT")
    return {
        "exit_code": exit_code,
        "output_bytes": len(raw),
        "output_kind": case["expected_kind"],
        "output_name": expected_name,
        "output_sha256": _sha256(raw),
        "source_post_sha256": source_post_sha256,
        "source_pre_sha256": source_pre_sha256,
        "stderr_bytes": 0,
        "stderr_sha256": _sha256(stderr),
        "stdout_bytes": 0,
        "stdout_sha256": _sha256(stdout),
        "timed_out": False,
        "trace_validation": trace_validation,
    }


def _limit_supervisor_trace() -> None:
    resource.setrlimit(
        resource.RLIMIT_FSIZE, (TRACE_FILE_LIMIT_BYTES, TRACE_FILE_LIMIT_BYTES)
    )


def execute_case(
    *, case_id: str, bundle_root: Path, output_directory: Path, trace_directory: Path,
) -> dict[str, Any]:
    case = case_record(case_id)
    output = _require_empty_private_directory(output_directory, "OUTPUT")
    trace = _require_empty_private_directory(trace_directory, "TRACE")
    source = bundle_root.resolve() / SOURCE_PREFIX / case["source_path"]
    source_pre_sha = _sha256(source.read_bytes())
    trace_prefix = trace / "syscalls"
    command = sandbox_command(
        bundle_root=bundle_root, output_directory=output,
        trace_prefix=trace_prefix, case_id=case_id,
    )
    started = time.monotonic_ns()
    timed_out = False
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True,
        start_new_session=True, preexec_fn=_limit_supervisor_trace,
    )
    try:
        stdout, stderr = process.communicate(timeout=WALL_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
    ended = time.monotonic_ns()
    source_post_sha = _sha256(source.read_bytes())
    trace_validation = validate_trace(trace_prefix)
    result = validate_closed_outcome(
        case=case, output_directory=output, exit_code=process.returncode,
        stdout=stdout, stderr=stderr, timed_out=timed_out,
        source_pre_sha256=source_pre_sha, source_post_sha256=source_post_sha,
        trace_validation=trace_validation,
    )
    result.update({
        "case_id": case_id,
        "command_sha256": _sha256(_canonical_bytes(command)),
        "duration_monotonic_ns": ended - started,
        "schema_version": "owner-capacity-isolated-run-v0.5",
        "status": "closed_outcome_verified",
        "training_authorized": False,
    })
    return result


def publish_authority_marker(directory: Path, receipt_sha256: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", receipt_sha256):
        raise IsolatedRunnerError("RECEIPT_SHA_INVALID")
    marker = directory / "AUTHORITY"
    descriptor = os.open(
        marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400
    )
    try:
        os.write(descriptor, (receipt_sha256 + "\n").encode("ascii"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
