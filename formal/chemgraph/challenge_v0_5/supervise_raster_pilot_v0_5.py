#!/usr/bin/env python3
"""Run and verify the v0.5 raster pilot in a read-only snapshot and hard limits."""

from __future__ import annotations

import argparse
import grp
import hashlib
import json
import math
import os
from pathlib import Path
import pwd
import shutil
import signal
import stat
import struct
import subprocess
import time
from typing import Any
import zlib


CONTRACT_SHA256 = "55ddcaacdf851166554ccd586b2a924c0e793f86cc7a6319f98132ef0ffb6b65"
TOTAL_TUPLES = 36_088_416
PILOT_TUPLES = 360_885
MEMORY_LIMIT = 85_899_345_920
CGROUP_MEMORY_MAX = 79 * 1024**3
WORKER_MEMORY_MAX = 78 * 1024**3
SUPERVISOR_MEMORY_MAX = 1 * 1024**3
PRE_MIGRATION_MEMORY_RESERVE = MEMORY_LIMIT - CGROUP_MEMORY_MAX
TEMPORARY_LIMIT = 2_199_023_255_552
TMPFS_SIZE = 8 * 1024**3
PILOT_WALL_LIMIT = 14_400
FULL_WALL_LIMIT = 345_600
PUBLICATION_RESERVE_SECONDS = 10
RECORD_DOMAIN = b"chemgraph-v0.5/tuple-raster-membership"
RESULT_ROOT = Path("/opt/chemgraph/results")
PRIVATE_RUN_UID = 61_184
PRIVATE_RUN_GID = 61_184
PRIVATE_RUN_NAME = "chemgraph-pilot"
REQUIRED_EXTERNAL_ROLES = {"apptainer", "mount", "runtime_sif", "umount"}
OOM_EVENT_KEYS = ("oom", "oom_kill", "oom_group_kill")
REQUIRED_SOURCE_ARTIFACTS = {
    "formal/chemgraph/ChemGraph.lean",
    "formal/chemgraph/ChemGraph/Counterexamples.lean",
    "formal/chemgraph/ChemGraph/EvidenceCore.lean",
    "formal/chemgraph/ChemGraph/FiniteAudit.lean",
    "formal/chemgraph/ChemGraph/Grammar.lean",
    "formal/chemgraph/ChemGraph/HiddenChallenge.lean",
    "formal/chemgraph/ChemGraph/Identifiability.lean",
    "formal/chemgraph/ChemGraph/Multiplicity.lean",
    "formal/chemgraph/ChemGraph/ObservationClosure.lean",
    "formal/chemgraph/ChemGraph/ObservationRecovery.lean",
    "formal/chemgraph/ChemGraph/RasterEvidence.lean",
    "formal/chemgraph/ChemGraph/Recovery.lean",
    "formal/chemgraph/ChemGraph/Render.lean",
    "formal/chemgraph/ChemGraph/Scene.lean",
    "formal/chemgraph/ChemGraph/Stability.lean",
    "formal/chemgraph/README.md",
    "formal/chemgraph/lake-manifest.json",
    "formal/chemgraph/lakefile.toml",
    "formal/chemgraph/lean-toolchain",
    "formal/chemgraph/challenge_v0_4/generate_pack_v0_4.py",
    "formal/chemgraph/challenge_v0_4/rasterizer_v0_4.py",
    "formal/chemgraph/challenge_v0_5/README.md",
    "formal/chemgraph/challenge_v0_5/__init__.py",
    "formal/chemgraph/challenge_v0_5/audit_domain_reference_v0_5.py",
    "formal/chemgraph/challenge_v0_5/canonical_raster_v0_5.py",
    "formal/chemgraph/challenge_v0_5/domain_scene_v0_5.py",
    "formal/chemgraph/challenge_v0_5/enumerate_domain_reference_v0_5.py",
    "formal/chemgraph/challenge_v0_5/enumerate_domain_v0_5.py",
    "formal/chemgraph/challenge_v0_5/prereg/p0_contract_v0_5.json",
    "formal/chemgraph/challenge_v0_5/run_raster_pilot_v0_5.py",
    "formal/chemgraph/challenge_v0_5/runtime/.gitignore",
    "formal/chemgraph/challenge_v0_5/runtime/README.md",
    "formal/chemgraph/challenge_v0_5/schemas/closure_manifest.schema.json",
    "formal/chemgraph/challenge_v0_5/schemas/domain_tuple.schema.json",
    "formal/chemgraph/challenge_v0_5/schemas/evidence_class.schema.json",
    "formal/chemgraph/challenge_v0_5/schemas/p0_contract.schema.json",
    "formal/chemgraph/challenge_v0_5/schemas/p1_source_freeze.schema.json",
    "formal/chemgraph/challenge_v0_5/schemas/raster_pilot_record.schema.json",
    "formal/chemgraph/challenge_v0_5/schemas/raster_pilot_resource.schema.json",
    "formal/chemgraph/challenge_v0_5/schemas/raster_pilot_validation.schema.json",
    "formal/chemgraph/challenge_v0_5/schemas/raster_class.schema.json",
    "formal/chemgraph/challenge_v0_5/supervise_raster_pilot_v0_5.py",
    "formal/chemgraph/challenge_v0_5/test_canonical_raster_v0_5.py",
    "formal/chemgraph/challenge_v0_5/test_domain_scene_v0_5.py",
    "formal/chemgraph/challenge_v0_5/test_enumerate_domain_reference_v0_5.py",
    "formal/chemgraph/challenge_v0_5/test_enumerate_domain_v0_5.py",
    "formal/chemgraph/challenge_v0_5/test_run_raster_pilot_v0_5.py",
    "formal/chemgraph/challenge_v0_5/test_supervise_raster_pilot_v0_5.py",
}


class SupervisorError(RuntimeError):
    """The authoritative resource supervisor failed closed."""


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("ascii")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_freeze(path: Path, expected_sha256: str, workspace: Path) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        raise SupervisorError("source_freeze_sha256_mismatch")
    record = json.loads(path.read_text(encoding="ascii"))
    if (record.get("schema_version") !=
            "chemgraph-observation-closure-p1-source-freeze-v0.5"):
        raise SupervisorError("source_freeze_schema_mismatch")
    if record.get("status") != "p1_source_frozen" or record.get("training_authorized") is not False:
        raise SupervisorError("source_freeze_status_mismatch")
    artifacts = record.get("artifacts", [])
    artifact_paths = {item.get("path") for item in artifacts}
    if artifact_paths != REQUIRED_SOURCE_ARTIFACTS or len(artifacts) != len(artifact_paths):
        raise SupervisorError("source_freeze_artifact_set_not_exact")
    for artifact in artifacts:
        target = workspace / artifact["path"]
        if not target.is_file() or sha256_file(target) != artifact["sha256"]:
            raise SupervisorError(f"source_artifact_changed:{artifact['path']}")
    dependencies = record.get("external_dependencies", [])
    roles = {item.get("role") for item in dependencies}
    if roles != REQUIRED_EXTERNAL_ROLES or len(dependencies) != len(roles):
        raise SupervisorError("external_dependency_roles_not_exact")
    for dependency in dependencies:
        target = Path(dependency["path"])
        if not target.is_file() or sha256_file(target) != dependency["sha256"]:
            raise SupervisorError(f"external_dependency_changed:{target}")
    return record


def dependency_map(record: dict[str, Any]) -> dict[str, Path]:
    return {item["role"]: Path(item["path"]).resolve()
            for item in record["external_dependencies"]}


def build_readonly_snapshot(
    record: dict[str, Any], workspace: Path, freeze_path: Path, snapshot: Path,
) -> Path:
    snapshot.mkdir(mode=0o755)
    for artifact in record["artifacts"]:
        source = workspace / artifact["path"]
        target = snapshot / artifact["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if sha256_file(target) != artifact["sha256"]:
            raise SupervisorError(f"snapshot_hash_mismatch:{artifact['path']}")
    frozen_record = snapshot / "p1_source_freeze_v0_5.json"
    shutil.copy2(freeze_path, frozen_record)
    for path in sorted(snapshot.rglob("*"), reverse=True):
        os.chmod(path, 0o555 if path.is_dir() else 0o444)
        os.chown(path, 0, 0)
    os.chmod(snapshot, 0o555)
    os.chown(snapshot, 0, 0)
    return frozen_record


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_tree(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    directories = [path for path in root.rglob("*") if path.is_dir()]
    for path in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        fsync_directory(path)
    fsync_directory(root)


def allocated_bytes(path: Path) -> int:
    stats = os.statvfs(path)
    return (stats.f_blocks - stats.f_bfree) * stats.f_frsize


def verify_physical_cores() -> list[int]:
    allowed = os.sched_getaffinity(0)
    cpus = list(range(16))
    if not set(cpus).issubset(allowed):
        raise SupervisorError(f"physical_cpu_set_not_allowed:{sorted(allowed)}")
    core_keys = []
    for cpu in cpus:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        core = (topology / "core_id").read_text(encoding="ascii").strip()
        package = (topology / "physical_package_id").read_text(encoding="ascii").strip()
        core_keys.append(f"{package}:{core}")
    if len(set(core_keys)) != 16:
        raise SupervisorError(f"cpu_set_not_16_physical_cores:{core_keys}")
    return cpus


def verify_private_run_identity(uid: int, gid: int) -> None:
    if uid != PRIVATE_RUN_UID or gid != PRIVATE_RUN_GID:
        raise SupervisorError(f"run_identity_not_frozen:{uid}:{gid}")
    try:
        account = pwd.getpwuid(uid)
        group = grp.getgrgid(gid)
    except KeyError as error:
        raise SupervisorError("private_run_account_missing") from error
    if (
        account.pw_name != PRIVATE_RUN_NAME
        or account.pw_gid != gid
        or account.pw_dir != "/nonexistent"
        or account.pw_shell != "/usr/sbin/nologin"
        or group.gr_name != PRIVATE_RUN_NAME
    ):
        raise SupervisorError(f"private_run_account_mismatch:{account}:{group}")
    for status_path in Path("/proc").glob("[0-9]*/status"):
        try:
            lines = status_path.read_text(encoding="ascii").splitlines()
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        uid_line = next((line for line in lines if line.startswith("Uid:")), "")
        if uid_line and uid in {int(value) for value in uid_line.split()[1:]}:
            raise SupervisorError(f"run_uid_already_active:{status_path.parent.name}")


def reject_unsafe_tree(root: Path) -> list[Path]:
    paths = [root, *root.rglob("*")]
    for path in paths:
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise SupervisorError(f"artifact_tree_symlink:{path}")
        if stat.S_ISREG(metadata.st_mode):
            if metadata.st_nlink != 1:
                raise SupervisorError(f"artifact_tree_file_not_unique:{path}")
        elif not stat.S_ISDIR(metadata.st_mode):
            raise SupervisorError(f"artifact_tree_special_file:{path}")
    return paths


def make_root_readonly(root: Path) -> None:
    paths = reject_unsafe_tree(root)
    for path in sorted(paths, key=lambda item: len(item.parts), reverse=True):
        metadata = path.lstat()
        os.chown(path, 0, 0, follow_symlinks=False)
        os.chmod(path, 0o555 if stat.S_ISDIR(metadata.st_mode) else 0o444)


def drain_cgroup(cgroup: Path) -> None:
    kill_file = cgroup / "cgroup.kill"
    for _ in range(100):
        pids = [int(value) for value in
                (cgroup / "cgroup.procs").read_text(encoding="ascii").split()]
        if not pids:
            return
        if kill_file.exists():
            kill_file.write_text("1", encoding="ascii")
        else:
            for pid in pids:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        time.sleep(0.01)
    raise SupervisorError("worker_cgroup_not_empty_after_kill")


def read_memory_events(cgroup: Path) -> dict[str, int]:
    return {
        line.split()[0]: int(line.split()[1])
        for line in (cgroup / "memory.events").read_text(encoding="ascii").splitlines()
    }


def sum_memory_events(*events: dict[str, int]) -> dict[str, int]:
    keys = set().union(*(event.keys() for event in events))
    return {key: sum(event.get(key, 0) for event in events) for key in keys}


def oom_events_increased(before: dict[str, int], after: dict[str, int]) -> bool:
    return any(after.get(key, 0) > before.get(key, 0) for key in OOM_EVENT_KEYS)


def self_virtual_memory_upper_bound() -> int:
    for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
        if line.startswith("VmSize:"):
            fields = line.split()
            if len(fields) != 3 or fields[2] != "kB":
                break
            return int(fields[1]) * 1024
    raise SupervisorError("self_vmsize_unavailable")


def artifact_tree_root(root: Path) -> str:
    reject_unsafe_tree(root)
    digest = hashlib.sha256()
    domain = b"chemgraph-v0.5/raster-pilot-artifact-tree"
    digest.update(len(domain).to_bytes(4, "big") + domain)
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("ascii")
        payload_hash = bytes.fromhex(sha256_file(path))
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(payload_hash)
    return digest.hexdigest()


def strict_zlib_decompress(payload: bytes, expected_size: int) -> bytes:
    decoder = zlib.decompressobj()
    decoded = decoder.decompress(payload, expected_size + 1)
    if len(decoded) != expected_size:
        raise SupervisorError(f"runner_frame_canonical_length:{len(decoded)}")
    if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise SupervisorError("runner_frame_zlib_not_exactly_one_stream")
    return decoded


def verify_result_vault(output: Path) -> Path:
    resolved_output = output.resolve()
    if resolved_output.parent != RESULT_ROOT:
        raise SupervisorError(f"output_not_in_result_vault:{resolved_output}")
    for path in (Path("/"), Path("/opt"), RESULT_ROOT.parent, RESULT_ROOT):
        metadata = path.lstat()
        if path.is_symlink() or metadata.st_uid != 0 or metadata.st_mode & 0o022:
            raise SupervisorError(f"result_vault_ancestor_not_frozen:{path}")
    if RESULT_ROOT.resolve() != RESULT_ROOT:
        raise SupervisorError(f"result_vault_not_canonical:{RESULT_ROOT}")
    return resolved_output


def verify_runner_artifacts(root: Path) -> dict[str, Any]:
    validation = json.loads((root / "validation.json").read_text(encoding="ascii"))
    if validation.get("status") != "completed_pending_resource_gate":
        raise SupervisorError("runner_status_not_pending_gate")
    records_path = root / "records.jsonl"
    frames_path = root / "canonical_frames.zlib"
    offsets_path = root / "canonical_frame_offsets.u64be"
    keys_path = root / "class_keys.sorted.tsv"
    if sha256_file(records_path) != validation["records_sha256"]:
        raise SupervisorError("runner_records_sha_mismatch")
    records: dict[int, str] = {}
    framed_root = hashlib.sha256()
    with records_path.open("rb") as handle:
        for index, payload in enumerate(handle):
            expected_tuple_id = index * 100
            framed_root.update(len(RECORD_DOMAIN).to_bytes(4, "big"))
            framed_root.update(RECORD_DOMAIN)
            framed_root.update(len(payload).to_bytes(8, "big"))
            framed_root.update(payload)
            line = payload.decode("ascii")
            record = json.loads(line)
            if record["tuple_id"] != expected_tuple_id or expected_tuple_id in records:
                raise SupervisorError(
                    f"runner_record_sequence_mismatch:{index}:{record['tuple_id']}"
                )
            records[record["tuple_id"]] = record["canonical_gray_sha256"]
    if len(records) != PILOT_TUPLES:
        raise SupervisorError("runner_record_count_mismatch")
    if framed_root.hexdigest() != validation["records_framed_root"]:
        raise SupervisorError("runner_records_framed_root_mismatch")
    expected_keys = sorted((digest, tuple_id) for tuple_id, digest in records.items())
    with keys_path.open("rt", encoding="ascii") as handle:
        actual_keys = [(line[:64], int(line[65:].strip())) for line in handle]
    if actual_keys != expected_keys:
        raise SupervisorError("runner_sorted_keys_mismatch")
    offsets = offsets_path.read_bytes()
    if len(offsets) != PILOT_TUPLES * 8:
        raise SupervisorError("runner_offsets_length_mismatch")
    with frames_path.open("rb") as frames:
        expected_offset = 0
        for index in range(PILOT_TUPLES):
            offset = struct.unpack_from(">Q", offsets, index * 8)[0]
            if offset != expected_offset:
                raise SupervisorError(f"runner_frame_offset_gap:{index}:{offset}:{expected_offset}")
            frames.seek(expected_offset)
            header = frames.read(12)
            if len(header) != 12:
                raise SupervisorError("runner_frame_header_truncated")
            tuple_id, length = struct.unpack(">QI", header)
            payload = frames.read(length)
            if len(payload) != length:
                raise SupervisorError(f"runner_frame_payload_truncated:{tuple_id}")
            canonical = strict_zlib_decompress(payload, 589_824)
            if tuple_id != index * 100:
                raise SupervisorError(f"runner_frame_tuple_sequence:{index}:{tuple_id}")
            if records.get(tuple_id) != hashlib.sha256(canonical).hexdigest():
                raise SupervisorError(f"runner_frame_bytes_mismatch:{tuple_id}")
            expected_offset = frames.tell()
        if expected_offset != frames_path.stat().st_size:
            raise SupervisorError("runner_frame_trailing_bytes")
    if validation.get("tuple_count") != PILOT_TUPLES:
        raise SupervisorError("runner_tuple_count_mismatch")
    if validation.get("padded_clipping_certificates") != PILOT_TUPLES:
        raise SupervisorError("runner_clipping_certificate_count_mismatch")
    if validation.get("deterministic_replay_count") != PILOT_TUPLES:
        raise SupervisorError("runner_replay_count_mismatch")
    return validation


def evaluate_gates(
    *, wall_upper: float, memory_peak_before: int, memory_peak_upper: int,
    memory_oom: int, temporary_upper: int,
    runner_validation: dict[str, Any], exit_code: int,
) -> tuple[dict[str, bool], int, int]:
    wall_upper_integer = math.ceil(wall_upper)
    projected_wall = (
        wall_upper_integer * TOTAL_TUPLES * 2 + PILOT_TUPLES - 1
    ) // PILOT_TUPLES
    projected_temporary = (
        temporary_upper * TOTAL_TUPLES * 2 + PILOT_TUPLES - 1
    ) // PILOT_TUPLES
    gates = {
        "runner_exit_zero": exit_code == 0,
        "pilot_wall_within_4h": wall_upper_integer <= PILOT_WALL_LIMIT,
        "memory_peak_before_receipt_within_80gib": memory_peak_before <= MEMORY_LIMIT,
        "memory_hard_upper_bound_within_80gib": memory_peak_upper <= MEMORY_LIMIT,
        "memory_oom_zero": memory_oom == 0,
        "temporary_peak_upper_within_2tib": temporary_upper <= TEMPORARY_LIMIT,
        "projected_temporary_within_2tib": projected_temporary <= TEMPORARY_LIMIT,
        "projected_full_wall_within_96h": projected_wall <= FULL_WALL_LIMIT,
        "tuple_count_exact": runner_validation.get("tuple_count") == PILOT_TUPLES,
        "padded_certificates_exact": (
            runner_validation.get("padded_clipping_certificates") == PILOT_TUPLES
        ),
        "deterministic_replay_exact": (
            runner_validation.get("deterministic_replay_count") == PILOT_TUPLES
        ),
    }
    return gates, projected_wall, projected_temporary


def run(args: argparse.Namespace) -> dict[str, Any]:
    if os.geteuid() != 0:
        raise SupervisorError("supervisor_requires_root")
    workspace = args.workspace.resolve()
    freeze_record = load_freeze(
        args.source_freeze, args.expected_source_freeze_sha256, workspace,
    )
    dependencies = dependency_map(freeze_record)
    requested = {
        "apptainer": Path(args.apptainer_binary).resolve(),
        "runtime_sif": Path(args.runtime_sif).resolve(),
        "mount": Path(args.mount_binary).resolve(),
        "umount": Path(args.umount_binary).resolve(),
    }
    for role, path in requested.items():
        if path != dependencies[role]:
            raise SupervisorError(f"cli_dependency_not_frozen:{role}:{path}")
    output = verify_result_vault(args.output)
    if output.exists():
        raise SupervisorError(f"output_already_exists:{output}")
    cpus = verify_physical_cores()
    uid, gid = args.run_uid, args.run_gid
    verify_private_run_identity(uid, gid)
    token = f"chemgraph-v05-pilot-{os.getpid()}"
    mountpoint = Path("/mnt") / token
    cgroup_parent = Path("/sys/fs/cgroup")
    worker_cgroup = cgroup_parent / f"{token}-worker"
    supervisor_cgroup = cgroup_parent / f"{token}-supervisor"
    final_stage = output.with_name(f".{output.name}.supervisor-stage-{os.getpid()}")
    if final_stage.exists():
        shutil.rmtree(final_stage)
    child = None
    mounted = False
    supervisor_in_cgroup = False
    started = time.perf_counter()
    peak_allocated = 0
    try:
        mountpoint.mkdir(mode=0o700)
        worker_cgroup.mkdir()
        supervisor_cgroup.mkdir()
        (worker_cgroup / "memory.max").write_text(str(WORKER_MEMORY_MAX), encoding="ascii")
        (supervisor_cgroup / "memory.max").write_text(
            str(SUPERVISOR_MEMORY_MAX), encoding="ascii",
        )
        pre_migration_memory_upper = self_virtual_memory_upper_bound()
        if pre_migration_memory_upper > PRE_MIGRATION_MEMORY_RESERVE:
            raise SupervisorError(
                f"pre_migration_memory_reserve_exceeded:{pre_migration_memory_upper}"
            )
        (supervisor_cgroup / "cgroup.procs").write_text(str(os.getpid()), encoding="ascii")
        supervisor_in_cgroup = True
        subprocess.run(
            [str(dependencies["mount"]), "-t", "tmpfs", "-o",
             f"size={TMPFS_SIZE},mode=0711", token, str(mountpoint)], check=True,
        )
        mounted = True
        initial_allocated = allocated_bytes(mountpoint)
        snapshot = mountpoint / "source"
        frozen_record = build_readonly_snapshot(
            freeze_record, workspace, args.source_freeze, snapshot,
        )
        runtime_snapshot = mountpoint / "runtime.sif"
        shutil.copy2(dependencies["runtime_sif"], runtime_snapshot)
        if sha256_file(runtime_snapshot) != next(
            item["sha256"] for item in freeze_record["external_dependencies"]
            if item["role"] == "runtime_sif"
        ):
            raise SupervisorError("runtime_snapshot_hash_mismatch")
        os.chown(runtime_snapshot, 0, 0)
        os.chmod(runtime_snapshot, 0o444)
        work = mountpoint / "work"
        work.mkdir(mode=0o700)
        os.chown(work, uid, gid)
        work_output = work / "pilot_output"
        stdout_path = work / "runner.stdout"
        stderr_path = work / "runner.stderr"

        def child_setup() -> None:
            os.sched_setaffinity(0, set(cpus))
            (worker_cgroup / "cgroup.procs").write_text(str(os.getpid()), encoding="ascii")
            os.setgroups([])
            os.setgid(gid)
            os.setuid(uid)

        command = [
            str(dependencies["apptainer"]), "exec", "--containall", "--cleanenv",
            "--no-home", "--bind", f"{snapshot}:/src:ro", "--bind", f"{work}:/work:rw",
            "--pwd", "/src", str(runtime_snapshot), "python",
            "/src/formal/chemgraph/challenge_v0_5/run_raster_pilot_v0_5.py",
            "--contract", "/src/formal/chemgraph/challenge_v0_5/prereg/p0_contract_v0_5.json",
            "--expected-contract-sha256", CONTRACT_SHA256,
            "--source-freeze", "/src/p1_source_freeze_v0_5.json",
            "--expected-source-freeze-sha256", args.expected_source_freeze_sha256,
            "--workspace", "/src",
            "--output", "/work/pilot_output",
            "--workers", "16",
        ]
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            child = subprocess.Popen(command, cwd=snapshot, stdout=stdout, stderr=stderr,
                                     preexec_fn=child_setup,
                                     env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"})
            while child.poll() is None:
                peak_allocated = max(
                    peak_allocated, allocated_bytes(mountpoint) - initial_allocated,
                )
                time.sleep(0.05)
        peak_allocated = max(peak_allocated, allocated_bytes(mountpoint) - initial_allocated)
        exit_code = child.returncode
        drain_cgroup(worker_cgroup)
        reject_unsafe_tree(work)
        if exit_code != 0 or not stat.S_ISDIR(work_output.lstat().st_mode):
            raise SupervisorError(f"runner_failed:{exit_code}")
        make_root_readonly(work)
        runner_validation = verify_runner_artifacts(work_output)
        source_runner_root = artifact_tree_root(work_output)
        memory_events = sum_memory_events(
            read_memory_events(worker_cgroup), read_memory_events(supervisor_cgroup),
        )
        memory_peak = (
            pre_migration_memory_upper
            + int((worker_cgroup / "memory.peak").read_text(encoding="ascii"))
            + int((supervisor_cgroup / "memory.peak").read_text(encoding="ascii"))
        )

        shutil.copytree(work_output, final_stage)
        copied_validation = verify_runner_artifacts(final_stage)
        if copied_validation != runner_validation:
            raise SupervisorError("copied_runner_validation_changed")
        copied_runner_root = artifact_tree_root(final_stage)
        if copied_runner_root != source_runner_root:
            raise SupervisorError("copied_runner_artifact_root_changed")
        shutil.copy2(stdout_path, final_stage / "runner.stdout")
        shutil.copy2(stderr_path, final_stage / "runner.stderr")
        wall_upper = math.ceil(time.perf_counter() - started) + PUBLICATION_RESERVE_SECONDS
        gates, projected_wall, projected_temporary = evaluate_gates(
            wall_upper=wall_upper,
            memory_peak_before=memory_peak,
            memory_peak_upper=pre_migration_memory_upper + CGROUP_MEMORY_MAX,
            memory_oom=memory_events.get("oom", 0),
            temporary_upper=TMPFS_SIZE,
            runner_validation=runner_validation,
            exit_code=exit_code,
        )
        passed = all(gates.values())
        result = {
            "schema_version": "chemgraph-raster-pilot-resource-v0.5",
            "status": "passed" if passed else "no_go",
            "contract_sha256": CONTRACT_SHA256,
            "p1_source_commit": freeze_record["p1_source_commit"],
            "p1_source_freeze_sha256": args.expected_source_freeze_sha256,
            "runner_artifact_tree_root": source_runner_root,
            "physical_cpus": cpus,
            "artifact_publication_wall_upper_bound_seconds": wall_upper,
            "projected_full_wall_upper_bound_seconds": projected_wall,
            "memory_peak_before_receipt_bytes": memory_peak,
            "memory_cgroup_max_bytes": CGROUP_MEMORY_MAX,
            "memory_peak_recorded_upper_bound_bytes": MEMORY_LIMIT,
            "memory_events_before_receipt": memory_events,
            "temporary_peak_observed_allocated_bytes": peak_allocated,
            "temporary_peak_hard_upper_bound_bytes": TMPFS_SIZE,
            "projected_temporary_upper_bound_bytes": projected_temporary,
            "temporary_limit_bytes": TEMPORARY_LIMIT,
            "gates": gates,
            "training_authorized": False,
        }
        resource_path = final_stage / "resource_validation.json"
        with resource_path.open("xb") as handle:
            handle.write(canonical_json_bytes(result))
            handle.flush()
            os.fsync(handle.fileno())
        make_root_readonly(final_stage)
        fsync_tree(final_stage)
        if time.perf_counter() - started > wall_upper:
            raise SupervisorError("publication_exceeded_recorded_wall_upper_bound")
        os.replace(final_stage, output)
        fsync_directory(output.parent)
        final_memory_peak = (
            pre_migration_memory_upper
            + int((worker_cgroup / "memory.peak").read_text(encoding="ascii"))
            + int((supervisor_cgroup / "memory.peak").read_text(encoding="ascii"))
        )
        if final_memory_peak > MEMORY_LIMIT:
            os.replace(output, final_stage)
            fsync_directory(output.parent)
            raise SupervisorError("final_memory_peak_exceeded_80gib")
        if time.perf_counter() - started > wall_upper:
            os.replace(output, final_stage)
            fsync_directory(output.parent)
            raise SupervisorError("atomic_publish_exceeded_recorded_wall_upper_bound")
        final_memory_events = sum_memory_events(
            read_memory_events(worker_cgroup), read_memory_events(supervisor_cgroup),
        )
        if oom_events_increased(memory_events, final_memory_events):
            os.replace(output, final_stage)
            fsync_directory(output.parent)
            raise SupervisorError("memory_oom_event_after_receipt")
        return result
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=30)
        if worker_cgroup.exists():
            try:
                drain_cgroup(worker_cgroup)
            except (OSError, SupervisorError):
                pass
        if mounted:
            subprocess.run([str(dependencies["umount"]), str(mountpoint)], check=False)
        if supervisor_in_cgroup:
            (cgroup_parent / "cgroup.procs").write_text(str(os.getpid()), encoding="ascii")
            supervisor_in_cgroup = False
        if mountpoint.exists():
            mountpoint.rmdir()
        for cgroup in (worker_cgroup, supervisor_cgroup):
            for _ in range(20):
                try:
                    cgroup.rmdir()
                    break
                except OSError:
                    time.sleep(0.05)
        if final_stage.exists():
            shutil.rmtree(final_stage)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--source-freeze", type=Path, required=True)
    parser.add_argument("--expected-source-freeze-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-uid", type=int, required=True)
    parser.add_argument("--run-gid", type=int, required=True)
    parser.add_argument("--apptainer-binary", required=True)
    parser.add_argument("--runtime-sif", required=True)
    parser.add_argument("--mount-binary", required=True)
    parser.add_argument("--umount-binary", required=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
