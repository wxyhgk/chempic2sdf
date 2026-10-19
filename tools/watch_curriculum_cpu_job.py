"""Recover a deadline for one existing, verified CPU renderer, without restarting.

The original launcher exited before writing its receipt. This monitor cannot
recover the lost child's exit status and explicitly distinguishes a complete
artifact set from an observed successful process exit. Run as a transient user
service so it is not owned by an interactive terminal session.
"""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import signal
import time


def write(path, payload):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


def process_identity(pid):
    root = Path("/proc") / str(pid)
    try:
        stat = (root / "stat").read_text().rsplit(")", 1)[1].split()
        command = (root / "cmdline").read_bytes().split(b"\0")
    except FileNotFoundError:
        return None
    if stat[0] == "Z":
        return None
    return {"start_ticks": int(stat[19]), "command": [s.decode() for s in command if s]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--original-seconds", type=int, required=True)
    args = parser.parse_args()
    original = process_identity(args.pid)
    if original is None or os.getpgid(args.pid) != args.pid:
        raise ValueError("Expected existing renderer session leader")
    command = original["command"]
    if (len(command) < 7 or command[1:4] != ["-m", "Data.sdf.chemvector_curriculum_v0_2.build", "--worker"]
            or "--output" not in command or command[command.index("--output") + 1] != str(args.dataset)):
        raise ValueError("Refuse to supervise unrelated process")
    if b"CUDA_VISIBLE_DEVICES=\0" not in (Path("/proc") / str(args.pid) / "environ").read_bytes():
        raise ValueError("Renderer is not CUDA-masked")
    if not 1 <= args.original_seconds <= 1800 or not args.runtime_dir.is_dir():
        raise ValueError("Invalid original bounded scope")
    # /proc start time and CLOCK_BOOTTIME share the boot origin. Subtract two
    # seconds conservatively for the original supervisor's process startup.
    start = original["start_ticks"] / os.sysconf("SC_CLK_TCK")
    deadline = start + args.original_seconds - 2
    clock = lambda: time.clock_gettime(time.CLOCK_BOOTTIME)
    record = {
        "schema_version": "chemvector-existing-cpu-supervision-recovery-0.1",
        "pid": args.pid, **original, "original_wall_budget_seconds": args.original_seconds,
        "conservative_deadline_boottime": deadline,
        "remaining_seconds_at_recovery": max(0, deadline - clock()),
        "original_launcher_exit_code": 143, "original_launcher_runtime_receipt_missing": True,
        "worker_exit_code": None, "automatic_retries": 0, "worker_restarted": False,
        "gpu_devices": [], "training": False,
        "plan_sha256": sha256((args.dataset / "plan.json").read_bytes()).hexdigest(),
        "monitor_source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    write(args.runtime_dir / "supervision-recovery.json", record)
    while process_identity(args.pid) == original and clock() < deadline:
        time.sleep(min(1, max(0, deadline - clock())))
    current = process_identity(args.pid)
    if current == original:
        if os.getpgid(args.pid) != args.pid:
            raise ValueError("Renderer process group changed")
        os.killpg(args.pid, signal.SIGKILL)
        status = "timeout"
    elif current is not None:
        raise ValueError("PID reused; refuse to signal replacement")
    else:
        status = "worker_exited_exit_status_unavailable"
    summary_path = args.dataset / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else None
    record.update(status=status, elapsed_since_worker_start_seconds=clock() - start,
                  complete_summary_present=bool(summary and summary.get("generation_complete")),
                  summary_sha256=sha256(summary_path.read_bytes()).hexdigest() if summary else None,
                  requires_independent_artifact_verification=True)
    write(args.runtime_dir / "runtime-recovery.json", record)


if __name__ == "__main__":
    main()
