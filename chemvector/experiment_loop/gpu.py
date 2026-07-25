"""Single-GPU locking and telemetry."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


class GpuLock:
    def __init__(self, path: str | Path, *, timeout_seconds: float = 3600.0) -> None:
        self.path = Path(path)
        self.timeout_seconds = timeout_seconds
        self.handle = None

    def __enter__(self) -> "GpuLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+")
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            try:
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    self.handle.close()
                    self.handle = None
                    raise TimeoutError(f"Timed out waiting for GPU lock: {self.path}")
                time.sleep(1.0)
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(f"pid={os.getpid()}\n")
        self.handle.flush()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None


def gpu_telemetry() -> dict:
    command = [
        "nvidia-smi",
        "--query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": str(exc)}
    if result.returncode:
        return {"available": False, "error": result.stderr.strip()}
    rows = []
    for line in result.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) == 7:
            rows.append(
                {
                    "timestamp": values[0],
                    "name": values[1],
                    "utilization_percent": float(values[2]),
                    "memory_used_mib": float(values[3]),
                    "memory_total_mib": float(values[4]),
                    "temperature_c": float(values[5]),
                    "power_w": float(values[6]),
                }
            )
    return {"available": True, "gpus": rows}


def append_telemetry(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(gpu_telemetry(), ensure_ascii=False) + "\n")
