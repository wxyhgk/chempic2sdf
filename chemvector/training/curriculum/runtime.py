"""Bounded single-process execution, independent of torch and native kernels.

Used by the diagnostic CLIs and the explicit-release training launcher. The
guard is not a data-release authority and never grants training admission.
"""

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

from .admission import require_training_admission


@dataclass(frozen=True)
class ResourceBudget:
    wall_seconds: float
    gpu_devices: tuple[str, ...] = ()

    def __post_init__(self):
        if (isinstance(self.wall_seconds, bool) or not math.isfinite(self.wall_seconds)
                or not 0 < self.wall_seconds <= 12 * 3600):
            raise ValueError("Runtime budget must be finite, positive, and at most 12 hours")
        if len(self.gpu_devices) > 1 or any(not str(d).isdigit() for d in self.gpu_devices):
            raise ValueError("This recipe permits zero or one explicitly selected GPU")


class StepLimit:
    """Consume a step BEFORE optimizer.step; empty batches still cost wall time."""

    def __init__(self, maximum):
        if type(maximum) is not int or maximum < 1:
            raise ValueError("Maximum optimizer steps must be a positive integer")
        self.maximum, self.consumed = maximum, 0

    def consume(self):
        if self.consumed >= self.maximum:
            raise RuntimeError("Optimizer step budget exhausted")
        self.consumed += 1


def run_budgeted(command, *, output, budget, training=False, admission=None):
    if not isinstance(budget, ResourceBudget):
        raise ValueError("Explicit resource budget is required")
    if training or budget.gpu_devices:
        require_training_admission(admission or {})
    output = Path(output)
    if output.exists():
        raise FileExistsError("Budgeted execution requires a fresh output directory")
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, budget.gpu_devices)),
               OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
    started = time.monotonic()
    receipt = {"schema_version": "chemvector-curriculum-runtime-0.1", "command": list(command),
               "wall_budget_seconds": budget.wall_seconds, "gpu_devices": list(budget.gpu_devices),
               "training": training, "automatic_retries": 0, "status": "launch_failed"}
    process = None
    try:
        with (output / "worker.log").open("xb") as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=max(.001, budget.wall_seconds - (time.monotonic() - started)))
                receipt.update(status="completed" if code == 0 else "worker_failed", exit_code=code)
            except subprocess.TimeoutExpired:
                receipt.update(status="timeout")
    finally:
        if process is not None:
            # Only the process group created above. Clean up descendants even
            # when the leader has exited. No wildcard or unrelated process kill.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=2)
        receipt["elapsed_seconds"] = time.monotonic() - started
        receipt["maximum_gpu_hours"] = receipt["elapsed_seconds"] * len(budget.gpu_devices) / 3600
        with (output / "runtime.json").open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write("\n")
    return receipt
