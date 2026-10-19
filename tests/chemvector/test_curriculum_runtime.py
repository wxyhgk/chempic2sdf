"""Tiny subprocesses only; never initialize CUDA or run a training command."""

import json
import os
from pathlib import Path
import sys
import time

import pytest

from chemvector.training.curriculum.runtime import ResourceBudget, StepLimit, run_budgeted


@pytest.mark.parametrize("seconds", [0, -1, True, float("nan"), float("inf"), 43201])
def test_bad_wall_budgets_fail(seconds):
    with pytest.raises(ValueError):
        ResourceBudget(seconds)


@pytest.mark.parametrize("devices", [("0", "1"), ("-1",), ("all",)])
def test_only_single_explicit_gpu_is_supported(devices):
    with pytest.raises(ValueError):
        ResourceBudget(1, devices)


def test_step_limit_fails_before_extra_step():
    limit = StepLimit(2)
    limit.consume()
    limit.consume()
    with pytest.raises(RuntimeError, match="exhausted"):
        limit.consume()
    assert limit.consumed == 2


@pytest.mark.parametrize("training,devices", [(True, ()), (False, ("0",)), (True, ("0",))])
def test_unadmitted_data_cannot_launch_worker_or_create_output(tmp_path, training, devices, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Worker must not start")
    monkeypatch.setattr("subprocess.Popen", forbidden)
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="must not start"):
        run_budgeted([sys.executable, "-c", "pass"], output=output,
                     budget=ResourceBudget(1, devices), training=training,
                     admission={"training_authorized": False})
    assert not output.exists()


def test_cpu_diagnostic_hides_cuda_preserves_parent_environment_and_never_overwrites(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    output = tmp_path / "run"
    command = [sys.executable, "-c", "import os; assert os.environ['CUDA_VISIBLE_DEVICES']==''; print('cpu only')"]
    result = run_budgeted(command, output=output, budget=ResourceBudget(5))
    assert result["status"] == "completed" and result["maximum_gpu_hours"] == 0
    assert json.loads((output / "runtime.json").read_text()) == result
    assert "cpu only" in (output / "worker.log").read_text()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    with pytest.raises(FileExistsError):
        run_budgeted(command, output=output, budget=ResourceBudget(5))


def test_hung_worker_is_killed_at_deadline_and_not_retried(tmp_path):
    output = tmp_path / "timeout"
    started = time.monotonic()
    result = run_budgeted([sys.executable, "-c", "import os,time; print(os.getpid(),flush=True); time.sleep(30)"],
                          output=output, budget=ResourceBudget(.25))
    assert time.monotonic() - started < 3
    assert result["status"] == "timeout" and result["automatic_retries"] == 0
    pid = int((output / "worker.log").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_worker_error_remains_error_not_success(tmp_path):
    result = run_budgeted([sys.executable, "-c", "raise RuntimeError('synthetic failure')"],
                          output=tmp_path / "error", budget=ResourceBudget(5))
    assert result["status"] == "worker_failed" and result["exit_code"] != 0
