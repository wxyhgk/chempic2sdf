"""Recoverable finite campaign runner."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import time

from chemvector.experiment_loop.artifacts import (
    sha256_file,
    verify_manifest,
    write_manifest,
)
from chemvector.experiment_loop.gates import evaluate_gates, load_result
from chemvector.experiment_loop.gpu import GpuLock, append_telemetry
from chemvector.experiment_loop.report import write_report
from chemvector.experiment_loop.schema import CampaignSpec, StageSpec, load_campaign
from chemvector.experiment_loop.state import (
    append_event,
    atomic_write_json,
    load_or_initialize_state,
    save_state,
    utc_now,
)


class CampaignRunner:
    def __init__(
        self,
        campaign_path: str | Path,
        *,
        project_root: str | Path = ".",
        allow_draft: bool = False,
        gpu_lock_path: str | Path = "/tmp/chempic2sdf-gpu0.lock",
        poll_seconds: float = 1.0,
        telemetry_seconds: float = 30.0,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.campaign_path = Path(campaign_path).resolve()
        self.spec: CampaignSpec = load_campaign(self.campaign_path)
        if self.spec.status == "draft" and not allow_draft:
            raise ValueError("Campaign is draft; pass --allow-draft to execute it")
        self.config_sha256 = sha256_file(self.campaign_path)
        self.run_dir = self.project_root / "outputs" / "experiments" / self.spec.campaign_id
        self.state_path = self.run_dir / "state.json"
        self.events_path = self.run_dir / "events.jsonl"
        self.manifest_path = self.run_dir / "manifest.json"
        self.report_path = self.run_dir / "report.md"
        self.gpu_lock_path = Path(gpu_lock_path)
        self.poll_seconds = poll_seconds
        self.telemetry_seconds = telemetry_seconds
        self.state = load_or_initialize_state(
            self.state_path, self.spec, self.config_sha256
        )
        if not self.manifest_path.is_file():
            write_manifest(
                self.manifest_path,
                project_root=self.project_root,
                campaign_path=self.campaign_path,
                source_paths=self.spec.source_paths,
                input_artifacts=self.spec.input_artifacts,
            )
        else:
            verify_manifest(self.manifest_path, project_root=self.project_root)
        write_report(self.report_path, self.spec, self.state)

    def _format(self, value: str, trial_id: str, stage_id: str) -> str:
        values = {
            "project_root": str(self.project_root),
            "campaign_dir": str(self.run_dir),
            "trial_dir": str(self.run_dir / "trials" / trial_id),
            "stage_dir": str(self.run_dir / "trials" / trial_id / stage_id),
            "trial_id": trial_id,
            "stage_id": stage_id,
        }
        try:
            return value.format(**values)
        except KeyError as exc:
            raise ValueError(f"Unknown command placeholder: {exc}") from exc

    def _resolve_result_path(
        self, value: str | None, trial_id: str, stage_id: str
    ) -> Path | None:
        if value is None:
            return None
        path = Path(self._format(value, trial_id, stage_id))
        return path if path.is_absolute() else self.project_root / path

    def _run_process(
        self,
        command: list[str],
        *,
        stage_dir: Path,
        environment: dict[str, str],
        requires_gpu: bool,
        timeout_seconds: int | None,
    ) -> tuple[int, float]:
        stage_dir.mkdir(parents=True, exist_ok=True)
        log_path = stage_dir / "stage.log"
        heartbeat_path = stage_dir / "heartbeat.json"
        telemetry_path = stage_dir / "gpu_telemetry.jsonl"
        env = os.environ.copy()
        env.update(environment)
        if requires_gpu:
            env.setdefault("CUDA_VISIBLE_DEVICES", "0")

        def execute() -> tuple[int, float]:
            started = time.monotonic()
            last_telemetry = float("-inf")
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n[{utc_now()}] command={json.dumps(command)}\n")
                log.flush()
                process = subprocess.Popen(
                    command,
                    cwd=self.project_root,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                while process.poll() is None:
                    elapsed = time.monotonic() - started
                    atomic_write_json(
                        heartbeat_path,
                        {"pid": process.pid, "elapsed_seconds": elapsed, "updated_at": utc_now()},
                    )
                    if requires_gpu and elapsed - last_telemetry >= self.telemetry_seconds:
                        append_telemetry(telemetry_path)
                        last_telemetry = elapsed
                    if timeout_seconds is not None and elapsed >= timeout_seconds:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        return 124, time.monotonic() - started
                    time.sleep(self.poll_seconds)
                return int(process.returncode), time.monotonic() - started

        if requires_gpu:
            with GpuLock(self.gpu_lock_path):
                return execute()
        return execute()

    def _run_stage(self, trial_id: str, stage: StageSpec) -> bool:
        trial_state = self.state["trials"][trial_id]
        stage_state = trial_state["stages"][stage.stage_id]
        if stage_state["status"] == "passed":
            return True
        interrupted = stage_state["status"] == "running"
        raw_command = (
            stage.resume_command
            if interrupted and stage.resume_command is not None
            else stage.command
        )
        command = [self._format(value, trial_id, stage.stage_id) for value in raw_command]
        environment = {
            key: self._format(value, trial_id, stage.stage_id)
            for key, value in stage.environment.items()
        }
        stage_dir = self.run_dir / "trials" / trial_id / stage.stage_id
        stage_state.update(
            {
                "status": "running",
                "attempts": int(stage_state.get("attempts", 0)) + 1,
                "command": command,
                "started_at": utc_now(),
            }
        )
        self.state.update(
            {"status": "running", "active_trial": trial_id, "active_stage": stage.stage_id}
        )
        save_state(self.state_path, self.state)
        append_event(
            self.events_path,
            {"event": "stage_started", "trial": trial_id, "stage": stage.stage_id},
        )
        return_code, elapsed = self._run_process(
            command,
            stage_dir=stage_dir,
            environment=environment,
            requires_gpu=stage.requires_gpu,
            timeout_seconds=stage.timeout_seconds,
        )
        if stage.requires_gpu:
            self.state["gpu_seconds"] = float(self.state.get("gpu_seconds", 0.0)) + elapsed
        result_path = self._resolve_result_path(
            stage.result_file, trial_id, stage.stage_id
        )
        result = load_result(result_path, return_code)
        gate_results = evaluate_gates(result, stage.gates)
        passed = return_code == 0 and all(item["passed"] for item in gate_results)
        atomic_write_json(
            stage_dir / "result.json",
            {
                "return_code": return_code,
                "elapsed_seconds": elapsed,
                "source_result_file": str(result_path) if result_path else None,
                "metrics": result,
                "gates": gate_results,
                "passed": passed,
            },
        )
        stage_state.update(
            {
                "status": "passed" if passed else "failed",
                "return_code": return_code,
                "elapsed_seconds": elapsed,
                "gate_results": gate_results,
                "finished_at": utc_now(),
            }
        )
        append_event(
            self.events_path,
            {
                "event": "stage_passed" if passed else "stage_failed",
                "trial": trial_id,
                "stage": stage.stage_id,
                "return_code": return_code,
            },
        )
        save_state(self.state_path, self.state)
        write_report(self.report_path, self.spec, self.state)
        return passed

    def run(self) -> dict:
        if self.state["status"] == "promotion_candidate":
            return self.state
        for trial in self.spec.trials:
            trial_state = self.state["trials"][trial.trial_id]
            if trial_state["status"] in {"passed", "failed"}:
                continue
            if self.state.get("gpu_seconds", 0.0) / 3600.0 >= self.spec.max_gpu_hours:
                self.state["status"] = "blocked"
                self.state["blocked_reason"] = "GPU hour budget exhausted"
                break
            trial_state["status"] = "running"
            save_state(self.state_path, self.state)
            all_passed = True
            for stage in trial.stages:
                if not self._run_stage(trial.trial_id, stage):
                    all_passed = False
                    break
            trial_state["status"] = "passed" if all_passed else "failed"
            if all_passed:
                self.state["status"] = "promotion_candidate"
                self.state["promotion_trial"] = trial.trial_id
                break
        else:
            self.state["status"] = "no_go"
        self.state["active_trial"] = None
        self.state["active_stage"] = None
        save_state(self.state_path, self.state)
        atomic_write_json(
            self.run_dir / "decision.json",
            {
                "campaign_id": self.spec.campaign_id,
                "decision": self.state["status"],
                "promotion_trial": self.state.get("promotion_trial"),
                "stable_baseline_unchanged": self.spec.stable_baseline,
            },
        )
        atomic_write_json(
            self.run_dir / "next_task.json",
            {
                "campaign_id": self.spec.campaign_id,
                "action": (
                    "human_review"
                    if self.state["status"] == "promotion_candidate"
                    else "propose_new_hypothesis"
                ),
                "reason": self.state["status"],
                "stable_baseline_unchanged": self.spec.stable_baseline,
            },
        )
        write_report(self.report_path, self.spec, self.state)
        return self.state
