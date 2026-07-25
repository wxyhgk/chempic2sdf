"""Validated campaign configuration schema."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import yaml


ALLOWED_OPERATORS = {"eq", "ne", "gt", "ge", "lt", "le"}


def _require_mapping(value: Any, name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


@dataclass(frozen=True, slots=True)
class GateSpec:
    metric: str
    operator: str
    value: int | float | str | bool

    @classmethod
    def from_dict(cls, payload: dict, *, name: str) -> "GateSpec":
        payload = _require_mapping(payload, name)
        metric = _require_string(payload.get("metric"), f"{name}.metric")
        operator = _require_string(payload.get("operator"), f"{name}.operator")
        if operator not in ALLOWED_OPERATORS:
            raise ValueError(f"{name}.operator must be one of {sorted(ALLOWED_OPERATORS)}")
        value = payload.get("value")
        if not isinstance(value, (int, float, str, bool)):
            raise ValueError(f"{name}.value must be a scalar")
        return cls(metric=metric, operator=operator, value=value)


@dataclass(frozen=True, slots=True)
class StageSpec:
    stage_id: str
    command: tuple[str, ...]
    resume_command: tuple[str, ...] | None
    requires_gpu: bool
    result_file: str | None
    gates: tuple[GateSpec, ...]
    timeout_seconds: int | None
    environment: dict[str, str]

    @classmethod
    def from_dict(cls, payload: dict, *, name: str) -> "StageSpec":
        payload = _require_mapping(payload, name)
        stage_id = _require_string(payload.get("id"), f"{name}.id")
        command = payload.get("command")
        if not isinstance(command, list) or not command or not all(
            isinstance(item, str) and item for item in command
        ):
            raise ValueError(f"{name}.command must be a nonempty string list")
        resume = payload.get("resume_command")
        if resume is not None and (
            not isinstance(resume, list)
            or not resume
            or not all(isinstance(item, str) and item for item in resume)
        ):
            raise ValueError(f"{name}.resume_command must be a string list")
        timeout = payload.get("timeout_seconds")
        if timeout is not None and (not isinstance(timeout, int) or timeout < 1):
            raise ValueError(f"{name}.timeout_seconds must be positive")
        environment = payload.get("environment", {})
        if not isinstance(environment, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in environment.items()
        ):
            raise ValueError(f"{name}.environment must map strings to strings")
        gates_payload = payload.get("gates", [])
        if not isinstance(gates_payload, list):
            raise ValueError(f"{name}.gates must be a list")
        return cls(
            stage_id=stage_id,
            command=tuple(command),
            resume_command=(tuple(resume) if resume is not None else None),
            requires_gpu=bool(payload.get("requires_gpu", False)),
            result_file=(
                _require_string(payload["result_file"], f"{name}.result_file")
                if payload.get("result_file") is not None
                else None
            ),
            gates=tuple(
                GateSpec.from_dict(item, name=f"{name}.gates[{index}]")
                for index, item in enumerate(gates_payload)
            ),
            timeout_seconds=timeout,
            environment=dict(environment),
        )


@dataclass(frozen=True, slots=True)
class TrialSpec:
    trial_id: str
    hypothesis: str
    stages: tuple[StageSpec, ...]

    @classmethod
    def from_dict(cls, payload: dict, *, name: str) -> "TrialSpec":
        payload = _require_mapping(payload, name)
        trial_id = _require_string(payload.get("id"), f"{name}.id")
        hypothesis = _require_string(payload.get("hypothesis"), f"{name}.hypothesis")
        stages_payload = payload.get("stages")
        if not isinstance(stages_payload, list) or not stages_payload:
            raise ValueError(f"{name}.stages must be a nonempty list")
        stages = tuple(
            StageSpec.from_dict(item, name=f"{name}.stages[{index}]")
            for index, item in enumerate(stages_payload)
        )
        stage_ids = [stage.stage_id for stage in stages]
        if len(stage_ids) != len(set(stage_ids)):
            raise ValueError(f"{name} contains duplicate stage ids")
        return cls(trial_id=trial_id, hypothesis=hypothesis, stages=stages)


@dataclass(frozen=True, slots=True)
class CampaignSpec:
    schema_version: str
    campaign_id: str
    objective: str
    status: str
    stable_baseline: str
    max_trials: int
    max_gpu_hours: float
    source_paths: tuple[str, ...]
    input_artifacts: tuple[str, ...]
    trials: tuple[TrialSpec, ...]

    @classmethod
    def from_dict(cls, payload: dict) -> "CampaignSpec":
        payload = _require_mapping(payload, "campaign")
        schema_version = _require_string(
            payload.get("schema_version"), "campaign.schema_version"
        )
        if schema_version != "chemvector-campaign-0.1":
            raise ValueError("Unsupported campaign schema version")
        status = _require_string(payload.get("status", "draft"), "campaign.status")
        if status not in {"draft", "active"}:
            raise ValueError("campaign.status must be draft or active")
        max_trials = payload.get("max_trials", 3)
        max_gpu_hours = payload.get("max_gpu_hours", 24.0)
        if not isinstance(max_trials, int) or max_trials < 1:
            raise ValueError("campaign.max_trials must be positive")
        if not isinstance(max_gpu_hours, (int, float)) or max_gpu_hours <= 0:
            raise ValueError("campaign.max_gpu_hours must be positive")
        source_paths = payload.get("source_paths", ["chemvector", "tests"])
        input_artifacts = payload.get("input_artifacts", [])
        if not isinstance(source_paths, list) or not all(
            isinstance(item, str) and item for item in source_paths
        ):
            raise ValueError("campaign.source_paths must be a string list")
        if not isinstance(input_artifacts, list) or not all(
            isinstance(item, str) and item for item in input_artifacts
        ):
            raise ValueError("campaign.input_artifacts must be a string list")
        trials_payload = payload.get("trials")
        if not isinstance(trials_payload, list) or not trials_payload:
            raise ValueError("campaign.trials must be a nonempty list")
        trials = tuple(
            TrialSpec.from_dict(item, name=f"campaign.trials[{index}]")
            for index, item in enumerate(trials_payload)
        )
        trial_ids = [trial.trial_id for trial in trials]
        if len(trial_ids) != len(set(trial_ids)):
            raise ValueError("campaign contains duplicate trial ids")
        if len(trials) > max_trials:
            raise ValueError("campaign defines more trials than max_trials")
        return cls(
            schema_version=schema_version,
            campaign_id=_require_string(payload.get("campaign_id"), "campaign.campaign_id"),
            objective=_require_string(payload.get("objective"), "campaign.objective"),
            status=status,
            stable_baseline=_require_string(
                payload.get("stable_baseline"), "campaign.stable_baseline"
            ),
            max_trials=max_trials,
            max_gpu_hours=float(max_gpu_hours),
            source_paths=tuple(source_paths),
            input_artifacts=tuple(input_artifacts),
            trials=trials,
        )


def load_campaign(path: str | Path) -> CampaignSpec:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    payload = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
    return CampaignSpec.from_dict(payload)
