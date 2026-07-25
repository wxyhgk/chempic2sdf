"""Atomic campaign state persistence."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from chemvector.experiment_loop.schema import CampaignSpec


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def initial_state(spec: CampaignSpec, config_sha256: str) -> dict:
    now = utc_now()
    return {
        "schema_version": "chemvector-campaign-state-0.1",
        "campaign_id": spec.campaign_id,
        "config_sha256": config_sha256,
        "status": "pending",
        "created_at": now,
        "updated_at": now,
        "gpu_seconds": 0.0,
        "active_trial": None,
        "active_stage": None,
        "trials": {
            trial.trial_id: {
                "status": "pending",
                "hypothesis": trial.hypothesis,
                "stages": {
                    stage.stage_id: {
                        "status": "pending",
                        "attempts": 0,
                    }
                    for stage in trial.stages
                },
            }
            for trial in spec.trials
        },
    }


def load_or_initialize_state(
    path: Path, spec: CampaignSpec, config_sha256: str
) -> dict:
    if not path.is_file():
        state = initial_state(spec, config_sha256)
        atomic_write_json(path, state)
        return state
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("campaign_id") != spec.campaign_id:
        raise ValueError("Campaign state belongs to another campaign")
    if state.get("config_sha256") != config_sha256:
        raise ValueError("Campaign configuration changed after state creation")
    return state


def save_state(path: Path, state: dict) -> None:
    state["updated_at"] = utc_now()
    atomic_write_json(path, state)


def append_event(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": utc_now(), **event}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()
