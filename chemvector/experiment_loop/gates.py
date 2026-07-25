"""Deterministic metric gates for campaign stages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from chemvector.experiment_loop.schema import GateSpec


def metric_value(payload: dict, path: str) -> Any:
    current: Any = payload
    if path.startswith("/"):
        parts = [
            part.replace("~1", "/").replace("~0", "~")
            for part in path[1:].split("/")
        ]
    else:
        parts = path.split(".")
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            raise KeyError(path)
        current = current[part]
    return current


def compare(actual: Any, operator: str, expected: Any) -> bool:
    operations = {
        "eq": lambda: actual == expected,
        "ne": lambda: actual != expected,
        "gt": lambda: actual > expected,
        "ge": lambda: actual >= expected,
        "lt": lambda: actual < expected,
        "le": lambda: actual <= expected,
    }
    return bool(operations[operator]())


def evaluate_gates(payload: dict, gates: tuple[GateSpec, ...]) -> list[dict]:
    results = []
    for gate in gates:
        try:
            actual = metric_value(payload, gate.metric)
            passed = compare(actual, gate.operator, gate.value)
            error = None
        except (KeyError, TypeError, ValueError) as exc:
            actual = None
            passed = False
            error = str(exc)
        results.append(
            {
                "metric": gate.metric,
                "operator": gate.operator,
                "expected": gate.value,
                "actual": actual,
                "passed": passed,
                "error": error,
            }
        )
    return results


def load_result(path: Path | None, return_code: int) -> dict:
    if path is None:
        return {"return_code": return_code}
    if not path.is_file():
        return {"return_code": return_code, "result_file_missing": True}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Stage result JSON must be an object")
    payload.setdefault("return_code", return_code)
    return payload
