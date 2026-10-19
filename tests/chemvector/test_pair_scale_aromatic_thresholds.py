from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "audit_pair_scale_aromatic_thresholds",
    ROOT / "tools/audit_pair_scale_aromatic_thresholds.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_threshold_audit_uses_saved_trace_without_forward(tmp_path: Path) -> None:
    out = tmp_path / "audit"
    result = MODULE.run(MODULE.TRACE, out)
    assert result["trace_scope"]["model_forwards"] == 0
    assert result["trace_scope"]["training_performed"] is False
    assert len(result["results"]) == 15
    assert json.loads((out / "report.json").read_text())["complete"] is True


def test_thresholds_are_fixed_and_production_untouched(tmp_path: Path) -> None:
    out = tmp_path / "audit"
    result = MODULE.run(MODULE.TRACE, out)
    assert result["fixed_thresholds"]["pair"] == 0.85
    assert result["fixed_thresholds"]["mean"] == 0.40
    assert result["fixed_thresholds"]["minimums"] == [0.0, 0.02, 0.03, 0.05, 0.08]
    assert result["trace_scope"]["production_modified"] is False
