from __future__ import annotations

import json

from tools.audit_identity_aligned_vertex_integration_shadow import run


def test_integration_shadow_fails_closed_without_exported_candidate_artifact(tmp_path):
    result = run(
        candidate_path=tmp_path / "missing-candidates.jsonl",
        output=tmp_path / "report.json",
    )
    assert result["complete"] is False
    assert result["training_performed"] is False
    assert result["sealed_test_read"] is False
    assert result["gate"]["go"] is False
    assert result["candidate_artifact"]["available"] is False
    assert json.loads((tmp_path / "report.json").read_text())["gate"]["go"] is False


def test_candidate_parser_rejects_target_fields(tmp_path):
    from tools.audit_identity_aligned_vertex_integration_shadow import _load_rows, _parse_candidates

    rows = _load_rows()
    key = next(iter(rows))
    path = tmp_path / "candidates.jsonl"
    path.write_text(json.dumps({
        "sample_key": key,
        "source": "shadow",
        "candidates": [{"x": 0.5, "y": 0.5, "score": 0.9}],
        "targets": [],
    }) + "\n")
    try:
        _parse_candidates(path, rows)
    except ValueError as exc:
        assert "forbidden target field" in str(exc)
    else:
        raise AssertionError("target-bearing candidate row was accepted")
