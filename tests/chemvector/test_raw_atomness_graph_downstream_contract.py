from pathlib import Path

from tools.audit_raw_atomness_graph_downstream_contract import audit


ROOT = Path(__file__).resolve().parents[2]


def test_current_release_is_read_only_and_graph_downstream_not_admitted():
    result = audit()
    assert result["read_only"] is True
    assert result["training_performed"] is False
    assert result["sealed_test_read"] is False
    assert result["checks"]["cache_valid"] is True
    assert result["checks"]["sidecar_feature_contract"] is True
    assert result["checks"]["graph_artifact_schema"] is False
    assert result["admission"]["graph_downstream_training_ready"] is False


def test_missing_graph_artifact_is_explicit_no_go(tmp_path):
    result = audit(graph_artifact=tmp_path / "missing.jsonl")
    status = result["graph_artifact"]
    assert status["provided"] is True
    assert status["available"] is False
    assert "does not exist" in status["reason"]
