"""All private calls are mocked; synthetic inputs only."""

import json

import pytest

from tools import review_curriculum_v02_private_identity as once


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


@pytest.fixture
def fixed(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    root = tmp_path / "request"
    root.mkdir()
    candidate = root / "candidate-identities.json"
    write(candidate, [{"molecule_id": "m", "image_count": 2, "sample_keys": ["m__a", "m__b"]}])
    monkeypatch.setattr(once, "COUNTS", (1, 2))
    monkeypatch.setattr(once, "CANDIDATES_SHA", once.digest(candidate))
    request = dict(schema_version="chemvector-identity-review-request-0.1", training_authorized=False,
                   candidate_molecules=1, candidate_images=2, candidate_identities_sha256=once.CANDIDATES_SHA,
                   input_sha256={}, evaluation_commitment_sha256="synthetic")
    write(root / "request.json", request)
    monkeypatch.setattr(once, "REQUEST_SHA", once.digest(root / "request.json"))
    output, evaluation = tmp_path / "receipt", tmp_path / "evaluation_never_opened"
    auth = dict(schema_version="chemvector-fixed-identity-authorization-record-0.1",
        maximum_real_private_invocations=1, automatic_retries=0, training_authorized=False,
        model_side_private_sample_access_authorized=False, images_sdf_or_tar_access_authorized=False,
        source_permission_changes_authorized=False, evaluation_split_changes_authorized=False,
        request_directory=str(root), output_directory=str(output), evaluation_root=str(evaluation),
        request_sha256=once.REQUEST_SHA, candidate_identities_sha256=once.CANDIDATES_SHA,
        candidate_molecules=1, candidate_images=2, evaluation_commitment_sha256="synthetic",
        runner_sha256=once.RUNNER_SHA, worker_sha256=once.WORKER_SHA, wrapper_sha256=once.digest(once.__file__))
    authorization = tmp_path / "authorization.json"
    write(authorization, auth)
    return root, evaluation, output, authorization


def test_reserved_before_single_call_and_no_replay(fixed, monkeypatch):
    calls = []
    def fake(*args, **kwargs):
        attempt = json.loads((fixed[0] / "private-attempt.json").read_text())
        assert attempt["maximum_invocations"] == 1 and attempt["automatic_retries"] == 0
        assert kwargs["authorized"] is True
        calls.append(args)
        return {"status": "checked", "training_authorized": False}
    monkeypatch.setattr(once.private_review, "run", fake)
    assert once.invoke_once(*fixed, authorized=True)["status"] == "checked"
    with pytest.raises(FileExistsError):
        once.invoke_once(*fixed, authorized=True)
    assert len(calls) == 1


def test_interruption_consumes_authorization(fixed, monkeypatch):
    calls = []
    def fake(*args, **kwargs):
        calls.append(args)
        raise TimeoutError("synthetic")
    monkeypatch.setattr(once.private_review, "run", fake)
    with pytest.raises(TimeoutError):
        once.invoke_once(*fixed, authorized=True)
    with pytest.raises(FileExistsError):
        once.invoke_once(*fixed, authorized=True)
    assert len(calls) == 1


@pytest.mark.parametrize("mutation", ["permission", "gpu", "training", "retry", "request", "candidate",
                                      "wrapper", "count", "output", "private_access"])
def test_bad_scope_cannot_reserve_or_invoke(fixed, monkeypatch, mutation):
    def forbidden(*args, **kwargs):
        raise AssertionError("Private invocation must not happen")
    monkeypatch.setattr(once.private_review, "run", forbidden)
    auth = json.loads(fixed[3].read_text())
    if mutation == "gpu": monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    elif mutation == "training": auth["training_authorized"] = True
    elif mutation == "retry": auth["automatic_retries"] = 1
    elif mutation == "request": auth["request_sha256"] = "changed"
    elif mutation == "candidate": auth["candidate_identities_sha256"] = "changed"
    elif mutation == "wrapper": auth["wrapper_sha256"] = "changed"
    elif mutation == "count": auth["candidate_images"] = 1
    elif mutation == "output": auth["output_directory"] += "-other"
    elif mutation == "private_access": auth["model_side_private_sample_access_authorized"] = True
    write(fixed[3], auth)
    with pytest.raises(ValueError):
        once.invoke_once(*fixed, authorized=mutation != "permission")
    assert not (fixed[0] / "private-attempt.json").exists()


def test_changed_candidate_side_input_blocks_before_attempt(fixed, monkeypatch):
    request_path = fixed[0] / "request.json"
    request = json.loads(request_path.read_text())
    dependency = fixed[0] / "source.json"
    write(dependency, {"version": 1})
    request["input_sha256"] = {str(dependency): once.digest(dependency)}
    write(request_path, request)
    monkeypatch.setattr(once, "REQUEST_SHA", once.digest(request_path))
    auth = json.loads(fixed[3].read_text())
    auth["request_sha256"] = once.REQUEST_SHA
    write(fixed[3], auth)
    write(dependency, {"version": 2})
    with pytest.raises(ValueError, match="input changed"):
        once.invoke_once(*fixed, authorized=True)
    assert not (fixed[0] / "private-attempt.json").exists()


def test_attempt_is_exclusive_and_cli_requires_permission(tmp_path):
    path = tmp_path / "attempt.json"
    once.reserve_once(path, {"first": True})
    saved = path.read_bytes()
    with pytest.raises(FileExistsError):
        once.reserve_once(path, {"first": False})
    assert path.read_bytes() == saved
    with pytest.raises(SystemExit):
        once.main([])
