"""Synthetic candidate inputs only; all private calls are mocked."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest
from rdkit import rdBase

from tools import review_structure_expansion_identity as review


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(review.encoded(value))


def cohort():
    molecules = [dict(molecule_id=mid, canonical_smiles=smi, split="quarantine",
                      training_authorized=False, external_overlap_status="unchecked")
                 for mid, smi in (("m1", "CCO"), ("m2", "CCN"))]
    samples = [dict(molecule_id=mid, sample_key=f"{mid}__{profile}", profile_id=profile)
               for mid, profile in (("m1", "oa00"), ("m1", "oa02"), ("m2", "oa00"))]
    return molecules, samples, [row["sample_key"] for row in samples]


def test_whole_cohort_preserves_all_styles_and_order():
    molecules, samples, retained = cohort()
    saved = deepcopy((molecules, samples, retained))
    rows = review.assemble_candidates(molecules, list(reversed(samples)), retained)
    assert (molecules, samples, retained) == saved
    assert [row["molecule_id"] for row in rows] == ["m1", "m2"]
    assert rows[0]["sample_keys"] == ["m1__oa00", "m1__oa02"]
    assert sum(row["image_count"] for row in rows) == 3


@pytest.mark.parametrize("mutation", ["duplicate_molecule", "duplicate_smiles", "duplicate_sample",
    "duplicate_retained", "foreign", "profile", "key", "missing", "training", "split", "overlap"])
def test_candidate_mutations_fail_closed(mutation):
    molecules, samples, retained = cohort()
    if mutation == "duplicate_molecule": molecules.append(dict(molecules[0]))
    elif mutation == "duplicate_smiles": molecules[1]["canonical_smiles"] = molecules[0]["canonical_smiles"]
    elif mutation == "duplicate_sample": samples.append(dict(samples[0]))
    elif mutation == "duplicate_retained": retained.append(retained[0])
    elif mutation == "foreign": samples[0]["molecule_id"] = "foreign"
    elif mutation == "profile": samples[0]["profile_id"] = "oa01"
    elif mutation == "key": samples[0]["sample_key"] = "../bad"
    elif mutation == "missing": retained.pop()
    elif mutation == "training": molecules[0]["training_authorized"] = True
    elif mutation == "split": molecules[0]["split"] = "train"
    else: molecules[0]["external_overlap_status"] = "checked"
    with pytest.raises(ValueError):
        review.assemble_candidates(molecules, samples, retained)


@pytest.fixture
def fixed(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    for name, suffix in (("DATASET", "data"), ("AUDIT", "audit.json"), ("REQUEST", "request"),
                         ("OUTPUT", "receipt"), ("PARTITION", "partition"),
                         ("AUTHORIZATION", "auth.json"), ("EVALUATION", "evaluation")):
        monkeypatch.setattr(review, name, tmp_path / suffix)
    monkeypatch.setattr(review, "COMMITMENT", review.EVALUATION / "commitment.json")
    monkeypatch.setattr(review, "COUNTS", (2, 3))
    molecules, samples, retained = cohort()
    for name, value in (("molecules.json", molecules), ("samples.json", samples),
                        ("quality-partition.json", dict(retained=retained,
                         conflict_identity_quarantine=[], same_identity_duplicates=[]))):
        write(review.DATASET / name, value)
    monkeypatch.setattr(review, "MOLECULES_SHA", review.digest(review.DATASET / "molecules.json"))
    dependency = tmp_path / "dependency.json"
    write(dependency, {"synthetic": True})
    audit = dict(integrity_passed=True, training_authorized=False, external_identity_review=False,
        retained_structures=2, retained_images=3, target_panel_passed=2,
        target_panel_failures=[], target_grid_collision_samples=[], dataset=str(review.DATASET),
        input_sha256={str(dependency): review.digest(dependency)},
        manifest_sha256={path.name: review.digest(path) for path in review.DATASET.iterdir()})
    write(review.AUDIT, audit)
    monkeypatch.setattr(review, "AUDIT_SHA", review.digest(review.AUDIT))
    commitment = dict(schema_version="chemvector-joint-commitment-1.0", dataset="fresh_eval_12k",
        sealed_access="locked_until_checkpoint_gate", rdkit_version=rdBase.rdkitVersion,
        split_counts={"fresh_dev": 8000, "sealed_test": 4000})
    digest = sha256((json.dumps(commitment, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest()
    commitment["commitment_sha256"] = digest
    monkeypatch.setattr(review, "COMMITMENT_SHA", digest)
    write(review.COMMITMENT, commitment)
    result = review.prepare()
    auth = dict(schema_version="chemvector-fixed-identity-authorization-record-0.1",
        maximum_real_private_invocations=1, automatic_retries=0, **{key: False for key in review.DENIED},
        request_directory=str(review.REQUEST), output_directory=str(review.OUTPUT), evaluation_root=str(review.EVALUATION),
        candidate_molecules=2, candidate_images=3, request_sha256=result["request_sha256"],
        candidate_identities_sha256=result["candidate_identities_sha256"], evaluation_commitment_sha256=digest,
        runner_sha256=review.RUNNER_SHA, worker_sha256=review.WORKER_SHA, wrapper_sha256=review.digest(review.__file__))
    write(review.AUTHORIZATION, auth)
    return dependency


def test_prepare_is_no_query_and_binds_source(fixed, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No private calls")
    monkeypatch.setattr(review.private_review, "run", forbidden)
    _, _, request, rows = review.validate_fixed_request()
    assert (len(rows), sum(r["image_count"] for r in rows)) == (2, 3)
    assert request["training_authorized"] is False
    assert request["private_identity_review_executed"] is False
    assert request["public_review"] == "not_performed"
    assert not (review.REQUEST / "private-attempt.json").exists()
    with pytest.raises(ValueError, match="Fresh request"):
        review.prepare()


def test_one_invocation_reserved_before_call_no_replay(fixed, monkeypatch):
    calls = []
    def fake(*args, **kwargs):
        reservation = json.loads((review.REQUEST / "private-attempt.json").read_bytes())
        assert reservation["maximum_invocations"] == 1 and reservation["automatic_retries"] == 0
        assert kwargs == {"authorized": True}
        assert args == (review.REQUEST, review.EVALUATION, review.OUTPUT)
        calls.append(args)
        return {"status": "checked", "training_authorized": False}
    monkeypatch.setattr(review.private_review, "run", fake)
    assert review.invoke_once(authorized=True)["status"] == "checked"
    with pytest.raises(FileExistsError): review.invoke_once(authorized=True)
    assert len(calls) == 1


def test_failed_invocation_still_consumes_reservation(fixed, monkeypatch):
    calls = []
    def fail(*args, **kwargs):
        calls.append(args)
        raise TimeoutError("synthetic")
    monkeypatch.setattr(review.private_review, "run", fail)
    with pytest.raises(TimeoutError): review.invoke_once(authorized=True)
    with pytest.raises(FileExistsError): review.invoke_once(authorized=True)
    assert len(calls) == 1


@pytest.mark.parametrize("mutation", ["authorization", "gpu", "training", "private_access", "retry",
    "bool_invocations", "wrapper", "request_sha", "candidate_bytes", "request_bytes", "dependency", "output"])
def test_invalid_binding_or_scope_does_not_reserve(fixed, monkeypatch, mutation):
    def forbidden(*args, **kwargs): raise AssertionError("No private calls")
    monkeypatch.setattr(review.private_review, "run", forbidden)
    auth = json.loads(review.AUTHORIZATION.read_bytes())
    if mutation == "gpu": monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    elif mutation == "training": auth["training_authorized"] = True
    elif mutation == "private_access": auth["model_side_private_sample_access_authorized"] = True
    elif mutation == "retry": auth["automatic_retries"] = 1
    elif mutation == "bool_invocations": auth["maximum_real_private_invocations"] = True
    elif mutation == "wrapper": auth["wrapper_sha256"] = "changed"
    elif mutation == "request_sha": auth["request_sha256"] = "changed"
    elif mutation == "candidate_bytes":
        path = review.REQUEST / "candidate-identities.json"
        path.write_bytes(path.read_bytes() + b" ")
    elif mutation == "request_bytes":
        path = review.REQUEST / "request.json"
        path.write_bytes(path.read_bytes() + b" ")
    elif mutation == "dependency": write(fixed, {"synthetic": "changed"})
    elif mutation == "output": auth["output_directory"] += "-other"
    write(review.AUTHORIZATION, auth)
    with pytest.raises(ValueError): review.invoke_once(authorized=mutation != "authorization")
    assert not (review.REQUEST / "private-attempt.json").exists()


def test_model_side_reader_never_opens_evaluation_manifests(fixed):
    for path in (review.EVALUATION / "molecules.csv", review.EVALUATION / "samples.csv",
                 review.EVALUATION / "sealed_test" / "molecules.csv"):
        with pytest.raises(ValueError, match="Only nonsymlink"):
            review.read(path)
    assert review.read(review.COMMITMENT)


def test_partition_drops_whole_structure_without_mutation():
    rows = review.assemble_candidates(*cohort())
    before = deepcopy(rows)
    receipt = dict(status="checked_conservative", excluded_candidates=[
        dict(source="curriculum", molecule_id="m1", rules=[review.RULES[2]])])
    remaining, rejected = review.partition_candidates(rows, receipt)
    assert rows == before
    assert len(remaining) == len(rejected) == 1
    assert remaining[0] == rows[1] and rejected[0]["sample_keys"] == rows[0]["sample_keys"]
    assert remaining[0]["image_count"] == 1 and rejected[0]["image_count"] == 2


@pytest.mark.parametrize("mutation", ["incomplete", "foreign", "duplicate", "rule"])
def test_partition_rejects_invalid_receipt(mutation):
    rows = review.assemble_candidates(*cohort())
    receipt = dict(status="checked", excluded_candidates=[
        dict(source="curriculum", molecule_id="m1", rules=[review.RULES[2]])])
    if mutation == "incomplete": receipt["status"] = "incomplete"
    elif mutation == "foreign": receipt["excluded_candidates"][0]["molecule_id"] = "private-canary"
    elif mutation == "duplicate": receipt["excluded_candidates"] *= 2
    else: receipt["excluded_candidates"][0]["rules"] = ["private-canary"]
    with pytest.raises(ValueError): review.partition_candidates(rows, receipt)


def test_cli_requires_explicit_invocation_flag():
    with pytest.raises(SystemExit): review.main(["invoke"])
