"""Synthetic private identities only, including the optional real OS sandbox."""

from copy import deepcopy
import csv
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest
from rdkit import rdBase

from Data.sdf.chemvector_joint_v1.identity_review_worker import RULES, ReviewFailure, review, same_legacy_scaffold, set_hash
from chemvector.training.curriculum.admission import identities, require_training_admission
from chemvector.training.curriculum.private_review import IDENTITIES, WORKER, failed, run, sandbox_command, validate_receipt


def encode(value):
    return (json.dumps(value, sort_keys=True) + "\n").encode()


def csv_bytes(rows):
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode()


def fixture(counts=None):
    counts = counts or {"fresh_dev": 4, "sealed_test": 4}
    manifests, molecules, committed_csv = {}, [], []
    for split in counts:
        smi = "[13CH3]CO" if split == "fresh_dev" else "Oc1ccccc1"
        ids = identities(smi)
        molrows, samples = [], []
        for i in range(counts[split] // 4):
            mid = f"PRIVATE_CANARY_{split}_{i}"
            row = dict(molecule_id=mid, inchikey=ids["inchikey"], scaffold=ids["legacy_scaffold"],
                       canonical_isomeric_smiles=smi, split=split)
            molrows.append(row)
            molecules.append(row)
            for variant in ("original", "rdkit", "coordgen", "stress"):
                samples.append(dict(sample_key=mid + variant, molecule_id=mid, inchikey=ids["inchikey"],
                                    scaffold=ids["legacy_scaffold"], split=split, variant=variant))
        payload = csv_bytes(samples)
        committed_csv.append(dict(path="samples.csv" if split == "fresh_dev" else "sealed_test/samples.csv",
                                  sha256=sha256(payload).hexdigest(), bytes=len(payload)))
        manifests[split] = payload, csv_bytes(molrows)
    commitment = dict(schema_version="chemvector-joint-commitment-1.0", dataset="fresh_eval_12k",
                      sealed_access="locked_until_checkpoint_gate", rdkit_version=rdBase.rdkitVersion,
                      split_counts=counts, samples=sum(counts.values()), molecules=len(molecules), samples_csv=committed_csv,
                      **{field + "_set_sha256": set_hash(row[field] for row in molecules)
                         for field in ("molecule_id", "inchikey", "scaffold")})
    commitment["commitment_sha256"] = sha256((json.dumps(commitment, ensure_ascii=True, sort_keys=True,
        separators=(",", ":")) + "\n").encode()).hexdigest()
    candidates = [dict(source="curriculum", molecule_id=f"candidate_{i}", canonical_smiles=smi,
                       sample_keys=[f"candidate_{i}__reference"], image_count=1, **identities(smi))
                  for i, smi in enumerate(("CCO", "Cc1ccccc1"))]
    request = dict(schema_version="chemvector-identity-review-request-0.1", training_authorized=False,
                   requested_checks=list(RULES), rdkit_version=rdBase.rdkitVersion,
                   identity_implementation_sha256=sha256(IDENTITIES.read_bytes()).hexdigest(),
                   candidate_identities_sha256=sha256(encode(candidates)).hexdigest(),
                   evaluation_commitment_sha256=commitment["commitment_sha256"], candidate_molecules=2, candidate_images=2)
    return encode(request), encode(candidates), encode(commitment), manifests


def call_core(args, identity_function=identities):
    counts = json.loads(args[2])["split_counts"]
    return review(*args, identity_function, sha256(IDENTITIES.read_bytes()).hexdigest(), rdBase.rdkitVersion,
                  expected_split_counts=counts)


def test_strict_full_coverage_with_only_candidate_side_output():
    args = fixture()
    before = deepcopy(args)
    result = call_core(args)
    assert args == before
    assert result["status"] == "checked"
    assert result["candidate_overlap_counts_by_rule"] == dict(full_inchikey=0, connectivity_prefix=1, stereo_isotope_independent_group_key=2)
    assert result["evaluation_molecules_by_split"] == dict(fresh_dev=1, sealed_test=1)
    assert {r["molecule_id"] for r in result["excluded_candidates"]} == {"candidate_0", "candidate_1"}
    encoded = json.dumps(result)
    for forbidden in ("PRIVATE_CANARY", "[13CH3]CO", "Oc1ccccc1", identities("[13CH3]CO")["inchikey"]):
        assert forbidden not in encoded
    with pytest.raises(ValueError, match="must not start"):
        require_training_admission(result)


def test_unresolved_is_incomplete_not_negative_evidence():
    def unresolved(smi):
        if smi == "[13CH3]CO":
            raise RuntimeError("PRIVATE_CANARY with secret SMILES")
        return identities(smi)
    result = call_core(fixture(), unresolved)
    assert result["status"] == "incomplete"
    assert result["unresolved_molecules_by_split"] == dict(fresh_dev=1, sealed_test=0)
    assert result["excluded_candidates"] == [{"source": "curriculum", "molecule_id": "candidate_1",
                                               "rules": ["stereo_isotope_independent_group_key"]}]
    assert "PRIVATE_CANARY" not in json.dumps(result)


@pytest.mark.parametrize("left,right,equal", [
    ("C1=CC=CC=C1", "c1ccccc1", True),
    ("ACYCLIC:CC(O)", "ACYCLIC:CCO", True),
    ("ACYCLIC:CCO", "ACYCLIC:[13CH3]CO", False),
    ("C[C@H]1CCCCO1", "C[C@@H]1CCCCO1", False),
    ("c1ccccc1", "C1CCCCC1", False),
    ("ACYCLIC:CCO", "CCO", False),
    ("ACYCLIC:C[NH3+]", "ACYCLIC:CN", False),
])
def test_legacy_scaffold_normalization_never_erases_chemical_attributes(left, right, equal):
    assert same_legacy_scaffold(left, right) is equal


@pytest.mark.parametrize("scaffold,expected_status", [("ACYCLIC:CCO", "checked"), ("ACYCLIC:CCCO", "checked_conservative")])
def test_requested_group_is_independently_bound_to_committed_scaffold(scaffold, expected_status):
    request, candidates, commitment, manifests = fixture()
    sample, mol = manifests["fresh_dev"]
    original = "ACYCLIC:[13CH3]CO".encode()
    manifests["fresh_dev"] = sample.replace(original, scaffold.encode()), mol.replace(original, scaffold.encode())
    comm = json.loads(commitment)
    comm.pop("commitment_sha256")
    comm["samples_csv"][0].update(sha256=sha256(manifests["fresh_dev"][0]).hexdigest(), bytes=len(manifests["fresh_dev"][0]))
    comm["scaffold_set_sha256"] = set_hash([scaffold, "c1ccccc1"])
    comm["commitment_sha256"] = sha256((json.dumps(comm, ensure_ascii=True, sort_keys=True,
        separators=(",", ":")) + "\n").encode()).hexdigest()
    req = json.loads(request)
    req["evaluation_commitment_sha256"] = comm["commitment_sha256"]
    candidate_rows = json.loads(candidates)
    candidate_rows.append(dict(source="curriculum", molecule_id="candidate_2", canonical_smiles="CCCO",
                               sample_keys=["candidate_2__reference"], image_count=1, **identities("CCCO")))
    candidates = encode(candidate_rows)
    req.update(candidate_molecules=3, candidate_images=3, candidate_identities_sha256=sha256(candidates).hexdigest())
    result = call_core((encode(req), candidates, encode(comm), manifests))
    assert result["status"] == expected_status
    if expected_status == "checked":
        assert result["legacy_scaffold_differences_resolved_by_requested_group_rule"] == 1
        assert result["committed_group_binding_mismatches"] == 0
    else:
        assert result["committed_group_binding_mismatches"] == 1
        assert result["unresolved_molecules_by_split"]["fresh_dev"] == 0
        # Candidate matching ONLY the committed legacy group is still excluded.
        assert any(row["molecule_id"] == "candidate_2" and row["rules"] == ["stereo_isotope_independent_group_key"]
                   for row in result["excluded_candidates"])
        assert not result["training_authorized"]


@pytest.mark.parametrize("mutation,code", [
    ("candidate_bytes", "candidate_binding_failed"), ("commitment_bytes", "commitment_binding_failed"),
    ("samples", "private_sample_binding_failed"), ("molecule_key", "private_manifest_invalid"),
    ("smiles", "private_key_mismatch"), ("missing", "private_manifest_invalid"),
    ("implementation", "identity_implementation_mismatch"), ("count", "candidate_binding_failed"),
])
def test_tampered_or_incomplete_inputs_fail_without_private_exception(mutation, code):
    request, candidates, commitment, manifests = fixture()
    if mutation == "candidate_bytes":
        candidates += b" "
    elif mutation == "commitment_bytes":
        value = json.loads(commitment); value["samples"] = 100; commitment = encode(value)
    elif mutation == "samples":
        sample, mol = manifests["sealed_test"]; manifests["sealed_test"] = (sample + b"x", mol)
    elif mutation in {"molecule_key", "smiles"}:
        sample, mol = manifests["sealed_test"]
        old, new = (identities("Oc1ccccc1")["inchikey"], "wrong") if mutation == "molecule_key" else ("Oc1ccccc1", "Cc1ccccc1")
        manifests["sealed_test"] = sample, mol.replace(old.encode(), new.encode())
    elif mutation == "missing":
        manifests.pop("sealed_test")
    else:
        value = json.loads(request)
        value["identity_implementation_sha256" if mutation == "implementation" else "candidate_molecules"] = "wrong"
        request = encode(value)
    with pytest.raises(ReviewFailure) as error:
        call_core((request, candidates, commitment, manifests))
    assert str(error.value) == code


def full_receipt():
    args = fixture({"fresh_dev": 8000, "sealed_test": 4000})
    result = call_core(args)
    result["worker_sha256"] = sha256(WORKER.read_bytes()).hexdigest()
    return args, result


@pytest.fixture(scope="module")
def complete_receipt():
    return full_receipt()


@pytest.mark.parametrize("mutation", ["private_field", "private_id", "private_rule", "bool_count", "count", "unknown", "schema", "worker", "omitted_candidate"])
def test_parent_rejects_private_or_unbound_fields(complete_receipt, mutation):
    args, original = complete_receipt
    value = deepcopy(original)
    if mutation == "private_field":
        value["private_smiles"] = "PRIVATE_CANARY"
    elif mutation == "private_id":
        value["excluded_candidates"][0]["molecule_id"] = "PRIVATE_CANARY"
    elif mutation == "private_rule":
        value["excluded_candidates"][0]["rules"] = ["PRIVATE_CANARY"]
    elif mutation == "bool_count":
        value["candidate_overlap_counts_by_rule"]["full_inchikey"] = False
    elif mutation == "count":
        value["evaluation_molecules_by_split"]["sealed_test"] -= 1
    elif mutation == "unknown":
        value["unresolved_molecules_by_split"]["sealed_test"] = 1
    elif mutation == "schema":
        value["schema_version"] = "other"
    elif mutation == "worker":
        value["worker_sha256"] = "wrong"
    else:
        value["excluded_candidates"].pop()
    with pytest.raises(ValueError, match="Unsafe"):
        validate_receipt(value, args[0], args[1], sha256(WORKER.read_bytes()).hexdigest())


def test_failure_receipts_never_relay_exception_text(complete_receipt):
    args, result = complete_receipt
    assert validate_receipt(result, args[0], args[1], result["worker_sha256"]) == result
    value = failed("worker_failed")
    assert validate_receipt(value, args[0], args[1], "unused") == value
    value["failure_code"] = "PRIVATE_CANARY"
    with pytest.raises(ValueError):
        validate_receipt(value, args[0], args[1], "unused")


def test_authorization_and_mount_scope(tmp_path):
    with pytest.raises(ValueError, match="authorization"):
        run(tmp_path, tmp_path, tmp_path / "out")
    command = sandbox_command([(3, "/private/fresh_dev_samples.csv")], [(4, "/runtime/lib")])
    assert "--unshare-all" in command
    assert command[command.index("--cap-drop") + 1] == "ALL"
    assert command[command.index("--ro-bind-data") + 2] == "/private/fresh_dev_samples.csv"
    assert not {"--bind", "--chmod", "--share-net", "--dev-bind"}.intersection(command)
    assert command[-4:] == ["/runtime/bin/python3.13", "-I", "-S", "/code/worker.py"]


@pytest.mark.parametrize("stdout", [b'PRIVATE_CANARY_RAW_STDOUT', b'{"schema_version":"wrong","secret":"PRIVATE_CANARY"}'])
def test_parent_never_persists_unvalidated_output_or_reads_private_files(tmp_path, monkeypatch, stdout):
    request, candidates, _, _ = fixture()
    root = tmp_path / "request"
    root.mkdir()
    (root / "request.json").write_bytes(request)
    (root / "candidate-identities.json").write_bytes(candidates)
    private = tmp_path / "private-not-opened"
    original = Path.read_bytes
    def checked_read(path):
        assert not path.is_relative_to(private)
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", checked_read)
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a, 0, stdout, b"PRIVATE_CANARY_STDERR"))
    output = tmp_path / "out"
    result = run(root, private, output, authorized=True)
    assert result == failed("unsafe_receipt")
    assert all("PRIVATE_CANARY" not in path.read_text() for path in output.iterdir())


def test_actual_sandbox_on_synthetic_locked_manifests(tmp_path, complete_receipt):
    if not shutil.which("bwrap") or not shutil.which("sudo") or subprocess.run(
            ["sudo", "-n", "/usr/bin/true"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
        pytest.skip("Requires bubblewrap and noninteractive custodian access")
    args, _ = complete_receipt
    request, candidates, commitment, manifests = args
    request_dir, root = tmp_path / "request", tmp_path / "private"
    request_dir.mkdir(); root.mkdir()
    (root / "sealed_test").mkdir()
    (request_dir / "request.json").write_bytes(request)
    (request_dir / "candidate-identities.json").write_bytes(candidates)
    (root / "commitment.json").write_bytes(commitment)
    locked = []
    for split, (samples, molecules) in manifests.items():
        directory = root if split == "fresh_dev" else root / "sealed_test"
        for name, data in (("samples.csv", samples), ("molecules.csv", molecules)):
            path = directory / name
            path.write_bytes(data)
            if split == "sealed_test":
                path.chmod(0)
                locked.append(path)
    sealed = root / "sealed_test"
    sealed.chmod(0)
    # Fake secrets/images outside the mount allowlist are never passed to bwrap.
    (root / "secret-image.png").write_bytes(b"PRIVATE_CANARY_IMAGE")
    before = sealed.stat().st_mode
    try:
        result = run(request_dir, root, tmp_path / "output", authorized=True)
        assert result["status"] == "checked", result
        assert result["candidate_overlap_counts_by_rule"]["connectivity_prefix"] == 1
        assert sealed.stat().st_mode == before
        assert "PRIVATE_CANARY" not in json.dumps(result)
        assert not result["images_or_sdf_read"]
        with pytest.raises(FileExistsError):
            run(request_dir, root, tmp_path / "output", authorized=True)
    finally:
        # Only restore this test's artificial private directory for pytest cleanup.
        sealed.chmod(0o700)
        for path in locked:
            assert path.stat().st_mode & 0o777 == 0
            path.chmod(0o600)
