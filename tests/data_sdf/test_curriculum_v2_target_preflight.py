"""Synthetic-only selected-target tests; no checkpoint or evaluation data."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest
import torch

from Data.sdf.chemvector_curriculum_v0_2 import build as builder
from Data.sdf.chemvector_curriculum_v0_2.catalog import entries
from tools.audit_curriculum_expansion_v02 import audit
from tools.prepare_curriculum_v02_quality_handoff import prepare_handoff
from tools import preflight_curriculum_v02_targets as preflight
from tools import prepare_curriculum_v02_identity_request as request_tool
from rdkit import rdBase


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    parent = tmp_path_factory.mktemp("selected-targets")
    root = parent / "dataset"
    records = [dict(record_id=f"test_{i}", name="test", stage="atoms_bonds", concepts=[], smiles=smiles)
               for i, smiles in enumerate(("CC", "CCO", "CC#N", "c1ccccc1", "F[C@H](Cl)Br",
                                            "CC=CCC", "C/C=C/CC"))]
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("CUDA_VISIBLE_DEVICES", "")
        patch.setattr(builder, "entries", lambda: entries(records))
        patch.setattr(builder, "bounded_map", lambda fn, jobs, **kwargs: (fn(job) for job in jobs))
        builder.build(root, workers=1)
    audit_path, handoff_path = parent / "audit.json", parent / "handoff.json"
    audit(root, audit_path)
    handoff = prepare_handoff(root, audit_path, handoff_path)
    return root, handoff_path, handoff


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    torch.set_num_threads(1)


def one_sample(corpus, smiles="CCO"):
    root, _, _ = corpus
    molecules = json.loads((root / "molecules.json").read_text())
    molecule = next(row for row in molecules if row["canonical_smiles"] == smiles)
    return preflight.data.load_sample(root, molecule["molecule_id"] + "__reference", purpose="diagnostic")


def test_complete_selected_only_and_immutable(corpus, tmp_path):
    root, handoff_path, handoff = corpus
    original = preflight.snapshot(root.rglob("*.*"))
    output = tmp_path / "preflight"
    result = preflight.run(handoff_path, output, max_seconds=120)
    assert result["complete"] is True and result["status"] == "passed_target_compatibility_only"
    assert result["checked_images"] == result["compatible_images"] == len(handoff["retained_candidates"])
    assert result["failed_images"] == 0 and result["training_authorized"] is False
    assert result["training_loader_rejections"] == result["training_collator_rejections"] == result["checked_images"]
    rows = [json.loads(line) for line in (output / "samples.jsonl").read_text().splitlines()]
    assert [row["sample_key"] for row in rows] == [row["sample_key"] for row in handoff["retained_candidates"]]
    assert result["checked_images"] < json.loads((root / "summary.json").read_text())["accepted_images"]
    assert "AROMATIC" in result["chemical_bonds"] and "TRIPLE" in result["chemical_bonds"]
    assert "BEGINWEDGE" in result["drawing_directions"] or "BEGINDASH" in result["drawing_directions"]
    assert preflight.snapshot(root.rglob("*.*")) == original
    with pytest.raises(FileExistsError):
        preflight.run(handoff_path, output)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "extra", "pixel", "authorization"])
def test_fixed_selection_cannot_be_mutated(corpus, mutation):
    handoff = deepcopy(corpus[2])
    if mutation == "missing":
        handoff["retained_candidates"].pop()
    elif mutation == "duplicate":
        handoff["retained_candidates"].append(handoff["retained_candidates"][0])
    elif mutation == "extra":
        handoff["retained_candidates"].append(handoff["quarantined_samples"][0])
    elif mutation == "pixel":
        handoff["retained_candidates"][0]["png_sha256"] = "0" * 64
    else:
        handoff["training_authorized"] = True
    with pytest.raises(ValueError):
        preflight.check_handoff(handoff)


@pytest.mark.parametrize("field", ["bond_chemical", "bond_display", "bond_direction", "bond_atom_refs",
                                  "atom_roles", "atom_anchors", "atom_mask", "pixel_values"])
def test_wrong_tensor_semantics_or_nonfinite_rejected(corpus, field):
    sample = one_sample(corpus)
    batch = preflight.data.CurriculumCollator(purpose="diagnostic")([sample])
    preflight.validate_targets(sample, batch, preflight.AtomBondSceneConfig())
    tensor = batch[field]
    if field == "pixel_values":
        tensor.flatten()[0] = float("nan")
    elif field == "atom_mask":
        tensor[0, 0] = False
    else:
        tensor.flatten()[0] += 1
    with pytest.raises(ValueError):
        preflight.validate_targets(sample, batch, preflight.AtomBondSceneConfig())


def test_collision_and_query_overflow_rejected(corpus):
    sample = one_sample(corpus, "CC")
    sample["layers"]["atoms"][1]["center"] = sample["layers"]["atoms"][0]["center"][:]
    sample["layers"]["bonds"][0]["end"] = sample["layers"]["bonds"][0]["begin"][:]
    batch = preflight.data.CurriculumCollator(purpose="diagnostic")([sample])
    with pytest.raises(ValueError, match="colliding"):
        preflight.validate_targets(sample, batch, preflight.AtomBondSceneConfig())
    tiny = replace(preflight.AtomBondSceneConfig(), atom_queries=1)
    with pytest.raises(ValueError, match="capacity"):
        preflight.data.CurriculumCollator(tiny, purpose="diagnostic")([sample])


def test_interrupted_job_cannot_report_success(corpus, tmp_path, monkeypatch):
    def fail(*args):
        raise InterruptedError("synthetic interruption")
    monkeypatch.setattr(preflight, "verify_sample", fail)
    output = tmp_path / "interrupted"
    with pytest.raises(InterruptedError):
        preflight.run(corpus[1], output, max_seconds=120)
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "failed_or_incomplete" and report["complete"] is False
    assert report["checked_images"] == 0 and report["training_authorized"] is False
    row = json.loads((output / "samples.jsonl").read_text())
    assert row["artifact_integrity_failure"] is True


def test_target_failure_keeps_denominator_but_blocks_pass(corpus, tmp_path, monkeypatch):
    def incompatible(*args):
        raise ValueError("synthetic incompatible target")
    monkeypatch.setattr(preflight, "validate_targets", incompatible)
    output = tmp_path / "incompatible"
    report = preflight.run(corpus[1], output, max_seconds=120)
    assert report["status"] == "failed_target_compatibility" and report["complete"] is True
    assert report["failed_images"] == report["selected_images"]
    assert report["compatible_images"] == 0


def test_cpu_guard(corpus, tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="CPU-only"):
        preflight.run(corpus[1], tmp_path / "forbidden")
    assert not (tmp_path / "forbidden").exists()


def test_request_packages_all_selected_and_never_reviews(corpus, tmp_path, monkeypatch):
    root, handoff_path, handoff = corpus
    preflight_output = tmp_path / "passed-preflight"
    preflight.run(handoff_path, preflight_output, max_seconds=120)
    commitment = dict(schema_version="chemvector-joint-commitment-1.0", dataset="fresh_eval_12k",
                      sealed_access="locked_until_checkpoint_gate", split_counts={"fresh_dev": 8000, "sealed_test": 4000},
                      rdkit_version=rdBase.rdkitVersion,
                      samples_csv="/this/path/must/not/be/read")
    canonical = (json.dumps(commitment, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode()
    commitment["commitment_sha256"] = preflight.digest(canonical)
    commitment_path = tmp_path / "synthetic-aggregate.json"
    preflight.write_new(commitment_path, commitment)
    monkeypatch.setattr(request_tool, "COMMITMENT", commitment_path)
    output = tmp_path / "request"
    report = request_tool.prepare(preflight_output, output)
    candidates_raw = (output / "candidate-identities.json").read_bytes()
    candidates = json.loads(candidates_raw)
    assert report["candidate_images"] == len(handoff["retained_candidates"])
    assert report["candidate_molecules"] == len({row["molecule_id"] for row in handoff["retained_candidates"]})
    assert report["candidate_identities_sha256"] == preflight.digest(candidates_raw)
    assert report["authorization_status"] == "not_granted_for_this_batch"
    assert report["private_identity_review_executed"] is report["training_authorized"] is False
    assert {key for candidate in candidates for key in candidate["sample_keys"]} == {
        row["sample_key"] for row in handoff["retained_candidates"]}
    assert not (output / "authorization.json").exists()
    with pytest.raises(ValueError, match="fresh"):
        request_tool.prepare(preflight_output, output)


@pytest.mark.parametrize("mutation", ["partial", "failure", "wrong_journal", "wrong_count", "authorized"])
def test_request_refuses_incomplete_or_changed_preflight(mutation):
    selected = [{"sample_key": "m__reference", "molecule_id": "m"}]
    journal = preflight.encoded({**selected[0], "compatible": True})
    # The journal is JSONL, unlike the pretty-printed canonical request bytes.
    journal = (json.dumps({**selected[0], "compatible": True}) + "\n").encode()
    report = dict(schema_version=preflight.SCHEMA, status="passed_target_compatibility_only", complete=True,
                  failed_images=0, training_authorized=False, training_performed=False,
                  external_identity_queries=0, external_overlap_status="unchecked", dataset="synthetic",
                  selected_images=1, checked_images=1, compatible_images=1, checked_molecules=1,
                  journal_sha256=preflight.digest(journal))
    handoff = {"retained_candidates": selected, "dataset": "synthetic"}
    request_tool.validate_preflight(report, handoff, journal)
    if mutation == "partial":
        report["complete"] = False
    elif mutation == "failure":
        report["failed_images"] = 1
    elif mutation == "wrong_journal":
        journal += b"\n"
    elif mutation == "wrong_count":
        report["selected_images"] = 2
    else:
        report["training_authorized"] = True
    with pytest.raises(ValueError):
        request_tool.validate_preflight(report, handoff, journal)


def test_identity_candidate_grouping_ignores_excluded_raw_images():
    molecules = [dict(molecule_id="m", canonical_smiles="CCO", split="quarantine", training_authorized=False,
                      external_overlap_status="unchecked"),
                 dict(molecule_id="excluded", canonical_smiles="CC", split="quarantine", training_authorized=False,
                      external_overlap_status="unchecked")]
    selected = [dict(molecule_id="m", sample_key="m__reference", profile_id="reference"),
                dict(molecule_id="m", sample_key="m__serif", profile_id="serif")]
    candidates = request_tool.assemble_candidates(molecules, selected)
    assert len(candidates) == 1 and candidates[0]["image_count"] == 2
    assert candidates[0]["group_key"] == "acyclic:CCO"
    with pytest.raises(ValueError, match="identity"):
        request_tool.assemble_candidates(molecules, selected + selected[:1])
    molecules[0]["training_authorized"] = True
    with pytest.raises(ValueError, match="quarantined"):
        request_tool.assemble_candidates(molecules, selected)
