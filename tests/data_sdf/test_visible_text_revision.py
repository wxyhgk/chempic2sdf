import copy
from dataclasses import replace
import json

import pytest

from Data.sdf.visible_text_revision import (
    REVISION_SCHEMA, VisibleTextCorrection, VisibleTextRevision,
    revise_training_visible_text, sha256,
)


def fixture():
    layers = {"schema_version": "chemvision-layers-0.2", "sample_key": "train:sample",
              "split": "train", "molecule_id": "synthetic", "width": 768, "height": 768,
              "atoms": [
                  {"atom_id": 1, "element": "C", "implicit_h_count": 2,
                   "formal_charge": 0, "aromatic": False, "visible_text": True,
                   "bbox": [10., 20., 20., 32.], "center": [15., 26.], "text": "CH2"},
                  {"atom_id": 2, "element": "N", "implicit_h_count": 2,
                   "formal_charge": 0, "visible_text": True,
                   "bbox": [40., 20., 70., 32.], "center": [45., 26.], "text": "NH2"},
              ], "bonds": [{"begin_atom": 1, "end_atom": 2, "chemical_type": "SINGLE"}]}
    payload = json.dumps(layers, sort_keys=True).encode()
    # These opaque bytes stand in for a caller-validated PNG in this unit test.
    png = b"source raster bytes"
    correction = VisibleTextCorrection(1, "CH2", "C", "reviewed_source_raster", "review:synthetic-C")
    revision = VisibleTextRevision("train:sample", sha256(payload), sha256(png), (correction,))
    return layers, payload, png, revision


def test_revision_changes_only_visible_text_and_version_provenance():
    layers, payload, png, revision = fixture()
    original = copy.deepcopy(layers)
    result = revise_training_visible_text(payload, png, revision)
    assert result["schema_version"] == REVISION_SCHEMA
    assert result["atoms"][0]["text"] == "C"
    assert result["atoms"][0]["implicit_h_count"] == 2
    assert result["atoms"][1] == layers["atoms"][1]
    assert result["bonds"] == layers["bonds"]
    assert result["visible_text_revision"]["other_text_labels"] == "legacy_unverified"
    restored = copy.deepcopy(result)
    restored["schema_version"] = layers["schema_version"]
    del restored["visible_text_revision"]
    restored["atoms"][0]["text"] = "CH2"
    assert restored == original
    assert json.loads(payload) == original


@pytest.mark.parametrize("split", ["fresh_dev", "sealed_test", "validation", "internal_test", None])
def test_evaluation_and_unknown_splits_cannot_be_revised(split):
    layers, _, png, revision = fixture()
    layers["split"] = split
    payload = json.dumps(layers).encode()
    revision = replace(revision, source_layers_sha256=sha256(payload))
    with pytest.raises(ValueError, match="Only train"):
        revise_training_visible_text(payload, png, revision)


@pytest.mark.parametrize("field,value", [
    ("source_layers_sha256", "wrong"), ("source_png_sha256", "wrong"),
    ("sample_key", "another-image"), ("corrections", ()),
])
def test_stale_or_unbound_requests_fail(field, value):
    _, payload, png, revision = fixture()
    with pytest.raises(ValueError):
        revise_training_visible_text(payload, png, replace(revision, **{field: value}))


@pytest.mark.parametrize("field,value", [
    ("atom_id", 99), ("atom_id", True), ("expected_text", "stale"),
    ("visible_text", ""), ("visible_text", "CH2"),
    ("evidence_kind", "chemical_fallback"), ("evidence_kind", "nofreetype_diagnostic"),
    ("evidence_reference", ""),
])
def test_unsupported_corrections_fail(field, value):
    _, payload, png, revision = fixture()
    correction = replace(revision.corrections[0], **{field: value})
    with pytest.raises(ValueError):
        revise_training_visible_text(payload, png, replace(revision, corrections=(correction,)))


def test_duplicate_corrections_fail():
    _, payload, png, revision = fixture()
    with pytest.raises(ValueError, match="duplicated"):
        revise_training_visible_text(payload, png, replace(revision, corrections=revision.corrections * 2))


@pytest.mark.parametrize("field,value", [("visible_text", False), ("bbox", None),
                                        ("bbox", [0, 0, 0, 1]), ("bbox", [0, 0, float('nan'), 1])])
def test_revision_cannot_create_text_objects_or_geometry(field, value):
    layers, _, png, revision = fixture()
    layers["atoms"][0][field] = value
    payload = json.dumps(layers).encode()
    with pytest.raises(ValueError):
        revise_training_visible_text(payload, png, replace(revision, source_layers_sha256=sha256(payload)))


def test_chemically_valid_displayed_CH2_is_not_globally_replaced():
    layers, _, png, revision = fixture()
    layers["atoms"][1].update(element="C", text="CH2")
    payload = json.dumps(layers).encode()
    result = revise_training_visible_text(payload, png, replace(revision, source_layers_sha256=sha256(payload)))
    assert [a["text"] for a in result["atoms"]] == ["C", "CH2"]


def test_reapplying_revision_is_rejected():
    _, payload, png, revision = fixture()
    result = revise_training_visible_text(payload, png, revision)
    revised_payload = json.dumps(result).encode()
    with pytest.raises(ValueError, match="unrevised"):
        revise_training_visible_text(revised_payload, png,
                                     replace(revision, source_layers_sha256=sha256(revised_payload)))


def test_rendered_linear_carbon_can_be_revised_without_losing_implicit_hydrogens():
    from rdkit import Chem
    from rdkit.Geometry import Point3D
    from Data.sdf.rendered_text_evidence import no_freetype_text_diagnostic

    mol = Chem.MolFromSmiles("CCC")
    conformer = Chem.Conformer(3)
    for index in range(3):
        conformer.SetAtomPosition(index, Point3D(index * 1.5, 0, 0))
    mol.AddConformer(conformer)
    # Diagnostic cross-check only; the revision below still requires explicit
    # reviewed-source evidence, not blind trust in another text backend.
    diagnostic = no_freetype_text_diagnostic(mol, {})
    assert diagnostic["labels"][1] == ["C"]
    assert diagnostic["revision_authorized"] is False
    _, payload, png, revision = fixture()
    result = revise_training_visible_text(payload, png, revision)
    assert result["atoms"][0]["text"] == "C"
    assert result["atoms"][0]["implicit_h_count"] == mol.GetAtomWithIdx(1).GetTotalNumHs() == 2


@pytest.mark.parametrize("sample_key", [None, "", "  ", 123])
def test_missing_or_invalid_sample_identity_is_rejected(sample_key):
    layers, _, png, revision = fixture()
    layers["sample_key"] = sample_key
    payload = json.dumps(layers).encode()
    with pytest.raises(ValueError, match="sample identity"):
        revise_training_visible_text(payload, png, replace(
            revision, sample_key=sample_key, source_layers_sha256=sha256(payload)))


@pytest.mark.parametrize("old_text", [None, "", "  ", 123])
def test_missing_or_invalid_expected_transcription_is_rejected(old_text):
    layers, _, png, revision = fixture()
    layers["atoms"][0]["text"] = old_text
    payload = json.dumps(layers).encode()
    correction = replace(revision.corrections[0], expected_text=old_text)
    with pytest.raises(ValueError, match="old transcription"):
        revise_training_visible_text(payload, png, replace(
            revision, corrections=(correction,), source_layers_sha256=sha256(payload)))


def test_boolean_coordinates_are_not_valid_geometry():
    layers, _, png, revision = fixture()
    layers["atoms"][0]["bbox"] = [False, False, True, True]
    payload = json.dumps(layers).encode()
    with pytest.raises(ValueError, match="bbox"):
        revise_training_visible_text(payload, png, replace(revision, source_layers_sha256=sha256(payload)))


def test_existing_provenance_is_not_silently_overwritten():
    layers, _, png, revision = fixture()
    layers["visible_text_revision"] = {"previous": "provenance"}
    payload = json.dumps(layers).encode()
    with pytest.raises(ValueError, match="provenance"):
        revise_training_visible_text(payload, png, replace(revision, source_layers_sha256=sha256(payload)))


@pytest.mark.parametrize("malformation", ["root", "atoms", "atom", "identity"])
def test_malformed_layers_fail_with_validation_error(malformation):
    layers, _, png, revision = fixture()
    if malformation == "root":
        layers = []
    elif malformation == "atoms":
        layers["atoms"] = None
    elif malformation == "atom":
        layers["atoms"][0] = None
    else:
        del layers["atoms"][0]["atom_id"]
    payload = json.dumps(layers).encode()
    with pytest.raises(ValueError):
        revise_training_visible_text(payload, png, replace(revision, source_layers_sha256=sha256(payload)))


def test_renderer_diagnostic_cannot_authorize_revision():
    from rdkit import Chem
    from Data.sdf.rendered_text_evidence import no_freetype_text_diagnostic

    diagnostic = no_freetype_text_diagnostic(Chem.MolFromSmiles("CCO"), {})
    _, payload, png, revision = fixture()
    correction = replace(revision.corrections[0], evidence_kind=diagnostic["evidence_kind"])
    with pytest.raises(ValueError, match="evidence"):
        revise_training_visible_text(payload, png, replace(revision, corrections=(correction,)))
