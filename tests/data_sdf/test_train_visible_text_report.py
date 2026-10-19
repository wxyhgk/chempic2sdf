import pytest

from Data.sdf.train_visible_text_report import candidate_record, neutral_visible_carbons, summarize, write_gallery


def fixture():
    layer = {"atom_id": 0, "element": "C", "text": "CH2", "visible_text": True,
             "bbox": [0, 0, 10, 10], "center": [5, 5]}
    atom = {"id": 0, "element": "C", "implicit_h_count": 2}
    diagnostic = {"labels": {0: ["C"]}, "evidence_kind": "nofreetype_diagnostic",
                  "revision_authorized": False}
    return layer, atom, diagnostic


def test_all_visible_carbons_include_plain_C_controls():
    layer, atom, _ = fixture()
    layers = [layer, {**layer, "atom_id": 1, "text": "C"}]
    result = neutral_visible_carbons({"atoms": layers}, {"atoms": [atom, {**atom, "id": 1}]})
    assert len(result) == 2


@pytest.mark.parametrize("field,value", [("formal_charge", 1), ("isotope", 13),
                                        ("radical_electrons", 1), ("element", "N")])
def test_non_neutral_carbon_out_of_scope(field, value):
    layer, atom, _ = fixture()
    atom[field] = value
    assert not neutral_visible_carbons({"atoms": [layer]}, {"atoms": [atom]})


def test_identity_mismatch_fails():
    layer, atom, _ = fixture()
    atom["id"] = 99
    with pytest.raises(ValueError):
        neutral_visible_carbons({"atoms": [layer]}, {"atoms": [atom]})


def test_diagnostic_never_authorizes_revision():
    layer, atom, diagnostic = fixture()
    row = candidate_record("train_s", "m", "stress", layer, atom, diagnostic, {})
    assert row["diagnostic_plain_c_disagreement"]
    assert row["review_status"] == "unreviewed"
    assert not row["revision_authorized"]
    diagnostic["labels"][0] = ["C", "H", "2"]
    assert not candidate_record("train_s", "m", "stress", layer, atom, diagnostic, {})["diagnostic_plain_c_disagreement"]
    diagnostic["revision_authorized"] = True
    with pytest.raises(ValueError):
        candidate_record("train_s", "m", "stress", layer, atom, diagnostic, {})


def test_denominators_include_images_without_candidates():
    layer, atom, diagnostic = fixture()
    row = candidate_record("train_s", "m", "stress", layer, atom, diagnostic, {})
    summary = summarize([row], [{"molecule_id": "m", "visible_text_objects": 5},
                                {"molecule_id": "n", "visible_text_objects": 7}])
    assert summary["images"] == 2 and summary["molecules"] == 2
    assert summary["visible_text_objects"] == 12
    assert summary["visible_neutral_carbons"] == 1
    assert summary["revisions_applied"] == 0


def test_empty_population_reports_zero_not_success():
    summary = summarize([], [{"molecule_id": "m", "visible_text_objects": 2}])
    assert summary["diagnostic_plain_c_disagreements"] == 0
    assert summary["review_status"] == "awaiting_source_raster_review"


def test_gallery_escapes_dataset_strings(tmp_path):
    layer, atom, diagnostic = fixture()
    row = candidate_record("train_s", "m", "stress", layer, atom, diagnostic, {})
    row["layer_text"] = "<script>alert(1)</script>"
    row["preview"] = {"source_png": "source.png", "crop": "crop.png", "context": "context.png"}
    write_gallery(tmp_path, [row])
    html = (tmp_path / "review.html").read_text()
    assert "<script>" not in html and "&lt;script&gt;" in html
