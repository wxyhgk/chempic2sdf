import json

from PIL import Image
import pytest

from Data.sdf.train_visible_text_contact_sheet import build
from Data.sdf.train_visible_text_report import digest


def evidence_bundle(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    Image.new("RGB", (30, 30), "white").save(source / "source.png")
    Image.new("RGB", (12, 12), "white").save(source / "crop.png")
    rows = [
        {"object_key": key, "molecule_id": mol, "layer_text": "CH2",
         "diagnostic_plain_c_disagreement": flagged,
         "source_hashes": {"png": digest((source / "source.png").read_bytes())},
         "preview": {"source_png": "source.png", "crop": "crop.png", "context": "crop.png"}}
        for key, mol, flagged in [("train_b", "m", True), ("train_a", "m", True),
                                  ("train_c", "n", True), ("train_d", "z", False)]
    ]
    (source / "records.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    artifacts = {name: digest((source / name).read_bytes()) for name in ("records.jsonl", "crop.png")}
    (source / "report.json").write_text(json.dumps({"artifacts": artifacts}))
    return source


def test_sheet_selects_one_per_flagged_molecule_without_review_decisions(tmp_path):
    source = evidence_bundle(tmp_path)
    output = tmp_path / "sheet"
    result = build(source, output)
    selection = json.loads((output / "selection.json").read_text())
    assert result["objects"] == 2
    assert [r["object_key"] for r in selection["records"]] == ["train_a", "train_c"]
    assert not selection["revision_authorized"]
    assert "observations" not in selection
    with pytest.raises(FileExistsError):
        build(source, output)


@pytest.mark.parametrize("filename", ["records.jsonl", "source.png", "crop.png"])
def test_sheet_rejects_altered_evidence(tmp_path, filename):
    source = evidence_bundle(tmp_path)
    (source / filename).write_bytes(b"tampered")
    with pytest.raises(ValueError):
        build(source, tmp_path / "sheet")
