import io
import json

from PIL import Image
import pytest

from Data.sdf import audit_train_visible_text as audit


def fake_inputs(monkeypatch, *, diagnostic_fails=False):
    from Data.sdf import build_chemvision_layers

    buffer = io.BytesIO()
    Image.new("RGB", (768, 768), "white").save(buffer, format="PNG")
    layers = {"width": 768, "height": 768, "render_parameters": {}, "atoms": [
        {"atom_id": 0, "element": "C", "visible_text": True, "text": "CH2",
         "bbox": [10, 10, 20, 20], "center": [15, 15]}]}
    chemical = {"atoms": [{"id": 0, "element": "C", "implicit_h_count": 2}]}
    sample = {"key": "train_000000_original", "metadata": {"molecule_id": "m", "variant": "original"},
              "layers": layers, "chemmol": chemical, "render": {"render_parameters": {}},
              "payloads": {"png": buffer.getvalue(), "layers": json.dumps(layers).encode(),
                           "chemmol": json.dumps(chemical).encode(), "render": b"{}"}}
    selection = {"selection": [sample["key"]], "sample_metadata": {sample["key"]: sample["metadata"]}}
    monkeypatch.setattr(audit, "freeze_selection", lambda *a, **k: selection)
    monkeypatch.setattr(audit, "iter_selected_samples", lambda *a: iter([sample]))
    monkeypatch.setattr(build_chemvision_layers, "restore_source_geometry", lambda *a: None)
    def diagnostic(*args):
        if diagnostic_fails:
            raise ValueError("synthetic renderer failure")
        return {"labels": {0: ["C"]}, "evidence_kind": "nofreetype_diagnostic", "revision_authorized": False}
    monkeypatch.setattr(audit, "no_freetype_text_diagnostic", diagnostic)
    return sample


def test_runner_freezes_original_bytes_and_does_not_revise(tmp_path, monkeypatch):
    sample = fake_inputs(monkeypatch)
    out = tmp_path / "review"
    result = audit.run(out, molecule_count=1)
    assert result["diagnostic_plain_c_disagreements"] == 1
    report = json.loads((out / "report.json").read_text())
    assert report["revisions_applied"] == 0 and not report["training_performed"]
    copied = out / "source-images" / f"{sample['key']}.png"
    assert copied.read_bytes() == sample["payloads"]["png"]
    assert sample["layers"]["atoms"][0]["text"] == "CH2"
    with pytest.raises(FileExistsError):
        audit.run(out, molecule_count=1)


def test_renderer_failure_remains_in_denominator(tmp_path, monkeypatch):
    fake_inputs(monkeypatch, diagnostic_fails=True)
    result = audit.run(tmp_path / "review", molecule_count=1)
    assert result["visible_neutral_carbons"] == 1
    assert result["unresolved_diagnostic_objects"] == 1
    assert result["diagnostic_plain_c_disagreements"] == 0
