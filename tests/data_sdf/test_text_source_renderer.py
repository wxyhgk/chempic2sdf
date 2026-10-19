"""Synthetic-only regressions: transcription follows rendered text, not valence."""

import io
import json
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import pytest
from rdkit import Chem

from Data.sdf.text_source_renderer import TextSourceConfig, render_molecule
from Data.sdf.text_source_renderer.layers import compile_layers, png_bytes, raster, sha256
from Data.sdf.text_source_renderer.smoke import CASES, make_molecule, run
from Data.sdf.text_source_renderer.svg_records import attributable_svg, isolated_svg


CONFIG = TextSourceConfig(width=384, height=384)


@pytest.fixture
def linear():
    return render_molecule(make_molecule("CCC", "linear"), config=CONFIG)


@pytest.mark.parametrize("case", CASES)
def test_synthetic_cases(case):
    smiles, layout, overrides, expected = CASES[case]
    mol = make_molecule(smiles, layout)
    before = Chem.MolToMolBlock(mol)
    if isinstance(expected, str):
        with pytest.raises(ValueError, match=expected):
            render_molecule(mol, config=CONFIG, label_overrides=overrides)
    else:
        result = render_molecule(mol, config=CONFIG, label_overrides=overrides)
        layers = json.loads(result[".layers.json"])
        assert [a["text"] for a in layers["atoms"]] == expected
        assert layers["training_authorized"] is False
        assert layers["renderer"]["legacy_pixel_equivalent"] is False
        assert len(layers["renderer"]["font"]["sha256"]) == 64
    assert Chem.MolToMolBlock(mol) == before


def test_linear_c_is_not_inferred_ch2(linear):
    atom = json.loads(linear[".layers.json"])["atoms"][1]
    assert atom["text"] == "C"
    assert atom["implicit_h_count"] == 2
    assert atom["visible_text"] and atom["bbox"]


def test_final_svg_replays_exact_png_and_masks(linear):
    layers = json.loads(linear[".layers.json"])
    records = json.loads(linear[".text-records.json"])
    assert png_bytes(raster(linear[".svg"], 384, 384).convert("RGB")) == linear[".png"]
    assert records["source_svg_sha256"] == layers["source_svg_sha256"] == sha256(linear[".svg"])
    assert records["source_png_sha256"] == layers["source_png_sha256"] == sha256(linear[".png"])
    assert layers["text_records_sha256"] == sha256(linear[".text-records.json"])
    assert records["same_source_raster"] is True
    for glyph in records["glyphs"]:
        isolated = isolated_svg(linear[".svg"], [glyph["node_index"]])
        assert list(raster(isolated, 384, 384).getchannel("A").getbbox()) == glyph["bbox"]
    with Image.open(io.BytesIO(linear[".png"])) as image:
        foreground = np.asarray(image).min(axis=2) < 255
    with Image.open(io.BytesIO(linear[".semantic.png"])) as image:
        semantic = np.asarray(image)
    assert np.array_equal(foreground, semantic > 0)
    assert set(np.unique(semantic)) == {0, 1, 2}


def test_repeat_render_is_identical(linear):
    assert render_molecule(make_molecule("CCC", "linear"), config=CONFIG) == linear


@pytest.mark.parametrize("mutation,error", [
    ("unowned", "Unattributed"), ("duplicate", "Duplicate"),
    ("unknown_atom", "unknown atom"), ("resource", "resources"),
    ("nested", "Nested"), ("transform", "style"),
    ("unsupported_glyph", "Unsupported visible glyph"),
])
def test_parser_rejects_unattributable_nodes(linear, mutation, error):
    root = ET.fromstring(linear[".svg"])
    glyph = next(node for node in root if node.tag.endswith("text"))
    if mutation == "unowned":
        ET.SubElement(root, "path", {"d": "M 1,1 L 2,2"})
    elif mutation == "duplicate":
        glyph.set("class", "atom-1 atom-1")
    elif mutation == "unknown_atom":
        glyph.set("class", "atom-99")
    elif mutation == "resource":
        glyph.set("href", "external.svg")
    elif mutation == "nested":
        ET.SubElement(glyph, "tspan")
    elif mutation == "transform":
        glyph.set("transform", "translate(1,1)")
    else:
        glyph.text = "?"
    with pytest.raises(ValueError, match=error):
        attributable_svg(ET.tostring(root), 3, 2)


@pytest.mark.parametrize("mutation,error", [
    ("no_ink", "no raster pixels"), ("clipped", "canvas edge"),
    ("overlap", "Overlapping glyph"), ("bond_overlap", "Overlapping text/bond"),
])
def test_raster_rejects_false_visible_evidence(linear, mutation, error):
    import copy

    root = ET.fromstring(linear[".svg"])
    glyph = next(node for node in root if node.tag.endswith("text"))
    if mutation == "no_ink":
        glyph.set("x", "-100")
    elif mutation == "clipped":
        glyph.set("x", "-2")
    elif mutation == "overlap":
        root.append(copy.deepcopy(glyph))
    else:
        ET.SubElement(root, "path", {"class": "bond-0 atom-0 atom-1",
                                    "d": "M 0,195 L 384,195", "stroke": "black",
                                    "stroke-width": "10"})
    svg, glyphs, bonds = attributable_svg(ET.tostring(root), 3, 2)
    layers = json.loads(linear[".layers.json"])
    with pytest.raises(ValueError, match=error):
        compile_layers(svg, glyphs, bonds, layers, [a["center"] for a in layers["atoms"]],
                       width=384, height=384, sample_key="test", provenance={})


def test_invalid_input_and_no_layout():
    with pytest.raises(ValueError, match="2D"):
        render_molecule(Chem.MolFromSmiles("CC"))
    with pytest.raises(ValueError, match="identifier"):
        render_molecule(make_molecule("CC", "auto"), sample_key="../unsafe")
    with pytest.raises(ValueError, match="Canvas"):
        render_molecule(make_molecule("CC", "auto"), config=TextSourceConfig(width=0))


def test_nonfinite_layout_rejected_before_native_drawing():
    mol = make_molecule("CC", "auto")
    mol.GetConformer().SetAtomPosition(0, (float("nan"), 0, 0))
    with pytest.raises(ValueError, match="finite"):
        render_molecule(mol)


@pytest.mark.parametrize("mode", ["crash", "timeout"])
def test_smoke_failure_never_creates_success_receipt(tmp_path, monkeypatch, mode):
    import subprocess
    from Data.sdf.text_source_renderer import smoke

    monkeypatch.setattr(smoke, "font_provenance", lambda: {"test": True})
    def fail(*args, **kwargs):
        if mode == "timeout":
            raise subprocess.TimeoutExpired(args[0], 30)
        return subprocess.CompletedProcess(args[0], -11, "", "synthetic crash")
    monkeypatch.setattr(smoke.subprocess, "run", fail)
    output = tmp_path / mode
    assert run(output) == 1
    receipt = json.loads((output / "receipt.json").read_bytes())
    assert receipt["status"] == "failed"
    assert all(r["status"] == "failed" for r in receipt["results"])


def test_smoke_subprocess_receipt_and_no_overwrite(tmp_path):
    output = tmp_path / "smoke"
    assert run(output) == 0
    receipt = json.loads((output / "receipt.json").read_bytes())
    assert receipt["sources_unchanged"]
    assert receipt["protocol_sha256"] == sha256((output / "protocol.json").read_bytes())
    assert len(receipt["results"]) == 10
    assert sum(r["status"] == "rendered" for r in receipt["results"]) == 8
    for result in receipt["results"]:
        for name, digest in result.get("files", {}).items():
            assert sha256((output / name).read_bytes()) == digest
    with pytest.raises(FileExistsError):
        run(output)
