"""v0.3 changes pixels, not acceptance thresholds or chemical/transcription truth."""

import copy
import json
import subprocess
import xml.etree.ElementTree as ET

import pytest
from rdkit import Chem

from Data.sdf.text_source_coverage.matrix import build_matrix
from Data.sdf.text_source_coverage.outcomes import summarize
from Data.sdf.text_source_coverage.run import environment, source_hashes as base_sources
from Data.sdf.text_source_layout import render_molecule, TextSourceConfig
from Data.sdf.text_source_layout.paired import check_paired_content, compare, launch, load_baseline, run, verify_result
from Data.sdf.text_source_layout.quality import quality_record
from Data.sdf.text_source_layout.renderer import LayoutRejected
from Data.sdf.text_source_layout.spacing import measure_boxes, space_labels
from Data.sdf.text_source_layout.worker import render_sample
from Data.sdf.text_source_renderer.layers import json_bytes, png_bytes, raster, sha256
from Data.sdf.text_source_renderer.smoke import make_molecule
from Data.sdf.text_source_renderer.v2.smoke import CASES
from Data.sdf.text_source_renderer.v2.svg_records import attributable_svg


CONFIG = TextSourceConfig(width=384, height=384)


@pytest.mark.parametrize("case", CASES)
def test_original_synthetic_cases_preserve_chemistry_and_transcription(case):
    smiles, layout, overrides, expected, _ = CASES[case]
    mol = make_molecule(smiles, layout)
    before = Chem.MolToMolBlock(mol)
    if isinstance(expected, str):
        with pytest.raises(ValueError, match=expected):
            render_molecule(mol, config=CONFIG, label_overrides=overrides)
    else:
        bundle = render_molecule(mol, config=CONFIG, label_overrides=overrides)
        layers = json.loads(bundle[".layers.json"])
        assert [a["text"] for a in layers["atoms"]] == expected
        assert layers["schema_version"] == "chemvision-text-source-layers-0.3"
        assert layers["training_authorized"] is False
        assert len(layers["bonds"]) == mol.GetNumBonds()
        for bond in layers["bonds"]:
            source = mol.GetBondWithIdx(bond["bond_id"])
            assert (bond["begin_atom"], bond["end_atom"]) == (source.GetBeginAtomIdx(), source.GetEndAtomIdx())
            assert bond["direction"] == str(source.GetBondDir())
    assert Chem.MolToMolBlock(mol) == before


@pytest.fixture
def isotope():
    return render_molecule(make_molecule("[13CH3][NH3+]", "auto"), config=CONFIG)


def test_exact_png_and_all_hash_bindings(isotope):
    layers, records, quality = [json.loads(isotope[s]) for s in (".layers.json", ".text-records.json", ".quality.json")]
    assert png_bytes(raster(isotope[".svg"], 384, 384).convert("RGB")) == isotope[".png"]
    for record in (layers, records, quality):
        assert record["source_svg_sha256"] == sha256(isotope[".svg"])
        assert record["source_png_sha256"] == sha256(isotope[".png"])
    assert layers["quality_record_sha256"] == sha256(isotope[".quality.json"])
    assert layers["text_records_sha256"] == sha256(isotope[".text-records.json"])
    assert quality["layout_input_svg_sha256"] == sha256(isotope[".layout-input.svg"])
    assert quality["after_spacing"]["same_atom_glyph_overlap_pixels"] == 0
    assert quality["after_spacing"]["text_bond_overlap_pixels"] == 0
    assert quality["layout_policy"]["acceptance"] == "unchanged_v02_compile_layers"


def test_horizontal_changes_only_and_bounded(isotope):
    before, after = ET.fromstring(isotope[".layout-input.svg"]), ET.fromstring(isotope[".svg"])
    assert len(before) == len(after)
    quality = json.loads(isotope[".quality.json"])
    assert any(a["dx"] for a in quality["adjustments"])
    for old, new in zip(before, after):
        if old.tag.endswith("text"):
            assert old.text == new.text
            old.attrib.pop("x")
            new.attrib.pop("x")
        assert ET.tostring(old) == ET.tostring(new)
    assert all(abs(a["dx"]) <= CONFIG.font_size / 2 + 1 for a in quality["adjustments"])


def test_repeat_outputs_identical(isotope):
    assert render_molecule(make_molecule("[13CH3][NH3+]", "auto"), config=CONFIG) == isotope


def parsed(bundle):
    layers = json.loads(bundle[".layers.json"])
    anchors = [a["center"] for a in layers["atoms"]]
    return layers, anchors, attributable_svg(bundle[".svg"], layers, anchors)


def test_measurement_finds_ink_beyond_original_canvas(isotope):
    layers, anchors, _ = parsed(isotope)
    root = ET.fromstring(isotope[".svg"])
    glyph = next(node for node in root if node.tag.endswith("text"))
    glyph.set("x", "383")
    svg, glyphs, bonds, auxiliary = attributable_svg(ET.tostring(root), layers, anchors)
    boxes = measure_boxes(svg, glyphs, 384, 384)
    assert boxes[glyphs[0]["node_id"]][2] > 384
    quality = quality_record(svg, glyphs, bonds, auxiliary, width=384, height=384)
    assert glyphs[0]["node_id"] in quality["clipped_or_edge_touching_glyph_ids"]


@pytest.mark.parametrize("same_atom", [True, False])
def test_quality_distinguishes_same_and_cross_atom_glyphs(isotope, same_atom):
    layers, anchors, _ = parsed(isotope)
    root = ET.fromstring(isotope[".svg"])
    original = next(node for node in root if node.tag.endswith("text"))
    duplicate = copy.deepcopy(original)
    duplicate.set("class", "atom-0" if same_atom else "atom-1")
    root.append(duplicate)
    svg, glyphs, bonds, aux = attributable_svg(ET.tostring(root), layers, anchors)
    quality = quality_record(svg, glyphs, bonds, aux, width=384, height=384)
    field = "same_atom_glyph_overlap_pixels" if same_atom else "cross_atom_glyph_overlap_pixels"
    assert quality[field] > 0
    assert quality["acceptance_override"] is False


def test_spacing_refuses_unbounded_expansion():
    bundle = render_molecule(make_molecule("CCC", "linear"), config=CONFIG)
    layers, anchors, _ = parsed(bundle)
    root = ET.fromstring(bundle[".svg"])
    original = next(node for node in root if node.tag.endswith("text"))
    root.append(copy.deepcopy(original))
    svg, glyphs, _, _ = attributable_svg(ET.tostring(root), layers, anchors)
    with pytest.raises(ValueError, match="fixed expansion budget"):
        space_labels(svg, glyphs, width=384, height=384)


def test_original_strict_compiler_rejection_is_never_overridden(monkeypatch):
    from Data.sdf.text_source_layout import renderer

    def reject(*args, **kwargs):
        raise ValueError("Overlapping text/geometry ink is outside the reference contract")
    monkeypatch.setattr(renderer, "compile_layers", reject)
    with pytest.raises(LayoutRejected, match="Overlapping text/geometry") as error:
        render_molecule(make_molecule("CC", "auto"), config=CONFIG)
    assert error.value.quality["stage"] == "strict_v02_compiler"
    assert error.value.quality["training_authorized"] is False


@pytest.fixture
def protocol():
    matrix = build_matrix()
    matrix["samples"] = [matrix["samples"][0], matrix["samples"][-1]]
    return {"matrix": matrix}


def test_paired_report_explicitly_counts_regressions_and_controls(protocol):
    primary, control = protocol["matrix"]["samples"]
    old = [{"sample_key": primary["sample_key"], "status": "rendered"},
           {"sample_key": control["sample_key"], "status": "rejected", "reason": control["expected_rejection"]}]
    new = copy.deepcopy(old)
    new[0].update(status="rejected", reason="glyph_overlap")
    report = compare(protocol, old, new)
    assert report["primary"]["attempted"] == 1
    assert report["paired"]["transitions"] == {"rendered->rejected": 1}
    assert report["paired"]["regressed_sample_keys"] == [primary["sample_key"]]
    with pytest.raises(ValueError):
        compare(protocol, old[:-1], new)


def test_worker_and_verifier(protocol, tmp_path):
    sample = protocol["matrix"]["samples"][0]
    result = render_sample(protocol, sample["sample_key"], tmp_path)
    assert result["status"] == "rendered"
    assert verify_result(result, sample, tmp_path) == result
    bad = copy.deepcopy(result)
    bad["files"][sample["sample_key"] + ".quality.json"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_result(bad, sample, tmp_path)
    with pytest.raises(FileExistsError):
        render_sample(protocol, sample["sample_key"], tmp_path)


@pytest.mark.parametrize("mode", ["timeout", "crash", "invalid_json", "missing_bundle"])
def test_worker_failures_are_errors(protocol, tmp_path, monkeypatch, mode):
    sample = protocol["matrix"]["samples"][0]
    def fake(*args, **kwargs):
        if mode == "timeout":
            raise subprocess.TimeoutExpired(args[0], 30)
        if mode == "crash":
            return subprocess.CompletedProcess(args[0], -11, "", "crash")
        if mode == "invalid_json":
            return subprocess.CompletedProcess(args[0], 0, "invalid", "")
        return subprocess.CompletedProcess(args[0], 0, json.dumps({"sample_key": sample["sample_key"], "status": "rendered"}), "")
    monkeypatch.setattr(subprocess, "run", fake)
    assert launch(sample, tmp_path, "hash")["status"] == "error"


def test_small_paired_run_freezes_and_does_not_overwrite(protocol, tmp_path):
    from Data.sdf.text_source_coverage.worker import render_sample as render_old_sample

    baseline = tmp_path / "baseline"
    baseline.mkdir()
    protocol.update(sources=base_sources(), environment=environment())
    payload = json_bytes(protocol)
    (baseline / "protocol.json").write_bytes(payload)
    samples = protocol["matrix"]["samples"]
    (baseline / "artifacts").mkdir()
    results = [render_old_sample(protocol, s["sample_key"], baseline / "artifacts") for s in samples]
    report = summarize(protocol, results)
    report["protocol_sha256"] = sha256(payload)
    (baseline / "report.json").write_bytes(json_bytes(report))
    output = tmp_path / "paired"
    assert run(baseline, output, workers=2) == 0
    paired = json.loads((output / "report.json").read_bytes())
    assert paired["sources_environment_protocol_unchanged"]
    assert paired["paired"]["transitions"] == {"rendered->rendered": 1}
    assert paired["paired_content_check"] == {"checked": 1, "mismatches": []}
    # Even a self-consistent new artifact hash must not conceal changed text.
    name = samples[0]["sample_key"] + ".layers.json"
    changed = json.loads((output / "artifacts" / name).read_bytes())
    changed["atoms"][0]["text"] = "WRONG"
    changed_bytes = json_bytes(changed)
    (output / "artifacts" / name).write_bytes(changed_bytes)
    new_rows = copy.deepcopy(paired["results"])
    next(r for r in new_rows if r["sample_key"] == samples[0]["sample_key"])["files"][name] = sha256(changed_bytes)
    content = check_paired_content(baseline, output, results, new_rows)
    assert len(content["mismatches"]) == 1
    with pytest.raises(FileExistsError):
        run(baseline, output)
    report["primary"]["rendered"] = 999
    (baseline / "report.json").write_bytes(json_bytes(report))
    with pytest.raises(ValueError, match="statistics"):
        load_baseline(baseline)
