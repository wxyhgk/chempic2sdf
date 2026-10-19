"""Synthetic coverage and adversarial junction witnesses; no corpus reads."""

import copy
import io
import json
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import pytest
from rdkit import Chem

from Data.sdf.text_source_renderer import render_molecule as render_v1
from Data.sdf.text_source_renderer.layers import png_bytes, raster, sha256
from Data.sdf.text_source_renderer.smoke import make_molecule
from Data.sdf.text_source_renderer.svg_records import isolated_svg
from Data.sdf.text_source_renderer.v2 import render_molecule, TextSourceConfig
from Data.sdf.text_source_renderer.v2.layers import compile_layers
from Data.sdf.text_source_renderer.v2.smoke import CASES, run
from Data.sdf.text_source_renderer.v2.svg_records import attributable_svg


CONFIG = TextSourceConfig(width=384, height=384)


def render_case(case):
    smiles, layout, overrides, _, _ = CASES[case]
    return render_molecule(make_molecule(smiles, layout), config=CONFIG, label_overrides=overrides)


@pytest.fixture
def bent():
    return render_case("bent_carbon")


@pytest.mark.parametrize("case", CASES)
def test_cases_preserve_chemistry_and_transcribe_only_visible_text(case):
    smiles, layout, overrides, expected, aux_count = CASES[case]
    mol = make_molecule(smiles, layout)
    before = Chem.MolToMolBlock(mol)
    if isinstance(expected, str):
        with pytest.raises(ValueError, match=expected):
            render_molecule(mol, config=CONFIG, label_overrides=overrides)
    else:
        result = render_molecule(mol, config=CONFIG, label_overrides=overrides)
        layers = json.loads(result[".layers.json"])
        assert [a["text"] for a in layers["atoms"]] == expected
        assert len(layers["auxiliary_geometry"]) == aux_count
        assert len(layers["atoms"]) == mol.GetNumAtoms()
        assert len(layers["bonds"]) == mol.GetNumBonds()
        assert not layers["training_authorized"]
        assert layers["schema_version"].endswith("0.2")
        for bond in layers["bonds"]:
            source = mol.GetBondWithIdx(bond["bond_id"])
            assert (bond["begin_atom"], bond["end_atom"]) == (source.GetBeginAtomIdx(), source.GetEndAtomIdx())
            assert bond["stereo_atoms"] == list(source.GetStereoAtoms())
    assert Chem.MolToMolBlock(mol) == before


@pytest.mark.parametrize("case,direction", [("wedge", "BEGINWEDGE"), ("hashed_wedge", "BEGINDASH")])
def test_stereo_draw_direction_does_not_overwrite_chemical_direction(case, direction):
    layers = json.loads(render_case(case)[".layers.json"])
    bond = layers["bonds"][0]
    assert bond["direction"] == "NONE"
    assert (bond["begin_atom"], bond["end_atom"]) == (0, 1)
    assert bond["drawing"]["direction"] == direction
    assert (bond["drawing"]["begin_atom"], bond["drawing"]["end_atom"]) == (1, 0)
    assert len(bond["svg_node_indices"]) == 1 if case == "wedge" else len(bond["svg_node_indices"]) > 1


def test_cis_trans_and_aromatic_chemistry_are_retained():
    trans = json.loads(render_case("trans_double")[".layers.json"])
    cis = json.loads(render_case("cis_double")[".layers.json"])
    assert trans["bonds"][1]["stereo"] == "STEREOE"
    assert cis["bonds"][1]["stereo"] == "STEREOZ"
    aromatic = json.loads(render_case("benzene")[".layers.json"])
    assert all(b["chemical_type"] == "AROMATIC" for b in aromatic["bonds"])
    assert {b["drawing"]["bond_type"] for b in aromatic["bonds"]} == {"SINGLE", "DOUBLE"}


def test_masks_keep_auxiliary_separate_and_replay_exact_png(bent):
    layers = json.loads(bent[".layers.json"])
    assert png_bytes(raster(bent[".svg"], 384, 384).convert("RGB")) == bent[".png"]
    arrays = {}
    for suffix in (".semantic.png", ".text.png", ".bond.png", ".auxiliary.png", ".png"):
        with Image.open(io.BytesIO(bent[suffix])) as image:
            arrays[suffix] = np.asarray(image)
    semantic = arrays[".semantic.png"]
    assert np.array_equal(semantic & 1 > 0, arrays[".bond.png"] > 0)
    assert np.array_equal(semantic & 2 > 0, arrays[".text.png"] > 0)
    assert np.array_equal(semantic & 4 > 0, arrays[".auxiliary.png"] > 0)
    assert np.array_equal(semantic > 0, arrays[".png"].min(axis=2) < 255)
    assert (semantic == 5).any()
    assert layers["bond_auxiliary_overlap_pixels"] == int((semantic == 5).sum())
    assert set(layers["auxiliary_geometry"][0]["incident_bond_ids"]) == {0, 1}
    aux = layers["auxiliary_geometry"][0]
    assert aux["anchor_atom_id"] == 1
    mask = raster(isolated_svg(bent[".svg"], [aux["node_index"]]), 384, 384).getchannel("A")
    assert list(mask.getbbox()) == aux["bbox"]
    assert sha256(bent[".svg"]) == layers["source_svg_sha256"]
    records = json.loads(bent[".text-records.json"])
    assert sha256(bent[".text-records.json"]) == layers["text_records_sha256"]
    assert records["source_png_sha256"] == sha256(bent[".png"])


def test_final_node_indices_and_attributes_survive_insertion():
    result = render_case("two_amine_ends")
    root = ET.fromstring(result[".svg"])
    layers = json.loads(result[".layers.json"])
    records = json.loads(result[".text-records.json"])
    for record in [*records["glyphs"], *layers["auxiliary_geometry"]]:
        node = root[record["node_index"]]
        assert node.attrib == record["svg_attributes"]
        assert node.attrib["id"] == record["node_id"]
    for bond in layers["bonds"]:
        for index in bond["svg_node_indices"]:
            assert f"bond-{bond['bond_id']}" in root[index].attrib["class"].split()
    replay, _, _, _ = attributable_svg(result[".svg"], layers, [a["center"] for a in layers["atoms"]])
    assert replay == result[".svg"]


@pytest.mark.parametrize("mutation,error", [
    ("unknown_path", "three-point"), ("far_pivot", "geometric witness"),
    ("long_leg", "geometric witness"), ("wrong_width", "geometric witness"),
    ("filled", "style"), ("transform", "attributes"), ("resource", "attributes"),
    ("duplicate", "Duplicate auxiliary"), ("wrong_graph", "chemical graph"),
    ("wrong_bond_owner", "chemical graph"), ("displaced_anchor", "geometric witness"),
    ("before_background", "must follow bond"), ("curve", "three-point"),
])
def test_invalid_junctions_fail_closed(bent, mutation, error):
    layers = json.loads(bent[".layers.json"])
    root = ET.fromstring(bent[".svg"])
    aux = root[layers["auxiliary_geometry"][0]["node_index"]]
    anchors = [a["center"] for a in layers["atoms"]]
    if mutation in {"unknown_path", "curve"}:
        aux.set("d", "M 10,10 L 20,20" if mutation == "unknown_path" else "M 1,1 C 2,2 3,3 4,4")
    elif mutation == "far_pivot":
        aux.set("d", "M 183.7,149.9 L 80,80 L 199.8,149.9")
    elif mutation == "long_leg":
        aux.set("d", "M 100,200 L 191.7,145.2 L 199.8,149.9")
    elif mutation == "wrong_width":
        aux.set("style", aux.attrib["style"].replace("2.0px", "3.0px"))
    elif mutation == "filled":
        aux.set("style", aux.attrib["style"].replace("fill:none", "fill:#000000"))
    elif mutation == "transform":
        aux.set("transform", "translate(1,1)")
    elif mutation == "resource":
        aux.set("href", "external.svg")
    elif mutation == "duplicate":
        duplicate = copy.deepcopy(aux)
        duplicate.attrib.pop("id")
        root.append(duplicate)
    elif mutation == "wrong_graph":
        layers["bonds"][0]["end_atom"] = 2
    elif mutation == "wrong_bond_owner":
        root[layers["bonds"][0]["svg_node_indices"][0]].set("class", "bond-0 atom-0 atom-2")
    elif mutation == "displaced_anchor":
        anchors[1] = [20, 20]
    else:
        root.remove(aux)
        root.insert(0, aux)
    with pytest.raises(ValueError, match=error):
        attributable_svg(ET.tostring(root), layers, anchors)


@pytest.mark.parametrize("size,rotation,width", [(256, 0, 1.0), (512, 30, 2.0), (768, 90, 3.0)])
def test_junctions_across_canvas_rotation_and_stroke(size, rotation, width):
    config = TextSourceConfig(width=size, height=size, rotation_degrees=rotation, bond_line_width=width)
    result = render_molecule(make_molecule("C1CCCCC1", "auto"), config=config)
    assert len(json.loads(result[".layers.json"])["auxiliary_geometry"]) == 6


def test_determinism_and_v1_boundary(bent):
    assert render_case("bent_carbon") == bent
    with pytest.raises(ValueError, match="Unattributed SVG primitive"):
        render_v1(make_molecule("CCC", "auto"), config=CONFIG)


@pytest.mark.parametrize("case", [name for name, record in CASES.items() if record[4] == 0])
def test_original_supported_fixtures_keep_same_pixels(case):
    smiles, layout, overrides, _, _ = CASES[case]
    old = render_v1(make_molecule(smiles, layout), config=CONFIG, label_overrides=overrides)
    new = render_case(case)
    assert new[".png"] == old[".png"]
    assert new[".text.png"] == old[".text.png"]
    assert new[".bond.png"] == old[".bond.png"]


@pytest.mark.parametrize("mutation,error", [
    ("no_ink", "no raster pixels"), ("clipped", "canvas edge"),
    ("overlap", "Overlapping glyph"), ("bond_overlap", "Overlapping text/geometry"),
])
def test_v2_retains_strict_visible_text_checks(mutation, error):
    result = render_case("linear_carbon")
    root = ET.fromstring(result[".svg"])
    layers = json.loads(result[".layers.json"])
    glyph = next(node for node in root if node.tag.endswith("text"))
    if mutation == "no_ink":
        glyph.set("x", "-100")
    elif mutation == "clipped":
        glyph.set("x", "-2")
    elif mutation == "overlap":
        root.append(copy.deepcopy(glyph))
    else:
        ET.SubElement(root, "path", {"class": "bond-0 atom-0 atom-1", "d": "M 0,195 L 384,195",
                                    "stroke": "black", "stroke-width": "10"})
    anchors = [a["center"] for a in layers["atoms"]]
    svg, glyphs, bonds, aux = attributable_svg(ET.tostring(root), layers, anchors)
    with pytest.raises(ValueError, match=error):
        compile_layers(svg, glyphs, bonds, aux, layers, anchors, [b["drawing"] for b in layers["bonds"]],
                       width=384, height=384, sample_key="test", provenance={})


def test_actual_connectivity_change_is_not_mistaken_for_stereo_reversal(monkeypatch):
    from rdkit.Chem.Draw import rdMolDraw2D
    from Data.sdf.text_source_renderer.v2.renderer import prepare_molecule

    mol = make_molecule("CCC", "auto")
    def wrong_graph(source, **kwargs):
        edited = Chem.RWMol(source)
        edited.RemoveBond(0, 1)
        edited.AddBond(0, 2, Chem.BondType.SINGLE)
        return edited.GetMol()
    monkeypatch.setattr(rdMolDraw2D, "PrepareMolForDrawing", wrong_graph)
    with pytest.raises(ValueError, match="connectivity"):
        prepare_molecule(mol)


def test_process_isolated_smoke(tmp_path):
    output = tmp_path / "smoke"
    assert run(output) == 0
    receipt = json.loads((output / "receipt.json").read_bytes())
    assert receipt["sources_unchanged"]
    assert receipt["protocol_sha256"] == sha256((output / "protocol.json").read_bytes())
    assert sum(r["status"] == "rendered" for r in receipt["results"]) == 16
    assert sum(r["status"] == "expected_rejection" for r in receipt["results"]) == 1
    for result in receipt["results"]:
        for name, digest in result.get("files", {}).items():
            assert sha256((output / name).read_bytes()) == digest
    with pytest.raises(FileExistsError):
        run(output)


@pytest.mark.parametrize("mode", ["crash", "timeout", "missing_bundle"])
def test_worker_failure_cannot_pass(tmp_path, monkeypatch, mode):
    import subprocess
    from Data.sdf.text_source_renderer.v2 import smoke

    monkeypatch.setattr(smoke, "CASES", {"bent_carbon": CASES["bent_carbon"]})
    monkeypatch.setattr(smoke, "font_provenance", lambda: {"test": True})
    def fail(*args, **kwargs):
        if mode == "timeout":
            raise subprocess.TimeoutExpired(args[0], 30)
        if mode == "missing_bundle":
            return subprocess.CompletedProcess(args[0], 0, json.dumps({"case": "bent_carbon", "status": "rendered"}), "")
        return subprocess.CompletedProcess(args[0], -11, "", "synthetic crash")
    monkeypatch.setattr(smoke.subprocess, "run", fail)
    assert run(tmp_path / mode) == 1
    receipt = json.loads((tmp_path / mode / "receipt.json").read_bytes())
    assert receipt["status"] == "failed"
