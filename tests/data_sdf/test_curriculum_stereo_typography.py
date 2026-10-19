"""CPU reproductions: preserve stereo paths, move ink instead, no corpus reads."""

from copy import deepcopy
import json
import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from rdkit import Chem

from Data.sdf.chemvector_curriculum_v0_1.catalog import merged_records
from Data.sdf.chemvector_curriculum_v0_1.rendering import profiles, render, make_molecule, sdf_bytes
from Data.sdf.chemvector_curriculum_v0_1 import stereo_typography as typography
from Data.sdf.text_source_renderer.layers import raster, png_bytes
from Data.sdf.text_source_renderer.svg_records import isolated_svg

BRIDGED = "O[C@@H]1C[C@]2(O)N[C@H]1C[C@@H](O)[C@@H]2O"
LARGE = "CC[C@H](CC[C@@H](C)[C@H]1CC[C@@H]2[C@@H]3C[C@H](Cl)[C@]45O[C@H]4[C@@H](O[C@H]4O[C@@H](CO)[C@H](O)[C@@H](O)[C@@H]4O)CC[C@]5(C)[C@H]3CC[C@@]21C)C(C)C"


def entry(smiles):
    return merged_records([dict(record_id="test", name="test", stage="full_molecule", concepts=[], smiles=smiles)])[0]


def profile(name):
    return next(p for p in profiles() if p.profile_id == name)


@pytest.mark.parametrize("smiles", [BRIDGED, LARGE])
@pytest.mark.parametrize("style", ["reference", "font_large", "serif"])
def test_old_rejected_chemistry_is_recovered_without_changing_graph(smiles, style):
    source, p = entry(smiles), profile(style)
    with pytest.raises(ValueError):
        render(source, p, typography="ink-aware-v1")
    files, meta = render(source, p, typography="stereo-aware-v1")
    assert files[".sdf"] == sdf_bytes(make_molecule(source["canonical_smiles"]))
    graph, layers = [json.loads(files[s]) for s in (".graph.json", ".layers.json")]
    assert graph["canonical_smiles"] == source["canonical_smiles"]
    assert meta["profile"]["font_size"] == p.font_size
    assert layers["renderer"]["renderer_id"] == "curriculum-same-svg-0.3"
    assert not meta["training_authorized"]
    assert files[".png"] == png_bytes(raster(files[".svg"], 768, 768).convert("RGB"))
    assert Chem.MolToSmiles(Chem.MolFromMolBlock(files[".sdf"].decode())) == source["canonical_smiles"]


def test_stereo_and_auxiliary_paths_anchors_and_font_are_unchanged(monkeypatch):
    snapshots = {}
    original = typography.adjust
    def observe(svg, glyphs, bonds, chemistry, anchors, drawing, **kwargs):
        snapshots.update(svg=svg, glyphs=deepcopy(glyphs), bonds=deepcopy(bonds),
                         chemistry=deepcopy(chemistry), anchors=deepcopy(anchors), drawing=deepcopy(drawing))
        return original(svg, glyphs, bonds, chemistry, anchors, drawing, **kwargs)
    monkeypatch.setattr(typography, "adjust", observe)
    files, _ = render(entry(LARGE), profile("serif"), typography="stereo-aware-v1")
    before, after = ET.fromstring(snapshots["svg"]), ET.fromstring(files[".svg"])
    layers, records = [json.loads(files[s]) for s in (".layers.json", ".text-records.json")]
    assert [a["center"] for a in layers["atoms"]] == [list(p) for p in snapshots["anchors"]]
    assert [b["drawing"] for b in layers["bonds"]] == snapshots["drawing"]
    stereo = {i for b, d in zip(snapshots["chemistry"]["bonds"], snapshots["drawing"])
              if d["direction"] != "NONE" for i in snapshots["bonds"][b["bond_id"]]}
    auxiliary = {i for i, n in enumerate(before) if n.tag.endswith("path") and not n.attrib.get("class")}
    assert stereo and auxiliary
    for i in stereo | auxiliary:
        assert before[i].attrib == after[i].attrib
    protocol = layers["renderer"]["typography"]
    assert protocol["protected_paths_unchanged"]
    assert protocol["label_moves"]
    for move in protocol["label_moves"]:
        assert 0 < math.hypot(move["dx_px"], move["dy_px"]) <= move["radius_px"] <= 8
    for old, new in zip(snapshots["glyphs"], records["glyphs"]):
        for field in ("atom_id", "text", "font_size", "node_index"):
            assert old[field] == new[field]
        assert new["svg_attributes"] == after[new["node_index"]].attrib
    text = np.asarray(raster(isolated_svg(files[".svg"], [g["node_index"] for g in records["glyphs"]]), 768, 768).getchannel("A")) > 0
    geometry = np.asarray(raster(isolated_svg(files[".svg"], [i for i, n in enumerate(after) if n.tag.endswith("path")]), 768, 768).getchannel("A")) > 0
    assert not (text & geometry).any()
    assert np.array_equal(text | geometry, np.asarray(raster(files[".svg"], 768, 768).convert("RGB")).min(axis=2) < 255)


@pytest.mark.parametrize("smiles", ["CCO", "CB(O)O", "C1CCOC1", "[13CH3]CO"])
@pytest.mark.parametrize("style", ["reference", "font_small", "serif"])
def test_collision_free_ordinary_examples_keep_pixels_and_chemistry(smiles, style):
    source, p = entry(smiles), profile(style)
    before, _ = render(source, p, typography="ink-aware-v1")
    after, _ = render(source, p, typography="stereo-aware-v1")
    for suffix in (".sdf", ".graph.json", ".png", ".text.png", ".bond.png", ".semantic.png"):
        assert before[suffix] == after[suffix]
    assert not json.loads(after[".layers.json"])["renderer"]["typography"]["label_moves"]


def test_legacy_remains_byte_reproducible_after_new_mode():
    e, p = entry("CCO"), profile("reference")
    before = render(e, p, typography="ink-aware-v1")
    render(e, p, typography="stereo-aware-v1")
    assert render(e, p, typography="ink-aware-v1") == before


def test_element_attribution_is_not_guessed_from_ambiguous_text():
    with pytest.raises(ValueError, match="uniquely"):
        typography._primary_indices([dict(text="N", node_index=1), dict(text="N", node_index=2)], "N")
    assert typography._primary_indices([dict(text=t, node_index=i) for i, t in enumerate("13CH3")], "C") == [2]


def test_no_canvas_clipping_or_font_shrinking_to_force_acceptance():
    # Vector attribution has not been relaxed to accept subpixel auxiliary legs.
    source = entry("CCCCCCCCCCCC[C@@H](O)[C@H]1CC[C@H]([C@H](O)CCC[C@@H]2OC(C)(C)O[C@@H]2CCCCC[C@@H](O)CC2=C[C@H](C)OC2=O)O1")
    with pytest.raises(ValueError, match="Degenerate auxiliary junction"):
        render(source, profile("font_large"), typography="stereo-aware-v1")


def test_protected_geometry_mutation_is_detected(monkeypatch):
    original = typography.adjust_v1
    def corrupt(svg, glyphs, bonds, chemistry, anchors, **kwargs):
        result, record = original(svg, glyphs, bonds, chemistry, anchors, **kwargs)
        if chemistry["bonds"]:  # After ordinary-bond clipping, mutate a stereo hatch.
            root = ET.fromstring(result)
            nodes = [n for n in root if "bond-10 " in n.attrib.get("class", "")]
            assert nodes
            nodes[0].set("d", "M 1,1 L 2,2")
            result = ET.tostring(root)
        return result, record
    monkeypatch.setattr(typography, "adjust_v1", corrupt)
    with pytest.raises(ValueError, match="Protected stereo"):
        render(entry(BRIDGED), profile("reference"), typography="stereo-aware-v1")
