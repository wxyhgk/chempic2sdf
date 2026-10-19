"""Same-source typography regressions, including formerly rejected structures."""

from copy import deepcopy
import io
import json
import multiprocessing as mp
import time
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import pytest

from Data.sdf.chemvector_curriculum_v0_1.bounded import bounded_map
from Data.sdf.chemvector_curriculum_v0_1.catalog import merged_records
from Data.sdf.chemvector_curriculum_v0_1.rendering import profiles, render
from Data.sdf.chemvector_curriculum_v0_1.typography import rectangle_interval
from Data.sdf.text_source_renderer.layers import png_bytes, raster
from Data.sdf.text_source_renderer.svg_records import parse_style


def entry(smiles):
    return next(r for r in merged_records() if r["canonical_smiles"] == smiles)


@pytest.mark.parametrize("smiles", ["COC(C)=O", "C1CCOC1", "CB(O)O", "CCO", "[13CH3]CO"])
@pytest.mark.parametrize("profile_id", ["reference", "font_small", "serif"])
def test_rejected_cases_have_strict_same_source_outputs(smiles, profile_id):
    profile = next(p for p in profiles() if p.profile_id == profile_id)
    source = entry(smiles)
    before = deepcopy(source)
    files, meta = render(source, profile, typography="ink-aware-v1")
    assert source == before
    assert files[".png"] == png_bytes(raster(files[".svg"], 768, 768).convert("RGB"))
    graph, layers, records = [json.loads(files[s]) for s in (".graph.json", ".layers.json", ".text-records.json")]
    assert graph["canonical_smiles"] == smiles
    assert len(graph["atoms"]) == len(layers["atoms"])
    assert len(graph["bonds"]) == len(layers["bonds"])
    assert layers["renderer"]["renderer_id"] == "curriculum-same-svg-0.2"
    assert layers["renderer"]["typography"]["mode"] == "ink-aware-v1"
    root = ET.fromstring(files[".svg"])
    for glyph in records["glyphs"]:
        assert root[glyph["node_index"]].attrib == glyph["svg_attributes"]
        assert parse_style(glyph["svg_attributes"]["style"])["font-family"] == profile.font_family
    with Image.open(io.BytesIO(files[".png"])) as image:
        foreground = np.asarray(image).min(axis=2) < 255
    with Image.open(io.BytesIO(files[".semantic.png"])) as image:
        semantic = np.asarray(image)
    assert np.array_equal(foreground, semantic > 0)
    assert not ((semantic & 2 > 0) & (semantic & 5 > 0)).any()
    assert meta["profile"]["font_size"] == profile.font_size
    assert not meta["training_authorized"]


def test_new_mode_does_not_change_legacy_or_hide_its_failures():
    profile = next(p for p in profiles() if p.profile_id == "font_small")
    with pytest.raises(ValueError, match="Overlapping glyph"):
        render(entry("CCO"), profile)
    with pytest.raises(ValueError, match="Overlapping glyph"):
        render(entry("CCO"), profile, typography="legacy")
    first = render(entry("CCO"), profile, typography="ink-aware-v1")
    assert first == render(entry("CCO"), profile, typography="ink-aware-v1")


def test_chemistry_anchors_baselines_and_glyph_sizes_stay_fixed():
    source, profile = entry("CCO"), profiles()[0]
    legacy, _ = render(source, profile)
    revised, _ = render(source, profile, typography="ink-aware-v1")
    assert legacy[".graph.json"] == revised[".graph.json"]
    assert legacy[".sdf"] == revised[".sdf"]
    old, new = [json.loads(f[".layers.json"]) for f in (legacy, revised)]
    assert [a["center"] for a in old["atoms"]] == [a["center"] for a in new["atoms"]]
    left, right = [json.loads(f[".text-records.json"])["glyphs"] for f in (legacy, revised)]
    for a, b in zip(left, right):
        for field in ("text", "atom_id", "node_index", "y", "font_size"):
            assert a[field] == b[field]


@pytest.mark.parametrize("begin,end,box,expected", [
    ((0, 0), (10, 0), (2, -1, 4, 1), (.2, .4)),
    ((0, 0), (0, 10), (-1, 2, 1, 4), (.2, .4)),
    ((0, 0), (10, 0), (2, 1, 4, 2), None),
    ((0, 0), (10, 10), (2, 2, 4, 4), (.2, .4)),
])
def test_segment_rectangle_intersection(begin, end, box, expected):
    assert rectangle_interval(begin, end, box) == expected


def delayed(job):
    delay, value = job
    time.sleep(delay)
    return value


def raises(_):
    raise ValueError("synthetic worker invariant")


def test_bounded_map_preserves_order():
    assert list(bounded_map(delayed, [(.15, 0), (0, 1), (0, 2)], workers=2, timeout_seconds=10)) == [0, 1, 2]


def test_worker_timeout_terminates_owned_children():
    previous = {p.pid for p in mp.active_children()}
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="no automatic retry"):
        list(bounded_map(delayed, [(30, 0)], workers=1, timeout_seconds=.5))
    assert time.monotonic() - started < 5
    assert {p.pid for p in mp.active_children()} <= previous


def test_worker_error_is_not_a_quality_rejection():
    with pytest.raises(RuntimeError, match="synthetic worker invariant"):
        list(bounded_map(raises, [None], workers=1, timeout_seconds=10))
