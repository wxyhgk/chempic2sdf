from copy import deepcopy

import numpy as np
from PIL import Image
import pytest
import torch

from chemvector.components.ocr.crop_proposals import predicted_box, propose_vertical, rasterize
from chemvector.ctc_ocr import bbox_crop
from chemvector.evaluation.audits.ocr.crop_coverage import score_crop, unique_anchor_matches, vertical_layout


@pytest.mark.parametrize('box', [(10.5, 11.5, 20.5, 31.5), (-2., -3., 5., 15.),
                                 (97.6, 97.4, 102.6, 111.4), (11., 10., 45., 20.)])
def test_original_raster_exactly_matches_runtime(box):
    pixels = np.random.default_rng(9).integers(0, 256, (100, 100, 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    proposal = propose_vertical(box, image.size)
    assert np.array_equal(np.asarray(rasterize(image, proposal['original'])),
                          np.asarray(bbox_crop(image, box, context=4.)))


def test_fp32_box_matches_tensor_arithmetic():
    center, extent = [.18621784448623657, .8866782188415527], [.014918106608092785, .019785095006227493]
    x, y = (torch.tensor(center, dtype=torch.float32) * 768).tolist()
    w, h = (torch.tensor(extent, dtype=torch.float32) * 768).tolist()
    assert predicted_box(center, extent) == (x - w, y - h, x + w, y + h)


@pytest.mark.parametrize('center,extent', [([float('nan'), .5], [.1, .1]),
    ([.5, .5], [0., .1]), ([.5, .5], [-.1, .1]), ([2., .5], [.1, .1])])
def test_invalid_predictions_rejected(center, extent):
    with pytest.raises(ValueError):
        predicted_box(center, extent)


def test_extension_bounded_and_original_retained():
    box = [40, 40, 55, 80]
    saved = deepcopy(box)
    p = propose_vertical(box, (100, 100))
    assert box == saved
    assert p['original'] == [36, 36, 59, 84]
    assert p['proposed'] == [36, 4, 59, 100]
    assert p['requested_extension_per_side_px'] == 32
    assert p['proposed_raster']['scale_y'] < p['original_raster']['scale_y']


def test_wide_box_unchanged_and_no_added_black_padding():
    p = propose_vertical([30, 40, 70, 60], (100, 100))
    assert not p['changed']
    assert p['original'] == p['proposed']
    p = propose_vertical([40, -1, 55, 24], (100, 100))
    assert p['proposed'][1] == p['original'][1] == -5


def test_unique_matching_retains_ambiguous_unmatched_and_invisible():
    atoms = [dict(atom_id=0, center=[10, 10]), dict(atom_id=1, center=[12, 10]),
             dict(atom_id=2, center=[50, 50]), dict(atom_id=3, center=[80, 80])]
    preds = [dict(candidate=i, anchor_px=xy) for i, xy in enumerate(
        ([10, 10], [50, 50], [50, 51], [80, 80], [95, 95], None))]
    result = unique_anchor_matches(preds, atoms)
    assert [r['status'] for r in result.values()] == [
        'ambiguous', 'ambiguous', 'ambiguous', 'matched', 'unmatched', 'unmatched']
    assert result[3]['atom_id'] == 3
    with pytest.raises(ValueError):
        unique_anchor_matches(preds + [preds[0]], atoms)


def test_coverage_neighbors_bonds_and_empty_are_separate():
    glyphs = [dict(node_id='n', atom_id=1, text='N', bbox=[10, 10, 20, 20]),
              dict(node_id='h', atom_id=1, text='H', bbox=[10, 25, 20, 35]),
              dict(node_id='o', atom_id=2, text='O', bbox=[10, 38, 20, 48])]
    text, bond = np.zeros((60, 60), dtype=bool), np.zeros((60, 60), dtype=bool)
    for g in glyphs:
        x0, y0, x1, y1 = g['bbox']; text[y0:y1, x0:x1] = True
    bond[36, 10:20] = True
    before = score_crop([8, 8, 22, 28], glyphs[:2], glyphs, 1, text, bond)
    after = score_crop([8, 8, 22, 42], glyphs[:2], glyphs, 1, text, bond)
    assert not before['complete'] and after['complete']
    assert before['covered_target_ink_pixels'] == 130
    assert after['covered_target_ink_pixels'] == 200
    assert before['neighboring_atom_ids'] == [] and after['neighboring_atom_ids'] == [2]
    assert after['neighboring_ink_pixels'] == 40 and after['bond_ink_pixels'] == 10
    assert vertical_layout(glyphs[:2])
    assert not vertical_layout(glyphs[:1])
    assert score_crop([8, 8, 22, 42], [], glyphs, None, text, bond)['complete'] is None
