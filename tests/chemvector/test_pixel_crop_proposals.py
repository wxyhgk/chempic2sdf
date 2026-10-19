from copy import deepcopy
import inspect

import numpy as np
from PIL import Image, ImageDraw
import pytest

from chemvector.components.ocr.pixel_crop_proposals import propose_from_pixels
from chemvector.components.ocr.crop_proposals import rasterize


BOX = [300, 300, 320, 330]  # Historical crop [296, 296, 324, 334].


def blank():
    return Image.new('RGB', (768, 768), 'white')


def rectangle(image, bounds, color='black'):
    ImageDraw.Draw(image).rectangle(bounds, fill=color)


def test_bottom_component_gets_tight_extension_and_keeps_original():
    image = blank(); rectangle(image, (302, 325, 316, 347))
    source = np.asarray(image).copy(); box = deepcopy(BOX)
    result = propose_from_pixels(image, box)
    assert result['original'] == [296, 296, 324, 334]
    assert result['proposed'] == [296, 296, 324, 350]
    assert result['broad'] == [296, 266, 324, 364]
    assert result['changed'] and result['status'] == 'candidate_only'
    assert result['proposed_raster']['scale_y'] > 48 / 98
    assert np.array_equal(np.asarray(image), source) and box == BOX


def test_top_component_and_both_sides_supported():
    image = blank(); rectangle(image, (302, 277, 316, 299))
    assert propose_from_pixels(image, BOX)['proposed'] == [296, 275, 324, 334]
    rectangle(image, (302, 325, 316, 347))
    assert propose_from_pixels(image, BOX)['proposed'] == [296, 275, 324, 350]


@pytest.mark.parametrize('bounds', [None, (302, 301, 316, 323), (302, 336, 316, 358)])
def test_blank_complete_and_fully_outside_ink_do_not_trigger(bounds):
    image = blank()
    if bounds:
        rectangle(image, bounds)
    p = propose_from_pixels(image, BOX)
    assert not p['changed'] and p['original'] == p['proposed']
    assert np.array_equal(np.asarray(rasterize(image, p['original'])), np.asarray(rasterize(image, p['proposed'])))


@pytest.mark.parametrize('bounds,reason', [((300, 332, 320, 336), 'height_outside_range'),
    ((310, 325, 311, 347), 'too_narrow'), ((296, 325, 316, 347), 'touches_search_boundary'),
    ((302, 325, 316, 370), 'touches_search_boundary')])
def test_geometry_rejections(bounds, reason):
    image = blank(); rectangle(image, bounds)
    p = propose_from_pixels(image, BOX)
    assert not p['changed']
    assert reason in p['components'][0]['rejection_reasons']


def test_sparse_diagonal_rejected():
    image = blank(); ImageDraw.Draw(image).line((301, 325, 319, 347), fill='black')
    p = propose_from_pixels(image, BOX)
    assert not p['changed'] and 'too_sparse' in p['components'][0]['rejection_reasons']


def test_two_crossing_components_are_ambiguous():
    image = blank()
    rectangle(image, (299, 325, 305, 347)); rectangle(image, (314, 325, 320, 347))
    p = propose_from_pixels(image, BOX)
    assert not p['changed'] and p['status'] == 'ambiguous_boundary_unchanged'


def test_component_budget_fails_closed():
    image = blank(); draw = ImageDraw.Draw(image)
    for x in range(298, 323, 2):
        for y in range(268, 363, 2):
            draw.point((x, y), fill='black')
    p = propose_from_pixels(image, BOX)
    assert p['status'] == 'component_budget_unchanged' and not p['changed']


def test_padding_is_not_ink_and_wide_box_not_expanded():
    assert not propose_from_pixels(blank(), [1, 1, 10, 30])['changed']
    assert propose_from_pixels(blank(), [300, 300, 340, 320])['status'] == 'wide_box_unchanged'
    assert not propose_from_pixels(blank(), [-100, -100, -80, -60])['changed']


@pytest.mark.parametrize('image', [Image.new('L', (768, 768)), Image.new('RGB', (100, 100)), None])
def test_original_image_contract(image):
    with pytest.raises(ValueError):
        propose_from_pixels(image, BOX)


def test_no_truth_ocr_or_identity_argument_and_determinism():
    assert list(inspect.signature(propose_from_pixels).parameters) == ['image', 'box']
    image = blank(); rectangle(image, (302, 325, 316, 347), color=(255, 0, 0))
    assert propose_from_pixels(image, BOX) == propose_from_pixels(image.copy(), BOX)
    with pytest.raises(TypeError):
        propose_from_pixels(image, BOX, expected_text='NH')


def test_geometric_invariants_on_random_boxes():
    image = blank(); rectangle(image, (300, 300, 350, 350))
    rng = np.random.default_rng(22)
    for _ in range(100):
        x, y = rng.uniform(-10, 770, size=2)
        w, h = rng.uniform(1, 60, size=2)
        p = propose_from_pixels(image, [x, y, x + w, y + h])
        a, b, c = p['original'], p['proposed'], p['broad']
        assert a[0] == b[0] == c[0] and a[2] == b[2] == c[2]
        assert c[1] <= b[1] <= a[1] < a[3] <= b[3] <= c[3]
