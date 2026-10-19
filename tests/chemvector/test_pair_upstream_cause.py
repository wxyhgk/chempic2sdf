from copy import deepcopy

import pytest
from PIL import Image
import torch

from chemvector.ctc_ocr import bbox_crop
from chemvector.evaluate_direct_atom_ocr import crop_tensor
from tools.diagnose_pair_upstream2 import as_tensor, crop_geometry, predicted_box, selected_parity


def test_bbox_reconstruction_uses_original_float32():
    row = dict(center=[.5365803837776184, .4393806755542755],
               half_extent=[.010362623259425163, .013219784945249557])
    x, y = (torch.tensor(row['center']) * 768).tolist()
    w, h = (torch.tensor(row['half_extent']) * 768).tolist()
    assert predicted_box(row) == [x - w, y - h, x + w, y + h]


def test_context_prevents_false_clipping_claim():
    box, target = [404.1, 327.3, 420.0, 347.6], [406, 328, 421, 346]
    assert not crop_geometry(box, target, 0)['covers_entire_target_bbox']
    assert crop_geometry(box, target, 4)['covers_entire_target_bbox']


def test_rounding_matches_pil_crop_dimensions():
    im = Image.new('RGB', (100, 100))
    box = [10.5, 20.5, 29.5, 40.5]
    actual = crop_geometry(box, box, 4)['actual_integer_crop']
    assert im.crop([box[0]-4, box[1]-4, box[2]+4, box[3]+4]).size == (actual[2]-actual[0], actual[3]-actual[1])


def test_control_tensor_matches_original_preprocessing():
    im = Image.new('RGB', (100, 100), (150, 30, 100))
    box = [15, 15, 35, 40]
    assert torch.equal(as_tensor(bbox_crop(im, box, context=4)), crop_tensor(im, box))


@pytest.mark.parametrize('extent', [[0, .01], [-.01, .01], [float('nan'), .01]])
def test_bad_predicted_bbox_rejected(extent):
    with pytest.raises(ValueError):
        predicted_box(dict(center=[.5, .5], half_extent=extent))


def test_parity_requires_ids_and_anchors_but_not_packed_order():
    raw = dict(atom_geometry=torch.tensor([[[.1, .2], [.3, .4]]]))
    selected = dict(atom_mask=torch.tensor([[True, True]]))
    trace = dict(compiled_candidate_indices=[1, 0], scene_before_fallback=dict(atom_objects=[
        dict(anchor=[307, 409]), dict(anchor=[102, 205])]))
    original = deepcopy(trace)
    assert selected_parity(raw, selected, trace)['selected_q1023_anchors_equal']
    assert trace == original
    trace['scene_before_fallback']['atom_objects'][0]['anchor'] = [307, 410]
    assert not selected_parity(raw, selected, trace)['selected_q1023_anchors_equal']
