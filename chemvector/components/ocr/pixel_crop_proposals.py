"""Pixel-only, audit-only bounded alternatives for vertically clipped ink.

Not a text detector or a semantic repair. A component may be a bond or noise.
Retain the original crop and expose evidence; never change recognized text.
"""

import numpy as np
from PIL import Image
from scipy import ndimage

from .crop_proposals import checked_box, propose_vertical, resize_geometry


RULE = dict(name='boundary-crossing-component-v1', dark_channel_threshold=128,
            connectivity=8, minimum_width_to_predicted_width=.30,
            minimum_height_to_predicted_height=.50, maximum_height_to_predicted_height=1.50,
            minimum_density=.15, minimum_pixels=6, minimum_inside_rows=2,
            component_padding_px=2, maximum_components=128,
            search='frozen_vertical_one_height_max32_each_side',
            reject_search_boundary_touch=True, reject_multiple_components_per_side=True,
            preserve_original_rectangle=True, output_size=[192, 48])


def propose_from_pixels(image, box):
    """Only RGB pixels and a predicted bbox enter; no IDs, OCR text or truth."""
    if not isinstance(image, Image.Image) or image.mode != 'RGB' or image.size != (768, 768):
        raise ValueError('Original 768x768 RGB image required')
    box = checked_box(box)
    broad = propose_vertical(box, image.size)
    original = broad['original']
    search = [max(0, broad['proposed'][0]), max(0, broad['proposed'][1]),
              min(768, broad['proposed'][2]), min(768, broad['proposed'][3])]
    result = dict(original=original, proposed=list(original), broad=broad['proposed'],
        eligible=broad['eligible'], changed=False, status='no_boundary_evidence',
        search_box=search, components=[], original_raster=broad['original_raster'],
        proposed_raster=broad['original_raster'])
    if not broad['eligible']:
        result['status'] = 'wide_box_unchanged'
        return result
    if search[0] >= search[2] or search[1] >= search[3]:
        result['status'] = 'outside_image_unchanged'
        return result
    # Outside-image padding is NOT evidence. Threshold is fixed before scoring.
    pixels = np.asarray(image.crop(tuple(search)))
    dark = pixels.min(axis=2) < RULE['dark_channel_threshold']
    labels, count = ndimage.label(dark, structure=np.ones((3, 3), dtype=np.uint8))
    result['component_count'] = int(count)
    if count > RULE['maximum_components']:
        result['status'] = 'component_budget_unchanged'
        return result
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    accepted = []
    for identity, slices in enumerate(ndimage.find_objects(labels), start=1):
        ys, xs = slices
        bounds = [xs.start + search[0], ys.start + search[1], xs.stop + search[0], ys.stop + search[1]]
        sides = []
        if bounds[1] < original[1] and bounds[3] >= original[1] + RULE['minimum_inside_rows']:
            sides.append('top')
        if bounds[3] > original[3] and bounds[1] <= original[3] - RULE['minimum_inside_rows']:
            sides.append('bottom')
        if not sides:
            continue
        width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
        density = int(sizes[identity]) / (width * height)
        reasons = []
        if any(bounds[i] == search[i] for i in range(4)):
            reasons.append('touches_search_boundary')
        if width < RULE['minimum_width_to_predicted_width'] * (box[2] - box[0]):
            reasons.append('too_narrow')
        if not (RULE['minimum_height_to_predicted_height'] * (box[3] - box[1]) <= height
                <= RULE['maximum_height_to_predicted_height'] * (box[3] - box[1])):
            reasons.append('height_outside_range')
        if density < RULE['minimum_density'] or sizes[identity] < RULE['minimum_pixels']:
            reasons.append('too_sparse')
        item = dict(component=identity, bbox=bounds, sides=sides, dark_pixels=int(sizes[identity]),
                    density=density, rejection_reasons=reasons)
        result['components'].append(item)
        if not reasons:
            accepted.append(item)
    if any(sum(side in c['sides'] for c in accepted) > 1 for side in ('top', 'bottom')):
        result['status'] = 'ambiguous_boundary_unchanged'
        return result
    candidate = list(original)
    for component in accepted:
        if 'top' in component['sides']:
            candidate[1] = min(candidate[1], max(search[1], component['bbox'][1] - RULE['component_padding_px']))
        if 'bottom' in component['sides']:
            candidate[3] = max(candidate[3], min(search[3], component['bbox'][3] + RULE['component_padding_px']))
    result.update(proposed=candidate, changed=candidate != original, proposed_raster=resize_geometry(candidate),
                  status='candidate_only' if candidate != original else 'no_boundary_evidence')
    return result
