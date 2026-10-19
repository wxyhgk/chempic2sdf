"""Audit-only crop proposals. No labels, truth, chemistry or model execution.

Keep the historical crop as well as a bounded vertical alternative. This is
deliberately not wired into inference: a larger crop is not an OCR correction.
"""

import math

import numpy as np
from PIL import Image


RULE = dict(name='vertical-one-predicted-height-v1', context_px=4,
            minimum_height_width_ratio=1., maximum_extension_per_side_px=32,
            output_size=[192, 48], original_crop_preserved=True)


def checked_box(box):
    if len(box) != 4 or any(isinstance(x, bool) or not math.isfinite(x) for x in box):
        raise ValueError('Finite xyxy box required')
    if box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError('Positive box dimensions required')
    return tuple(float(x) for x in box)


def predicted_box(center, half_extent):
    """Reproduce saved normalized FP32 tensor multiplication BEFORE subtraction."""
    if (len(center) != 2 or len(half_extent) != 2
            or any(isinstance(x, bool) or not math.isfinite(x) for x in [*center, *half_extent])
            or any(x <= 0 or x > 1 for x in half_extent)
            or any(x < 0 or x > 1 for x in center)):
        raise ValueError('Normalized finite center and positive extents required')
    x, y = (np.asarray(center, dtype=np.float32) * np.float32(768)).tolist()
    w, h = (np.asarray(half_extent, dtype=np.float32) * np.float32(768)).tolist()
    return checked_box((x - w, y - h, x + w, y + h))


def propose_vertical(box, image_size):
    x0, y0, x1, y1 = checked_box(box)
    if len(image_size) != 2 or any(type(x) is not int or x <= 0 for x in image_size):
        raise ValueError('Positive integer image size required')
    # Pillow rounds float boundaries to nearest-even integers. Historical
    # out-of-image black padding is preserved, not silently corrected here.
    original = tuple(round(v) for v in (x0 - 4, y0 - 4, x1 + 4, y1 + 4))
    width, height = x1 - x0, y1 - y0
    extension = min(height, 32.) if height >= width else 0.
    expanded = (original[0], min(original[1], max(0, round(y0 - 4 - extension))),
                original[2], max(original[3], min(image_size[1], round(y1 + 4 + extension))))
    return dict(original=list(original), proposed=list(expanded),
                eligible=height >= width, changed=expanded != original,
                requested_extension_per_side_px=extension,
                original_raster=resize_geometry(original), proposed_raster=resize_geometry(expanded))


def resize_geometry(box):
    x0, y0, x1, y1 = checked_box(box)
    width, height = x1 - x0, y1 - y0
    scale = min(192 / width, 48 / height)
    resized = [max(1, round(width * scale)), max(1, round(height * scale))]
    return dict(source_size=[width, height], resized_size=resized,
                scale_x=resized[0] / width, scale_y=resized[1] / height)


def rasterize(image, box):
    """Same aspect-preserving canvas as bbox_crop, from integer crop bounds."""
    checked_box(box)
    if any(type(x) is not int for x in box):
        raise ValueError('Integer Pillow crop bounds required')
    crop = image.crop(tuple(box)).convert('RGB')
    size = resize_geometry(box)['resized_size']
    resized = crop.resize(tuple(size), Image.Resampling.LANCZOS)
    canvas = Image.new('RGB', (192, 48), 'white')
    canvas.paste(resized, ((192 - size[0]) // 2, (48 - size[1]) // 2))
    return canvas
