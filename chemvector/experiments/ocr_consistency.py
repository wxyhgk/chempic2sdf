"""Fixed prediction-box perturbations and review-only OCR disagreement checks.

No glyph labels, atom chemistry, crop winner selection, or output rewriting.
Agreement is stability under these probes, not proof that a character is right.
"""

import math

# One predeclared probe set, not a search for the crop with the best label score.
VARIANTS = (
    ('base', 0., 0., 1., 4.),
    ('left2', -2., 0., 1., 4.), ('right2', 2., 0., 1., 4.),
    ('up2', 0., -2., 1., 4.), ('down2', 0., 2., 1., 4.),
    ('scale090', 0., 0., .9, 4.), ('scale110', 0., 0., 1.1, 4.),
    ('context2', 0., 0., 1., 2.), ('context6', 0., 0., 1., 6.),
)


def crop_variants(box, image_size):
    if (len(box) != 4 or any(not math.isfinite(v) for v in box)
            or box[0] >= box[2] or box[1] >= box[3]
            or len(image_size) != 2 or any(type(v) is not int or v <= 0 for v in image_size)):
        raise ValueError('Finite nonempty predicted box and positive image dimensions required')
    x0, y0, x1, y1 = box
    cx, cy, hw, hh = (x0+x1)/2, (y0+y1)/2, (x1-x0)/2, (y1-y0)/2
    rows = []
    for name, dx, dy, scale, context in VARIANTS:
        new_box = [cx+dx-hw*scale, cy+dy-hh*scale, cx+dx+hw*scale, cy+dy+hh*scale]
        actual = [round(new_box[0]-context), round(new_box[1]-context),
                  round(new_box[2]+context), round(new_box[3]+context)]
        available = (0 <= actual[0] < actual[2] <= image_size[0]
                     and 0 <= actual[1] < actual[3] <= image_size[1])
        rows.append(dict(variant=name, box=new_box, context=context, actual_crop=actual, available=available))
    return rows


def summarize(decoded):
    """Each variant has a text or None if unavailable; no majority correction."""
    if (len(decoded) != len(VARIANTS) or {r['variant'] for r in decoded} != {v[0] for v in VARIANTS}
            or any(r['text'] is not None and not isinstance(r['text'], str) for r in decoded)):
        raise ValueError('Exactly one result for each fixed OCR variant required')
    by_name = {r['variant']: r['text'] for r in decoded}
    distinct = sorted({v for v in by_name.values() if v is not None})
    missing = sorted(k for k, v in by_name.items() if v is None)
    blank = any(v == '' for v in by_name.values())
    return dict(original_text=by_name['base'], output_text=by_name['base'], output_changed=False,
                distinct_texts=distinct, unavailable_variants=missing,
                needs_review=bool(missing or blank or len(distinct) != 1),
                action='review' if missing or blank or len(distinct) != 1 else 'keep_with_stability_note',
                agreement_is_correctness=False)
