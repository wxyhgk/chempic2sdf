"""Truth-side scoring only; never imported by crop proposal/inference code."""

from collections import Counter
import math

from chemvector.components.ocr.crop_proposals import checked_box, resize_geometry


def intersection(a, b):
    return (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))


def area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def mask_count(mask, box):
    box = intersection(box, (0, 0, mask.shape[1], mask.shape[0]))
    if not area(box):
        return 0
    x0, y0, x1, y1 = map(int, box)
    return int(mask[y0:y1, x0:x1].sum())


def unique_anchor_matches(predictions, atoms, radius=6.):
    """Conservative pixel-space audit, NOT historical training pair matching.

    No nearest fallback: multiple targets or competing crops stay ambiguous.
    All atoms (including invisible carbons) participate in ownership checks.
    """
    if len({p['candidate'] for p in predictions}) != len(predictions):
        raise ValueError('Duplicate predicted candidate')
    if len({a['atom_id'] for a in atoms}) != len(atoms):
        raise ValueError('Duplicate target atom')
    options = {}
    for p in predictions:
        point = p.get('anchor_px')
        options[p['candidate']] = [] if point is None else [a['atom_id'] for a in atoms
            if math.dist(point, a['center']) <= radius]
    claims = Counter(a for ids in options.values() for a in ids)
    result = {}
    for candidate, ids in options.items():
        status = 'unmatched' if not ids else (
            'matched' if len(ids) == 1 and claims[ids[0]] == 1 else 'ambiguous')
        result[candidate] = dict(status=status, nearby_atom_ids=ids,
                                 atom_id=ids[0] if status == 'matched' else None)
    return result


def vertical_layout(glyphs):
    # Scoring category only; never selects or changes a proposal.
    return any(a['bbox'][1] >= b['bbox'][3] or b['bbox'][1] >= a['bbox'][3]
               for i, a in enumerate(glyphs) for b in glyphs[i + 1:])


def score_crop(box, glyphs, all_glyphs, atom_id, text_mask, bond_mask):
    checked_box(box)
    raster = resize_geometry(box)
    own = []
    for glyph in glyphs:
        target = checked_box(glyph['bbox'])
        overlap = intersection(box, target)
        own.append(dict(node_id=glyph['node_id'], text=glyph['text'], bbox=glyph['bbox'],
            complete=area(overlap) == area(target), bbox_area_coverage=area(overlap) / area(target),
            ink_pixels=mask_count(text_mask, target), covered_ink_pixels=mask_count(text_mask, overlap),
            full_glyph_height_after_resize_px=(target[3] - target[1]) * raster['scale_y']))
    neighbors = [g for g in all_glyphs if g['atom_id'] != atom_id and area(intersection(box, g['bbox'])) > 0]
    return dict(glyphs=own, complete=all(g['complete'] for g in own) if own else None,
        neighboring_atom_ids=sorted({g['atom_id'] for g in neighbors}),
        neighboring_ink_pixels=sum(mask_count(text_mask, intersection(box, g['bbox'])) for g in neighbors),
        bond_ink_pixels=mask_count(bond_mask, box),
        target_ink_pixels=sum(g['ink_pixels'] for g in own),
        covered_target_ink_pixels=sum(g['covered_ink_pixels'] for g in own),
        minimum_full_glyph_height_after_resize_px=min(
            (g['full_glyph_height_after_resize_px'] for g in own), default=None))
