"""Offline location evidence for saved candidates, NOT inference or relabeling."""

import math
import re
import xml.etree.ElementTree as ET

import torch
from scipy.optimize import linear_sum_assignment

NUMBER = r'[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?'
COMMAND = re.compile(r'([ML])\s*(' + NUMBER + r')[,\s]+(' + NUMBER + r')')


def linear_points(value):
    """Only explicit absolute M/L paths; never guess curved/closed geometry."""
    matches = list(COMMAND.finditer(value))
    if not matches or COMMAND.sub('', value).strip(' ,\t\n\r'):
        return None
    if matches[0][1] != 'M' or any(m[1] != 'L' for m in matches[1:]):
        return None
    points = [(float(m[2]), float(m[3])) for m in matches]
    return points if len(points) >= 2 and all(math.isfinite(v) for p in points for v in p) else None


def segment_distance(point, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    den = dx * dx + dy * dy
    t = 0. if not den else max(0., min(1., ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / den))
    return math.dist(point, (a[0] + t * dx, a[1] + t * dy))


def strokes(svg, layers):
    root = ET.fromstring(svg)
    if root.get('transform') or root.get('viewBox') != '0 0 768 768':
        raise ValueError('Unsupported SVG root coordinate transform')
    result, unsupported = [], []
    for bond in layers['bonds']:
        for node_id in bond['svg_node_indices']:
            node = root[node_id]
            points = linear_points(node.get('d', ''))
            if node.get('transform') or points is None:
                unsupported.append(dict(bond_id=bond['bond_id'], node_index=node_id))
                continue
            if node.tag.rsplit('}', 1)[-1] != 'path' or f'bond-{bond["bond_id"]}' not in node.get('class', '').split():
                raise ValueError('Bond attribution differs from layer node indices')
            result.append(dict(bond_id=bond['bond_id'], node_index=node_id, points=points,
                chemical_type=bond['chemical_type'], drawing_type=bond['drawing']['bond_type'],
                begin=bond['begin'], end=bond['end']))
    return result, unsupported


def inside(point, box):
    return bool(box is not None and box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3])


def location(candidate, image_size, radius, glyphs, paths):
    point = [v * image_size for v in candidate['geometry'][:2]]
    hits = [dict(atom_id=g['atom_id'], text=g['text'], bbox=g['bbox']) for g in glyphs if inside(point, g['bbox'])]
    ends, segments = [], []
    for path in paths:
        for endpoint in (path['points'][0], path['points'][-1]):
            ends.append(dict(distance_px=math.dist(point, endpoint), point=endpoint,
                bond_id=path['bond_id'], node_index=path['node_index'],
                chemical_type=path['chemical_type'], drawing_type=path['drawing_type'],
                endpoint_axis_offset_px=segment_distance(endpoint, path['begin'], path['end'])))
        segments.extend(segment_distance(point, a, b) for a, b in zip(path['points'], path['points'][1:]))
    nearest = min(ends, key=lambda p: p['distance_px']) if ends else None
    # This bucket is descriptive only; it never changes labels or selection.
    if candidate['raw_target'] >= 0:
        bucket = 'raw_positive'
    elif candidate['nearest_distance_px'] <= radius:
        bucket = 'near_target_unassigned'  # Could be replacement OR duplicate; selected mapping decides.
    elif hits:
        bucket = 'glyph_bbox_outside_target_radius'
    elif nearest and nearest['distance_px'] <= radius:
        multiple = nearest['drawing_type'] in ('DOUBLE', 'TRIPLE')
        bucket = ('multiple_bond_off_axis_endpoint' if multiple and nearest['endpoint_axis_offset_px'] > radius
                  else 'other_bond_stroke_endpoint')
    else:
        bucket = 'other_or_unresolved'
    return dict(pixel_xy=point, bucket=bucket, glyph_bbox_hits=hits, nearest_stroke_endpoint=nearest,
        nearest_linear_stroke_distance_px=min(segments) if segments else None)


@torch.no_grad()
def assignment(geometry, slots):
    """Replay unchanged CPU Hungarian then 6px gate on saved float32 geometry."""
    candidates = {c['slot']: c for c in geometry['candidates']}
    chosen = sorted(slots)
    mapping = {s: -1 for s in chosen}
    if not chosen:
        return mapping
    points = torch.tensor([candidates[s]['geometry'][:2] for s in chosen], dtype=torch.float32)
    targets = torch.tensor(geometry['target_anchors'], dtype=torch.float32)
    costs = torch.cdist(points, targets) * geometry['image_size']
    rows, columns = linear_sum_assignment(costs.numpy())
    for r, c in zip(rows, columns):
        if float(costs[r, c]) <= geometry['match_radius_px']:
            mapping[chosen[r]] = int(c)
    return mapping


def check_geometry(geometry, layers):
    candidates = geometry['candidates']
    if geometry['image_size'] != 768 or geometry['match_radius_px'] != 6:
        raise ValueError('Original fixed geometry contract changed')
    ids = [c['slot'] for c in candidates]
    if len(set(ids)) != len(ids) or not ids:
        raise ValueError('Empty/duplicate candidate slots')
    for c in candidates:
        if (len(c['geometry']) != 8 or not all(math.isfinite(v) for v in c['geometry'])
                or any(not 0 <= v <= 1 for v in c['geometry'][:2])):
            raise ValueError('Invalid saved geometry')
    anchors = geometry['target_anchors']
    atoms = sorted(layers['atoms'], key=lambda a: a['atom_id'])
    expected = [[round(v / 767 * 1023) / 1023 for v in a['center']] for a in atoms]
    if len(anchors) != len(expected) or any(abs(x - y) > 1e-7 for a, b in zip(anchors, expected) for x, y in zip(a, b)):
        raise ValueError('Saved target anchors disagree with same-render metadata')
    restored = assignment(geometry, ids)
    if any(restored[c['slot']] != c['raw_target'] for c in candidates):
        raise ValueError('Saved raw supervision changed on geometry replay')


def decisions(geometry, record, locations):
    if record['key'] != geometry['key']:
        raise ValueError('Prediction/geometry identity mismatch')
    metrics = record['metrics']
    candidates = {c['slot']: c for c in geometry['candidates']}
    logits = metrics['logits']
    if {int(k) for k in logits} != set(candidates) or any(not math.isfinite(z) for z in logits.values()):
        raise ValueError('Candidate/logit coverage mismatch')
    selected = {int(s) for s, z in logits.items() if z >= -.25}
    mapping = assignment(geometry, selected)
    matched = sorted(t for t in mapping.values() if t >= 0)
    positives = {s for s, c in candidates.items() if c['raw_target'] >= 0}
    if (matched != metrics['matched_target_ids'] or len(selected) != metrics['selected_atoms']
            or sorted(positives) != metrics['positive_slots']
            or sorted(selected & positives) != metrics['retained_positive_slots']
            or sorted(selected - positives) != metrics['selected_negative_slots']
            or sum(t < 0 for t in mapping.values()) != metrics['unmatched_selected']):
        raise ValueError('Saved assessment changed on selected geometry replay')
    return [dict(slot=s, raw_target=c['raw_target'], selected=s in selected,
        selected_target=mapping.get(s, -1), logit=logits[str(s)],
        nearest_target=c['nearest_target'], nearest_distance_px=c['nearest_distance_px'],
        **locations[s]) for s, c in candidates.items() if s in selected or s in positives]
