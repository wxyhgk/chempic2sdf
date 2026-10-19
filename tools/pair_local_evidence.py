"""Prediction-only local edge evidence. Diagnostics, never an edge-deletion rule.

No target graph, matching labels, learned parameters, or inference is accepted.
All geometric/pixel tolerances are exploratory and fixed before the saved audit.
"""

from collections import deque
from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Config:
    ribbon_half_width: float = 12.0
    narrow_half_width: float = 3.0
    ink_threshold: int = 200
    endpoint_padding: float = 2.0
    interior_start: float = .04
    interior_end: float = .96
    minimum_gap_px: float = 12.0
    minimum_gap_fraction: float = .15
    maximum_path_hops: int = 4

    def to_dict(self):
        return asdict(self)


DEFAULT = Config()
ORDERS = {'SINGLE': 1, 'DOUBLE': 2, 'TRIPLE': 3}


def prediction_graph(trace, image_size):
    width, height = image_size
    scene, lineage = trace['scene_before_fallback'], trace['compiled_candidate_indices']
    atoms = scene['atom_objects']
    if (width <= 0 or height <= 0 or len(atoms) != len(lineage)
            or len(set(lineage)) != len(lineage)
            or any(type(i) is not int or i < 0 for i in lineage)
            or [a['id'] for a in atoms] != list(range(len(atoms)))):
        raise ValueError('Unique packed saved candidate lineage required')
    nodes = {}
    for slot, atom in zip(lineage, atoms):
        anchor = atom['anchor']
        if len(anchor) != 2 or not all(math.isfinite(v) and 0 <= v <= 1023 for v in anchor):
            raise ValueError('Finite q1023 anchor required')
        text = atom.get('text')
        box = None if text is None else text.get('bbox')
        if box is not None:
            if (len(box) != 4 or not all(math.isfinite(v) and 0 <= v <= 1023 for v in box)
                    or box[0] > box[2] or box[1] > box[3]):
                raise ValueError('Valid saved text box required')
            box = [box[0]*width/1023, box[1]*height/1023, box[2]*width/1023, box[3]*height/1023]
        nodes[slot] = dict(xy=[anchor[0]*width/1023, anchor[1]*height/1023],
            text_box=box, text=None if text is None else text['value'],
            chemical=dict(atom['chemical']))
    edges = {}
    for edge in scene['bond_objects']:
        refs = edge['atom_refs']
        if len(refs) != 2 or any(type(i) is not int or not 0 <= i < len(lineage) for i in refs):
            raise ValueError('Invalid saved scene edge')
        pair = tuple(sorted(lineage[i] for i in refs))
        if pair[0] == pair[1] or pair in edges:
            raise ValueError('Self or duplicate scene edge')
        edges[pair] = edge['chemical_type']
    return dict(nodes=nodes, edges=edges)


def projection(a, b, point):
    delta, relative = np.asarray(b, float)-a, np.asarray(point, float)-a
    squared = float(delta @ delta)
    if not math.isfinite(squared) or squared <= 0:
        raise ValueError('Distinct finite endpoints required')
    return float(relative @ delta / squared), float(abs(relative[0]*delta[1]-relative[1]*delta[0]) / math.sqrt(squared))


def corridor_path(graph, pair, interior, maximum_hops):
    allowed = {pair[0], pair[1], *interior}
    adjacency = {i: [] for i in allowed}
    for edge in graph['edges']:
        if edge != pair and set(edge) <= allowed:
            a, b = edge; adjacency[a].append(b); adjacency[b].append(a)
    queue, seen = deque([[pair[0]]]), {pair[0]}
    while queue:
        path = queue.popleft()
        if path[-1] == pair[1]:
            return path
        if len(path)-1 >= maximum_hops:
            continue
        for node in sorted(adjacency[path[-1]]):
            if node not in seen:
                seen.add(node); queue.append(path + [node])
    return None


def _text_exit(anchor, direction, box, padding):
    if box is None:
        return padding, 'implicit_or_no_text_box'
    x, y = anchor
    bounds = (box[0]-padding, box[1]-padding, box[2]+padding, box[3]+padding)
    if not (bounds[0] <= x <= bounds[2] and bounds[1] <= y <= bounds[3]):
        return padding, 'anchor_outside_text_box'
    exits = []
    for axis in (0, 1):
        value = direction[axis]
        if abs(value) > 1e-12:
            boundary = bounds[axis+2] if value > 0 else bounds[axis]
            exits.append((boundary-anchor[axis])/value)
    return float(max(padding, min(exits))), 'endpoint_text_box'


def _longest_blank(present, spacing):
    longest = run = 0
    for value in present:
        run = 0 if value else run + 1
        longest = max(longest, run)
    # Report extent between sample centers, not a biased (N+1)-pixel length.
    return max(0, longest - 1) * spacing


def pixel_support(gray, first, second, config=DEFAULT):
    pixels = np.asarray(gray)
    if pixels.ndim != 2 or pixels.dtype != np.uint8:
        raise ValueError('Unmodified uint8 grayscale source raster required')
    a, b = np.asarray(first['xy'], float), np.asarray(second['xy'], float)
    length = float(np.linalg.norm(b-a))
    if not math.isfinite(length) or length <= 0:
        raise ValueError('Distinct finite endpoints required')
    direction = (b-a)/length; normal = np.array([-direction[1], direction[0]])
    start, reason_a = _text_exit(a, direction, first['text_box'], config.endpoint_padding)
    end, reason_b = _text_exit(b, -direction, second['text_box'], config.endpoint_padding)
    span = length-start-end
    base = dict(length_px=length, start_trim_px=start, end_trim_px=end,
                trim_reasons=[reason_a, reason_b], checked_span_px=max(0., span))
    if span <= 2:
        return dict(**base, available=False, reason='endpoint_regions_cover_segment', gap_warning=False)
    distances = np.linspace(start, length-end, math.ceil(span)+1)
    spacing = float(distances[1]-distances[0])
    center = a + distances[:, None]*direction
    profiles = {}
    for name, radius in (('narrow', config.narrow_half_width), ('wide', config.ribbon_half_width)):
        offsets = np.linspace(-radius, radius, math.ceil(radius*2)+1)
        points = np.rint(center[:, None, :] + offsets[None, :, None]*normal).astype(int)
        valid = ((points[..., 0] >= 0) & (points[..., 0] < pixels.shape[1])
                 & (points[..., 1] >= 0) & (points[..., 1] < pixels.shape[0]))
        if not bool(valid.all()):
            return dict(**base, available=False, reason='ribbon_outside_source', gap_warning=False)
        ink = (pixels[points[..., 1], points[..., 0]] < config.ink_threshold).any(axis=1)
        profiles[name] = dict(ink_coverage=float(ink.mean()), max_blank_run_px=_longest_blank(ink, spacing))
    cutoff = max(config.minimum_gap_px, config.minimum_gap_fraction*span)
    # Never use the narrow corridor alone: double/triple bonds may have no
    # ink on their centerline. This is still a warning, not proof of nonedge.
    return dict(**base, available=True, profiles=profiles, gap_cutoff_px=cutoff,
                gap_warning=bool(profiles['wide']['max_blank_run_px'] >= cutoff))


def carbon_conflicts(graph, pair, chemical_type):
    edges = dict(graph['edges']); edges.setdefault(pair, chemical_type)
    result = []
    for slot in pair:
        chemistry = graph['nodes'][slot]['chemical']
        incident = [(edge, kind) for edge, kind in edges.items() if slot in edge]
        eligible = (chemistry.get('element') == 'C' and chemistry.get('formal_charge', 0) == 0
                    and chemistry.get('radical_electrons', 0) == 0)
        order_available = eligible and all(kind in ORDERS for _, kind in incident)
        valence = (sum(ORDERS[kind] for _, kind in incident) + chemistry.get('explicit_h_count', 0)
                   if order_available else None)
        conflict = eligible and (len(incident) > 4 or valence is not None and valence > 4)
        result.append(dict(candidate=slot, eligible_neutral_carbon=eligible, degree=len(incident),
            nonaromatic_order_plus_explicit_h=valence, conflict=bool(conflict),
            incident_pairs=[list(edge) for edge, _ in incident]))
    return result


def inspect_pair(gray, graph, pair, chemical_type, config=DEFAULT):
    pair = tuple(pair)
    if (len(pair) != 2 or any(type(i) is not int for i in pair) or pair[0] >= pair[1]
            or any(i not in graph['nodes'] for i in pair)):
        raise ValueError('Ordered distinct existing candidate pair required')
    first, second = [graph['nodes'][i] for i in pair]
    interior = []
    for slot, node in graph['nodes'].items():
        if slot in pair: continue
        t, distance = projection(first['xy'], second['xy'], node['xy'])
        if config.interior_start <= t <= config.interior_end and distance <= config.ribbon_half_width:
            interior.append(dict(candidate=slot, fraction=t, distance_px=distance, predicted_text=node['text']))
    interior.sort(key=lambda row: row['fraction'])
    path = corridor_path(graph, pair, [r['candidate'] for r in interior], config.maximum_path_hops)
    pixels = pixel_support(gray, first, second, config)
    carbon = carbon_conflicts(graph, pair, chemical_type)
    local_warning = path is not None or pixels['gap_warning']
    conflict = any(c['conflict'] for c in carbon)
    return dict(pair=list(pair), already_present_in_scene=pair in graph['edges'],
        interior_candidates=interior, predicted_alternate_corridor_path=path,
        pixels=pixels, carbon_endpoints=carbon,
        flags=dict(interior_candidate=bool(interior), alternate_corridor_path=path is not None,
                   raster_gap=pixels['gap_warning'], carbon_conflict=conflict,
                   combined_review=local_warning and conflict),
        diagnostic_only=True, action='none')
