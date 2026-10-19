"""Read-only coordinate diagnostics. No changed matching or training labels."""

import math

import torch

from tools import selector_node_groups as groups


def source_xy(normalized, size=768):
    """Inverse of the source adapter's pixel/(size-1), before q1023 rounding."""
    return [float(v) * (size - 1) for v in normalized]


def quantized_anchor(center, size=768):
    return [round(float(v) / (size - 1) * 1023) / 1023 for v in center]


def cell_lower_bound(candidate, target, size=768, stride=4):
    """Best possible distance within this saved cell, not a new prediction.

    The closed cell is a conservative infimum; sigmoid offsets do not include
    the cell boundaries. Cells/peaks stay fixed in this geometry-only bound.
    """
    width = size // stride
    x, y = candidate['grid_id'] % width, candidate['grid_id'] // width
    return math.sqrt(sum(max(cell*stride - value*size, 0., value*size - (cell+1)*stride)**2
                         for cell, value in zip((x, y), target)))


def check_cells(geometry):
    size, width = geometry['image_size'], 192
    if size != 768 or geometry['match_radius_px'] != 6:
        raise ValueError('Only original768/6px contract')
    for c in geometry['candidates']:
        if type(c['grid_id']) is not int or not 0 <= c['grid_id'] < width**2:
            raise ValueError('Invalid saved stride4 cell')
        cell = (c['grid_id'] % width, c['grid_id'] // width)
        if any(not -1e-5 <= v*width - index <= 1+1e-5 for v, index in zip(c['geometry'][:2], cell)):
            raise ValueError('Candidate coordinate does not belong to saved cell')


@torch.no_grad()
def target_diagnostics(geometry, layers):
    check_cells(geometry)
    size, radius = geometry['image_size'], geometry['match_radius_px']
    atoms = sorted(layers['atoms'], key=lambda a:a['atom_id'])
    anchors, candidates = geometry['target_anchors'], geometry['candidates']
    if [a['atom_id'] for a in atoms] != list(range(len(anchors))):
        raise ValueError('Target identity mapping differs')
    for a, target in zip(atoms, anchors):
        expected = torch.tensor(quantized_anchor(a['center']), dtype=torch.float32).tolist()
        if expected != target:
            raise ValueError('Saved anchor differs from adapter quantization')
    costs = torch.cdist(torch.tensor([c['geometry'][:2] for c in candidates], dtype=torch.float32),
                        torch.tensor(anchors, dtype=torch.float32)) * size
    rows, numeric_flips, maximum_error = [], 0, 0.
    for i, (atom, target) in enumerate(zip(atoms, anchors)):
        unquantized = [v/(size-1) for v in atom['center']]
        direct = [math.dist(c['geometry'][:2], target)*size for c in candidates]
        unquantized_distances = [math.dist(c['geometry'][:2], unquantized)*size for c in candidates]
        for j, d in enumerate(direct):
            numeric_flips += (float(costs[j,i]) <= radius) != (d <= radius)
            maximum_error = max(maximum_error, abs(float(costs[j,i]) - d))
        nearest = min(range(len(candidates)), key=lambda j:direct[j])
        nearest_unquantized = min(range(len(candidates)), key=lambda j:unquantized_distances[j])
        c = candidates[nearest]
        point = source_xy(c['geometry'][:2], size)
        box = atom['bbox']
        center = [(box[0]+box[2])/2, (box[1]+box[3])/2] if box else None
        target_cell = [min(191, int(v*192)) for v in target]
        rows.append(dict(target_id=i, element=atom['element'], visible_text=atom['visible_text'],
            text=atom['text'], bbox=box, source_anchor=atom['center'], normalized_saved_target=target,
            quantized_source_anchor=source_xy(target), nearest_slot=c['slot'],
            nearest_model_xy=[v*size for v in c['geometry'][:2]], nearest_source_xy=point,
            legacy_float32_nearest_px=float(costs[:,i].min()), direct_nearest_px=direct[nearest],
            span_only_nearest_px=direct[nearest]*(size-1)/size,
            unquantized_model_nearest_px=unquantized_distances[nearest_unquantized],
            unquantized_nearest_slot=candidates[nearest_unquantized]['slot'],
            q1023_shift_model_px=math.dist(target,unquantized)*size,
            source_overlay_shift_px=math.dist(point,[v*size for v in c['geometry'][:2]]),
            anchor_to_glyph_union_center_px=math.dist(atom['center'],center) if center else None,
            candidate_to_glyph_union_center_px=math.dist(point,center) if center else None,
            target_cell=target_cell, target_cell_in_raw=any(c['grid_id']==target_cell[1]*192+target_cell[0] for c in candidates),
            nearest_cell_lower_bound_px=cell_lower_bound(c,target),
            best_existing_cell_lower_bound_px=min(cell_lower_bound(c,target) for c in candidates),
            legacy_no_candidate=bool(costs[:,i].min()>radius)))
    return dict(targets=rows, float32_direct_membership_flips=numeric_flips,
                maximum_float32_direct_distance_difference_px=maximum_error)


def pair_robustness(geometry, layers, glyphs, paths, unsupported, proposal):
    """Review old proposals under inverse projection and unquantized anchors.

    Do not replace the old matching target or publish corrected training labels.
    Any new membership is only a sensitivity finding requiring review.
    """
    byslot = {c['slot']: c for c in geometry['candidates']}
    bonds = {b['bond_id']:b for b in layers['bonds']}
    unknown = {n['bond_id'] for n in unsupported}
    size, radius = geometry['image_size'], geometry['match_radius_px']
    rows = []
    for pair in proposal['proposed_pairs']:
        c = byslot[pair['fragment_slot']]
        point = source_xy(c['geometry'][:2], size)
        evidence = groups.ownership(point, glyphs, paths, bonds, radius*(size-1)/size)
        owners = evidence['possible_owners']
        reasons = []
        if owners != [pair['owner_target']]:
            reasons.append('source_projection_owner_not_confirmed')
        if set(evidence['kinds']) != set(pair['kinds']):
            reasons.append('source_projection_evidence_kind_changed')
        if any(t['owner'] is None for t in evidence['nearby_tips']):
            reasons.append('source_projection_tip_ambiguous')
        if any(t['bond_id'] in unknown for t in evidence['nearby_tips']):
            reasons.append('source_projection_related_path_unparsed')
        unquantized_members = [a['atom_id'] for a in layers['atoms']
            if math.dist(c['geometry'][:2],[v/(size-1) for v in a['center']])*size <= radius]
        if unquantized_members:
            reasons.append('q1023_membership_sensitive')
        rows.append(dict(fragment_slot=c['slot'], owner_target=pair['owner_target'], old_kinds=pair['kinds'],
            old_overlay_xy=[v*size for v in c['geometry'][:2]], source_xy=point,
            source_evidence=evidence, unquantized_members=unquantized_members,
            review_reasons=reasons, coordinate_stable=not reasons,
            training_authorized=False, requires_review=True))
    return rows
