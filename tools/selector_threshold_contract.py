"""Review-only singleton threshold discrimination and affine readout diagnostics.

No optimizer, model mutation, label admission or production imports. The fixed
threshold is part of both the objective and decisions, not a tuned hyperparameter.
"""

import math

import torch
from torch.nn import functional as F

THRESHOLD = -.25


def threshold_contract(logits, contrasts):
    """Equal node means of balanced positive/owned-negative threshold BCE.

    This only defines a local diagnostic, not a complete graph training loss.
    Multi-candidate nodes and shared supervision slots are explicitly rejected.
    """
    if (not isinstance(logits, torch.Tensor) or logits.device.type != 'cpu'
            or logits.dtype not in (torch.float32, torch.float64) or logits.ndim != 1
            or not torch.isfinite(logits).all() or not contrasts):
        raise ValueError('Finite CPU score vector and nonempty contrasts required')
    occupied, identities, losses, decisions = set(), set(), [], []
    for node in contrasts:
        if (not isinstance(node.node_id, str) or not node.node_id or node.node_id in identities
                or len(node.valid_slots) != 1 or not node.fragment_slots):
            raise ValueError('Unique named singleton nodes with owned fragments required')
        slots = tuple(node.valid_slots) + tuple(node.fragment_slots)
        if (any(type(s) is not int or not 0 <= s < len(logits) for s in slots)
                or len(set(slots)) != len(slots) or occupied & set(slots)):
            raise ValueError('Disjoint unique in-range supervision slots required')
        occupied.update(slots)
        identities.add(node.node_id)
        positive = logits[node.valid_slots[0]]
        negative = logits[list(node.fragment_slots)]
        losses.append((F.softplus(THRESHOLD - positive)
                       + F.softplus(negative - THRESHOLD).mean()) / 2)
        decisions.append(dict(node_id=node.node_id,
            valid_selected=bool(positive.detach() >= THRESHOLD),
            selected_fragments=int((negative.detach() >= THRESHOLD).sum()),
            both_decisions_correct=bool(positive.detach() >= THRESHOLD)
                and bool((negative.detach() < THRESHOLD).all())))
    return dict(loss=torch.stack(losses).mean(), per_node_losses=torch.stack(losses),
        decisions=decisions, supervised_slots=sorted(occupied), threshold=THRESHOLD,
        training_authorized=False, graph_training_objective_ready=False)


def review_contrast(node, subset, geometry, radius_case):
    """Check saved provenance/ownership for diagnostic use, NOT label approval."""
    if (subset.get('training_authorized') is not False or subset.get('requires_review') is not True
            or radius_case.get('training_authorized') is not False
            or geometry['image_size'] != 768 or geometry['match_radius_px'] != 6
            or len(node['valid_slots']) != 1 or not node['stable_fragment_slots']):
        raise ValueError('Original singleton review-only evidence required')
    target, valid = node['target_id'], node['valid_slots'][0]
    proposal = subset['stable_proposal']
    originals = [n for n in proposal['nodes'] if n['target_id'] == target]
    if len(originals) != 1:
        raise ValueError('Unique source target required')
    original = originals[0]
    candidates = {c['slot']: c for c in geometry['candidates']}
    if len(candidates) != len(geometry['candidates']):
        raise ValueError('Unique candidate slots required')
    edges = {(e['slot'], e['target_id']) for e in radius_case['raw']['radius_reference']['eligible_edges']}
    if (original['status'] != 'reviewable' or original['possible_slots'] != [valid]
            or original['raw_positive_slots'] != [valid] or original['alternative_slots']
            or {s for s, t in edges if t == target} != {valid}
            or {t for s, t in edges if s == valid} != {target}
            or candidates[valid]['raw_target'] != target):
        raise ValueError('Sole representative must agree with full spatial eligibility')
    fragments = node['stable_fragment_slots']
    expected = [p for p in proposal['proposed_pairs'] if p['owner_target'] == target]
    if (len(fragments) != len(set(fragments))
            or set(fragments) != {p['fragment_slot'] for p in expected}
            or len(expected) != len(fragments)):
        raise ValueError('Complete original owned fragment set required')
    held = {p['fragment_slot'] for p in subset['excluded_pairs']}
    details = []
    for pair in expected:
        slot = pair['fragment_slot']
        reviews = [r for r in subset['stable_pair_reviews']
                   if r['fragment_slot'] == slot and r['owner_target'] == target]
        if (len(reviews) != 1 or slot in held or any(s == slot for s, t in edges)
                or candidates[slot]['raw_target'] >= 0
                or pair.get('training_authorized') is not False or pair.get('requires_review') is not True):
            raise ValueError('Outside-node stable review must remain unapproved and unambiguous')
        review = reviews[0]
        evidence = review['source_evidence']
        if (review['coordinate_stable'] is not True or review['review_reasons']
                or review['unquantized_members'] or evidence['possible_owners'] != [target]
                or set(evidence['kinds']) != set(pair['kinds'])
                or review.get('training_authorized') is not False or review.get('requires_review') is not True):
            raise ValueError('Contradictory or sensitive coordinate review')
        point = [v * 767 for v in candidates[slot]['geometry'][:2]]
        if math.dist(point, review['source_xy']) > 1e-7:
            raise ValueError('Source projection differs from saved candidate')
        supported = False
        if 'glyph_fragment' in pair['kinds'] and original['visible_text']:
            supported = any(g['atom_id'] == target and g['bbox'][0] <= point[0] <= g['bbox'][2]
                            and g['bbox'][1] <= point[1] <= g['bbox'][3]
                            for g in evidence['glyph_hits'])
        if 'multiple_bond_off_axis_tip' in pair['kinds']:
            supported |= any(t['owner'] == target and t['multiple_off_axis']
                             and t['axis_offset_px'] > 6 * 767 / 768
                             and math.dist(point, t['point']) <= 6 * 767 / 768
                             for t in evidence['nearby_tips'])
        if not supported:
            raise ValueError('No matching glyph or off-axis endpoint ownership evidence')
        details.append(dict(fragment_slot=slot, kinds=pair['kinds'], source_xy=point,
                            coordinate_evidence_consistent=True))
    return dict(target_id=target, valid_slot=valid, fragments=details,
        diagnostic_evidence_consistent=True, human_semantic_review_performed=False,
        label_release_authorized=False, training_authorized=False)


def affine_gradient(hidden, derivatives):
    """Chain rule for ONLY the final residual Linear weight and bias."""
    if (hidden.ndim != 2 or derivatives.shape != (len(hidden),)
            or hidden.shape[1] < 1 or not len(hidden)
            or any(x.device.type != 'cpu' or x.requires_grad
                   or x.dtype not in (torch.float32, torch.float64)
                   or not torch.isfinite(x).all() for x in (hidden, derivatives))):
        raise ValueError('Detached finite CPU features and output derivatives required')
    return torch.cat((hidden.double().T @ derivatives.double(), derivatives.double().sum()[None]))


def feasible_ray(constraints):
    """Intersect exact linear decision inequalities along one fixed eta >= 0 ray.

    Each row has id, logit, slope, selected. Positive uses >= threshold, negative
    uses STRICT < threshold. No parameter is updated and no step-size is searched.
    """
    if not constraints or len({r['id'] for r in constraints}) != len(constraints):
        raise ValueError('Nonempty unique decision constraints required')
    lower, upper, lower_open, upper_open = 0., math.inf, False, False
    lower_ids, upper_ids, impossible = ['nonnegative_step'], [], []
    for row in constraints:
        z, slope, selected = row['logit'], row['slope'], row['selected']
        if not math.isfinite(z) or not math.isfinite(slope) or type(selected) is not bool:
            raise ValueError('Finite scalars and explicit selected decisions required')
        # a * eta >= b (positive) or a * eta > b (negative).
        a, b, strict = (slope, THRESHOLD-z, False) if selected else (-slope, z-THRESHOLD, True)
        if a == 0:
            if not (0 > b if strict else 0 >= b):
                impossible.append(row['id'])
            continue
        bound = b / a
        if not math.isfinite(bound):
            raise ValueError('Unrepresentable decision crossing; no rounded feasibility claim')
        if a > 0:
            if bound > lower:
                lower, lower_open, lower_ids = bound, strict, [row['id']]
            elif bound == lower:
                lower_open |= strict
                lower_ids.append(row['id'])
        else:
            if bound < upper:
                upper, upper_open, upper_ids = bound, strict, [row['id']]
            elif bound == upper:
                upper_open |= strict
                upper_ids.append(row['id'])
    feasible = not impossible and (lower < upper or lower == upper and not (lower_open or upper_open))
    return dict(feasible=feasible, lower=lower, lower_exclusive=lower_open,
        upper=None if math.isinf(upper) else upper, upper_exclusive=upper_open,
        lower_bound_ids=lower_ids, upper_bound_ids=upper_ids, impossible_constant_ids=impossible,
        constraint_count=len(constraints), parameter_update_applied=False,
        scope='Only this frozen affine readout direction, not all parameters or all directions')
