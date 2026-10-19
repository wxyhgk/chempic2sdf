"""Saved-coordinate matching comparison, not inference or training labels.

Reuse the previously tested radius_assignment comparator. Original Hungarian
then gate metrics remain separate and authoritative. Freeze the BEFORE cost
matrix for removal comparisons; also measure shape-dependent cdist changes.
"""

import math

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from tools.audit_curriculum_v02_selector_matching import radius_assignment

RADIUS = 6
IMAGE_SIZE = 768


def _slots(values):
    values = list(values)
    if any(type(s) is not int or s < 0 for s in values) or len(set(values)) != len(values):
        raise ValueError('Unique nonnegative integer slots required')
    return sorted(values)


@torch.no_grad()
def distance_matrix(geometry, slots):
    """Identical float32 cdist path to the historical matcher, no model call."""
    if geometry['image_size'] != IMAGE_SIZE or geometry['match_radius_px'] != RADIUS:
        raise ValueError('Fixed 768/6px contract required')
    raw = geometry['candidates']
    _slots(c['slot'] for c in raw)
    by_slot = {c['slot']: c for c in raw}
    chosen = _slots(slots)
    if not set(chosen) <= set(by_slot):
        raise ValueError('Unknown selected slot')
    anchors = geometry['target_anchors']
    points = [c['geometry'][:2] for c in raw]
    if any(len(p) != 2 or any(not math.isfinite(x) or not 0 <= x <= 1 for x in p)
           for p in points + anchors):
        raise ValueError('Finite normalized xy pairs required')
    if not chosen or not anchors:
        return np.zeros((len(chosen), len(anchors)), dtype=np.float32)
    xy = torch.tensor([by_slot[s]['geometry'][:2] for s in chosen], dtype=torch.float32)
    targets = torch.tensor(anchors, dtype=torch.float32)
    return (torch.cdist(xy, targets) * IMAGE_SIZE).numpy()


def coverage(cost, slots, layers, *, policy):
    """Record identities and gated-out assignments, not just match counts.

    There is deliberately no extra tie-break rule. Distinct optimal matchings
    can cover different target IDs and hence different chemical bonds.
    """
    slots = list(slots)
    if slots != _slots(slots):
        raise ValueError('Cost rows must follow sorted slots')
    cost = np.asarray(cost)
    n_targets = len(layers['atoms'])
    if (cost.shape != (len(slots), n_targets) or not np.isfinite(cost).all()
            or (cost < 0).any()
            or sorted(a['atom_id'] for a in layers['atoms']) != list(range(n_targets))):
        raise ValueError('Invalid cost/target identity contract')
    bonds = layers['bonds']
    if (len({b['bond_id'] for b in bonds}) != len(bonds)
            or any(b[k] not in range(n_targets) for b in bonds for k in ('begin_atom', 'end_atom'))):
        raise ValueError('Invalid bond identity contract')
    if policy == 'original':
        rows, cols = linear_sum_assignment(cost)
        pairs = list(zip(rows, cols))
    elif policy == 'radius_reference':
        pairs = radius_assignment(cost, RADIUS)
    else:
        raise ValueError('Unknown matching policy')
    trace = [dict(slot=slots[r], target_id=int(t), distance_px=float(cost[r, t]),
                  accepted=bool(cost[r, t] <= RADIUS)) for r, t in pairs]
    mapping = {s: -1 for s in slots}
    mapping.update((p['slot'], p['target_id']) for p in trace if p['accepted'])
    matched = sorted(t for t in mapping.values() if t >= 0)
    eligible = [dict(slot=slots[r], target_id=int(t), distance_px=float(cost[r, t]))
                for r, t in zip(*np.nonzero(cost <= RADIUS))]
    available = {p['target_id'] for p in eligible}
    return dict(mapping=mapping, assignment_before_gate=trace, eligible_edges=eligible,
        matched_target_ids=matched, missing_target_ids=sorted(set(range(n_targets)) - set(matched)),
        no_eligible_candidate_target_ids=sorted(set(range(n_targets)) - available),
        eligible_but_unassigned_target_ids=sorted(available - set(matched)),
        reachable_bond_ids=sorted(b['bond_id'] for b in bonds
            if b['begin_atom'] in matched and b['end_atom'] in matched),
        unmatched_selected=len(slots) - len(matched))


def difference(before, after):
    """Identity-sensitive differences, including equal-cardinality changes."""
    result = {}
    for noun, field in (('target', 'matched_target_ids'), ('bond', 'reachable_bond_ids')):
        old, new = set(before[field]), set(after[field])
        result['lost_' + noun + '_ids'] = sorted(old - new)
        result['gained_' + noun + '_ids'] = sorted(new - old)
    old = {(s, t) for s, t in before['mapping'].items() if t >= 0}
    new = {(s, t) for s, t in after['mapping'].items() if t >= 0}
    result.update(lost_relations=sorted(old - new), gained_relations=sorted(new - old),
                  matched_count_delta=len(new) - len(old))
    return result


def compare(geometry, slots, layers):
    slots = _slots(slots)
    cost = distance_matrix(geometry, slots)
    original = coverage(cost, slots, layers, policy='original')
    reference = coverage(cost, slots, layers, policy='radius_reference')
    result = dict(slots=slots, original=original, radius_reference=reference,
                  policy_difference=difference(original, reference))
    if original['mapping'] != reference['mapping']:
        result['distance_matrix_px'] = cost.tolist()
    return result


def compare_removal(geometry, slots, removed, layers):
    """Only remove isolated (zero eligible edge) rows of a frozen cost graph.

    Original AFTER always recomputes cdist, reproducing historic behavior.
    Reference AFTER uses the before matrix sliced to the retained rows. A
    separate reference recomputation exposes any numerical/shape effect.
    """
    slots, removed = _slots(slots), _slots(removed)
    if not set(removed) <= set(slots):
        raise ValueError('Removal must be a subset of selected slots')
    remaining = [s for s in slots if s not in removed]
    cost = distance_matrix(geometry, slots)
    if any(np.any(cost[i] <= RADIUS) for i, s in enumerate(slots) if s in removed):
        raise ValueError('Cannot remove any spatially eligible candidate')
    frozen_after = cost[[i for i, s in enumerate(slots) if s not in removed], :]
    recomputed = distance_matrix(geometry, remaining)
    old_before = coverage(cost, slots, layers, policy='original')
    old_after = coverage(recomputed, remaining, layers, policy='original')
    ref_before = coverage(cost, slots, layers, policy='radius_reference')
    ref_after = coverage(frozen_after, remaining, layers, policy='radius_reference')
    ref_recomputed = coverage(recomputed, remaining, layers, policy='radius_reference')
    shifts = [dict(slot=remaining[r], target_id=int(t), frozen_px=float(frozen_after[r, t]),
                   recomputed_px=float(recomputed[r, t]))
              for r, t in zip(*np.nonzero((frozen_after <= RADIUS) != (recomputed <= RADIUS)))]
    original_delta = difference(old_before, old_after)
    reference_delta = difference(ref_before, ref_after)
    eligible_before = {(e['slot'], e['target_id'], e['distance_px']) for e in ref_before['eligible_edges']}
    eligible_after = {(e['slot'], e['target_id'], e['distance_px']) for e in ref_after['eligible_edges']}
    result = dict(selected_slots=slots, removed_slots=removed, remaining_slots=remaining,
        original=dict(before=old_before, after=old_after, removal_difference=original_delta),
        radius_reference=dict(before=ref_before, after=ref_after, removal_difference=reference_delta),
        policy_difference_before=difference(old_before, ref_before),
        policy_difference_after=difference(old_after, ref_after),
        frozen_eligible_graph_identical=eligible_before == eligible_after,
        reference_relations_preserved=not reference_delta['lost_relations'],
        numerical_check=dict(max_abs_distance_change_px=float(np.max(np.abs(frozen_after - recomputed)))
                if recomputed.size else 0., eligibility_changes=shifts,
            recomputed_reference=ref_recomputed,
            difference_from_frozen=difference(ref_after, ref_recomputed)),
        uses_ground_truth=True, training_authorized=False, requires_review=True)
    if (old_before['mapping'] != ref_before['mapping'] or old_after['mapping'] != ref_after['mapping']
            or original_delta['lost_relations'] or reference_delta['lost_relations'] or shifts):
        result.update(distance_matrix_before_px=cost.tolist(),
                      distance_matrix_frozen_after_px=frozen_after.tolist(),
                      distance_matrix_recomputed_after_px=recomputed.tolist())
    return result
