"""Prediction-only orphan-node review requests; never automatic atom recovery.

A candidate is only queued for inspection. This module neither adds an atom nor
claims that a geometric two-segment bridge is a chemical bond sequence.
"""

import math
from statistics import median

CONFIG = dict(minimum_observed_edges=3, endpoint_span_in_bond_lengths=[1.5, 2.5],
              leg_length_range=[.5, 1.5], interior_fraction=[.25, .75],
              corridor_half_width_px=12., selected_clearance_px=6., maximum_requests=4)


def review_requests(proposals, edges):
    """Only active raw predictions, their selector scores, and predicted edges."""
    by_slot = {}
    for p in proposals:
        if (set(p) != {'slot', 'xy', 'keep_logit', 'selected'}
                or type(p['slot']) is not int or p['slot'] < 0 or p['slot'] in by_slot
                or type(p['selected']) is not bool or len(p['xy']) != 2
                or any(not math.isfinite(v) or not 0 <= v <= 768 for v in p['xy'])
                or not math.isfinite(p['keep_logit'])):
            raise ValueError('Unique finite prediction-only candidate records required')
        by_slot[p['slot']] = p
    selected = {s for s, p in by_slot.items() if p['selected']}
    neighbors = {s: set() for s in selected}
    pairs = set()
    lengths = []
    for edge in edges:
        if len(edge) != 2 or any(type(s) is not int for s in edge) or len(set(edge)) != 2 or not set(edge) <= selected:
            raise ValueError('Predicted edges must have distinct selected endpoints')
        pair = tuple(sorted(edge))
        if pair in pairs:
            raise ValueError('Duplicate edge')
        pairs.add(pair)
        a, b = pair
        length = math.dist(by_slot[a]['xy'], by_slot[b]['xy'])
        if length <= 0:
            raise ValueError('Coincident connected anchors')
        lengths.append(length)
        neighbors[a].add(b)
        neighbors[b].add(a)
    result = dict(action='review_only', graph_edits=0, selected_slots=sorted(selected),
                  orphan_slots=sorted(s for s in selected if not neighbors[s]), requests=[])
    if len(lengths) < CONFIG['minimum_observed_edges']:
        return dict(result, status='insufficient_predicted_length_reference', median_bond_length=None)
    unit = median(lengths)
    requests = []
    for orphan in result['orphan_slots']:
        a = by_slot[orphan]['xy']
        for partner in sorted(s for s in selected if neighbors[s]):
            b = by_slot[partner]['xy']
            span = math.dist(a, b)
            if not 1.5 * unit <= span <= 2.5 * unit:
                continue
            dx, dy = b[0] - a[0], b[1] - a[1]
            candidates = []
            for slot, p in by_slot.items():
                if p['selected']:
                    continue
                x, y = p['xy']
                u = ((x-a[0])*dx + (y-a[1])*dy) / span**2
                perpendicular = abs((x-a[0])*dy - (y-a[1])*dx) / span
                legs = [math.dist(a, p['xy']), math.dist(b, p['xy'])]
                clearance = min(math.dist(p['xy'], by_slot[s]['xy']) for s in selected)
                if (.25 <= u <= .75 and perpendicular <= 12 and clearance > 6
                        and all(.5 * unit <= length <= 1.5 * unit for length in legs)):
                    candidates.append(dict(slot=slot, keep_logit=p['keep_logit'], projection=u,
                                           perpendicular_px=perpendicular, leg_lengths_px=legs))
            if not candidates:
                continue
            candidates.sort(key=lambda r: (-r['keep_logit'], r['slot']))
            tied = len(candidates) > 1 and abs(candidates[0]['keep_logit'] - candidates[1]['keep_logit']) <= 1e-8
            requests.append(dict(orphan_slot=orphan, partner_slot=partner, span_px=span,
                candidates=candidates, first_review_slot=None if tied else candidates[0]['slot'],
                reason='tied_ranking_abstain' if tied else 'geometry_shortlist_then_existing_selector_rank'))
    if len(requests) > CONFIG['maximum_requests']:
        return dict(result, status='review_budget_exceeded', median_bond_length=unit,
                    proposed_request_count=len(requests))
    return dict(result, status='requests_ready' if requests else 'no_geometric_request',
                median_bond_length=unit, requests=requests)
