"""Compare saved CPU/historical candidates by grid identity, not slot numbers.

Descriptive only: this never remaps training labels or relaxes a parity gate.
"""

from collections import Counter
import math


def compare(geometry, old_logits, snapshot):
    slots = [int(s) for s in snapshot['slots']]
    old = geometry['candidates']
    fresh = [dict(slot=s, grid_id=int(snapshot['grid_ids'][s]),
                  geometry=snapshot['geometry'][s].tolist(), raw_score=float(snapshot['raw_scores'][s]),
                  logit=float(snapshot['logits'][s])) for s in slots]
    if (not old or not fresh or len(set(slots)) != len(slots)
            or len({c['slot'] for c in old}) != len(old)
            or set(old_logits) != {str(c['slot']) for c in old}
            or geometry['image_size'] != 768 or geometry['match_radius_px'] != 6):
        raise ValueError('Complete original768/6px candidate and score identities required')
    prior = [dict(c, logit=old_logits[str(c['slot'])]) for c in old]
    for rows in (prior, fresh):
        if (len({r['grid_id'] for r in rows}) != len(rows)
                or any(not math.isfinite(v) for r in rows
                       for v in [r['logit'], r['raw_score']] + r['geometry'])):
            raise ValueError('Unique grids and finite saved numbers required')
    a, b = ({r['grid_id']: r for r in rows} for rows in (prior, fresh))
    common = sorted(a.keys() & b.keys())
    removed, added = sorted(a.keys()-b.keys()), sorted(b.keys()-a.keys())
    reordered = [dict(grid_id=g, old_slot=a[g]['slot'], new_slot=b[g]['slot'])
                 for g in common if a[g]['slot'] != b[g]['slot']]
    def plateau(rows):
        score = min(r['raw_score'] for r in rows)
        tied = [r for r in rows if r['raw_score'] == score]
        return dict(cutoff_score=score, retained_cutoff_ties=len(tied),
                    tied_grid_rows=dict(Counter(str(r['grid_id']//192) for r in tied)))
    old_selected = sorted(g for g, r in a.items() if r['logit'] >= -.25)
    new_selected = sorted(g for g, r in b.items() if r['logit'] >= -.25)
    anchors = geometry['target_anchors']
    def relations(rows):
        return sorted([r['grid_id'], i] for r in rows for i, xy in enumerate(anchors)
                      if math.dist(r['geometry'][:2], xy)*768 <= 6)
    def changed(rows):
        return [dict(grid_id=r['grid_id'], slot=r['slot'], raw_score=r['raw_score'], logit=r['logit'],
            nearest_target_px=min(math.dist(r['geometry'][:2], xy)*768 for xy in anchors)) for r in rows]
    changes = changed([a[g] for g in removed]) + changed([b[g] for g in added])
    return dict(old_candidates=len(a), new_candidates=len(b), common_grids=len(common),
        removed_grid_ids=removed, added_grid_ids=added, reordered_common_candidates=reordered,
        old_cutoff_plateau=plateau(prior), new_cutoff_plateau=plateau(fresh),
        all_changed_candidates_at_own_cutoff=all(a[g]['raw_score'] == min(r['raw_score'] for r in prior) for g in removed)
            and all(b[g]['raw_score'] == min(r['raw_score'] for r in fresh) for g in added),
        old_selected_grid_ids=old_selected, new_selected_grid_ids=new_selected,
        selected_grid_ids_equal=old_selected == new_selected,
        eligible_grid_target_relations_equal=relations(prior) == relations(fresh),
        changed_candidate_details=changes,
        all_changed_candidates_outside_all_target_radii=all(r['nearest_target_px'] > 6 for r in changes),
        max_common_logit_change=max((abs(a[g]['logit']-b[g]['logit']) for g in common), default=None),
        max_common_xy_change_px=max((math.dist(a[g]['geometry'][:2], b[g]['geometry'][:2])*768 for g in common), default=None),
        same_complete_candidate_grid_set=not removed and not added,
        may_reuse_historical_slot_labels=not removed and not added and not reordered,
        parameter_direction_established=False, training_authorized=False,
        limitations=['Selected-grid equality does not imply equal full candidate context or selector features',
            'Cutoff plateau evidence does not isolate all CPU/GPU numerical effects on selector scores',
            'Only saved retained candidates are available, not the entire dense peak population'])
