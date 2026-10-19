"""Reviewed-coordinate subset checks; not labels, inference, or training.

The original float32 Hungarian/6px matcher and -0.25 selector cutoff remain
unchanged. Removing an outside-radius point need not be matcher-monotone, so
every saved phase is checked again rather than inheriting the old all-pair test.
"""

from copy import deepcopy
import json
import math

from tools import selector_coordinate_contract as coordinates
from tools import selector_node_groups as groups
from tools import selector_node_semantics as semantics


def serialized_equal(left, right):
    """Compare persisted JSON values, normalizing tuples without rounding."""
    return json.loads(json.dumps(left, allow_nan=False)) == json.loads(json.dumps(right, allow_nan=False))


def stable_subset(proposal, pair_reviews):
    """Partition exactly the original pairs; never synthesize new negatives."""
    pairs = proposal['proposed_pairs']
    identity = lambda row: (row['fragment_slot'], row['owner_target'])
    by_id = {identity(p): p for p in pairs}
    reviews = {identity(r): r for r in pair_reviews}
    if len(by_id) != len(pairs) or len(reviews) != len(pair_reviews) or set(by_id) != set(reviews):
        raise ValueError('Pair/review identity must be unique and exactly aligned')
    if len({p['fragment_slot'] for p in pairs}) != len(pairs):
        raise ValueError('A fragment cannot receive two owners')
    if proposal.get('training_authorized') is not False or proposal.get('requires_review') is not True:
        raise ValueError('Unapproved review-only proposal required')
    stable, excluded = [], []
    for pair in pairs:
        review = reviews[identity(pair)]
        if (review.get('training_authorized') is not False or review.get('requires_review') is not True
                or pair.get('training_authorized') is not False or pair.get('requires_review') is not True
                or type(review['coordinate_stable']) is not bool
                or review['coordinate_stable'] != (not review['review_reasons'])
                or set(review['old_kinds']) != set(pair['kinds'])):
            raise ValueError('Review flags or evidence differ from original pair')
        if review['coordinate_stable']:
            if (review['source_evidence']['possible_owners'] != [pair['owner_target']]
                    or set(review['source_evidence']['kinds']) != set(pair['kinds'])
                    or review['unquantized_members']):
                raise ValueError('Stable review contains contradictory coordinate evidence')
            stable.append(deepcopy(pair))
        else:
            excluded.append(dict(fragment_slot=pair['fragment_slot'], owner_target=pair['owner_target'],
                pair=deepcopy(pair), coordinate_review=deepcopy(review)))
    subset = deepcopy(proposal)
    subset['proposed_pairs'] = stable
    subset['subset_policy'] = 'Original pair AND coordinate_stable; no new pair, threshold, or target'
    return dict(stable_proposal=subset, excluded_pairs=excluded,
                stable_pair_reviews=[deepcopy(reviews[identity(p)]) for p in stable],
                training_authorized=False, requires_review=True)


def removal_check(geometry, subset, record, layers, glyphs):
    """Re-run original matching and retain explicit per-node protections.

    Source-glyph evidence uses inverse projection (767), not the old 768
    visualization. Spatial matching still uses the unchanged 768 metric.
    """
    logits = record['metrics']['logits']
    raw = {c['slot']: c for c in geometry['candidates']}
    if ({int(s) for s in logits} != set(raw) or len(logits) != len(raw)
            or any(not math.isfinite(value) for value in logits.values())):
        raise ValueError('Exact finite candidate/logit coverage required')
    if geometry['image_size'] != 768 or geometry['match_radius_px'] != 6:
        raise ValueError('Original 768/6px geometry required')
    proposal = subset['stable_proposal']
    selected = {int(s) for s, value in logits.items() if value >= -.25}
    check = groups.removal_check(geometry, proposal, record, layers)
    removed = set(check['removed_fragment_slots'])
    held = {row['pair']['fragment_slot'] for row in subset['excluded_pairs']}
    if removed & held:
        raise ValueError('Held coordinate-sensitive fragment was removed')
    before = groups.coverage(geometry, selected, layers)
    after = groups.coverage(geometry, selected - removed, layers)
    matched = {s: t for s, t in before['mapping'].items() if t >= 0}
    source_glyph_matched = sorted(s for s in matched if any(
        semantics.inside(coordinates.source_xy(raw[s]['geometry'][:2]), glyph['bbox']) for glyph in glyphs))
    text_targets = {a['atom_id'] for a in layers['atoms'] if a['visible_text']}
    correct_text = sorted(text_targets & set(before['matched_target_ids']))
    lost_text = sorted(set(correct_text) - set(after['matched_target_ids']))
    removed_correct = sorted(set(matched) & removed)
    lost_relations = [dict(slot=s, target_id=t, after_target_id=after['mapping'].get(s, -1))
                     for s, t in sorted(matched.items()) if after['mapping'].get(s, -1) != t]
    changed_replacements = [row for row in lost_relations if raw[row['slot']]['raw_target'] < 0]
    protected_slots = sorted(c['slot'] for c in proposal['candidates'] if c['memberships'])
    removed_eligible = sorted(set(protected_slots) & removed)
    check.update(original_correct_slots=sorted(matched), removed_correct_slots=removed_correct,
        source_projected_glyph_matched_slots=source_glyph_matched,
        removed_source_projected_glyph_matched_slots=sorted(set(source_glyph_matched) & removed),
        original_visible_text_target_ids=correct_text, lost_visible_text_target_ids=lost_text,
        lost_original_assignment_relations=lost_relations,
        changed_valid_replacement_relations=changed_replacements,
        protected_spatially_eligible_slots=protected_slots,
        removed_spatially_eligible_slots=removed_eligible,
        held_sensitive_selected_slots=sorted(held & selected),
        all_protections_preserved=bool(check['original_matcher_nonregression']
            and not removed_correct and not lost_text and not lost_relations and not removed_eligible
            and not check['removed_correct_glyph_slots'] and not check['removed_valid_replacement_slots']),
        training_authorized=False, requires_review=True)
    return check
