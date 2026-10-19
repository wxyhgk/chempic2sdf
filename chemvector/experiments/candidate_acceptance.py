"""One-interior-node hypothesis gate. Offline regression only, not deployment.

Consumes predicted queues/scenes/logits only. Chemical validity is a necessary
check, never evidence that the molecular identity is correct.
"""

from copy import deepcopy
from itertools import combinations
import math

from chemvector.experiments.single_candidate_replay import (
    compare_scenes, first_review_candidate, scene_state,
)
from chemvector.training.curriculum.validation_truth import canonical_molblock

PAIR_THRESHOLD = .85
RECORD_FIELDS = {'prediction', 'trace', 'pairs', 'presence_logits', 'chemical_logits',
                 'postprocessed_chemical_classes'}


def _pair_logits(record):
    if set(record) != RECORD_FIELDS:
        raise ValueError('Only the saved prediction-only record schema is accepted')
    lineage = record['trace']['compiled_candidate_indices']
    pairs, logits = record['pairs'], record['presence_logits']
    if len(pairs) != len(logits) or any(not math.isfinite(v) for v in logits):
        raise ValueError('Aligned finite pair logits required')
    result = {}
    for pair, logit in zip(pairs, logits):
        if (len(pair) != 2 or any(type(s) is not int for s in pair) or pair[0] >= pair[1]
                or tuple(pair) in result or not set(pair) <= set(lineage)):
            raise ValueError('Unique ordered pairs with exact scene lineage required')
        result[tuple(pair)] = logit
    if set(result) != set(combinations(sorted(lineage), 2)):
        raise ValueError('Complete all-pairs prediction required')
    return result


def assess_candidate(queue, before, after):
    """Accept only the stated two-sided hypothesis; do not search alternatives."""
    slot = first_review_candidate(queue)
    request = queue['requests'][0]
    endpoints = [request['orphan_slot'], request['partner_slot']]
    if (len(set(endpoints)) != 2 or any(type(s) is not int for s in endpoints)
            or not set(endpoints) <= set(queue['selected_slots'])):
        raise ValueError('Two distinct original requested endpoints required')
    # Unavailable compilation is a refusal, not a reason to invent a scene.
    if 'scene_before_fallback' not in after.get('trace', {}):
        return dict(accepted=False, selected='original', reason='candidate_scene_unavailable')
    old_logits, new_logits = _pair_logits(before), _pair_logits(after)
    old_atoms, old_edges = scene_state(before['trace'])
    _, new_edges = scene_state(after['trace'])
    if sorted(old_atoms) != queue['selected_slots']:
        raise ValueError('Stale review queue does not describe the baseline')
    delta = compare_scenes(before['trace'], after['trace'])
    expected = {tuple(sorted([slot, endpoint])) for endpoint in endpoints}
    added_edges = set(new_edges) - set(old_edges)
    cutoff = math.log(PAIR_THRESHOLD / (1 - PAIR_THRESHOLD))
    two_sided = all(pair in new_logits and new_logits[pair] >= cutoff for pair in expected)
    text = lambda r: {v['candidate']: v['text'] for v in r['trace']['ocr']}
    roles = lambda r: {v['candidate']: v['text'] for v in r['trace']['roles']}
    before_text, after_text = text(before), text(after)
    before_roles, after_roles = roles(before), roles(after)
    checks = dict(
        exactly_one_new_slot=delta['added_slots'] == [slot] and not delta['removed_slots'],
        old_atom_fields_preserved=not delta['changed_old_atoms'],
        old_edges_preserved=not delta['removed_edges'] and not delta['changed_old_edges'],
        old_ocr_preserved=all(after_text.get(s) == t for s, t in before_text.items()),
        old_text_routing_preserved=all(after_roles.get(s) == t for s, t in before_roles.items()),
        two_requested_pair_scores_supported=two_sided,
        exactly_two_requested_new_scene_edges=added_edges == expected,
        compiled_without_error_or_fallback=(after['prediction']['error'] is None
            and after['prediction']['fallback'] is False
            and canonical_molblock(after['prediction']['molblock']) is not None))
    accepted = all(checks.values())
    return dict(accepted=accepted, selected='candidate' if accepted else 'original',
        checks=checks, failed_checks=[k for k, passed in checks.items() if not passed],
        requested_pairs=[list(pair) for pair in sorted(expected)],
        requested_logits=[new_logits.get(pair) for pair in sorted(expected)],
        old_pair_rows=len(old_logits), pair_threshold=PAIR_THRESHOLD,
        proves_identity_correct=False, deployment_authorized=False)


def select_prediction(queue, before, after):
    """Return a detached copy; never overwrite either saved prediction."""
    decision = assess_candidate(queue, before, after)
    return decision, deepcopy(after if decision['accepted'] else before)
