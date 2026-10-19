"""Saved-score diagnosis of missing node groups, never labels or predictions."""

import math

THRESHOLD = -.25
PHASES = ('initial', 'retention-200', 'control-200', 'retention-400')


def score_state(valid_slots, fragment_slots, logits):
    """Keep threshold and ordering failures separate; no winner tie-break."""
    valid, fragments = list(valid_slots), list(fragment_slots)
    if (not valid or len(set(valid)) != len(valid) or len(set(fragments)) != len(fragments)
            or set(valid) & set(fragments)
            or any(type(s) is not int or s < 0 or s not in logits for s in valid + fragments)
            or any(not math.isfinite(v) for v in logits.values())):
        raise ValueError('Distinct disjoint valid/fragment slots with finite scores required')
    best = max(logits[s] for s in valid)
    best_slots = sorted(s for s in valid if logits[s] == best)
    bad = max((logits[s] for s in fragments), default=None)
    retained = sorted(s for s in valid if logits[s] >= THRESHOLD)
    selected_bad = sorted(s for s in fragments if logits[s] >= THRESHOLD)
    second = sorted((logits[s] for s in valid), reverse=True)[1] if len(valid) > 1 else None
    lower = max((v for v in (bad, second) if v is not None), default=None)
    kind = ('no_stable_fragment_context' if bad is None else
            'fragment_ties_or_outranks_valid' if bad >= best else
            'valid_over_fragment')
    return dict(best_valid_logit=best, best_valid_slots=best_slots,
        best_fragment_logit=bad, valid_minus_fragment_margin=best-bad if bad is not None else None,
        best_valid_minus_threshold=best-THRESHOLD, valid_candidate_count=len(valid),
        selected_valid_slots=retained, selected_fragment_slots=selected_bad,
        node_state='missing' if not retained else 'single' if len(retained) == 1 else 'duplicate',
        comparison=kind, valid_fragment_threshold_separable=bad is None or best > bad,
        exactly_one_threshold_separable=lower is None or best > lower,
        exactly_one_threshold_interval=dict(lower_exclusive=lower, upper_inclusive=best),
        fixed_threshold_contract_pass=bool(retained) and not selected_bad,
        warning='Spatial candidates and reviewed GT-assisted fragments only; not predicted semantic correctness')


def inspect_case(geometry, subset, radius_case, phase_records):
    """Join existing fixed58 evidence by target IDs, never by nearest labels."""
    if (subset.get('training_authorized') is not False or subset.get('requires_review') is not True
            or radius_case.get('training_authorized') is not False
            or set(phase_records) != set(PHASES)):
        raise ValueError('Review-only subset and exact four saved phases required')
    proposal = subset['stable_proposal']
    raw = {c['slot']: c for c in geometry['candidates']}
    if len(raw) != len(geometry['candidates']):
        raise ValueError('Unique raw slots required')
    nodes = proposal['nodes']
    if [n['target_id'] for n in nodes] != list(range(len(geometry['target_anchors']))):
        raise ValueError('Complete ordered source atom identities required')
    membership = {s:[] for s in raw}
    for edge in radius_case['raw']['radius_reference']['eligible_edges']:
        membership[edge['slot']].append(edge['target_id'])
    fragments, held = {}, {}
    for pair in proposal['proposed_pairs']:
        slot, owner = pair['fragment_slot'], pair['owner_target']
        if (slot not in raw or membership[slot] or not 0 <= owner < len(nodes)
                or pair['training_authorized'] is not False or pair['requires_review'] is not True):
            raise ValueError('Stable fragment cannot be an eligible node or new label')
        if any(slot in slots for slots in fragments.values()):
            raise ValueError('Fragment cannot have multiple owners')
        fragments.setdefault(owner, []).append(slot)
    for pair in subset['excluded_pairs']:
        held.setdefault(pair['owner_target'], []).append(pair['fragment_slot'])
    if set(s for ss in held.values() for s in ss) & set(s for ss in fragments.values() for s in ss):
        raise ValueError('Sensitive fragments must remain outside stable evidence')
    logits = {}
    for phase, record in phase_records.items():
        values = record['metrics']['logits']
        if set(values) != {str(s) for s in raw} or any(not math.isfinite(z) for z in values.values()):
            raise ValueError('Complete finite saved scores required')
        logits[phase] = {int(s):z for s,z in values.items()}
        expected_selected = sorted(s for s,z in logits[phase].items() if z >= THRESHOLD)
        if expected_selected != radius_case['phases'][phase]['selected_slots']:
            raise ValueError('Saved phase selection changed')
    raw_missing = set(radius_case['raw']['radius_reference']['no_eligible_candidate_target_ids'])
    original_missing = set(radius_case['phases']['initial']['original']['before']['missing_target_ids'])
    selected_missing = set(radius_case['phases']['initial']['radius_reference']['before']['missing_target_ids'])
    focused = selected_missing - raw_missing
    metric_only = original_missing - selected_missing
    result = []
    for node in nodes:
        target = node['target_id']
        slots = sorted(s for s, members in membership.items() if target in members)
        if slots != sorted(node['possible_slots']):
            raise ValueError('Saved node group differs from radius-eligibility evidence')
        if not slots:
            continue
        if any(len(membership[s]) != 1 for s in slots) or node['status'] != 'reviewable':
            raise ValueError('This fixed cohort must have unambiguous reviewed node membership')
        own_fragments = sorted(fragments.get(target, []))
        states = {phase:score_state(slots,own_fragments,logits[phase]) for phase in PHASES}
        for phase in PHASES:
            ref_missing = radius_case['phases'][phase]['radius_reference']['before']['no_eligible_candidate_target_ids']
            if (states[phase]['node_state']=='missing') != (target in ref_missing):
                raise ValueError('Node-state and saved selected eligibility disagree')
        if target in focused and states['initial']['selected_valid_slots']:
            raise ValueError('Focused target has selected eligible candidates')
        result.append(dict(target_id=target, initial_selector_missing=target in focused,
            initial_original_matcher_only_missing=target in metric_only,
            element=node['element'], visible_text=node['visible_text'], text=node['label_text'],
            valid_slots=slots, raw_positive_slots=node['raw_positive_slots'],
            protected_alternative_slots=node['alternative_slots'],
            stable_fragment_slots=own_fragments, held_sensitive_fragment_slots=sorted(held.get(target,[])),
            fragment_kinds=sorted({kind for p in proposal['proposed_pairs'] if p['owner_target']==target for kind in p['kinds']}),
            phases=states))
    if focused != {n['target_id'] for n in result if n['initial_selector_missing']}:
        raise ValueError('Focused target population changed')
    return dict(nodes=result, focused_target_ids=sorted(focused), raw_missing_target_ids=sorted(raw_missing),
        original_matcher_only_missing_ids=sorted(metric_only),
        training_authorized=False, requires_review=True, model_predictions_changed=False)
