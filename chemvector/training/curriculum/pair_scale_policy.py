"""Proposed scale-run selection rules, distinct from the immutable fixed58 rules."""


def _index(records):
    if not records:
        raise ValueError('Full fixed development records required')
    result = {}
    for row in records:
        if (row['key'] in result or row['domain'] not in ('full_molecule', 'curriculum')
                or any(type(row[k]) is not bool for k in ('exact', 'valid', 'fallback'))
                or row['exact'] and not row['valid']):
            raise ValueError('Invalid development metrics/cohort')
        result[row['key']] = row
    if {r['domain'] for r in records} != {'full_molecule', 'curriculum'}:
        raise ValueError('Both fixed development domains required')
    return result


def development_decision(baseline, current, *, complete_coverage):
    """Called at predeclared checkpoints, not on arbitrary sample prefixes."""
    if type(complete_coverage) is not bool:
        raise ValueError('Explicit exposure completion flag required')
    before, after = _index(baseline), _index(current)
    if before.keys() != after.keys() or any(before[k]['domain'] != after[k]['domain'] for k in before):
        raise ValueError('Cannot change fixed development cohort/domain')
    groups = {}
    for domain in ('full_molecule', 'curriculum'):
        keys = [k for k in before if before[k]['domain'] == domain]
        totals = [{field: sum(rows[k][field] for k in keys) for field in ('exact', 'valid', 'fallback')}
                  for rows in (before, after)]
        groups[domain] = dict(images=len(keys), baseline=totals[0], current=totals[1],
            rescued=sum(not before[k]['exact'] and after[k]['exact'] for k in keys),
            broken=sum(before[k]['exact'] and not after[k]['exact'] for k in keys))
    full, course = groups['full_molecule'], groups['curriculum']
    # Integer arithmetic: equality at 5pp is a stop, never rounding-dependent.
    severe = any((full['baseline'][k] - full['current'][k]) * 100 >= 5 * full['images']
                 for k in ('exact', 'valid'))
    reasons = []
    if severe:
        reasons.append('full_development_drop_at_least_5pp')
    if complete_coverage:
        if full['current']['exact'] <= full['baseline']['exact']:
            reasons.append('no_full_exact_gain')
        if full['current']['valid'] < full['baseline']['valid']:
            reasons.append('full_valid_regression')
        if full['current']['fallback'] > full['baseline']['fallback']:
            reasons.append('full_fallback_regression')
        if (course['baseline']['exact'] - course['current']['exact']) * 100 > 2 * course['images']:
            reasons.append('curriculum_exact_drop_over_2pp')
    eligible = complete_coverage and not reasons
    return dict(action='stop_severe_regression' if severe else
                ('complete_candidate_for_review' if eligible else 'complete_keep_initial') if complete_coverage
                else 'continue_to_next_fixed_checkpoint',
                reasons=reasons, groups=groups, complete_coverage=complete_coverage,
                candidate_eligible=eligible, deployment_authorized=False, independent_test_measured=False)
