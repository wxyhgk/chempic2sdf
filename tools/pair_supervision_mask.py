"""Review-only pair supervision masks; no new positive labels or inference edits.

Two fixed policies: strict matched-endpoint supervision, and a conditional
diagnostic-only extension that keeps nonedges under explicit ownership sets.
Neither policy is wired into production training.
"""

from collections import Counter
import math

import torch

from chemvector.train_direct_atom_pair_bond import select_training_pairs
from tools import selector_missing_graph_trace as trace

POLICIES = ('matched_only', 'conditional_ownership')


def ownership_hypotheses(geometry, selected, mapping):
    """All radius-eligible owners plus an optional conservative spatial hypothesis.

    A local diagnostic correspondence is NOT a released atom match. When none
    is available, leave ownership unresolved, never assume background.
    """
    diagnostic = trace.spatial_correspondence(geometry, selected)
    by_slot = {r['slot']: r for r in diagnostic}
    owners = {}
    for candidate in selected:
        slot = candidate['slot']
        if mapping[slot] >= 0:
            owners[slot] = dict(target_ids=[mapping[slot]], source='original_match')
            continue
        eligible = {i for i, a in enumerate(geometry['target_anchors'])
                    if math.dist(candidate['geometry'][:2], a) * 768 <= 6}
        row = by_slot[slot]
        if row['accepted_for_diagnostic_only']:
            eligible.add(row['target_id'])
        source = ('radius_and_or_diagnostic_hypothesis' if eligible else 'unresolved')
        owners[slot] = dict(target_ids=sorted(eligible), source=source)
    return owners, diagnostic


def classify_pairs(mapping, labels, owners, bonds):
    """Preserve original positive values; distinguish negative from ignored.

    Conditional negatives are conditional on owner-set correctness, not newly
    certified negatives. An unresolved endpoint always forces abstention.
    """
    bond_set = {tuple(sorted((b['begin_atom'], b['end_atom']))) for b in bonds}
    rows = []
    for pair, positive in sorted(labels.items()):
        a, b = pair
        if type(positive) is not bool or a >= b or min(a, b) < 0:
            raise ValueError('Unique ordered pairs and boolean original labels required')
        ma, mb = mapping[a], mapping[b]
        matched_distinct = ma >= 0 and mb >= 0 and ma != mb
        is_original_bond = matched_distinct and tuple(sorted((ma, mb))) in bond_set
        if positive != is_original_bond:
            raise ValueError('Original presence disagrees with unchanged matched graph')
        for slot in pair:
            ids = owners[slot]['target_ids']
            if len(ids) != len(set(ids)) or any(type(t) is not int or t < 0 for t in ids):
                raise ValueError('Unique nonnegative possible owners required')
            if mapping[slot] >= 0 and ids != [mapping[slot]]:
                raise ValueError('Do not replace original matched endpoints')
        if positive:
            strict, conditional, reason = 'positive', 'positive', 'original_positive'
        elif matched_distinct:
            strict, conditional, reason = 'negative', 'negative', 'original_matched_nonedge'
        else:
            strict = 'ignore'
            aa, bb = set(owners[a]['target_ids']), set(owners[b]['target_ids'])
            if not aa or not bb:
                conditional, reason = 'ignore', 'unresolved_owner'
            elif aa & bb:
                conditional, reason = 'ignore', 'possible_same_atom'
            elif any(tuple(sorted((x, y))) in bond_set for x in aa for y in bb):
                conditional, reason = 'ignore', 'possible_true_bond'
            else:
                conditional, reason = 'negative', 'conditional_nonedge'
        rows.append(dict(pair=list(pair), original_presence=positive,
                         matched_only=strict, conditional_ownership=conditional, reason=reason))
    return rows


def sampling_capacity(positives, negatives):
    """Count capacity under the existing 6x / minimum48 sampler, not sampled IDs."""
    return min(negatives, max(48, 6 * max(positives, 1)))


def summarize(rows):
    positives = sum(r['original_presence'] for r in rows)
    negatives = len(rows) - positives
    result = dict(current=dict(positive=positives, negative=negatives, ignore=0,
                               negative_sampling_capacity=sampling_capacity(positives, negatives)))
    for policy in POLICIES:
        counts = Counter(r[policy] for r in rows)
        if counts['positive'] != positives:
            raise ValueError('A mask must preserve every original positive')
        result[policy] = {s: counts[s] for s in ('positive', 'negative', 'ignore')}
        capacity = sampling_capacity(positives, counts['negative'])
        result[policy].update(negative_sampling_capacity=capacity,
            negative_sampling_capacity_loss=result['current']['negative_sampling_capacity'] - capacity)
    result['conditional_ownership']['hypothesis_dependent_negatives'] = sum(
        r['reason'] == 'conditional_nonedge' for r in rows)
    return result


def inspect_case(geometry, record, layers):
    selected = trace.selected_candidates(geometry, record)
    mapping, labels = trace.replay_pair_labels(geometry, record, layers)
    owners, diagnostic = ownership_hypotheses(geometry, selected, mapping)
    rows = classify_pairs(mapping, labels, owners, layers['bonds'])
    references = {r['slot']: r['target_id'] for r in diagnostic if r['accepted_for_diagnostic_only']}
    gt_edges = {tuple(sorted((b['begin_atom'], b['end_atom']))): b['bond_id'] for b in layers['bonds']}
    reference_counts = {p: Counter() for p in ('current',) + POLICIES}
    for row in rows:
        a, b = row['pair']
        if a in references and b in references and references[a] != references[b]:
            key = tuple(sorted((references[a], references[b])))
            reference = 'bond' if key in gt_edges else 'nonedge'
            row['diagnostic_reference'] = reference
            row['diagnostic_bond_id'] = gt_edges.get(key)
        else:
            row['diagnostic_reference'] = 'unresolved'
            row['diagnostic_bond_id'] = None
        for policy in ('current',) + POLICIES:
            state = ('positive' if row['original_presence'] else 'negative') if policy == 'current' else row[policy]
            reference_counts[policy][row['diagnostic_reference'] + '_' + state] += 1
    return dict(sample_key=record['sample_key'], group=record['group'],
        selected_atoms=len(selected), original_matched_atoms=sum(mapping[c['slot']] >= 0 for c in selected),
        ownership_hypotheses=owners, diagnostic_correspondence=diagnostic, pairs=rows,
        summary=summarize(rows), diagnostic_reference_counts=reference_counts,
        training_authorized=False, requires_review=True, labels_or_predictions_changed=False)


def sample_with_mask(pair_indices, sample_indices, atoms, pair_targets, supervision_mask,
                     *, negative_ratio=6, minimum_negatives=48):
    """Prototype adapter: discard ignored rows BEFORE the existing sampler.

    Returns original row IDs. Only synthetic tests call this adapter in this
    audit; real58 results report pool and quota counts without stochastic draws.
    """
    count = len(pair_indices)
    if (supervision_mask.dtype != torch.bool or supervision_mask.shape != (count,)
            or supervision_mask.device != pair_indices.device):
        raise ValueError('Aligned boolean supervision mask required')
    if bool((pair_targets['presence'] & ~supervision_mask).any()):
        raise ValueError('Cannot mask an original positive')
    eligible = supervision_mask.nonzero().flatten()
    filtered = dict(pair_targets)
    for name in ('presence', 'chemical', 'structured_negative'):
        filtered[name] = pair_targets[name][eligible]
    local = select_training_pairs(pair_indices[eligible], sample_indices[eligible], atoms, filtered,
                                  negative_ratio=negative_ratio, minimum_negatives=minimum_negatives)
    return eligible[local]
