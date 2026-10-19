"""Pure saved-plan exposure accounting. Counts are NOT real optimizer updates."""

from collections import Counter

import torch

from chemvector.training.curriculum.pair_anchor_wiring import PerImageOffsetCycle, sampled_positions
from tools.prepare_pair_anchor_augmentation import counts


def project(cases, *, attempts, updates, legacy=False):
    if (type(attempts) is not int or attempts < 1 or type(updates) is not int or updates < 1
            or not cases or type(legacy) is not bool):
        raise ValueError('Explicit finite positive simulation limits required')
    keys = [c['sample_key'] for c in cases]
    cycle = PerImageOffsetCycle(keys)
    observations = {key: {} for key in keys}
    visits, simulated_updates, steps = Counter(), 0, []
    total_base, total_extra = Counter(), Counter()
    for case in cases:
        rows = case['sampled_rows']
        ids = [r['row_id'] for r in rows]
        if len(set(ids)) != len(ids) or len(case['views']) != 9:
            raise ValueError('Original unique sampled rows and nine views required')
        for rid in ids: observations[case['sample_key']][rid] = Counter()
    for attempt in range(attempts):
        if simulated_updates == updates: break
        case = cases[attempt % len(cases)]
        key = case['sample_key']
        visit = cycle.next(key)
        index = 1 + attempt % 8 if legacy else visit['view_index']
        view = case['views'][index]
        chosen = torch.tensor([r['row_id'] for r in case['sampled_rows']], dtype=torch.long)
        kept = torch.tensor(view['row_ids'], dtype=torch.long)
        positions = sampled_positions(chosen, kept).tolist()
        extra = [case['sampled_rows'][p] for p in positions]
        if counts(extra) != view['counts']:
            raise ValueError('Saved view counts do not reconstruct from mapped labels')
        visits[key] += 1
        for row in case['sampled_rows']: observations[key][row['row_id']]['base'] += 1
        for rid in kept.tolist(): observations[key][rid]['view_' + str(index)] += 1
        total_base.update(counts(case['sampled_rows'])); total_extra.update(counts(extra))
        simulated_updates += bool(len(chosen))
        steps.append(dict(attempt=attempt, sample_key=key, visit=visit['visit'], view_index=index,
            offset_pixels=view['offset_pixels'], would_update=bool(len(chosen)),
            augmented_row_ids=kept.tolist(), augmented_sampled_positions=positions,
            augmented_counts=counts(extra)))
    per_case, unseen_base, unseen_extra, targets = [], [], [], []
    for case in cases:
        rows, key = [], case['sample_key']
        for row in case['sampled_rows']:
            observed = observations[key][row['row_id']]
            views = {str(i): observed['view_' + str(i)] for i in range(1, 9)}
            count = sum(views.values())
            rows.append(dict(**row, base_observations=observed['base'], augmented_observations=count,
                             augmented_view_counts=views))
            if not observed['base']: unseen_base.append(row)
            if not count: unseen_extra.append(row)
        by_bond = {r['bond_id']: r for r in rows if r['presence']}
        for target in case['source_target_bonds']:
            observed = by_bond.get(target['bond_id'])
            targets.append(dict(sample_key=key, **target, reachable=observed is not None,
                base_observations=observed['base_observations'] if observed else 0,
                augmented_observations=observed['augmented_observations'] if observed else 0))
        per_case.append(dict(sample_key=key, visits=visits[key], rows=rows))
    summary = dict(policy='legacy_global_attempt' if legacy else 'per_image_visit_v1',
        maximum_attempts=attempts, maximum_simulated_updates=updates, attempts=len(steps),
        simulated_updates=simulated_updates, empty_base_attempts=sum(not s['would_update'] for s in steps),
        empty_extra_on_nonempty_base=sum(s['would_update'] and not s['augmented_row_ids'] for s in steps),
        unique_images_visited=len(visits), unique_images_with_base=sum(bool(c['visits'] and c['rows']) for c in per_case),
        base_observations=total_base, augmented_observations=total_extra,
        never_base_sampled_rows=counts(unseen_base), never_augmented_sampled_rows=counts(unseen_extra),
        po_target_exposures=[t for t in targets if t['po']],
        target_count=len(targets), unreachable_targets=sum(not t['reachable'] for t in targets),
        distinct_scheduled_views_per_image={key: sorted({s['view_index'] for s in steps if s['sample_key'] == key}) for key in keys},
        actual_optimizer_steps=0, training_authorized=False)
    return dict(summary=summary, steps=steps, per_case=per_case, targets=targets)
