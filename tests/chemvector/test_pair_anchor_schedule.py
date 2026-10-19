"""Small synthetic metadata, no corpus/model/optimizer."""

from copy import deepcopy

import pytest

from chemvector.training.curriculum.pair_anchor_augmentation import OFFSETS
from tools.pair_anchor_schedule import project, counts


def cases():
    result = []
    for i in range(2):
        rows = [] if i == 0 else [dict(row_id=9, presence=True, kind='DOUBLE', bond_id=2, po=True),
                                  dict(row_id=3, presence=False, kind=None, bond_id=None, po=False)]
        views = [dict(offset_pixels=list(offset), row_ids=[r['row_id'] for r in rows] if v != 1 else [],
                      counts=counts(rows if v != 1 else [])) for v, offset in enumerate(OFFSETS)]
        result.append(dict(sample_key=str(i), sampled_rows=rows, views=views,
                           source_target_bonds=[dict(bond_id=2, presence=True, kind='DOUBLE', po=True)]))
    return result


def test_empty_extra_does_not_retry_and_empty_base_consumes_visit():
    records = cases(); before = deepcopy(records)
    result = project(records, attempts=4, updates=2)
    assert [r['view_index'] for r in result['steps']] == [1, 1, 2, 2]
    assert result['summary']['empty_base_attempts'] == 2
    assert result['summary']['empty_extra_on_nonempty_base'] == 1
    assert result['summary']['simulated_updates'] == 2
    assert result['steps'][-1]['augmented_sampled_positions'] == [0, 1]
    assert records == before


def test_update_cap_stops_without_completing_sweep_or_hiding_targets():
    result = project(cases(), attempts=16, updates=1)
    assert result['summary']['attempts'] == 2
    assert result['summary']['target_count'] == 2 and result['summary']['unreachable_targets'] == 1
    assert result['summary']['never_augmented_sampled_rows']['positive'] == 1


def test_full_visit_cycle_is_finite_metadata_only_not_budget_extension():
    result = project(cases(), attempts=16, updates=8)
    assert all(v == list(range(1, 9)) for v in result['summary']['distinct_scheduled_views_per_image'].values())
    assert result['summary']['actual_optimizer_steps'] == 0
    assert not result['summary']['training_authorized']
    assert result['summary']['augmented_observations']['positive'] == 7
    assert result['summary']['never_augmented_sampled_rows']['rows'] == 0


def test_bad_cached_subset_rejected():
    records = cases(); records[0]['views'][1]['row_ids'] = [99]
    with pytest.raises(ValueError): project(records, attempts=4, updates=2)
