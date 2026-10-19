"""Do not confuse a suppressed GT cell with every valid alternative missing."""

import pytest

from tools.review_selector_dense_observation import neighborhood_category


def node(*, peak=False, threshold=True, topk=False, active=False):
    return dict(grid_id=0, direct_target_distance_px=4., raw_active=active,
                is_local_peak=peak, passes_threshold=threshold, in_topk_before_threshold=topk)


@pytest.mark.parametrize('near,expected', [
    ([], 'no_local_peak_within_radius'),
    ([node()], 'no_local_peak_within_radius'),
    ([node(peak=True)], 'above_threshold_local_peak_excluded_by_topk'),
    ([node(peak=True, threshold=False)], 'local_peaks_below_threshold_and_all_outside_topk'),
    ([node(peak=True, threshold=False, topk=True)], 'local_peaks_below_threshold'),
])
def test_category_is_based_on_whole_neighborhood_not_gt_stage(near, expected):
    target = dict(cpu_still_missing=True, gt_cell=dict(first_excluding_stage='local_peak_suppressed'),
                  dense_cells_within_6px=near)
    assert neighborhood_category(target) == expected


@pytest.mark.parametrize('change', ['not_missing', 'active', 'contradictory_peak', 'outside', 'duplicate'])
def test_neighborhood_contract(change):
    target = dict(cpu_still_missing=True, dense_cells_within_6px=[node()])
    if change == 'not_missing': target['cpu_still_missing'] = False
    elif change == 'active': target['dense_cells_within_6px'][0]['raw_active'] = True
    elif change == 'contradictory_peak': target['dense_cells_within_6px'][0] = node(peak=True, topk=True)
    elif change == 'outside': target['dense_cells_within_6px'][0]['direct_target_distance_px'] = 7
    else: target['dense_cells_within_6px'].append(node())
    with pytest.raises(ValueError):
        neighborhood_category(target)
