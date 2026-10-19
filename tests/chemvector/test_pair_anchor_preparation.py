"""Synthetic saved-plan tests; never access the fixed58 corpus."""

from copy import deepcopy

import pytest

from chemvector.training.curriculum import pair_anchor_augmentation as aug
from tests.chemvector.test_pair_anchor_augmentation import fixture
from tools import prepare_pair_anchor_augmentation as prep


def scalar_fixture(monkeypatch):
    args = fixture()
    # Include both a radius failure and a crossing nonendpoint in the ribbon.
    args[0][0, 0] += 4 / 768
    views = aug.plan_views(*args)
    chosen = args[-1].tolist()
    rows = [dict(row_id=rid, candidates=args[4][rid].tolist(),
                 owner_targets=args[2][args[4][rid]].tolist()) for rid in chosen]
    case = dict(sampled_rows=rows, views=[])
    for view in views:
        kept = view['rows'].tolist()
        flags = aug.other_nodes_in_ribbon(view['anchors'], args[4][view['rows']], args[2], args[3])
        case['views'].append(dict(offset_pixels=view['offset_pixels'], row_ids=kept,
            guard_reason_row_ids={reason: [rid for rid in chosen if view['exclusion_masks'][reason][rid]] for reason in prep.REASONS},
            ribbon_other_node_row_ids=[rid for rid, flag in zip(kept, flags.tolist()) if flag]))
    geometry = dict(target_anchors=args[3].tolist(), candidates=[dict(slot=i, geometry=point.tolist()) for i, point in enumerate(args[0])])
    monkeypatch.setattr(prep.wiring.trace, 'selected_candidates', lambda geo, _: geo['candidates'])
    return case, (None, None, None, geometry, None, None)


def test_independent_scalar_replay_matches_tensor_guards(monkeypatch):
    case, context = scalar_fixture(monkeypatch)
    prep.check_scalar_geometry(case, context)


@pytest.mark.parametrize('change', ['kept_rows', 'reasons', 'ribbon', 'owner', 'offset'])
def test_scalar_replay_rejects_corrupted_rows_reasons_or_lineage(monkeypatch, change):
    case, context = scalar_fixture(monkeypatch)
    if change == 'kept_rows': case['views'][0]['row_ids'].pop()
    elif change == 'reasons': case['views'][0]['guard_reason_row_ids']['outside_image'].append(0)
    elif change == 'ribbon': case['views'][0]['ribbon_other_node_row_ids'].clear()
    elif change == 'owner': case['sampled_rows'][0]['owner_targets'][0] = 1
    else: case['views'][1]['offset_pixels'] = [100., 100.]
    with pytest.raises(ValueError): prep.check_scalar_geometry(case, context)


def test_counts_keep_nonedge_separate_from_types_and_po():
    rows = [dict(presence=True, kind='SINGLE', po=True), dict(presence=True, kind='DOUBLE', po=True),
            dict(presence=False, kind=None, po=False), dict(presence=True, kind='AROMATIC', po=False)]
    before = deepcopy(rows)
    assert prep.counts(rows) == dict(rows=4, positive=3, negative=1, SINGLE=1, DOUBLE=1, TRIPLE=0, AROMATIC=1, PO_SINGLE=1, PO_DOUBLE=1)
    assert rows == before
    assert prep.counts([])['negative'] == 0


def test_inventory_uses_chemical_not_display_bond_type():
    graph = dict(atoms=[dict(element='P'), dict(element='O'), dict(element='C')],
                 bonds=[dict(bond_id=4, begin_atom=0, end_atom=1, chemical_type='DOUBLE', display_type='SINGLE'),
                        dict(bond_id=5, begin_atom=0, end_atom=2, chemical_type='AROMATIC', display_type='DOUBLE')])
    assert prep.inventory(graph) == [dict(bond_id=4, kind='DOUBLE', presence=True, po=True),
                                     dict(bond_id=5, kind='AROMATIC', presence=True, po=False)]


def test_resource_defaults_do_not_enable_gpu_or_training():
    assert prep.runtime.ResourceBudget(120).gpu_devices == ()
    assert prep.OUTPUT != prep.POSTCHECK
    assert prep.MANIFEST_SHA == 'f5e3a7163eef5c49118ad78dcf05778f4ad889dc9b6389097ef20896b79e2111'
