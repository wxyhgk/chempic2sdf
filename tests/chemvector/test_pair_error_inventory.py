from copy import deepcopy

import pytest

from tools import pair_error_inventory as inv


def target(tid, x, y=100, **extra):
    return dict(atom_id=tid, center=[x, y], element='C', text='', visible_text=False,
                formal_charge=0, isotope=0, radical_electrons=0, explicit_h_count=0,
                implicit_h_count=0, chiral_tag='CHI_UNSPECIFIED', **extra)


def atom(aid, x, y=100):
    return dict(id=aid, anchor=inv.adapted_anchor([x, y]), role='IMPLICIT', text=None,
                chemical=dict(element='C', formal_charge=0, isotope=0, radical_electrons=0,
                              explicit_h_count=0, canonical_chiral_tag='CHI_UNSPECIFIED'), incident_ports=[])


def bond(bid, a, b):
    return dict(bond_id=bid, begin_atom=a, end_atom=b, chemical_type='SINGLE',
                drawing=dict(direction='NONE'), stereo='STEREONONE')


def fixture():
    layers = dict(atoms=[target(i, 100 + i * 100) for i in range(4)],
                  bonds=[bond(0, 0, 1), bond(1, 1, 2), bond(2, 2, 3)])
    scene = dict(atom_objects=[atom(0, 100), atom(1, 200), atom(2, 400)], bond_objects=[
        dict(id=0, atom_refs=[0, 1], chemical_type='SINGLE', direction='NONE', stereo='STEREONONE')])
    record = dict(trace=dict(scene_before_fallback=scene, compiled_candidate_indices=[2, 5, 8]),
                  metrics=dict(candidate_label_sha256='same', text_sha256='same', correct_typed_bond_ids=[0]),
                  pairs=[[2, 5], [2, 8], [5, 8]], presence_logits=[4., -4., -4.],
                  chemical_logits=[[4., 0., 0., 0.]] * 3)
    coverage = dict(selected_atoms=3, unmatched_selected_atoms=0, reachable_positive_pairs=1)
    labels = [dict(row_id=0, label=True), dict(row_id=1, label=False)]
    return layers, [record, deepcopy(record)], coverage, labels


def test_target_adapter_uses_767_not_768():
    from chemvector.training.curriculum.data import _point
    for p in ([0, 767], [490.219560754508, 376.8720366944824], [123.4, 567.8]):
        assert inv.adapted_anchor(p) == list(_point(p))
    assert inv.adapted_anchor([767, 767]) == [1023, 1023]


@pytest.mark.parametrize('point', [[float('nan'), 1], [768, 0], [-1, 1], [1]])
def test_bad_source_coordinates(point):
    with pytest.raises(ValueError):
        inv.adapted_anchor(point)


def test_missing_atom_not_net_selected_count():
    layers, records, coverage, labels = fixture()
    result = inv.inventory(layers, records, coverage, labels)
    assert result['nodes'][2]['candidate_slot'] is None
    assert result['nodes'][2]['no_selected_within_radius_even_with_rounding']
    assert result['phases']['50']['counts'] == {'present_raw_type_correct': 1, 'missing_local_selected_endpoint': 2}
    assert len(result['phases']['50']['components']) == 2
    assert result['receipt_derived_mapping_complete']
    assert not result['original_float_matching_replayed']


def test_input_not_mutated():
    args = fixture()
    copy = deepcopy(args)
    inv.inventory(*args)
    assert args == copy


def test_ambiguous_duplicate_candidates_stay_unresolved():
    rows, _ = inv.correspondence([atom(0, 100), atom(1, 100)], [5, 8], [target(0, 100), target(1, 200)])
    assert all(r['diagnostic_target'] is None for r in rows)


def test_distant_target_not_force_assigned():
    rows, _ = inv.correspondence([atom(0, 150)], [5], [target(0, 100), target(1, 200)])
    assert rows[0]['diagnostic_target'] is None


def test_boundary_is_uncertain_until_receipts():
    a = atom(0, 100)
    a['anchor'][1] += 8
    rows, _ = inv.correspondence([a, atom(1, 200)], [5, 8], [target(0, 100), target(1, 200)])
    assert rows[0]['radius_status'] == 'quantization_boundary'
    mem, _ = inv.receipt_membership([[5, 8]], [dict(row_id=0, label=False)],
        dict(selected_atoms=2, unmatched_selected_atoms=0, reachable_positive_pairs=0), rows)
    assert mem[0]['status'] == 'matched'
    assert mem[0]['target_id'] == 0


def test_exhaustive_receipts_identify_unmatched_slot():
    rows, _ = inv.correspondence([atom(0, 100), atom(1, 200), atom(2, 300)], [2, 5, 8],
                                [target(i, 100 + i * 100) for i in range(3)])
    mem, _ = inv.receipt_membership([[2, 5], [2, 8], [5, 8]], [dict(row_id=0, label=False)],
        dict(selected_atoms=3, unmatched_selected_atoms=1, reachable_positive_pairs=0), rows)
    assert mem[-1]['status'] == 'unmatched'
    assert mem[-1]['target_id'] is None


def test_unsampled_node_not_automatically_unmatched():
    rows, _ = inv.correspondence([atom(0, 100), atom(1, 200), atom(2, 300)], [2, 5, 8],
                                [target(i, 100 + i * 100) for i in range(3)])
    mem, _ = inv.receipt_membership([[2, 5], [2, 8], [5, 8]], [dict(row_id=0, label=False)],
        dict(selected_atoms=3, unmatched_selected_atoms=0, reachable_positive_pairs=0), rows)
    assert mem[-1]['status'] == 'unresolved'


def test_missing_edge_and_wrong_type_are_both_recorded():
    layers, records, coverage, labels = fixture()
    for r in records:
        r['trace']['scene_before_fallback']['bond_objects'] = []
        r['presence_logits'][0] = 0
        r['chemical_logits'][0] = [0, 0, 0, 4]
        r['metrics']['correct_typed_bond_ids'] = []
    result = inv.inventory(layers, records, coverage, labels)
    b = result['phases']['50']['target_bonds'][0]
    assert b['status'] == 'missing_edge' and b['raw_type_mismatch']
    assert b['prediction']['sampled_label'] is True


def test_scene_type_is_separate_from_raw_type():
    args = fixture()
    for r in args[1]:
        r['trace']['scene_before_fallback']['bond_objects'][0]['chemical_type'] = 'AROMATIC'
    b = inv.inventory(*args)['phases']['50']['target_bonds'][0]
    assert b['prediction']['scene_type'] == 'AROMATIC'
    assert b['status'] == 'present_raw_type_correct'


@pytest.mark.parametrize('field', ['candidate_label_sha256', 'text_sha256'])
def test_changed_lineage_rejected(field):
    args = fixture()
    args[1][1]['metrics'][field] = 'changed'
    with pytest.raises(ValueError, match='lineage'):
        inv.inventory(*args)


def test_metric_mismatch_is_not_silently_accepted():
    args = fixture()
    args[1][0]['metrics']['correct_typed_bond_ids'] = []
    with pytest.raises(ValueError, match='cross-check'):
        inv.inventory(*args)


def test_duplicate_sampler_row_rejected():
    args = fixture()
    args[3].append(args[3][0])
    with pytest.raises(ValueError, match='sampled row'):
        inv.inventory(*args)


def test_known_label_contradiction_rejected():
    args = fixture()
    args[3][0]['label'] = False
    args[3][1]['label'] = True
    with pytest.raises(ValueError, match='contradicts'):
        inv.inventory(*args)


def test_hn_nh_layout_not_chemical_error():
    t, a = target(0, 100), atom(0, 100)
    t.update(element='N', text='HN', visible_text=True, implicit_h_count=1)
    a.update(role='TEXT', text={'value': 'NH'})
    a['chemical'].update(element='N', explicit_h_count=1)
    assert inv.literal_text_status(t, a) == 'hydrogen_layout_equivalent_not_element_error'


def test_carbon_without_text_is_role_miss_not_wrong_element():
    t, a = target(0, 100), atom(0, 100)
    t.update(text='C', visible_text=True)
    assert inv.literal_text_status(t, a) == 'visible_text_not_routed_to_ocr'
    a.update(role='TEXT', text={'value': 'F'})
    a['chemical']['element'] = 'F'
    assert inv.literal_text_status(t, a) == 'literal_mismatch'


def test_stereo_annotation_absence_not_conflated_with_typed_bond():
    args = fixture()
    args[0]['atoms'][0]['chiral_tag'] = 'CHI_TETRAHEDRAL_CW'
    result = inv.inventory(*args)
    assert result['nodes'][0]['stereo_annotation_missing']
    assert result['phases']['50']['target_bonds'][0]['status'] == 'present_raw_type_correct'
