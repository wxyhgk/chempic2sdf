"""CPU-only proposal partition/protection tests; no model or optimizer."""

from copy import deepcopy

import pytest

from tools import selector_node_groups as groups
from tools import selector_stable_subset as stable


def example():
    points = ((100, 100), (200, 100), (103, 100), (113, 100), (114, 100))
    raw = (0, 1, -1, -1, -1)
    geo = dict(image_size=768, match_radius_px=6,
        target_anchors=[[100/768, 100/768], [200/768, 100/768]],
        candidates=[dict(slot=i, geometry=[x/768, y/768, 0, 0, 0, 0, 0, 0], raw_target=raw[i])
                    for i, (x, y) in enumerate(points)])
    layers = dict(atoms=[dict(atom_id=i, element='C', text='', visible_text=i == 0) for i in range(2)],
        bonds=[dict(bond_id=0, begin_atom=0, end_atom=1)])
    glyphs = [dict(atom_id=0, text='r', bbox=(110, 95, 120, 105)),
              dict(atom_id=0, text='B', bbox=(95, 95, 106, 105))]
    proposal = groups.propose(geo, layers, glyphs, [], [])
    reviewed = []
    for pair in proposal['proposed_pairs']:
        flag = pair['fragment_slot'] == 3
        reviewed.append(dict(fragment_slot=pair['fragment_slot'], owner_target=pair['owner_target'],
            old_kinds=pair['kinds'], coordinate_stable=flag,
            review_reasons=[] if flag else ['source_projection_owner_not_confirmed'],
            source_evidence=dict(possible_owners=[0] if flag else [], kinds=pair['kinds']),
            unquantized_members=[], training_authorized=False, requires_review=True))
    return geo, layers, glyphs, proposal, reviewed


def record(geo, layers, selected):
    cov = groups.coverage(geo, selected, layers)
    return dict(metrics={**{k: v for k, v in cov.items() if k != 'mapping'},
        'logits': {str(c['slot']): 1. if c['slot'] in selected else -1. for c in geo['candidates']}})


def test_partition_is_exact_nonmutating_and_review_only():
    _, _, _, proposal, reviewed = example()
    original = deepcopy((proposal, reviewed))
    subset = stable.stable_subset(proposal, reviewed)
    assert [p['fragment_slot'] for p in subset['stable_proposal']['proposed_pairs']] == [3]
    assert [p['pair']['fragment_slot'] for p in subset['excluded_pairs']] == [4]
    assert subset['stable_proposal']['nodes'][0]['alternative_slots'] == [2]
    assert subset['training_authorized'] is False and subset['requires_review'] is True
    assert (proposal, reviewed) == original


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'owner', 'flag', 'kinds', 'permission',
                                       'unquantized', 'evidence_owner'])
def test_inconsistent_reviews_fail_closed(mutation):
    _, _, _, proposal, reviewed = example()
    if mutation == 'missing': reviewed.pop()
    elif mutation == 'duplicate': reviewed.append(deepcopy(reviewed[0]))
    elif mutation == 'owner': reviewed[0]['owner_target'] = 1
    elif mutation == 'flag': reviewed[0]['coordinate_stable'] = False
    elif mutation == 'kinds': reviewed[0]['old_kinds'] = ['unknown']
    elif mutation == 'permission': reviewed[0]['training_authorized'] = True
    elif mutation == 'unquantized': reviewed[0]['unquantized_members'] = [0]
    else: reviewed[0]['source_evidence']['possible_owners'] = [1]
    with pytest.raises(ValueError):
        stable.stable_subset(proposal, reviewed)


def test_duplicate_fragments_cannot_have_different_owners():
    _, _, _, proposal, reviewed = example()
    proposal['proposed_pairs'][1]['fragment_slot'] = 3
    proposal['proposed_pairs'][1]['owner_target'] = 1
    reviewed[1]['fragment_slot'] = 3
    reviewed[1]['owner_target'] = 1
    with pytest.raises(ValueError, match='two owners'):
        stable.stable_subset(proposal, reviewed)


def test_tuple_list_normalization_does_not_relax_values():
    assert stable.serialized_equal({'xy': (1.0, 2.0)}, {'xy': [1.0, 2.0]})
    assert not stable.serialized_equal([1.0], [1.000001])
    with pytest.raises(ValueError):
        stable.serialized_equal([float('nan')], [float('nan')])


def test_actual_subset_keeps_sensitive_slot_text_replacement_and_bond():
    geo, layers, glyphs, proposal, reviewed = example()
    subset = stable.stable_subset(proposal, reviewed)
    rec = record(geo, layers, {1, 2, 3, 4})
    original = deepcopy((geo, layers, glyphs, subset, rec))
    check = stable.removal_check(geo, subset, rec, layers, glyphs)
    assert check['removed_fragment_slots'] == [3]
    assert check['held_sensitive_selected_slots'] == [4]
    assert check['original_valid_replacement_slots'] == [2]
    assert check['source_projected_glyph_matched_slots'] == [2]
    assert check['original_visible_text_target_ids'] == [0]
    assert check['before']['reachable_bond_ids'] == check['after']['reachable_bond_ids'] == [0]
    assert check['all_protections_preserved']
    assert (geo, layers, glyphs, subset, rec) == original


def test_all_held_produces_identity_counterfactual():
    geo, layers, glyphs, proposal, reviewed = example()
    reviewed[0].update(coordinate_stable=False, review_reasons=['projection_sensitive'])
    subset = stable.stable_subset(proposal, reviewed)
    check = stable.removal_check(geo, subset, record(geo, layers, {0, 1, 3, 4}), layers, glyphs)
    assert not subset['stable_proposal']['proposed_pairs']
    assert check['before'] == check['after'] and check['all_protections_preserved']


@pytest.mark.parametrize('mutation', ['nan', 'missing_slot', 'radius', 'image_size'])
def test_fixed_logits_and_geometry_contract(mutation):
    geo, layers, glyphs, proposal, reviewed = example()
    subset = stable.stable_subset(proposal, reviewed)
    rec = record(geo, layers, {0, 1, 3})
    if mutation == 'nan': rec['metrics']['logits']['3'] = float('nan')
    elif mutation == 'missing_slot': del rec['metrics']['logits']['4']
    elif mutation == 'radius': geo['match_radius_px'] = 7
    else: geo['image_size'] = 767
    with pytest.raises(ValueError):
        stable.removal_check(geo, subset, rec, layers, glyphs)


def test_original_matcher_nonmonotonicity_is_reported_not_hidden():
    points = ((104, 100), (118, 100), (91, 100))
    geo = dict(image_size=768, match_radius_px=6,
        target_anchors=[[100/768, 100/768], [110/768, 100/768]],
        candidates=[dict(slot=i, geometry=[x/768, y/768], raw_target=0 if i == 0 else -1)
                    for i, (x, y) in enumerate(points)])
    layers = dict(atoms=[dict(atom_id=0, visible_text=True), dict(atom_id=1, visible_text=False)], bonds=[])
    memberships = groups.memberships(geo)
    proposal = dict(nodes=[], proposed_pairs=[dict(fragment_slot=1)],
        candidates=[dict(slot=c['slot'], raw_target=c['raw_target'], memberships=memberships[c['slot']],
                         evidence=dict(glyph_hits=[])) for c in geo['candidates']])
    subset = dict(stable_proposal=proposal, excluded_pairs=[])
    check = stable.removal_check(geo, subset, record(geo, layers, {0, 1, 2}), layers, [])
    assert check['lost_target_ids'] == [0]
    assert check['lost_visible_text_target_ids'] == [0]
    assert check['lost_original_assignment_relations'] == [dict(slot=0, target_id=0, after_target_id=1)]
    assert not check['all_protections_preserved']


def test_original_missing_nodes_are_not_repaired():
    geo, layers, glyphs, proposal, reviewed = example()
    subset = stable.stable_subset(proposal, reviewed)
    check = stable.removal_check(geo, subset, record(geo, layers, {3, 4}), layers, glyphs)
    assert not check['before']['matched_target_ids'] and not check['after']['matched_target_ids']
    assert not check['gained_target_ids']
    assert check['all_protections_preserved']


def test_saved_triple_bond_failure_keeps_correct_slot_but_changes_assignment():
    # Numeric fixture from the already inspected C#[13CH] saved geometry.
    # Slot4 (the other correct endpoint) was never selected in this phase.
    points = {1: (.5613572001457214, .48636168241500854),
              2: (.472441166639328, .5021392107009888),
              3: (.4551868140697479, .49676835536956787),
              5: (.4400653541088104, .4872252643108368)}
    geo = dict(image_size=768, match_radius_px=6,
        target_anchors=[[.5620723366737366, .4995112419128418], [.4682306945323944, .4995112419128418]],
        candidates=[dict(slot=s, geometry=list(p), raw_target=1 if s == 2 else -1) for s, p in points.items()])
    layers = dict(atoms=[dict(atom_id=0, visible_text=False), dict(atom_id=1, visible_text=True)], bonds=[])
    members = groups.memberships(geo)
    proposal = dict(nodes=[], proposed_pairs=[dict(fragment_slot=1), dict(fragment_slot=3)],
        candidates=[dict(slot=c['slot'], raw_target=c['raw_target'], memberships=members[c['slot']],
                         evidence=dict(glyph_hits=[])) for c in geo['candidates']])
    subset = dict(stable_proposal=proposal, excluded_pairs=[dict(pair=dict(fragment_slot=5))])
    check = stable.removal_check(geo, subset, record(geo, layers, set(points)), layers, [])
    assert check['removed_fragment_slots'] == [1, 3]
    assert check['held_sensitive_selected_slots'] == [5]
    assert check['original_correct_slots'] == [2] and not check['removed_correct_slots']
    assert check['lost_target_ids'] == [1]
    assert not check['all_protections_preserved']
