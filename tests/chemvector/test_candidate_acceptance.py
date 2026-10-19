from copy import deepcopy
from itertools import combinations
import math

import pytest
from rdkit import Chem

from chemvector.experiments.candidate_acceptance import assess_candidate, select_prediction


def record(slots, edges, smiles):
    pairs = list(combinations(slots, 2))
    return dict(prediction=dict(molblock=Chem.MolToMolBlock(Chem.MolFromSmiles(smiles)), fallback=False, error=None),
        trace=dict(compiled_candidate_indices=slots, roles=[dict(candidate=s, text=False) for s in slots], ocr=[],
            scene_before_fallback=dict(atom_objects=[dict(id=i, anchor=[s, s]) for i, s in enumerate(slots)],
                bond_objects=[dict(id=i, atom_refs=[slots.index(a), slots.index(b)], chemical_type='SINGLE')
                              for i, (a, b) in enumerate(edges)])),
        pairs=[list(p) for p in pairs], presence_logits=[4. if p in edges else -4. for p in pairs],
        chemical_logits=[[4., 0., 0., 0.] for _ in pairs], postprocessed_chemical_classes=[0]*len(pairs))


def fixture():
    queue = dict(action='review_only', graph_edits=0, status='requests_ready', selected_slots=[0, 1, 2],
        requests=[dict(orphan_slot=0, partner_slot=1, first_review_slot=3, candidates=[dict(slot=3, keep_logit=-1.)])])
    return queue, record([0, 1, 2], [(1, 2)], 'C.CC'), record([0, 1, 2, 3], [(0, 3), (1, 2), (1, 3)], 'CCCC')


def test_accepts_exact_two_sided_hypothesis_without_claiming_identity():
    q, before, after = fixture()
    snapshot = deepcopy((q, before, after))
    decision, selected = select_prediction(q, before, after)
    assert decision['accepted'] and selected == after and selected is not after
    assert not decision['proves_identity_correct'] and not decision['deployment_authorized']
    assert (q, before, after) == snapshot


@pytest.mark.parametrize('change', ['weak_side', 'missing_edge', 'extra_edge', 'old_type', 'old_atom',
                                  'fallback', 'error', 'invalid_sdf', 'old_role', 'missing_scene'])
def test_refuses_and_returns_exact_original_copy(change):
    q, before, after = fixture()
    if change == 'weak_side': after['presence_logits'][after['pairs'].index([0, 3])] = 0.
    elif change == 'missing_edge': after['trace']['scene_before_fallback']['bond_objects'].pop(0)
    elif change == 'extra_edge':
        after['trace']['scene_before_fallback']['bond_objects'].append(dict(id=3, atom_refs=[2, 3], chemical_type='SINGLE'))
    elif change == 'old_type': after['trace']['scene_before_fallback']['bond_objects'][1]['chemical_type'] = 'DOUBLE'
    elif change == 'old_atom': after['trace']['scene_before_fallback']['atom_objects'][0]['anchor'] = [9, 9]
    elif change == 'fallback': after['prediction']['fallback'] = True
    elif change == 'error': after['prediction']['error'] = 'failed'
    elif change == 'invalid_sdf': after['prediction']['molblock'] = None
    elif change == 'old_role': after['trace']['roles'][0]['text'] = True
    else: del after['trace']['scene_before_fallback']
    decision, selected = select_prediction(q, before, after)
    assert not decision['accepted'] and selected == before and selected is not before


def test_rejects_truth_fields_and_incomplete_pairs():
    q, before, after = fixture(); after['expected_smiles'] = 'CCCC'
    with pytest.raises(ValueError, match='prediction-only'): assess_candidate(q, before, after)
    del after['expected_smiles']; after['pairs'].pop(); after['presence_logits'].pop()
    with pytest.raises(ValueError, match='all-pairs'): assess_candidate(q, before, after)


def test_original_threshold_boundary_is_inclusive():
    q, before, after = fixture()
    after['presence_logits'][after['pairs'].index([0, 3])] = math.log(.85/.15)
    assert assess_candidate(q, before, after)['accepted']


def test_nonfinite_scores_fail_closed():
    q, before, after = fixture(); after['presence_logits'][0] = float('nan')
    with pytest.raises(ValueError, match='finite'): assess_candidate(q, before, after)
