from copy import deepcopy

import pytest

from chemvector.experiments import ocr_consistency as ocr, selector_recheck as selector


def proposals():
    positions = [(0, 100), (140, 100), (210, 100), (210, 170), (140, 170),
                 (70, 100), (70, 105), (60, 70), (135, 100)]
    return [dict(slot=i, xy=list(xy), keep_logit=1 if i < 5 else -1-(i-5), selected=i < 5)
            for i, xy in enumerate(positions)]


EDGES = [[1, 2], [2, 3], [3, 4]]


def test_orphan_shortlist_preserves_selected_and_graph():
    data = proposals()
    before = deepcopy(data)
    result = selector.review_requests(data, EDGES)
    assert result['selected_slots'] == [0, 1, 2, 3, 4]
    assert result['orphan_slots'] == [0]
    assert len(result['requests']) == 1
    assert result['requests'][0]['first_review_slot'] == 5
    assert [p['slot'] for p in result['requests'][0]['candidates']] == [5, 6]
    assert result['graph_edits'] == 0 and result['action'] == 'review_only'
    assert data == before


def test_no_orphan_no_recovery_request():
    assert selector.review_requests(proposals(), EDGES + [[0, 1]])['requests'] == []


def test_tied_rank_abstains():
    data = proposals()
    data[6]['keep_logit'] = data[5]['keep_logit']
    group = selector.review_requests(data, EDGES)['requests'][0]
    assert group['first_review_slot'] is None


def test_no_length_reference_abstains():
    r = selector.review_requests(proposals(), [])
    assert r['status'] == 'insufficient_predicted_length_reference' and not r['requests']


def test_budget_refuses_instead_of_silent_truncation():
    data = proposals()
    data += [dict(slot=10+i, xy=[0, 101+i], keep_logit=1., selected=True) for i in range(4)]
    result = selector.review_requests(data, EDGES)
    assert result['status'] == 'review_budget_exceeded'
    assert result['requests'] == []


def test_target_input_rejected():
    data = proposals()
    data[5]['target_id'] = 123
    with pytest.raises(ValueError, match='prediction-only'):
        selector.review_requests(data, EDGES)


def test_duplicate_slots_rejected():
    data = proposals()
    data.append(deepcopy(data[-1]))
    with pytest.raises(ValueError):
        selector.review_requests(data, EDGES)


@pytest.mark.parametrize('bad_edge', [[5, 1], [1, 1], [1, 100]])
def test_nonselected_or_invalid_edges_rejected(bad_edge):
    with pytest.raises(ValueError):
        selector.review_requests(proposals(), EDGES + [bad_edge])


def decoded(text='F'):
    return [dict(variant=v[0], text=text) for v in ocr.VARIANTS]


def test_nine_fixed_variants_and_unmodified_base():
    box = [100, 100, 120, 130]
    rows = ocr.crop_variants(box, (768, 768))
    assert len(rows) == 9 and rows[0]['box'] == box
    assert rows[0]['context'] == 4
    assert rows[1]['box'][0] == 98 and rows[2]['box'][0] == 102
    assert all(r['available'] for r in rows)


def test_outside_image_variant_unavailable():
    rows = ocr.crop_variants([1, 1, 8, 10], (100, 100))
    assert not rows[0]['available']


def test_disagreement_does_not_vote_or_rewrite():
    rows = decoded('C')
    rows[0]['text'] = 'F'
    result = ocr.summarize(rows)
    assert result['needs_review'] and result['output_text'] == 'F'
    assert not result['output_changed']
    assert result['distinct_texts'] == ['C', 'F']


def test_unanimous_can_still_be_wrong():
    result = ocr.summarize(decoded())
    assert not result['needs_review']
    assert not result['agreement_is_correctness']


@pytest.mark.parametrize('replacement', [None, ''])
def test_blank_or_missing_requires_review(replacement):
    rows = decoded()
    rows[1]['text'] = replacement
    assert ocr.summarize(rows)['needs_review']


def test_duplicate_variant_rejected():
    rows = decoded()
    rows[-1]['variant'] = rows[0]['variant']
    with pytest.raises(ValueError):
        ocr.summarize(rows)


@pytest.mark.parametrize('box', [[1, 2, 1, 4], [1, 2, float('nan'), 4]])
def test_invalid_crop_rejected(box):
    with pytest.raises(ValueError):
        ocr.crop_variants(box, (768, 768))


def test_ocr_helper_does_not_modify_inputs():
    rows = decoded()
    before = deepcopy(rows)
    ocr.summarize(rows)
    assert rows == before
