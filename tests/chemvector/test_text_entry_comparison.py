import copy
import itertools

import numpy as np
import pytest

from chemvector.audit_text_entry_comparison import validate_predictions
from chemvector.text_entry_metrics import (
    box_arrays, decide, evaluate_image, match_boxes, summarize,
)
from chemvector.text_entry_protocol import ARMS, select_samples


def test_selection_is_deterministic_grouped_and_order_independent():
    metadata = {
        f"{m}:{v}": {"molecule_id": m, "variant": v}
        for m in range(10) for v in ("original", "rdkit", "coordgen", "stress")
    }
    selection = {"sample_metadata": metadata, "sample_keys": list(metadata)}
    selected = select_samples(selection, 3)
    selection["sample_keys"].reverse()
    assert selected == select_samples(selection, 3)
    assert len(selected) == 12
    assert len({row["molecule_id"] for row in selected.values()}) == 3
    with pytest.raises(ValueError, match="Insufficient"):
        select_samples(selection, 11)
    selection["sample_keys"] = list(selected)[:-1]
    with pytest.raises(ValueError, match="four complete"):
        select_samples(selection, 3)


@pytest.mark.parametrize("boxes", [[[0, 0, 0, 1]], [[0, 2, 1, 1]], [[0, 0, np.nan, 1]]])
def test_invalid_boxes_fail_closed(boxes):
    with pytest.raises(ValueError):
        box_arrays(boxes)


def test_empty_and_exact_threshold_matching():
    assert match_boxes([], [[0, 0, 1, 1]])[0] == {}
    assert match_boxes([[0, 0, 1, 1]], [])[0] == {}
    assert match_boxes([], [])[0] == {}
    assert len(match_boxes([[0, 0, 2, 1]], [[0, 0, 1, 1]], 0.5)[0]) == 1
    assert match_boxes([[0, 0, 2, 1]], [[0, 0, 1, 1]], 0.51)[0] == {}
    with pytest.raises(ValueError):
        match_boxes([], [], 0)


def test_matching_prioritizes_cardinality_then_iou(monkeypatch):
    import chemvector.text_entry_metrics as metrics

    matrices = [np.array([[1., .5], [.5, .49]])]
    rng = np.random.default_rng(91)
    matrices.extend(rng.uniform(size=(3, 3)) for _ in range(30))
    for matrix in matrices:
        monkeypatch.setattr(metrics, "overlap_matrices", lambda *_: (matrix, matrix))
        matched, _ = metrics.match_boxes([], [], .5)
        actual = (len(matched), sum(row["iou"] for row in matched.values()))
        expected = max(
            (sum(matrix[i, j] >= .5 for i, j in enumerate(order)),
             sum(matrix[i, j] for i, j in enumerate(order) if matrix[i, j] >= .5))
            for order in itertools.permutations(range(len(matrix)))
        )
        assert actual == pytest.approx(expected)


def prediction(key, text, box=(0, 0, 10, 10)):
    return {"object_key": key, "text": text, "bbox": list(box)}


def test_assignment_is_text_blind_and_duplicates_are_not_recovery():
    targets = [{"gt_key": "t", "truth": "N", "bbox": [0, 0, 10, 10]}]
    predictions = [prediction("best_geometry", "O"),
                   prediction("right_text", "N", (0, 0, 10, 11)),
                   prediction("extra", "", (20, 20, 30, 30))]
    sample, rows = evaluate_image("s", "stress", predictions, targets)
    assert rows[0]["prediction_key"] == "best_geometry"
    assert sample["matched"] == 1
    assert sample["grounded_exact"] == 0
    assert sample["duplicate_predictions"] == 1
    assert sample["unmatched_predictions"] == 2
    assert sample["unmatched_nonempty"] == 1
    assert sample["covered_90"] == 1
    assert not sample["all_text_exact"]
    predictions[0]["text"] = "N"
    improved, _ = evaluate_image("s", "stress", predictions, targets)
    assert summarize([improved])["conditional_ocr_exact"] == 1
    assert summarize([improved])["box_precision"] == pytest.approx(1 / 3)


def test_empty_text_image_requires_no_predictions_for_all_exact():
    empty, _ = evaluate_image("s", "stress", [], [])
    extra, _ = evaluate_image("s", "stress", [prediction("p", "")], [])
    assert empty["all_text_exact"]
    assert not extra["all_text_exact"]
    assert summarize([empty])["grounded_exact_rate"] is None
    assert summarize([])["unmatched_per_image"] is None


def decision_inputs():
    arms = dict(zip(ARMS, [
        {"grounded_exact_rate": .6, "box_precision": .9, "unmatched_per_image": .2},
        {"grounded_exact_rate": .65, "box_precision": .9, "unmatched_per_image": .3},
    ]))
    gates = {"grounded_exact_delta_pp_minimum": 2.,
             "precision_regression_pp_maximum": 1.,
             "extra_boxes_per_image_increase_maximum": 1.}
    return arms, {"stress": copy.deepcopy(arms)}, gates


def test_diagnostic_pass_never_authorizes_training_or_promotion():
    report = decide(*decision_inputs())
    assert report["diagnostic_gate_passed"]
    assert not report["training_authorized"]
    assert not report["promotion_authorized"]
    assert report["architecture_verdict"].startswith("not_identified")


@pytest.mark.parametrize("field,value", [("grounded_exact_rate", .61),
                                        ("box_precision", .88),
                                        ("box_precision", None),
                                        ("unmatched_per_image", 1.3)])
def test_each_overall_gate_fails_closed(field, value):
    arms, variants, gates = decision_inputs()
    arms[ARMS[1]][field] = value
    assert not decide(arms, variants, gates)["diagnostic_gate_passed"]


def test_stress_regression_and_missing_targets():
    arms, variants, gates = decision_inputs()
    variants["stress"][ARMS[1]]["grounded_exact_rate"] = .59
    assert not decide(arms, variants, gates)["diagnostic_gate_passed"]
    variants["stress"][ARMS[1]]["grounded_exact_rate"] = None
    with pytest.raises(ValueError, match="nonempty"):
        decide(arms, variants, gates)


def prediction_inputs():
    protocol = {"sample_metadata": {"s": {}}}
    receipt = {"rows": 1, "sample_keys": ["s"], "detector_repeat_exact": True,
               "training_performed": False, "scoring_truth_used_for_proposals": False,
               "models_before_sha256": "same", "models_after_sha256": "same"}
    rows = [{**prediction("p", "N"), "sample_key": "s", "arm": ARMS[0]}]
    return protocol, receipt, rows


def test_prediction_contract_accepts_zero_predictions_for_an_arm():
    validate_predictions(*prediction_inputs())


@pytest.mark.parametrize("field,value", [("rows", 2), ("sample_keys", []),
                                        ("detector_repeat_exact", False),
                                        ("training_performed", True),
                                        ("scoring_truth_used_for_proposals", True),
                                        ("models_after_sha256", "changed")])
def test_prediction_receipt_tampering_fails(field, value):
    protocol, receipt, rows = prediction_inputs()
    receipt[field] = value
    with pytest.raises(ValueError):
        validate_predictions(protocol, receipt, rows)


@pytest.mark.parametrize("field,value", [("sample_key", "outside"), ("arm", "unknown"),
                                        ("text", None), ("bbox", [0, 0, 0, 0])])
def test_prediction_row_tampering_fails(field, value):
    protocol, receipt, rows = prediction_inputs()
    rows[0][field] = value
    with pytest.raises(ValueError):
        validate_predictions(protocol, receipt, rows)


def test_duplicate_predictions_fail():
    protocol, receipt, rows = prediction_inputs()
    rows.append(copy.deepcopy(rows[0]))
    receipt["rows"] = 2
    with pytest.raises(ValueError, match="Duplicate"):
        validate_predictions(protocol, receipt, rows)
