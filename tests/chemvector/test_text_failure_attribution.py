import copy
import json

import pytest

from chemvector.audit_text_failure_attribution import selected_records, unique_index
from chemvector.text_failure_attribution import (
    attribute_failures, geometry_evidence, locate_box_failure, summarize_failures,
)
from chemvector.text_failure_review import write_gallery


def inputs():
    target = {"gt_key": "s:t000", "sample_key": "s", "molecule_id": "m",
              "variant": "stress", "truth": "N", "bbox": [0, 0, 10, 10],
              "arms": {"atom_conditioned": {"matched": False, "grounded_exact": False},
                       "independent_text": {"grounded_exact": False}}}
    ladder = {**{k: target[k] for k in ("gt_key", "sample_key", "molecule_id", "variant", "truth")},
              "role": "text", "anchor": [5, 5],
              "stages": {stage: {"matched": False} for stage in (
                  "deployed", "decoded", "raw_top128", "all_local_peaks", "dense_anchors")}}
    atom = {"sample_key": "s", "predicted_atom_index": 3,
            "selected_as_text": True, "crop_object_key": "s:p003"}
    proposal = {"object_key": "A:s:p003", "bbox": [1, 1, 9, 9], "text": "O"}
    return target, ladder, {("s", 3): atom}, {"A:s:p003": proposal}


def test_role_requires_the_same_deployed_atom_identity():
    target, ladder, atoms, predictions = inputs()
    ladder["stages"]["deployed"] = {"matched": True, "proposal_index": 3}
    atoms[("s", 3)]["selected_as_text"] = False
    assert locate_box_failure(target, ladder, atoms, predictions) == ("role_rejected", None)
    with pytest.raises(ValueError, match="identity missing"):
        locate_box_failure(target, ladder, {}, predictions)


def test_bbox_geometry_distinguished_from_assignment_competition():
    target, ladder, atoms, predictions = inputs()
    ladder["stages"]["deployed"] = {"matched": True, "proposal_index": 3}
    assert locate_box_failure(target, ladder, atoms, predictions)[0] == "bbox_assignment_competition"
    predictions["A:s:p003"]["bbox"] = [20, 20, 30, 30]
    assert locate_box_failure(target, ladder, atoms, predictions)[0] == "bbox_misaligned"
    with pytest.raises(ValueError, match="missing its frozen crop"):
        locate_box_failure(target, ladder, atoms, {})


def test_set_filtering_requires_score_and_candidate_absence():
    target, ladder, atoms, predictions = inputs()
    ladder["stages"]["decoded"] = {"matched": True, "proposal_index": 3, "presence_logit": -.5}
    assert locate_box_failure(target, ladder, {}, predictions)[0] == "set_rejected"
    with pytest.raises(ValueError, match="unexpectedly deployed"):
        locate_box_failure(target, ladder, atoms, predictions)
    ladder["stages"]["decoded"]["presence_logit"] = -.25
    assert locate_box_failure(target, ladder, atoms, predictions)[0] == "anchor_assignment_competition"
    with pytest.raises(ValueError, match="absent"):
        locate_box_failure(target, ladder, {}, predictions)


@pytest.mark.parametrize("stage,category", [
    ("raw_top128", "decode_or_assignment_loss"),
    ("all_local_peaks", "raw_capacity_or_assignment_loss"),
    ("dense_anchors", "local_peak_unreachable"),
    (None, "anchor_field_unreachable"),
])
def test_upstream_reachability_categories(stage, category):
    target, ladder, atoms, predictions = inputs()
    if stage:
        ladder["stages"][stage]["matched"] = True
    assert locate_box_failure(target, ladder, atoms, predictions)[0] == category


def test_raw_box_incomplete_does_not_mean_context_crop_truncates():
    evidence = geometry_evidence([2, 2, 8, 8], [0, 0, 10, 10])
    assert evidence["gt_box_coverage"] == .36
    assert evidence["context_gt_box_coverage"] == 1
    assert not evidence["context_misses_gt_box_area"]
    assert geometry_evidence([0, 0, 4, 10], [0, 0, 10, 10])["context_misses_gt_box_area"]


def attributed():
    target, ladder, atoms, predictions = inputs()
    target["arms"]["atom_conditioned"].update(
        matched=True, prediction_key="A:s:p003", predicted_text="O", iou=.64)
    peaks = {target["gt_key"]: {"stages": {"a3_bf16_dense_top384": {"matched": True}}}}
    rows = attribute_failures([target], {target["gt_key"]: ladder}, atoms, predictions, peaks)
    oracle = {"s:t000": {"baseline_text": "O", "oracle_text": "N", "truth": "N"}}
    return rows, oracle


def test_matched_ocr_error_kept_separate_and_counterfactual_counted():
    rows, oracle = attributed()
    assert rows[0]["category"] == "matched_ocr_error"
    report = summarize_failures(rows, oracle)
    assert report["ocr_counterfactual"]["oracle_exact"] == 1
    assert report["ocr_counterfactual"]["context_misses_gt_box_area"] == 0
    assert not report["training_authorized"] and not report["promotion_authorized"]
    oracle["s:t000"]["oracle_text"] = "NH"
    assert summarize_failures(rows, oracle)["ocr_counterfactual"]["oracle_still_wrong"] == 1


@pytest.mark.parametrize("field,value", [("baseline_text", "changed"), ("truth", "changed")])
def test_counterfactual_replay_and_identity_fail_closed(field, value):
    rows, oracle = attributed()
    oracle["s:t000"][field] = value
    with pytest.raises(ValueError):
        summarize_failures(rows, oracle)


def test_missing_oracle_population_fails():
    rows, _ = attributed()
    with pytest.raises(ValueError, match="population"):
        summarize_failures(rows, {})


def test_ladder_identity_mismatch_fails():
    target, ladder, atoms, predictions = inputs()
    ladder["truth"] = "O"
    with pytest.raises(ValueError, match="identity differs"):
        attribute_failures([target], {"s:t000": ladder}, atoms, predictions, {})


def test_geometry_replay_mismatch_fails():
    target, ladder, atoms, predictions = inputs()
    target["arms"]["atom_conditioned"].update(
        matched=True, prediction_key="A:s:p003", predicted_text="O", iou=.9)
    with pytest.raises(ValueError, match="geometry differs"):
        attribute_failures([target], {"s:t000": ladder}, atoms, predictions, {})


def test_successful_targets_are_not_selected():
    target, *_ = inputs()
    target["arms"]["atom_conditioned"]["grounded_exact"] = True
    assert attribute_failures([target], {}, {}, {}, {}) == []


def test_indices_and_streamed_population_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="Duplicate"):
        unique_index([{"key": "a"}, {"key": "a"}], lambda r: r["key"])
    path = tmp_path / "targets.jsonl"
    path.write_text(json.dumps({"gt_key": "a"}) + "\n" + json.dumps({"gt_key": "b"}) + "\n")
    assert set(selected_records(path, {"a"}, 2)) == {"a"}
    with pytest.raises(ValueError, match="population"):
        selected_records(path, {"missing"}, 2)
    with pytest.raises(ValueError, match="population"):
        selected_records(path, {"a"}, 3)


def test_gallery_escapes_label_markup(tmp_path):
    rows, _ = attributed()
    row = copy.deepcopy(rows[0])
    row["truth"] = '<script>alert("x")</script>'
    row["associated_prediction"] = None
    write_gallery(tmp_path, [row], {}, {"s:t000": "visuals/test.png"})
    html = (tmp_path / "review.html").read_text()
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
