"""Synthetic-only replay gates and honest incremental shadow scoring."""

import io
import tarfile

import numpy as np
from PIL import Image
import pytest

from chemvector.evaluation.audits.ocr.set_rejected_text.checks import check_baseline_text, check_deployed, rejected_mask
from chemvector.evaluation.audits.ocr.set_rejected_text.inputs import index_selected_images, selected_images
from chemvector.evaluation.audits.ocr.set_rejected_text.metrics import aggregate, decide, evaluate_image
from chemvector.external_ocr_r0_contracts import sha256_bytes


def target(name, box, matched=False, truth="N"):
    return {"sample_key": "sample", "gt_key": name, "bbox": box, "truth": truth,
            "arms": {"atom_conditioned": {"matched": matched}}}


def prediction(name, box, text="N"):
    return {"sample_key": "sample", "object_key": name, "bbox": box, "text": text}


def test_shadow_does_not_rematch_or_steal_A_targets():
    targets = [target("covered", [0, 0, 10, 10], True), target("missing", [20, 0, 30, 10])]
    shadow = [prediction("dupA", [0, 0, 10, 10]), prediction("new", [20, 0, 30, 10])]
    sample, rows, details = evaluate_image("sample", "stress", shadow, targets, {"missing"})
    assert sample["A_covered"] == 1 and sample["new_boxes"] == 1
    assert sample["duplicates"] == 1 and sample["set_loss_new_boxes"] == 1
    assert len(rows) == 1 and rows[0]["gt_key"] == "missing"
    assert details[0]["overlaps_A_covered"]


def test_duplicates_and_unmatched_partition_all_shadow_boxes():
    targets = [target("missing", [0, 0, 10, 10])]
    shadow = [prediction("one", [0, 0, 10, 10]), prediction("two", [0, 0, 10, 10]),
              prediction("extra", [50, 50, 60, 60], "")]
    sample, _, _ = evaluate_image("sample", "original", shadow, targets, set())
    assert (sample["new_boxes"], sample["duplicates"], sample["unmatched_no_qualified_target"]) == (1, 1, 1)
    assert sample["nonempty_B_crops"] == 2


def test_strings_do_not_drive_assignment_and_CH2_is_not_corrected():
    targets = [target("carbon", [0, 0, 10, 10], truth="CH2")]
    shadow = [prediction("perfectbox", [0, 0, 10, 10], "C"), prediction("worsebox", [1, 0, 11, 10], "CH2")]
    sample, rows, _ = evaluate_image("sample", "stress", shadow, targets, set())
    assert rows[0]["prediction_key"] == "perfectbox"
    assert rows[0]["historical_CH2_disputed"] and not rows[0]["historical_raw_exact"]
    assert sample["new_CH2_boxes"] == 1 and sample["new_nonCH2_boxes"] == 0


@pytest.mark.parametrize("with_target", [True, False])
def test_empty_shadow_keeps_denominator(with_target):
    targets = [target("one", [0, 0, 10, 10])] if with_target else []
    sample, _, _ = evaluate_image("sample", "stress", [], targets, set())
    summary = aggregate([sample])
    assert summary["images"] == 1 and summary["B_crops"] == 0
    assert summary["incremental_box_precision"] is None


def test_extra_boxes_without_targets_count_as_unmatched():
    sample, _, _ = evaluate_image("sample", "stress", [prediction("extra", [0, 0, 10, 10])], [], set())
    assert sample["unmatched_no_qualified_target"] == 1


def test_multi_target_overlap_reported_not_double_counted():
    targets = [target("one", [0, 0, 10, 10]), target("two", [2, 0, 12, 10])]
    sample, _, _ = evaluate_image("sample", "stress", [prediction("both", [1, 0, 11, 10])], targets, set())
    assert sample["new_boxes"] == 1 and sample["ambiguous_multiple_targets"] == 1


@pytest.mark.parametrize("kind", ["duplicate", "mixed"])
def test_invalid_identity_is_rejected(kind):
    rows = [prediction("same", [0, 0, 10, 10])]
    if kind == "duplicate":
        rows *= 2
    else:
        rows[0]["sample_key"] = "different"
    with pytest.raises(ValueError):
        evaluate_image("sample", "stress", rows, [], set())


def test_partition_never_expands_decoded_set():
    decoded = np.array([[True, True, False]])
    selected = np.array([[True, False, False]])
    assert rejected_mask(decoded, selected).tolist() == [[False, True, False]]
    with pytest.raises(ValueError):
        rejected_mask(decoded, np.array([[True, False, True]]))


def test_deployed_ids_and_coordinates_require_exact_replay():
    geometry = np.array([[0.25, 0.5], [0.75, 0.5]], dtype=np.float32)
    expected = [{"predicted_atom_index": 0, "anchor": [0.25, 0.5]}]
    check_deployed("sample", geometry, np.array([True, False]), expected)
    with pytest.raises(ValueError, match="IDs"):
        check_deployed("sample", geometry, np.array([True, True]), expected)
    geometry[0, 0] += .0001
    with pytest.raises(ValueError, match="anchor"):
        check_deployed("sample", geometry, np.array([True, False]), expected)


@pytest.mark.parametrize("field", ["bbox", "score", "crop_sha256", "object_key"])
def test_baseline_text_replay_is_not_tolerance_tuned(field):
    expected = [{"object_key": "A:sample:p0", "bbox": [0, 0, 10, 10], "score": 0.5, "crop_sha256": "hash"}]
    actual = [dict(expected[0])]
    actual[0][field] = "different"
    with pytest.raises(ValueError):
        check_baseline_text("sample", actual, expected)


def test_gate_cannot_pass_on_recall_alone():
    gate = {"minimum_new_boxes": 5, "max_extra_unmatched_per_image": 1,
            "max_duplicate_fraction": .5, "minimum_stress_new_boxes": 1}
    overall = {"new_boxes": 10, "extra_unmatched_per_image": 4, "duplicate_fraction": .7}
    result = decide(overall, {"new_boxes": 2}, gate)
    assert not result["gate_passed"] and not result["graph_gain_measured"]


def test_reader_opens_only_allowlisted_png_members(tmp_path):
    payload = io.BytesIO()
    Image.new("RGB", (768, 768), "white").save(payload, format="PNG")
    png = payload.getvalue()
    path = tmp_path / "fresh_dev-000000.tar"
    with tarfile.open(path, "w") as archive:
        for name, data in [("other.png", b"not an image"), ("sample.meta.json", b"not json"), ("sample.png", png)]:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    metadata = {"sample": {"png_sha256": sha256_bytes(png)}}
    order, index = index_selected_images({str(path): "hash"}, metadata)
    protocol = {"shards": {str(path): "hash"}, "sample_order": order, "image_index": index, "sample_metadata": metadata}
    samples = list(selected_images(protocol))
    assert len(samples) == 1 and samples[0]["key"] == "sample"
    metadata["sample"]["png_sha256"] = "wrong"
    with pytest.raises(ValueError, match="hash"):
        list(selected_images(protocol))


@pytest.mark.parametrize("fault", [None, "missing", "duplicate", "count", "partition", "score", "unknown_arm"])
def test_saved_shadow_exhausts_qualified_candidates(fault):
    from chemvector.evaluation.audits.ocr.set_rejected_text.checks import check_shadow_records

    candidates = [{"sample_key": "sample", "proposal_index": 1, "decoded": True,
                   "selected": False, "shadow_eligible": True, "set_logit": -1., "role_score": .8}]
    rows = [{"arm": "shadow_rejected", "sample_key": "sample", "proposal_index": 1, "score": .8}]
    samples = [{"sample_key": "sample", "decoded": 1, "selected": 0, "rejected": 1, "shadow_crops": 1}]
    if fault == "missing":
        rows = []
    elif fault == "duplicate":
        rows *= 2
    elif fault == "count":
        samples[0]["decoded"] = 2
    elif fault == "partition":
        candidates[0]["selected"] = True
    elif fault == "score":
        rows[0]["score"] = .9
    elif fault == "unknown_arm":
        rows[0]["arm"] = "other"
    if fault:
        with pytest.raises(ValueError):
            check_shadow_records(candidates, rows, samples, {"sample"}, -.25, .5)
    else:
        check_shadow_records(candidates, rows, samples, {"sample"}, -.25, .5)
