import torch

from chemvector.evaluation.audits.graph import (
    audit_g3_p19e5_old_new_logit_fusion as p19e5,
)


def _record(key, probabilities, presence):
    return {
        "sample_key": key,
        "variant": "stress",
        "stress_operation": "near_nonbonded",
        "probabilities": torch.tensor(probabilities),
        "presence": torch.tensor(presence),
    }


def test_fusion_endpoints_are_normalized_reference_margins():
    new = [_record("a", [p19e5.NEW_REFERENCE_THRESHOLD, 0.5], [True, False])]
    old = [_record("a", [p19e5.OLD_REFERENCE_THRESHOLD, 0.5], [True, False])]
    only_new = p19e5.fuse_probability_records(new, old, alpha=1.0)[0]
    only_old = p19e5.fuse_probability_records(new, old, alpha=0.0)[0]
    assert abs(float(only_new["probabilities"][0])) < 1e-5
    assert abs(float(only_old["probabilities"][0])) < 1e-5


def test_score_threshold_finds_shared_exact_interval():
    records = [
        _record("a", [2.0, -1.0], [True, False]),
        _record("b", [1.0, 0.5], [True, False]),
    ]
    threshold, audit = p19e5.select_score_threshold(records)
    assert 0.5 < threshold <= 1.0
    assert audit["metrics"]["overall"]["skeleton_exact"] == 2


def test_fusion_rejects_target_mismatch():
    new = [_record("a", [0.9, 0.1], [True, False])]
    old = [_record("a", [0.9, 0.1], [False, True])]
    try:
        p19e5.fuse_probability_records(new, old, alpha=0.5)
    except ValueError as error:
        assert "targets differ" in str(error)
    else:
        raise AssertionError("mismatched targets were accepted")
