import torch

from tools.audit_r17_pair_incidence_calibration_cpu import evaluate, _select_profile


class AlwaysPositive:
    def __call__(self, x):
        return torch.full((len(x),), 10.0)


def item(profile="oa13"):
    return {"key": "sample", "profile": profile, "features": torch.zeros(2, 262),
            "base_logits": torch.tensor([2.0, 0.0]), "chemical_logits": torch.zeros(2, 6),
            "labels": torch.tensor([1.0, 0.0]), "type_labels": ["SINGLE", None]}


def test_explicit_abstention_preserves_only_baseline():
    scored, profiles = evaluate([item()], AlwaysPositive(), (1.0, 4.0, 0.0), {"oa13": None})
    assert scored["predicted_edges"] == 1
    assert profiles["oa13"]["predicted_edges"] == 1


def test_profile_selection_supports_targeted_cohort():
    rows = [item()]
    base = {"oa13": {"images": 1, "predicted_edges": 1, "mapped_edges": 1,
                     "true_edges": 1, "typed_true_edges": 0, "target_edges": 1,
                     "edge_precision": 1.0, "typed_edge_precision": 0.0,
                     "edge_recall": 1.0, "typed_edge_recall": 0.0}}
    params, thresholds, diag = _select_profile(rows, AlwaysPositive(), base, ("oa13",), (1.0, 4.0, 0.0))
    assert params == (1.0, 4.0, 0.0)
    assert set(thresholds) == {"oa13"}
    assert set(diag) == {"oa13"}


def test_decision_digest_does_not_depend_on_targets():
    a = item()
    b = {**a, "labels": torch.tensor([0.0, 1.0]), "type_labels": [None, "DOUBLE"]}
    *_, digest_a = evaluate([a], AlwaysPositive(), (1.0, 4.0, 0.0), 0.8, decisions=True)
    *_, digest_b = evaluate([b], AlwaysPositive(), (1.0, 4.0, 0.0), 0.8, decisions=True)
    assert digest_a == digest_b
