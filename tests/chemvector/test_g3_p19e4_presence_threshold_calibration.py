import torch

from chemvector.evaluation.audits.graph import (
    audit_g3_p19e4_presence_threshold_calibration as p19e4,
)


def _record(key, probabilities, presence):
    return {
        "sample_key": key,
        "variant": "stress",
        "stress_operation": "near_nonbonded",
        "probabilities": torch.tensor(probabilities),
        "presence": torch.tensor(presence),
    }


def test_select_threshold_uses_global_exact_intervals():
    records = [
        _record("a", [0.9, 0.1], [True, False]),
        _record("b", [0.8, 0.7], [True, False]),
    ]
    threshold, audit = p19e4.select_global_threshold(records)
    assert 0.7 < threshold <= 0.800001
    assert audit["selected"]["overall"]["skeleton_exact"] == 2


def test_metrics_at_threshold_counts_false_edges():
    metrics = p19e4.metrics_at_threshold(
        [_record("a", [0.9, 0.6, 0.1], [True, False, False])], 0.5
    )
    assert metrics["overall"]["true_positive"] == 1
    assert metrics["overall"]["false_positive"] == 1
    assert metrics["overall"]["skeleton_exact"] == 0


def test_split_selection_is_scaffold_isolated(monkeypatch):
    monkeypatch.setattr(p19e4, "MOLECULES_PER_SPLIT", 1)
    monkeypatch.setattr(p19e4, "IMAGES_PER_SPLIT", 4)

    def rows(molecule, scaffold):
        return [
            {
                "sample_key": f"{molecule}_{variant}",
                "molecule_id": molecule,
                "scaffold": scaffold,
                "variant": variant,
            }
            for variant in p19e4.VARIANTS
        ]

    pilot = rows("pilot", "p")
    old_dev = rows("old", "d")
    full = pilot + old_dev + rows("a", "x") + rows("b", "y")
    calibration, evaluation, audit = p19e4.select_calibration_eval(
        pilot, old_dev, full
    )
    assert len(calibration) == len(evaluation) == 4
    assert audit["calibration_eval_scaffold_overlap"] == 0
    assert audit["calibration_excluded_scaffold_overlap"] == 0
    assert audit["eval_excluded_scaffold_overlap"] == 0
