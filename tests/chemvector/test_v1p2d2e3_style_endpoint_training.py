from collections import Counter

from chemvector.training.bonds.objects.train_v1p2d2e3_style_endpoint_field import (
    _checks,
    _reachability_metrics,
    load_style_manifest,
    split_rows,
)


def test_style_endpoint_manifest_counts_match_frozen_safe_split() -> None:
    rows = load_style_manifest()
    assert [len(split_rows(rows, split)) for split in ("fit", "calibration", "holdout")] == [
        91_740,
        3_588,
        4_008,
    ]


def test_reachability_metrics_report_port_and_pairable_recall() -> None:
    metrics = _reachability_metrics(
        Counter(
            samples=2,
            ports=10,
            bonds=8,
            outputs=20,
            saturated=1,
            recovered_ports=9,
            pairable_bonds=7,
            port_exact=1,
        )
    )
    assert metrics["unique_port_recall"] == 0.9
    assert metrics["pairable_bond_recall"] == 0.875
    assert metrics["port_exact_rate"] == 0.5


def test_style_endpoint_checks_enforce_fresh_stress_gate() -> None:
    style = {
        "samples": 4_008,
        "budgets": {
            "max512": {"unique_port_recall": 0.99, "pairable_bond_recall": 0.99}
        },
    }
    fresh = {
        "samples": 400,
        "budgets": {
            "max512": {"unique_port_recall": 0.99, "pairable_bond_recall": 0.99}
        },
        "by_variant": {"stress": {"max512": {"pairable_bond_recall": 0.94}}},
    }
    checks = _checks(style, fresh)
    assert not checks["fresh_stress_pairable_bond_recall_min95pct"]
    assert not all(checks.values())
