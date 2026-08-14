from chemvector.evaluation.audits.graph import (
    audit_v1p0_edge_centric_oracle10k as v1p0,
)


def test_first_four_real_samples_obey_edge_centric_oracle():
    report = v1p0.audit_archive(v1p0.SHARDS[-1], limit=4)
    assert report["counts"] == {"total": 4, "passed": 4}
    assert report["object_totals"]["ports"] == 2 * report["object_totals"]["bonds"]
    assert report["generated_png"] is False


def test_aggregate_preserves_skip_coverage():
    reports = [
        {
            "counts": {"total": 3, "passed": 2, "skipped_coordinate_collision": 1},
            "object_totals": {"atoms": 4, "bonds": 3, "ports": 6},
            "variants": {"stress": 2},
            "stress_operations": {"crossed_bonds": 2},
            "failures": [],
        },
        {
            "counts": {"total": 1, "passed": 1},
            "object_totals": {"atoms": 2, "bonds": 1, "ports": 2},
            "variants": {"rdkit": 1},
            "stress_operations": {},
            "failures": [],
        },
    ]
    result = v1p0.aggregate_archives(reports)
    assert result["coverage"] == 0.75
    assert result["object_totals"]["ports"] == 8
    assert result["generated_png"] is False
