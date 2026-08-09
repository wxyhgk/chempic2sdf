from chemvector.evaluation.audits.graph.audit_g3_p18e0a_stress_bond_ceiling import (
    classify_bottleneck,
)


def test_classify_ranking_truncation():
    assert classify_bottleneck(0.35, 0.70, 0.75) == "top192_ranking_truncation"


def test_classify_endpoint_localization():
    assert classify_bottleneck(0.35, 0.45, 0.70) == "endpoint_localization_precision"


def test_classify_dense_generation():
    assert classify_bottleneck(0.35, 0.50, 0.58) == "dense_bond_generation"
