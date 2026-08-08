import pytest

from chemvector.evaluation.audits.atoms.audit_g3_p5e6_refine_error_decomposition import (
    distance_bucket,
    nearest_neighbor_bucket,
    rank_bucket,
)


def test_p5e6_bucket_boundaries_are_frozen() -> None:
    assert distance_bucket(6.1) == "6-8"
    assert distance_bucket(8.0) == "6-8"
    assert distance_bucket(8.1) == "8-10"
    assert distance_bucket(10.1) == "10-12"
    with pytest.raises(ValueError):
        distance_bucket(6.0)
    assert nearest_neighbor_bucket(15.9) == "lt16"
    assert nearest_neighbor_bucket(16.0) == "16-32"
    assert nearest_neighbor_bucket(32.0) == "ge32"
    assert rank_bucket(16) == "top16"
    assert rank_bucket(17) == "17-64"
    assert rank_bucket(65) == "gt64"
