import pytest

from chemvector.evaluation.audits.graph.build_v1p0r_edge_centric_safe_manifest import (
    VARIANTS,
    partition_complete_molecule_rows,
)


def _rows(molecule_id: str):
    return [
        {
            "sample_key": f"{molecule_id}-{variant}",
            "molecule_id": molecule_id,
            "variant": variant,
        }
        for variant in VARIANTS
    ]


def test_partition_excludes_the_entire_molecule_group():
    safe, excluded = partition_complete_molecule_rows(
        _rows("safe") + _rows("bad"), {"bad"}
    )
    assert {row["molecule_id"] for row in safe} == {"safe"}
    assert {row["molecule_id"] for row in excluded} == {"bad"}
    assert len(safe) == len(excluded) == 4


def test_partition_rejects_partial_molecule_groups():
    with pytest.raises(ValueError, match="Incomplete four-variant molecule"):
        partition_complete_molecule_rows(_rows("partial")[:-1], set())
