import csv

from chemvector.evaluation.audits.bonds.audit_v1p2d2e0_fresh_candidate_oracle import (
    FRESH_ROOT,
    split_fresh_rows,
)


def test_fresh_candidate_oracle_split_is_molecule_disjoint() -> None:
    with (FRESH_ROOT / "samples.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    calibration, holdout = split_fresh_rows(rows)
    assert (len(calibration), len(holdout)) == (400, 400)
    calibration_ids = {row["molecule_id"] for row in calibration}
    holdout_ids = {row["molecule_id"] for row in holdout}
    assert (len(calibration_ids), len(holdout_ids)) == (100, 100)
    assert not calibration_ids & holdout_ids
    assert {row["variant"] for row in calibration} == {
        "coordgen",
        "original",
        "rdkit",
        "stress",
    }
