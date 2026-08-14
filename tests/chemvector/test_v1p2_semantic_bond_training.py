from chemvector.training.bonds.objects.train_v1p2_semantic_bond_object import (
    THRESHOLDS,
    split_train_calibration_rows,
)


def test_p2_split_keeps_complete_molecule_groups():
    variants = ("original", "rdkit", "coordgen", "stress")
    rows = [
        {
            "sample_key": f"{molecule}-{variant}",
            "molecule_id": f"m{molecule:03d}",
            "split": "train",
            "variant": variant,
        }
        for molecule in range(2214)
        for variant in variants
    ]
    fit, calibration = split_train_calibration_rows(rows)
    assert len(fit) == 8456
    assert len(calibration) == 400
    assert {row["molecule_id"] for row in fit}.isdisjoint(
        {row["molecule_id"] for row in calibration}
    )


def test_p2_threshold_grid_is_fixed():
    assert THRESHOLDS == tuple(round(index * 0.05, 2) for index in range(1, 20))
