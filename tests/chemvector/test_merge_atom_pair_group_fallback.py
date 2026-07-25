from chemvector.merge_atom_pair_group_fallback import select_record


def test_group_primary_and_valid_baseline_fallback() -> None:
    selected, source = select_record(
        {"sdf_valid": True, "seconds": 1.0},
        {"sdf_valid": True, "seconds": 2.0},
    )
    assert source == "atom_pair_group_primary"
    assert selected["seconds"] == 1.0

    selected, source = select_record(
        {"sdf_valid": False, "seconds": 1.0},
        {"sdf_valid": True, "seconds": 2.0},
    )
    assert source == "baseline_invalid_fallback"
    assert selected["seconds"] == 3.0
