from tools.current_initial_cohort import split_group_disjoint


def _row(i, group, molecule=None):
    return {
        "sample_key": f"k{i}",
        "group_key": group,
        "molecule_id": molecule or f"m{i}",
        "canonical_smiles": molecule or f"s{i}",
        "split": "train",
    }


def test_split_keeps_groups_whole():
    rows = [_row(0, "a"), _row(1, "a"), _row(2, "b"), _row(3, "c")]
    train, holdout, meta = split_group_disjoint(rows, train_target=2)
    assert len(train) == 2
    assert len(holdout) == 2
    assert meta["group_overlap"] == 0


def test_split_rejects_molecule_overlap():
    rows = [_row(0, "a", "same"), _row(1, "b", "same"), _row(2, "c")]
    try:
        split_group_disjoint(rows, train_target=1)
    except AssertionError:
        pass
    else:
        raise AssertionError("molecule overlap must be rejected")
