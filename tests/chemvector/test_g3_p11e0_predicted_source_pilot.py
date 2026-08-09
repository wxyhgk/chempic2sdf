from chemvector.evaluation.audits.atoms.build_g3_p11e0_predicted_source_pilot import (
    EXPECTED_MOVE,
    KEEP_PER_IMAGE,
    select_keep_assignments,
)


def test_p11_keep_selection_is_deterministic_and_excludes_move_sources() -> None:
    assignments = [3, 1, None, 2, 4]
    first = select_keep_assignments("sample", assignments, {2}, maximum=2)
    second = select_keep_assignments("sample", assignments, {2}, maximum=2)
    assert first == second
    assert len(first) == 2
    assert all(source != 2 for _, source in first)


def test_p11_protocol_counts_are_fixed() -> None:
    assert EXPECTED_MOVE == 264
    assert KEEP_PER_IMAGE == 8
