from chemvector.evaluation.audits.graph import (
    audit_g3_p19e2_direct_atom_pair_fresh640 as p19e2,
)


def _row(key, *, exact, variant="stress", operation="crossed_bonds"):
    return {
        "sample_key": key,
        "variant": variant,
        "stress_operation": operation,
        "skeleton_exact": exact,
    }


def test_paired_summary_counts_rescue_and_break():
    old = [_row("a", exact=False), _row("b", exact=True), _row("c", exact=True)]
    new = [_row("a", exact=True), _row("b", exact=False), _row("c", exact=True)]
    result = p19e2.paired_summary(old, new)
    assert result["overall"] == {
        "images": 3,
        "old_exact": 2,
        "new_exact": 2,
        "rescue": 1,
        "break": 1,
    }
    assert result["stress_by_operation"]["crossed_bonds"]["rescue"] == 1


def test_paired_summary_is_key_order_invariant():
    old = [_row("a", exact=False), _row("b", exact=True)]
    new = [_row("b", exact=True), _row("a", exact=True)]
    assert p19e2.paired_summary(old, new)["overall"]["rescue"] == 1


def test_paired_summary_rejects_key_mismatch():
    try:
        p19e2.paired_summary([_row("a", exact=False)], [_row("b", exact=True)])
    except ValueError as error:
        assert "keys differ" in str(error)
    else:
        raise AssertionError("mismatched paired rows were accepted")
