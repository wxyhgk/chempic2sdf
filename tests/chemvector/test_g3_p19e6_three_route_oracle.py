from chemvector.evaluation.audits.graph import (
    audit_g3_p19e6_three_route_oracle as p19e6,
)


def _new(key, exact):
    return {
        "sample_key": key,
        "variant": "stress",
        "stress_operation": "crossed_bonds",
        "skeleton_exact": exact,
    }


def _old(key, exact):
    return {
        "sample_key": key,
        "variant": "stress",
        "stress_operation": "crossed_bonds",
        "old_skeleton_exact": exact,
    }


def test_oracle_union_and_disagreement():
    rows = p19e6.build_oracle_rows(
        [_new("a", True), _new("b", False)],
        [_new("a", False), _new("b", True)],
        [_old("a", False), _old("b", False)],
    )
    summary = p19e6.summarize(rows)
    assert summary["oracle_exact"] == 2
    assert summary["routes_disagree"] == 2
    assert rows[0]["exact_routes"] == ["new"]
    assert rows[1]["exact_routes"] == ["fusion"]


def test_oracle_rejects_key_mismatch():
    try:
        p19e6.build_oracle_rows(
            [_new("a", True)], [_new("b", True)], [_old("a", True)]
        )
    except ValueError as error:
        assert "keys differ" in str(error)
    else:
        raise AssertionError("mismatched route keys were accepted")
