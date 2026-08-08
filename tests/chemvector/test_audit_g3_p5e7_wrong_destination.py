from chemvector.evaluation.audits.atoms.audit_g3_p5e7_wrong_destination import (
    destination_class,
)


def test_p5e7_destination_precedence_is_frozen() -> None:
    assert destination_class(5.0, 2.0) == "TARGET_HIT"
    assert destination_class(8.0, 5.0) == "OTHER_GT_HIT"
    assert destination_class(10.0, 8.0) == "OTHER_GT_CLOSER"
    assert destination_class(8.0, 10.0) == "TARGET_STILL_NEAREST"
