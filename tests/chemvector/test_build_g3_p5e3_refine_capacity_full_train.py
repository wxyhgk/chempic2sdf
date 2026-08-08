from chemvector.evaluation.audits.atoms.build_g3_p5e3_refine_capacity_full_train import (
    EXPECTED_ELIGIBLE_IMAGES,
    EXPECTED_ELIGIBLE_MOLECULES,
    MINIMUM_REFINE_ACTIONS,
    MINIMUM_REFINE_MOLECULES,
)


def test_p5e3_capacity_protocol_is_frozen() -> None:
    assert EXPECTED_ELIGIBLE_IMAGES == 58_376
    assert EXPECTED_ELIGIBLE_MOLECULES == 14_594
    assert MINIMUM_REFINE_ACTIONS == 800
    assert MINIMUM_REFINE_MOLECULES == 500
