from chemvector.evaluation.audits.vision.audit_v1p1ac_binary_ink_internal_confirm import (
    CHECKPOINT_SHA256,
    P1A_RESULT_SHA256,
)


def test_internal_confirm_inputs_are_frozen_sha256_values():
    assert len(CHECKPOINT_SHA256) == 64
    assert len(P1A_RESULT_SHA256) == 64
    int(CHECKPOINT_SHA256, 16)
    int(P1A_RESULT_SHA256, 16)
