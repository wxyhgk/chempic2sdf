from chemvector.evaluation.audits.bonds.audit_v1p2d2e5_slots4_local_partner_oracle import (
    BASELINE_TOP8_PAIR_RECALL,
    ENDPOINT_CHECKPOINT,
    ENDPOINT_CHECKPOINT_SHA256,
)
from chemvector.g3_structural_data import file_sha256


def test_slots4_local_partner_oracle_pins_endpoint_checkpoint() -> None:
    assert file_sha256(ENDPOINT_CHECKPOINT) == ENDPOINT_CHECKPOINT_SHA256
    assert 0.92 < BASELINE_TOP8_PAIR_RECALL < 0.93
