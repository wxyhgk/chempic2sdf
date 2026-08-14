from chemvector.training.bonds.objects.train_v1p2d2e8_stable_bond_query_set_capacity64 import (
    DEFAULT_ACCUMULATION_STEPS,
    MODEL_NAME,
)


def test_stable_query_set_protocol_is_frozen() -> None:
    assert MODEL_NAME == "chemvector-v1p2d2e8-stable-bond-query-set-capacity64"
    assert DEFAULT_ACCUMULATION_STEPS == 8
