import torch

from chemvector.evaluation.audits.atoms.build_g3_p5e8_generic_direction_capacity import (
    deterministic_offset,
    degree_bucket,
)


def test_p5e8_direction_generation_is_fixed_and_in_annulus() -> None:
    first = deterministic_offset("sample", 3)
    second = deterministic_offset("sample", 3)
    torch.testing.assert_close(first, second)
    radius = float(torch.linalg.vector_norm(first) * 768)
    assert 6.0 < radius <= 12.0
    assert degree_bucket(1) == "degree1"
    assert degree_bucket(2) == "degree2"
    assert degree_bucket(3) == "degree3plus"
