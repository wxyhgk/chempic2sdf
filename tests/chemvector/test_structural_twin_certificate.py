from pathlib import Path

import torch

from chemvector.confirm_structural_twin_certificate import (
    _committed_confirm_tars,
    _safe_precision_lower,
)
from chemvector.structural_twin_certificate import (
    StructuralTwinCertificateConfig,
    structural_twin_delete_is_certified,
    structural_twin_witnesses,
)


def _sample(
    *,
    witness_x: float = 1.0,
    direct_relation: bool = False,
    second_chemical: int = 1,
) -> dict:
    pairs = [[0, 2], [1, 2]]
    probabilities = [0.99, 0.98]
    chemical = [1, second_chemical]
    if direct_relation:
        pairs.append([0, 1])
        probabilities.append(0.97)
        chemical.append(1)
    return {
        "anchors": torch.tensor(
            [[0.5, 0.5], [0.5 + witness_x / 768.0, 0.5], [0.7, 0.5]],
            dtype=torch.float32,
        ),
        "pairs": torch.tensor(pairs, dtype=torch.long),
        "probabilities": torch.tensor(probabilities),
        "chemical": torch.tensor(chemical),
    }


def test_identical_typed_neighborhood_is_certified() -> None:
    sample = _sample()
    assert structural_twin_witnesses(sample, 0, presence_threshold=0.9) == (1,)
    assert structural_twin_delete_is_certified(
        sample, 1, presence_threshold=0.9
    )


def test_chemical_mismatch_is_rejected() -> None:
    assert not structural_twin_delete_is_certified(
        _sample(second_chemical=2), 0, presence_threshold=0.9
    )


def test_direct_relation_between_nodes_is_rejected() -> None:
    assert not structural_twin_delete_is_certified(
        _sample(direct_relation=True), 0, presence_threshold=0.9
    )


def test_distance_boundary_is_explicit() -> None:
    config = StructuralTwinCertificateConfig(maximum_distance_px=2.0)
    assert structural_twin_delete_is_certified(
        _sample(witness_x=2.0), 0, presence_threshold=0.9, config=config
    )
    assert not structural_twin_delete_is_certified(
        _sample(witness_x=2.01), 0, presence_threshold=0.9, config=config
    )


def test_empty_neighborhood_is_rejected_by_default() -> None:
    sample = _sample()
    sample["probabilities"].fill_(0.1)
    assert not structural_twin_delete_is_certified(
        sample, 0, presence_threshold=0.9
    )


def test_zero_failure_confirm_sample_size_reaches_safety_bound() -> None:
    assert _safe_precision_lower(2_995, 0) >= 0.999
    assert _safe_precision_lower(2_994, 0) < 0.999
    assert _safe_precision_lower(10_000, 1) == 0.0


def test_confirm_tar_paths_remain_relative_to_dataset_root() -> None:
    commitment = {
        "tars": [
            {"path": "route_confirm/shards/route_confirm-000000.tar"},
            {"path": "route_select/shards/route_select-000000.tar"},
        ]
    }
    assert _committed_confirm_tars(
        commitment, data_root=Path("/dataset")
    ) == [
        (
            Path("/dataset/route_confirm/shards/route_confirm-000000.tar"),
            commitment["tars"][0],
        )
    ]
