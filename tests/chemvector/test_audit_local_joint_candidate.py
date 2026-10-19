import pytest
import torch

from tools.audit_local_joint_candidate import _top_k_metrics, local_pair_mask


def test_local_pair_mask_uses_only_predicted_geometry_and_bounds():
    geometry = torch.tensor([[[0.10, 0.10], [0.20, 0.10], [0.50, 0.10], [0.90, 0.10]]])
    mask = torch.ones(1, 4, dtype=torch.bool)
    pairs = torch.tensor([[0, 1], [0, 2], [0, 3], [1, 2]])
    samples = torch.zeros(4, dtype=torch.long)
    selected, distances = local_pair_mask(
        geometry, mask, pairs, samples, min_distance_px=1, max_distance_px=100,
    )
    assert selected.tolist() == [True, False, False, False]
    assert distances.tolist() == pytest.approx([76.8, 307.2, 614.4, 230.4])


def test_local_pair_mask_rejects_unsorted_or_invalid_rows():
    geometry = torch.zeros(1, 2, 2)
    mask = torch.ones(1, 2, dtype=torch.bool)
    with pytest.raises(ValueError, match="sorted"):
        local_pair_mask(geometry, mask, torch.tensor([[1, 0]]), torch.tensor([0]))


def test_top_k_excludes_unknown_rows_and_reports_strict_type_counts():
    scores = torch.tensor([2.0, 1.0, 3.0, 0.0])
    known = torch.tensor([True, False, True, True])
    positive = torch.tensor([True, False, False, True])
    chemical = torch.tensor([0, 1, 2, 3])
    chemical_targets = chemical.clone()
    result = _top_k_metrics(scores, known, positive, chemical, chemical_targets, top_ks=(1, 2, 4))
    assert result["1"]["selected"] == 1
    assert result["1"]["true_positive"] == 0
    assert result["2"]["selected"] == 2
    assert result["2"]["true_positive"] == 1
    assert result["4"]["selected"] == 3
    assert result["4"]["true_positive"] == 2
    assert result["4"]["chemical_type_counts"]["SINGLE"] == 1
    assert result["4"]["chemical_type_counts"]["AROMATIC"] == 1


def test_top_k_rejects_unknown_positive_label():
    with pytest.raises(ValueError, match="Unknown"):
        _top_k_metrics(
            torch.zeros(1), torch.tensor([False]), torch.tensor([True]),
            torch.tensor([0]), torch.tensor([0]),
        )
