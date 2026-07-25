import torch

from chemvector.dense_bond_object import (
    BondProposalResidualAdapter,
    BondProposalResidualConfig,
    DenseBondObjectConfig,
    DenseBondObjectDetector,
)
from chemvector.train_bond_proposal_style_replay import selection_key


def _official(chemvision: float, full_v0: float) -> dict:
    return {
        "domains": {
            "chemvision": {
                "bond_proposal_recall": chemvision,
                "bond_deployed_exact": 100,
                "full_object_exact": 80,
            },
            "full_v0": {
                "bond_proposal_recall": full_v0,
                "bond_deployed_exact": 100,
                "full_object_exact": 80,
            },
        }
    }


def _style(all_renderings: int, mixed: int, coverage: float) -> dict:
    return {
        "overall": {"bond_coverage": coverage, "representable": coverage - 0.01},
        "paired": {
            "bond_coverage": {
                "all_renderings": all_renderings,
                "mixed": mixed,
            }
        },
    }


def test_selection_rejects_official_domain_regression() -> None:
    baseline = _official(0.91, 0.92)
    unsafe = selection_key(
        _official(0.92, 0.91), _style(100, 5, 0.95), baseline
    )
    safe = selection_key(baseline, _style(90, 8, 0.90), baseline)

    assert safe > unsafe


def test_selection_prefers_complete_paired_style_coverage_when_safe() -> None:
    baseline = _official(0.91, 0.92)
    first = selection_key(baseline, _style(90, 4, 0.91), baseline)
    second = selection_key(baseline, _style(91, 10, 0.90), baseline)

    assert second > first


def test_selection_rejects_downstream_exact_regression() -> None:
    baseline = _official(0.91, 0.92)
    candidate = _official(0.92, 0.93)
    candidate["domains"]["chemvision"]["bond_deployed_exact"] = 99

    unsafe = selection_key(candidate, _style(100, 4, 0.95), baseline)
    safe = selection_key(baseline, _style(90, 8, 0.90), baseline)

    assert safe > unsafe


def test_zero_residual_preserves_features_types_and_geometry() -> None:
    base = DenseBondObjectDetector(DenseBondObjectConfig())
    adapter = BondProposalResidualAdapter(base, BondProposalResidualConfig())
    features = torch.randn(2, 96, 8, 8)

    expected = base(features)
    actual = adapter(features)

    for name in (
        "object_features",
        "heatmap_logits",
        "endpoints",
        "display_logits",
        "chemical_logits",
        "direction_logits",
    ):
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    assert all(parameter.requires_grad for parameter in adapter.adapter_parameters())
    assert not any(parameter.requires_grad for parameter in adapter.base_detector.parameters())
