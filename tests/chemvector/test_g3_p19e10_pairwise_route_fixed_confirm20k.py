import pytest
import torch

from chemvector.evaluation.audits.graph import (
    audit_g3_p19e10_pairwise_route_fixed_confirm20k as p19e10,
)


def _rows(molecule, scaffold):
    return [
        {
            "sample_key": f"{molecule}_{variant}",
            "molecule_id": molecule,
            "scaffold": scaffold,
            "variant": variant,
        }
        for variant in p19e10.VARIANTS
    ]


def test_clopper_pearson_zero_success_matches_closed_form():
    observed = p19e10.clopper_pearson_upper(0, 200)
    expected = 1.0 - 0.05 ** (1.0 / 200)
    assert observed == pytest.approx(expected, abs=1e-12)
    assert observed < 0.02


def test_clopper_pearson_rejects_invalid_counts():
    with pytest.raises(ValueError):
        p19e10.clopper_pearson_upper(2, 1)
    with pytest.raises(ValueError):
        p19e10.clopper_pearson_upper(0, 1, alpha=1.0)


def test_selected_outcome_counts_uses_chosen_route():
    actions = torch.tensor([0, 1, 2])
    cache = {
        "outcomes": torch.tensor([[0, 0], [2, 3], [3, 2]]),
        "variants": ["rdkit", "stress", "original"],
        "stress_operations": [None, "crossed_bonds", None],
    }
    result = p19e10.selected_outcome_counts(actions, cache)
    assert result["by_route"]["USE_NEW"]["RESCUE"] == 1
    assert result["by_route"]["USE_OLD"]["RESCUE"] == 1
    assert sum(
        value for values in result["by_route"].values() for value in values.values()
    ) == 2


def test_confirm_selection_excludes_prior_scaffolds(monkeypatch):
    monkeypatch.setattr(p19e10, "CONFIRM_MOLECULES", 2)
    monkeypatch.setattr(p19e10, "CONFIRM_IMAGES", 8)
    excluded = _rows("x", "excluded")
    full = excluded + _rows("a", "sa") + _rows("b", "sb")
    rows, audit = p19e10.select_confirm_rows(full, excluded)
    assert len(rows) == 8
    assert audit["scaffolds"] == 2
    assert audit["excluded_molecule_overlap"] == 0
    assert audit["excluded_scaffold_overlap"] == 0
