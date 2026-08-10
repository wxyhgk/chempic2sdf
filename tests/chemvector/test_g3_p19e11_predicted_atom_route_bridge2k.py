import torch

from chemvector.evaluation.audits.graph import (
    audit_g3_p19e11_predicted_atom_route_bridge2k as p19e11,
)


def _rows(molecule, scaffold):
    return [
        {
            "sample_key": f"{molecule}_{variant}",
            "molecule_id": molecule,
            "scaffold": scaffold,
            "variant": variant,
        }
        for variant in p19e11.p19e9.VARIANTS
    ]


def test_route_exact_requires_atom_exact():
    selected = {
        "fusion": torch.tensor([True, False]),
        "new": torch.tensor([True, False]),
        "old": torch.tensor([False, False]),
    }
    presence = torch.tensor([True, False])
    route, pair = p19e11.route_exact_flags(selected, presence, atom_exact=False)
    assert pair.tolist() == [True, True, False]
    assert route.tolist() == [False, False, False]
    route, _ = p19e11.route_exact_flags(selected, presence, atom_exact=True)
    assert route.tolist() == [True, True, False]


def test_screen_selection_keeps_complete_molecules(monkeypatch):
    monkeypatch.setattr(p19e11, "SCREEN_MOLECULES", 2)
    monkeypatch.setattr(p19e11, "SCREEN_IMAGES", 8)
    rows = _rows("a", "sa") + _rows("b", "sb") + _rows("c", "sc")
    selected, audit = p19e11.select_screen_rows(rows)
    assert len(selected) == 8
    assert audit["molecules"] == 2
    assert audit["scaffolds"] == 2
    assert audit["new_scaffolds_consumed"] == 0


def test_cache_zero_fills_non_router_eligible_features():
    record = {
        "sample_key": "a",
        "variant": "stress",
        "stress_operation": "crossed_bonds",
        "features": None,
        "route_exact": torch.tensor([False, False, False]),
        "pair_exact": torch.tensor([True, True, True]),
        "router_eligible": False,
        "atom_exact": False,
        "bond_representable": False,
        "target_atoms": 1,
        "selected_atoms": 1,
        "matched_atoms": 0,
        "target_bonds": 0,
        "represented_bonds": 0,
        "routes_observably_disagree": False,
    }
    cache = p19e11._records_to_cache([record], 78)
    assert cache["features"].shape == (1, 78)
    assert not bool(cache["features"].any())
    assert not bool(cache["router_eligible"][0])


def test_structural_targets_maps_g3_bond_type_protocol():
    batch = {
        "atom_anchors": torch.zeros(1, 2, 2),
        "atom_mask": torch.ones(1, 2, dtype=torch.bool),
        "bond_atom_refs": torch.tensor([[[0, 1]]]),
        "bond_types": torch.tensor([[3]]),
        "bond_mask": torch.ones(1, 1, dtype=torch.bool),
        "unrelated": torch.tensor([1]),
    }
    targets = p19e11.structural_targets(batch, torch.device("cpu"))
    assert set(targets) == {
        "atom_anchors",
        "atom_mask",
        "bond_atom_refs",
        "bond_chemical",
        "bond_mask",
    }
    assert targets["bond_chemical"].tolist() == [[3]]


def test_force_keep_for_ineligible_is_not_in_place():
    source = torch.tensor([1, 2, 1])
    selected = p19e11.force_keep_for_ineligible(
        source, torch.tensor([True, False, True])
    )
    assert source.tolist() == [1, 2, 1]
    assert selected.tolist() == [1, 0, 1]
