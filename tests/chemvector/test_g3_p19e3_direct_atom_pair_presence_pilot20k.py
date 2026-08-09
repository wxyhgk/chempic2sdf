import torch

from chemvector.training.graph.relations import (
    train_g3_p19e3_direct_atom_pair_presence_pilot20k as p19e3,
)


def _molecule_rows(index, scaffold):
    return [
        {
            "sample_key": f"k{index}_{variant}",
            "molecule_id": f"m{index}",
            "scaffold": scaffold,
            "variant": variant,
        }
        for variant in p19e3.VARIANTS
    ]


def test_select_internal_dev_excludes_pilot_molecules_and_scaffolds(monkeypatch):
    monkeypatch.setattr(p19e3, "DEV_MOLECULES", 2)
    monkeypatch.setattr(p19e3, "DEV_IMAGES", 8)
    pilot = _molecule_rows(0, "shared")
    full = pilot + _molecule_rows(1, "shared") + _molecule_rows(2, "a") + _molecule_rows(3, "b")
    selected, audit = p19e3.select_internal_dev640(pilot, full)
    assert len(selected) == 8
    assert {row["molecule_id"] for row in selected} == {"m2", "m3"}
    assert audit["pilot_molecule_overlap"] == 0
    assert audit["pilot_scaffold_overlap"] == 0


def test_balanced_presence_loss_is_averaged_per_image():
    logits = torch.tensor([2.0, -1.0, 1.0, -2.0], requires_grad=True)
    targets = torch.tensor([True, False, False, True])
    samples = torch.tensor([0, 0, 1, 1])
    losses = p19e3.balanced_presence_loss_by_sample(logits, targets, samples)
    expected = torch.stack(
        [
            p19e3.p19e1.balanced_presence_loss(logits[:2], targets[:2])["loss"],
            p19e3.p19e1.balanced_presence_loss(logits[2:], targets[2:])["loss"],
        ]
    ).mean()
    assert torch.allclose(losses["loss"], expected)
    losses["loss"].backward()
    assert logits.grad is not None


def test_capability_checks_require_distribution_and_hard_gains(monkeypatch):
    monkeypatch.setattr(p19e3, "DEV_IMAGES", 4)
    baseline = {
        "overall": {"skeleton_exact": 2},
        "by_variant": {
            "coordgen": {"skeleton_exact": 0},
            "original": {"skeleton_exact": 0},
            "rdkit": {"skeleton_exact": 1},
            "stress": {"skeleton_exact": 1},
        },
        "stress_by_operation": {
            "crossed_bonds": {"skeleton_exact": 0},
            "near_nonbonded": {"skeleton_exact": 0},
        },
    }
    best = {
        "overall": {"skeleton_exact": 12},
        "by_variant": {
            "coordgen": {"skeleton_exact": 1},
            "original": {"skeleton_exact": 0},
            "rdkit": {"skeleton_exact": 1},
            "stress": {"skeleton_exact": 10},
        },
        "stress_by_operation": {
            "crossed_bonds": {"skeleton_exact": 2},
            "near_nonbonded": {"skeleton_exact": 2},
        },
    }
    checks = p19e3.capability_checks(
        baseline,
        best,
        {"overall": {"rescue": 10, "break": 0}},
        frozen_states_unchanged=True,
        repeat_exact=True,
        split_audit={
            "selected_images": 4,
            "pilot_molecule_overlap": 0,
            "pilot_scaffold_overlap": 0,
        },
    )
    assert all(checks.values())
