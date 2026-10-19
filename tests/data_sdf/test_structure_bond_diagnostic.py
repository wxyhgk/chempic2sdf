"""Deterministic synthetic panel selection; no checkpoints or real images."""

from copy import deepcopy

import pytest

from tools import diagnose_structure_expansion_bonds as diagnostic


def cohort():
    candidates, molecules = [], []
    for i, smi in enumerate(("C#N", "CCC#N", "CCCCC#N", "CCCCCCC#N",
                            "C=O", "CCC=O", "CCCCC=O", "CCCCCCC=O",
                            "c1ccccc1", "Cc1ccccc1", "CCCc1ccccc1", "CCCCCc1ccccc1",
                            "CO", "CCCO", "CCCCCO", "CCCCCCCO")):
        mid = f"m{i}"
        candidates.append(dict(source="curriculum", molecule_id=mid, canonical_smiles=smi, image_count=2,
                               sample_keys=[mid+"__"+style for style in diagnostic.PROFILES]))
        molecules.append(dict(molecule_id=mid, canonical_smiles=smi, split="quarantine", training_authorized=False))
    return candidates, molecules


def test_deterministic_size_spread_and_paired_styles():
    inputs = cohort()
    original = deepcopy(inputs)
    rows = diagnostic.select_panel(*inputs, per_kind=2)
    assert inputs == original
    assert len(rows) == 16 and len({r["molecule_id"] for r in rows}) == 8
    assert rows == diagnostic.select_panel(*[list(reversed(v)) for v in inputs], per_kind=2)
    assert [r["selected_rank"] for r in rows[:4]] == [1, 1, 3, 3]
    for left, right in zip(rows[::2], rows[1::2]):
        assert left["molecule_id"] == right["molecule_id"]
        assert (left["profile_id"], right["profile_id"]) == diagnostic.PROFILES


@pytest.mark.parametrize("mutation", ["duplicate", "source", "training", "profile", "count", "identity", "coverage"])
def test_invalid_panel_rejected(mutation):
    candidates, molecules = cohort()
    if mutation == "duplicate": candidates.append(candidates[0])
    elif mutation == "source": candidates[0]["source"] = "foreign"
    elif mutation == "training": molecules[0]["training_authorized"] = True
    elif mutation == "profile": candidates[0]["sample_keys"][0] = "m0__oa02"
    elif mutation == "count": candidates[0]["image_count"] = 1
    elif mutation == "identity": molecules[0]["canonical_smiles"] = "CO"
    else: candidates = candidates[4:]
    with pytest.raises(ValueError): diagnostic.select_panel(candidates, molecules, per_kind=2)


def test_cpu_guard_prevents_any_input_read(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="CPU-only"): diagnostic.run()
