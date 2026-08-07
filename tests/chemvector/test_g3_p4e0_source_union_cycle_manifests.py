from __future__ import annotations

from chemvector.evaluation.audits.atoms.build_g3_p4e0_source_union_cycle_manifests import (
    allocate_cycle_molecules,
)


def _molecule(index: int, count: int, scaffold: str | None = None) -> dict:
    return {
        "molecule_id": f"m{index}",
        "scaffold": scaffold or f"s{index}",
        "heavy_atoms": str(count),
    }


def test_allocation_is_blocked_and_confirm_disjoint(monkeypatch) -> None:
    from chemvector.evaluation import g3_p4_source_union_cycle as cycle

    monkeypatch.setattr(cycle, "CONFIRM_V2_BUCKET_MOLECULES", {
        "1-20": 1,
        "21-40": 1,
        "41-60": 1,
        "61-80": 1,
        "81-100": 1,
    })
    monkeypatch.setattr(cycle, "FULL_TRAIN_MOLECULES", 5)
    monkeypatch.setattr(cycle, "PILOT_TRAIN_MOLECULES", 2)
    counts = (10, 30, 50, 70, 90)
    molecules = [_molecule(0, 10, "blocked")]
    molecules.extend(
        _molecule(1 + bucket * 3 + offset, count)
        for bucket, count in enumerate(counts)
        for offset in range(3)
    )
    allocated, _ = allocate_cycle_molecules(
        molecules,
        blocked_molecules={"m0"},
        blocked_scaffolds={"blocked"},
    )
    train = {row["molecule_id"] for row in allocated["full_train"]}
    pilot = {row["molecule_id"] for row in allocated["pilot_train"]}
    confirm = {row["molecule_id"] for row in allocated["confirm_v2"]}
    assert len(train) == 5
    assert len(confirm) == 5
    assert pilot <= train
    assert not train & confirm
    assert "m0" not in train | confirm


def test_allocation_rejects_insufficient_high_count_bucket(monkeypatch) -> None:
    import pytest
    from chemvector.evaluation import g3_p4_source_union_cycle as cycle

    monkeypatch.setattr(cycle, "CONFIRM_V2_BUCKET_MOLECULES", {
        "1-20": 0,
        "21-40": 0,
        "41-60": 0,
        "61-80": 0,
        "81-100": 2,
    })
    with pytest.raises(RuntimeError, match="81-100"):
        allocate_cycle_molecules(
            [_molecule(1, 90)],
            blocked_molecules=set(),
            blocked_scaffolds=set(),
        )
