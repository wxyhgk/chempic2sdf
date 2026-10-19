"""Deterministic, identity-disjoint split for the public current-initial cohort."""

from __future__ import annotations

from collections import defaultdict


def split_group_disjoint(rows: list[dict], train_target: int = 120) -> tuple[list[dict], list[dict], dict]:
    """Split rows by ``group_key`` without changing row order semantics.

    Rows are sorted by sample key for reproducibility. Whole groups are added
    until the requested train size is reached; a group is never split. The
    resulting train size may exceed ``train_target`` by the final group size.
    """
    if train_target < 1 or train_target >= len(rows):
        raise ValueError("train_target must be inside the cohort")
    ordered = sorted(rows, key=lambda row: row["sample_key"])
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in ordered:
        key = row.get("group_key")
        if not isinstance(key, str) or not key:
            raise ValueError("Every cohort row requires a nonempty group_key")
        groups[key].append(row)
    train: list[dict] = []
    used: set[str] = set()
    for row in ordered:
        group = row["group_key"]
        if group in used:
            continue
        train.extend(groups[group])
        used.add(group)
        if len(train) >= train_target:
            break
    train_keys = {row["sample_key"] for row in train}
    holdout = [row for row in ordered if row["sample_key"] not in train_keys]
    train_groups = {row["group_key"] for row in train}
    holdout_groups = {row["group_key"] for row in holdout}
    if train_groups & holdout_groups:
        raise AssertionError("group-disjoint split invariant failed")
    train_molecules = {row["molecule_id"] for row in train}
    holdout_molecules = {row["molecule_id"] for row in holdout}
    train_smiles = {row["canonical_smiles"] for row in train}
    holdout_smiles = {row["canonical_smiles"] for row in holdout}
    if train_molecules & holdout_molecules or train_smiles & holdout_smiles:
        raise AssertionError("molecule/canonical split invariant failed")
    return train, holdout, {
        "requested_train_images": train_target,
        "train_images": len(train),
        "holdout_images": len(holdout),
        "train_groups": len(train_groups),
        "holdout_groups": len(holdout_groups),
        "group_overlap": len(train_groups & holdout_groups),
        "molecule_overlap": len(train_molecules & holdout_molecules),
        "canonical_smiles_overlap": len(train_smiles & holdout_smiles),
    }


def assert_public_training_rows(rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Empty cohort")
    if any(row.get("split") != "train" for row in rows):
        raise ValueError("Current-initial cohort must contain released training rows")
