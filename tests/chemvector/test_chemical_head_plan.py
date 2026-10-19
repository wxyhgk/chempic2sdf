import copy

import pytest

from chemvector.training.curriculum.chemical_head_plan import make_sample_plan


def row(index, *, smiles=None, domain="full_molecule", size=12, bond="AROMATIC", profile="reference"):
    key = f"{index:064x}"
    bond_counts = {"SINGLE": 4} if bond == "SINGLE" else {bond: 4}
    return {
        "source": "full_molecule_expansion" if domain == "full_molecule" else "curriculum",
        "sample_key": f"sample-{index}",
        "split": "train",
        "domain": domain,
        "canonical_smiles": smiles or f"C{index}N",
        "group_key": f"group-{index}",
        "meta_sha256": key,
        "pixel_sha256": f"{index + 1:064x}",
        "png_sha256": f"{index + 2:064x}",
        "graph_sha256": f"{index + 3:064x}",
        "molecule_id": f"molecule-{index}",
        "heavy_atoms": size,
        "bond_counts": bond_counts,
        "profile_id": profile,
        "stages": ["full_molecule"],
    }


def rows_with_style_variants(count=24):
    profiles = ("reference", "font_small", "line_thin")
    rows = []
    for index in range(count):
        for profile_index, profile in enumerate(profiles):
            rows.append(row(index * 10 + profile_index, smiles=f"C{index}N", profile=profile))
    return rows


def test_sample_plan_is_deterministic_and_uses_one_image_per_structure():
    rows = rows_with_style_variants()
    first = make_sample_plan(rows, count=18)
    second = make_sample_plan(list(reversed(rows)), count=18)
    assert first == second
    assert first["summary"]["images"] == 18
    assert first["summary"]["distinct_structures"] == 18
    assert len({record["canonical_smiles"] for record in first["samples"]}) == 18
    assert set(first["summary"]["profiles"]) == {"reference", "font_small", "line_thin"}


def test_sample_plan_interleaves_size_bond_and_domain_strata():
    rows = []
    for index in range(60):
        rows.append(row(index, domain="curriculum" if index % 3 == 0 else "full_molecule",
                        size=5 if index % 2 else 28,
                        bond=("TRIPLE", "DOUBLE", "AROMATIC", "SINGLE")[index % 4],
                        profile=("reference", "serif", "font_small")[index % 3]))
    plan = make_sample_plan(rows, count=24)
    assert plan["summary"]["images"] == 24
    assert set(plan["summary"]["domains"]) == {"curriculum", "full_molecule"}
    assert len(plan["summary"]["size_buckets"]) == 2
    assert set(plan["summary"]["dominant_bond_buckets"]) == {"TRIPLE", "DOUBLE", "AROMATIC", "SINGLE"}


def test_sample_plan_rejects_validation_and_insufficient_unique_structures():
    rows = rows_with_style_variants(3)
    validation = copy.deepcopy(rows)
    validation[0]["split"] = "validation"
    with pytest.raises(ValueError, match="train-split"):
        make_sample_plan(validation, count=2)
    with pytest.raises(ValueError, match="distinct structures"):
        make_sample_plan(rows, count=4)


def test_sample_plan_prefers_full_molecule_variant_for_duplicate_identity():
    curriculum = row(1, smiles="CCO", domain="curriculum", profile="reference")
    full = row(2, smiles="CCO", domain="full_molecule", profile="serif")
    other = row(3, smiles="CCN", domain="full_molecule", profile="reference")
    plan = make_sample_plan([curriculum, full, other], count=2)
    selected = {record["canonical_smiles"]: record for record in plan["samples"]}
    assert selected["CCO"]["domain"] == "full_molecule"

