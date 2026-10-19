from tools.prepare_oa13_oa33_cap_stroke_panel import build_plan


def test_default_cap_stroke_panel_prioritizes_cov13():
    result = build_plan()
    split = result["identity_split"]
    assert result["parents"] == ["cov13"]
    assert len(result["profiles"]) == 3
    assert split["train_molecules"] == 128
    assert split["holdout_molecules"] == 32
    assert split["images"] == 480
    assert split["molecule_overlap"] == 0
    assert split["connection_identity_overlap"] == 0
    assert split["scaffold_overlap"] == 0
    assert result["gate"]["training_admissible"] is False


def test_optional_cov33_uses_same_identity_split_and_six_profiles():
    result = build_plan(include_cov33=True)
    assert result["parents"] == ["cov13", "cov33"]
    assert len(result["profiles"]) == 6
    assert result["identity_split"]["images"] == 960
    assert result["identity_split"]["source_split_preserved"] is True
    assert all(row["render_status"] == "not_rendered" for row in result["rows"])
    assert {row["changed_factor"] for row in result["rows"]} <= {"none", "bond_length_cap", "bond_line_width"}
