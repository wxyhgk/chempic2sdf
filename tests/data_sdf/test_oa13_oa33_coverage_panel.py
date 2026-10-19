from pathlib import Path

from tools.prepare_oa13_oa33_coverage_panel import build_plan


ROOT = Path(__file__).resolve().parents[2]


def test_coverage_panel_is_fixed_eight_profiles_and_320_identity_split():
    result = build_plan()
    split = result["identity_split"]
    assert split["train_molecules"] == 256
    assert split["holdout_molecules"] == 64
    assert split["images"] == 2560
    assert split["molecule_overlap"] == 0
    assert split["connection_identity_overlap"] == 0
    assert split["scaffold_overlap"] == 0
    assert len(result["profiles"]) == 8
    assert result["gate"]["training_admissible"] is False


def test_coverage_profile_changes_are_single_factor():
    result = build_plan()
    profiles = result["profiles"]
    assert {p["parent_profile_id"] for p in profiles} == {"oa13", "oa33"}
    assert all(p["changed_factor"] in {"none", "font_size", "rotation_degrees", "font_family"}
               for p in profiles)
    assert all(row["render_status"] == "not_rendered" for row in result["rows"])
