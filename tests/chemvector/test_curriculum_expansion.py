"""Small CPU-only train expansion fixtures; never read real or sealed corpora."""

from copy import deepcopy

import pytest
from rdkit import Chem

from chemvector.training.curriculum.admission import identities
from chemvector.training.curriculum.expansion_plan import (
    PROFILE_IDS, make_plan, prefilter, selected_profiles, validation_protection,
)
from chemvector.training.curriculum.full_molecule_data import prepare_selected
from tests.chemvector.test_full_molecule_data import exclusion, molecule


def manifest(*smiles):
    rows = []
    for i, value in enumerate(smiles or ("CO",)):
        value = Chem.MolToSmiles(Chem.MolFromSmiles(value))
        rows.append(dict(split="validation", molecule_id=f"val-{i}", canonical_smiles=value,
                         group_key=identities(value)["group_key"]))
    return {"samples": rows}


def test_six_fixed_styles_do_not_change_existing_renderer_profiles():
    values = selected_profiles()
    assert tuple(p.profile_id for p in values) == PROFILE_IDS
    assert len(values) == 6 and len({p.font_family for p in values}) == 2
    assert {p.font_size for p in values} == {14, 24, 32}
    assert all(p.width == p.height == 768 for p in values)


@pytest.mark.parametrize("val,candidate", [
    ("CCCCCCCN", "CCCCCCC[NH3+]"),
    ("CCc1ccccc1", "CCCc1ccccc1"),
    ("CCCCCCC[13CH3]", "CCCCCCCC"),
    ("CCCC[C@H](F)CCO", "CCCC[C@@H](F)CCO"),
])
def test_protected_validation_excludes_connectivity_isotopes_stereo_and_core(val, candidate):
    rows = [molecule(candidate, 0), molecule("CCCCCCCO", 1)]
    kept, rejected, _ = prefilter(rows, manifest(val), exclusion(), [], [])
    assert [r["molecule_id"] for r in kept] == [rows[1]["molecule_id"]]
    assert rejected[0]["reason"] == "existing_validation_identity_connectivity_or_core"


def test_public_unresolved_floor_is_conservative_and_not_admission():
    rows = [molecule("CCCCCCNN", 0), molecule("CCCCCCCO", 1)]
    unresolved = [dict(sample_id="public-opaque", native_probe={"key": None},
        inventory={"guaranteed_non_hydrogen_inventory": {"N": 2}})]
    kept, rejected, _ = prefilter(rows, manifest(), exclusion(), [], unresolved)
    assert kept == [rows[1]]
    assert rejected[0]["reason"] == "possible_overlap_with_cached_unresolved_public_record"
    assert rejected[0]["public_record_ids"] == ["public-opaque"]
    unresolved[0]["inventory"] = None
    kept, rejected, _ = prefilter(rows, manifest(), exclusion(), [], unresolved)
    assert not kept and len(rejected) == 2


def test_additional_public_keys_are_positive_exclusion_only():
    rows = [molecule("CCCCCCCC", 0), molecule("CCCCCCCO", 1)]
    kept, rejected, _ = prefilter(rows, manifest(), exclusion(), [{"inchikey": rows[0]["inchikey"]}], [])
    assert kept == [rows[1]] and rejected[0]["reason"] == "cached_public_key_or_connectivity"


def test_plan_is_order_independent_does_not_relabel_old_train_or_touch_validation():
    rows = [molecule("C" * n, n) for n in range(8, 12)]
    m = manifest()
    m["samples"].append(dict(split="train", canonical_smiles=rows[0]["canonical_isomeric_smiles"]))
    before = deepcopy(m)
    plan = make_plan(rows, m, exclusion(), [], [], count=4)
    assert plan == make_plan(list(reversed(rows)), m, exclusion(), [], [], count=4)
    assert plan["planned_images"] == 24
    assert plan["selected_already_in_previous_train"] == 1
    assert not plan["training_authorized"] and not plan["validation_changed"]
    assert plan["public_review"] == "cached_record_prefilter_only_not_new_candidate_clearance"
    assert m == before


def test_missing_or_inconsistent_validation_is_not_ignored():
    with pytest.raises(ValueError, match="protection"):
        validation_protection({"samples": []})
    m = manifest()
    m["samples"][0]["group_key"] = "wrong"
    with pytest.raises(ValueError, match="group"):
        validation_protection(m)


def test_no_padding_when_protected_candidates_leave_too_few():
    rows = [molecule("CCCCCCCC", 0), molecule("CCCCCCCO", 1)]
    with pytest.raises(ValueError, match="Not enough"):
        make_plan(rows, manifest("CCCCCCCC"), exclusion(), [], [], count=2)


def test_nontrain_rows_fail_before_asset_reads():
    row = molecule("CCCCCCCC", 0)
    row["split"] = "validation"
    with pytest.raises(ValueError, match="train-only"):
        prefilter([row], manifest(), exclusion(), [], [])
    with pytest.raises(ValueError, match="train-only"):
        prepare_selected("must-not-open", [row], [], {}, {}, {})


def test_pixel_duplicates_count_once_and_conflicting_labels_never_qualify():
    from chemvector.training.curriculum.expansion_inventory import unique_candidates
    def row(key, digest="same", smiles="CC", compatible=True, profile="serif"):
        return dict(sample_key=key, pixel_sha256=digest, canonical_smiles=smiles,
                    target_compatible=compatible, profile_id=profile)
    rows = [row("a"), row("b", profile="reference"), row("c", compatible=False),
            row("d", "conflict", "CN"), row("e", "conflict", "CO"), row("f", "bad", compatible=False)]
    selected, duplicates, conflicts = unique_candidates(rows)
    assert [r["sample_key"] for r in selected] == ["b"]
    assert duplicates == ["a"]
    assert conflicts == [{"pixel_sha256": "conflict", "sample_keys": ["d", "e"]}]
    assert unique_candidates(list(reversed(rows))) == (selected, duplicates, conflicts)


def test_chemistry_gate_excludes_whole_identity_without_repairs():
    from Data.sdf.chemvector_curriculum_v0_1.catalog import merged_records
    from chemvector.training.curriculum.expansion_plan import chemistry_preflight
    bad = "CCCC1=C2OC(=O)C3=C2[C@@H](CC1)C[C@H]1C2CC[C@@]4(C5=C2[C@@H](CN2C(=O)C=CC2=O)c2ccnc(c2)NCSSC[C@H](C)C/C=C\\4OC5=O)[C@@H]31"
    entries = merged_records([dict(record_id=str(i), name=str(i), stage="full_molecule", concepts=[], smiles=s)
                              for i, s in enumerate(("CCO", bad))])
    before = deepcopy(entries)
    supported, failed = chemistry_preflight(entries)
    assert len(supported) == len(failed) == 1 and entries == before
    assert supported[0]["canonical_smiles"] == "CCO"
    assert failed[0]["reason"] == "SDF roundtrip changed chemical identity"
    assert failed[0]["excluded_all_profiles"] == list(PROFILE_IDS)
    assert not failed[0]["chemical_labels_modified"]


def test_chemistry_gate_does_not_reclassify_unknown_bugs_as_rejections(monkeypatch):
    from chemvector.training.curriculum.expansion_plan import chemistry_preflight
    def fail(*args):
        raise ValueError("unexpected ownership bug")
    monkeypatch.setattr("Data.sdf.chemvector_curriculum_v0_1.rendering.sdf_bytes", fail)
    with pytest.raises(ValueError, match="unexpected"):
        chemistry_preflight([dict(molecule_id="a", canonical_smiles="CCO")])


def test_inventory_binds_preflight_and_hashes_pixels_without_authorizing_training(tmp_path):
    from Data.sdf.chemvector_curriculum_v0_1.build import build
    from Data.sdf.chemvector_curriculum_v0_1.catalog import merged_records
    from chemvector.training.curriculum.expansion import write_json
    from chemvector.training.curriculum.expansion_inventory import run
    from chemvector.training.curriculum.full_molecule_data import file_hash
    from chemvector.training.curriculum.preflight import run as preflight
    root = tmp_path / "expanded"
    root.mkdir()
    entries = merged_records([dict(record_id="ethanol", name="ethanol", stage="full_molecule", concepts=[], smiles="CCO")])
    summary = build(root / "dataset", entries=entries, selected_profiles=selected_profiles()[:1], workers=1,
                    gallery=False, typography="stereo-aware-v1")
    write_json(root / "plan.json", {"planned_images": 1})
    write_json(root / "report.json", {"plan_sha256": file_hash(root / "plan.json"), "chemistry_excluded_images": 0})
    checked = tmp_path / "checked"
    preflight(root / "dataset", checked)
    report = run(root, checked, tmp_path / "inventory")
    assert report["compatible_unique_candidate_images"] == summary["accepted_images"] == 1
    assert report["admitted_new_train_images"] == 0 and not report["training_authorized"]
    assert not report["minimum_candidate_count_reached"]
    write_json(root / "unrelated.json", {})
    # A different plan cannot silently inherit the old generation binding.
    (root / "plan.json").write_text('{"planned_images": 2}')
    with pytest.raises(ValueError, match="plan binding"):
        run(root, checked, tmp_path / "must-not-publish")
    assert not (tmp_path / "must-not-publish").exists()
