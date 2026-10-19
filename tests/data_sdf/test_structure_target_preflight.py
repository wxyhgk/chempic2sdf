"""Selection and journal tests only; no real corpus, models or identity calls."""

from copy import deepcopy

import pytest

from tools import preflight_structure_expansion_targets as check


def fixture():
    samples = [dict(sample_key="m__" + profile, molecule_id="m", profile_id=profile,
                    png_sha256="png-" + profile, rgb_sha256="rgb-" + profile) for profile in ("oa02", "oa00")]
    candidates = [dict(source="curriculum", molecule_id="m", sample_keys=[r["sample_key"] for r in samples], image_count=2)]
    return samples, candidates


def test_all_styles_preserved_without_source_mutation():
    samples, candidates = fixture()
    saved = deepcopy((samples, candidates))
    rows = check.select_rows(samples, candidates)
    assert (samples, candidates) == saved
    assert [r["sample_key"] for r in rows] == ["m__oa00", "m__oa02"]
    assert [r["decoded_rgb_sha256"] for r in rows] == ["rgb-oa00", "rgb-oa02"]


@pytest.mark.parametrize("mutation", ["duplicate_index", "duplicate_candidate", "duplicate_key", "count", "missing", "foreign", "profile"])
def test_bad_selection_is_rejected(mutation):
    samples, candidates = fixture()
    if mutation == "duplicate_index": samples.append(dict(samples[0]))
    elif mutation == "duplicate_candidate": candidates.append(dict(candidates[0]))
    elif mutation == "duplicate_key": candidates[0]["sample_keys"][1] = candidates[0]["sample_keys"][0]
    elif mutation == "count": candidates[0]["image_count"] = 1
    elif mutation == "missing": samples.pop()
    elif mutation == "foreign": samples[0]["molecule_id"] = "foreign"
    else: samples[0]["profile_id"] = "oa01"
    with pytest.raises(ValueError): check.select_rows(samples, candidates)


def test_failure_is_counted_in_full_style_denominator():
    selection = check.select_rows(*fixture())
    journal = [{**row, "compatible": i == 0} for i, row in enumerate(selection)]
    result = check.summarize_journal(selection, journal)
    assert result["checked_images"] == 2
    assert result["checked_molecules"] == result["compatible_images"] == result["failed_images"] == 1
    assert result["checked_images_by_profile"] == {"oa00": 1, "oa02": 1}
    assert result["compatible_images_by_profile"] == {"oa00": 1, "oa02": 0}


@pytest.mark.parametrize("mutation", ["missing", "order", "identity", "nonbool"])
def test_incomplete_or_changed_journal_cannot_pass(mutation):
    selection = check.select_rows(*fixture())
    journal = [{**row, "compatible": True} for row in selection]
    if mutation == "missing": journal.pop()
    elif mutation == "order": journal.reverse()
    elif mutation == "identity": journal[0]["molecule_id"] = "foreign"
    else: journal[0]["compatible"] = 1
    with pytest.raises(ValueError): check.summarize_journal(selection, journal)
