"""Pure synthetic selection/metrics; no checkpoints or real images."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from chemvector.training.curriculum.evaluation import FrozenOCRPredictor
from tests.chemvector.test_curriculum_pair_gap_diagnostic import args
from tools import diagnose_curriculum_v02_bond_reachability as diagnostic


def panel_inputs():
    candidates, molecules = [], []
    for i, smiles in enumerate(("C#N", "C=O", "c1ccccc1", "CO")):
        mid = f"m{i}"
        profiles = ["font_large", "line_thin"] if i == 0 else ["font_large", "reference"]
        candidates.append(dict(source="curriculum", molecule_id=mid, canonical_smiles=smiles,
                               image_count=2, sample_keys=[mid + "__" + style for style in profiles]))
        molecules.append(dict(molecule_id=mid, canonical_smiles=smiles, training_authorized=False, split="quarantine"))
    return candidates, molecules


def test_fixed_selection_surviving_profiles_and_immutable():
    candidates, molecules = panel_inputs()
    original = deepcopy((candidates, molecules))
    result = diagnostic.select_panel(candidates, molecules, per_kind=1)
    assert len(result) == 4 and [r["coverage_bucket"] for r in result] == list(diagnostic.KINDS)
    assert result[0]["profile_id"] == "font_large"  # Dedup removed reference, do not restore it.
    assert all(r["profile_id"] == "reference" for r in result[1:])
    assert diagnostic.select_panel(list(reversed(candidates)), list(reversed(molecules)), per_kind=1) == result
    assert (candidates, molecules) == original


@pytest.mark.parametrize("mutation", ["duplicate", "missing_kind", "not_quarantine", "bad_profile", "wrong_identity"])
def test_panel_rejects_scope_or_coverage_changes(mutation):
    candidates, molecules = panel_inputs()
    if mutation == "duplicate": candidates.append(candidates[0])
    elif mutation == "missing_kind": candidates.pop(0)
    elif mutation == "not_quarantine": molecules[0]["training_authorized"] = True
    elif mutation == "bad_profile": candidates[0]["sample_keys"][0] = "m0__not_in_plan"
    else: molecules[0]["canonical_smiles"] = "CC"
    with pytest.raises(ValueError):
        diagnostic.select_panel(candidates, molecules, per_kind=1)


def inputs(present=True, chemical="SINGLE"):
    _, _, batch, raw, structure, _, config, threshold = args(present=present, chemical=chemical)
    batch["bond_ids"] = [[0]]
    return raw, structure["atoms"], {k: structure[k] for k in ("pairs", "output", "chemistry")}, batch, config, threshold


@pytest.mark.parametrize("present,kind,stage", [(True, "SINGLE", "reachable_correct"),
    (False, "SINGLE", "reachable_missing"), (True, "DOUBLE", "reachable_wrong_type")])
def test_reachable_stages_keep_target_denominator(present, kind, stage):
    result = diagnostic.measure(*inputs(present, kind))
    assert result["bond_stages"][stage] == sum(result["bond_stages"].values()) == 1
    assert result["coverage"]["reachable_positive_pairs"] == 1


@pytest.mark.parametrize("raw_gap", [False, True])
def test_raw_and_selector_gaps_remain_distinct(raw_gap):
    values = list(inputs())
    if raw_gap: values[0]["atom_mask"][0, 1] = False
    values[1]["atom_mask"][0, 1] = False
    values[2] = dict(pairs=torch.empty(0, 2, dtype=torch.long), chemistry=torch.empty(0, dtype=torch.long),
                     output={"presence_logits": torch.empty(0), "chemical_logits": torch.empty(0, 4)})
    result = diagnostic.measure(*values)
    stage = "raw_correspondence_gap" if raw_gap else "selection_correspondence_gap"
    assert result["bond_stages"][stage] == 1 and result["pair_contract"]["unreachable_bonds"] == 1


def test_unknown_is_not_negative_and_summary_does_not_claim_accuracy():
    values = list(inputs())
    values[1]["atom_mask"][0, 2] = True
    values[2] = dict(pairs=torch.tensor([[0, 1], [0, 2], [1, 2]]), chemistry=torch.zeros(3, dtype=torch.long),
                     output={"presence_logits": torch.full((3,), 10.), "chemical_logits": torch.zeros(3, 4)})
    metrics = diagnostic.measure(*values)
    assert metrics["unknown_present_pairs"] == 2 and metrics["confirmed_false_positive_pairs"] == 0
    report = diagnostic.summarize([{"sample_key": "m", "metrics": metrics}])
    assert report["images"] == 1 and "accuracy" not in report and "exact" not in report
    assert report["by_chemical_type"]["SINGLE"]["targets"] == 1
    with pytest.raises(ValueError):
        diagnostic.summarize([{"sample_key": "m", "metrics": metrics}] * 2)


def test_nonfinite_predictions_fail_closed():
    values = list(inputs())
    values[2]["output"]["chemical_logits"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="Invalid fixed"):
        diagnostic.measure(*values)


def test_pair_path_parity_with_existing_predictor_without_ocr():
    raw, selected, predicted, _, _, threshold = inputs()
    selected.update(atom_features=torch.zeros(1, 3, 2), atom_scores=torch.ones(1, 3))
    def pair(*args):
        return predicted["output"]
    model = SimpleNamespace(pair=pair)
    actual = diagnostic.predict_pair(model, torch.zeros(1), selected, threshold)
    reference = FrozenOCRPredictor.from_predicted_atoms(SimpleNamespace(provenance={"pair_threshold": threshold}),
                                                        pair, torch.zeros(1), selected)
    for key in ("pairs", "chemistry"):
        assert torch.equal(actual[key], reference[key])
    for key in ("presence_logits", "chemical_logits"):
        assert torch.equal(actual["output"][key], reference["output"][key])


def test_cpu_and_duration_guards(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="CPU-only"):
        diagnostic.run(tmp_path / "forbidden")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(ValueError, match="300"):
        diagnostic.run(tmp_path / "too-long", max_seconds=301)


def test_real_prediction_snapshot_container_contract():
    raw, selected, predicted, batch, config, threshold = inputs()
    before = diagnostic.prediction_digest(raw, selected, predicted)
    diagnostic.measure(raw, selected, predicted, batch, config, threshold)
    assert diagnostic.prediction_digest(raw, selected, predicted) == before
    predicted["output"]["presence_logits"][0] += 1
    assert diagnostic.prediction_digest(raw, selected, predicted) != before
