"""CPU-only interface selection, caching, tracing and no-update contracts."""

from collections import Counter
from copy import deepcopy
from types import SimpleNamespace

from PIL import Image
import pytest
import torch

from chemvector.training.curriculum import evaluation
from chemvector.training.curriculum.interface_diagnostic import (
    interface_delta, run_cases, summarize, tensor_tree_digest, write_json,
)
from chemvector.training.curriculum.interface_plan import (
    ARMS, PER_STYLE, VARIANTS, select_cases, stratum,
)
from chemvector.training.curriculum.validation_truth import canonical_molblock


def selection_inputs():
    rows, initial, last = [], [], []
    for group, count in PER_STYLE:
        for variant in VARIANTS:
            for i in range(count + 2):
                key = f"{group}_{variant}_{i}"
                rows.append(dict(source="dev", sample_key=key, molecule_id=key, domain="full_molecule",
                                 split="validation", canonical_smiles="CN", variant=variant))
                common = dict(key="dev:" + key, expected_smiles="CN", domain="full_molecule")
                initial.append(dict(common, exact=group != "stable_wrong",
                                    predicted_smiles="CC" if group == "stable_wrong" else "CN"))
                last.append(dict(common, exact=group == "stable_correct", predicted_smiles=(
                    "CN" if group == "stable_correct" else None if group == "broken_other" else "CC")))
    return {"samples": rows}, {"predictions": initial}, {"predictions": last}


def test_selection_is_fixed_distinct_style_stratified_and_order_independent():
    manifest, initial, last = selection_inputs()
    plan = select_cases(manifest, initial, last)
    assert len(plan["samples"]) == len({r["molecule_id"] for r in plan["samples"]}) == 24
    assert plan["arms"] == list(ARMS) and plan["optimizer_steps"] == 0
    assert Counter((r["stratum"], r["variant"]) for r in plan["samples"]) == {
        (g, v): n for g, n in PER_STYLE for v in VARIANTS}
    manifest["samples"].reverse()
    initial["predictions"].reverse()
    last["predictions"].reverse()
    assert select_cases(manifest, initial, last) == plan


@pytest.mark.parametrize("change", ["duplicate", "missing", "truth", "domain", "shortage"])
def test_selection_fails_closed_on_changed_records_or_insufficient_strata(change):
    manifest, initial, last = selection_inputs()
    if change == "duplicate":
        last["predictions"].append(last["predictions"][0])
    elif change == "missing":
        last["predictions"].pop()
    elif change == "truth":
        last["predictions"][0]["expected_smiles"] = "CO"
    elif change == "domain":
        last["predictions"][0]["domain"] = "curriculum"
    else:
        for record in last["predictions"]:
            record["exact"] = True
    with pytest.raises(ValueError):
        select_cases(manifest, initial, last)


@pytest.mark.parametrize("left,right,expected", [
    ("CN", "CC", "broken_composition"), ("CN", "C[NH3+]", "broken_composition"),
    ("CC", "[13CH3]C", "broken_composition"), ("CC", "C", "broken_other"),
    ("CN", None, "broken_other"),
])
def test_composition_stratum_preserves_charge_and_isotope(left, right, expected):
    assert stratum({"exact": True, "predicted_smiles": left}, {"exact": False, "predicted_smiles": right}) == expected


def synthetic_predictor(monkeypatch):
    atoms = {"atom_mask": torch.tensor([[True, True]]), "atom_geometry": torch.tensor([[[.3, .5], [.7, .5]]]),
             "atom_features": torch.zeros(1, 2, 4), "atom_scores": torch.ones(1, 2)}
    output = {"presence_logits": torch.tensor([10.]), "chemical_logits": torch.tensor([[10., -10., -10., -10.]])}
    def structure(*args):
        assert not torch.is_grad_enabled()
        return torch.zeros(1), None, None, None, None, None, atoms
    monkeypatch.setattr(evaluation, "structure_forward", structure)
    predictor = evaluation.FrozenOCRPredictor.__new__(evaluation.FrozenOCRPredictor)
    predictor.provenance = dict(atom_threshold=-.25, pair_threshold=.85, role_threshold=.85)
    predictor.ocr = lambda crops: torch.zeros(len(crops), 3, 2)
    monkeypatch.setattr(evaluation, "decode_ctc_with_score", lambda row: ("N", .99))
    def bbox(pixels, anchors, indices, rows):
        return {"centers": anchors[rows, indices], "half_extents": torch.ones(len(indices), 2) * .02}
    model = SimpleNamespace(pair=lambda *args: output, role=lambda *args: torch.tensor([-10., 10.]), bbox=bbox)
    return predictor, model


def test_cached_and_default_inference_agree_trace_is_inert_and_structure_unchanged(monkeypatch):
    predictor, model = synthetic_predictor(monkeypatch)
    pixels, image = torch.ones(1, 3, 768, 768), Image.new("RGB", (768, 768), "white")
    default = predictor(model, None, pixels, image)
    structure = predictor.predict_structure(model, None, pixels)
    digest, trace = tensor_tree_digest(structure), {}
    cached = predictor.from_structure(structure, model.role, model.bbox, pixels, image, trace=trace)
    assert default == cached
    assert canonical_molblock(cached["molblock"]) == "CN"
    assert tensor_tree_digest(structure) == digest
    assert [r["text"] for r in trace["roles"]] == [False, True]
    assert trace["ocr"][0]["candidate"] == 1 and trace["ocr"][0]["text"] == "N"
    assert trace["ocr_application"]["parsed_text"] == 1
    no_text = lambda *args: torch.tensor([-10., -10.])
    implicit_trace = {}
    implicit = predictor.from_structure(structure, no_text, model.bbox, pixels, image, trace=implicit_trace)
    assert canonical_molblock(implicit["molblock"]) == "CC" and implicit_trace["ocr"] == []
    assert tensor_tree_digest(structure) == digest
    trace["atoms_after_fallback"][0]["anchor"][0] = 999
    assert predictor(model, None, pixels, image) == default


@pytest.mark.parametrize("bad", ["role", "bbox"])
def test_nonfinite_interface_predictions_are_rejected(monkeypatch, bad):
    predictor, model = synthetic_predictor(monkeypatch)
    if bad == "role":
        model.role = lambda *args: torch.tensor([float("nan"), 0.])
    else:
        model.bbox = lambda *args: {"centers": torch.tensor([[float("nan"), .5]]), "half_extents": torch.ones(1, 2)}
    with pytest.raises(ValueError, match="Non-finite text"):
        predictor(model, None, torch.ones(1, 3, 768, 768), Image.new("RGB", (768, 768)))


def test_delta_records_open_close_ocr_and_disallows_changed_candidates():
    before = {"roles": [{"candidate": 0, "text": False}, {"candidate": 1, "text": True}],
              "ocr": [{"candidate": 1, "text": "N", "center": [.2, .3], "half_extent": [.02, .03]}]}
    after = deepcopy(before)
    after["roles"][0]["text"] = True
    after["ocr"][0]["text"] = "O"
    after["ocr"][0]["center"][0] += .1
    delta = interface_delta(before, after)
    assert delta["opened_candidates"] == [0] and delta["closed_candidates"] == []
    assert delta["changed_ocr"] == [{"candidate": 1, "before": "N", "after": "O"}]
    assert delta["box_coordinate_absolute_delta_pixels"] == pytest.approx(76.8)
    after["roles"].pop()
    with pytest.raises(ValueError, match="same candidates"):
        interface_delta(before, after)


def test_runner_reuses_last_structure_without_targets_or_weight_updates(monkeypatch, tmp_path):
    predictor, initial = synthetic_predictor(monkeypatch)
    last = SimpleNamespace(**vars(initial))
    last.role = lambda *args: torch.tensor([-10., -10.])
    last.bbox = lambda *args: initial.bbox(*args)
    calls = []
    interfaces = []
    original = predictor.predict_structure
    def counted(model, config, pixels):
        calls.append(model)
        return original(model, config, pixels)
    monkeypatch.setattr(predictor, "predict_structure", counted)
    original_interface = predictor.from_structure
    def record_interface(structure, role, bbox, pixels, image, *, trace):
        interfaces.append((role is initial.role, bbox is initial.bbox))
        return original_interface(structure, role, bbox, pixels, image, trace=trace)
    monkeypatch.setattr(predictor, "from_structure", record_interface)
    row = dict(source="dev", sample_key="a", split="validation", domain="full_molecule")
    selection = dict(key="dev:a", stratum="broken_composition", variant="original", molecule_id="a")
    sample = {"image": Image.new("RGB", (768, 768), "white"), "purpose": "diagnostic",
              "graph": {"canonical_smiles": "CN"}}
    saved = {k: {"dev:a": dict(predicted_smiles=s, expected_smiles="CN", valid=True, exact=k == "initial", fallback=False)}
             for k, s in (("initial", "CN"), ("last", "CC"))}
    cases = run_cases({"samples": [selection]}, {"samples": [row]}, lambda *args: sample,
                     predictor, dict(initial=initial, last=last), None, saved, tmp_path)
    assert len(calls) == 2
    assert interfaces == [(True, True), (False, False), (True, False), (False, True), (True, True)]
    case = cases[0]
    assert all(r["matches"] for r in case["saved_replay"].values())
    assert [case["arms"][a]["exact"] for a in ARMS] == [True, False, True, False, True]
    assert len({case["arms"][a]["structure_sha256"] for a in ARMS[1:]}) == 1
    assert summarize(cases)["all_diagnostic_cases"]["arms"]["last_initial_role"]["rescued_vs_last"] == 1
    assert (tmp_path / "a.json").exists()
    with pytest.raises(FileExistsError):
        write_json(tmp_path / "a.json", {})


@pytest.mark.parametrize("bad", ["purpose", "split", "domain"])
def test_runner_rejects_non_diagnostic_inputs_before_inference(tmp_path, bad):
    row = dict(source="dev", sample_key="a", split="validation", domain="full_molecule")
    sample = {"purpose": "diagnostic"}
    if bad == "purpose":
        sample["purpose"] = "training"
    else:
        row[bad] = "train" if bad == "split" else "curriculum"
    with pytest.raises(ValueError):
        run_cases({"samples": [{"key": "dev:a"}]}, {"samples": [row]}, lambda *args: sample,
                  None, {}, None, {}, tmp_path)


def test_tensor_digest_includes_shape_dtype_and_values():
    tensor = torch.tensor([1., 2.])
    original = tensor_tree_digest({"x": tensor})
    assert original != tensor_tree_digest({"x": tensor.reshape(1, 2)})
    assert original != tensor_tree_digest({"x": tensor.double()})
    assert original != tensor_tree_digest({"x": tensor + 1})


def test_invalid_predictions_remain_in_summary_denominator():
    case = {"stratum": "broken_other", "arms": {a: dict(exact=False, valid=False, fallback=False) for a in ARMS}}
    report = summarize([case])
    assert report["all_diagnostic_cases"]["images"] == 1
    assert report["broken_other"]["images"] == 1
    assert report["all_diagnostic_cases"]["arms"]["last"]["valid"] == 0
