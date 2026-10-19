"""Synthetic original-image/SDF evaluation without any training annotations."""

import numpy as np
from PIL import Image
import pytest
from rdkit import Chem
import torch

from chemvector.training.curriculum.evaluation import evaluate, expected_identity, summarize_predictions
from chemvector.training.curriculum.pixels import image_tensor


def molblock(smiles):
    return Chem.MolToMolBlock(Chem.MolFromSmiles(smiles))


def test_normalization_is_bitwise_equal_to_old_training_formula():
    pixels = np.random.default_rng(13).integers(0, 256, (768, 768, 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    before = image.tobytes()
    old = (torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255 - .5) / .5
    assert torch.equal(old, image_tensor(image))
    assert image.tobytes() == before


@pytest.mark.parametrize("mode,size", [("L", (768, 768)), ("RGBA", (768, 768)), ("RGB", (256, 256))])
def test_no_implicit_image_conversion_or_resize(mode, size):
    with pytest.raises(ValueError, match="shape/mode"):
        image_tensor(Image.new(mode, size))


@pytest.mark.parametrize("smiles", ["CCO", "[13CH3]CO", "F[C@H](Cl)Br", "C[NH3+]", "[Na+].[Cl-]"])
def test_sdf_and_legacy_graph_truth_agree_without_losing_attributes(smiles):
    canonical = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True)
    assert expected_identity({"target_molblock": molblock(smiles) + "$$$$\n"}) == canonical
    assert expected_identity({"graph": {"canonical_smiles": canonical}}) == canonical


@pytest.mark.parametrize("sample", [{}, {"target_molblock": ""}, {"target_molblock": "invalid"},
    {"target_molblock": molblock("C"), "graph": {"canonical_smiles": "C"}},
    {"target_molblock": molblock("C") + "$$$$\n" + molblock("CC") + "$$$$\n"}])
def test_ambiguous_missing_or_multirecord_truth_is_rejected(sample):
    with pytest.raises(ValueError):
        expected_identity(sample)


def test_original_image_sdf_evaluation_needs_no_training_labels_and_retains_failures(monkeypatch):
    image = Image.new("RGB", (768, 768), "white")
    samples = {
        "a": {"image": image, "target_molblock": molblock("CCO"), "purpose": "diagnostic"},
        "b": {"image": image, "target_molblock": molblock("CCN"), "purpose": "diagnostic"},
    }
    rows = [dict(source="original", sample_key=k, domain="full_molecule", split="validation") for k in samples]
    rows.append(dict(source="train", sample_key="must_not_load", split="train"))
    manifest = {"samples": rows}
    model = torch.nn.Linear(1, 1).train()
    calls = []
    def no_training_targets(*args, **kwargs):
        raise AssertionError("Evaluation must not build training targets")
    monkeypatch.setattr("chemvector.training.curriculum.data.CurriculumCollator.__call__", no_training_targets)
    def predictor(model, config, pixels, original_image):
        assert not model.training and not torch.is_grad_enabled()
        assert pixels.shape == (1, 3, 768, 768) and torch.all(pixels == 1)
        assert original_image is image
        calls.append(True)
        return {"molblock": molblock("CCO") if len(calls) == 1 else None, "fallback": False, "error": None}
    result = evaluate(model, None, manifest, lambda m, row: samples[row["sample_key"]], predictor, device="cpu")
    assert model.training and len(calls) == 2
    assert summarize_predictions(result)["sdf_exact_rate"] == .5
    assert summarize_predictions(result)["images"] == 2
    assert not result[1]["valid"]


def test_wrong_sample_purpose_cannot_reach_predictor_and_model_mode_restored():
    model = torch.nn.Linear(1, 1).train()
    manifest = {"samples": [{"split": "validation"}]}
    def forbidden(*args):
        raise AssertionError("Bad-purpose sample must fail before inference")
    with pytest.raises(ValueError, match="diagnostic-only"):
        evaluate(model, None, manifest, lambda *args: {"purpose": "training"}, forbidden, device="cpu")
    assert model.training
