"""Portable complete structure checkpoint round trips on tiny CPU modules."""

import torch
import pytest

from tests.chemvector.test_curriculum_continuation import tiny
from chemvector.model import ChemVectorConfig
from chemvector.training.curriculum.bundle import load_bundle, save_bundle
from chemvector.training.curriculum.checkpoints import file_digest
from chemvector.training.curriculum.continuation import continuation_losses, structure_forward
from chemvector.training.curriculum.gradient_preflight import state_digest


def configure(model):
    model.vision_config = ChemVectorConfig(d_model=32, decoder_heads=4, vision_dims=(8, 16, 32, 64),
                                          vision_depths=(1, 1, 1, 1), dropout=0)


def test_updated_all_component_weights_roundtrip_without_legacy_sources(tiny, tmp_path, monkeypatch):
    model, config, batch = tiny
    configure(model)
    optimizer = torch.optim.AdamW(model.parameter_groups())
    losses, _ = continuation_losses(model, batch, config, atom_threshold=-1e6)
    losses["loss"].backward()
    optimizer.step()
    model.eval()
    with torch.no_grad():
        expected = structure_forward(model, batch["pixel_values"], config, 0)
    record = save_bundle(tmp_path / "bundle.pt", model, config, provenance={"synthetic": True}, progress={"optimizer_steps": 1})
    def forbidden(*args, **kwargs):
        raise AssertionError("Portable bundle must not load legacy upstream checkpoints")
    monkeypatch.setattr("chemvector.training.curriculum.checkpoints.load_current_components", forbidden)
    restored, restored_config, payload = load_bundle(record["path"], expected_sha256=record["sha256"])
    assert state_digest(restored) == state_digest(model)
    assert restored_config == config and payload["progress"]["optimizer_steps"] == 1
    assert not payload["resume_supported"]
    with torch.no_grad():
        actual = structure_forward(restored, batch["pixel_values"], restored_config, 0)
    for left, right in zip(expected, actual):
        if isinstance(left, dict):
            assert all(torch.equal(left[k], right[k]) for k in left)
        else:
            assert torch.equal(left, right)
    with pytest.raises(FileExistsError):
        save_bundle(record["path"], model, config, provenance={}, progress={})


def test_corrupt_or_incomplete_bundle_never_falls_back_to_old_components(tiny, tmp_path):
    model, config, _ = tiny
    configure(model)
    path = tmp_path / "bundle.pt"
    record = save_bundle(path, model, config, provenance={}, progress={})
    with pytest.raises(ValueError, match="hash"):
        load_bundle(path, expected_sha256="0" * 64)
    payload = torch.load(path, weights_only=True)
    payload["model"].pop(next(k for k in payload["model"] if k.startswith("vision.")))
    incomplete = tmp_path / "incomplete.pt"
    torch.save(payload, incomplete)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_bundle(incomplete, expected_sha256=file_digest(incomplete))


def test_failed_bundle_write_leaves_no_published_or_partial_checkpoint(tiny, tmp_path, monkeypatch):
    model, config, _ = tiny
    configure(model)
    def fail(*args, **kwargs):
        raise OSError("synthetic disk failure")
    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="disk failure"):
        save_bundle(tmp_path / "failed.pt", model, config, provenance={}, progress={})
    assert list(tmp_path.iterdir()) == []
