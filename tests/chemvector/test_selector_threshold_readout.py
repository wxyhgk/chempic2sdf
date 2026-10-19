"""Snapshot boundaries using a tiny fake forward; no real models or inputs."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools import audit_selector_threshold_readout as audit


def fake_model():
    model = torch.nn.Module()
    model.set_head = torch.nn.Module()
    model.set_head.atom_presence_residual = torch.nn.Sequential(torch.nn.Linear(128, 1))
    return model.eval().requires_grad_(False)


def fake_forward(model, tensor, config, threshold):
    hidden = torch.ones(1, 128, 128)
    logits = model.set_head.atom_presence_residual(hidden).squeeze(-1)
    atoms = dict(atom_mask=torch.ones(1, 128, dtype=torch.bool),
        atom_indices=torch.arange(128)[None], atom_geometry=torch.zeros(1, 128, 8),
        atom_scores=torch.ones(1, 128))
    output = dict(atom_pre_presence_logits=torch.zeros(1, 128), atom_presence_logits=logits)
    return None, None, None, atoms, None, output, None


def test_capture_uses_one_forward_and_removes_hook_without_mutation(monkeypatch):
    model = fake_model()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    calls = []
    def forward(*args):
        calls.append(1)
        return fake_forward(*args)
    monkeypatch.setattr(audit.continuation, 'structure_forward', forward)
    snapshot = audit.capture(model, torch.zeros(1, 3, 768, 768), SimpleNamespace())
    assert len(calls) == 1 and snapshot['hidden'].shape == (128, 128)
    assert not model.set_head.atom_presence_residual[-1]._forward_pre_hooks
    assert all(not t.requires_grad for t in snapshot.values())
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())


def test_exception_also_removes_capture_hook(monkeypatch):
    model = fake_model()
    def fail(*args): raise RuntimeError('synthetic failure')
    monkeypatch.setattr(audit.continuation, 'structure_forward', fail)
    with pytest.raises(RuntimeError, match='synthetic'):
        audit.capture(model, torch.zeros(1, 3, 768, 768), None)
    assert not model.set_head.atom_presence_residual[-1]._forward_pre_hooks


@pytest.mark.parametrize('case', ['train', 'grad', 'shape'])
def test_capture_refuses_out_of_scope_inputs(case):
    model = fake_model(); tensor = torch.zeros(1, 3, 768, 768)
    if case == 'train': model.train()
    if case == 'grad': model.requires_grad_(True)
    if case == 'shape': tensor = torch.zeros(1, 3, 16, 16)
    with pytest.raises(ValueError): audit.capture(model, tensor, None)


def test_snapshot_roundtrip_and_exclusive_publication(tmp_path):
    path = tmp_path / 'test.npz'; values = dict(hidden=torch.ones(3, 4), slots=torch.arange(3))
    assert len(audit.save_snapshot(path, values)) == 64
    with np.load(path, allow_pickle=False) as loaded:
        assert np.array_equal(loaded['hidden'], values['hidden'].numpy())
    with pytest.raises(FileExistsError): audit.save_snapshot(path, values)


def test_forbidden_forward_never_silently_runs():
    with pytest.raises(RuntimeError): audit.forbidden_forward(None, None)
