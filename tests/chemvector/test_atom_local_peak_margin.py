import torch

from chemvector.components.atoms.objects.spatial_atom_object import atom_local_peak_margin_loss


def test_local_margin_ignores_other_positive_cells():
    logits = torch.zeros(1, 1, 5, 5, requires_grad=True)
    positives = torch.zeros_like(logits, dtype=torch.bool)
    positives[0, 0, 2, 2] = True
    positives[0, 0, 2, 3] = True
    value = atom_local_peak_margin_loss(logits, positives, margin=0.5)
    assert torch.isfinite(value)
    value.backward()
    assert torch.isfinite(logits.grad).all()


def test_local_margin_decreases_when_true_peak_beats_negative():
    positives = torch.zeros(1, 1, 5, 5, dtype=torch.bool)
    positives[0, 0, 2, 2] = True
    weak = torch.zeros(1, 1, 5, 5)
    strong = weak.clone()
    strong[0, 0, 2, 2] = 3.0
    assert atom_local_peak_margin_loss(strong, positives) < atom_local_peak_margin_loss(weak, positives)
