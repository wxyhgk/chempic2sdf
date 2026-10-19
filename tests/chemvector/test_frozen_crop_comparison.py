from copy import deepcopy

import numpy as np
from PIL import Image
import pytest
import torch

from chemvector.evaluation.audits.ocr.frozen_crop_comparison import image_tensor, infer_frozen, score_pair, summarize_text
from chemvector.evaluate_direct_atom_ocr import crop_tensor
from chemvector.components.ocr.crop_proposals import propose_vertical, rasterize


class FakeOCR(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.calls = 0
        self.eval()

    def forward(self, pixels):
        assert not torch.is_grad_enabled()
        self.calls += 1
        logits = torch.full((len(pixels), 48, 257), -10.)
        logits[:, :, 0] = 10.
        logits[:, 0, 0] = -10.
        logits[:, 0, ord('N') + 1] = 10.
        return logits


def test_tensor_exact_runtime_preprocessing():
    image = Image.fromarray(np.random.default_rng(22).integers(0, 256, (768, 768, 3), dtype=np.uint8))
    box = [130.2, 660.5, 152.9, 698.7]
    canvas = rasterize(image, propose_vertical(box, image.size)['original'])
    assert torch.equal(image_tensor(canvas), crop_tensor(image, box))


def test_bounded_frozen_inference_and_tail_batch():
    model = FakeOCR(); pixels = torch.zeros(9, 3, 48, 192); before = pixels.clone()
    decoded, logits, calls = infer_frozen(model, pixels)
    assert calls == model.calls == 2 and logits.shape == (9, 48, 257)
    assert [r['text'] for r in decoded] == ['N'] * 9
    assert torch.equal(before, pixels) and model.anchor.grad is None


@pytest.mark.parametrize('count', [0, 293])
def test_input_budget_checked_before_forward(count):
    model = FakeOCR()
    with pytest.raises(ValueError):
        infer_frozen(model, torch.zeros(count, 3, 48, 192))
    assert model.calls == 0


@pytest.mark.parametrize('mode', ['train', 'autograd', 'wrong_shape', 'nan', 'wrong_dtype'])
def test_bad_runtime_contract(mode):
    model = FakeOCR(); pixels = torch.zeros(1, 3, 48, 192)
    if mode == 'train': model.train()
    if mode == 'autograd': model.anchor.requires_grad_(True)
    if mode == 'wrong_shape': pixels = pixels[:, :, :40]
    if mode == 'nan': pixels.fill_(float('nan'))
    if mode == 'wrong_dtype': pixels = pixels.double()
    with pytest.raises(ValueError): infer_frozen(model, pixels)
    assert model.calls == 0


def record(candidate, expected='NH', original='N', proposed='NH', status='matched_visible'):
    return dict(key='train:sample', candidate=candidate, expected_text=expected, status=status,
        original=dict(text=original), proposed=dict(text=proposed), saved_text=original,
        layout='vertical_multiglyph', coverage_rescued=True, crop_changed=True)


def test_unmatched_and_missing_targets_not_scored_as_correct():
    records = [record(0), record(1, status='unmatched')]; before = deepcopy(records)
    summary = summarize_text(records, visible_targets=3)
    assert summary['visible_targets_without_unique_crop'] == 2
    assert summary['groups']['all_matched_visible']['raw']['proposed_correct'] == 1
    assert summary['changed_text_all_statuses'] == 2 and records == before


def test_regression_not_hidden_by_net_gain_or_confidence():
    rows = [record(0), record(1), record(2, expected='P', original='P', proposed='O')]
    summary = summarize_text(rows, visible_targets=3)
    assert summary['decision'] == 'regression_keep_default'
    assert len(summary['groups']['all_matched_visible']['raw']['regressed']) == 1


def test_historical_device_drift_reported_separately():
    row = record(0, original='NH', proposed='NH'); row['saved_text'] = 'N'
    summary = summarize_text([row], visible_targets=1)
    assert len(summary['cpu_original_vs_saved_text_mismatches']) == 1
    assert summary['decision'] == 'no_text_gain_keep_default'


def test_invalid_parses_not_equal_by_none_and_duplicate_rejected():
    assert not score_pair('not_a_label', '??', '??')['proposed_state_exact']
    with pytest.raises(ValueError): summarize_text([record(0), record(0)], visible_targets=2)
    with pytest.raises(ValueError): summarize_text([record(0)], visible_targets=0)
