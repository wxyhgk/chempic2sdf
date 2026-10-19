"""Bounded frozen CPU OCR and separate truth-side paired metrics.

Inference accepts only tensors. Scoring never selects an OCR output or repairs
a graph. Unmatched candidates and missing targets remain explicit denominators.
"""

from collections import Counter
import math

import numpy as np
import torch

from chemvector.ctc_ocr import decode_ctc_with_score
from chemvector.evaluate_direct_atom_ocr import atom_state


def image_tensor(image):
    if image.mode != 'RGB' or image.size != (192, 48):
        raise ValueError('Original OCR RGB canvas required')
    array = np.asarray(image, dtype=np.float32) / 255.0
    return (torch.from_numpy(array).permute(2, 0, 1) - .5) / .5


def infer_frozen(model, pixels, *, batch_size=8, maximum_inputs=292, progress=None):
    """No labels, candidate IDs, image names, routing or target text accepted."""
    if (type(batch_size) is not int or not 1 <= batch_size <= 8
            or type(maximum_inputs) is not int or not 1 <= maximum_inputs <= 292
            or pixels.ndim != 4 or pixels.shape[1:] != (3, 48, 192)
            or not 0 < len(pixels) <= maximum_inputs or pixels.device.type != 'cpu'
            or pixels.dtype != torch.float32 or pixels.requires_grad
            or not bool(torch.isfinite(pixels).all()) or bool((pixels.abs() > 1).any())):
        raise ValueError('Bounded detached CPU FP32 OCR inputs required')
    if (any(m.training for m in model.modules())
            or any(p.requires_grad or p.grad is not None for p in model.parameters())
            or any(t.device.type != 'cpu' for t in (*model.parameters(), *model.buffers()))):
        raise ValueError('Frozen eval-only CPU model required')
    rows, outputs, calls = [], [], 0
    with torch.inference_mode():
        for start in range(0, len(pixels), batch_size):
            batch = pixels[start:start + batch_size]
            logits = model(batch)
            calls += 1
            if (logits.shape != (len(batch), 48, 257) or logits.device.type != 'cpu'
                    or logits.dtype != torch.float32 or not bool(torch.isfinite(logits).all())):
                raise ValueError('Finite original 48-step FP32 CTC logits required')
            for row in logits:
                text, score = decode_ctc_with_score(row)
                if not math.isfinite(score):
                    raise ValueError('Nonfinite CTC confidence')
                rows.append(dict(text=text, score=score))
            outputs.append(logits.detach().clone())
            if progress is not None:
                progress(calls, min(start + batch_size, len(pixels)))
    return rows, torch.cat(outputs), calls


def score_pair(expected, original, proposed):
    target, old, new = (atom_state(t) for t in (expected, original, proposed))
    return dict(original_raw_exact=original == expected, proposed_raw_exact=proposed == expected,
        expected_parsed_state=target, original_parsed_state=old, proposed_parsed_state=new,
        original_state_exact=target is not None and target == old,
        proposed_state_exact=target is not None and target == new)


def summarize_text(records, *, visible_targets):
    if len({(r['key'], r['candidate']) for r in records}) != len(records):
        raise ValueError('Duplicate crop identity')
    visible = [r for r in records if r['status'] == 'matched_visible']
    if type(visible_targets) is not int or visible_targets < len(visible):
        raise ValueError('Complete visible target denominator required')
    groups = {}
    for name in ('all_matched_visible', 'vertical_multiglyph', 'coverage_rescued', 'changed_no_coverage_gain'):
        subset = visible if name == 'all_matched_visible' else [r for r in visible if
            (r['layout'] == name if name == 'vertical_multiglyph' else r['coverage_rescued']
             if name == 'coverage_rescued' else r['crop_changed'] and not r['coverage_rescued'])]
        scored = [(r, score_pair(r['expected_text'], r['original']['text'], r['proposed']['text'])) for r in subset]
        groups[name] = dict(crops=len(subset), changed_text=sum(r['original']['text'] != r['proposed']['text'] for r in subset),
            **{metric: dict(original_correct=sum(s['original_' + metric + '_exact'] for _, s in scored),
                proposed_correct=sum(s['proposed_' + metric + '_exact'] for _, s in scored),
                rescued=[dict(key=r['key'], candidate=r['candidate']) for r, s in scored
                         if not s['original_' + metric + '_exact'] and s['proposed_' + metric + '_exact']],
                regressed=[dict(key=r['key'], candidate=r['candidate']) for r, s in scored
                           if s['original_' + metric + '_exact'] and not s['proposed_' + metric + '_exact']])
               for metric in ('raw', 'state')})
    all_group = groups['all_matched_visible']
    regressions = any(all_group[k]['regressed'] for k in ('raw', 'state'))
    gains = any(all_group[k]['rescued'] for k in ('raw', 'state'))
    return dict(predicted_crops=len(records), statuses=dict(Counter(r['status'] for r in records)),
        visible_targets=visible_targets, visible_targets_without_unique_crop=visible_targets - len(visible),
        changed_text_all_statuses=sum(r['original']['text'] != r['proposed']['text'] for r in records),
        cpu_original_vs_saved_text_mismatches=[dict(key=r['key'], candidate=r['candidate'], saved=r['saved_text'],
            cpu=r['original']['text']) for r in records if r['saved_text'] != r['original']['text']],
        groups=groups, decision='regression_keep_default' if regressions else
        'text_gain_only_no_deployment' if gains else 'no_text_gain_keep_default',
        independent_accuracy_measured=False, graph_accuracy_measured=False, deployment_authorized=False)
