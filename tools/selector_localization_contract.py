"""CPU-only explanation of the unchanged peak decoder on supplied tensors.

No files, models, checkpoints, data loaders, optimization, or producer entrypoint.
This is a numerical observation helper, not a replacement proposal decoder.
"""

import torch
from torch.nn import functional as F

from chemvector.components.graph.joint.predicted_incidence import _local_peak_topk


@torch.no_grad()
def inspect_decoder_cell(heatmap_logits, grid_id, *, maximum, threshold):
    """Explain one cell's actual local-peak/top-k/threshold decisions.

    All rejection flags are reported; the first excluding production stage is
    not a claim about which training mechanism caused the dense heatmap.
    Equal scores use the production topk result, never an invented tie-break.
    """
    if (not isinstance(heatmap_logits, torch.Tensor)
            or heatmap_logits.device.type != "cpu"
            or not heatmap_logits.is_floating_point()
            or heatmap_logits.ndim != 4 or heatmap_logits.shape[:2] != (1, 1)
            or min(heatmap_logits.shape[-2:]) < 1
            or not bool(torch.isfinite(heatmap_logits).all())):
        raise ValueError("Expected one finite floating CPU heatmap [1,1,H,W]")
    size = heatmap_logits.numel()
    if type(grid_id) is not int or not 0 <= grid_id < size:
        raise ValueError("Grid ID outside heatmap")
    if type(maximum) is not int or maximum < 1 or not 0 <= threshold <= 1:
        raise ValueError("Invalid unchanged decoder configuration")
    probabilities = heatmap_logits.sigmoid()
    maxima = F.max_pool2d(probabilities, 3, stride=1, padding=1)
    peaks = probabilities == maxima
    values, indices, masks = _local_peak_topk(
        heatmap_logits, maximum=maximum, threshold=threshold)
    # The production decoder pads indices with zero. A padding slot is not an
    # additional selection of the real cell zero.
    count = min(maximum, size)
    slots = (indices[0, :count] == grid_id).nonzero().flatten()
    score = probabilities.flatten()[grid_id]
    peak = bool(peaks.flatten()[grid_id])
    above = bool(score >= threshold)
    in_topk = bool(slots.numel())
    active = bool(masks[0, slots].any()) if in_topk else False
    peak_scores = probabilities[peaks]
    stage = ("local_peak_suppressed" if not peak else "topk_excluded" if not in_topk
             else "below_threshold" if not above else "raw_active")
    return {
        "grid_id": grid_id, "score": float(score),
        "local_maximum_score": float(maxima.flatten()[grid_id]),
        "is_local_peak": peak, "passes_threshold": above,
        "in_topk_before_threshold": in_topk, "raw_active": active,
        "first_excluding_stage": stage,
        "strictly_higher_peak_count": int((peak_scores > score).sum()),
        "equal_peak_count_including_self_if_peak": int((peak_scores == score).sum()),
        "ties_at_topk_boundary": peak and bool(score == values[0, count - 1]),
        "exclusion_flags": {
            "local_peak": not peak, "topk": not in_topk, "threshold": not above,
        },
        "valid_topk_slots": count, "padding_slots": maximum - count,
        "model_forward_performed": False, "training_authorized": False,
    }
