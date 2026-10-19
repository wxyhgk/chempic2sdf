"""Frozen-feature heatmap parameter derivatives; no optimizer or weight edits."""

import math

import torch

from chemvector.vertex import build_vertex_heatmap, focal_heatmap_loss
from tools.selector_dense_observation import image_only_snapshot


def image_only_features(model, pixels, config):
    """Capture hidden features during exactly the existing image-only forward."""
    captured = []

    def capture(module, arguments, output):
        captured.append(output['object_features'].detach().clone())

    handle = model.atom.register_forward_hook(capture)
    try:
        snapshot = image_only_snapshot(model, pixels, config)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise ValueError('Exactly one atom forward required')
    return snapshot, captured[0]


def check_tensor(value, shape=None):
    if (not isinstance(value, torch.Tensor) or value.device.type != 'cpu'
            or value.dtype != torch.float32 or value.requires_grad
            or not torch.isfinite(value).all()
            or shape is not None and tuple(value.shape) != tuple(shape)):
        raise ValueError('Finite detached CPU float32 tensors with matching shapes required')


def parameter_gradient(hidden, logits, anchors):
    """Exact chain rule for a 1x1 readout of frozen features, all atoms included.

    Return weight and bias derivatives of the existing, per-image normalized
    focal loss. No extra model/readout forward or parameter .grad is needed.
    """
    check_tensor(hidden)
    if hidden.ndim != 4 or hidden.shape[0] != 1 or min(hidden.shape[1:]) < 1:
        raise ValueError('One nonempty feature image required')
    _, channels, height, width = hidden.shape
    check_tensor(logits, (1, 1, height, width))
    check_tensor(anchors)
    if (anchors.ndim != 3 or anchors.shape[0] != 1 or anchors.shape[2] != 2
            or anchors.shape[1] < 1 or (anchors < 0).any() or (anchors > 1).any()):
        raise ValueError('Nonempty in-range atom anchors required')
    heat = build_vertex_heatmap(anchors, torch.ones(anchors.shape[:2], dtype=torch.bool), height, width, 1)
    if int((heat == 1).sum()) != anchors.shape[1]:
        raise ValueError('Multiple atoms in one target cell are not supported')
    z = logits.detach().clone().requires_grad_(True)
    loss = focal_heatmap_loss(z, heat)
    dz, = torch.autograd.grad(loss, z)
    # Float64 accumulation improves the directional diagnostic, not production inference.
    features = hidden[0].flatten(1).double()
    derivatives = dz.flatten().double()
    gradient = torch.cat((features @ derivatives, derivatives.sum().reshape(1)))
    if gradient.shape != (channels + 1,) or not torch.isfinite(gradient).all():
        raise ValueError('Invalid readout parameter gradient')
    return dict(gradient=gradient, heatmap=heat, logit_gradient=dz.detach(),
                loss=float(loss.detach()), positive_cells=anchors.shape[1])


def descent_field(hidden, gradient):
    check_tensor(hidden)
    if (hidden.ndim != 4 or hidden.shape[0] != 1 or gradient.device.type != 'cpu'
            or gradient.dtype != torch.float64 or gradient.requires_grad
            or gradient.shape != (hidden.shape[1] + 1,) or not torch.isfinite(gradient).all()):
        raise ValueError('Matching detached float64 readout derivative required')
    return -(gradient[:-1] @ hidden[0].flatten(1).double() + gradient[-1]).reshape(hidden.shape[-2:])


def local_margin(logits, direction, x, y):
    """Right derivative of z[cell] minus the largest OTHER 3x3 logit.

    If maximum neighbors tie, the largest directional derivative wins for a
    positive infinitesimal step. Never choose an arbitrary favorable neighbor.
    """
    h, w = logits.shape
    if (direction.shape != logits.shape or not 0 <= x < w or not 0 <= y < h
            or not torch.isfinite(logits).all() or not torch.isfinite(direction).all()):
        raise ValueError('Finite, matching local grids and in-range coordinates required')
    ids = [(xx, yy) for yy in range(max(0, y-1), min(h, y+2))
           for xx in range(max(0, x-1), min(w, x+2)) if (xx, yy) != (x, y)]
    if not ids:
        raise ValueError('A distinct neighboring cell is required')
    maximum = max(float(logits[yy, xx]) for xx, yy in ids)
    tied = [(xx, yy) for xx, yy in ids if float(logits[yy, xx]) == maximum]
    return dict(grid_id=y*w+x, neighbor_grid_ids=[yy*w+xx for xx, yy in tied],
                margin=float(logits[y, x])-maximum,
                descent_margin_derivative=float(direction[y, x])-max(float(direction[yy, xx]) for xx, yy in tied),
                descent_logit_derivative=float(direction[y, x]))


def describe_image(hidden, logits, heatmap, geometry, target_ids, gradient, own_gradient):
    """Describe all source atoms and actually matched original raw peaks."""
    h, w = logits.shape[-2:]
    z = logits[0, 0]
    field, own = (descent_field(hidden, g) for g in (gradient, own_gradient))
    targets = torch.tensor(geometry['target_anchors'], dtype=torch.float32)
    cells = (targets * torch.tensor([w, h])).clamp_min(0)
    cells[:, 0].clamp_max_(w - 1e-4)
    cells[:, 1].clamp_max_(h - 1e-4)
    cells = cells.long()
    if (len(set(target_ids)) != len(target_ids)
            or any(type(t) is not int or not 0 <= t < len(cells) for t in target_ids)):
        raise ValueError('Unique in-range focus IDs required')
    records = []
    for target_id, (x, y) in enumerate(cells.tolist()):
        record = local_margin(z, field, x, y)
        own_record = local_margin(z, own, x, y)
        record.update(target_id=target_id, focused=target_id in target_ids,
            gt_heatmap_target=float(heatmap[0, 0, y, x]),
            own_image_descent_margin_derivative=own_record['descent_margin_derivative'],
            neighbor_positive_target_ids=[i for i, (cx, cy) in enumerate(cells.tolist())
                if cy*w+cx in record['neighbor_grid_ids']],
            derivative_scope='mean of all seven image losses; first-order only')
        records.append(record)
    matched = []
    for candidate in geometry['candidates']:
        if candidate['raw_target'] < 0:
            continue
        y, x = divmod(candidate['grid_id'], w)
        record = local_margin(z, field, x, y)
        record.update(slot=candidate['slot'], target_id=candidate['raw_target'],
            logit_threshold_margin=float(z[y, x])-math.log(.01/.99))
        matched.append(record)
    return dict(atoms=records, original_raw_positive_peaks=matched,
        loss_descent_derivative=-float(torch.dot(own_gradient, gradient)),
        parameters_changed=False, candidate_recovery_measured=False,
        caution='Negative peak-margin slope is risk, not an observed lost candidate; top-k and threshold also matter')


def diagnostic_gate(cases):
    """Predeclared conservative gate, not a mathematical impossibility claim.

    Require a strict majority (>=6/10) of the fixed missing targets to improve
    their local margin and the summed missing-target margin to improve.
    Failure is an obvious conflict for THIS full-image mean-loss direction;
    success only permits trial preparation, never deployment or automatic fit.
    """
    focused = [a for case in cases for a in case['atoms'] if a['focused']]
    if len(cases) != 7 or len(focused) != 10:
        raise ValueError('Fixed seven images and ten missing targets required')
    slopes = [a['descent_margin_derivative'] for a in focused]
    if any(not math.isfinite(v) for v in slopes):
        raise ValueError('Finite directional evidence required')
    improves = sum(v > 1e-8 for v in slopes)
    total = sum(slopes)
    reasons = []
    if improves < 6:
        reasons.append('no_strict_majority_of_missing_target_local_margins_improves')
    if total <= 1e-8:
        reasons.append('summed_missing_target_local_margin_does_not_improve')
    if any(a['neighbor_positive_target_ids'] for a in focused):
        reasons.append('focused_competitor_is_another_true_atom')
    return dict(status='NO_GO' if reasons else 'DIRECTION_SUPPORTED', reasons=reasons,
        focused_targets=10, improving_local_margins=improves,
        worsening_local_margins=sum(v < -1e-8 for v in slopes),
        flat_local_margins=sum(abs(v) <= 1e-8 for v in slopes),
        summed_local_margin_derivative=total, numerical_zero_tolerance=1e-8,
        minimum_improving_targets=6, model_updated=False,
        supports_bounded_trial_preparation=not reasons,
        limitation='Not all possible 129-parameter directions, optimizer steps, full58 gradient, or architecture capacity')
