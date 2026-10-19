"""Versioned CPU wiring prototype, not a training launcher or data admission.

Build plans from the CURRENT candidate tensors. Historical saved row IDs are
for audits only; never import them into a new execution-device cache.
"""

import torch

from .pair_anchor_augmentation import OFFSETS, plan_views, typed_augmentation_loss
from .supervision_contract import pair_targets_with_unknown


class PerImageOffsetCycle:
    """Advance on every visit, including empty supervision/augmentation.

    No labels, eligibility masks or scores enter scheduling. Starting at the
    first already-fixed offset is not a search over favorable directions.
    This object is ephemeral; checkpoint/resume is intentionally unsupported.
    """

    def __init__(self, keys):
        keys = list(keys)
        if not keys or any(not isinstance(k, str) or not k for k in keys) or len(keys) != len(set(keys)):
            raise ValueError('Nonempty unique image keys required')
        self._visits = dict.fromkeys(keys, 0)

    def next(self, key):
        if key not in self._visits:
            raise ValueError('Image outside the fixed schedule')
        visit = self._visits[key]
        self._visits[key] += 1
        index = 1 + visit % (len(OFFSETS) - 1)
        return dict(visit=visit, view_index=index, offset_pixels=list(OFFSETS[index]))


def sampled_positions(chosen, rows):
    """Map all-pair row IDs to sampled positions, preserving rows' order."""
    if (any(t.ndim != 1 or t.dtype != torch.long for t in (chosen, rows))
            or chosen.device != rows.device
            or any(bool((t < 0).any()) or len(t.unique()) != len(t) for t in (chosen, rows))):
        raise ValueError('Unique nonnegative row IDs on one device required')
    if not len(rows):
        return rows.clone()
    if not len(chosen):
        raise ValueError('Augmented rows must be a subset of sampled rows')
    sorted_ids, positions = chosen.sort()
    insertion = torch.searchsorted(sorted_ids, rows)
    if bool((insertion >= len(chosen)).any()) or not torch.equal(sorted_ids[insertion], rows):
        raise ValueError('Augmented rows must be a subset of sampled rows')
    return positions[insertion]


def paired_objective(pair, feature_maps, atoms, mapping, targets, chosen, *, view_index, weight=1.):
    """Single-image CPU, original plus at most one shifted pair forward.

    Synthetic tests exercise the production pair-call signature. No caller
    may use a saved plan here: matching/labels are rebuilt from the supplied
    current lineage and the plan is derived in this call. No optimizer exists.
    """
    if type(view_index) is not int or not 1 <= view_index < len(OFFSETS):
        raise ValueError('One of the eight nonzero views required')
    if type(weight) not in (int, float) or weight not in (0., 1.):
        raise ValueError('Fixed unit weight or zero-weight parity test required')
    tensors = [feature_maps, mapping, chosen, *atoms.values(), *targets.values()]
    if (any(t.device.type != 'cpu' for t in tensors) or atoms['atom_mask'].shape[0] != 1
            or feature_maps.shape[0] != 1 or any(m.training for m in pair.modules())
            or any(p.device.type != 'cpu' for p in pair.parameters())):
        raise ValueError('CPU single image and deterministic eval-mode pair module required')
    contract = pair_targets_with_unknown(atoms, mapping, targets)
    pairs, samples = contract['pair_indices'], contract['sample_indices']
    anchors = atoms['atom_geometry'][..., :2].detach()
    view = plan_views(anchors[0], atoms['atom_mask'][0], mapping[0], targets['atom_anchors'][0],
                      pairs, contract['known_mask'], chosen)[view_index]
    positions = sampled_positions(chosen, view['rows'])
    details = dict(base_row_ids=chosen.clone(), planned_augmented_row_ids=view['rows'].clone(),
                   planned_sampled_positions=positions.clone(), offset_pixels=view['offset_pixels'],
                   pair_forwards=0, skipped_empty_base=not len(chosen))
    if not len(chosen):
        return dict(losses=None, **details)

    def forward(xy, rows):
        details['pair_forwards'] += 1
        return pair(feature_maps.detach(), atoms['atom_features'].detach(), xy.detach(),
                    atoms['atom_scores'].detach(), pairs[rows], samples[rows])

    base_output = forward(anchors, chosen)
    # Zero weight skips the extra forward entirely: protects RNG/state parity,
    # not merely the scalar loss value. Do not call the head with empty rows.
    apply_extra = weight == 1 and bool(len(positions))
    augmented_output = forward(view['anchors'][None], view['rows']) if apply_extra else None
    applied_positions = positions if apply_extra else positions[:0]
    labels = contract['legacy_targets']
    losses = typed_augmentation_loss(base_output, augmented_output, labels['presence'][chosen],
        labels['chemical'][chosen], applied_positions, weight=weight)
    if not bool(torch.isfinite(losses['loss'])):
        raise ValueError('Nonfinite wired loss')
    return dict(losses=losses, **details)
