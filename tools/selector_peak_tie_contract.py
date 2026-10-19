"""CPU reference decoder with an explicit exact-score tie order.

Diagnostic only. Production is unchanged. Sigmoid, 3x3 equality-to-max peaks,
capacity, threshold comparison and padding mirror the existing decoder.
Only the order inside EXACT equal score groups changes: flattened ID ascending.
For bonds the flattened ID includes the prediction channel, not just (x, y).
"""

import math

import torch
from torch.nn import functional as F

POLICY = 'exact-probability-desc-flat-id-asc-v1'


@torch.no_grad()
def stable_local_peak_topk(logits, *, maximum, threshold):
    if (not isinstance(logits, torch.Tensor) or logits.device.type != 'cpu'
            or logits.dtype not in (torch.float32, torch.float64)
            or logits.ndim != 4 or min(logits.shape) < 1
            or not torch.isfinite(logits).all() or type(maximum) is not int or maximum < 1
            or isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError('Finite CPU [B,C,H,W] logits, positive capacity and finite threshold required')
    probabilities = logits.sigmoid()
    peaks = probabilities == F.max_pool2d(probabilities, 3, stride=1, padding=1)
    scores = probabilities.masked_fill(~peaks, -1.).flatten(1)
    count = min(maximum, scores.shape[1])
    # Flattening already supplies ascending identity. Stable sort keeps it for
    # exact equal probability scores; no epsilon, quantization, or near-tie band.
    indices = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :count]
    values = scores.gather(1, indices)
    mask = values >= threshold
    if count < maximum:
        values = F.pad(values, (0, maximum-count), value=-1.)
        indices = F.pad(indices, (0, maximum-count), value=0)
        mask = F.pad(mask, (0, maximum-count), value=False)
    return values, indices, mask


@torch.no_grad()
def canonical_atom_proposals(output, config):
    """Isolated diagnostic adapter; same gathers as the production atom decoder."""
    scores, indices, mask = stable_local_peak_topk(output['heatmap_logits'],
        maximum=config.maximum_atoms, threshold=config.atom_threshold)
    if output['heatmap_logits'].shape[1] != 1:
        raise ValueError('Atom identity requires one heatmap channel')
    h, w = output['heatmap_logits'].shape[-2:]
    spatial = indices.remainder(h*w)
    y, x = spatial // w, spatial % w
    batch = torch.arange(len(indices))[:, None]
    def gather(name):
        return output[name].permute(0, 2, 3, 1)[batch, y, x]
    geometry = torch.cat((gather('anchors'), gather('role_logits').softmax(-1),
                          gather('text_centers'), gather('text_extents')), dim=-1)
    return dict(atom_scores=scores, atom_indices=spatial, atom_mask=mask,
                atom_features=gather('object_features'), atom_geometry=geometry)


@torch.no_grad()
def canonical_bond_proposals(output, config):
    """Keep channel-major flat IDs; two bond slots at one pixel are distinct."""
    scores, indices, mask = stable_local_peak_topk(output['heatmap_logits'],
        maximum=config.maximum_bonds, threshold=config.bond_threshold)
    h, w = output['heatmap_logits'].shape[-2:]
    channel, spatial = indices // (h*w), indices % (h*w)
    y, x = spatial // w, spatial % w
    batch = torch.arange(len(indices))[:, None]
    features = output['object_features'].permute(0, 2, 3, 1)[batch, y, x]
    endpoints = output['endpoints'].permute(0, 1, 4, 5, 2, 3)[batch, channel, y, x]
    display = output['display_logits'].permute(0, 1, 3, 4, 2)[batch, channel, y, x].softmax(-1)
    chemical = output['chemical_logits'].permute(0, 1, 3, 4, 2)[batch, channel, y, x].softmax(-1)
    return dict(bond_scores=scores, bond_indices=indices, bond_mask=mask, bond_features=features,
                bond_endpoints=endpoints, bond_types=torch.cat((display, chemical), dim=-1))


@torch.no_grad()
def selection_difference(logits, legacy, canonical, *, maximum, threshold):
    """Explain a decoder-only difference on the SAME complete dense tensor.

    No inference scores, labels or features are reused after a changed selection.
    A retained-only list is deliberately not an accepted input here.
    """
    expected = stable_local_peak_topk(logits, maximum=maximum, threshold=threshold)
    if len(legacy) != 3 or len(canonical) != 3 or any(not torch.equal(a, b) for a, b in zip(canonical, expected)):
        raise ValueError('Canonical selection must come from the supplied complete heatmap')
    values, indices, mask = canonical
    old_values, old_indices, old_mask = legacy
    if (old_values.shape != values.shape or old_indices.shape != indices.shape or old_mask.shape != mask.shape
            or old_indices.dtype != torch.int64 or old_mask.dtype != torch.bool
            or not torch.equal(values, old_values) or not torch.equal(mask, old_mask)):
        raise ValueError('Tie policy must preserve the full sorted score and activity vectors')
    probabilities = logits.sigmoid()
    peaks = probabilities == F.max_pool2d(probabilities, 3, stride=1, padding=1)
    flat = probabilities.masked_fill(~peaks, -1.).flatten(1)
    count = min(maximum, flat.shape[1])
    if ((old_indices < 0).any() or (old_indices >= flat.shape[1]).any()
            or not torch.equal(flat.gather(1, old_indices[:, :count]), old_values[:, :count])
            or not torch.equal(old_mask, old_values >= threshold)
            or any(len(set(row[:count].tolist())) != count for row in old_indices)):
        raise ValueError('Legacy selection is not aligned with the same full score field')
    if count < maximum and (old_indices[:, count:].count_nonzero() or old_mask[:, count:].any()):
        raise ValueError('Padding is not a real repeated candidate')
    records = []
    for batch in range(len(logits)):
        old = old_indices[batch, old_mask[batch]].tolist()
        new = indices[batch, mask[batch]].tolist()
        a, b = set(old), set(new)
        removed, added = sorted(a-b), sorted(b-a)
        cutoff = float(values[batch, count-1])
        if any(float(flat[batch, i]) != cutoff for i in removed+added):
            raise ValueError('A non-tied candidate changed membership')
        old_slots, new_slots = ({idx: slot for slot, idx in enumerate(ids)} for ids in (old, new))
        reordered = [dict(flat_id=i, old_slot=old_slots[i], new_slot=new_slots[i])
                     for i in sorted(a & b) if old_slots[i] != new_slots[i]]
        records.append(dict(batch=batch, active_count=len(new), old_active_flat_ids=old,
            canonical_active_flat_ids=new, removed_flat_ids=removed, added_flat_ids=added,
            reordered_common=reordered, cutoff_score=cutoff,
            complete_field_cutoff_ties=int((flat[batch] == cutoff).sum()),
            active_set_unchanged=a == b, active_order_unchanged=old == new,
            can_reuse_slot_indexed_context=old == new,
            changed_members_only_at_exact_cutoff_tie=True))
    return dict(policy=POLICY, batches=records, sorted_score_vectors_equal=True,
        activity_vectors_equal=True, production_decoder_changed=False,
        cross_device_dense_numeric_parity_proven=False, training_authorized=False)
