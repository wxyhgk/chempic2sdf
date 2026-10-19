"""Explicit training-only pair supervision; unchanged prediction and matching.

No diagnostic correspondence, graph repair, new positive labels, or optimizer.
Legacy is the default and retains the original sampler call without filtering.
"""

import torch

from chemvector.train_direct_atom_pair_bond import select_training_pairs

LEGACY = 'legacy-all-pairs-v1'
MATCHED_ONLY = 'matched-only-v1'
POLICIES = (LEGACY, MATCHED_ONLY)


def validate_policy(policy):
    if policy not in POLICIES:
        raise ValueError('Unknown pair supervision policy')


def supervision_mask(pair_indices, sample_indices, atom_mapping, presence, *, policy=LEGACY):
    validate_policy(policy)
    count = len(pair_indices)
    if (pair_indices.dtype != torch.long or pair_indices.shape != (count, 2)
            or sample_indices.dtype != torch.long or sample_indices.shape != (count,)
            or atom_mapping.dtype != torch.long or atom_mapping.ndim != 2
            or presence.dtype != torch.bool or presence.shape != (count,)
            or any(t.device != pair_indices.device for t in (sample_indices, atom_mapping, presence))):
        raise ValueError('Aligned pair IDs, mapping and boolean presence required')
    if count and (bool((sample_indices < 0).any()) or bool((sample_indices >= len(atom_mapping)).any())
            or bool((pair_indices < 0).any()) or bool((pair_indices >= atom_mapping.shape[1]).any())
            or bool((pair_indices[:, 0] >= pair_indices[:, 1]).any())):
        raise ValueError('Pair indices outside mapping or not strictly ordered')
    if policy == LEGACY:
        return torch.ones_like(presence)
    mapped = atom_mapping[sample_indices[:, None], pair_indices]
    eligible = (mapped >= 0).all(-1) & (mapped[:, 0] != mapped[:, 1])
    if bool((presence & ~eligible).any()):
        raise ValueError('Original positive has no distinct matched endpoints')
    return eligible


def sample_supervised_pairs(pair_indices, sample_indices, atom_mapping, atoms, pair_targets,
                            *, policy=LEGACY, negative_ratio=6, minimum_negatives=48):
    """Return original row IDs and mask; remove uncertainty before negative quota."""
    mask = supervision_mask(pair_indices, sample_indices, atom_mapping, pair_targets['presence'], policy=policy)
    if policy == LEGACY:
        chosen = select_training_pairs(pair_indices, sample_indices, atoms, pair_targets,
                                       negative_ratio=negative_ratio, minimum_negatives=minimum_negatives)
        return chosen, mask
    eligible = mask.nonzero().flatten()
    filtered = dict(pair_targets)
    for field in ('presence', 'chemical', 'structured_negative'):
        filtered[field] = pair_targets[field][eligible]
    local = select_training_pairs(pair_indices[eligible], sample_indices[eligible], atoms, filtered,
                                  negative_ratio=negative_ratio, minimum_negatives=minimum_negatives)
    return eligible[local], mask
