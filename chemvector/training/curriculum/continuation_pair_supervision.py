"""Versioned opt-in loss entry; the SHA-bound historical continuation is intact.

Default calls the historical loss verbatim. The matched-only branch preserves
its component losses and text path, changing only pair sampling and telemetry.
Existing launchers are NOT switched to this entry and no optimizer API is added.
"""

import torch

from . import continuation as legacy
from .pair_supervision import LEGACY, MATCHED_ONLY, sample_supervised_pairs, validate_policy


def continuation_losses(model, batch, config, *, atom_threshold=0.0, bond_threshold=-0.5,
                        pair_supervision=LEGACY):
    validate_policy(pair_supervision)  # Reject before a forward or side effect.
    if pair_supervision == LEGACY:
        return legacy.continuation_losses(model, batch, config,
            atom_threshold=atom_threshold, bond_threshold=bond_threshold)
    return _matched_only_losses(model, batch, config,
        atom_threshold=atom_threshold, bond_threshold=bond_threshold)


def _matched_only_losses(model, batch, config, *, atom_threshold, bond_threshold):
    # This bounded versioned body intentionally retains the historical loss
    # sequence. Parity tests exercise every loss, labels and text coverage;
    # archived historical implementation/source hashes remain reproducible.
    features, atom_output, bond_output, atoms, bonds, set_output, selected = legacy.structure_forward(
        model, batch['pixel_values'], config, atom_threshold)
    proposals = legacy.associate_predicted_proposals(atoms, bonds, batch, config)
    atom_losses = legacy.spatial_atom_object_loss(atom_output, batch)
    bond_targets = dict(endpoints=batch['bond_ports'], display=batch['bond_display'],
                        chemical=batch['bond_chemical'], direction=batch['bond_direction'], mask=batch['bond_mask'])
    bond_losses = legacy.dense_bond_object_loss(bond_output, bond_targets, heatmap_set_weight=.1,
        slot_matching='hungarian', cell_count_weight=.05, duplicate_weight=.02)
    set_losses = legacy.proposal_object_set_loss(set_output, proposals['atom_presence_targets'],
        proposals['bond_presence_targets'], atoms['atom_mask'], bonds['bond_mask'],
        proposals['target_atom_refs'], proposals['bond_supervision_mask'],
        atom_decision_threshold=atom_threshold, bond_decision_threshold=bond_threshold)
    pairs, rows = legacy.enumerate_unordered_atom_pairs(selected['atom_mask'])
    with torch.no_grad():
        mapping, _ = legacy.match_atom_targets(selected, batch, config)
    pair_targets = legacy.build_pair_targets(pairs, rows, mapping, batch, selected)
    chosen, pair_mask = sample_supervised_pairs(pairs, rows, mapping, selected, pair_targets,
                                               policy=MATCHED_ONLY, negative_ratio=6, minimum_negatives=48)
    # Keep the original differentiable zero and skip the head on an empty branch.
    zero = atom_losses['loss'] * 0
    pair_loss = role_loss = bbox_loss = zero
    if len(chosen):
        pair_output = model.pair(features, selected['atom_features'], selected['atom_geometry'][..., :2],
                                selected['atom_scores'], pairs[chosen], rows[chosen])
        pair_loss = legacy.direct_atom_pair_loss(pair_output, pair_targets['presence'][chosen],
                                                 pair_targets['chemical'][chosen])['loss']
    atom_indices, sample_indices = legacy.enumerate_selected_atoms(selected['atom_mask'])
    mapped = mapping[sample_indices, atom_indices]
    valid = mapped >= 0
    atom_indices, sample_indices, mapped = atom_indices[valid], sample_indices[valid], mapped[valid]
    text = batch['atom_text_mask'][sample_indices, mapped]
    if len(atom_indices):
        anchors = selected['atom_geometry'][..., :2].detach()
        if model.update_recipe != 'freeze-role-v1':
            role_logits = model.role(batch['pixel_values'], anchors, atom_indices, sample_indices)
            labels = batch['atom_roles'][sample_indices, mapped].float()
            balance = ((labels == 0).sum().clamp_min(1) / (labels == 1).sum().clamp_min(1)).clamp(1, 8)
            role_loss = legacy.F.binary_cross_entropy_with_logits(role_logits.float(), labels, pos_weight=balance)
        if bool(text.any()):
            output = model.bbox(batch['pixel_values'], anchors, atom_indices[text], sample_indices[text])
            bbox_loss = legacy.direct_atom_text_bbox_loss(output, anchors[sample_indices[text], atom_indices[text]],
                batch['atom_text_bbox'][sample_indices[text], mapped[text]], model.bbox.config)['loss']
    losses = dict(atom=atom_losses['loss'], bond=bond_losses['loss'], set_head=set_losses['loss'],
                  pair=pair_loss, role=role_loss, bbox=bbox_loss)
    losses['loss'] = sum(losses.values())
    if any(not bool(torch.isfinite(value)) for value in losses.values()):
        raise ValueError('Non-finite continuation loss; no optimizer step is allowed')
    coverage = dict(target_atoms=int(batch['atom_mask'].sum()), target_bonds=int(batch['bond_mask'].sum()),
        raw_predicted_atoms=int(atoms['atom_mask'].sum()), selected_atoms=int(selected['atom_mask'].sum()),
        matched_raw_atoms=int(proposals['atom_target_matched'].sum()), matched_selected_atoms=len(mapped),
        pair_rows=len(chosen), reachable_positive_pairs=int(pair_targets['presence'].sum()),
        role_rows=len(mapped), bbox_rows=int(text.sum()),
        role_update_rows=0 if model.update_recipe == 'freeze-role-v1' else len(mapped),
        role_frozen=model.update_recipe == 'freeze-role-v1', oracle_candidates_injected=False,
        pair_supervision=MATCHED_ONLY, pair_total_rows=len(pairs), pair_ignored_rows=int((~pair_mask).sum()),
        pair_supervised_rows=int(pair_mask.sum()),
        pair_supervised_negative_rows=int((pair_mask & ~pair_targets['presence']).sum()),
        pair_sampled_positive_rows=int(pair_targets['presence'][chosen].sum()),
        pair_sampled_negative_rows=int((~pair_targets['presence'][chosen]).sum()),
        pair_branch_skipped=not len(chosen))
    return losses, coverage
