"""Opt-in presence-readout loss on frozen predicted features.

This prototype has no loader, launcher, optimizer step or release authority.
It is not the full-pair objective and is not registered with continuation.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from chemvector.bond_object import BOND_TYPES
from chemvector.train_direct_atom_pair_bond import match_atom_targets
from .continuation import structure_forward
from .supervision_contract import pair_targets_with_unknown


def presence_loss(logits, targets):
    """Existing positive-weighted BCE convention, without any chemical loss."""
    if (logits.ndim != 1 or not logits.numel() or targets.shape != logits.shape
            or targets.dtype != torch.bool or targets.device != logits.device
            or not bool(torch.isfinite(logits).all())):
        raise ValueError('Finite nonempty logits and aligned boolean targets required')
    positive = targets.sum().clamp_min(1)
    negative = (~targets).sum().clamp_min(1)
    loss = F.binary_cross_entropy_with_logits(logits.float(), targets.float(),
        pos_weight=(negative / positive).clamp(1., 16.))
    if not bool(torch.isfinite(loss)):
        raise ValueError('Non-finite presence loss')
    return {'loss': loss, 'presence_loss': loss}


class FrozenFeaturePresenceStage(nn.Module):
    """Only the existing pair.presence linear weight/bias may receive gradients.

The whole original pair forward runs under no_grad in eval mode. Its detached
hidden features feed the SAME existing presence layer once more with autograd.
Upstream features, candidate selection, shared pair trunk and raw chemical
logits are fixed; downstream graph/aromatic decisions can still depend on the
new presence probabilities and therefore are NOT promised to stay identical.
"""

    def __init__(self, components, config, *, atom_threshold):
        super().__init__()
        if isinstance(atom_threshold, bool) or not math.isfinite(atom_threshold):
            raise ValueError('Finite existing atom threshold required')
        if any(p.grad is not None for p in components.parameters()):
            raise ValueError('Use fresh components before gradients or optimizer creation')
        presence, chemical = components.pair.presence, components.pair.chemical
        if (not isinstance(presence, nn.Linear) or presence.out_features != 1 or presence.bias is None
                or not isinstance(chemical, nn.Linear) or chemical.out_features != len(BOND_TYPES)
                or chemical.in_features != presence.in_features):
            raise ValueError('Expected existing linear presence/chemical readouts of the same hidden features')
        expected = frozenset(id(p) for p in presence.parameters())
        if any(id(p) in expected and not name.startswith('pair.presence.')
               for name, p in components.named_parameters(remove_duplicate=False)):
            raise ValueError('Presence parameters may not alias a frozen component')
        self.components, self.config, self.atom_threshold = components, config, atom_threshold
        self._presence_parameter_ids = expected
        self.components.requires_grad_(False)
        presence.requires_grad_(True)
        self.train()

    def train(self, mode=True):
        super().train(mode)
        self.components.eval()
        self.components.pair.presence.train(mode)
        return self

    def presence_parameters(self):
        parameters = list(self.components.pair.presence.parameters())
        expected = {id(p) for p in parameters}
        actual = {id(p) for p in self.parameters() if p.requires_grad}
        if expected != self._presence_parameter_ids or actual != expected:
            raise ValueError('Only the original presence readout weight/bias may be trainable')
        if any(id(p) in expected and not name.startswith('pair.presence.')
               for name, p in self.components.named_parameters(remove_duplicate=False)):
            raise ValueError('Presence parameters may not alias a frozen component')
        if any(p.grad is not None for p in self.parameters() if id(p) not in expected):
            raise ValueError('Frozen parameters have stale gradients')
        return parameters

    def forward(self, batch):
        self.presence_parameters()
        readout = self.components.pair.presence
        if any(m.training for m in self.components.modules() if m is not readout):
            raise ValueError('All frozen components, including the shared pair trunk, must remain in eval mode')
        with torch.no_grad():
            forward = structure_forward(self.components, batch['pixel_values'], self.config, self.atom_threshold)
            features, selected = forward[0], forward[6]
            mapping, _ = match_atom_targets(selected, batch, self.config)
            contract = pair_targets_with_unknown(selected, mapping, batch)
            known = contract['known_mask']
            coverage = {**contract['coverage'], 'loss_rows': int(known.sum()),
                'loss_positive_rows': int(contract['legacy_targets']['presence'][known].sum()),
                'loss_negative_rows': int((~contract['legacy_targets']['presence'][known]).sum()),
                'all_confirmed_non_edges_retained': True, 'sampled_unknown_pairs': 0,
                'oracle_candidates_injected': False}
            if not bool(known.any()):
                return {'losses': None, 'coverage': coverage, 'skip_reason': 'no_known_pairs'}
            # Use EVERY predicted pair, including unknown ones, in original
            # order. Truth changes loss eligibility only, never pair features.
            output = self.components.pair(features.detach(), selected['atom_features'].detach(),
                selected['atom_geometry'][..., :2].detach(), selected['atom_scores'].detach(),
                contract['pair_indices'], contract['sample_indices'])
            hidden, chemical = output['hidden'], output['chemical_logits']
            count = len(contract['pair_indices'])
            if (hidden.shape != (count, readout.in_features)
                    or chemical.shape != (count, len(BOND_TYPES))
                    or output['presence_logits'].shape != (count,)
                    or any(not bool(torch.isfinite(output[k]).all())
                           for k in ('hidden', 'chemical_logits', 'presence_logits'))):
                raise ValueError('Invalid frozen pair output or hidden feature contract')
        logits = readout(hidden.detach()).squeeze(-1)
        # Exact same linear readout and row order; this is not a new head.
        if not torch.equal(logits.detach(), output['presence_logits']):
            raise ValueError('Presence readout differs from the unmodified pair path')
        losses = presence_loss(logits[known], contract['legacy_targets']['presence'][known])
        return {'losses': losses, 'coverage': coverage, 'skip_reason': None,
            'presence_logits': logits, 'frozen_chemical_logits': chemical.detach(),
            'frozen_hidden': hidden.detach(), 'known_mask': known.detach(),
            'pair_indices': contract['pair_indices'], 'sample_indices': contract['sample_indices']}
