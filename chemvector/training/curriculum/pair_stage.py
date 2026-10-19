"""Opt-in pair-only objective. No default wiring, launcher or optimizer update.

The caller owns a separate component instance. Existing candidate generation
and thresholds are unchanged; OCR/role/bbox are not called by this objective.
"""

import torch
from torch import nn

from chemvector.direct_atom_pair_bond import direct_atom_pair_loss
from chemvector.train_direct_atom_pair_bond import match_atom_targets
from .continuation import structure_forward
from .supervision_contract import pair_targets_with_unknown, select_known_pair_rows


class FrozenUpstreamPairStage(nn.Module):
    """Only the existing pair head can learn; no ground-truth candidates."""

    def __init__(self, components, config, *, atom_threshold):
        super().__init__()
        if any(p.grad is not None for p in components.parameters()):
            raise ValueError('Use fresh components before gradients or optimizer creation')
        if not torch.isfinite(torch.tensor(atom_threshold)):
            raise ValueError('Finite existing atom threshold required')
        self.components, self.config, self.atom_threshold = components, config, atom_threshold
        self.components.requires_grad_(False)
        self.components.pair.requires_grad_(True)
        self.train()

    def train(self, mode=True):
        super().train(mode)
        self.components.eval()
        self.components.pair.train(mode)
        return self

    def pair_parameters(self):
        expected = {id(p) for p in self.components.pair.parameters()}
        actual = {id(p) for p in self.parameters() if p.requires_grad}
        if actual != expected:
            raise ValueError('Only all pair parameters may be trainable')
        return list(self.components.pair.parameters())

    def forward(self, batch):
        self.pair_parameters()
        if any(m.training for name in ('vision', 'atom', 'bond', 'set_head', 'role', 'bbox')
               for m in getattr(self.components, name).modules()):
            raise ValueError('Frozen interfaces must remain in eval mode')
        # no_grad, not inference_mode: resulting tensors may safely be inputs
        # to an autograd-enabled pair head without retaining the upstream graph.
        with torch.no_grad():
            forward = structure_forward(self.components, batch['pixel_values'], self.config, self.atom_threshold)
            features, selected = forward[0], forward[6]
            mapping, _ = match_atom_targets(selected, batch, self.config)
            contract = pair_targets_with_unknown(selected, mapping, batch)
            chosen = select_known_pair_rows(contract, selected, negative_ratio=6, minimum_negatives=48)
        coverage = {**contract['coverage'], 'loss_rows': len(chosen),
                    'sampled_unknown_pairs': int(contract['unknown_mask'][chosen].sum()),
                    'oracle_candidates_injected': False}
        if not len(chosen):
            # Explicit skip; do not fabricate a zero loss/optimizer step.
            return {'losses': None, 'coverage': coverage, 'skip_reason': 'no_known_pairs'}
        output = self.components.pair(features.detach(), selected['atom_features'].detach(),
            selected['atom_geometry'][..., :2].detach(), selected['atom_scores'].detach(),
            contract['pair_indices'][chosen], contract['sample_indices'][chosen])
        targets = contract['legacy_targets']
        losses = direct_atom_pair_loss(output, targets['presence'][chosen], targets['chemical'][chosen])
        if any(not bool(torch.isfinite(v)) for v in losses.values()):
            raise ValueError('Non-finite pair loss')
        return {'losses': losses, 'coverage': coverage, 'skip_reason': None}
