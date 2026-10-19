"""Read-only hooks and pair-only gradient checks for the real loss entry.

No replacement predictions, label-fed model input, optimizer or monkeypatch.
CPU arm equality is distinct from historical CPU/GPU candidate correspondence.
"""

import math

import torch

from chemvector.training.curriculum import continuation as legacy
from chemvector.training.curriculum import continuation_pair_supervision as wired
from chemvector.training.curriculum import pair_supervision as supervision
from chemvector.training.curriculum.gradient_preflight import state_digest
from chemvector.training.curriculum.interface_diagnostic import tensor_tree_digest


def clone_tree(value):
    return {k: v.detach().clone() for k, v in value.items()}


class Capture:
    """Observe only image-derived module outputs and actual sampled pair IDs."""

    def __init__(self, model, config):
        self.model, self.config = model, config
        self.raw = self.logits = None
        self.input_digest = None
        self.pairs, self.samples = None, None
        self.calls = dict(atom=0, selector=0, pair=0, role=0, bbox=0)
        self.text_traces = dict(role=[], bbox=[])
        self.handles = []

    def atom(self, module, args, output):
        self.calls['atom'] += 1
        # Decode the SAME output only to observe grid IDs; the actual selector
        # inputs are checked below. This adds no image/model forward.
        self.raw = clone_tree(legacy.decode_predicted_atom_proposals(output, self.config))

    def selector_before(self, module, args):
        if self.raw is None:
            raise ValueError('Selector preceded atom decoder')
        for field, observed in zip(('atom_features', 'atom_geometry', 'atom_scores', 'atom_mask'), args[:4]):
            if not torch.equal(self.raw[field], observed):
                raise ValueError('Observed decoder differs from actual selector input')
        self.input_digest = tensor_tree_digest({str(i): t for i, t in enumerate(args)})

    def selector_after(self, module, args, output):
        self.calls['selector'] += 1
        self.logits = output['atom_presence_logits'].detach().clone()

    def pair(self, module, args):
        self.calls['pair'] += 1
        self.pairs, self.samples = args[4].detach().clone(), args[5].detach().clone()

    def text(self, name, args, output):
        self.calls[name] += 1
        self.text_traces[name].append(dict(input=tensor_tree_digest({str(i): t for i, t in enumerate(args)}),
                                           output=tensor_tree_digest(output)))

    def __enter__(self):
        self.handles = [self.model.atom.register_forward_hook(self.atom),
            self.model.set_head.register_forward_pre_hook(self.selector_before),
            self.model.set_head.register_forward_hook(self.selector_after),
            self.model.pair.register_forward_pre_hook(self.pair)]
        for name in ('role', 'bbox'):
            self.handles.append(getattr(self.model, name).register_forward_hook(
                lambda module, args, output, name=name: self.text(name, args, output)))
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()


def require_scope(model, batch):
    if batch.get('purpose') != 'diagnostic' or batch.get('training_authorized') is not False:
        raise ValueError('Diagnostic-only batch required')
    if (batch['pixel_values'].device.type != 'cpu' or any(m.training for m in model.modules())
            or any(p.device.type != 'cpu' or p.requires_grad != n.startswith('pair.')
                   for n, p in model.named_parameters())):
        raise ValueError('CPU eval and pair-only autograd scope required')


def run_arm(model, batch, config, *, policy, atom_threshold, bond_threshold, seed):
    supervision.validate_policy(policy)
    require_scope(model, batch)
    before = state_digest(model)
    model.zero_grad(set_to_none=True)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        with Capture(model, config) as observed:
            losses, coverage = wired.continuation_losses(model, batch, config,
                atom_threshold=atom_threshold, bond_threshold=bond_threshold, pair_supervision=policy)
        values = {k: float(v.detach()) for k, v in losses.items()}
        backward = bool(losses['pair'].requires_grad)
        if backward:
            losses['pair'].backward()
    if observed.calls['atom'] != 1 or observed.calls['selector'] != 1:
        raise ValueError('Exactly one upstream forward per arm required')
    gradients = [p.grad for p in model.pair.parameters() if p.grad is not None]
    if any(not bool(torch.isfinite(g).all()) for g in gradients):
        raise ValueError('Nonfinite pair gradient')
    if any(p.grad is not None for n, p in model.named_parameters() if not n.startswith('pair.')):
        raise ValueError('Frozen component received gradient')
    if before != state_digest(model):
        raise ValueError('Forward/backward changed weights or buffers')
    raw, logits = observed.raw, observed.logits
    selected = {**raw, 'atom_mask': raw['atom_mask'] & (logits >= atom_threshold)}
    with torch.no_grad():
        mapping, matched = legacy.match_atom_targets(selected, batch, config)
        all_pairs, all_samples = legacy.enumerate_unordered_atom_pairs(selected['atom_mask'])
        labels = legacy.build_pair_targets(all_pairs, all_samples, mapping, batch, selected)
        mask = supervision.supervision_mask(all_pairs, all_samples, mapping, labels['presence'], policy=policy)
    by_pair = {(int(sample), *pair): i for i, (sample, pair) in enumerate(zip(all_samples, all_pairs.tolist()))}
    seen = [] if observed.pairs is None else [by_pair[(int(sample), *pair)]
        for sample, pair in zip(observed.samples, observed.pairs.tolist())]
    if len(seen) != len(set(seen)) or any(not mask[i] for i in seen):
        raise ValueError('Actual head received duplicate or ignored pairs')
    positives = labels['presence'].nonzero().flatten().tolist()
    if set(positives) - set(seen):
        raise ValueError('Actual head lost original positive pairs')
    if observed.calls['pair'] != int(bool(seen)) or len(seen) != coverage['pair_rows']:
        raise ValueError('Pair head call/count mismatch')
    if bool(seen) != backward or bool(seen) != bool(gradients):
        raise ValueError('Pair gradient/empty-branch contract failed')
    if coverage['matched_selected_atoms'] != int(matched.sum()):
        raise ValueError('Loss and replay matching disagree')
    # Identify risk without importing historical slot labels: an unmatched
    # endpoint may be a misplaced atom or a fragment. Do not assert its identity.
    uncertain = (~supervision.supervision_mask(all_pairs, all_samples, mapping, labels['presence'],
                                              policy=supervision.MATCHED_ONLY)).nonzero().flatten().tolist()
    summary = dict(policy=policy, losses=values, coverage=coverage, module_calls=observed.calls,
        upstream_sha256=tensor_tree_digest(dict(raw=raw, logits=logits)),
        selector_input_sha256=observed.input_digest, text_traces=observed.text_traces,
        selected_slot_ids=selected['atom_mask'][0].nonzero().flatten().tolist(),
        original_positive_rows=positives, sampled_pair_rows=seen,
        sampled_candidate_pairs=[] if observed.pairs is None else observed.pairs.tolist(),
        sampled_pairs_with_unmatched_endpoint=len(set(seen) & set(uncertain)),
        ignored_rows=int((~mask).sum()), pair_backward_performed=backward,
        pair_gradient_tensors=len(gradients),
        pair_gradient_l2=math.sqrt(sum(float(g.double().square().sum()) for g in gradients)),
        frozen_gradients_absent=True, state_before=before, state_after=state_digest(model))
    snapshot = dict(slots=raw['atom_mask'][0].nonzero().flatten(), grid_ids=raw['atom_indices'][0],
                    geometry=raw['atom_geometry'][0], raw_scores=raw['atom_scores'][0], logits=logits[0])
    model.zero_grad(set_to_none=True)
    return summary, snapshot


def compare_arms(old, new):
    if (old['policy'] != supervision.LEGACY or new['policy'] != supervision.MATCHED_ONLY
            or old['upstream_sha256'] != new['upstream_sha256']
            or old['selector_input_sha256'] != new['selector_input_sha256']
            or old['selected_slot_ids'] != new['selected_slot_ids']):
        raise ValueError('CPU arm candidates/context differ: do not attribute effects to the mask')
    for name in ('atom', 'bond', 'set_head', 'role', 'bbox'):
        if old['losses'][name] != new['losses'][name]:
            raise ValueError('Non-pair loss changed: ' + name)
    if old['text_traces'] != new['text_traces'] or old['original_positive_rows'] != new['original_positive_rows']:
        raise ValueError('Text predictions or positive labels changed')
    if new['sampled_pairs_with_unmatched_endpoint']:
        raise ValueError('Unmatched endpoint still entered the masked pair head')
    return dict(exact_cpu_candidate_context_parity=True, non_pair_losses_exact=True,
                text_path_exact=True, original_positives_preserved=True,
                all_losses_equal=old['losses'] == new['losses'],
                baseline_sampled_unmatched_pairs=old['sampled_pairs_with_unmatched_endpoint'],
                masked_sampled_unmatched_pairs=new['sampled_pairs_with_unmatched_endpoint'])
