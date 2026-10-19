"""One saved update's local gradient directions, NEVER an optimizer step.

Loss-gradient projections describe an infinitesimal SGD direction, not an Adam
counterfactual. The actual Adam parameter delta projection is first-order too;
its residual against the observed logit change must remain visible.
"""

from copy import deepcopy

import torch
from torch.nn import functional as F

from . import pair_cached_trial as cached
from .pair_observed_capture import ATOL, RTOL, cpu_copy, mixed_digest


def check_adam_transition(event):
    """Read-only arithmetic on saved tensors; no model or optimizer invocation."""
    state, before, after = event['optimizer_before'], event['pair_before'], event['pair_after']
    groups = state['param_groups']
    if len(groups) != 1 or groups[0]['weight_decay'] != 0 or groups[0]['amsgrad'] or groups[0]['maximize']:
        raise ValueError('Only the unchanged single-group AdamW recipe is supported')
    group = groups[0]; b1, b2 = group['betas']
    error = 0.
    for name, param_id in zip(event['parameter_names'], group['params'], strict=True):
        p, gradient = before[name], event['gradients_after_clip'][name]
        if gradient is None:
            predicted = p
        else:
            previous = state['state'].get(param_id, {})
            step = int(previous.get('step', 0)) + 1
            m = previous.get('exp_avg', torch.zeros_like(p)) * b1 + gradient * (1. - b1)
            v = previous.get('exp_avg_sq', torch.zeros_like(p)) * b2 + gradient.square() * (1. - b2)
            predicted = p - (group['lr'] / (1. - b1**step)) * m / (
                v.sqrt() / (1. - b2**step)**.5 + group['eps'])
        error = max(error, float((predicted - after[name]).abs().max()))
        if not torch.allclose(predicted, after[name], atol=1e-6, rtol=1e-5):
            raise ValueError('Saved gradients/Adam state cannot explain actual parameter update')
    return dict(max_parameter_absolute_error=error, atol=1e-6, rtol=1e-5,
                optimizer_steps=0, verifies_update_arithmetic_not_loss_causality=True)


def choose_analysis_event(events):
    choices = [(p['presence_logit_delta'], -step, p['probe']['key'], step)
               for step, e in events.items() for p in e['record']['probes']
               if p['presence_logit_delta'] > ATOL]
    if not choices: return None
    _, _, key, step = max(choices)
    return step, key


def analyze_one_event(pair_head, event, probe_cache, binding):
    device = next(pair_head.parameters()).device
    devices = [device.index] if device.type == 'cuda' else []
    original = mixed_digest(pair_head.state_dict())
    with torch.random.fork_rng(devices=devices):
        torch.random.set_rng_state(event['cpu_rng'])
        if device.type == 'cuda': torch.cuda.set_rng_state(event['cuda_rng'], device)
        model = deepcopy(pair_head).to(device).eval()
        model.load_state_dict(event['pair_before'], strict=True)
        for p in model.parameters(): p.grad = None
        params = list(model.parameters())
        names = [n for n, _ in model.named_parameters()]
        if names != event['parameter_names']: raise ValueError('Captured parameter names changed')
        cache = event['cache']; cached.verify_cache(cache); cached.verify_cache(probe_cache)
        data = cache['tensors']
        rows = torch.tensor(event['record']['update']['sampled_row_ids'], dtype=torch.long)
        features, selected = cached.pair_inputs(cache, device)
        output = model(features, selected['atom_features'], selected['atom_geometry'][..., :2],
            selected['atom_scores'], data['pair_indices'][rows].to(device), data['sample_indices'][rows].to(device))
        positive = data['labels']['presence'][rows].to(device)
        losses = cached.direct_atom_pair_loss(output, positive, data['labels']['chemical'][rows].to(device))
        for k, v in losses.items():
            if not torch.isclose(v.detach().cpu(), torch.tensor(event['record']['update']['losses'][k]), atol=ATOL, rtol=RTOL):
                raise ValueError('Saved event loss replay mismatch')
        weight = ((~positive).sum().clamp_min(1) / positive.sum().clamp_min(1)).clamp(1., 16.)
        bce = F.binary_cross_entropy_with_logits(output['presence_logits'].float(), positive.float(),
                                                pos_weight=weight, reduction='none')
        terms = dict(presence_positive=(bce * positive).mean(),
                     presence_negative=(bce * ~positive).mean(), chemical=losses['chemical_loss'])
        gradients = {k: torch.autograd.grad(v, params, retain_graph=True, allow_unused=True)
                     for k, v in terms.items()}
        total_gradient = [sum((g[i] for g in gradients.values() if g[i] is not None),
                             torch.zeros_like(p)) for i, p in enumerate(params)]
        clipped_error = 0.
        for name, gradient in zip(names, total_gradient):
            recorded = event['gradients_after_clip'][name]
            if recorded is None: recorded = torch.zeros_like(gradient, device='cpu')
            predicted = (gradient.detach() * event['clip_coefficient']).cpu()
            clipped_error = max(clipped_error, float((predicted - recorded).abs().max()))
            if not torch.allclose(predicted, recorded, atol=ATOL, rtol=RTOL):
                raise ValueError('Loss decomposition does not reconstruct clipped gradient')
        probe_data = probe_cache['tensors']; r = binding['row_id']
        features, selected = cached.pair_inputs(probe_cache, device)
        z = model(features, selected['atom_features'], selected['atom_geometry'][..., :2],
            selected['atom_scores'], probe_data['pair_indices'][r:r + 1].to(device),
            probe_data['sample_indices'][r:r + 1].to(device))['presence_logits'][0]
        record = next(p for p in event['record']['probes'] if p['probe']['key'] == binding['key'])
        if not torch.isclose(z.detach().cpu(), torch.tensor(record['before']['presence_logit']), atol=ATOL, rtol=RTOL):
            raise ValueError('Saved probe replay mismatch')
        sensitivity = torch.autograd.grad(z, params, allow_unused=True)

        def dot(left, right):
            return sum(float((a.detach().double() * b.detach().to(device).double()).sum())
                       for a, b in zip(left, right) if a is not None and b is not None)

        actual_delta = [event['pair_after'][n] - event['pair_before'][n] for n in names]
        projected = dot(sensitivity, actual_delta)
        report = dict(step=event['step'], probe_key=binding['key'],
            updated_sample_key=event['cache']['key'],
            loss_values={k: float(v.detach()) for k, v in terms.items()},
            infinitesimal_sgd_logit_direction={k: -dot(sensitivity, g) for k, g in gradients.items()},
            actual_parameter_delta_first_order_logit_change=projected,
            observed_logit_change=record['presence_logit_delta'],
            first_order_residual=record['presence_logit_delta'] - projected,
            clipped_gradient_replay_max_absolute_error=clipped_error,
            clip_coefficient=event['clip_coefficient'], pair_forwards=2, optimizer_steps=0,
            loss_component_causality_proven=False,
            interpretation='Local SGD directions are not additive Adam causal contributions; no counterfactual update')
        del model, gradients, params
    if mixed_digest(pair_head.state_dict()) != original:
        raise ValueError('Analysis mutated the trained head')
    cached.verify_cache(cache); cached.verify_cache(probe_cache)
    return report
