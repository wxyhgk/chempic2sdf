"""Opt-in negative-pair telemetry for a FUTURE authorized trial; no optimizer.

Not wired into the historical launcher. Resolve targets on each fresh cache,
never carry slot IDs from historical CPU/GPU candidates. These target-selected
probes are diagnostics, not inference inputs or a new supervision source.
"""

import torch

from .pair_cached_trial import pair_inputs, verify_cache


def bind_negative_probe(cache, target_pair):
    verify_cache(cache)
    targets = list(target_pair)
    if (len(targets) != 2 or any(type(t) is not int or t < 0 for t in targets)
            or targets[0] >= targets[1]):
        raise ValueError('Two ordered distinct target IDs required')
    data = cache['tensors']
    if data['mapping'].shape[0] != 1:
        raise ValueError('Single-image cache required')
    slots = []
    for target in targets:
        candidates = torch.where((data['mapping'][0] == target) & data['selected']['atom_mask'][0])[0]
        if len(candidates) != 1:
            raise ValueError('Probe target must have exactly one current selected match')
        slots.append(int(candidates[0]))
    pair = sorted(slots)
    rows = torch.where((data['pair_indices'] == torch.tensor(pair)).all(-1)
                       & (data['sample_indices'] == 0))[0]
    if len(rows) != 1:
        raise ValueError('Probe pair is missing or ambiguous')
    row = int(rows[0])
    if not bool(data['known'][row]) or bool(data['labels']['presence'][row]):
        raise ValueError('Probe must be a confirmed nonedge, not a positive or unknown')
    return dict(key=cache['key'], target_pair=targets, candidate_pair=pair, row_id=row,
                cache_sha256=cache['cache_sha256'], label=False)


@torch.no_grad()
def observe(pair_head, cache, binding):
    if any(module.training for module in pair_head.modules()):
        raise ValueError('Probe requires the trial eval mode')
    # Revalidate metadata too: callers cannot retain a true digest while
    # silently replacing the row, target IDs or candidate pair of a probe.
    if bind_negative_probe(cache, binding['target_pair']) != binding:
        raise ValueError('Probe binding differs from the current immutable cache')
    device = next(pair_head.parameters()).device
    features, selected = pair_inputs(cache, device)
    row = binding['row_id']
    data = cache['tensors']
    output = pair_head(features, selected['atom_features'], selected['atom_geometry'][..., :2],
        selected['atom_scores'], data['pair_indices'][row:row + 1].to(device),
        data['sample_indices'][row:row + 1].to(device))
    if (output['presence_logits'].shape != (1,) or output['chemical_logits'].shape[0] != 1
            or not bool(torch.isfinite(output['presence_logits']).all())
            or not bool(torch.isfinite(output['chemical_logits']).all())):
        raise ValueError('Invalid probe output')
    return dict(binding=binding, presence_logit=float(output['presence_logits'][0]),
                chemical_logits=output['chemical_logits'][0].tolist(), model_mode='eval')


def paired_update_record(before, after, *, step, updated_sample_key):
    if (type(step) is not int or step <= 0 or not isinstance(updated_sample_key, str)
            or not updated_sample_key or before['binding'] != after['binding']):
        raise ValueError('Exact same probe and explicit update identity required')
    a, b = before['presence_logit'], after['presence_logit']
    if not all(torch.isfinite(torch.tensor(v)) for v in (a, b)):
        raise ValueError('Finite probe scores required')
    return dict(step=step, updated_sample_key=updated_sample_key, probe=before['binding'],
        own_sample_update=updated_sample_key == before['binding']['key'],
        before=before, after=after, presence_logit_delta=b - a,
        negative_score_increased=b > a, diagnostic_only=True,
        identifies_observed_update_not_loss_component_causality=True)
