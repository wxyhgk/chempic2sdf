"""129-parameter fitting primitives and identity-based regression gates."""

from hashlib import sha256
import random

import torch

from chemvector.predicted_incidence import decode_predicted_atom_proposals
from chemvector.vertex import focal_heatmap_loss
from .selector_retention_stage import call_selector, tensors_to
from tools import selector_radius_reference as matching

PARAMETER_NAMES = {'atom.heatmap.weight', 'atom.heatmap.bias'}


def configure(model):
    model.eval().requires_grad_(False)
    model.atom.heatmap.requires_grad_(True)
    return parameters(model)


def parameters(model):
    if (any(m.training for m in model.modules())
            or any(p.requires_grad != (n in PARAMETER_NAMES) for n, p in model.named_parameters())
            or any(p.grad is not None for n, p in model.named_parameters() if n not in PARAMETER_NAMES)):
        raise ValueError('Only heatmap weight/bias may train, all modules must remain eval')
    result = tuple(p for n, p in model.named_parameters() if n in PARAMETER_NAMES)
    if len(result) != 2 or sum(p.numel() for p in result) != 129:
        raise ValueError('Exactly 129 original readout parameters required')
    return result


def frozen_digest(model):
    digest = sha256()
    for name, value in sorted(model.state_dict().items()):
        if name not in PARAMETER_NAMES:
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def refresh_proposals(model, cache, config, device):
    """Cache features/static outputs, NEVER the heatmap-derived proposals."""
    output = tensors_to(cache['atom_static'], device)
    output['heatmap_logits'] = model.atom.heatmap(output['object_features'])
    raw = decode_predicted_atom_proposals(output, config)
    selected_logits = call_selector(model.set_head, raw, tensors_to(cache['bonds'], device))
    if any(not bool(torch.isfinite(v).all()) for v in selected_logits.values()):
        raise ValueError('Nonfinite selector output')
    selected = {**raw, 'atom_mask': raw['atom_mask'] & (selected_logits['atom_presence_logits'] >= -.25)}
    return raw, selected, selected_logits


def order(caches, manifest_sha):
    if (len(caches) != 58 or len({c['key'] for c in caches}) != 58
            or any(c['purpose'] != 'training' or c['training_authorized'] is not True
                   or c['manifest_sha256'] != manifest_sha for c in caches)):
        raise ValueError('Exact admitted58 training caches required')
    pools = {group: [c for c in caches if c['group'] == group] for group in ('old_full_molecule', 'witness')}
    if [len(pools[g]) for g in pools] != [32, 26]:
        raise ValueError('Fixed replay/witness counts required')
    rng, queues = random.Random(20260920), {g: [] for g in pools}
    while True:
        for group in ('old_full_molecule', 'old_full_molecule', 'witness'):
            if not queues[group]:
                queues[group] = list(pools[group])
                rng.shuffle(queues[group])
            yield queues[group].pop()


def update(model, cache, optimizer, limit, device, reserve, manifest_sha):
    ps = parameters(model)
    if (cache.get('purpose') != 'training' or cache.get('training_authorized') is not True
            or cache.get('manifest_sha256') != manifest_sha):
        raise ValueError('Unadmitted/diagnostic cache cannot fit')
    ids = [id(p) for g in optimizer.param_groups for p in g['params']]
    if len(ids) != 2 or set(ids) != {id(p) for p in ps}:
        raise ValueError('Optimizer scope differs')
    if type(optimizer) is not torch.optim.SGD or any(g['lr'] != .001 or g['momentum'] != 0
            or g['weight_decay'] != 0 or g['nesterov'] or g['maximize'] for g in optimizer.param_groups):
        raise ValueError('Only fixed SGD 1e-3, no momentum/decay permitted')
    hidden, heat = cache['atom_static']['object_features'], cache['target_heatmap']
    if hidden.requires_grad or heat.requires_grad or hidden.device.type != 'cpu' or heat.device.type != 'cpu':
        raise ValueError('Detached frozen features and targets required')
    optimizer.zero_grad(set_to_none=True)
    loss = focal_heatmap_loss(model.atom.heatmap(hidden.to(device)), heat.to(device))
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite loss')
    loss.backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in ps):
        raise ValueError('Missing/nonfinite readout gradients')
    norm = torch.nn.utils.clip_grad_norm_(ps, 1., error_if_nonfinite=True)
    limit.consume()
    reserve(limit.consumed)
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in ps):
        raise ValueError('Nonfinite readout weights')
    return dict(loss=float(loss.detach()), gradient_norm=float(norm))


@torch.no_grad()
def coverage(raw, selected, logits, layers, target_anchors):
    raw, selected, logits = (tensors_to(v, 'cpu') for v in (raw, selected, logits))
    slots = raw['atom_mask'][0].nonzero().flatten().tolist()
    geometry = dict(image_size=768, match_radius_px=6, target_anchors=target_anchors,
        candidates=[dict(slot=s, grid_id=int(raw['atom_indices'][0, s]),
            geometry=raw['atom_geometry'][0, s].tolist(), raw_score=float(raw['atom_scores'][0, s]),
            selector_logit=float(logits['atom_presence_logits'][0, s]),
            selected=bool(selected['atom_mask'][0, s])) for s in slots])
    chosen = selected['atom_mask'][0].nonzero().flatten().tolist()
    return dict(geometry=geometry, raw=matching.compare(geometry, slots, layers),
                selected=matching.compare(geometry, chosen, layers))


def gate(baseline, current, focus, step):
    before, after = ({r['sample_key']: r for r in rows} for rows in (baseline, current))
    if len(before) != len(baseline) or before.keys() != after.keys() or len(after) != len(current):
        raise ValueError('Same unique paired panel required')
    lost, recovered = [], []
    for key, old in before.items():
        new = after[key]
        for policy in ('original', 'radius_reference'):
            for which in ('raw', 'selected'):
                for field in ('matched_target_ids', 'reachable_bond_ids'):
                    ids = sorted(set(old[which][policy][field]) - set(new[which][policy][field]))
                    if ids:
                        lost.append(dict(sample_key=key, stage=which, policy=policy, field=field, ids=ids))
        if old['prediction']['exact'] and not new['prediction']['exact']:
            lost.append(dict(sample_key=key, stage='sdf', field='exact', ids=[]))
        actual_lost = sorted(set(old['correct_typed_bond_ids']) - set(new['correct_typed_bond_ids']))
        if actual_lost:
            lost.append(dict(sample_key=key, stage='pair', field='correct_typed_bond_ids', ids=actual_lost))
        focus_ids = set(focus.get(key, []))
        old_missing = set(old['raw']['original']['no_eligible_candidate_target_ids'])
        if not focus_ids <= old_missing:
            raise ValueError('Focused historical raw gaps not reproduced at baseline')
        new_present = set(new['raw']['original']['matched_target_ids'])
        recovered.extend(dict(sample_key=key, target_id=t) for t in sorted(focus_ids & new_present))
    stop = []
    if lost:
        stop.append('baseline_correct_atom_or_bond_or_graph_lost')
    if step >= 200 and not recovered:
        stop.append('no_actual_raw_recovery_by_step200')
    extra_before = sum(r['selected']['original']['unmatched_selected'] for r in baseline)
    extra_after = sum(r['selected']['original']['unmatched_selected'] for r in current)
    false_bonds_before = sum(r['false_or_wrong_typed_pairs'] for r in baseline)
    false_bonds_after = sum(r['false_or_wrong_typed_pairs'] for r in current)
    return dict(step=step, stop_reasons=stop, lost_baseline_relations=lost,
        recovered_focused_targets=recovered, recovered_count=len(recovered),
        selected_extra_before=extra_before, selected_extra_after=extra_after,
        false_typed_pairs_before=false_bonds_before, false_typed_pairs_after=false_bonds_after,
        local_pass=not stop and len(recovered) >= 3 and extra_after <= extra_before
                   and false_bonds_after <= false_bonds_before,
        generalization_established=False, deployment_allowed=False)


def summary(rows):
    result = dict(images=len(rows), strict_sdf_exact=sum(r['prediction']['exact'] for r in rows),
        correct_typed_bonds=sum(len(r['correct_typed_bond_ids']) for r in rows))
    for stage in ('raw', 'selected'):
        for field in ('matched_target_ids', 'reachable_bond_ids'):
            result[stage+'_'+field] = sum(len(r[stage]['original'][field]) for r in rows)
        result[stage+'_extra'] = sum(r[stage]['original']['unmatched_selected'] for r in rows)
    return result
