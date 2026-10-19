"""Device-bound adapter for a NEW execution's cache. No model/data loading."""

import torch

from chemvector.direct_atom_pair_bond import direct_atom_pair_loss
from . import pair_cached_trial as cached, pair_anchor_ab_contract as rules
from .pair_anchor_augmentation import plan_views, typed_augmentation_loss
from .pair_anchor_wiring import sampled_positions
from .interface_diagnostic import tensor_tree_digest
from .runtime import StepLimit
from .supervision_contract import select_known_pair_rows


def make_plan(cache, target_anchors, *, ordinal, execution_device):
    cached.verify_cache(cache)
    if type(ordinal) is not int or not 0 <= ordinal < 58 or str(torch.device(execution_device)) != execution_device:
        raise ValueError('Bound ordinal and canonical execution device required')
    t = cache['tensors']
    targets = target_anchors.detach().cpu().clone()
    if targets.shape != (1, cache['coverage']['target_atoms'], 2) or targets.dtype != torch.float32:
        raise ValueError('Same-execution packed target anchors required')
    contract = dict(known_mask=t['known'], legacy_targets=t['labels'],
                    pair_indices=t['pair_indices'], sample_indices=t['sample_indices'])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(rules.protocol()['recipe']['seed'] + ordinal)
        chosen = select_known_pair_rows(contract, t['selected'], negative_ratio=6, minimum_negatives=48)
    if set(t['labels']['presence'].nonzero().flatten().tolist()) - set(chosen.tolist()):
        raise ValueError('Every original positive must remain in the base branch')
    selected = t['selected']
    # One predeclared visit per image. The CPU guard uses a copy of current
    # execution-device predictions; matching is NOT recomputed on the CPU.
    view = plan_views(selected['atom_geometry'][0, :, :2], selected['atom_mask'][0], t['mapping'][0],
                      targets[0], t['pair_indices'], t['known'], chosen)[1]
    tensors = dict(target_anchors=targets, chosen=chosen, augmented_rows=view['rows'],
                   augmented_positions=sampled_positions(chosen, view['rows']), shifted_anchors=view['anchors'][None])
    result = dict(tensors=tensors, tensor_sha256=tensor_tree_digest(tensors), cache_sha256=cache['cache_sha256'],
                  ordinal=ordinal, view_index=1, offset_pixels=view['offset_pixels'], execution_device=execution_device)
    result['binding_sha256'] = cached.json_digest({k: v for k, v in result.items() if k != 'tensors'})
    return result


def verify_plan(cache, plan):
    cached.verify_cache(cache)
    if (cache['cache_sha256'] != plan['cache_sha256']
            or tensor_tree_digest(plan['tensors']) != plan['tensor_sha256']
            or cached.json_digest({k: v for k, v in plan.items() if k not in ('tensors', 'binding_sha256')}) != plan['binding_sha256']
            or plan['view_index'] != 1 or plan['offset_pixels'] != [-4., -4.]):
        raise ValueError('Bound current-candidate plan changed')


def objective(pair, cache, plan, *, weight):
    verify_plan(cache, plan)
    if type(weight) not in (int, float) or weight not in (0, 1) or any(m.training for m in pair.modules()):
        raise ValueError('Fixed weight and eval-mode pair required')
    params = list(pair.parameters())
    device = torch.device(plan['execution_device'])
    if not params or any(p.device != device for p in params):
        raise ValueError('No device migration/fallback across a frozen cache')
    t, p = cache['tensors'], plan['tensors']
    chosen = p['chosen']
    if not len(chosen): return dict(losses=None, forwards=0)
    features, selected = cached.pair_inputs(cache, device)
    def forward(anchors, rows):
        return pair(features, selected['atom_features'], anchors, selected['atom_scores'],
                    t['pair_indices'][rows].to(device), t['sample_indices'][rows].to(device))
    base = forward(selected['atom_geometry'][..., :2], chosen)
    use_extra = weight == 1 and len(p['augmented_rows']) > 0
    shifted = forward(p['shifted_anchors'].to(device), p['augmented_rows']) if use_extra else None
    positions = (p['augmented_positions'] if use_extra else p['augmented_positions'][:0]).to(device)
    losses = typed_augmentation_loss(base, shifted, t['labels']['presence'][chosen].to(device),
                                    t['labels']['chemical'][chosen].to(device), positions, weight=weight)
    if not bool(torch.isfinite(losses['loss'])): raise ValueError('Nonfinite loss before update')
    return dict(losses=losses, forwards=1+int(use_extra), base_output=base)


@torch.no_grad()
def zero_weight_parity(pair, cache, plan):
    adapted = objective(pair, cache, plan, weight=0)
    if adapted['losses'] is None: return dict(passed=True, forwards=0)
    t, chosen = cache['tensors'], plan['tensors']['chosen']
    device = torch.device(plan['execution_device'])
    features, atoms = cached.pair_inputs(cache, device)
    output = pair(features, atoms['atom_features'], atoms['atom_geometry'][..., :2], atoms['atom_scores'],
                  t['pair_indices'][chosen].to(device), t['sample_indices'][chosen].to(device))
    loss = direct_atom_pair_loss(output, t['labels']['presence'][chosen].to(device), t['labels']['chemical'][chosen].to(device))
    if (any(not torch.equal(output[k], adapted['base_output'][k]) for k in ('presence_logits', 'chemical_logits'))
            or any(not torch.equal(value, adapted['losses']['original'][k]) for k, value in loss.items())):
        raise ValueError('Fresh device zero-weight adapter parity failed')
    return dict(passed=True, forwards=2)


def update(stage, cache, plan, optimizer, arm_limit, total_limit, *, weight, emit):
    params = cached.assert_stage(stage)
    if (not isinstance(arm_limit, StepLimit) or not isinstance(total_limit, StepLimit)
            or arm_limit.maximum != 54 or total_limit.maximum != 108):
        raise ValueError('Exact per-arm and aggregate step limits required')
    opt_params = [p for group in optimizer.param_groups for p in group['params']]
    if len(opt_params) != len(params) or {id(p) for p in opt_params} != {id(p) for p in params}:
        raise ValueError('Optimizer must include exactly the active pair head')
    verify_plan(cache, plan)
    if not len(plan['tensors']['chosen']): return dict(updated=False, forwards=0)
    if arm_limit.consumed >= 54 or total_limit.consumed >= 108:
        raise RuntimeError('Step budget exhausted before forward')
    optimizer.zero_grad(set_to_none=True)
    result = objective(stage.components.pair, cache, plan, weight=weight)
    result['losses']['loss'].backward()
    gradients = [p.grad for p in params if p.grad is not None]
    if not gradients or any(not bool(torch.isfinite(g).all()) for g in gradients):
        raise ValueError('Nonfinite/absent gradients; no update')
    cached.assert_stage(stage)
    norm = torch.nn.utils.clip_grad_norm_(params, rules.protocol()['recipe']['gradient_clip'], error_if_nonfinite=True)
    arm_limit.consume(); total_limit.consume()
    emit('step_reserved', arm_step=arm_limit.consumed, total_step=total_limit.consumed)
    optimizer.step()
    if any(not bool(torch.isfinite(p).all()) for p in params): raise ValueError('Nonfinite updated pair')
    value = dict(updated=True, forwards=result['forwards'], loss=float(result['losses']['loss'].detach()),
                 gradient_norm=float(norm), base_rows=len(plan['tensors']['chosen']),
                 augmented_rows=len(plan['tensors']['augmented_rows']) if weight else 0)
    optimizer.zero_grad(set_to_none=True)
    return value


def score_case(cache, structure, prediction, trace, *, key, expected):
    metric = cached.score_case(cache, structure, prediction, trace, key=key, expected=expected)
    t = cache['tensors']; output = structure['output']
    present = output['presence_logits'].detach().cpu().sigmoid() >= rules.protocol()['recipe']['pair_threshold']
    positive, known = t['labels']['presence'], t['known']
    chemical = structure['chemistry'].detach().cpu()
    if chemical.shape != positive.shape or chemical.dtype != torch.long or bool(((chemical < 0) | (chemical > 3)).any()):
        raise ValueError('Promoted chemical labels differ from full pair population')
    typed = present & positive & (chemical == t['labels']['chemical'])
    metric.update(pair_count=len(present), promoted_correct_typed_bond_ids=sorted(t['bond_slots'][typed].tolist()),
                  confirmed_false_pair_row_ids=(present & known & ~positive).nonzero().flatten().tolist(),
                  unknown_present_row_ids=(present & ~known).nonzero().flatten().tolist())
    return metric
