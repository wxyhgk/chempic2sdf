"""Cached frozen-upstream pair objective and metrics; no data/authorization I/O."""

from hashlib import sha256
import json

import torch

from chemvector.direct_atom_pair_bond import direct_atom_pair_loss
from chemvector.train_direct_atom_pair_bond import match_atom_targets
from .interface_diagnostic import tensor_tree_digest
from .pair_trial_contract import RECIPE
from .runtime import StepLimit
from .supervision_contract import pair_targets_with_unknown, select_known_pair_rows
from .validation_truth import canonical_molblock


def json_digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def copy_tree(tree, device='cpu'):
    if isinstance(tree, dict):
        return {k: copy_tree(v, device) for k, v in tree.items()}
    if not isinstance(tree, torch.Tensor):
        raise ValueError('Only tensor dictionaries may enter the frozen cache')
    if tree.is_floating_point() and (tree.dtype != torch.float32 or not bool(torch.isfinite(tree).all())):
        raise ValueError('Finite FP32 frozen features required')
    return tree.detach().to(device).clone()


def frozen_digest(model):
    return tensor_tree_digest({k: v for k, v in model.state_dict().items() if not k.startswith('pair.')})


@torch.no_grad()
def make_cache(forward, batch, config):
    """Targets are attached AFTER predicted candidates; never reused cross-device."""
    features, _, _, raw, bonds, context, selected = forward
    mapping, _ = match_atom_targets(selected, batch, config)
    contract = pair_targets_with_unknown(selected, mapping, batch)
    if selected['atom_mask'].shape[0] != 1:
        raise ValueError('The fixed protocol requires batch size one')
    target_count = int(batch['bond_mask'].sum())
    lookup = {tuple(sorted(ref)): i for i, ref in enumerate(batch['bond_atom_refs'][0, :target_count].tolist())}
    bond_slots = torch.tensor([lookup.get(tuple(sorted(pair)), -1)
        for pair in mapping[contract['sample_indices'][:, None], contract['pair_indices']].tolist()],
        dtype=torch.long, device=mapping.device)
    if not torch.equal(bond_slots >= 0, contract['legacy_targets']['presence']):
        raise ValueError('Positive pair to original bond-slot correspondence changed')
    tensors = copy_tree(dict(features=features, raw=raw, bonds=bonds, context=context,
        selected=selected, mapping=mapping, pair_indices=contract['pair_indices'],
        sample_indices=contract['sample_indices'], known=contract['known_mask'],
        labels=contract['legacy_targets'], bond_slots=bond_slots))
    signature = tensor_tree_digest({k: v for k, v in tensors.items() if k != 'features'})
    return dict(tensors=tensors, coverage=contract['coverage'],
                candidate_label_sha256=signature, feature_sha256=tensor_tree_digest({'features': tensors['features']}),
                cache_sha256=tensor_tree_digest(tensors))


def verify_cache(cache):
    if tensor_tree_digest(cache['tensors']) != cache['cache_sha256']:
        raise ValueError('Frozen cache changed')


def pair_inputs(cache, device):
    # Do not transfer scoring labels to the prediction interface.
    return cache['tensors']['features'].to(device), copy_tree(cache['tensors']['selected'], device)


def assert_stage(stage):
    params = stage.pair_parameters()
    if any(m.training for m in stage.modules()):
        raise ValueError('The entire pair-only stage must stay in eval mode')
    if any(p.grad is not None for name, p in stage.components.named_parameters() if not name.startswith('pair.')):
        raise ValueError('A frozen parameter received a gradient')
    return params


def update_once(stage, cache, optimizer, limit, *, attempt, emit):
    """A single real pair-head update; empty branches never call optimizer.step."""
    params = assert_stage(stage)
    if (not isinstance(limit, StepLimit) or limit.maximum > RECIPE['maximum_steps']
            or type(attempt) is not int or not 0 <= attempt < RECIPE['maximum_batch_attempts']):
        raise ValueError('Bounded update/attempt budget required')
    optimizer_params = [p for group in optimizer.param_groups for p in group['params']]
    if len(optimizer_params) != len(params) or {id(p) for p in optimizer_params} != {id(p) for p in params}:
        raise ValueError('Optimizer must contain exactly the pair parameters')
    verify_cache(cache)
    t = cache['tensors']
    contract = dict(known_mask=t['known'], legacy_targets=t['labels'],
                    pair_indices=t['pair_indices'], sample_indices=t['sample_indices'])
    # Fixed CPU sampler avoids device-specific RNG changes; geometry/labels
    # were nevertheless freshly built on this execution's GPU candidates.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(RECIPE['seed'] + attempt)
        chosen = select_known_pair_rows(contract, t['selected'], negative_ratio=6, minimum_negatives=48)
    optimizer.zero_grad(set_to_none=True)
    if not len(chosen):
        return dict(updated=False, skip_reason='no_known_pairs', sampled_rows=0)
    if bool((~t['known'][chosen]).any()):
        raise ValueError('Unknown row entered pair supervision')
    if set(t['labels']['presence'].nonzero().flatten().tolist()) - set(chosen.tolist()):
        raise ValueError('An original positive was omitted')
    device = params[0].device
    features, selected = pair_inputs(cache, device)
    output = stage.components.pair(features, selected['atom_features'], selected['atom_geometry'][..., :2],
        selected['atom_scores'], t['pair_indices'][chosen].to(device), t['sample_indices'][chosen].to(device))
    losses = direct_atom_pair_loss(output, t['labels']['presence'][chosen].to(device),
                                  t['labels']['chemical'][chosen].to(device))
    if any(not bool(torch.isfinite(v)) for v in losses.values()):
        raise ValueError('Non-finite pair loss; no update')
    losses['loss'].backward()
    gradients = [p.grad for p in params if p.grad is not None]
    if not gradients or any(not bool(torch.isfinite(g).all()) for g in gradients):
        raise ValueError('Non-finite or absent pair gradients; no update')
    assert_stage(stage)
    norm = torch.nn.utils.clip_grad_norm_(params, RECIPE['gradient_clip'], error_if_nonfinite=True)
    limit.consume()
    emit('step_reserved', step=limit.consumed, attempt=attempt)
    optimizer.step()
    if any(not bool(torch.isfinite(p).all()) for p in params):
        raise ValueError('Non-finite updated pair state')
    result = dict(updated=True, step=limit.consumed, sampled_rows=len(chosen),
        positive_rows=int(t['labels']['presence'][chosen].sum()),
        sampled_unknown_pairs=0, gradient_norm=float(norm),
        losses={k: float(v.detach()) for k, v in losses.items()}, sampled_row_ids=chosen.tolist())
    optimizer.zero_grad(set_to_none=True)
    return result


def score_case(cache, structure, prediction, trace, *, key, expected):
    t = cache['tensors']
    pairs = structure['pairs'].detach().cpu()
    output = {k: v.detach().cpu() for k, v in structure['output'].items()}
    if not torch.equal(pairs, t['pair_indices']):
        raise ValueError('Scoring pairs differ from frozen supervision rows')
    for k in ('presence_logits', 'chemical_logits'):
        if not bool(torch.isfinite(output[k]).all()):
            raise ValueError('Non-finite full-pair evaluation')
    present = output['presence_logits'].sigmoid() >= RECIPE['pair_threshold']
    positive, known = t['labels']['presence'], t['known']
    typed = present & positive & (output['chemical_logits'].argmax(-1) == t['labels']['chemical'])
    identity = canonical_molblock(prediction['molblock'])
    return dict(key=key, target_bonds=cache['coverage']['target_bonds'],
        reachable_positive_pairs=cache['coverage']['reachable_positive_pairs'],
        correct_typed_bond_ids=sorted(t['bond_slots'][typed].tolist()),
        confirmed_false_positives=int((present & known & ~positive).sum()),
        unknown_present=int((present & ~known).sum()),
        sdf_exact=identity is not None and identity == expected, sdf_valid=identity is not None,
        fallback=prediction['fallback'], predicted_smiles=identity, expected_smiles=expected,
        candidate_label_sha256=cache['candidate_label_sha256'],
        text_sha256=json_digest({k: trace[k] for k in ('roles', 'ocr')}))


def summarize(records):
    return {group: dict(images=len(rows),
        correct_typed_bonds=sum(len(r['correct_typed_bond_ids']) for r in rows),
        **{k: sum(int(r[k]) for r in rows) for k in ('target_bonds', 'reachable_positive_pairs',
            'confirmed_false_positives', 'unknown_present', 'sdf_exact', 'sdf_valid', 'fallback')})
        for group in ('all', 'witness', 'old_full_molecule')
        for rows in [[r for r in records if group == 'all' or r['group'] == group]]}
