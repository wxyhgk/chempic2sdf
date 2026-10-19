"""Small cached-readout fitting primitives; no file access or model loading."""

from hashlib import sha256

import torch

from .admission import require_training_admission
from .interface_diagnostic import tensor_tree_digest
from .pair_presence_stage import presence_loss
from .runtime import StepLimit

PROTOCOL = {'device': 'cpu', 'wall_seconds': 300, 'maximum_steps': 200,
    'optimizer': 'Adam', 'learning_rate': .01, 'weight_decay': 0., 'gradient_clip': 1.,
    'batch': 'all_known_pairs_from_fixed_24_training_images', 'seed': 20260918,
    'threshold_search': False, 'automatic_retries': 0,
    'selection': 'last_step_for_diagnosis_only_initial_remains_default'}


def frozen_state_digest(model):
    digest = sha256()
    for name, value in sorted(model.state_dict().items()):
        if name.startswith('pair.presence.'):
            continue
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def fitting_tensors(caches, manifest_sha256):
    if not caches or len({r['sample_key'] for r in caches}) != len(caches):
        raise ValueError('Nonempty distinct training caches required')
    hidden, targets = [], []
    for cache in caches:
        if (cache.get('purpose') != 'training' or cache.get('training_authorized') is not True
                or cache.get('manifest_sha256') != manifest_sha256):
            raise ValueError('Validation/quarantine/unbound caches cannot train')
        h, t, known = cache['hidden'], cache['targets'], cache['known']
        if (h.ndim != 2 or t.shape != (len(h),) or t.dtype != torch.bool or known.shape != t.shape
                or known.dtype != torch.bool or any(v.device.type != 'cpu' or v.requires_grad for v in (h, t, known))
                or not bool(torch.isfinite(h).all()) or bool((t & ~known).any())):
            raise ValueError('Invalid frozen CPU features/known-pair labels')
        hidden.append(h[known])
        targets.append(t[known])
    h, t = torch.cat(hidden).detach(), torch.cat(targets).detach()
    if not len(t):
        raise ValueError('No reachable known pairs; no optimizer may be created')
    return h, t


def fit(stage, caches, *, admission, manifest_sha256, step_limit, emit):
    require_training_admission(admission)
    if admission.get('approved_manifest_sha256') != manifest_sha256:
        raise ValueError('Fit admission is not bound to the exact cached manifest')
    if (not isinstance(step_limit, StepLimit) or step_limit.consumed != 0
            or not 1 <= step_limit.maximum <= PROTOCOL['maximum_steps']):
        raise ValueError('Fresh bounded step budget required; no resume')
    features, targets = fitting_tensors(caches, manifest_sha256)
    frozen_before = frozen_state_digest(stage.components)
    cache_before = tensor_tree_digest({'hidden': features, 'targets': targets})
    parameters = stage.presence_parameters()
    if any(p.device.type != 'cpu' for p in parameters):
        raise ValueError('Presence probe is CPU only')
    optimizer = torch.optim.Adam(parameters, lr=PROTOCOL['learning_rate'], weight_decay=0.)
    with torch.no_grad():
        initial_loss = float(presence_loss(stage.components.pair.presence(features).squeeze(-1), targets)['loss'])
    losses = []
    for _ in range(step_limit.maximum):
        stage.presence_parameters()
        if any(m.training for m in stage.components.modules() if m is not stage.components.pair.presence):
            raise ValueError('Frozen component mode changed during fitting')
        optimizer.zero_grad(set_to_none=True)
        logits = stage.components.pair.presence(features).squeeze(-1)
        loss = presence_loss(logits, targets)['loss']
        loss.backward()
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters):
            raise ValueError('Invalid presence gradient; no update allowed')
        norm = torch.nn.utils.clip_grad_norm_(parameters, PROTOCOL['gradient_clip'], error_if_nonfinite=True)
        step_limit.consume()
        emit('step_reserved', step=step_limit.consumed)
        optimizer.step()
        if any(not bool(torch.isfinite(p).all()) for p in parameters):
            raise ValueError('Non-finite updated presence weights')
        losses.append(float(loss.detach()))
        emit('step', step=step_limit.consumed, loss_before_update=losses[-1], gradient_norm=float(norm))
    with torch.no_grad():
        final_loss = float(presence_loss(stage.components.pair.presence(features).squeeze(-1), targets)['loss'])
    if frozen_state_digest(stage.components) != frozen_before or tensor_tree_digest(
            {'hidden': features, 'targets': targets}) != cache_before:
        raise ValueError('Frozen model/cache changed during readout fitting')
    stage.presence_parameters()
    return {'optimizer_steps': step_limit.consumed, 'initial_loss': initial_loss, 'final_loss': final_loss,
        'known_pairs': len(targets), 'positive_pairs': int(targets.sum()), 'negative_pairs': int((~targets).sum()),
        'feature_sha256': cache_before, 'frozen_state_sha256': frozen_before,
        'trainable_parameters': sum(p.numel() for p in parameters), 'losses_before_update': losses}


def pair_counts(structure, contract, threshold):
    predicted = structure['output']['presence_logits'].sigmoid() >= threshold
    positive = contract['legacy_targets']['presence']
    known = contract['known_mask']
    return {**contract['coverage'], 'missed_reachable': int((positive & ~predicted).sum()),
        'confirmed_false_positive': int((known & ~positive & predicted).sum()),
        'unknown_present': int((~known & predicted).sum()),
        'present_wrong_type': int((positive & predicted & (
            structure['chemistry'] != contract['legacy_targets']['chemical'])).sum())}
