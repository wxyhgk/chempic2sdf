"""Opt-in cross-image confirmed-nonedge replay; no data or job authorization.

Keep the original main-image objective and sampler. Add one independently
normalized, unit-weight negative BCE term, then clip/step ONCE. Replay neither
calls direct_atom_pair_loss nor changes its positive weighting or chemistry.
"""

from copy import deepcopy

import torch
from torch.nn import functional as F

from . import pair_cached_trial as cached
from .pair_observed_capture import CapturedUpdates, cpu_copy, tensor_bytes
from .pair_observed_update import ObservedPairUpdates
from .pair_negative_observer import paired_update_record
from .pair_trial_contract import RECIPE
from .runtime import StepLimit
from .supervision_contract import select_known_pair_rows

MAX_REPLAY_ROWS = 32
REPLAY_WEIGHT = 1.0


def bind_replay(cache, initial_logits):
    """Rank only confirmed nonedges, by initial logit then immutable row ID."""
    cached.verify_cache(cache)
    data = cache['tensors']
    if (not isinstance(cache.get('key'), str) or not cache['key']
            or initial_logits.device.type != 'cpu' or initial_logits.dtype != torch.float32
            or initial_logits.shape != data['known'].shape
            or not bool(torch.isfinite(initial_logits).all()) or initial_logits.requires_grad):
        raise ValueError('Explicit cache key and aligned finite detached CPU FP32 initial logits required')
    eligible = torch.where(data['known'] & ~data['labels']['presence'])[0].tolist()
    chosen = sorted(eligible, key=lambda r: (-float(initial_logits[r]), r))[:MAX_REPLAY_ROWS]
    if not chosen: raise ValueError('Replay donor has no confirmed nonedges')
    return dict(key=cache['key'], cache_sha256=cache['cache_sha256'],
        initial_logits_sha256=cached.tensor_tree_digest({'presence': initial_logits}),
        eligible_confirmed_nonedges=len(eligible), row_ids=chosen,
        presence_target=False, unknown_rows=0, positive_rows=0)


def donor_key(donor_rows, main_row, completed_steps):
    """Metadata-only schedule shared by preparation and actual execution."""
    if type(completed_steps) is not int or not 0 <= completed_steps < 50:
        raise ValueError('Replay selection requires a bounded real-update index')
    main_key = main_row['source'] + ':' + main_row['sample_key']
    for offset in range(len(donor_rows)):
        row = donor_rows[(completed_steps + offset) % len(donor_rows)]
        key = row['source'] + ':' + row['sample_key']
        if key != main_key and row['molecule_id'] != main_row['molecule_id']:
            return key
    raise ValueError('No different-molecule replay donor; refuse silent fallback')


class ReplayBank:
    def __init__(self, caches, initial_logits, donor_rows):
        self.entries, self.keys = {}, []
        self.rows = deepcopy(list(donor_rows))
        self.rows_sha256 = cached.json_digest(self.rows)
        molecules = set()
        for row in self.rows:
            key = row['source'] + ':' + row['sample_key']
            if (row.get('split') != 'train' or row.get('fit_domain') != 'old_full_molecule'
                    or key in self.entries or row['molecule_id'] in molecules):
                raise ValueError('Distinct admitted full-molecule training donors required')
            binding = bind_replay(caches[key], initial_logits[key])
            if binding['key'] != key: raise ValueError('Donor cache key differs from manifest')
            self.keys.append(key); molecules.add(row['molecule_id'])
            self.entries[key] = dict(cache=caches[key], initial_logits=initial_logits[key].clone(),
                                     binding=binding, molecule_id=row['molecule_id'])
        if len(self.keys) < 2: raise ValueError('At least two distinct donor molecules required')

    def select(self, main_row, completed_steps):
        if cached.json_digest(self.rows) != self.rows_sha256:
            raise ValueError('Frozen donor schedule changed')
        key = donor_key(self.rows, main_row, completed_steps)
        entry = self.entries[key]
        if (entry['binding']['key'] != key
                or bind_replay(entry['cache'], entry['initial_logits']) != entry['binding']):
            raise ValueError('Frozen replay bank changed')
        return entry


def replay_loss(logits):
    if logits.ndim != 1 or not 0 < len(logits) <= MAX_REPLAY_ROWS or not bool(torch.isfinite(logits).all()):
        raise ValueError('Finite nonempty bounded nonedge logits required')
    return F.softplus(logits.float()).mean()  # BCE(logit, 0), no pos_weight coupling.


def _forward_rows(stage, cache, chosen):
    device = next(stage.components.pair.parameters()).device
    features, selected = cached.pair_inputs(cache, device)
    t = cache['tensors']
    return stage.components.pair(features, selected['atom_features'], selected['atom_geometry'][..., :2],
        selected['atom_scores'], t['pair_indices'][chosen].to(device), t['sample_indices'][chosen].to(device))


def update_once(stage, main, optimizer, limit, *, attempt, emit, replay_entry=None, weight=REPLAY_WEIGHT):
    """The sole experimental change is + weight * mean negative BCE.

    Weight=0 is a synthetic parity control, not a second real training arm.
    Empty main supervision never turns into a replay-only update.
    """
    params = cached.assert_stage(stage)
    if (not isinstance(limit, StepLimit) or limit.maximum > 50 or limit.consumed >= limit.maximum
            or type(attempt) is not int or not 0 <= attempt < 58
            or type(weight) not in (int, float) or weight not in (0., REPLAY_WEIGHT)):
        raise ValueError('Exact bounded replay recipe or zero-weight synthetic control required')
    actual = [p for g in optimizer.param_groups for p in g['params']]
    if list(map(id, actual)) != list(map(id, params)):
        raise ValueError('Exactly the original ordered pair parameters required')
    cached.verify_cache(main)
    t = main['tensors']
    contract = dict(known_mask=t['known'], legacy_targets=t['labels'],
                    pair_indices=t['pair_indices'], sample_indices=t['sample_indices'])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(RECIPE['seed'] + attempt)
        chosen = select_known_pair_rows(contract, t['selected'], negative_ratio=6, minimum_negatives=48)
    optimizer.zero_grad(set_to_none=True)
    if not len(chosen): return dict(updated=False, skip_reason='no_known_pairs', sampled_rows=0)
    if bool((~t['known'][chosen]).any()) or set(t['labels']['presence'].nonzero().flatten().tolist()) - set(chosen.tolist()):
        raise ValueError('Original main-image supervision changed')
    replay_rows = None
    if weight:
        if replay_entry is None: raise ValueError('A fixed different-image donor is mandatory')
        donor, binding = replay_entry['cache'], replay_entry['binding']
        if donor['key'] == main['key'] or bind_replay(donor, replay_entry['initial_logits']) != binding:
            raise ValueError('Self replay or changed donor rejected')
        replay_rows = torch.tensor(binding['row_ids'], dtype=torch.long)
    output = _forward_rows(stage, main, chosen)
    device = params[0].device
    losses = cached.direct_atom_pair_loss(output, t['labels']['presence'][chosen].to(device),
                                          t['labels']['chemical'][chosen].to(device))
    total, penalty = losses['loss'], None
    if weight:
        replay_output = _forward_rows(stage, donor, replay_rows)
        penalty = replay_loss(replay_output['presence_logits'])
        total = total + weight * penalty
    if any(not bool(torch.isfinite(v)) for v in [*losses.values(), total]):
        raise ValueError('Nonfinite loss; no update')
    total.backward()
    gradients = [p.grad for p in params if p.grad is not None]
    if not gradients or any(not bool(torch.isfinite(g).all()) for g in gradients):
        raise ValueError('Nonfinite or absent gradient')
    cached.assert_stage(stage)
    norm = torch.nn.utils.clip_grad_norm_(params, RECIPE['gradient_clip'], error_if_nonfinite=True)
    cached.verify_cache(main)
    if weight: cached.verify_cache(donor)
    limit.consume(); emit('step_reserved', step=limit.consumed, attempt=attempt)
    optimizer.step()
    if any(not bool(torch.isfinite(p).all()) for p in params): raise ValueError('Nonfinite updated pair state')
    result = dict(updated=True, step=limit.consumed, sampled_rows=len(chosen),
        positive_rows=int(t['labels']['presence'][chosen].sum()), sampled_unknown_pairs=0,
        gradient_norm=float(norm), losses={k: float(v.detach()) for k, v in losses.items()},
        sampled_row_ids=chosen.tolist(), objective_loss=float(total.detach()),
        replay=None if not weight else dict(key=donor['key'], binding=deepcopy(binding),
            weight=weight, negative_bce=float(penalty.detach()), rows=len(replay_rows),
            chemical_loss_added=False, actual_optimizer_steps=1))
    optimizer.zero_grad(set_to_none=True)
    return result


class ObservedReplayUpdates(ObservedPairUpdates):
    """Explicit new objective adapter; old observer and launcher stay immutable."""
    def __init__(self, *args, bank, rows, **kwargs):
        super().__init__(*args, **kwargs)
        self.bank, self.rows = bank, deepcopy(list(rows))

    def _update(self, cache, *, attempt):
        if (type(attempt) is not int or attempt != self._attempt or attempt >= len(self.rows)
                or attempt >= self._maximum_attempts or self._limit.maximum != self._maximum_steps
                or self._limit.consumed != self._steps or self._steps >= self._maximum_steps
                or self._pair_state() != self._state):
            raise ValueError('Observed replay sequence/state/budget changed')
        row = self.rows[attempt]; key = row['source'] + ':' + row['sample_key']
        if cache['key'] != key: raise ValueError('Main manifest order changed')
        cached.assert_stage(self._stage); cached.verify_cache(cache)
        for _, binding in self._probes:
            if key == binding['key'] and cache['cache_sha256'] != binding['cache_sha256']:
                raise ValueError('Own probe cache mismatch')
        has_supervision = bool(cache['tensors']['known'].any())
        entry = self.bank.select(row, self._steps) if has_supervision else None
        before = self._observe() if has_supervision else []
        self._record('observed_attempt_before', attempt=attempt, updated_sample_key=key,
            consumed_steps=self._steps, has_supervision=has_supervision, probes=before,
            replay_binding=None if entry is None else entry['binding'])
        result = update_once(self._stage, cache, self._optimizer, self._limit,
                             attempt=attempt, emit=self._record, replay_entry=entry)
        expected = self._steps + int(has_supervision)
        if result['updated'] != has_supervision or self._limit.consumed != expected:
            raise ValueError('Replay update result differs from bracketed step')
        self._state = self._pair_state(); records = []
        if result['updated']:
            for a, b in zip(before, self._observe()):
                record = paired_update_record(a, b, step=result['step'], updated_sample_key=key)
                record.update(attempt=attempt,
                    probe_row_sampled_in_update=record['own_sample_update'] and a['binding']['row_id'] in result['sampled_row_ids'],
                    replay_image_is_probe=entry['cache']['key'] == a['binding']['key'],
                    probe_row_in_replay=(entry['cache']['key'] == a['binding']['key']
                                         and a['binding']['row_id'] in entry['binding']['row_ids']))
                records.append(record)
        cached.verify_cache(cache)
        self._record('observed_attempt_complete', attempt=attempt, updated_sample_key=key, result=result, probes=records)
        self._steps, self._attempt = expected, attempt + 1
        return dict(update=result, probes=records)


class CapturedReplayUpdates(CapturedUpdates):
    def __init__(self, stage, optimizer, limit, probes, rows, *, bank, emit):
        order = [r['source'] + ':' + r['sample_key'] for r in rows]
        super().__init__(stage, optimizer, limit, probes, order, emit=emit)
        self.bank = bank
        self.adapter = ObservedReplayUpdates(stage, optimizer, limit, self.probes,
            emit=self._emit, maximum_attempts=len(order), bank=bank, rows=rows)

    def update(self, cache, *, attempt):
        try:
            result = super().update(cache, attempt=attempt)
            step = result['update'].get('step')
            if step in self.events:
                binding = result['update']['replay']['binding']
                entry = self.bank.entries[binding['key']]
                self.events[step].update(replay_cache=entry['cache'], replay_binding=cpu_copy(binding),
                                        replay_initial_logits=cpu_copy(entry['initial_logits']))
                if tensor_bytes(self.events) > 1024**3: raise ValueError('Replay snapshot tensor cap')
            return result
        except BaseException:
            self.failed = True
            raise
