"""Opt-in telemetry adapter; not a launcher, new objective, or authorization.

Use only on fresh, authorized caches. ``emit`` must synchronously persist its
event or raise. A failed receipt/update poisons this instance: parameters may
already have changed, so retry/resume or promotion is not safe. CPU unit tests
do not establish GPU numerical parity or authorize a real invocation.
"""

from copy import deepcopy

import torch

from . import pair_cached_trial as cached
from . import pair_negative_observer as observer
from .interface_diagnostic import tensor_tree_digest
from .runtime import StepLimit


class ObservedPairUpdates:
    """Bracket the unchanged objective with two fixed negative-pair probes."""

    def __init__(self, stage, optimizer, limit, probes, *, emit, maximum_attempts=58):
        if (not isinstance(limit, StepLimit) or not 1 <= limit.maximum <= 50
                or limit.consumed != 0 or type(maximum_attempts) is not int
                or not 1 <= maximum_attempts <= 58 or not callable(emit)):
            raise ValueError('Fresh <=50-step, <=58-attempt bounded observer required')
        cached.assert_stage(stage)
        probes = list(probes)
        if len(probes) != 2:
            raise ValueError('Exactly two fixed negative probes required')
        self._probes = [(cache, observer.bind_negative_probe(cache, pair)) for cache, pair in probes]
        keys = [binding['key'] for _, binding in self._probes]
        if any(not isinstance(k, str) or not k for k in keys) or len(set(keys)) != 2:
            raise ValueError('Two distinct explicit sample keys required')
        self._stage, self._optimizer, self._limit = stage, optimizer, limit
        self._emit, self._maximum_attempts = emit, maximum_attempts
        self._maximum_steps = limit.maximum
        self._attempt, self._steps, self._failed = 0, 0, False
        self._state = self._pair_state()

    def _pair_state(self):
        return tensor_tree_digest(self._stage.components.pair.state_dict())

    def _observe(self):
        # Even a stochastic custom head must not advance the update's RNG.
        devices = sorted({p.device.index for p in self._stage.components.pair.parameters()
                          if p.device.type == 'cuda'})
        with torch.random.fork_rng(devices=devices):
            result = [observer.observe(self._stage.components.pair, cache, binding)
                      for cache, binding in self._probes]
        if self._pair_state() != self._state:
            raise ValueError('Probe forward mutated pair parameters or buffers')
        for cache, binding in self._probes:
            if observer.bind_negative_probe(cache, binding['target_pair']) != binding:
                raise ValueError('Probe forward mutated cache or binding')
        return result

    def _record(self, event, **values):
        # A logger cannot mutate a live binding or the returned result.
        self._emit(event, **deepcopy(values))

    def update(self, cache, *, attempt):
        if self._failed:
            raise RuntimeError('Incomplete observed update: no retry or resume')
        try:
            return self._update(cache, attempt=attempt)
        except Exception:
            self._failed = True
            raise

    def _update(self, cache, *, attempt):
        if (type(attempt) is not int or attempt != self._attempt
                or attempt >= self._maximum_attempts
                or self._limit.maximum != self._maximum_steps
                or self._limit.consumed != self._steps
                or self._steps >= self._maximum_steps
                or self._pair_state() != self._state):
            raise ValueError('Attempt/step/state differs from observed sequence or budget')
        cached.assert_stage(self._stage)
        cached.verify_cache(cache)
        key = cache.get('key')
        if not isinstance(key, str) or not key:
            raise ValueError('Explicit update sample key required')
        for probe_cache, binding in self._probes:
            if key == binding['key'] and cache['cache_sha256'] != binding['cache_sha256']:
                raise ValueError('Own-sample update must use the identical fresh probe cache')
        has_supervision = bool(cache['tensors']['known'].any())
        before = self._observe() if has_supervision else []
        self._record('observed_attempt_before', attempt=attempt, updated_sample_key=key,
                     consumed_steps=self._steps, probes=before,
                     has_supervision=has_supervision)
        result = cached.update_once(self._stage, cache, self._optimizer, self._limit,
                                   attempt=attempt, emit=self._record)
        expected_steps = self._steps + int(has_supervision)
        if (result['updated'] != has_supervision or self._limit.consumed != expected_steps
                or (has_supervision and result['step'] != expected_steps)):
            raise ValueError('Update result differs from the bracketed step')
        self._state = self._pair_state()
        records = []
        if result['updated']:
            for prior, after in zip(before, self._observe()):
                record = observer.paired_update_record(prior, after,
                    step=result['step'], updated_sample_key=key)
                record['attempt'] = attempt
                record['probe_row_sampled_in_update'] = (
                    record['own_sample_update']
                    and prior['binding']['row_id'] in result['sampled_row_ids'])
                records.append(record)
        cached.verify_cache(cache)
        self._record('observed_attempt_complete', attempt=attempt,
                     updated_sample_key=key, result=result, probes=records)
        self._steps, self._attempt = expected_steps, attempt + 1
        return dict(update=result, probes=records)
