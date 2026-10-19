"""Bounded diagnostic snapshots around the unchanged observed pair update."""

from copy import deepcopy
from hashlib import sha256
import json
import math
import os

import torch

from . import pair_cached_trial as cached
from .pair_negative_observer import observe
from .pair_observed_update import ObservedPairUpdates

ATOL, RTOL = 1e-4, 1e-5


def cpu_copy(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list): return [cpu_copy(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu_copy(v) for v in value)
    return deepcopy(value)


def mixed_digest(value):
    digest = sha256()
    def visit(v):
        if isinstance(v, torch.Tensor):
            digest.update(str((v.dtype, tuple(v.shape))).encode())
            digest.update(v.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(v, dict):
            for k in sorted(v, key=lambda k: (type(k).__name__, str(k))):
                visit(k); visit(v[k])
        elif isinstance(v, (tuple, list)):
            digest.update(type(v).__name__.encode())
            for item in v: visit(item)
        else: digest.update((type(v).__name__ + ':' + repr(v)).encode())
    visit(value)
    return digest.hexdigest()


def tensor_bytes(value):
    if isinstance(value, torch.Tensor): return value.numel() * value.element_size()
    if isinstance(value, dict): return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)): return sum(tensor_bytes(v) for v in value)
    return 0


class DurableEvents:
    def __init__(self, stream, elapsed):
        self.stream, self.elapsed, self.failed = stream, elapsed, False

    def __call__(self, event, **values):
        if self.failed: raise RuntimeError('Failed event sink cannot be reused')
        try:
            line = json.dumps(dict(event=event, elapsed_seconds=self.elapsed(), **values),
                              allow_nan=False)
            self.stream.write(line + '\n'); self.stream.flush(); os.fsync(self.stream.fileno())
        except BaseException:
            self.failed = True
            raise


def check_probe_parity(single, full, row):
    a = torch.tensor([single['presence_logit'], *single['chemical_logits']], dtype=torch.float32)
    b = torch.cat((full['presence_logits'][row:row + 1].detach().cpu(),
                   full['chemical_logits'][row].detach().cpu()))
    if a.shape != b.shape or not torch.isfinite(b).all() or not torch.allclose(a, b, atol=ATOL, rtol=RTOL):
        raise ValueError('Single-row / full-pair diagnostic parity failed')
    return dict(max_absolute_difference=float((a - b).abs().max()), atol=ATOL, rtol=RTOL)


class CapturedUpdates:
    """One ordered pass, <=4 retained events, no state-changing extra forward.

    Snapshot slots: each probe's own update and largest positive logit change.
    Replaced max-rise candidates are discarded from RAM only, never overwrite
    an artifact. All scalar events are durably retained. Snapshots are written
    only after a completed run; a killed process is explicitly incomplete.
    """

    def __init__(self, stage, optimizer, limit, probes, order, *, emit):
        if len(order) > 58 or len(order) != len(set(order)) or not order or optimizer.state:
            raise ValueError('One distinct fixed pass and fresh optimizer required')
        self.stage, self.optimizer, self.limit = stage, optimizer, limit
        self.order, self.emit = list(order), emit
        self.probes = list(probes)
        if any(cache['key'] not in self.order for cache, _ in self.probes):
            raise ValueError('Every probe must be in the fixed pass')
        self.pending, self.events, self.slots, self.best = None, {}, {}, {}
        self.attempt, self.failed = 0, False
        self.names = [n for n, _ in stage.components.pair.named_parameters()]
        actual = [p for g in optimizer.param_groups for p in g['params']]
        expected = list(stage.components.pair.parameters())
        if list(map(id, actual)) != list(map(id, expected)):
            raise ValueError('Optimizer parameter order must match named pair parameters')
        self.optimizer_digest = mixed_digest(optimizer.state_dict())
        self.adapter = ObservedPairUpdates(stage, optimizer, limit, probes,
            emit=self._emit, maximum_attempts=len(order))

    def _emit(self, event, **values):
        self.emit(event, **values)
        if event == 'step_reserved':
            self.pending['gradients_after_clip'] = {
                n: None if p.grad is None else cpu_copy(p.grad)
                for n, p in self.stage.components.pair.named_parameters()}

    def update(self, cache, *, attempt):
        if self.failed: raise RuntimeError('Incomplete capture; no retry')
        try:
            if (type(attempt) is not int or attempt != self.attempt or attempt >= len(self.order)
                    or cache['key'] != self.order[attempt]
                    or mixed_digest(self.optimizer.state_dict()) != self.optimizer_digest):
                raise ValueError('Fixed order or optimizer state changed outside observed update')
            cached.verify_cache(cache)
            if bool(cache['tensors']['known'].any()):
                device = next(self.stage.components.pair.parameters()).device
                self.pending = dict(pair_before=cpu_copy(self.stage.components.pair.state_dict()),
                    optimizer_before=cpu_copy(self.optimizer.state_dict()), parameter_names=list(self.names),
                    cpu_rng=torch.random.get_rng_state().clone(),
                    cuda_rng=torch.cuda.get_rng_state(device).cpu().clone() if device.type == 'cuda' else None,
                    execution_device=str(device), cache=cache, attempt=attempt)
            result = self.adapter.update(cache, attempt=attempt)
            if result['update']['updated']:
                step = result['update']['step']
                selected = []
                for probe in result['probes']:
                    key, delta = probe['probe']['key'], probe['presence_logit_delta']
                    if probe['own_sample_update']: selected.append('own:' + key)
                    if delta > ATOL and delta > self.best.get(key, ATOL):
                        self.best[key] = delta; selected.append('maxrise:' + key)
                if selected:
                    self.pending.update(pair_after=cpu_copy(self.stage.components.pair.state_dict()),
                        record=deepcopy(result), step=step,
                        clip_coefficient=min(1., 1. / (result['update']['gradient_norm'] + 1e-6)))
                    self.events[step] = self.pending
                    for slot in selected: self.slots[slot] = step
                    self.events = {s: e for s, e in self.events.items() if s in self.slots.values()}
                    if len(self.events) > 4 or tensor_bytes(self.events) > 1024**3:
                        raise ValueError('Diagnostic snapshot memory bound exceeded')
            self.pending = None
            self.optimizer_digest = mixed_digest(self.optimizer.state_dict())
            self.attempt += 1
            return result
        except BaseException:
            self.failed = True
            raise


def summarize_trajectories(records):
    grouped = {}
    for result in records:
        for probe in result['probes']:
            key = probe['probe']['key']
            grouped.setdefault(key, []).append(probe)
    report = {}
    cutoff = math.log(.85 / .15)
    for key, rows in grouped.items():
        for left, right in zip(rows, rows[1:]):
            if abs(left['after']['presence_logit'] - right['before']['presence_logit']) > ATOL:
                raise ValueError('Probe trajectory is discontinuous')
        own = [r for r in rows if r['own_sample_update']]
        total = rows[-1]['after']['presence_logit'] - rows[0]['before']['presence_logit']
        deltas = sum(r['presence_logit_delta'] for r in rows)
        if abs(total - deltas) > ATOL: raise ValueError('Trajectory delta sum mismatch')
        maximum = max(rows, key=lambda r: (r['presence_logit_delta'], -r['step']))
        report[key] = dict(initial_logit=rows[0]['before']['presence_logit'],
            final_logit=rows[-1]['after']['presence_logit'], total_logit_change=total,
            own_update_logit_change=sum(r['presence_logit_delta'] for r in own),
            other_updates_logit_change=sum(r['presence_logit_delta'] for r in rows if not r['own_sample_update']),
            own_updates=own, largest_rise=maximum,
            crossing_steps=[r['step'] for r in rows if r['before']['presence_logit'] < cutoff <= r['after']['presence_logit']],
            positive_change_steps=sum(r['presence_logit_delta'] > ATOL for r in rows),
            negative_change_steps=sum(r['presence_logit_delta'] < -ATOL for r in rows),
            optimizer_updates=len(rows), loss_component_causality_proven=False)
    return report


def termination_status(steps):
    if type(steps) is not int or not 0 <= steps <= 50: raise ValueError('Invalid step count')
    return dict(actual_final_step=steps, full_50_step_diagnostic=steps == 50,
                status='completed_50' if steps == 50 else 'incomplete_attempt_cap',
                needs_actual_final_evaluation=True, deployment_authorized=False)
