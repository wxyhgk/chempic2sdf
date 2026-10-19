"""Saved-output CPU postcheck only; no model construction or optimizer step."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sys

import torch

from chemvector.training.curriculum import runtime
from chemvector.training.curriculum.interface_diagnostic import write_json
from chemvector.training.curriculum.validation_truth import canonical_molblock

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'outputs/experiments/chemvector-pair-observed-trial-20260922'
OLD = ROOT / 'outputs/experiments/chemvector-pair-only-matched-trial-20260921'
OUTPUT = ROOT / 'outputs/audits/chemvector-pair-observed-postcheck-20260922'


def digest(path): return sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_bytes())


def worker():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('CPU saved-artifact postcheck required')
    torch.set_num_threads(2)
    report, receipt = read(RUN / 'report.json'), read(RUN / 'runtime.json')
    assert receipt['status'] == 'completed' and receipt['elapsed_seconds'] < 600
    assert report['optimizer_steps'] == report['actual_final_step'] == 50
    assert report['full_50_step_diagnostic'] and not report['deployment_authorized']
    assert report['frozen_state_before'] == report['frozen_state_after']
    assert report['ocr_state_before'] == report['ocr_state_after']
    checked = Counter()
    for kind, bindings in [('input', report['input_sha256']), ('implementation', report['implementation_sha256']),
                           ('artifact', report['artifact_sha256'])]:
        for path, expected in bindings.items():
            resolved = RUN / path if kind == 'artifact' else ROOT / path
            assert digest(resolved) == expected, str(resolved)
            checked[kind] += 1
    assert digest(OLD / 'report.json') == 'd12d93e5e012f7566f4cc3123abf411b67f49c44ecb898404034105b3c4603eb'
    historical = read(OLD / 'report.json')
    assert report['pair_state_before'] == historical['pair_state_before']
    assert report['pair_state_after'] == historical['pair_state_after']
    plan = read(RUN / 'plan.json'); rows = plan['samples']
    lookup = {r['source'] + ':' + r['sample_key']: r for r in rows}
    events = [json.loads(line) for line in (RUN / 'events.jsonl').read_bytes().splitlines()]
    attempts = [r for r in events if r['event'] == 'batch_complete']
    assert len(attempts) == 54 and [r['attempt'] for r in attempts] == list(range(54))
    assert [r['key'] for r in attempts] == list(lookup)[:54]
    assert [r['step'] for r in attempts if r['updated']] == list(range(1, 51))
    assert [r['step'] for r in events if r['event'] == 'step_reserved'] == list(range(1, 51))
    assert sum(r['event'] == 'observed_attempt_before' for r in events) == 54
    assert sum(r['event'] == 'observed_attempt_complete' for r in events) == 54
    old_events = [json.loads(line) for line in (OLD / 'events.jsonl').read_bytes().splitlines()]
    old_attempts = [r for r in old_events if r['event'] == 'batch_complete']
    fields = ('key', 'attempt', 'updated', 'step', 'sampled_row_ids', 'losses', 'gradient_norm')
    assert [{k: r.get(k) for k in fields} for r in attempts] == [{k: r.get(k) for k in fields} for r in old_attempts]
    comparisons = 0
    for label, step in [('baseline', 0), ('final', 50)]:
        records = read(RUN / (label + '.json'))['records']
        assert {r['key'] for r in records} == set(lookup)
        for row in rows:
            new = read(RUN / ('check-' + label + '--' + row['sample_key'] + '.json'))
            old = read(OLD / (f'check-{step:06d}--' + row['sample_key'] + '.json'))
            assert new == old, row['sample_key']  # All logits, trace, MolBlock and metric fields.
            metric = new['metrics']; identity = canonical_molblock(new['prediction']['molblock'])
            assert metric['sdf_valid'] == (identity is not None)
            assert metric['sdf_exact'] == (identity is not None and identity == metric['expected_smiles'])
            comparisons += 1
        for group in ('all', 'witness', 'old_full_molecule'):
            subset = [r for r in records if group == 'all' or r['group'] == group]
            summary = report[label][group]
            assert summary['images'] == len(subset)
            assert summary['correct_typed_bonds'] == sum(len(r['correct_typed_bond_ids']) for r in subset)
            for field in ('target_bonds', 'reachable_positive_pairs', 'confirmed_false_positives',
                          'unknown_present', 'sdf_exact', 'sdf_valid', 'fallback'):
                assert summary[field] == sum(r[field] for r in subset)
    assert report['decision'] == read(OLD / 'decision-000050.json') or (
        report['decision'] == {k: v for k, v in read(OLD / 'decision-000050.json').items() if k != 'step'})
    trajectory = read(RUN / 'trajectories.json')
    complete_events = [e for e in events if e['event'] == 'observed_attempt_complete']
    assert trajectory['updates'] == [dict(update=e['result'], probes=e['probes']) for e in complete_events]
    derived, expected_slots = {}, {}
    for key in trajectory['summary']:
        track = [p for r in trajectory['updates'] for p in r['probes'] if p['probe']['key'] == key]
        assert len(track) == 50
        for a, b in zip(track, track[1:]): assert a['after']['presence_logit'] == b['before']['presence_logit']
        own = [r for r in track if r['own_sample_update']]
        assert len(own) == 1 and own[0]['updated_sample_key'] == key and own[0]['probe_row_sampled_in_update']
        for record in track:
            assert record['presence_logit_delta'] == record['after']['presence_logit'] - record['before']['presence_logit']
        maximum = max(track, key=lambda p: (p['presence_logit_delta'], -p['step']))
        expected_slots['own:' + key] = own[0]['step']
        expected_slots['maxrise:' + key] = maximum['step']
        cutoff = math.log(.85 / .15)
        summary = trajectory['summary'][key]
        assert summary['crossing_steps'] == [r['step'] for r in track if r['before']['presence_logit'] < cutoff <= r['after']['presence_logit']]
        assert summary['own_update_logit_change'] == own[0]['presence_logit_delta'] < 0
        assert summary['other_updates_logit_change'] == sum(r['presence_logit_delta'] for r in track if not r['own_sample_update'])
        derived[key] = dict(initial_probability=1 / (1 + math.exp(-summary['initial_logit'])),
            final_probability=1 / (1 + math.exp(-summary['final_logit'])),
            own_step=own[0]['step'], own_logit_change=own[0]['presence_logit_delta'],
            other_updates_logit_change=summary['other_updates_logit_change'],
            crossing_steps=summary['crossing_steps'], largest_rise_step=maximum['step'],
            by_domain_logit_change={group: sum(r['presence_logit_delta'] for r in track
                if lookup[r['updated_sample_key']]['fit_domain'] == group) for group in ('witness', 'old_full_molecule')})
    assert expected_slots == report['selected_event_slots']
    snapshot = read(RUN / 'snapshot-manifest.json')
    assert snapshot['slots'] == expected_slots and len(snapshot['events']) == 3 <= 4
    assert snapshot['snapshot_tensor_bytes'] < 1024**3
    adam_errors = {}
    for step, record in snapshot['events'].items():
        saved = torch.load(RUN / record['file'], map_location='cpu', weights_only=True)
        e = saved['event']; assert e['step'] == int(step)
        assert e['record'] == next(r for r in trajectory['updates'] if r['update'].get('step') == int(step))
        state = e['optimizer_before']; group = state['param_groups'][0]; b1, b2 = group['betas']
        max_error = 0.
        for name, index in zip(e['parameter_names'], group['params'], strict=True):
            g, before = e['gradients_after_clip'][name], e['pair_before'][name]
            if g is None:
                expected = before
            else:
                st = state['state'].get(index, {}); t = float(st.get('step', 0)) + 1
                m = st.get('exp_avg', torch.zeros_like(g)).lerp(g, 1 - b1)
                v = st.get('exp_avg_sq', torch.zeros_like(g)) * b2 + g.square() * (1 - b2)
                expected = before - group['lr'] * (m / (1 - b1**t)) / ((v / (1 - b2**t)).sqrt() + group['eps'])
            error = float((expected - e['pair_after'][name]).abs().max()); max_error = max(error, max_error)
            assert error < 1e-6
        adam_errors[step] = max_error
        del e, saved
    result = dict(schema_version='pair-observed-postcheck-0.1', checks_passed=True,
        report_sha256=digest(RUN / 'report.json'), runtime_sha256=digest(RUN / 'runtime.json'),
        hashes_checked=dict(checked), historical_saved_cases_exactly_reproduced=comparisons,
        historical_update_rows_losses_and_gradient_norms_identical=True,
        historical_final_pair_state_identical=True, confirmed_training_steps=50,
        optimizer_steps_performed_by_postcheck=0, model_forwards=0, model_instantiations=0,
        cpu_tensor_snapshot_reads=3, gpu_used=False, new_identity_queries=0,
        derived_trajectories=derived, adam_arithmetic_max_errors=adam_errors,
        snapshot_events=3, assessment='cross_sample_increase_with_small_own_negative_correction; no_loss_component_causality_claim',
        source_sha256=digest(__file__))
    write_json(OUTPUT / 'report.json', result)
    print(json.dumps(result, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true'); args = parser.parse_args(argv)
    if args.worker: worker(); return 0
    receipt = runtime.run_budgeted([sys.executable, '-m', 'tools.audit_pair_observed_result', '--worker'],
                                  output=OUTPUT, budget=runtime.ResourceBudget(120))
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__': raise SystemExit(main())
