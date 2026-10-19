"""CPU saved-result postcheck; no model construction, forward or optimizer."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sys

import torch

from chemvector.training.curriculum import runtime, pair_replay_trial as launch
from chemvector.training.curriculum import pair_cached_trial as cached, pair_trial_contract as rules
from chemvector.training.curriculum.interface_diagnostic import write_json, tensor_tree_digest
from chemvector.training.curriculum.pair_observed_capture import tensor_bytes
from chemvector.training.curriculum.validation_truth import canonical_molblock

ROOT, RUN, CONTROL = launch.ROOT, launch.OUTPUT, launch.CONTROL
OUTPUT = ROOT / 'outputs/audits/chemvector-pair-replay-postcheck-20260922'


def digest(path): return sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_bytes())


def verify_saved_binding(cache, logits, binding):
    cached.verify_cache(cache)
    assert cache['key'] == binding['key'] and cache['cache_sha256'] == binding['cache_sha256']
    assert tensor_tree_digest({'presence': logits}) == binding['initial_logits_sha256']
    t = cache['tensors']; eligible = torch.where(t['known'] & ~t['labels']['presence'])[0].tolist()
    ranked = sorted(eligible, key=lambda r: (-float(logits[r]), r))[:32]
    assert binding['row_ids'] == ranked and len(eligible) == binding['eligible_confirmed_nonedges']
    assert binding['presence_target'] is False and binding['unknown_rows'] == binding['positive_rows'] == 0


def adam_error(event):
    state = event['optimizer_before']; assert len(state['param_groups']) == 1
    group = state['param_groups'][0]; b1, b2 = group['betas']; maximum = 0.
    assert group['weight_decay'] == 0 and not group['amsgrad'] and not group['maximize']
    for name, index in zip(event['parameter_names'], group['params'], strict=True):
        gradient, before = event['gradients_after_clip'][name], event['pair_before'][name]
        if gradient is None:
            expected = before
        else:
            previous = state['state'].get(index, {}); step = float(previous.get('step', 0)) + 1
            m = previous.get('exp_avg', torch.zeros_like(gradient)).lerp(gradient, 1 - b1)
            v = previous.get('exp_avg_sq', torch.zeros_like(gradient)) * b2 + gradient.square() * (1 - b2)
            expected = before - group['lr'] * (m / (1 - b1**step)) / ((v / (1 - b2**step)).sqrt() + group['eps'])
        error = float((expected - event['pair_after'][name]).abs().max())
        assert error < 1e-6
        maximum = max(maximum, error)
    return maximum


def worker():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('CPU-only saved-result verification required')
    torch.set_num_threads(2)
    report_sha = digest(RUN / 'report.json'); source_sha = digest(__file__)
    report, receipt = read(RUN / 'report.json'), read(RUN / 'runtime.json')
    manifest, _, auth, control = launch.authorize()  # Read-only, no reservation.
    assert receipt['status'] == 'completed' and receipt['elapsed_seconds'] < 600
    assert receipt['training'] and receipt['gpu_devices'] == ['0'] and receipt['automatic_retries'] == 0
    assert report['optimizer_steps'] == report['actual_final_step'] == 50
    assert report['complete'] and not report['default_changed'] and not report['deployment_authorized']
    assert report['frozen_state_before'] == report['frozen_state_after']
    assert report['ocr_state_before'] == report['ocr_state_after']
    assert report['authorization_sha256'] == digest(launch.AUTHORIZATION)
    assert read(launch.ATTEMPT)['authorization_sha256'] == report['authorization_sha256']
    checked = Counter()
    for kind, bindings in [('input', report['input_sha256']), ('control_input', report['control_input_sha256']),
                           ('implementation', report['implementation_sha256']), ('artifact', report['artifact_sha256'])]:
        for path, expected in bindings.items():
            resolved = RUN / path if kind == 'artifact' else ROOT / path
            assert digest(resolved) == expected, str(resolved)
            checked[kind] += 1
    assert digest(CONTROL / 'report.json') == launch.CONTROL_SHA
    for name in ('final.json', 'trajectories.json'):
        assert digest(CONTROL / name) == control['artifact_sha256'][name]
    plan, bank = read(RUN / 'plan.json'), read(RUN / 'replay-bank.json')
    rows = manifest['training_samples']; assert plan['samples'] == rows
    assert plan['environment'] == read(CONTROL / 'plan.json')['environment']
    assert read(RUN / 'cache-manifest.json') == read(CONTROL / 'cache-manifest.json')
    lookup = {r['source'] + ':' + r['sample_key']: r for r in rows}
    donors = [k for k, r in lookup.items() if r['fit_domain'] == 'old_full_molecule']
    assert donors == bank['donor_keys'] and len(donors) == 32
    baseline, final = [read(RUN / (label + '.json'))['records'] for label in ('baseline', 'final')]
    old_final = read(CONTROL / 'final.json')['records']
    cases = {}
    for label, records in [('baseline', baseline), ('final', final)]:
        assert [r['key'] for r in records] == list(lookup)
        for row, metric in zip(rows, records, strict=True):
            name = 'check-' + label + '--' + row['sample_key'] + '.json'
            case = read(RUN / name); assert case['metrics'] == metric
            cases[label, metric['key']] = case
            if label == 'baseline': assert case == read(CONTROL / name)
            identity = canonical_molblock(case['prediction']['molblock'])
            assert metric['sdf_valid'] == (identity is not None)
            assert metric['sdf_exact'] == (identity is not None and identity == metric['expected_smiles'])
            assert metric['fallback'] == case['prediction']['fallback']
        assert cached.summarize(records) == report[label]
    decision = rules.checkpoint_decision(baseline, final, step=50)
    decision.update(bounded_repair_stopped=True, further_training_authorized=False, deployment_authorized=False)
    assert decision == report['decision'] == read(RUN / 'decision.json')
    for key, entry in bank['entries'].items():
        b = entry['binding']; logits = torch.tensor(cases['baseline', key]['presence_logits'], dtype=torch.float32)
        assert b['key'] == key and entry['molecule_id'] == lookup[key]['molecule_id']
        assert b['initial_logits_sha256'] == tensor_tree_digest({'presence': logits})
        assert len(b['row_ids']) == len(set(b['row_ids'])) == 32
        assert all(type(i) is int and 0 <= i < len(logits) for i in b['row_ids'])
        assert b['row_ids'] == sorted(b['row_ids'], key=lambda i: (-float(logits[i]), i))
    events = [json.loads(line) for line in (RUN / 'events.jsonl').read_bytes().splitlines()]
    attempts = [e for e in events if e['event'] == 'batch_complete']
    old_attempts = [e for line in (CONTROL / 'events.jsonl').read_bytes().splitlines()
                    if (e := json.loads(line))['event'] == 'batch_complete']
    fields = ('key', 'attempt', 'updated', 'step', 'sampled_row_ids')
    assert [{k: e.get(k) for k in fields} for e in attempts] == [{k: e.get(k) for k in fields} for e in old_attempts]
    assert len(attempts) == 54 and sum(e['updated'] for e in attempts) == 50
    assert [e['step'] for e in events if e['event'] == 'step_reserved'] == list(range(1, 51))
    visits, replay_pairs = Counter(), 0
    for e in attempts:
        if not e['updated']:
            assert e['sampled_rows'] == 0 and 'replay' not in e
            continue
        main = lookup[e['key']]; replay = e['replay']; chosen = None
        for offset in range(32):
            candidate = donors[(e['step'] - 1 + offset) % 32]
            if candidate != e['key'] and lookup[candidate]['molecule_id'] != main['molecule_id']:
                chosen = candidate; break
        assert chosen == replay['key'] and replay['binding'] == bank['entries'][chosen]['binding']
        assert replay['weight'] == 1. and replay['rows'] == 32 and replay['actual_optimizer_steps'] == 1
        assert replay['chemical_loss_added'] is False and e['sampled_unknown_pairs'] == 0
        assert math.isclose(e['objective_loss'], e['losses']['loss'] + replay['negative_bce'], abs_tol=1e-6, rel_tol=1e-6)
        visits[chosen] += 1; replay_pairs += replay['rows']
    assert len(visits) == 32 and replay_pairs == 1600
    assert report['counters'] == dict(upstream_forwards=58, evaluation_images=116, parity_probe_forwards=4,
                                    batch_attempts=54, replay_forwards=50, replay_pair_rows=1600)
    trajectory, old_trajectory = read(RUN / 'trajectories.json'), read(CONTROL / 'trajectories.json')
    complete = [e for e in events if e['event'] == 'observed_attempt_complete']
    assert trajectory['updates'] == [dict(update=e['result'], probes=e['probes']) for e in complete]
    assert sum(e['event'] == 'observed_attempt_before' for e in events) == len(complete) == 54
    derived, slots = {}, {}
    for key, summary in trajectory['summary'].items():
        track = [p for u in trajectory['updates'] for p in u['probes'] if p['probe']['key'] == key]
        assert len(track) == 50
        for a, b in zip(track, track[1:]): assert a['after']['presence_logit'] == b['before']['presence_logit']
        for p in track:
            e = next(e for e in attempts if e.get('step') == p['step'])
            direct = e['replay']['key'] == key
            assert p['replay_image_is_probe'] is direct
            assert p['probe_row_in_replay'] == (direct and p['probe']['row_id'] in e['replay']['binding']['row_ids'])
        own = [p for p in track if p['own_sample_update']]; assert len(own) == 1
        slots['own:' + key] = own[0]['step']
        positive = [p for p in track if p['presence_logit_delta'] > 1e-4]
        if positive: slots['maxrise:' + key] = max(positive, key=lambda p: (p['presence_logit_delta'], -p['step']))['step']
        crossings = [p['step'] for p in track if p['before']['presence_logit'] < math.log(.85 / .15) <= p['after']['presence_logit']]
        assert summary['crossing_steps'] == crossings
        derived[key] = dict(initial_probability=1 / (1 + math.exp(-summary['initial_logit'])),
            control_final_probability=1 / (1 + math.exp(-old_trajectory['summary'][key]['final_logit'])),
            repair_final_probability=1 / (1 + math.exp(-summary['final_logit'])), crossing_steps=crossings,
            direct_replay_steps=[p['step'] for p in track if p['probe_row_in_replay']],
            own_update_logit_change=summary['own_update_logit_change'])
    snapshots = read(RUN / 'snapshot-manifest.json'); assert snapshots['slots'] == slots
    assert len(snapshots['events']) == len(set(slots.values())) <= 4
    snapshot_bytes, errors, saved_donors = 0, {}, set()
    for key, name in snapshots['probes'].items():
        package = torch.load(RUN / name, map_location='cpu', weights_only=True)
        cached.verify_cache(package['cache']); snapshot_bytes += tensor_bytes(package)
        logits = torch.tensor(cases['baseline', key]['presence_logits'], dtype=torch.float32)
        verify_saved_binding(package['cache'], logits, bank['entries'][key]['binding'])
        saved_donors.add(key); del package
    for step, record in snapshots['events'].items():
        package = torch.load(RUN / record['file'], map_location='cpu', weights_only=True)
        e = package['event']; snapshot_bytes += tensor_bytes(e)
        assert e['step'] == int(step) and package['protocol_sha256'] == auth['protocol_sha256']
        assert e['record'] == next(u for u in trajectory['updates'] if u['update'].get('step') == int(step))
        cached.verify_cache(e['cache'])
        verify_saved_binding(e['replay_cache'], e['replay_initial_logits'], e['replay_binding'])
        saved_donors.add(e['replay_cache']['key']); errors[step] = adam_error(e)
        del package, e
    assert snapshot_bytes == snapshots['snapshot_tensor_bytes'] < 1024**3
    delta = torch.load(RUN / 'final-pair-repair.pt', map_location='cpu', weights_only=True)
    state_sha = sha256()
    for name, value in sorted(delta['pair_state'].items()):
        state_sha.update(name.encode()); state_sha.update(value.detach().cpu().contiguous().numpy().tobytes())
    assert state_sha.hexdigest() == report['pair_state_after']
    assert delta['optimizer_steps'] == 50 and not delta['deployment_authorized']
    changes = []
    for before, after in zip(old_final, final, strict=True):
        lost = sorted(set(before['correct_typed_bond_ids']) - set(after['correct_typed_bond_ids']))
        gained = sorted(set(after['correct_typed_bond_ids']) - set(before['correct_typed_bond_ids']))
        fields = {k: [before[k], after[k]] for k in ('confirmed_false_positives', 'unknown_present',
            'sdf_exact', 'sdf_valid', 'fallback', 'predicted_smiles') if before[k] != after[k]}
        if lost or gained or fields:
            changes.append(dict(key=after['key'], lost_vs_control=lost, gained_vs_control=gained, metric_changes=fields))
    assert digest(RUN / 'report.json') == report_sha and digest(__file__) == source_sha
    result = dict(schema_version='pair-nonedge-replay-postcheck-0.1', checks_passed=True,
        report_sha256=report_sha, runtime_sha256=digest(RUN / 'runtime.json'), source_sha256=source_sha,
        hashes_checked=dict(checked), confirmed_steps=50, confirmed_attempts=54, replay_pairs=1600,
        replay_donors_visited=32, historical_baseline_cases_identical=58, same_main_order_and_rows=True,
        saved_snapshot_donors_label_and_top32_verified=sorted(saved_donors), adam_arithmetic_max_errors=errors,
        snapshot_events=len(errors), snapshot_tensor_bytes=snapshot_bytes,
        initial=report['baseline']['all'], control=control['final']['all'], repair=report['final']['all'],
        gate=decision, derived_probes=derived, per_image_changes_vs_control=changes,
        model_instantiations=0, model_forwards=0, optimizer_steps_performed_by_postcheck=0, gpu_used=False,
        default_changed=False, independent_accuracy_measured=False, new_identity_queries=0,
        limitations=['Total typed-bond count equality does not imply identical bond IDs',
                     'Probe negatives were directly replayed; this is not unseen-sample generalization',
                     'Only saved donor caches independently recheck labels; runtime checks all donor caches'])
    write_json(OUTPUT / 'report.json', result)
    print(json.dumps(result, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--worker', action='store_true')
    if parser.parse_args(argv).worker:
        worker(); return 0
    receipt = runtime.run_budgeted([sys.executable, '-m', 'tools.audit_pair_replay_result', '--worker'],
                                  output=OUTPUT, budget=runtime.ResourceBudget(120))
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__': raise SystemExit(main())
