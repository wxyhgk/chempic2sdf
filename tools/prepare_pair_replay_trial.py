"""CPU metadata/synthetic preparation only; never grants or invokes training."""

import argparse
from collections import Counter
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

from chemvector.training.curriculum import pair_replay_trial as launch, runtime
from chemvector.training.curriculum.checkpoints import file_digest
from chemvector.training.curriculum.pair_cached_trial import json_digest
from chemvector.training.curriculum.pair_nonedge_replay import donor_key

TESTS = (
    'tests/chemvector/test_pair_nonedge_replay.py',
    'tests/chemvector/test_pair_observed_trial.py',
    'tests/chemvector/test_pair_observed_update.py',
    'tests/chemvector/test_pair_negative_observer.py',
    'tests/chemvector/test_pair_only_trial.py',
    'tests/chemvector/test_pair_trial_contract.py',
    'tests/chemvector/test_curriculum_pair_stage.py',
    'tests/chemvector/test_curriculum_supervision_contract.py',
    'tests/chemvector/test_curriculum_runtime.py',
)


def metadata_plan(rows, receipts, attempts):
    """Only image-level scheduling, never old candidate-slot supervision."""
    launch.rules.make_proposal(rows)
    donors = [r for r in rows if r['fit_domain'] == 'old_full_molecule']
    if len({r['molecule_id'] for r in donors}) != 32:
        raise ValueError('Distinct32 original full-molecule donors required')
    keys = [r['source'] + ':' + r['sample_key'] for r in rows]
    if [r['key'] for r in receipts] != keys:
        raise ValueError('Exact ordered58 cache receipts required')
    by_key = {r['key']: r for r in receipts}
    donor_keys = [r['source'] + ':' + r['sample_key'] for r in donors]
    counts = {k: by_key[k]['coverage']['known_negative_pairs'] for k in donor_keys}
    if any(type(n) is not int or n < 32 for n in counts.values()):
        raise ValueError('Every donor must have at least32 confirmed nonedges')
    if not 50 <= len(attempts) <= 58: raise ValueError('Bounded historical attempt count required')
    steps, assignments = 0, []
    for i, (event, row) in enumerate(zip(attempts, rows, strict=False)):
        key = keys[i]; coverage = by_key[key]['coverage']
        updated = coverage['known_negative_pairs'] + coverage['reachable_positive_pairs'] > 0
        if (steps >= 50 or event['key'] != key or type(event['attempt']) is not int
                or event['attempt'] != i or event['updated'] is not updated):
            raise ValueError('Historical order/empty supervision changed')
        if updated:
            donor = donor_key(donors, row, steps)
            steps += 1
            if event.get('step') != steps: raise ValueError('Historical real-update index changed')
            assignments.append(dict(attempt=i, step=steps, main_key=key, donor_key=donor,
                                    maximum_pairs=32))
        elif event.get('step') is not None:
            raise ValueError('Empty attempt must not consume a real update')
    if steps != 50: raise ValueError('Verified original50-step control required')
    return dict(donor_images=32, eligible_confirmed_nonedges=sum(counts.values()),
        minimum_confirmed_nonedges_per_donor=min(counts.values()),
        planned_fixed_bank_pairs=32 * 32, planned_replay_pair_rows=32 * steps,
        historical_attempts=len(attempts), empty_attempts=len(attempts) - steps,
        planned_optimizer_updates=steps, donor_keys=donor_keys,
        donor_visit_counts=dict(Counter(r['donor_key'] for r in assignments)), assignments=assignments,
        pair_row_selection='deferred_to_fresh_execution_device_cache_and_identical_initial_logits',
        historical_pair_slots_reused=False, no_probe_specific_selection=True)


def write(name, value):
    with (launch.PREPARATION / name).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def worker():
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '' or not launch.PREPARATION.is_dir()
            or any(p.exists() for p in (launch.AUTHORIZATION, launch.ATTEMPT, launch.OUTPUT))):
        raise ValueError('Fresh CPU-only preparation, without authorization or training invocation, required')
    manifest, _, control = launch.source_checks()
    implementations = {p: file_digest(launch.ROOT / p) for p in launch.IMPLEMENTATIONS}
    inputs = {str(launch.DATA / 'manifest.json'): launch.rules.MANIFEST_SHA,
        str(launch.DATA / 'admission.json'): launch.rules.ADMISSION_SHA,
        str(launch.INITIAL): launch.rules.INITIAL_SHA,
        str(launch.CONTROL / 'report.json'): launch.CONTROL_SHA,
        str(launch.POSTCHECK): launch.POSTCHECK_SHA}
    for name in ('cache-manifest.json', 'events.jsonl'):
        path = launch.CONTROL / name; expected = control['artifact_sha256'][name]
        if file_digest(path) != expected: raise ValueError('Historical metadata changed: ' + name)
        inputs[str(path)] = expected
    for path in TESTS:
        inputs[str(launch.ROOT / path)] = file_digest(launch.ROOT / path)
    receipts = json.loads((launch.CONTROL / 'cache-manifest.json').read_text())
    attempts = [event for line in (launch.CONTROL / 'events.jsonl').read_bytes().splitlines()
                if (event := json.loads(line))['event'] == 'batch_complete']
    plan = metadata_plan(manifest['training_samples'], receipts, attempts)
    write('protocol-draft.json', launch.protocol())
    write('replay-metadata-plan.json', plan)
    result = subprocess.run([sys.executable, '-m', 'pytest', '-q', *TESTS,
        '--junitxml=' + str(launch.PREPARATION / 'synthetic-tests.xml')],
        cwd=launch.ROOT, timeout=60, check=False)
    suites = ET.parse(launch.PREPARATION / 'synthetic-tests.xml').getroot().findall('testsuite')
    totals = {k: sum(int(s.attrib.get(k, 0)) for s in suites) for k in ('tests', 'failures', 'errors', 'skipped')}
    passed = bool(totals['tests']) and result.returncode == 0 and not any(totals[k] for k in ('failures', 'errors', 'skipped'))
    for path, digest in inputs.items():
        if file_digest(path) != digest: raise ValueError('Preparation input changed: ' + path)
    for path, digest in implementations.items():
        if file_digest(launch.ROOT / path) != digest: raise ValueError('Preparation implementation changed: ' + path)
    launch.source_checks()
    if any(p.exists() for p in (launch.AUTHORIZATION, launch.ATTEMPT, launch.OUTPUT)):
        raise ValueError('Preparation must not authorize or reserve a trial')
    write('report.json', dict(schema_version='pair-nonedge-replay-preparation-0.1',
        protocol_sha256=json_digest(launch.protocol()), implementation_sha256=implementations,
        input_sha256=inputs, synthetic_tests=totals, synthetic_tests_passed=passed,
        ready_except_new_authorization=passed, training_authorized=False, training_performed=False,
        gpu_used=False, real_model_forwards=0, real_optimizer_steps=0, real_checkpoint_loads=0,
        initial_checkpoint_bytes_hashed_only=True, donor_images=plan['donor_images'],
        eligible_confirmed_nonedges=plan['eligible_confirmed_nonedges'],
        planned_replay_pair_rows=plan['planned_replay_pair_rows'], historical_control_unchanged=True,
        gpu_numerical_parity_established=False, repair_effectiveness_measured=False,
        deployment_authorized=False, validation_sample_artifacts_read=False, sealed_test_read=False,
        new_identity_queries=0, automatic_retries=0,
        artifact_sha256={n: file_digest(launch.PREPARATION / n) for n in
            ('protocol-draft.json', 'replay-metadata-plan.json', 'synthetic-tests.xml')},
        pending=['New explicit one-shot repair authorization',
                 'At authorized execution: exact environment/fresh58 cache/baseline parity before first update'],
        limitations=['Synthetic contracts are not evidence of real repair success',
                     'Fixed58 are training/replay images, not independent test accuracy']))
    if not passed: raise RuntimeError('Synthetic replay tests failed')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args(argv)
    if args.worker:
        worker(); return 0
    receipt = runtime.run_budgeted([sys.executable, '-m', 'tools.prepare_pair_replay_trial', '--worker'],
        output=launch.PREPARATION, budget=runtime.ResourceBudget(120))
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__': raise SystemExit(main())
