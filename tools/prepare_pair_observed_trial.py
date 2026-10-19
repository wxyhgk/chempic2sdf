"""CPU-only preparation and synthetic tests; never launch a real trial."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

from chemvector.training.curriculum import runtime

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'outputs/audits/chemvector-pair-observed-preparation-20260921'
MANIFEST = ROOT / 'outputs/datasets/chemvector-selector-retention-fit-20260919/manifest.json'
MANIFEST_SHA = 'f5e3a7163eef5c49118ad78dcf05778f4ad889dc9b6389097ef20896b79e2111'
TESTS = ['tests/chemvector/test_pair_observed_update.py',
         'tests/chemvector/test_pair_negative_observer.py', 'tests/chemvector/test_pair_only_trial.py']


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def worker():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or not OUTPUT.is_dir():
        raise ValueError('Budgeted CPU preparation required')
    payload = MANIFEST.read_bytes()
    if sha256(payload).hexdigest() != MANIFEST_SHA:
        raise ValueError('Fixed58 manifest changed')
    rows = json.loads(payload)['training_samples']
    if len(rows) != 58 or any(r['split'] != 'train' for r in rows):
        raise ValueError('Fixed58 training-only rows required')
    probes = []
    for key, target_pair in [('curr_4ef506af64e8f0b8__bond_short', [31, 34]),
                             ('curr_08d02946e879406f__font_large', [7, 14])]:
        matching = [r for r in rows if r['sample_key'] == key]
        if len(matching) != 1:
            raise ValueError('Probe not uniquely found in fixed manifest')
        probes.append(dict(key=matching[0]['source'] + ':' + key, target_pair=target_pair,
                           binding='resolve_on_fresh_cache; must_be_confirmed_nonedge'))
    paths = [MANIFEST, Path(__file__), ROOT / 'tests/chemvector/test_curriculum_pair_stage.py']
    paths += [ROOT / t for t in TESTS]
    paths += [ROOT / 'chemvector/training/curriculum' / (name + '.py') for name in
              ('pair_observed_update', 'pair_negative_observer', 'pair_cached_trial',
               'pair_only_trial', 'pair_only_worker', 'pair_trial_contract',
               'pair_stage', 'runtime', 'supervision_contract', 'interface_diagnostic')]
    hashes = {str(p): sha256(p.read_bytes()).hexdigest() for p in paths}
    plan = dict(schema_version='pair-observed-preparation-0.1',
        status='adapter_preparation_not_real_launcher', probes=probes,
        proposed_real_trial=dict(maximum_steps=50, maximum_batch_attempts=58, wall_seconds=600,
            gpu_devices=['0'], maximum_invocations=1, automatic_retries=0,
            initial='original_initial_not_step50', samples=58, pair_threshold=.85,
            objective_sampler_optimizer='unchanged_pair_cached_trial.update_once',
            order='original_manifest_order_once', all_upstream_frozen=True,
            whole_graph_protections='unchanged_original_pair_trial_contract',
            observation_forwards_at_most=200),
        observation_scope='own_or_other_sample_update_timing_not_loss_component_causality',
        prerequisite=['New bounded launcher with durable event receipts and external wall-time guard',
                      'Final SHA-bound protocol and new one-shot training authorization; old grant consumed',
                      'Fresh device caches and initial full58 evaluation; do not reuse historical slot IDs',
                      'Abort on missing/nonnegative/unknown probe or incomplete telemetry; no automatic retry'],
        integration_contract=['After fresh caches: construct ObservedPairUpdates with exact two probes',
                              'Replace only update_once call in a NEW opt-in launcher with adapter.update',
                              'Use unchanged final full58 scoring and protection checks; no deployment'],
        training_authorized=False, real_trial_run=False, historical_launchers_modified=False,
        real_model_forwards=0, real_optimizer_steps=0, checkpoint_loads=0, gpu_used=False,
        validation_sample_artifacts_read=False, sealed_test_read=False, identity_queries=0,
        input_sha256=hashes)
    write(OUTPUT / 'plan.json', plan)
    result = subprocess.run([sys.executable, '-m', 'pytest', '-q', *TESTS,
        '--junitxml=' + str(OUTPUT / 'synthetic-tests.xml')], cwd=ROOT, timeout=60, check=False)
    suites = ET.parse(OUTPUT / 'synthetic-tests.xml').getroot().findall('testsuite')
    totals = {k: sum(int(s.attrib.get(k, 0)) for s in suites) for k in ('tests', 'failures', 'errors', 'skipped')}
    passed = bool(totals['tests']) and result.returncode == 0 and not totals['failures'] and not totals['errors']
    for path, digest in hashes.items():
        if sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError('Preparation inputs changed')
    write(OUTPUT / 'report.json', dict(**plan, synthetic_tests=totals, synthetic_tests_passed=passed,
        launch_ready=False, adapter_ready_for_future_integration=passed,
        synthetic_cpu_optimization_only=True,
        gpu_numerical_parity_established=False,
        artifact_sha256={name: sha256((OUTPUT / name).read_bytes()).hexdigest()
                         for name in ('plan.json', 'synthetic-tests.xml')}))
    if not passed:
        raise RuntimeError('Synthetic observer tests failed')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args(argv)
    if args.worker:
        worker(); return 0
    receipt = runtime.run_budgeted([sys.executable, '-m', 'tools.prepare_pair_observed_trial', '--worker'],
                                  output=OUTPUT, budget=runtime.ResourceBudget(120))
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
