"""CPU synthetic/metadata preparation; never grants, reserves or starts training."""

import argparse
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

import torch

from chemvector.training.curriculum import pair_anchor_ab_trial as launch, runtime
from chemvector.training.curriculum.checkpoints import file_digest
from chemvector.training.curriculum.interface_diagnostic import write_json
from chemvector.training.curriculum.pair_cached_trial import json_digest

TESTS = (
    'tests/chemvector/test_pair_anchor_ab.py', 'tests/chemvector/test_pair_anchor_device.py',
    'tests/chemvector/test_pair_anchor_wiring.py', 'tests/chemvector/test_pair_anchor_schedule.py',
    'tests/chemvector/test_pair_anchor_visual_review.py', 'tests/chemvector/test_pair_anchor_augmentation.py',
    'tests/chemvector/test_pair_anchor_preparation.py', 'tests/chemvector/test_pair_only_trial.py',
    'tests/chemvector/test_pair_trial_contract.py', 'tests/chemvector/test_curriculum_supervision_contract.py',
    'tests/chemvector/test_curriculum_pair_supervision.py', 'tests/chemvector/test_curriculum_pair_stage.py',
    'tests/chemvector/test_curriculum_runtime.py', 'tests/data_sdf/test_pair_sampling_ensemble.py',
)


def worker():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('CPU-only preparation required')
    if any(p.exists() for p in (launch.AUTHORIZATION, launch.ATTEMPT, launch.OUTPUT)):
        raise ValueError('Preparation cannot authorize, reserve or reuse a training attempt')
    manifest, _ = launch.source_checks()
    implementations = {p: file_digest(launch.ROOT / p) for p in launch.IMPLEMENTATIONS}
    inputs = {str(launch.ROOT / p): file_digest(launch.ROOT / p) for p in TESTS}
    inputs.update({str(launch.DATA / 'manifest.json'): launch.rules.original.MANIFEST_SHA,
        str(launch.DATA / 'admission.json'): launch.rules.original.ADMISSION_SHA,
        str(launch.INITIAL): launch.rules.original.INITIAL_SHA,
        str(launch.WIRING / 'report.json'): launch.WIRING_SHA, str(launch.WIRING_POST): launch.WIRING_POST_SHA})
    report = launch.old.read_bound(launch.WIRING / 'report.json', launch.WIRING_SHA)
    path = launch.WIRING / 'one_sweep.json'
    source = launch.old.read_bound(path, report['artifact_sha256'][path.name]); inputs[str(path)] = report['artifact_sha256'][path.name]
    keys = [r['sample_key'] for r in manifest['training_samples']]
    if ([s['sample_key'] for s in source['steps']] != keys or len(keys) != 58
            or source['summary']['simulated_updates'] != 54
            or source['summary']['never_base_sampled_rows']['rows'] != 0
            or any(s['view_index'] != 1 for s in source['steps'])):
        raise ValueError('Previously verified complete single pass changed')
    protocol = launch.rules.protocol(); launch.rules.validate_protocol(protocol)
    write_json(launch.PREPARATION / 'protocol-draft.json', protocol)
    write_json(launch.PREPARATION / 'metadata-plan.json', dict(
        sample_keys=keys, historical_summary=source['summary'], historical_row_ids_reused_for_training=False,
        new_cache_before_update=True, two_arms_share_original_supervision=True,
        prospective_limits=protocol['limits'], training_authorized=False))
    test_path = launch.PREPARATION / 'synthetic-tests.xml'
    result = subprocess.run([sys.executable, '-m', 'pytest', '-q', *TESTS, '--junitxml=' + str(test_path)],
                            cwd=launch.ROOT, timeout=60, check=False)
    suites = ET.parse(test_path).getroot().findall('testsuite')
    totals = {k: sum(int(s.attrib.get(k, 0)) for s in suites) for k in ('tests', 'failures', 'errors', 'skipped')}
    passed = result.returncode == 0 and totals['tests'] > 0 and not any(totals[k] for k in ('failures', 'errors', 'skipped'))
    for path, digest in inputs.items():
        if file_digest(path) != digest: raise ValueError('Preparation input changed')
    for path, digest in implementations.items():
        if file_digest(launch.ROOT / path) != digest: raise ValueError('Implementation changed during preparation')
    launch.source_checks()
    if torch.cuda.is_initialized() or any(p.exists() for p in (launch.AUTHORIZATION, launch.ATTEMPT, launch.OUTPUT)):
        raise ValueError('Preparation crossed its no-execution boundary')
    write_json(launch.PREPARATION / 'report.json', dict(schema_version='pair-anchor-ab-preparation-0.1',
        protocol_sha256=json_digest(protocol), input_sha256=inputs, implementation_sha256=implementations,
        synthetic_tests=totals, synthetic_tests_passed=passed, launch_gates_prepared=passed,
        training_authorized=False, training_performed=False, gpu_used=False,
        real_model_loads=0, real_model_forwards=0, real_optimizer_steps=0,
        gpu_numerical_parity_established=False, initial_checkpoint_bytes_hashed_only=True,
        default_changed=False, old_authorizations_reused=False, new_identity_queries=0,
        validation_sample_artifacts_read=False, sealed_test_read=False,
        pending=['New exact data-use and one-shot two-arm authorization',
                 'Within that authorized invocation: GPU fresh58 cache and adapter/baseline parity before updates'],
        artifact_sha256={n: file_digest(launch.PREPARATION / n) for n in
                        ('protocol-draft.json', 'metadata-plan.json', 'synthetic-tests.xml')},
        limitations=['CPU synthetic wiring is not real GPU execution or a repair result',
                     'One pass observes one fixed shift, not eight-direction robustness',
                     'Fixed58 train-fit only; no new data diversity or independent accuracy',
                     'GPU driver availability was not tested; no automatic CPU fallback exists']))
    if not passed: raise RuntimeError('Synthetic preparation failed')
    print(json.dumps(dict(synthetic_tests=totals, limits=protocol['limits'], training_authorized=False)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker: worker(); return 0
    receipt = runtime.run_budgeted([sys.executable, '-m', 'tools.prepare_pair_anchor_ab_trial', '--worker'],
                                  output=launch.PREPARATION, budget=runtime.ResourceBudget(120))
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__': raise SystemExit(main())
