"""New one-shot observed fixed58 diagnostic; never resumes or deploys."""

import argparse
import json
from pathlib import Path
import sys

from . import pair_trial_contract as rules
from . import pair_only_trial as old
from .admission import require_training_admission
from .checkpoints import file_digest
from .interface_diagnostic import write_json
from .pair_cached_trial import json_digest
from .runtime import ResourceBudget, run_budgeted

ROOT, INITIAL, DATA = old.ROOT, old.INITIAL, old.DATA
OUTPUT = ROOT / 'outputs/experiments/chemvector-pair-observed-trial-20260922'
ATTEMPT = ROOT / 'outputs/experiments/chemvector-pair-observed-attempt-20260922.json'
PROTOCOL = ROOT / 'docs/model/training/releases/20260922-pair-observed-protocol.json'
AUTHORIZATION = ROOT / 'docs/model/training/releases/20260922-pair-observed-authorization.json'
PREPARATION = ROOT / 'outputs/audits/chemvector-pair-observed-preparation-20260921/report.json'
PREPARATION_SHA = '750278e754d51d6ac11f22911575382d7668228d5ab82629841af5a5dace6c06'
USE = 'local_pair_observed_regression58_once'
PROBES = [('full_molecule_expansion:curr_4ef506af64e8f0b8__bond_short', [31, 34]),
          ('full_molecule:curr_08d02946e879406f__font_large', [7, 14])]
IMPLEMENTATIONS = tuple(dict.fromkeys((*old.IMPLEMENTATIONS,
    'chemvector/training/curriculum/pair_observed_trial.py',
    'chemvector/training/curriculum/pair_observed_worker.py',
    'chemvector/training/curriculum/pair_observed_capture.py',
    'chemvector/training/curriculum/pair_event_analysis.py',
    'chemvector/training/curriculum/pair_observed_update.py',
    'chemvector/training/curriculum/pair_negative_observer.py',
    'tests/chemvector/test_pair_observed_trial.py',
    'tests/chemvector/test_pair_observed_update.py',
    'tests/chemvector/test_pair_negative_observer.py')))


def protocol():
    return dict(schema_version='pair-observed-protocol-0.1', original_recipe=rules.RECIPE,
        initial_sha256=rules.INITIAL_SHA, manifest_sha256=rules.MANIFEST_SHA,
        probes=[dict(key=k, target_pair=p) for k, p in PROBES],
        limits=dict(maximum_steps=50, maximum_batch_attempts=58, wall_seconds=600,
                    gpu_devices=['0'], maximum_invocations=1, automatic_retries=0,
                    upstream_forwards=58, full_cohort_evaluations=2,
                    snapshot_events=4, snapshot_tensor_bytes=1024**3,
                    local_event_analyses=1, analysis_pair_forwards=2),
        order='original_fixed58_manifest_order_once', thresholds_unchanged=True,
        snapshot_selection='each_probe_own_update_and_largest_positive_logit_change; earliest_tie',
        parity=dict(atol=1e-4, rtol=1e-5, comparison='single_probe_vs_full_pair_same_state_at_baseline_and_final'),
        runtime='one_supervised_worker_including_load_cache_eval_updates_analysis_and_writes',
        diagnostic_not_fix=True, deployment_authorized=False, resume_supported=False,
        early_exit='actual_final_step_metrics_or_explicit_failure; never_baseline_as_final',
        missing_or_unknown_probe='abort_before_optimizer',
        crossing='record_only; preserve_sequence_through_own_updates_within_50_step_cap',
        protections='unchanged_pair_trial_contract; evaluate_all58; no_promotion_even_if_passed',
        forbidden=['new_data', 'private_identity_queries', 'validation_sample_artifacts',
                   'sealed_test', 'automatic_followup', 'threshold_changes', 'optimizer_counterfactual_steps'])


def authorization(implementation):
    return dict(schema_version='pair-observed-authorization-0.1',
        user_statement='可以试试看',
        approved_context='Three-agent recommendation: original initial/fixed58, <=50 updates, one GPU, <=600 seconds, <=4 key event snapshots; no deployment or automatic extra experiment',
        protocol_sha256=json_digest(protocol()),
        manifest_sha256=rules.MANIFEST_SHA, original_admission_sha256=rules.ADMISSION_SHA,
        initial_sha256=rules.INITIAL_SHA, implementation_sha256=implementation,
        training_authorized=True, deployment_authorized=False, maximum_invocations=1,
        output=str(OUTPUT.relative_to(ROOT)), limits=protocol()['limits'],
        data_use_extension=dict(original_use='local_selector_retention_fit_only', approved_use=USE,
            training_images=58, new_data=False, validation_sample_artifacts=False, identity_queries=0))


def validate_authorization(auth):
    implementation = auth.get('implementation_sha256', {})
    if (set(implementation) != set(IMPLEMENTATIONS)
            or any(not isinstance(v, str) or len(v) != 64 or set(v) - set('0123456789abcdef') for v in implementation.values())
            or json_digest(auth) != json_digest(authorization(implementation))):
        raise ValueError('New exact one-shot observed diagnostic authorization required')


def source_checks():
    manifest = old.read_bound(DATA / 'manifest.json', rules.MANIFEST_SHA)
    admission = old.read_bound(DATA / 'admission.json', rules.ADMISSION_SHA)
    require_training_admission(admission)
    if admission['use'] != 'local_selector_retention_fit_only' or admission['training_run_authorized'] is not False:
        raise ValueError('Historical admission changed')
    rules.make_proposal(manifest['training_samples'])  # Exact 58 / 26+32 / train-only contract.
    prep = old.read_bound(PREPARATION, PREPARATION_SHA)
    if not prep['synthetic_tests_passed'] or prep['real_trial_run']:
        raise ValueError('Expected completed synthetic preparation only')
    if file_digest(INITIAL) != rules.INITIAL_SHA:
        raise ValueError('Initial bytes changed')
    return manifest, admission


def prepare():
    # Records the CURRENT user confirmation; never borrows the consumed old grant.
    if any(p.exists() for p in (PROTOCOL, AUTHORIZATION, ATTEMPT, OUTPUT)):
        raise FileExistsError('Preparation is exclusive; existing records must not be replaced')
    source_checks()
    hashes = {p: file_digest(ROOT / p) for p in IMPLEMENTATIONS}
    write_json(PROTOCOL, protocol())
    write_json(AUTHORIZATION, authorization(hashes))


def authorize():
    auth = json.loads(AUTHORIZATION.read_text()); validate_authorization(auth)
    if json_digest(json.loads(PROTOCOL.read_text())) != auth['protocol_sha256']:
        raise ValueError('Diagnostic protocol changed')
    for path, digest in auth['implementation_sha256'].items():
        if file_digest(ROOT / path) != digest: raise ValueError('Implementation changed: ' + path)
    manifest, admission = source_checks()
    return manifest, dict(admission, use=USE, training_run_authorized=True,
        original_admission_sha256=rules.ADMISSION_SHA,
        execution_authorization_sha256=file_digest(AUTHORIZATION)), auth


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--prepare', action='store_true')
    modes.add_argument('--check-only', action='store_true')
    modes.add_argument('--worker', action='store_true')
    args = parser.parse_args(argv)
    if args.prepare:
        prepare(); print('Recorded current user confirmation; no invocation consumed'); return 0
    if args.worker:
        from .pair_observed_worker import worker
        worker(); return 0
    _, admission, _ = authorize()
    if args.check_only:
        print(json.dumps(dict(checks_passed=True, model_loaded=False, attempt_consumed=ATTEMPT.exists(),
                              output_exists=OUTPUT.exists(), limits=protocol()['limits']))); return 0
    if OUTPUT.exists(): raise FileExistsError('No retry or overwrite of diagnostic output')
    old.reserve_once(ATTEMPT, file_digest(AUTHORIZATION))
    receipt = run_budgeted([sys.executable, '-m', __package__ + '.pair_observed_trial', '--worker'],
        output=OUTPUT, budget=ResourceBudget(600, ('0',)), training=True, admission=admission)
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
