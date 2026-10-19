"""Single-factor nonedge-replay proposal and gated launcher; no implicit grant."""

import argparse
from copy import deepcopy
import json
import sys

from . import pair_observed_trial as previous, pair_trial_contract as rules
from .admission import require_training_admission
from .checkpoints import file_digest
from .pair_cached_trial import json_digest
from .runtime import ResourceBudget, run_budgeted

ROOT, INITIAL, DATA = previous.ROOT, previous.INITIAL, previous.DATA
CONTROL = previous.OUTPUT
CONTROL_SHA = 'c828baa4ada80200c98add2e6058abded6a21cbf3871bd5611a66cd71a2aee9f'
POSTCHECK = ROOT / 'outputs/audits/chemvector-pair-observed-postcheck-20260922/report.json'
POSTCHECK_SHA = '121eb63b5bf6fe78f3e46887635c5d10514249af1ff9657eea340c5be9ef9ae8'
PREPARATION = ROOT / 'outputs/audits/chemvector-pair-replay-preparation-20260922'
OUTPUT = ROOT / 'outputs/experiments/chemvector-pair-replay-trial-20260922'
ATTEMPT = ROOT / 'outputs/experiments/chemvector-pair-replay-attempt-20260922.json'
AUTHORIZATION = ROOT / 'docs/model/training/releases/20260922-pair-replay-authorization.json'
IMPLEMENTATIONS = tuple(dict.fromkeys((*previous.IMPLEMENTATIONS,
    'chemvector/training/curriculum/pair_nonedge_replay.py',
    'chemvector/training/curriculum/pair_replay_trial.py',
    'chemvector/training/curriculum/pair_replay_worker.py',
    'tests/chemvector/test_pair_nonedge_replay.py',
    'tools/prepare_pair_replay_trial.py')))


def protocol():
    return dict(schema_version='pair-nonedge-replay-protocol-0.1',
        initial_sha256=rules.INITIAL_SHA, manifest_sha256=rules.MANIFEST_SHA,
        historical_control_report_sha256=CONTROL_SHA, original_recipe=deepcopy(rules.RECIPE),
        experimental_change=dict(objective='original_main_loss + 1.0 * mean(softplus(other_image_nonedge_logits))',
            replay_weight=1.0, weight_selection='single_unit_coefficient; no_scan_or_outcome_tuning',
            donor_cohort='exact_32_old_full_molecule_train_rows_in_manifest_order',
            donor_schedule='completed_update_index_mod32; advance_if_same_key_or_molecule',
            donor_rows='top32_initial_logits_among_confirmed_nonedges; row_id_breaks_ties',
            dynamic_mining=False, probe_specific_rows=False, chemical_replay_loss=False,
            original_main_pos_weight_unchanged=True, zero_main_supervision='skip_without_replay_or_step'),
        limits=dict(maximum_steps=50, maximum_batch_attempts=58, wall_seconds=600,
            gpu_devices=['0'], maximum_invocations=1, automatic_retries=0,
            maximum_replay_pairs_per_update=32, maximum_replay_pair_rows=1600,
            upstream_forwards=58, full58_evaluations=2, snapshot_events=4,
            snapshot_tensor_bytes=1024**3),
        equality_required_before_update=['same_original_initial', 'same_environment_as_control',
            'all58_fresh_cache_hashes_equal_control', 'all58_initial_saved_predictions_equal_control'],
        control='reuse_verified_original_objective_run; no_second_control_training_arm',
        execution_scope='limits_above_override_historical_recipe_budgets; one_manifest_pass; stop_at50_even_if_gate_passes',
        main_sampler_order_optimizer_clip_thresholds='unchanged',
        telemetry='same_two_probes; explicitly_distinguish_main_and_replay_probe_supervision',
        decision='original_full58_protections; no_promotion_on_any_regression; independent_accuracy_not_measured',
        targets='known_negative_mask_only; original_one_to_one_matching; no_unknown_or_positive_replay',
        data_use='same_admitted_fixed58_training_pool_only',
        training_authorized=False, deployment_authorized=False, resume_supported=False,
        forbidden=['new_data', 'development_sample_artifacts', 'sealed_tests', 'identity_queries',
                   'weight_sweeps', 'automatic_followup', 'extra_optimizer_step_for_replay'])


def source_checks():
    manifest = previous.old.read_bound(DATA / 'manifest.json', rules.MANIFEST_SHA)
    admission = previous.old.read_bound(DATA / 'admission.json', rules.ADMISSION_SHA)
    require_training_admission(admission)
    if admission['use'] != 'local_selector_retention_fit_only' or admission['training_run_authorized'] is not False:
        raise ValueError('Historical admission changed')
    rules.make_proposal(manifest['training_samples'])
    control = previous.old.read_bound(CONTROL / 'report.json', CONTROL_SHA)
    postcheck = previous.old.read_bound(POSTCHECK, POSTCHECK_SHA)
    if (not postcheck['checks_passed'] or postcheck['report_sha256'] != CONTROL_SHA
            or not postcheck['historical_final_pair_state_identical']
            or control['optimizer_steps'] != 50 or file_digest(INITIAL) != rules.INITIAL_SHA):
        raise ValueError('Verified initial/control evidence required')
    for path, digest in control['implementation_sha256'].items():
        if file_digest(ROOT / path) != digest: raise ValueError('Historical control code changed')
    return manifest, admission, control


def validate_authorization(auth):
    expected = dict(schema_version='pair-nonedge-replay-authorization-0.1',
        protocol_sha256=json_digest(protocol()), manifest_sha256=rules.MANIFEST_SHA,
        initial_sha256=rules.INITIAL_SHA, original_admission_sha256=rules.ADMISSION_SHA,
        historical_control_report_sha256=CONTROL_SHA, training_authorized=True,
        deployment_authorized=False, maximum_invocations=1, limits=protocol()['limits'],
        use='local_pair_nonedge_replay58_once', output=str(OUTPUT.relative_to(ROOT)))
    if (any(json_digest(auth.get(k)) != json_digest(v) for k, v in expected.items())
            or not isinstance(auth.get('user_statement'), str) or not auth['user_statement'].strip()
            or set(auth.get('implementation_sha256', {})) != set(IMPLEMENTATIONS)):
        raise ValueError('New exact single-factor repair authorization required; old grant cannot be reused')
    for value in auth['implementation_sha256'].values():
        if not isinstance(value, str) or len(value) != 64 or set(value) - set('0123456789abcdef'):
            raise ValueError('Invalid implementation binding')


def authorize():
    # This module deliberately has no prepare/auto-authorization switch.
    auth = json.loads(AUTHORIZATION.read_text()); validate_authorization(auth)
    preparation = json.loads((PREPARATION / 'report.json').read_text())
    if (not preparation['synthetic_tests_passed'] or not preparation['ready_except_new_authorization']
            or preparation['protocol_sha256'] != auth['protocol_sha256']
            or preparation['training_authorized'] is not False
            or preparation['implementation_sha256'] != auth['implementation_sha256']
            or file_digest(PREPARATION / 'report.json') != auth.get('preparation_report_sha256')):
        raise ValueError('Final preparation report not bound to new grant')
    for name, digest in preparation['artifact_sha256'].items():
        if file_digest(PREPARATION / name) != digest: raise ValueError('Preparation evidence changed: ' + name)
    for path, digest in auth['implementation_sha256'].items():
        if file_digest(ROOT / path) != digest: raise ValueError('Repair implementation changed: ' + path)
    manifest, admission, control = source_checks()
    return manifest, dict(admission, use=auth['use'], training_run_authorized=True,
        execution_authorization_sha256=file_digest(AUTHORIZATION)), auth, control


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--check-only', action='store_true')
    modes.add_argument('--worker', action='store_true')
    args = parser.parse_args(argv)
    if args.check_only:
        manifest, _, _ = source_checks()
        print(json.dumps(dict(source_checks_passed=True, training_rows=len(manifest['training_samples']),
            authorization_present=AUTHORIZATION.exists(), attempt_consumed=ATTEMPT.exists(),
            model_loaded=False, training_performed=False, protocol=protocol()))); return 0
    if args.worker:
        from .pair_replay_worker import worker
        worker(); return 0
    _, admission, _, _ = authorize()
    if OUTPUT.exists(): raise FileExistsError('Repair output exists; no retry or overwrite')
    previous.old.reserve_once(ATTEMPT, file_digest(AUTHORIZATION))
    receipt = run_budgeted([sys.executable, '-m', __package__ + '.pair_replay_trial', '--worker'],
        output=OUTPUT, budget=ResourceBudget(600, ('0',)), training=True, admission=admission)
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__': raise SystemExit(main())
