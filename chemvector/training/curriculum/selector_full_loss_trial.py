"""One authorized full-loss-only control; prior retention trial remains immutable."""

import argparse
import json
from pathlib import Path
import sys

from . import selector_retention_trial as previous
from .selector_retention_trial import (
    release, file_digest, verify_bindings, reserve_once, INITIAL, INITIAL_SHA,
    INITIAL_STATE, MANIFEST_SHA, ADMISSION_SHA, PROPOSAL_SHA, CORE_PLAN,
)
from .runtime import ResourceBudget, run_budgeted

AUTHORIZATION = Path('docs/model/training/releases/20260920-selector-full-loss-control-authorization.json')
OUTPUT = Path('outputs/experiments/chemvector-selector-full-loss-control-20260920')
ATTEMPT = Path('outputs/experiments/chemvector-selector-full-loss-control-attempt-20260920.json')
REFERENCE = previous.OUTPUT
REFERENCE_REPORT_SHA = 'e05e574bc43a5743942cc61ec717ebf913e80ab0634a99dc94d45fa8054afeb2'
REFERENCE_RUNTIME_SHA = 'ae0f74b755e9dbdd2cd0804eeee96831888f761e68744eff39b00c1614dbd9b2'
PROTOCOL = {**previous.PROTOCOL, 'retention_weight': 0., 'conditional_regression_cpu_seconds': 0}
DATA_USE_REVIEW = dict(use='same58_local_selector_retention_ablation_only',
    source_use='local_selector_retention_fit_only', manifest_sha256=MANIFEST_SHA,
    admission_sha256=ADMISSION_SHA, reference_proposal_sha256=PROPOSAL_SHA,
    fitting_images=58, development_only_images=64, changed_splits=0, copied_images=0,
    original_release_unchanged=True, permitted_change='retention_weight: 1 -> 0',
    private_identity_queries=0, full_dataset_training_authorized=False,
    reference_run_authorization_reused=False)


def authorize():
    manifest, admission, _ = previous.authorize()  # Validation only; never executes the consumed trial.
    verify_bindings({str(REFERENCE / 'report.json'): REFERENCE_REPORT_SHA,
                     str(REFERENCE / 'runtime.json'): REFERENCE_RUNTIME_SHA})
    reference = json.loads((REFERENCE / 'report.json').read_text())
    runtime = json.loads((REFERENCE / 'runtime.json').read_text())
    if (reference['optimizer_steps'] != 400 or reference['deployed']
            or runtime['status'] != 'completed' or runtime['elapsed_seconds'] > 600):
        raise ValueError('Expected completed fixed 400-step reference')
    auth = json.loads(AUTHORIZATION.read_text())
    required = dict(schema_version='chemvector-selector-full-loss-control-authorization-0.1',
        user_statement='开始下一步', protocol=PROTOCOL, data_use_review=DATA_USE_REVIEW,
        manifest_sha256=MANIFEST_SHA, admission_sha256=ADMISSION_SHA,
        initial_sha256=INITIAL_SHA, reference_report_sha256=REFERENCE_REPORT_SHA,
        reference_authorization_sha256=file_digest(previous.AUTHORIZATION),
        maximum_invocations=1, output=str(OUTPUT), training_authorized=True,
        full_regression_authorized=False, deployment_authorized=False)
    if any(auth.get(k) != v for k, v in required.items()):
        raise ValueError('Missing/mismatched exact ablation authorization or data-use extension')
    required_files = {str(Path(__file__).with_name(name).relative_to(release.ROOT)) for name in (
        'selector_full_loss_trial.py', 'selector_full_loss_worker.py', 'selector_full_loss_stage.py',
        'selector_full_loss_evidence.py')}
    if not required_files <= set(auth.get('implementation_sha256', {})):
        raise ValueError('Control implementation is not fully bound')
    expected_records = {str(REFERENCE / f'{phase}--{r["sample_key"]}.json')
        for r in manifest['training_samples']
        for phase in ('baseline', 'check-000100', 'check-000200', 'check-000400')}
    expected_records |= {str(REFERENCE / f'{phase}--{r["sample_key"]}.json')
        for r in manifest['validation_only_samples'] for phase in ('baseline', 'final-development')}
    expected_records |= {str(REFERENCE / n) for n in ('steps.jsonl', 'plan.json', 'report.json', 'runtime.json')}
    if set(auth.get('reference_records_sha256', {})) != expected_records:
        raise ValueError('Reference pairing records are incomplete')
    verify_bindings(auth['implementation_sha256'])
    verify_bindings(auth['reference_records_sha256'])
    return manifest, admission, auth


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        if args.check_only:
            parser.error('Worker cannot be check-only')
        from .selector_full_loss_worker import worker
        worker()
        return 0
    _, admission, _ = authorize()
    if args.check_only:
        print(json.dumps(dict(checks_passed=True, model_loaded=False, training_performed=False,
                              protocol=PROTOCOL, data_use_review=DATA_USE_REVIEW)))
        return 0
    if OUTPUT.exists():
        raise FileExistsError('Existing output; no retry')
    reserve_once(ATTEMPT, AUTHORIZATION)
    receipt = run_budgeted([sys.executable, '-m', __package__ + '.selector_full_loss_trial', '--worker'],
        output=OUTPUT, budget=ResourceBudget(600, ('0',)), training=True, admission=admission)
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
