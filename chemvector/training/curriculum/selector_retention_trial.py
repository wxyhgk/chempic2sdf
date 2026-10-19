"""One fixed, explicitly authorized selector-only trial; never resumes or deploys."""

import argparse
import json
import os
from pathlib import Path
import sys

from . import selector_retention_release as release
from .checkpoints import file_digest
from .runtime import ResourceBudget, run_budgeted

AUTHORIZATION = Path('docs/model/training/releases/20260920-selector-retention-training-authorization.json')
OUTPUT = Path('outputs/experiments/chemvector-selector-retention-trial-20260920')
ATTEMPT = Path('outputs/experiments/chemvector-selector-retention-attempt-20260920.json')
INITIAL = Path('outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt')
INITIAL_SHA = '09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1'
INITIAL_STATE = '9d7f66f79329479e05c38f8b91d95118adb9f12c32a44e5bd9d83a458a8b452e'
MANIFEST_SHA = 'f5e3a7163eef5c49118ad78dcf05778f4ad889dc9b6389097ef20896b79e2111'
ADMISSION_SHA = '77c4b42da2f675e61924ee7904032deb05d30b8de7767f92bfd9888aed1e85f9'
PROPOSAL_SHA = '4111f716f2833c444a356550ad5332929c7b5bcbd74333fa349d201965b5afcf'
CORE_PLAN = Path('outputs/audits/curriculum-v02-selector-loss-terms4-20260919/plan.json')
CORE_SHA = 'bbefebd295061f28d278f91af61306789e0bf59ce4ee831b94c9cfcf824595e6'
PROTOCOL = dict(wall_seconds=600, maximum_steps=400, gpu_devices=['0'], seed=20260920,
    local_checks=[0, 100, 200, 400], batch_size=1, learning_rate=5e-6, eps=1e-8,
    betas=[.9, .999], weight_decay=0, clip_norm=1., retention_weight=1.,
    training_cycle=['old_full_molecule', 'old_full_molecule', 'witness'],
    parameter_scope='set_head only', model_mode='eval', automatic_retries=0,
    conditional_regression_cpu_seconds=2400, conditional_regression_gpu_count=0,
    tf32=False, cudnn_benchmark=False)


def verify_bindings(bindings):
    for path, expected in bindings.items():
        if file_digest(path) != expected:
            raise ValueError('Bound input/implementation changed: ' + str(path))


def authorize():
    verify_bindings({str(release.OUTPUT / 'manifest.json'): MANIFEST_SHA,
        str(release.OUTPUT / 'admission.json'): ADMISSION_SHA,
        str(release.OUTPUT / 'trial-proposal.json'): PROPOSAL_SHA,
        str(INITIAL): INITIAL_SHA, str(CORE_PLAN): CORE_SHA})
    manifest, admission = release.verify_release(release.OUTPUT)
    auth = json.loads(AUTHORIZATION.read_text())
    required = {'schema_version': 'chemvector-selector-retention-training-authorization-0.1',
        'user_statement': '继续推进', 'protocol': PROTOCOL, 'manifest_sha256': MANIFEST_SHA,
        'admission_sha256': ADMISSION_SHA, 'proposal_sha256': PROPOSAL_SHA,
        'initial_sha256': INITIAL_SHA, 'maximum_invocations': 1, 'output': str(OUTPUT),
        'training_authorized': True, 'conditional_full_regression_authorized': True,
        'deployment_authorized': False}
    if any(auth.get(k) != v for k, v in required.items()):
        raise ValueError('Missing/mismatched one-shot selector trial authorization')
    files = {str(Path(__file__).with_name(name).relative_to(release.ROOT)) for name in (
        'selector_retention_trial.py', 'selector_retention_worker.py', 'selector_retention_stage.py',
        'selector_retention_release.py', 'runtime.py', 'bundle.py', 'evaluation.py', 'continuation.py')}
    if not files <= set(auth.get('implementation_sha256', {})):
        raise ValueError('Authorization omits a critical implementation')
    verify_bindings(auth['implementation_sha256'])
    verify_bindings(auth['protected_inputs_sha256'])
    return manifest, admission, auth


def reserve_once(path, authorization_path):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(dict(authorization_sha256=file_digest(authorization_path), automatic_retries=0), stream)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        if args.check_only:
            parser.error('Worker cannot be check-only')
        from .selector_retention_worker import worker
        worker()
        return 0
    _, admission, _ = authorize()
    if args.check_only:
        print(json.dumps(dict(checks_passed=True, protocol=PROTOCOL,
                              model_loaded=False, training_performed=False)))
        return 0
    if OUTPUT.exists():
        raise FileExistsError('Existing trial output; no retry')
    reserve_once(ATTEMPT, AUTHORIZATION)
    receipt = run_budgeted([sys.executable, '-m', __package__ + '.selector_retention_trial', '--worker'],
        output=OUTPUT, budget=ResourceBudget(600, ('0',)), training=True, admission=admission)
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
