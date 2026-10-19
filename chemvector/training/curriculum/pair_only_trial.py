"""One-shot fixed58 pair-only training launcher. No resume or deployment."""

import argparse
import json
import os
from pathlib import Path
import sys

from .admission import require_training_admission
from .checkpoints import file_digest
from .pair_cached_trial import json_digest
from . import pair_trial_contract as contract
from .runtime import ResourceBudget, run_budgeted

ROOT = Path(__file__).resolve().parents[3]
PREPARATION = ROOT / 'outputs/audits/chemvector-pair-only-trial-preparation-20260921'
PROPOSAL_SHA = '1ce38bbba63ac89778ddb68c4a695a3acc59590b5210e6c0f2b70a964e18572b'
PREPARATION_SHA = '2ef3343ca9d5121a8d9e09b3f78c724d073fd87b13ff3c217b3dabf9989e86ab'
DATA = ROOT / 'outputs/datasets/chemvector-selector-retention-fit-20260919'
INITIAL = ROOT / 'outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt'
AUTHORIZATION = ROOT / 'docs/model/training/releases/20260921-pair-only-training-authorization.json'
OUTPUT = ROOT / 'outputs/experiments/chemvector-pair-only-matched-trial-20260921'
ATTEMPT = ROOT / 'outputs/experiments/chemvector-pair-only-matched-attempt-20260921.json'

IMPLEMENTATIONS = (
    'chemvector/training/curriculum/pair_only_trial.py',
    'chemvector/training/curriculum/pair_only_worker.py',
    'chemvector/training/curriculum/pair_cached_trial.py',
    'chemvector/training/curriculum/pair_trial_contract.py',
    'chemvector/training/curriculum/pair_stage.py',
    'chemvector/training/curriculum/supervision_contract.py',
    'chemvector/training/curriculum/pair_supervision.py',
    'chemvector/training/curriculum/runtime.py',
    'chemvector/training/curriculum/admission.py',
    'chemvector/training/curriculum/bundle.py',
    'chemvector/training/curriculum/checkpoints.py',
    'chemvector/training/curriculum/data.py',
    'chemvector/training/curriculum/pixels.py',
    'chemvector/training/curriculum/continuation.py',
    'chemvector/training/curriculum/evaluation.py',
    'chemvector/training/curriculum/validation_truth.py',
    'chemvector/training/curriculum/interface_diagnostic.py',
    'chemvector/training/curriculum/gradient_preflight.py',
    'chemvector/training/graph/relations/train_direct_atom_pair_bond.py',
    'chemvector/components/graph/joint/direct_atom_pair_bond.py',
    'chemvector/components/graph/joint/predicted_incidence.py',
    'chemvector/components/graph/proposals/proposal_object_set.py',
    'chemvector/core/model.py',
    'tests/chemvector/test_pair_only_trial.py',
)


def validate_authorization(auth):
    expected = dict(schema_version='chemvector-pair-only-training-authorization-0.1',
        user_statement='持续推进，开始第一轮的训练', reaffirmation='继续',
        protocol_sha256=PROPOSAL_SHA, manifest_sha256=contract.MANIFEST_SHA,
        original_admission_sha256=contract.ADMISSION_SHA, initial_sha256=contract.INITIAL_SHA,
        training_authorized=True, deployment_authorized=False, maximum_invocations=1,
        output=str(OUTPUT.relative_to(ROOT)), recipe=contract.RECIPE,
        data_use_extension=dict(previous_use='local_selector_retention_fit_only',
            approved_use=contract.USE, training_images=58, new_data=False,
            validation_sample_artifacts=False, identity_queries=0))
    if any(json_digest(auth.get(k)) != json_digest(v) for k, v in expected.items()):
        raise ValueError('Missing or mismatched new one-shot pair trial authorization')
    if not set(IMPLEMENTATIONS) <= set(auth.get('implementation_sha256', {})):
        raise ValueError('Missing critical implementation binding')


def read_bound(path, expected):
    path = Path(path)
    if path.resolve() != path or 'sealed_test' in path.parts or file_digest(path) != expected:
        raise ValueError('Changed/aliased/forbidden bound input: ' + str(path))
    return json.loads(path.read_text())


def authorize():
    auth = json.loads(AUTHORIZATION.read_text())
    validate_authorization(auth)
    for path, digest in auth['implementation_sha256'].items():
        if not (ROOT / path).resolve().is_relative_to(ROOT) or file_digest(ROOT / path) != digest:
            raise ValueError('Trial implementation changed: ' + path)
    manifest = read_bound(DATA / 'manifest.json', contract.MANIFEST_SHA)
    old = read_bound(DATA / 'admission.json', contract.ADMISSION_SHA)
    require_training_admission(old)
    if old.get('use') != 'local_selector_retention_fit_only' or old['training_run_authorized'] is not False:
        raise ValueError('Original data-only admission changed')
    proposal = read_bound(PREPARATION / 'proposal.json', PROPOSAL_SHA)
    preparation = read_bound(PREPARATION / 'report.json', PREPARATION_SHA)
    if not preparation['old_stage_equivalent_on_saved58']:
        raise ValueError('Missing pair supervision equivalence evidence')
    contract.validate_proposal(proposal, manifest['training_samples'])
    if file_digest(INITIAL) != contract.INITIAL_SHA:
        raise ValueError('Initial checkpoint changed')
    # This additive, exact-purpose extension never rewrites the historical
    # release. Its authority is the user's new request, not the draft protocol.
    admission = {**old, 'use': contract.USE, 'training_run_authorized': True,
        'execution_authorization_sha256': file_digest(AUTHORIZATION),
        'original_admission_sha256': contract.ADMISSION_SHA,
        'approved_protocol_sha256': PROPOSAL_SHA}
    return manifest, admission, auth


def reserve_once(path, authorization_sha256):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(dict(authorization_sha256=authorization_sha256, automatic_retries=0), stream)
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
        from .pair_only_worker import worker
        worker()
        return 0
    _, admission, _ = authorize()
    if args.check_only:
        print(json.dumps(dict(checks_passed=True, model_loaded=False, training_performed=False,
                              attempt_consumed=ATTEMPT.exists(), recipe=contract.RECIPE)))
        return 0
    if OUTPUT.exists():
        raise FileExistsError('Existing trial output; no automatic retry')
    reserve_once(ATTEMPT, file_digest(AUTHORIZATION))
    receipt = run_budgeted([sys.executable, '-m', __package__ + '.pair_only_trial', '--worker'],
        output=OUTPUT, budget=ResourceBudget(contract.RECIPE['wall_seconds'], ('0',)),
        training=True, admission=admission)
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
