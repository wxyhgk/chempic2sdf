"""Gated one-pass pair training entry. Preparation/check-only never authorizes GPU."""

import argparse
import json
import sys

from . import pair_anchor_ab_trial as prior
from .checkpoints import file_digest
from .interface_diagnostic import write_json
from .pair_scale_plan import digest, draft_protocol, exposure_plan, sample_id
from .runtime import ResourceBudget, run_budgeted
from tools import prepare_pair_scale_training as prep

ROOT = prep.ROOT
PREPARATION = prep.OUTPUT
READINESS = ROOT / 'outputs/audits/chemvector-pair-scale-readiness-20260924'
AUTHORIZATION = ROOT / 'docs/model/training/releases/20260924-pair-scale-authorization.json'
OUTPUT = ROOT / 'outputs/experiments/chemvector-pair-scale-20260924'
ATTEMPT = ROOT / 'outputs/experiments/chemvector-pair-scale-attempt-20260924.json'
USE = 'local_pair_scale_unique12985_once'
IMPLEMENTATIONS = tuple(dict.fromkeys((*prior.IMPLEMENTATIONS, *prep.IMPLEMENTATIONS,
    'chemvector/training/curriculum/release.py', 'chemvector/training/curriculum/fresh_dev.py',
    'chemvector/training/curriculum/pair_scale_training.py',
    'chemvector/training/curriculum/pair_scale_worker.py',
    'tests/chemvector/test_pair_scale_runtime.py', 'tools/check_pair_scale_readiness.py')))


def execution_contract(protocol):
    return dict(schema_version='pair-scale-execution-0.1', use=USE, seed=20260924,
        protocol_sha256=digest(protocol), cache_mode='stream_one_training_image_no_feature_corpus',
        profiling='64_fixed_unique_training_images_forward_backward_without_optimizer; measured_p95_cost',
        projected_cost='1.5*p95_or_mean_train_time*attempts + 1.25*4*baseline_dev_time +120_seconds',
        updates='one_single_pass_only; initial_checkpoint; no_augmentation; no_automatic_resume',
        validation_media='existing944_only_with_new_authorization; not_sealed_test',
        safety='fixed_stopping_policy_plus_finite_gradients_and_frozen_state_checks',
        maximum_training_attempts=protocol['proposed_budget']['maximum_attempts'],
        maximum_upstream_forwards=protocol['proposed_budget']['maximum_attempts']+64+5*944,
        deployment_authorized=False)


def source_checks():
    report = json.loads((PREPARATION / 'report.json').read_text())
    if not report['complete'] or report['training_authorized'] or report['unique_images'] != 12985:
        raise ValueError('Complete non-authorizing scale preparation required')
    for name, value in report['artifact_sha256'].items():
        if '/' in name or file_digest(PREPARATION/name) != value:
            raise ValueError('Prepared artifact changed')
    for path, value in report['implementation_sha256'].items():
        if file_digest(ROOT/path) != value:
            raise ValueError('Prepared implementation changed')
    manifest = json.loads(prep.bound_bytes(prep.RELEASE/'manifest.json', prep.MANIFEST_SHA))
    admission = json.loads(prep.bound_bytes(prep.RELEASE/'admission.json', prep.ADMISSION_SHA))
    prep.require_training_admission(admission)
    if file_digest(prep.INITIAL) != prep.INITIAL_SHA:
        raise ValueError('Initial checkpoint changed')
    rows = json.loads((PREPARATION/'unique-training-rows.json').read_text())['records']
    schedule_doc = json.loads((PREPARATION/'schedule.json').read_text())
    schedule = schedule_doc['schedule']
    protocol = json.loads((PREPARATION/'protocol-draft.json').read_text())
    original = {sample_id(r): r for r in manifest['samples']}
    if (len(rows) != 12985 or digest(rows) != schedule_doc['rows_sha256']
            or schedule != exposure_plan(rows) or protocol != draft_protocol(len(schedule))):
        raise ValueError('Exposure plan changed')
    for row in rows:
        source = original[sample_id(row)]
        if source['split'] != 'train' or any(row[k] != v for k, v in source.items()):
            raise ValueError('Only original train references may be scheduled')
    dev = json.loads((PREPARATION/'development-reference.json').read_text())
    if dev['records'] != [r for r in manifest['samples'] if r['split']=='validation']:
        raise ValueError('Fixed development selection changed')
    return dict(manifest=manifest, admission=admission, rows=rows, schedule=schedule, protocol=protocol,
                preparation_report_sha256=file_digest(PREPARATION/'report.json'))


def expected_authorization(sources):
    return dict(schema_version='pair-scale-authorization-0.1', use=USE,
        training_authorized=True, deployment_authorized=False, maximum_invocations=1,
        preparation_report_sha256=sources['preparation_report_sha256'],
        manifest_sha256=prep.MANIFEST_SHA, admission_sha256=prep.ADMISSION_SHA, initial_sha256=prep.INITIAL_SHA,
        protocol_sha256=digest(sources['protocol']), execution_contract=execution_contract(sources['protocol']),
        budget=sources['protocol']['proposed_budget'], output=str(OUTPUT.relative_to(ROOT)),
        data_use_extension=dict(previous_use='local_research_only', approved_use=USE,
            distinct_training_images=12985, development_images=944, development_media_read=True,
            new_data=False, sealed_test_read=False, identity_queries=0))


def validate_authorization(auth, sources):
    if any(digest(auth.get(k)) != digest(v) for k, v in expected_authorization(sources).items()):
        raise ValueError('New exact full-pass authorization required')
    if (not isinstance(auth.get('user_statement'), str) or not auth['user_statement'].strip()
            or set(auth.get('implementation_sha256', {})) != set(IMPLEMENTATIONS)):
        raise ValueError('Explicit statement and complete implementation bindings required')
    for value in [auth.get('readiness_report_sha256'), *auth['implementation_sha256'].values()]:
        if not isinstance(value,str) or len(value)!=64 or set(value)-set('0123456789abcdef'):
            raise ValueError('Full readiness and implementation digests required')


def authorize():
    auth = json.loads(AUTHORIZATION.read_text())  # Deny before corpus checks/model/CUDA.
    sources = source_checks()
    validate_authorization(auth,sources)
    ready = json.loads(prep.bound_bytes(READINESS/'report.json',auth['readiness_report_sha256']))
    if (not ready['tests_passed'] or ready['preparation_report_sha256'] != sources['preparation_report_sha256']
            or ready['implementation_sha256'] != auth['implementation_sha256']
            or ready['training_authorized'] is not False):
        raise ValueError('Readiness does not match grant')
    for path,value in auth['implementation_sha256'].items():
        if file_digest(ROOT/path)!=value:
            raise ValueError('Bound implementation changed: '+path)
    for name,value in ready['artifact_sha256'].items():
        if '/' in name or file_digest(READINESS/name)!=value:
            raise ValueError('Readiness evidence changed')
    return sources,auth


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    flags=parser.add_mutually_exclusive_group()
    flags.add_argument('--check-only',action='store_true')
    flags.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args.check_only:
        data=source_checks()
        print(json.dumps(dict(checks_passed=True,distinct_images=len(data['rows']),
            attempts=len(data['schedule']),training_performed=False,authorization_present=AUTHORIZATION.exists())))
        return 0
    if args.worker:
        from .pair_scale_worker import worker
        worker()
        return 0
    sources,_=authorize()
    if OUTPUT.exists():
        raise FileExistsError('No overwrite or retry of scale run')
    prior.old.reserve_once(ATTEMPT,file_digest(AUTHORIZATION))
    receipt=run_budgeted([sys.executable,'-m',__package__+'.pair_scale_training','--worker'],
        output=OUTPUT,budget=ResourceBudget(3600,('0',)),training=True,
        admission=dict(sources['admission'],use=USE,training_run_authorized=True))
    print(json.dumps(receipt,indent=2))
    return 0 if receipt['status']=='completed' else 1


if __name__=='__main__':
    raise SystemExit(main())
