"""One-shot paired trial gate. No command creates training authorization."""

import argparse
import json
from pathlib import Path
import sys

from . import pair_only_trial as old, pair_anchor_ab_contract as rules
from .admission import require_training_admission
from .checkpoints import file_digest
from .pair_cached_trial import json_digest
from .runtime import ResourceBudget, run_budgeted

ROOT, INITIAL, DATA = old.ROOT, old.INITIAL, old.DATA
PREPARATION = ROOT / 'outputs/audits/chemvector-pair-anchor-ab-preparation-20260924'
OUTPUT = ROOT / 'outputs/experiments/chemvector-pair-anchor-ab-20260924'
AUTHORIZATION = ROOT / 'docs/model/training/releases/20260924-pair-anchor-ab-authorization.json'
ATTEMPT = ROOT / 'outputs/experiments/chemvector-pair-anchor-ab-attempt-20260924.json'
WIRING = ROOT / 'outputs/audits/chemvector-pair-anchor-wiring58-20260924'
WIRING_SHA = '598388de9b4e40c7ac4e8f414b403394a685523ca9b39271a0e81433052765fc'
WIRING_POST = WIRING.with_name(WIRING.name + '-postcheck') / 'report.json'
WIRING_POST_SHA = 'cc7b34ecad8435ae34e23707d2c6108e1a6aa0d2b55e1320bc799257dfa0e706'
IMPLEMENTATIONS = tuple(dict.fromkeys((*old.IMPLEMENTATIONS,
    'chemvector/training/curriculum/pair_anchor_augmentation.py',
    'chemvector/training/curriculum/pair_anchor_wiring.py',
    'chemvector/training/curriculum/pair_anchor_device.py',
    'chemvector/training/curriculum/pair_anchor_ab_contract.py',
    'chemvector/training/curriculum/pair_anchor_ab_trial.py',
    'chemvector/training/curriculum/pair_anchor_ab_worker.py',
    'chemvector/components/atoms/chemistry/direct_atom_pair_chemistry.py',
    'chemvector/direct_scene_inference.py',
    'chemvector/evaluation/shadows/shadow_chemscene_v02_oracle.py',
    'chemvector/components/atoms/objects/direct_atom_role.py',
    'chemvector/components/ocr/ctc_ocr.py',
    'chemvector/evaluation/evaluators/ocr/evaluate_direct_atom_ocr.py',
    'Data/sdf/chemscene_v0_2.py',
    'chemvector/placeholder_scene.py',
    'tools/prepare_pair_anchor_ab_trial.py',
    'tests/chemvector/test_pair_anchor_ab.py',
    'tests/chemvector/test_pair_anchor_device.py',
    'docs/model/training/20260924-pair-anchor-ab-protocol.md')))


def source_checks():
    manifest = old.read_bound(DATA / 'manifest.json', rules.original.MANIFEST_SHA)
    admission = old.read_bound(DATA / 'admission.json', rules.original.ADMISSION_SHA)
    require_training_admission(admission)
    rules.original.make_proposal(manifest['training_samples'])
    if admission['use'] != 'local_selector_retention_fit_only' or admission['training_run_authorized'] is not False:
        raise ValueError('Historical data-only admission changed')
    report = old.read_bound(WIRING / 'report.json', WIRING_SHA)
    checked = old.read_bound(WIRING_POST, WIRING_POST_SHA)
    if not checked['passed'] or checked['source_report_sha256'] != WIRING_SHA:
        raise ValueError('Completed wiring postcheck required')
    for path, digest in report['input_sha256'].items():
        if file_digest(path) != digest: raise ValueError('Wiring source changed: ' + path)
    for name, digest in report['artifact_sha256'].items():
        if Path(name).name != name or file_digest(WIRING / name) != digest:
            raise ValueError('Wiring artifact changed')
    if file_digest(INITIAL) != rules.original.INITIAL_SHA:
        raise ValueError('Initial checkpoint changed')
    return manifest, admission


def validate_authorization(auth):
    expected = dict(schema_version='pair-anchor-ab-authorization-0.1', use=rules.USE,
        protocol_sha256=json_digest(rules.protocol()), manifest_sha256=rules.original.MANIFEST_SHA,
        initial_sha256=rules.original.INITIAL_SHA, original_admission_sha256=rules.original.ADMISSION_SHA,
        training_authorized=True, deployment_authorized=False, maximum_invocations=1,
        output=str(OUTPUT.relative_to(ROOT)), limits=rules.LIMITS,
        data_use_extension=dict(previous_use='local_selector_retention_fit_only', approved_use=rules.USE,
            training_images=58, new_data=False, validation_sample_artifacts=False, identity_queries=0))
    if (any(json_digest(auth.get(k)) != json_digest(v) for k, v in expected.items())
            or not isinstance(auth.get('user_statement'), str) or not auth['user_statement'].strip()
            or set(auth.get('implementation_sha256', {})) != set(IMPLEMENTATIONS)):
        raise ValueError('New exact two-arm purpose and run authorization required')
    for digest in [auth.get('preparation_report_sha256'), *auth['implementation_sha256'].values()]:
        if not isinstance(digest, str) or len(digest) != 64 or set(digest) - set('0123456789abcdef'):
            raise ValueError('Full preparation/implementation hash binding required')


def authorize():
    auth = json.loads(AUTHORIZATION.read_text())
    validate_authorization(auth)
    preparation = old.read_bound(PREPARATION / 'report.json', auth['preparation_report_sha256'])
    if (not preparation['synthetic_tests_passed'] or preparation['protocol_sha256'] != auth['protocol_sha256']
            or preparation['training_authorized'] is not False
            or preparation['implementation_sha256'] != auth['implementation_sha256']):
        raise ValueError('Preparation differs from the new grant')
    for path, digest in auth['implementation_sha256'].items():
        if file_digest(ROOT / path) != digest: raise ValueError('Bound implementation changed: ' + path)
    for name, digest in preparation['artifact_sha256'].items():
        if Path(name).name != name or file_digest(PREPARATION / name) != digest:
            raise ValueError('Preparation artifact changed')
    manifest, admission = source_checks()
    return manifest, dict(admission, use=rules.USE, training_run_authorized=True,
        execution_authorization_sha256=file_digest(AUTHORIZATION)), auth


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--check-only', action='store_true')
    group.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.check_only:
        manifest, _ = source_checks()
        print(json.dumps(dict(source_checks_passed=True, training_rows=len(manifest['training_samples']),
            authorization_present=AUTHORIZATION.exists(), attempt_consumed=ATTEMPT.exists(),
            gpu_used=False, training_performed=False, protocol=rules.protocol())))
        return 0
    if args.worker:
        from .pair_anchor_ab_worker import worker
        worker(); return 0
    _, admission, _ = authorize()
    if OUTPUT.exists(): raise FileExistsError('Existing paired trial; no overwrite/retry')
    old.reserve_once(ATTEMPT, file_digest(AUTHORIZATION))
    receipt = run_budgeted([sys.executable, '-m', __package__ + '.pair_anchor_ab_trial', '--worker'],
        output=OUTPUT, budget=ResourceBudget(rules.LIMITS['wall_seconds'], ('0',)), training=True, admission=admission)
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__': raise SystemExit(main())
