"""Exact 24-image local fitting release; never changes the mixed-training schema.

Consumes prior identity receipts only. Copies 17 quarantine bundles using the
existing lifecycle transition, and 7 admitted controls byte-for-byte. The 16
old clean validation images remain validation-only references.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from . import pair_gap_diagnostic as fixed
from .admission import identities, require_training_admission
from .checkpoints import file_digest
from .clean_metrics import select_clean_rows
from .data import CurriculumCollator, load_sample
from .pair_gap_generation import bind_digest
from .pair_gap_metrics import validate_sample
from .publish import SUFFIXES, transition, write
from .release import CHECKS, verify_release
from .release_evidence import read, verify_reviews
from .validation_truth import expected_identity

DIAG = fixed.OUTPUT
DIAG_PLAN_SHA = '332881b77c744a0c49e4b46aca028c683a896e8290e2a757d7b2aa8ed8e86453'
DIAG_REPORT_SHA = '22ad8bc18ec710bd9ec96be21c0f0d81f5c3b5dcc7bf3d02487c6917ff28a689'
CLEAN = Path('outputs/audits/chemvector-clean16-diagnostic-20260918')
OUTPUT = Path('outputs/datasets/chemvector-presence-probe-release-20260918')
PLAN = Path('outputs/audits/chemvector-presence-release-plan-20260918.json')
RIGHTS = Path('docs/model/training/releases/20260918-presence-source-use.json')
REVIEW = Path('docs/model/training/releases/20260918-presence-promotion-review.json')
SCHEMA = 'chemvector-presence-probe-release-0.1'
USE = 'local_presence_readout_fit_only_not_mixed_training_or_deployment'


def disjoint_training_validation(train, validation):
    cache = {r['canonical_smiles']: identities(r['canonical_smiles']) for r in train + validation}
    for field in ('inchikey', 'connectivity_prefix', 'group_key'):
        if {cache[r['canonical_smiles']][field] for r in train} & {
                cache[r['canonical_smiles']][field] for r in validation}:
            raise ValueError('Presence fit overlaps existing validation: ' + field)
    if {r['molecule_id'] for r in train} & {r['molecule_id'] for r in validation}:
        raise ValueError('Presence fit reuses a validation molecule ID')


def build_plan():
    fixed.verify_bindings({str(DIAG / 'plan.json'): DIAG_PLAN_SHA, str(DIAG / 'report.json'): DIAG_REPORT_SHA,
                           str(fixed.PANEL): fixed.PANEL_SHA})
    diagnostic = json.loads((DIAG / 'plan.json').read_text())
    inputs = dict(diagnostic['input_sha256'])
    for p in (DIAG / 'plan.json', DIAG / 'report.json', fixed.PANEL):
        bind_digest(inputs, p, file_digest(p))
    fixed.verify_bindings(inputs)
    panel = json.loads(fixed.PANEL.read_text())
    evidence = {'request': fixed.REQUEST / 'request.json',
        'request_candidates': fixed.REQUEST / 'candidate-identities.json',
        'remaining': fixed.PUBLIC / 'review/remaining-candidates.json',
        'public_report': fixed.PUBLIC / 'review/report.json', 'private_receipt': fixed.PRIVATE_RECEIPT}
    raw = {k: p.read_bytes() for k, p in evidence.items()}
    docs = {k: json.loads(v) for k, v in raw.items()}
    receipt = verify_reviews(raw, docs)
    manifest, _, bindings = verify_release(fixed.RELEASE / 'manifest.json', fixed.RELEASE / 'admission.json')
    rows = fixed.fixed_rows(panel, docs['remaining'], manifest)
    for expected, saved in zip(rows, diagnostic['samples']):
        if any(saved.get(k) != v for k, v in expected.items()):
            raise ValueError('Fixed diagnostic selection changed')
    validation = [{**r, 'root': manifest['sources'][r['source']]} for r in select_clean_rows(manifest)]
    disjoint_training_validation(rows, [r for r in manifest['samples'] if r['split'] == 'validation'])
    rights = json.loads(RIGHTS.read_text())
    if (rights.get('schema_version') != 'chemvector-presence-source-use-0.1'
            or rights.get('use') != 'local_research_only' or rights.get('redistribution_authorized') is not False
            or rights.get('remaining_candidates_sha256') != file_digest(evidence['remaining'])
            or set(rights.get('sources', {})) != {'hand_authored', 'inherited_curriculum_controls', 'dejavu'}
            or any(not s.get('basis') or not s.get('attribution') for s in rights['sources'].values())):
        raise ValueError('Missing fixed-subset source-use review')
    fixed.verify_bindings(rights['input_sha256'])
    for name, digest in rights['input_sha256'].items():
        bind_digest(inputs, name, digest)
    for p in (RIGHTS, Path(__file__), Path(__file__).with_name('publish.py'),
              CLEAN / 'report.json', CLEAN / 'plan.json'):
        bind_digest(inputs, p, file_digest(p))
    collator = CurriculumCollator(purpose='diagnostic')
    artifacts = {}
    for row in rows + validation:
        key, root = row['sample_key'], Path(row['root'])
        meta_path = root / 'samples' / (key + '.meta.json')
        digest = file_digest(meta_path)
        if 'meta_sha256' in row and row['meta_sha256'] != digest:
            raise ValueError('Original admitted metadata changed')
        bind_digest(inputs, meta_path, digest)
        row['meta_sha256'] = digest
        payloads = {s: read((root / 'samples' / (key + s)).absolute()) for s in sorted(SUFFIXES)}
        artifacts[key] = {s: sha256(p).hexdigest() for s, p in payloads.items()}
        for s, value in artifacts[key].items():
            bind_digest(inputs, root / 'samples' / (key + s), value)
        sample = load_sample(root, key, purpose='diagnostic')
        if row in rows:
            validate_sample(sample, row)
        elif (sample['meta']['split'] != 'validation' or sample['meta']['training_authorized'] is not False
              or sample['meta']['external_overlap_status'] != 'checked'):
            raise ValueError('Validation controls must remain validation-only')
        batch = collator([sample])
        if batch['spatial_cell_collision_pairs'][0] or expected_identity(
                {'target_molblock': payloads['.sdf'].decode()}) != row['canonical_smiles']:
            raise ValueError('Target/SDF compatibility failed')
        old_case = (DIAG if row in rows else CLEAN) / (key + '.json')
        bind_digest(inputs, old_case, file_digest(old_case))
    fixed.verify_bindings(inputs)
    return {'schema_version': 'chemvector-presence-promotion-plan-0.1', 'use': USE,
        'samples': rows, 'validation_controls': validation, 'source_artifacts': artifacts,
        'input_sha256': inputs, 'inherited_release_bindings': bindings, 'private_review_status': receipt['status'],
        'release_checks': {k: 'checked' for k in CHECKS}, 'checked_images': 40,
        'training_images': 24, 'validation_images': 16, 'new_identity_queries': 0,
        'source_use_review_sha256': file_digest(RIGHTS), 'training_performed': False,
        'limitations': ['Readout-only local fitting probe; 24 fitting images are not held-out accuracy',
                       'Original 800-image development benchmark not evaluated by this limited probe',
                       'Existing evaluation internal group issue and 13 global public identity gaps remain',
                       'Private clearance inherited exactly once; no private queries or sealed sample access',
                       'No redistribution, default inference change, general mixed training or deployment']}


def publish():
    plan, review = json.loads(PLAN.read_text()), json.loads(REVIEW.read_text())
    plan_hash = file_digest(PLAN)
    if (review.get('schema_version') != 'chemvector-presence-promotion-review-0.1'
            or review.get('decision') != 'approve_local_presence_probe'
            or review.get('approved_plan_sha256') != plan_hash or not review.get('reviewer')
            or not review.get('rationale') or review.get('release_checks') != plan['release_checks']
            or review.get('source_use_review_sha256') != plan['source_use_review_sha256']):
        raise ValueError('Exact-plan publication review required')
    if plan != build_plan():
        raise ValueError('Reviewed publication inputs changed')
    output = OUTPUT.resolve()
    if any(p.is_symlink() for p in (OUTPUT, *OUTPUT.parents)):
        raise ValueError('Publication output must not be symlinked')
    if any(output.is_relative_to(Path(r['root']).resolve()) or Path(r['root']).resolve().is_relative_to(output)
           for r in plan['samples'] + plan['validation_controls']):
        raise ValueError('Publication must be independent of original sources')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'samples').mkdir()
    write(output / 'promotion-plan.json', plan)
    write(output / 'promotion-review.json', REVIEW.read_bytes())
    write(output / 'source-use-review.json', RIGHTS.read_bytes())
    rows, preserved = [], 0
    for original in plan['samples']:
        row = {**original, 'root': str(output), 'split': 'train', 'group_key': identities(original['canonical_smiles'])['group_key']}
        source = Path(original['root']) / 'samples'
        payloads = {s: read((source / (row['sample_key'] + s)).absolute()) for s in sorted(SUFFIXES)}
        if {s: sha256(v).hexdigest() for s, v in payloads.items()} != plan['source_artifacts'][row['sample_key']]:
            raise ValueError('Source bundle changed before copying')
        copied = transition(payloads, row, plan_sha256=plan_hash, original_root=original['root']) \
            if original['origin'] == 'new_quarantine' else payloads
        for suffix, value in copied.items():
            write(output / 'samples' / (row['sample_key'] + suffix), value)
            if value == payloads[suffix]:
                preserved += 1
        row['meta_sha256'] = sha256(copied['.meta.json']).hexdigest()
        sample = load_sample(output, row['sample_key'], purpose='training')
        CurriculumCollator(purpose='training')([sample])
        rows.append(row)
    manifest = {'schema_version': SCHEMA, 'use': USE, 'samples': rows,
                'validation_controls': plan['validation_controls'], 'promotion_plan_sha256': plan_hash}
    write(output / 'manifest.json', manifest)
    if plan != build_plan():
        raise ValueError('Original evidence changed during publication')
    admission = {'schema_version': 'chemvector-curriculum-admission-0.1', 'training_authorized': True,
        'external_overlap_status': 'checked', 'coverage_gaps': [], 'checked_images': 40,
        'approved_manifest_sha256': file_digest(output / 'manifest.json'), 'use': USE,
        'release_checks': plan['release_checks'], 'promotion_plan_sha256': plan_hash,
        'promotion_review_sha256': file_digest(REVIEW), 'training_performed': False,
        'byte_identical_copied_artifacts': preserved, 'limitations': plan['limitations']}
    write(output / 'admission.json', admission)
    try:
        verify()
    except Exception:
        (output / 'admission.json').rename(output / 'admission.failed.json')
        raise
    return admission


def verify():
    """Dedicated immutable release; normal mixed launcher deliberately rejects it."""
    admission = json.loads((OUTPUT / 'admission.json').read_text())
    require_training_admission(admission)
    manifest = json.loads((OUTPUT / 'manifest.json').read_text())
    plan = json.loads((OUTPUT / 'promotion-plan.json').read_text())
    review = json.loads((OUTPUT / 'promotion-review.json').read_text())
    if (manifest.get('schema_version') != SCHEMA or manifest.get('use') != USE or admission.get('use') != USE
            or admission['approved_manifest_sha256'] != file_digest(OUTPUT / 'manifest.json')
            or admission.get('release_checks') != {k: 'checked' for k in CHECKS}
            or admission.get('promotion_plan_sha256') != file_digest(OUTPUT / 'promotion-plan.json')
            or manifest.get('promotion_plan_sha256') != admission['promotion_plan_sha256']
            or admission.get('promotion_review_sha256') != file_digest(OUTPUT / 'promotion-review.json')
            or review.get('approved_plan_sha256') != admission['promotion_plan_sha256']
            or file_digest(OUTPUT / 'source-use-review.json') != plan['source_use_review_sha256']
            or plan != build_plan()):
        raise ValueError('Presence-only release evidence differs')
    expected = []
    for original in plan['samples']:
        row = {**original, 'root': str(OUTPUT.resolve()), 'split': 'train',
               'group_key': identities(original['canonical_smiles'])['group_key']}
        source = Path(original['root']) / 'samples'
        payloads = {s: read((source / (row['sample_key'] + s)).absolute()) for s in sorted(SUFFIXES)}
        copied = transition(payloads, row, plan_sha256=admission['promotion_plan_sha256'], original_root=original['root']) \
            if original['origin'] == 'new_quarantine' else payloads
        for suffix, value in copied.items():
            if read((OUTPUT / 'samples' / (row['sample_key'] + suffix)).absolute()) != value:
                raise ValueError('Published copy differs from exact permitted transition')
        row['meta_sha256'] = sha256(copied['.meta.json']).hexdigest()
        expected.append(row)
    if (manifest['samples'] != expected or manifest['validation_controls'] != plan['validation_controls']
            or len(expected) != 24 or len(manifest['validation_controls']) != 16 or admission['checked_images'] != 40):
        raise ValueError('Presence train/validation partition changed')
    return manifest, admission


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'publish', 'check'))
    args = parser.parse_args(argv)
    if args.action == 'plan':
        write(PLAN, build_plan())
        print(json.dumps({'plan_sha256': file_digest(PLAN)}))
    elif args.action == 'publish':
        print(json.dumps(publish(), indent=2))
    else:
        manifest, _ = verify()
        print(json.dumps({'train': len(manifest['samples']), 'validation': len(manifest['validation_controls'])}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
