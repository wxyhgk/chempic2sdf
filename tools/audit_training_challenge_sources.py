"""Bounded training-metadata exposure audit; no pictures or private queries."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

from rdkit import Chem, rdBase

from chemvector.experiments import training_exposure as exposure
from chemvector.training.curriculum import admission, runtime

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'outputs/audits/chemvector-training-challenge-sources-20260921'
EXPANDED = 'outputs/datasets/chemvector-expanded-release-20260915'
FIXED = 'outputs/datasets/chemvector-selector-retention-fit-20260919'
HISTORICAL = {
    'chemvector-mixed-continuation-20260915': ('127d5665dee8cb797e83dd85c2dae49592fdc0324649c5b076e2704c68913411', None),
    'chemvector-selected-matching-trial-20260915': ('3da86a51a7ea54569066e3027c6d88c5d6fc7acf59c55496b009e1f232c5dd6b',
        'bf525a124aafc942c3bec19a19fd8d190a314705874e73e99b0eeaf73504d478'),
    'chemvector-expanded-freeze-role-20260915': ('0c645f70fc064d2c22b50f6314adc7de950e21593c603dfa85b56c7296366555',
        '37b17610a198a3da792ebce271e572eb1e47098f0f6b6c0fd105cd2a02b312df'),
}


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def worker():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or not OUTPUT.is_dir():
        raise ValueError('Budgeted CPU metadata audit required')
    hashes = {}

    def read(relative, expected=None, *, lines=False):
        path = ROOT / relative
        payload = path.read_bytes(); digest = sha256(payload).hexdigest()
        if expected is not None and expected != digest:
            raise ValueError('Historical metadata hash changed: ' + str(path))
        hashes[str(path)] = digest
        return [json.loads(line) for line in payload.splitlines() if line.strip()] if lines else json.loads(payload)

    manifest = read(EXPANDED + '/manifest.json', '67ccb1307f7af3ca19f16ee6fd900d1a2e8c3c62dd0b9c0ec28704ddd93aa05e')
    release = read(EXPANDED + '/admission.json')
    publication = read(EXPANDED + '/publication-report.json')
    if (release['approved_manifest_sha256'] != hashes[str(ROOT / EXPANDED / 'manifest.json')]
            or release['external_overlap_status'] != 'checked' or release['coverage_gaps']
            or not publication['validation_changed'] is False):
        raise ValueError('Inherited publication/separation evidence changed')
    # Only train rows enter local identity grouping; do not inspect dev structures.
    rows = [r for r in manifest['samples'] if r['split'] == 'train']
    fixed = read(FIXED + '/manifest.json', 'f5e3a7163eef5c49118ad78dcf05778f4ad889dc9b6389097ef20896b79e2111')
    design = [dict(r, evidence_id='fixed58_repeated_rule_design') for r in fixed['training_samples']]
    for name, digest in [('chemvector-initial-headroom32-20260918',
                         '2b5aedf18a4ab18613c19ca33a658a3471aaec2dff4cda73fa913ff5ce4d2aac'),
                        ('chemvector-pair-gap-diagnostic24-20260918',
                         '332881b77c744a0c49e4b46aca028c683a896e8290e2a757d7b2aa8ed8e86453')]:
        plan = read('outputs/audits/' + name + '/plan.json', digest)
        for row in plan['samples']:
            if row.get('split') not in (None, 'train'):
                raise ValueError('Development rows are outside this ledger')
            design.append(dict(row, evidence_id=name))
    updates, histories = defaultdict(set), []
    for name, (report_sha, steps_sha) in HISTORICAL.items():
        base = 'outputs/experiments/' + name
        report = read(base + '/report.json', report_sha)
        steps = read(base + '/steps.jsonl', steps_sha, lines=True)
        if ([r['optimizer_steps'] for r in steps] != list(range(1, report['optimizer_steps'] + 1))
                or any(len(r['keys']) != 1 for r in steps)):
            raise ValueError('Historical single-image update sequence incomplete')
        keys = {r['keys'][0] for r in steps}
        for key in keys: updates[key].add(name)
        histories.append(dict(run=name, optimizer_steps=len(steps), distinct_updated_rows=len(keys),
            steps_historical_sha_available=steps_sha is not None,
            note='Mixed-continuation steps digest frozen at this audit' if steps_sha is None else 'Prior steps hash checked'))
    v02 = read('outputs/audits/curriculum-v02-release-preparation-20260919/report.json',
               '39e66a3a4b0ba17651a5834fe3ac65fa8661fef247df98f58195f9e6fe496310')
    if v02['dataset_published'] is not False:
        raise ValueError('Unpublished v02 proposal status changed; re-scope explicitly')
    for path in (Path(__file__), Path(exposure.__file__), Path(admission.__file__), Path(runtime.__file__),
                 ROOT / 'tests/chemvector/test_training_exposure.py'):
        hashes[str(path)] = sha256(path.read_bytes()).hexdigest()
    write(OUTPUT / 'plan.json', dict(input_sha256=hashes, cpu_wall_seconds=60,
        selection_limit=128, selection_seed='20260921-training-challenge-v1',
        selection_rule='No selection unless design-unexposed history can be established; unknown excluded',
        grouping='transitive molecule ID / declared and computed core group / connectivity-prefix union',
        images_read=0, validation_sample_artifacts_read=False, new_identity_queries=0,
        new_data_generation=False, real_model_forwards=0, optimizer_steps=0))
    ledger = exposure.build_ledger(rows, design, updates)
    write(OUTPUT / 'exposure-ledger.json', ledger)
    molecular = {}
    for row in rows:
        smiles = row['canonical_smiles']
        if smiles not in molecular:
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None: raise ValueError('Invalid published training structure')
            molecular[smiles] = dict(multifragment=len(Chem.GetMolFrags(molecule)) > 1,
                atoms=molecule.GetNumAtoms(), bond_types=sorted({str(b.GetBondType()) for b in molecule.GetBonds()}))
    coverage = dict(training_rows=len(rows),
        distinct_molecule_ids=len({r['molecule_id'] for r in rows}), declared_groups=len({r['group_key'] for r in rows}),
        by_source=dict(Counter(r['source'] for r in rows)),
        by_style=dict(Counter(r['sample_key'].rsplit('__', 1)[-1] for r in rows)),
        reference_style_rows=sum(r['sample_key'].endswith('__reference') for r in rows),
        visual_cleanliness_verified=False,
        multifragment_rows=sum(molecular[r['canonical_smiles']]['multifragment'] for r in rows),
        multifragment_structures=sum(m['multifragment'] for m in molecular.values()),
        structures_with_bond_type=dict(Counter(t for m in molecular.values() for t in m['bond_types'])),
        distinct_png_inherited=publication['unique_train_images'])
    for path, digest in hashes.items():
        if sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError('Metadata source changed during audit')
    report = dict(schema_version='training-challenge-source-audit-0.1', coverage=coverage,
        exposure=ledger['counts'], audited_update_histories=histories,
        distinct_rows_in_audited_update_logs=len(updates), diagnostic_exclusion_records=len(design),
        challenge_ready=False, selected_challenge_images=0, new_challenge_manifest_published=False,
        history_complete=False, exposure_not_found_means='unknown_not_unseen',
        excluded_pools=dict(old944='development/model-selection; not a new test',
            new64='already observed in retention/full-loss/node-semantics; not new',
            v02_proposals=dict(published=False, train_candidates=v02['image_partition']['train'],
                               validation_candidates=v02['image_partition']['validation'])),
        gaps=['No complete per-sample catalog of all historical design/diagnostic observation',
              'Initial component pretraining exposure not reconstructed; never claim model-unseen',
              'Only the named metadata plans and three training histories audited; counts are lower bounds',
              'No new visual inspection; reference preset cannot certify clean visual examples'],
        next_action='Finish provenance-only exposure catalog before fixing <=128 training-side challenge groups',
        rdkit_version=rdBase.rdkitVersion, input_sha256=hashes,
        artifact_sha256={name: sha256((OUTPUT / name).read_bytes()).hexdigest()
                         for name in ('plan.json', 'exposure-ledger.json')},
        model_forwards=0, checkpoint_loads=0, optimizer_steps=0, training_performed=False,
        gpu_used=False, images_read=0, validation_sample_artifacts_read=False, sealed_test_read=False,
        private_identity_queries=0, local_training_smiles_grouping_only=True,
        new_data_generated=False, original_outputs_changed=False, training_authorized=False)
    write(OUTPUT / 'report.json', report)
    print(json.dumps(dict(coverage=coverage, exposure=ledger['counts'], challenge_ready=False), indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args(argv)
    if args.worker:
        worker(); return 0
    receipt = runtime.run_budgeted([sys.executable, '-m', 'tools.audit_training_challenge_sources', '--worker'],
                                  output=OUTPUT, budget=runtime.ResourceBudget(60))
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
