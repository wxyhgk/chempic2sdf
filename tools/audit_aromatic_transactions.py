"""Bounded CPU-only saved-prediction aromatic decoder experiment; no deployment."""

import argparse
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

from rdkit import Chem
import torch

from Data.sdf.chemscene_v0_2 import mol_from_chemscene_v0_2
from chemvector.components.atoms.chemistry.aromatic_transactions import guarded_promotions, scene_digest
from chemvector.evaluation.audits.aromatic_saved_replay import reconstruct
from chemvector.placeholder_scene import compile_carbon_placeholder
from chemvector.training.curriculum import runtime, pair_trial_contract as rules
from chemvector.training.curriculum.interface_diagnostic import write_json
from chemvector.training.curriculum.validation_truth import canonical_molblock

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'outputs/experiments/chemvector-pair-replay-trial-20260922'
CONTROL = ROOT / 'outputs/experiments/chemvector-pair-observed-trial-20260922'
OUTPUT = ROOT / 'outputs/audits/chemvector-aromatic-transactions-20260922'
MANIFEST = ROOT / 'outputs/datasets/chemvector-selector-retention-fit-20260919/manifest.json'
RUN_SHA = 'b7a6c66c9fabc2a346edf13a2552f97f9fae3c4656be8ed680ca328838b93aae'
CONTROL_SHA = 'c828baa4ada80200c98add2e6058abded6a21cbf3871bd5611a66cd71a2aee9f'
POSTCHECK = ROOT / 'outputs/audits/chemvector-pair-replay-postcheck-20260922/report.json'
POSTCHECK_SHA = '1c5e8b73d1e77ec0297e2417178540afbaf2601722ba88457b4c797e207d4bb8'
TESTS = ('tests/chemvector/test_aromatic_transactions.py', 'tests/chemvector/test_direct_atom_pair_chemistry.py',
         'tests/chemvector/test_shadow_chemscene_v02_oracle.py', 'tests/chemvector/test_curriculum_evaluation_input.py')
IMPLEMENTATIONS = (
    'chemvector/components/atoms/chemistry/aromatic_transactions.py',
    'chemvector/evaluation/audits/aromatic_saved_replay.py', 'tools/audit_aromatic_transactions.py',
    'chemvector/components/atoms/chemistry/direct_atom_pair_chemistry.py',
    'chemvector/evaluation/shadows/shadow_chemscene_v02_oracle.py', 'chemvector/placeholder_scene.py',
    'chemvector/direct_scene_inference.py', 'chemvector/training/curriculum/evaluation.py',
    'chemvector/training/curriculum/runtime.py', 'chemvector/training/curriculum/validation_truth.py',
    'chemvector/training/curriculum/interface_diagnostic.py', 'chemvector/training/curriculum/pair_trial_contract.py',
    'Data/sdf/chemscene_v0_1.py', 'Data/sdf/chemscene_v0_2.py', *TESTS)


def digest(path): return sha256(Path(path).read_bytes()).hexdigest()


class ChemistryBudget:
    def __init__(self, maximum=4096): self.maximum, self.calls = maximum, 0

    def take(self):
        if self.calls >= self.maximum: raise TimeoutError('Chemistry compilation budget exhausted')
        self.calls += 1

    def compile(self, scene):
        self.take()
        return mol_from_chemscene_v0_2(deepcopy(scene))


def legacy_prediction(scene, budget):
    """Reproduce the historical success/fallback/error semantics exactly."""
    diagnostics = dict(global_fallback_attempted=False, global_demoted_bonds=0)
    try:
        fallback = False
        try:
            molecule = budget.compile(scene)
        except (ValueError, RuntimeError) as error:
            diagnostics.update(global_fallback_attempted=True, sanitize_error=str(error))
            budget.take()
            _, molecule, demoted = compile_carbon_placeholder(deepcopy(scene))
            diagnostics['global_demoted_bonds'] = demoted; fallback = True
        prediction = dict(molblock=Chem.MolToMolBlock(molecule), fallback=fallback, error=None)
    except (ValueError, RuntimeError) as error:
        prediction = dict(molblock=None, fallback=False, error=str(error))
    return prediction, diagnostics


def compare_case(case, budget):
    """Only the outer scoring code sees expected identity and training metrics."""
    t = case['trace']; saved = t['scene_before_fallback']
    raw, proposals, reconstruction = reconstruct(saved, t['compiled_candidate_indices'],
        torch.tensor(case['pairs'], dtype=torch.long).reshape(-1, 2),
        torch.tensor(case['presence_logits'], dtype=torch.float32),
        torch.tensor(case['chemical_logits'], dtype=torch.float32).reshape(-1, 4))
    legacy, legacy_details = legacy_prediction(saved, budget)
    if legacy != case['prediction']: raise ValueError('Saved legacy prediction was not exactly reproduced')
    if reconstruction['use_legacy']:
        guarded = deepcopy(saved)
        journal = dict(status='unmapped_proposal_use_legacy', use_legacy=True, local_rollback=False, transactions=[])
    else:
        guarded, journal = guarded_promotions(raw, proposals, compiler=budget.compile)
    if journal['use_legacy']:
        guarded = deepcopy(saved); prediction = deepcopy(legacy)
        guarded_details = deepcopy(legacy_details)
    else:
        # Every committed transaction is already validated; unexpected failure is
        # an implementation error, not permission for a silent global fallback.
        molecule = budget.compile(guarded)
        prediction = dict(molblock=Chem.MolToMolBlock(molecule), fallback=False, error=None)
        guarded_details = dict(global_fallback_attempted=False, global_demoted_bonds=0)
    rollback_ids = {b for r in journal['transactions'] if r['action'] == 'rolled_back' for b in r['bond_ids']}
    changes = []
    if guarded['atom_objects'] != saved['atom_objects']: raise ValueError('Saved atom/OCR properties changed')
    for a, b in zip(saved['bond_objects'], guarded['bond_objects'], strict=True):
        if a != b:
            if a['id'] not in rollback_ids or any(a[k] != b[k] for k in a if k not in ('chemical_type', 'display_type')):
                raise ValueError('Edit escaped rejected aromatic transactions')
            changes.append(dict(bond_id=a['id'], atom_refs=a['atom_refs'],
                                before={k: a[k] for k in ('chemical_type', 'display_type')},
                                after={k: b[k] for k in ('chemical_type', 'display_type')}))
    # Expected identity is introduced AFTER decoding, and never enters a proposal.
    expected = case['metrics']['expected_smiles']
    def score(prediction, details):
        identity = canonical_molblock(prediction['molblock'])
        return dict(sdf_valid=identity is not None, sdf_exact=identity is not None and identity == expected,
                    fallback=prediction['fallback'], predicted_smiles=identity, **details)
    old, new = score(legacy, legacy_details), score(prediction, guarded_details)
    if any(old[k] != case['metrics'][k] for k in ('sdf_valid', 'sdf_exact', 'fallback', 'predicted_smiles')):
        raise ValueError('Legacy saved metric identity mismatch')
    unaffected_aromatic = [b['id'] for b in saved['bond_objects']
                          if b['chemical_type'] == 'AROMATIC' and b['id'] not in rollback_ids]
    return dict(legacy=old, guarded=new, journal=journal, reconstruction=reconstruction, proposals=proposals,
        changed_bonds=changes, unchanged_atom_attributes=True, unchanged_connectivity=True,
        outside_rejected_aromatic_bond_ids_preserved=unaffected_aromatic,
        any_correction_warning=journal['local_rollback'] or new['global_fallback_attempted'],
        legacy_scene_sha256=scene_digest(saved), guarded_scene_sha256=scene_digest(guarded),
        guarded_scene=guarded, prediction=prediction, expected_smiles=expected,
        raw_pair_predictions_and_original_training_gate_unchanged=True)


def summarize(records):
    summary = {}
    for arm in ('initial', 'control50', 'replay50'):
        arm_records = [r for r in records if r['arm'] == arm]
        if len(arm_records) != 58: raise ValueError('All58 must remain in each comparison')
        groups = {}
        for group in ('all', 'witness', 'old_full_molecule'):
            subset = [r for r in arm_records if group == 'all' or r['group'] == group]
            protections = [r['key'] for r in subset if (r['legacy']['sdf_exact'] and not r['guarded']['sdf_exact'])
                or (r['legacy']['sdf_valid'] and not r['guarded']['sdf_valid'])
                or (not r['legacy']['global_fallback_attempted'] and r['guarded']['global_fallback_attempted'])]
            groups[group] = dict(images=len(subset),
                legacy={k: sum(r['legacy'][k] for r in subset) for k in
                    ('sdf_exact', 'sdf_valid', 'fallback', 'global_fallback_attempted')},
                guarded={k: sum(r['guarded'][k] for r in subset) for k in
                    ('sdf_exact', 'sdf_valid', 'fallback', 'global_fallback_attempted')},
                local_rollback_images=sum(r['journal']['local_rollback'] for r in subset),
                any_correction_warning_images=sum(r['any_correction_warning'] for r in subset),
                use_legacy_images=sum(r['journal']['use_legacy'] for r in subset), protection_failures=protections)
        summary[arm] = groups
    return summary


def worker():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('Saved-prediction CPU worker required')
    torch.set_num_threads(2)
    hashes, artifacts = {}, {}
    def read_bound(path, expected):
        if digest(path) != expected: raise ValueError('Bound source changed: ' + str(path))
        hashes[str(path)] = expected
        return json.loads(path.read_bytes())
    def write(name, value):
        write_json(OUTPUT / name, value); artifacts[name] = digest(OUTPUT / name)
    report = read_bound(RUN / 'report.json', RUN_SHA)
    control = read_bound(CONTROL / 'report.json', CONTROL_SHA)
    postcheck = read_bound(POSTCHECK, POSTCHECK_SHA)
    manifest = read_bound(MANIFEST, rules.MANIFEST_SHA); rows = manifest['training_samples']
    rules.make_proposal(rows)
    if not postcheck['checks_passed'] or postcheck['report_sha256'] != RUN_SHA:
        raise ValueError('Verified saved comparison required')
    implementations = {p: digest(ROOT / p) for p in IMPLEMENTATIONS}
    for p, expected in report['implementation_sha256'].items():
        if digest(ROOT / p) != expected: raise ValueError('Historical training implementation changed')
        hashes[str(ROOT / p)] = expected
    write('plan.json', dict(schema_version='aromatic-transaction-plan-0.1',
        data_scope='same_fixed58_train_predictions_only', arms=['initial', 'control50', 'replay50'],
        cases=174, gpu_devices=[], wall_seconds=120, maximum_chemistry_compiles=4096,
        thresholds=dict(presence=.85, aromatic_mean=.40, aromatic_minimum=.05),
        rule='after_OCR; validate_raw; overlap_atom_groups_atomic; failed_transaction_restores_previous_state',
        preexisting_invalid='record_and_keep_legacy_path; no_search',
        no_deployment=True, no_training=True, no_model_or_OCR_forward=True,
        implementation_sha256=implementations))
    test = subprocess.run([sys.executable, '-m', 'pytest', '-q', *TESTS,
        '--junitxml=' + str(OUTPUT / 'synthetic-tests.xml')], cwd=ROOT, timeout=45, check=False)
    suites = ET.parse(OUTPUT / 'synthetic-tests.xml').getroot().findall('testsuite')
    totals = {k: sum(int(s.attrib.get(k, 0)) for s in suites) for k in ('tests', 'failures', 'errors', 'skipped')}
    if test.returncode or not totals['tests'] or any(totals[k] for k in ('failures', 'errors', 'skipped')):
        raise ValueError('Synthetic regression tests did not pass')
    artifacts['synthetic-tests.xml'] = digest(OUTPUT / 'synthetic-tests.xml')
    budget, records = ChemistryBudget(), []
    for arm, directory, run_report, label in [('initial', RUN, report, 'baseline'),
            ('control50', CONTROL, control, 'final'), ('replay50', RUN, report, 'final')]:
        for row in rows:
            name = 'check-' + label + '--' + row['sample_key'] + '.json'
            case = read_bound(directory / name, run_report['artifact_sha256'][name])
            key = row['source'] + ':' + row['sample_key']
            if case['metrics']['key'] != key: raise ValueError('Fixed training sample changed')
            result = compare_case(case, budget)
            result.update(arm=arm, key=key, group=row['fit_domain'])
            filename = arm + '--' + row['sample_key'] + '.json'
            write(filename, result)
            records.append({k: v for k, v in result.items() if k not in ('guarded_scene', 'prediction')})
        print(json.dumps(dict(arm_complete=arm, cases=len(records), chemistry_compiles=budget.calls)), flush=True)
    summary = summarize(records)
    changes = [dict(arm=r['arm'], key=r['key'], legacy=r['legacy'], guarded=r['guarded'],
        status=r['journal']['status'], local_rollback=r['journal']['local_rollback'],
        changed_bonds=r['changed_bonds'], outside_rejected_aromatic_bond_ids_preserved=r['outside_rejected_aromatic_bond_ids_preserved'])
        for r in records if r['legacy'] != r['guarded'] or r['journal']['local_rollback']]
    for path, expected in hashes.items():
        if digest(path) != expected: raise ValueError('Input changed during replay')
    for path, expected in implementations.items():
        if digest(ROOT / path) != expected: raise ValueError('Implementation changed during replay')
    write_json(OUTPUT / 'report.json', dict(schema_version='aromatic-transaction-result-0.1', complete=True,
        synthetic_tests=totals, cases=len(records), summary=summary, changes=changes,
        status_counts=dict(Counter(r['journal']['status'] for r in records)), chemistry_compiles=budget.calls,
        all174_legacy_scenes_and_predictions_exactly_reproduced=True,
        guard_protection_failures=[dict(arm=arm, keys=v['all']['protection_failures'])
                                   for arm, v in summary.items() if v['all']['protection_failures']],
        original_training_gate=report['decision'], original_training_gate_reinterpreted=False,
        input_sha256=hashes, implementation_sha256=implementations, artifact_sha256=artifacts,
        model_loads=0, model_forwards=0, ocr_forwards=0, optimizer_steps=0, gpu_used=False,
        deployment_authorized=False, default_changed=False, independent_accuracy_measured=False,
        validation_sample_artifacts_read=False, sealed_test_read=False, new_identity_queries=0,
        limitations=['Valid chemistry does not imply correct identity or recovered OCR',
                     'A local rollback is an explicit correction warning, not a hidden fallback success',
                     'Preexisting-invalid raw scenes retain the historical prediction unchanged',
                     'Frozen raw typed-bond protection failure is not reclassified as training success']))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--worker', action='store_true')
    if parser.parse_args(argv).worker:
        worker(); return 0
    receipt = runtime.run_budgeted([sys.executable, '-m', 'tools.audit_aromatic_transactions', '--worker'],
                                  output=OUTPUT, budget=runtime.ResourceBudget(120))
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__': raise SystemExit(main())
