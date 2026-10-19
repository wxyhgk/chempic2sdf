"""Prepare a draft, checking fixed58 saved supervision on CPU; never train."""

import argparse
from collections import Counter
from hashlib import sha256
import inspect
import json
import os
from pathlib import Path
import sys

import torch

from chemvector.bond_object import BOND_TYPE_TO_ID
from chemvector.training.curriculum import pair_trial_contract as trial
from chemvector.training.curriculum import pair_stage, pair_supervision, runtime, supervision_contract
from tools import audit_pair_supervision_wiring as wiring
from tools import audit_selector_stable_subset as source

ROOT = source.ROOT
OUTPUT = ROOT / 'outputs/audits/chemvector-pair-only-trial-preparation-20260921'
DRYRUN = ROOT / 'outputs/audits/chemvector-pair-initial-dryrun4-20260921/report.json'
DRYRUN_SHA = 'be373defdcd25fb29ba5a2845d2f5923cb1c2df4951c7f2806a77c6a61384efc'
WIRING = wiring.OUTPUT / 'report.json'
WIRING_SHA = '3699d01515776b5e37996ae8f9e2089a18af0bd847aca32f4eb725cf41d0fe83'


def worker():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('CPU-only metadata/saved-candidate preparation required')
    torch.set_num_threads(2)
    inputs = source.Inputs()
    _, manifest, rows, bindings = source.preflight(inputs)
    if inputs.hashes[str(source.MANIFEST)] != trial.MANIFEST_SHA:
        raise ValueError('Fixed58 manifest changed')
    admission = inputs.json(source.MANIFEST.with_name('admission.json'), trial.ADMISSION_SHA)
    dryrun = inputs.json(DRYRUN, DRYRUN_SHA)
    previous = inputs.json(WIRING, WIRING_SHA)
    if (dryrun['current_cpu_arm_comparisons_passed'] != 4 or dryrun['optimizer_steps'] != 0
            or dryrun['initial_state_before'] != dryrun['initial_state_after']):
        raise ValueError('Expected completed no-update initial dry-run evidence')
    for module in (Path(__file__), Path(trial.__file__), Path(pair_stage.__file__),
                   Path(supervision_contract.__file__), Path(pair_supervision.__file__),
                   Path(wiring.__file__), Path(source.__file__), Path(runtime.__file__),
                   Path(inspect.getfile(pair_stage.structure_forward)),
                   Path(inspect.getfile(pair_stage.direct_atom_pair_loss)),
                   Path(inspect.getfile(wiring.match_atom_targets)),
                   ROOT / 'tests/chemvector/test_pair_trial_contract.py'):
        expected = dryrun['input_sha256'].get(str(module), previous['input_sha256'].get(str(module)))
        inputs.read(module, expected)
    source.write(OUTPUT / 'plan.json', dict(schema_version='pair-only-preparation-plan-0.1',
        sample_keys=[r['sample_key'] for r in rows], wall_seconds=120,
        model_forwards=0, optimizer_steps=0, checkpoint_loads=0, gpu_devices=[],
        sampler_calls=116, training_authorized=False, validation_sample_artifacts_read=False,
        new_identity_queries=0, input_sha256=dict(inputs.hashes)))
    proposal = trial.make_proposal(rows)
    trial.validate_proposal(proposal, rows)
    source.write(OUTPUT / 'proposal.json', proposal)
    totals, cases = Counter(), []
    for ordinal, row in enumerate(rows):
        key = row['sample_key']
        base = Path(manifest['sources'][row['source']]) / 'samples'
        meta = inputs.json(base / (key + '.meta.json'), row['meta_sha256'])
        layers = inputs.json(base / (key + '.layers.json'), meta['sha256']['.layers.json'])
        paths = [source.CONTROL / (prefix + '--' + key + '.json') for prefix in ('geometry', 'baseline')]
        geometry, record = [inputs.json(path, bindings[str(path)]) for path in paths]
        if (layers['sample_key'] != key or layers['molecule_id'] != row['molecule_id']
                or geometry['key'] != row['source'] + ':' + key or record['key'] != geometry['key']):
            raise ValueError('Saved case identity changed')
        _, _, mapping, atoms, _ = wiring.tensors(geometry, record, layers)
        bonds = layers['bonds']
        targets = dict(atom_mask=torch.ones(1, len(layers['atoms']), dtype=torch.bool),
            bond_mask=torch.ones(1, len(bonds), dtype=torch.bool),
            bond_atom_refs=torch.tensor([[[b['begin_atom'], b['end_atom']] for b in bonds]],
                                       dtype=torch.long).reshape(1, len(bonds), 2),
            bond_chemical=torch.tensor([[BOND_TYPE_TO_ID[b['chemical_type']] for b in bonds]], dtype=torch.long))
        result = trial.compare_supervision(atoms, mapping, targets, seed=trial.RECIPE['seed'] + ordinal)
        totals.update({k: int(v) for k, v in result.items()})
        cases.append(dict(sample_key=key, group=row['fit_domain'], **result))
    for field, previous_field in (('sampled_positive_rows', 'sampled_positive_rows'),
                                  ('sampled_negative_rows', 'sampled_negative_rows'),
                                  ('unknown_pairs', 'ignored_pool_rows'), ('empty_branch', 'empty_pair_branches')):
        if totals[field] != previous['totals'][pair_supervision.MATCHED_ONLY][previous_field]:
            raise ValueError('Saved fixed58 sampler accounting changed: ' + field)
    source.write(OUTPUT / 'supervision-equivalence.json', dict(cases=cases, totals=totals))
    for path, digest in list(inputs.hashes.items()):
        inputs.read(path, digest)
    if torch.cuda.is_initialized():
        raise ValueError('Unexpected CUDA initialization')
    report = dict(schema_version='pair-only-trial-preparation-0.1',
        proposal_sha256=sha256((OUTPUT / 'proposal.json').read_bytes()).hexdigest(),
        equivalence_sha256=sha256((OUTPUT / 'supervision-equivalence.json').read_bytes()).hexdigest(),
        input_sha256=inputs.hashes, images=58, totals=totals,
        proposal_valid=True, old_stage_equivalent_on_saved58=True,
        real_initial_dryrun_current_cpu_passed=4, historical_cpu_gpu_candidate_passed=2,
        existing_admission_use=admission['use'], proposed_use=trial.USE,
        readiness=dict(data_use_extension_approved=False, new_execution_authorization=False,
                       fresh_execution_device_cache=False, pair_trial_launcher_implemented=False,
                       protocol_prepared=True, safe_to_launch=False),
        blockers=['Exact fixed58 data-use extension beyond local_selector_retention_fit_only',
                  'New one-shot authorization bound to final protocol and implementation; old grants consumed',
                  'Budgeted cache/evaluation/optimizer launcher implementing this draft',
                  'Fresh device-bound candidates, labels and baseline before any optimizer update'],
        model_forwards=0, optimizer_steps=0, checkpoint_loads=0, checkpoint_writes=0,
        gpu_used=False, training_performed=False, training_authorized=False, default_changed=False,
        validation_sample_artifacts_read=False, sealed_test_read=False, new_identity_queries=0,
        limitations=['Saved supervision equivalence is not a fresh GPU cache or a full training-stage forward',
            '58 train/replay cases cannot measure independent recognition accuracy',
            'Unknown endpoints remain ignored; no missing positives or graph connections were added',
            'Single-arm full-pair trial cannot isolate mask causality from the changed update scope',
            'Existing strict matching still includes previously diagnosed ambiguous correspondence',
            'No CUDA availability or performance test was performed'])
    source.write(OUTPUT / 'report.json', report)
    print(json.dumps(dict(images=58, totals=totals, readiness=report['readiness']), ensure_ascii=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        worker()
        return 0
    receipt = runtime.run_budgeted([sys.executable, '-m', 'tools.prepare_pair_only_trial', '--worker'],
                                  output=OUTPUT, budget=runtime.ResourceBudget(120))
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
