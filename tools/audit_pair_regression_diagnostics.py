"""Fixed58 saved decoding and sampled BCE direction audit; no model updates."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

import torch
from rdkit import rdBase

from chemvector.training.curriculum import runtime
from tools import pair_regression_diagnostics as diagnostics
from tools.audit_selector_stable_subset import Inputs, ROOT, write

TRIAL = ROOT / 'outputs/experiments/chemvector-pair-only-matched-trial-20260921'
OUTPUT = ROOT / 'outputs/audits/chemvector-pair-regression-diagnostics58-20260921'
REPORT_SHA = 'd12d93e5e012f7566f4cc3123abf411b67f49c44ecb898404034105b3c4603eb'
POSTCHECK_SHA = 'b09b82c98fe5280de0d79a6cca858a2854398fa66dca1d29205997f4cece1f54'
MANIFEST = ROOT / 'outputs/datasets/chemvector-selector-retention-fit-20260919/manifest.json'
MANIFEST_SHA = 'f5e3a7163eef5c49118ad78dcf05778f4ad889dc9b6389097ef20896b79e2111'
PROBES = {'curr_4ef506af64e8f0b8__bond_short': (15, 18),
          'curr_08d02946e879406f__font_large': (4, 6)}


def worker():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('CPU saved-artifact audit only')
    torch.set_num_threads(2)
    inputs = Inputs()
    report = inputs.json(TRIAL / 'report.json', REPORT_SHA)
    post = inputs.json(TRIAL / 'postcheck.json', POSTCHECK_SHA)
    manifest = inputs.json(MANIFEST, MANIFEST_SHA)
    auth = inputs.json(ROOT / 'docs/model/training/releases/20260921-pair-only-training-authorization.json',
                       report['authorization_sha256'])
    # Recover labels only under the exact historical batch=1 sampler order,
    # not an assumed convention of an arbitrary training pipeline.
    if (auth['recipe']['batch_size'] != 1 or auth['recipe']['pair_supervision'] != 'matched-only-v1'
            or auth['recipe']['pair_threshold'] != diagnostics.PAIR_THRESHOLD):
        raise ValueError('Unsupported historical sampler receipt semantics')
    for relative in ('chemvector/training/graph/relations/train_direct_atom_pair_bond.py',
                     'chemvector/training/curriculum/supervision_contract.py',
                     'chemvector/training/curriculum/pair_cached_trial.py',
                     'chemvector/training/curriculum/validation_truth.py',
                     'chemvector/components/graph/joint/direct_atom_pair_bond.py'):
        inputs.read(ROOT / relative, auth['implementation_sha256'][relative])
    for path in (Path(__file__), Path(diagnostics.__file__), Path(runtime.__file__),
                 ROOT / 'tools/audit_selector_stable_subset.py',
                 ROOT / 'tests/chemvector/test_pair_regression_diagnostics.py',
                 ROOT / 'chemvector/training/curriculum/pair_negative_observer.py',
                 ROOT / 'tests/chemvector/test_pair_negative_observer.py'):
        inputs.read(path)
    events = [json.loads(line) for line in inputs.read(TRIAL / 'events.jsonl',
                post['artifact_sha256']['events.jsonl']).decode().splitlines()]
    updates = [r for r in events if r['event'] == 'batch_complete' and r['updated']]
    if ([r['step'] for r in updates] != list(range(1, 51))
            or len({r['key'] for r in updates}) != 50):
        raise ValueError('Expected one pass with 50 distinct updated images')
    by_key = {r['key']: r for r in updates}
    rows = manifest['training_samples']
    if (len(rows) != 58 or len({r['sample_key'] for r in rows}) != 58
            or any(r['split'] != 'train' for r in rows)
            or Counter(r['fit_domain'] for r in rows) != {'witness': 26, 'old_full_molecule': 32}):
        raise ValueError('Fixed58 training/replay scope required')
    write(OUTPUT / 'plan.json', dict(schema_version='pair-regression-diagnostics-plan-0.1',
        sample_keys=[r['sample_key'] for r in rows], probes={k: list(v) for k, v in PROBES.items()},
        wall_seconds=120, model_forwards=0, checkpoint_loads=0, optimizer_steps=0,
        maximum_saved_logit_backwards=100, original_acceptance_changed=False,
        no_gpu=True, validation_sample_artifacts_read=False,
        boundary='Saved-logit gradients are not historical model/optimizer gradients',
        input_sha256=dict(inputs.hashes)))
    totals, case_hashes, focused, false_edges = Counter(), {}, [], []
    failures = {f['key']: f for f in report['decisions'][0]['protection_failures']}
    for row in rows:
        sample, key = row['sample_key'], row['source'] + ':' + row['sample_key']
        before, after = [inputs.json(TRIAL / (name := f'check-{step:06d}--{sample}.json'),
                                    post['artifact_sha256'][name]) for step in (0, 50)]
        if before['metrics']['key'] != key or after['metrics']['key'] != key:
            raise ValueError('Saved sample outside fixed scope')
        decoding = diagnostics.compare_decoding(before, after)
        totals.update(images=1, **{field: decoding[field] for field in (
            'pairs', 'raw_presence_added', 'raw_presence_removed', 'raw_chemical_changes_on_present_pairs',
            'final_structure_equal', 'new_invalid_sdf', 'new_fallback', 'baseline_exact_broken')})
        result = dict(sample_key=sample, group=row['fit_domain'], decoding=decoding, presence_loss_replay=None)
        if key in by_key:
            replay = diagnostics.presence_replay(before, after, by_key[key])
            result['presence_loss_replay'] = replay
            labels = diagnostics.recovered_sample_labels(by_key[key], len(before['pairs']))
            totals.update(updated_images_audited=1, saved_logit_backwards=2,
                          sampled_positive_rows=sum(labels.values()), sampled_negative_rows=sum(not v for v in labels.values()))
            if sample in PROBES:
                target = list(PROBES[sample]); index = before['pairs'].index(target)
                if labels.get(index) is not False:
                    raise ValueError('Fixed false-edge witness was not a sampled negative')
                arms = {phase: next(r for r in values['per_sampled_row'] if r['row_id'] == index)
                        for phase, values in replay['arms'].items()}
                if any(arm['direct_logit_gradient'] <= 0 for arm in arms.values()):
                    raise ValueError('Witness gradient did not push the negative logit down')
                false_edges.append(dict(sample_key=sample, pair=target, row_id=index,
                    sampled_as='confirmed_negative', step=by_key[key]['step'], arms=arms,
                    snapshot_cannot_identify_which_update_raised_score=True))
        if key in failures:
            focused.append(dict(sample_key=sample, historical_alarm=failures[key], decoding=decoding))
        path = OUTPUT / (sample + '.json')
        write(path, result); case_hashes[path.name] = sha256(path.read_bytes()).hexdigest()
    if (totals['images'] != 58 or totals['updated_images_audited'] != 50
            or totals['saved_logit_backwards'] != 100 or len(false_edges) != 2):
        raise ValueError('Incomplete diagnostic scope')
    for path, digest in list(inputs.hashes.items()): inputs.read(path, digest)
    if torch.cuda.is_initialized(): raise ValueError('Unexpected GPU initialization')
    write(OUTPUT / 'report.json', dict(schema_version='pair-regression-diagnostics58-0.1',
        totals=totals, historical_alarm_cases=focused, fixed_false_edges=false_edges,
        versions=dict(torch=torch.__version__, rdkit=rdBase.rdkitVersion),
        case_sha256=case_hashes, input_sha256=inputs.hashes,
        model_forwards=0, checkpoint_loads=0, optimizer_steps=0, gpu_used=False,
        training_performed=False, original_acceptance_changed=False, default_changed=False,
        validation_sample_artifacts_read=False, sealed_test_read=False, new_identity_queries=0,
        future_observer_wired=False,
        limitations=['No historical features, intermediate weights or Adam states were saved; cannot locate causal step',
            'Positive-prefix convention is valid only for the bound one-image sampler',
            'Unselected rows are not labeled negative or unknown by this replay',
            'Direct-logit BCE direction is not the direction of a shared-parameter Adam update',
            'Same final structure does not mean correct structure; failed compilation is never equality',
            'Diagnostic stratification does not retroactively relax the training acceptance gate']))
    print(json.dumps(dict(totals=totals, fixed_false_edges=false_edges), ensure_ascii=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        worker(); return 0
    receipt = runtime.run_budgeted([sys.executable, '-m', 'tools.audit_pair_regression_diagnostics', '--worker'],
                                  output=OUTPUT, budget=runtime.ResourceBudget(120))
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
