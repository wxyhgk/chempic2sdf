"""Close the three saved counterfactuals with a prediction-only refusal gate."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

from chemvector.experiments import candidate_acceptance as gate
from chemvector.training.curriculum import runtime
from tools.preview_pair_false_edges import ROOT, read

PRIOR = ROOT / 'outputs/audits/chemvector-candidate-safety58-20260921'
OUTPUT = ROOT / 'outputs/audits/chemvector-candidate-acceptance3-20260921'
EXPECTED = {'curr_08d02946e879406f__font_large': True,
            'curr_20d6303f5788a4d7__serif': False, 'curr_cb5b7da426023d14__font_large': False}


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def worker():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or not OUTPUT.is_dir():
        raise ValueError('Fresh CPU budgeted worker required')
    hashes = {}
    prior = json.loads(read(PRIOR / 'report.json', hashes,
                           '08c60f9d87014fd4c65e50976a79b8b05e82914bbab8db56e678af19e5875656'))
    records = {}
    for key in EXPECTED:
        name = key + '-prediction-only.json'
        records[key] = json.loads(read(PRIOR / name, hashes, prior['artifact_sha256'][name]))
    for path in (Path(__file__), Path(gate.__file__), ROOT / 'tests/chemvector/test_candidate_acceptance.py',
                 ROOT / 'chemvector/experiments/single_candidate_replay.py',
                 ROOT / 'chemvector/training/curriculum/validation_truth.py', Path(runtime.__file__),
                 ROOT / 'tools/preview_pair_false_edges.py'):
        read(path, hashes, prior['input_sha256'].get(str(path.resolve())))
    write(OUTPUT / 'plan.json', dict(sample_keys=list(EXPECTED), expected_regression_decisions=EXPECTED,
        pair_threshold=.85, wall_seconds=120, new_model_forwards=0, optimizer_steps=0,
        new_data=False, training=False, production_changed=False, input_sha256=dict(hashes),
        scope='One saved-candidate hypothesis each; no alternative, retuning, recursive round or independent safety claim'))
    decisions, selected_hashes = {}, {}
    for key, record in records.items():
        decision, selected = gate.select_prediction(record['queue'], record['baseline'], record['admitted'])
        decisions[key] = decision
        name = key + '-selected-copy.json'
        write(OUTPUT / name, selected)
        selected_hashes[name] = sha256((OUTPUT / name).read_bytes()).hexdigest()
    write(OUTPUT / 'decisions-prediction-only.json', decisions)
    passed = all(decisions[k]['accepted'] == expected for k, expected in EXPECTED.items())
    for path, digest in list(hashes.items()):
        read(path, hashes, digest)
    write(OUTPUT / 'report.json', dict(schema_version='candidate-acceptance3-0.1',
        regression_pass=passed, decisions=decisions, input_sha256=hashes,
        artifact_sha256={**selected_hashes, **{name: sha256((OUTPUT / name).read_bytes()).hexdigest()
            for name in ('plan.json', 'decisions-prediction-only.json')}},
        model_forwards=0, training_performed=False, optimizer_steps=0, gpu_used=False,
        checkpoint_loads=0, new_sample_artifacts_read=False, original_outputs_changed=False,
        independent_safety_proven=False, production_admission=False,
        next_action='close_candidate_branch' if passed else 'stop_without_retuning'))
    print(json.dumps(dict(regression_pass=passed, decisions=decisions), indent=2))
    if not passed:
        raise RuntimeError('Regression failed; stop without changing rule')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args(argv)
    if args.worker:
        worker(); return 0
    receipt = runtime.run_budgeted([sys.executable, '-m', 'tools.audit_candidate_acceptance3', '--worker'],
                                  output=OUTPUT, budget=runtime.ResourceBudget(120))
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
