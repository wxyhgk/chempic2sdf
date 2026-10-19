"""Fixed8/25 saved-evidence and threshold-objective CPU probe, not training."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

import torch

from chemvector.training.curriculum import runtime
from tools.audit_selector_stable_subset import Inputs, write
from tools import audit_selector_stable_subset as source
from tools import selector_threshold_contract as contract
from tools.selector_singleton_rank import SingleNodeContrast

ROOT = source.ROOT
PREVIOUS = ROOT / 'outputs/audits/chemvector-selector-rejected21-20260920'
PREVIOUS_SHA = '20d63abbfeb85bf6dae75a96866681c999f9ae25ae05aebc8847b469a546ec97'
RADIUS = ROOT / 'outputs/audits/chemvector-selector-radius-reference-20260920'
OUTPUT = ROOT / 'outputs/audits/chemvector-selector-threshold-contract8-20260921'


def load_scope(inputs):
    report = inputs.json(PREVIOUS / 'report.json', PREVIOUS_SHA)
    focused = inputs.json(PREVIOUS / 'focused21.json', report['focused_sha256'])['nodes']
    nodes = [n for n in focused if n['phases']['initial']['comparison'] == 'fragment_ties_or_outranks_valid']
    if (len(nodes) != 8 or len({n['sample_key'] for n in nodes}) != 6
            or sum(len(n['stable_fragment_slots']) for n in nodes) != 25
            or any(len(n['valid_slots']) != 1 for n in nodes)):
        raise ValueError('Exactly the previously fixed eight singletons and25 fragments required')
    return report, nodes


def worker():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('CPU-only saved-logit probe required')
    if not OUTPUT.is_dir() or (OUTPUT / 'plan.json').exists():
        raise ValueError('Fresh output required; no retries')
    torch.set_num_threads(2)
    inputs = Inputs()
    prior, nodes = load_scope(inputs)
    for path in (Path(__file__), Path(contract.__file__), Path(source.__file__), Path(runtime.__file__),
                 ROOT / 'tools/selector_singleton_rank.py',
                 ROOT / 'tests/chemvector/test_selector_threshold_contract.py'):
        inputs.read(path)
    write(OUTPUT / 'plan.json', dict(sample_target_ids=[dict(sample_key=n['sample_key'], target_id=n['target_id']) for n in nodes],
        threshold=-.25, objective='mean_node 0.5*(softplus(tau-z_valid)+mean_fragment softplus(z_fragment-tau))',
        phases=[p[0] for p in source.PHASES], wall_budget_seconds=120, automatic_retries=0,
        model_forwards=0, optimizer_steps=0, gpu_devices=[], training_authorized=False,
        unknown_negatives_supervised=False, weight_search=False))
    by_key = defaultdict(list)
    reviews = []
    for node in nodes:
        by_key[node['sample_key']].append(node)
    for key, owned in by_key.items():
        paths = [source.OUTPUT / (key+'.json'), source.CONTROL / ('geometry--'+key+'.json'), RADIUS / (key+'.json')]
        subset, geometry, radius = [inputs.json(p, prior['input_sha256'][str(p)]) for p in paths]
        for node in owned:
            reviews.append(dict(sample_key=key, **contract.review_contrast(node, subset, geometry, radius)))
    phases = {}
    for phase, parent, prefix in source.PHASES:
        values, groups, origins = [], [], []
        for key, owned in by_key.items():
            path = parent / (prefix+'--'+key+'.json')
            saved = inputs.json(path, prior['input_sha256'][str(path)])
            scores = {int(s): v for s, v in saved['metrics']['logits'].items()}
            positions = {s: len(values)+i for i, s in enumerate(sorted(scores))}
            values.extend(scores[s] for s in sorted(scores))
            for node in owned:
                groups.append(SingleNodeContrast(key+':'+str(node['target_id']),
                    tuple(positions[s] for s in node['valid_slots']),
                    tuple(positions[s] for s in node['stable_fragment_slots'])))
                origins.append(node)
        z = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        result = contract.threshold_contract(z, groups)
        gradient, = torch.autograd.grad(result['loss'], z)
        records = []
        for i, (group, origin) in enumerate(zip(groups, origins)):
            valid, bad = group.valid_slots[0], list(group.fragment_slots)
            records.append(dict(sample_key=origin['sample_key'], target_id=origin['target_id'],
                valid_slot=origin['valid_slots'][0], fragment_slots=origin['stable_fragment_slots'],
                valid_logit=values[valid], fragment_logits=[values[s] for s in bad],
                loss=float(result['per_node_losses'][i].detach()),
                valid_derivative=float(gradient[valid]), fragment_derivatives=gradient[bad].tolist(),
                **result['decisions'][i]))
        outside = sorted(set(range(len(values))) - set(result['supervised_slots']))
        if any(r['valid_derivative'] >= 0 or any(d <= 0 for d in r['fragment_derivatives']) for r in records):
            raise ValueError('Direct threshold derivative signs differ')
        if torch.count_nonzero(gradient[outside]):
            raise ValueError('Unreferenced logit has a direct derivative')
        phases[phase] = dict(loss=float(result['loss'].detach()), nodes=records,
            common_shift_derivative=float(gradient.sum()), unreferenced_direct_gradient_nonzero=0,
            both_decisions_correct=sum(r['both_decisions_correct'] for r in records))
    for path, digest in list(inputs.hashes.items()):
        inputs.read(path, digest)
    write(OUTPUT / 'report.json', dict(schema_version='selector-threshold-contract8-0.1',
        input_sha256=inputs.hashes, evidence_reviews=reviews, phases=phases, nodes=8, fragments=25, images=6,
        diagnostic_evidence_consistent=True, mathematical_contract_passed=True,
        human_semantic_review_performed=False, new_labels_released=False, training_authorized=False,
        graph_training_objective_ready=False, model_forwards=0, optimizer_steps=0, gpu_used=False,
        parameter_direction_checked=False, validation_sample_artifacts_read=False, sealed_test_read=False,
        default_changed=False, predictions_changed=False,
        limitations=['Saved ownership consistency is not human label approval',
            'Absolute threshold BCE is a local diagnostic, not a new complete training loss',
            'Output derivatives do not guarantee safe shared-parameter directions or finite decisions',
            'Multi-candidate nodes, unknown fragments, and ten raw-missing targets remain outside scope']))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        worker()
        return 0
    receipt = runtime.run_budgeted([sys.executable, '-m', 'tools.probe_selector_threshold_contract', '--worker'],
        output=OUTPUT, budget=runtime.ResourceBudget(120))
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
