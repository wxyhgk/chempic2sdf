"""Two fixed offline masks on the saved initial fixed58; no training or inference."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

import torch

from chemvector.training.curriculum import runtime
from tools import audit_selector_stable_subset as source
from tools import pair_supervision_mask as masks
from tools import selector_missing_graph_trace as trace

ROOT = source.ROOT
OUTPUT = ROOT / 'outputs/audits/chemvector-pair-supervision-mask58-20260921'
TRACE = ROOT / 'outputs/audits/chemvector-selector-missing10-graph-trace-20260921/report.json'
TRACE_SHA = '4254d5c8568234f2346951aa0c4b9bf829d424cebd36efdc3634d581a46dbd54'


def preflight(inputs):
    prior, manifest, rows, bindings = source.preflight(inputs)
    previous_trace = inputs.json(TRACE, TRACE_SHA)
    focused = defaultdict(list)
    for node in previous_trace['focused_nodes']:
        for bond in node['expected_incident_bonds']:
            focused[node['sample_key']].append(dict(target_id=node['target_id'], **bond))
    if sum(map(len, focused.values())) != 19 or len(focused) != 7:
        raise ValueError('Expected the unchanged nineteen prior diagnosed connections')
    return manifest, rows, bindings, focused, previous_trace


def worker():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('CPU-only saved-artifact audit required')
    torch.set_num_threads(2)
    inputs = source.Inputs()
    manifest, rows, bindings, focused, previous_trace = preflight(inputs)
    for path in (Path(__file__), Path(masks.__file__), Path(trace.__file__), Path(source.__file__),
            Path(runtime.__file__), ROOT / 'tools/selector_node_groups.py',
            ROOT / 'chemvector/training/graph/relations/train_direct_atom_pair_bond.py',
            ROOT / 'chemvector/training/curriculum/continuation.py', ROOT / 'frontend/inference_profiles.json',
            ROOT / 'tests/chemvector/test_pair_supervision_mask.py'):
        inputs.read(path, previous_trace['input_sha256'].get(str(path), bindings.get(str(path))))
    source.write(OUTPUT / 'plan.json', dict(schema_version='pair-supervision-mask-plan-0.1',
        sample_keys=[r['sample_key'] for r in rows], policies=list(masks.POLICIES), images=58,
        user_request='继续推进', model_forwards=0, optimizer_steps=0, gpu_devices=[],
        training_authorized=False, wall_budget_seconds=120, automatic_retries=0,
        phase='saved_initial_only', original_match_radius_px=6, original_selector_threshold=-.25,
        positive_relabeling=False, real_data_sampling=False, production_integration=False,
        ownership_policy='Conditional ownership is an unapproved spatial hypothesis, never a new positive label',
        counters=['original positives preserved', 'negative pool retained/ignored',
                  'per-image negative sampling capacity', 'nineteen prior connections isolated',
                  'diagnostic reference nonedge losses and unresolved rows']))
    totals = defaultdict(lambda: defaultdict(Counter))
    references = defaultdict(lambda: defaultdict(Counter))
    case_hashes, focused_checks, capacities = {}, [], []
    for row in rows:
        key = row['sample_key']
        base = Path(manifest['sources'][row['source']]) / 'samples'
        meta = inputs.json(base / (key + '.meta.json'), row['meta_sha256'])
        layers = inputs.json(base / (key + '.layers.json'), meta['sha256']['.layers.json'])
        geo_path = source.CONTROL / ('geometry--' + key + '.json')
        record_path = source.CONTROL / ('baseline--' + key + '.json')
        geometry = inputs.json(geo_path, bindings[str(geo_path)])
        record = inputs.json(record_path, bindings[str(record_path)])
        if (record['key'] != row['source'] + ':' + key or record['key'] != geometry['key']
                or layers['sample_key'] != key or layers['molecule_id'] != row['molecule_id']
                or record['group'] != row['fit_domain']):
            raise ValueError('Fixed initial source identities changed')
        result = masks.inspect_case(geometry, record, layers)
        for group in ('all', row['fit_domain']):
            for policy, counts in result['summary'].items():
                totals[group][policy].update(counts)
                totals[group][policy]['images'] += 1
                references[group][policy].update(result['diagnostic_reference_counts'][policy])
        lookup = {tuple(p['pair']): p for p in result['pairs']}
        for connection in focused.get(key, []):
            pair = tuple(connection['candidate_pair'])
            found = lookup[pair]
            if found['original_presence'] or found['diagnostic_bond_id'] != connection['bond_id']:
                raise ValueError('Original diagnosed connection changed')
            states = {p: found[p] for p in masks.POLICIES}
            focused_checks.append(dict(sample_key=key, target_id=connection['target_id'],
                bond_id=connection['bond_id'], candidate_pair=list(pair), states=states))
        for policy in masks.POLICIES:
            counts = result['summary'][policy]
            if counts['negative_sampling_capacity_loss'] or counts['ignore']:
                capacities.append(dict(sample_key=key, group=row['fit_domain'], policy=policy, **counts))
        path = OUTPUT / (key + '.json')
        source.write(path, result)
        case_hashes[path.name] = sha256(path.read_bytes()).hexdigest()
    if len(focused_checks) != 19 or any(c['states'][p] != 'ignore' for c in focused_checks for p in masks.POLICIES):
        raise ValueError('A policy failed to isolate an original diagnosed false negative')
    for path, digest in list(inputs.hashes.items()):
        inputs.read(path, digest)
    if torch.cuda.is_initialized():
        raise ValueError('Unexpected CUDA initialization')
    source.write(OUTPUT / 'report.json', dict(schema_version='pair-supervision-mask58-0.1',
        totals=totals, diagnostic_reference_counts=references, focused19_checks=focused_checks,
        changed_image_policy_counts=capacities, input_sha256=inputs.hashes, case_sha256=case_hashes,
        model_forwards=0, optimizer_steps=0, gpu_used=False, training_performed=False,
        training_authorized=False, positive_labels_changed=False, production_integrated=False,
        original_labels_changed=False, thresholds_changed=False, default_changed=False,
        real_data_pairs_sampled=0, validation_sample_artifacts_read=False, sealed_test_read=False,
        new_identity_queries=0, limitations=[
            'Negative-pool retention is not model recognition accuracy or a historical training effect',
            'Diagnostic ownership is not a complete true-identity oracle or released training labels',
            'Conditional negatives remain assumption-dependent and are reported separately',
            'Ignoring a false negative does not supply its missing positive bond supervision',
            'Only the saved initial fixed58 state was audited, not changing candidates during training']))
    print(json.dumps(dict(totals=totals, focused19_isolated=True), ensure_ascii=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        worker()
        return 0
    preflight(source.Inputs())
    receipt = runtime.run_budgeted([sys.executable, '-m', 'tools.audit_pair_supervision_mask', '--worker'],
                                  output=OUTPUT, budget=runtime.ResourceBudget(120))
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
