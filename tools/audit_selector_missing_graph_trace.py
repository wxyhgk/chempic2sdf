"""Bounded CPU audit of the original ten misses in seven saved training images."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

import torch

from chemvector.training.curriculum import runtime
from tools.audit_selector_stable_subset import Inputs, write
from tools import selector_missing_graph_trace as trace

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / 'outputs/audits/chemvector-selector-coordinate-contract-20260920-r1/report.json'
PRIOR_SHA = '834bcf6eeb2e119bc8e0bfd64db1fbe20259bcd6275c384dd72ff0c5bbc77e62'
MANIFEST = ROOT / 'outputs/datasets/chemvector-selector-retention-fit-20260919/manifest.json'
CONTROL = ROOT / 'outputs/experiments/chemvector-selector-full-loss-control-20260920'
OUTPUT = ROOT / 'outputs/audits/chemvector-selector-missing10-graph-trace-20260921'


def preflight(inputs):
    prior = inputs.json(PRIOR, PRIOR_SHA)
    bindings = prior['input_sha256']
    manifest = inputs.json(MANIFEST, bindings[str(MANIFEST)])
    focus = defaultdict(list)
    for r in prior['missing_targets']:
        focus[r['sample_key']].append(r['target_id'])
    if len(focus) != 7 or sum(map(len, focus.values())) != 10:
        raise ValueError('Exactly the original seven images / ten targets required')
    rows = {r['sample_key']: r for r in manifest['training_samples']}
    for key, targets in focus.items():
        row = rows[key]
        if row['split'] != 'train' or row['fit_domain'] != 'old_full_molecule' or len(set(targets)) != len(targets):
            raise ValueError('Fixed old-training subset only')
    return manifest, rows, focus, bindings


def worker():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('CPU saved-artifact audit only')
    torch.set_num_threads(2)
    inputs = Inputs()
    manifest, rows, focus, bindings = preflight(inputs)
    source_paths = [Path(__file__), Path(trace.__file__), Path(runtime.__file__),
        ROOT / 'tools/audit_selector_stable_subset.py',
        ROOT / 'chemvector/training/graph/relations/train_direct_atom_pair_bond.py',
        ROOT / 'chemvector/training/curriculum/continuation.py',
        ROOT / 'chemvector/training/curriculum/evaluation.py',
        ROOT / 'chemvector/evaluation/shadows/shadow_chemscene_v02_oracle.py',
        ROOT / 'Data/sdf/chemscene_v0_2.py', ROOT / 'Data/sdf/chemscene_v0_1.py',
        ROOT / 'Data/sdf/chemsvg_sequence_v4.py', ROOT / 'frontend/inference_profiles.json']
    for path in source_paths:
        inputs.read(path, bindings.get(str(path)))
    write(OUTPUT / 'plan.json', dict(schema_version='missing10-graph-trace-plan-0.1',
        focus=dict(focus), model_forwards=0, optimizer_steps=0, gpu_devices=[], wall_budget_seconds=120,
        training_authorized=False, label_or_threshold_changes=False, automatic_retries=0,
        policy='Exact saved SDF/candidate lineage; coordinate-only diagnostic correspondence; unchanged pair-target replay',
        unresolved_policy='No MolBlock means unknown final local graph, never infer it from sanitize failure'))
    totals, case_hashes, focused = Counter(), {}, []
    for key, targets in focus.items():
        row = rows[key]
        base = Path(manifest['sources'][row['source']]) / 'samples'
        meta = inputs.json(base / (key + '.meta.json'), row['meta_sha256'])
        layers = inputs.json(base / (key + '.layers.json'), meta['sha256']['.layers.json'])
        geo_path, pred_path = CONTROL / ('geometry--' + key + '.json'), CONTROL / ('baseline--' + key + '.json')
        geometry = inputs.json(geo_path, bindings[str(geo_path)])
        record = inputs.json(pred_path, bindings[str(pred_path)])
        if (record['key'] != geometry['key'] or record['key'] != row['source'] + ':' + key
                or layers['sample_key'] != key or layers['molecule_id'] != row['molecule_id']
                or record['prediction']['expected_smiles'] != row['canonical_smiles']):
            raise ValueError('Saved sample identity differs')
        result = trace.inspect_case(geometry, record, layers, targets)
        totals.update(images=1, final_graph_available=result['compiled_graph'] is not None,
                      whole_molecule_exact=record['prediction']['exact'])
        for node in result['focused_nodes']:
            focused.append(dict(sample_key=key, **node))
            totals.update(focused_nodes=1, **{node['local_status']: 1})
            totals['missing_incident_bonds'] += len(node['missing_bond_ids'] or [])
            totals['wrong_type_incident_bonds'] += len(node['wrong_type_bonds'] or [])
            totals['extra_incident_bonds'] += len(node['extra_edges'] or [])
            totals['expected_incident_pairs_in_current_negative_pool'] += sum(
                b['current_negative_pool_eligible'] is True for b in node['expected_incident_bonds'])
        path = OUTPUT / (key + '.json')
        write(path, result)
        case_hashes[path.name] = sha256(path.read_bytes()).hexdigest()
    for path, digest in list(inputs.hashes.items()):
        inputs.read(path, digest)
    if torch.cuda.is_initialized():
        raise ValueError('Unexpected CUDA initialization')
    write(OUTPUT / 'report.json', dict(schema_version='missing10-graph-trace-0.1',
        totals=totals, focused_nodes=focused, case_sha256=case_hashes, input_sha256=inputs.hashes,
        model_forwards=0, optimizer_steps=0, gpu_used=False, training_performed=False,
        training_authorized=False, default_changed=False, thresholds_changed=False,
        validation_sample_artifacts_read=False, sealed_test_read=False, new_identity_queries=0,
        limitations=['Coordinate-only association is diagnostic evidence, not a new acceptance radius or training labels',
            'Local comparisons cover elements, incident endpoints and bond types, not complete atom/stereo equivalence',
            'Missing saved MolBlocks remain unresolved; no pre-compile graph was reconstructed',
            'Current negative-pool replay does not prove historical sampling, gradient impact or training causality',
            'Selected old-training cases are not an independent recognition accuracy estimate']))
    print(json.dumps(dict(totals=totals), ensure_ascii=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        worker()
        return 0
    preflight(Inputs())
    receipt = runtime.run_budgeted([sys.executable, '-m', 'tools.audit_selector_missing_graph_trace', '--worker'],
        output=OUTPUT, budget=runtime.ResourceBudget(120))
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
