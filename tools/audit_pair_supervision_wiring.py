"""Runtime sampler dry-run on fixed58 saved candidates; no model or optimizer."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

from chemvector.bond_object import BOND_TYPE_TO_ID
from chemvector.direct_atom_pair_bond import enumerate_unordered_atom_pairs
from chemvector.train_direct_atom_pair_bond import build_pair_targets, match_atom_targets
from chemvector.training.curriculum import pair_supervision as supervision
from chemvector.training.curriculum import continuation_pair_supervision as wired
from chemvector.training.curriculum import runtime
from tools import audit_selector_stable_subset as source
from tools import selector_missing_graph_trace as trace

ROOT = source.ROOT
PREVIOUS = ROOT / 'outputs/audits/chemvector-pair-supervision-mask58-20260921'
PREVIOUS_SHA = 'e94be067444ce1cbaa6a1f507365f958c94a5a40264fc381a18299c9e8b314e5'
OUTPUT = ROOT / 'outputs/audits/chemvector-pair-supervision-wiring58-20260921'


def preflight(inputs):
    previous = inputs.json(PREVIOUS / 'report.json', PREVIOUS_SHA)
    _, manifest, rows, bindings = source.preflight(inputs)
    if set(previous['case_sha256']) != {r['sample_key'] + '.json' for r in rows}:
        raise ValueError('Original fixed58 mask results required')
    return previous, manifest, rows, bindings


def tensors(geometry, record, layers):
    selected = trace.selected_candidates(geometry, record)
    capacity = max(c['slot'] for c in geometry['candidates']) + 1
    xy, mask = torch.zeros(1, capacity, 8), torch.zeros(1, capacity, dtype=torch.bool)
    for c in selected:
        xy[0, c['slot']] = torch.tensor(c['geometry'])
        mask[0, c['slot']] = True
    atoms = dict(atom_geometry=xy, atom_mask=mask, atom_indices=torch.arange(capacity)[None])
    bonds = layers['bonds']
    targets = dict(atom_anchors=torch.tensor([geometry['target_anchors']], dtype=torch.float32),
        atom_mask=torch.ones(1, len(layers['atoms']), dtype=torch.bool),
        bond_mask=torch.ones(1, len(bonds), dtype=torch.bool),
        bond_atom_refs=torch.tensor([[[b['begin_atom'], b['end_atom']] for b in bonds]],
                                   dtype=torch.long).reshape(1, len(bonds), 2),
        bond_chemical=torch.tensor([[BOND_TYPE_TO_ID[b['chemical_type']] for b in bonds]], dtype=torch.long))
    mapping, matched = match_atom_targets(atoms, targets, SimpleNamespace(image_size=768, atom_match_radius_px=6))
    if matched[0].nonzero().flatten().tolist() != record['metrics']['matched_target_ids']:
        raise ValueError('Historical original selected matching changed')
    pairs, samples = enumerate_unordered_atom_pairs(mask)
    labels = build_pair_targets(pairs, samples, mapping, targets, atoms)
    return pairs, samples, mapping, atoms, labels


def check_case(geometry, record, layers, reference, *, seed):
    pairs, samples, mapping, atoms, labels = tensors(geometry, record, layers)
    by_pair = {tuple(r['pair']): r for r in reference['pairs']}
    if len(by_pair) != len(pairs):
        raise ValueError('Historical pair population differs')
    expected = torch.tensor([by_pair[tuple(p)]['matched_only'] != 'ignore' for p in pairs.tolist()], dtype=torch.bool)
    original = torch.tensor([by_pair[tuple(p)]['original_presence'] for p in pairs.tolist()], dtype=torch.bool)
    if not torch.equal(original, labels['presence']):
        raise ValueError('Historical original positive labels changed')
    outputs = {}
    before_rng = torch.get_rng_state().clone()
    for policy in supervision.POLICIES:
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            chosen, mask = supervision.sample_supervised_pairs(pairs, samples, mapping, atoms, labels, policy=policy)
        if (len(chosen) != len(chosen.unique()) or not bool(mask[chosen].all())
                or set(labels['presence'].nonzero().flatten().tolist()) - set(chosen.tolist())):
            raise ValueError('Sampled duplicate, ignored row or missing original positive')
        if policy == supervision.MATCHED_ONLY and not torch.equal(mask, expected):
            raise ValueError('Runtime supervision mask differs from audited prototype')
        if policy == supervision.LEGACY:
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                old = supervision.select_training_pairs(pairs, samples, atoms, labels,
                                                       negative_ratio=6, minimum_negatives=48)
            if not torch.equal(old, chosen):
                raise ValueError('Default sampler changed original row IDs')
        name = 'current' if policy == supervision.LEGACY else 'matched_only'
        count = reference['summary'][name]
        positives, negatives = int(labels['presence'][chosen].sum()), int((~labels['presence'][chosen]).sum())
        if positives != count['positive'] or negatives != count['negative_sampling_capacity']:
            raise ValueError('Actual sampler failed original positive or per-image quota contract')
        risk = [p for p in chosen.tolist() if not labels['presence'][p]
                and by_pair[tuple(pairs[p].tolist())]['diagnostic_reference'] == 'bond']
        outputs[policy] = dict(sampled_row_ids=chosen.tolist(), sampled_candidate_pairs=pairs[chosen].tolist(),
            positive_rows=positives, negative_rows=negatives, ignored_pool_rows=int((~mask).sum()),
            diagnostic_risk_negative_rows=risk, empty_pair_branch=len(chosen) == 0)
        if policy == supervision.MATCHED_ONLY and risk:
            raise ValueError('A diagnosed risk pair entered the masked sampler')
    if not torch.equal(before_rng, torch.get_rng_state()):
        raise ValueError('Dry-run leaked its random state')
    return dict(sample_key=record['sample_key'], seed=seed, policies=outputs,
                original_presence=labels['presence'].tolist(), training_authorized=False,
                model_forward_performed=False, parameter_update_performed=False)


def worker():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('CPU-only saved-candidate wiring check required')
    torch.set_num_threads(2)
    inputs = source.Inputs()
    previous, manifest, rows, bindings = preflight(inputs)
    for path in (Path(__file__), Path(supervision.__file__), Path(wired.__file__), Path(source.__file__),
                 Path(trace.__file__), Path(runtime.__file__),
                 ROOT / 'chemvector/training/curriculum/continuation.py',
                 ROOT / 'chemvector/training/graph/relations/train_direct_atom_pair_bond.py',
                 ROOT / 'frontend/inference_profiles.json',
                 ROOT / 'tests/chemvector/test_curriculum_pair_supervision.py'):
        inputs.read(path, previous['input_sha256'].get(str(path), bindings.get(str(path))))
    source.write(OUTPUT / 'plan.json', dict(schema_version='pair-wiring-plan-0.1', images=58,
        sample_keys=[r['sample_key'] for r in rows], policies=list(supervision.POLICIES),
        wall_budget_seconds=120, gpu_devices=[], optimizer_steps=0, model_forwards=0,
        checkpoint_loads=0, real_saved_candidate_sampling=True, training_authorized=False,
        maximum_policy_sampler_calls=116, legacy_exact_replay_comparisons=58,
        seed_rule='20260921 + original manifest row ordinal; same seed for both policies',
        automatic_retries=0, production_launcher_changed=False, default_changed=False,
        scope='Actual runtime masks and sampler only; full head/loss wiring is tested separately on synthetic tensors'))
    totals = {p: Counter() for p in supervision.POLICIES}
    case_hashes = {}
    for ordinal, row in enumerate(rows):
        key = row['sample_key']
        base = Path(manifest['sources'][row['source']]) / 'samples'
        meta = inputs.json(base / (key + '.meta.json'), row['meta_sha256'])
        layers = inputs.json(base / (key + '.layers.json'), meta['sha256']['.layers.json'])
        geo_path, rec_path = source.CONTROL / ('geometry--' + key + '.json'), source.CONTROL / ('baseline--' + key + '.json')
        geometry, record = inputs.json(geo_path, bindings[str(geo_path)]), inputs.json(rec_path, bindings[str(rec_path)])
        reference = inputs.json(PREVIOUS / (key + '.json'), previous['case_sha256'][key + '.json'])
        if (record['sample_key'] != key or reference['sample_key'] != key or layers['sample_key'] != key
                or record['key'] != row['source'] + ':' + key or geometry['key'] != record['key']
                or layers['molecule_id'] != row['molecule_id']):
            raise ValueError('Saved source identity changed')
        result = check_case(geometry, record, layers, reference, seed=20260921 + ordinal)
        for policy, counts in result['policies'].items():
            totals[policy].update(images=1, sampled_positive_rows=counts['positive_rows'],
                sampled_negative_rows=counts['negative_rows'], ignored_pool_rows=counts['ignored_pool_rows'],
                sampled_diagnostic_risk_negatives=len(counts['diagnostic_risk_negative_rows']),
                empty_pair_branches=counts['empty_pair_branch'])
        path = OUTPUT / (key + '.json')
        source.write(path, result)
        case_hashes[path.name] = sha256(path.read_bytes()).hexdigest()
    for path, digest in list(inputs.hashes.items()):
        inputs.read(path, digest)
    if torch.cuda.is_initialized():
        raise ValueError('Unexpected CUDA initialization')
    source.write(OUTPUT / 'report.json', dict(schema_version='pair-wiring58-0.1', totals=totals,
        case_sha256=case_hashes, input_sha256=inputs.hashes, original_sampler_exact_replay_passed=58,
        masks_match_offline_reference_images=58, positive_and_quota_contract_passed_images=58,
        model_forwards=0, optimizer_steps=0, checkpoint_loads=0, gpu_used=False,
        training_performed=False, training_authorized=False, production_launcher_changed=False,
        default_changed=False, historical_implementations_changed=False,
        validation_sample_artifacts_read=False, sealed_test_read=False, new_identity_queries=0,
        limitations=['Actual sampling uses one fixed CPU seed per image, not historical sampled gradients',
            'No full real-checkpoint forward/backward was run; full loss tests use synthetic models/candidates',
            'Ignoring risk negatives does not create missing positives or establish an accuracy gain',
            'The versioned loss entry is opt-in; old launchers and checkpoint metadata remain unchanged']))
    print(json.dumps(dict(totals=totals), ensure_ascii=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        worker()
        return 0
    preflight(source.Inputs())
    receipt = runtime.run_budgeted([sys.executable, '-m', 'tools.audit_pair_supervision_wiring', '--worker'],
                                  output=OUTPUT, budget=runtime.ResourceBudget(120))
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
