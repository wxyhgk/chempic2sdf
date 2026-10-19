"""Fixed four-image real initial CPU forward/pair-backward; never optimize."""

import argparse
from hashlib import sha256
import inspect
import json
import os
from pathlib import Path
import sys

import torch

from chemvector.training.curriculum import bundle, data, runtime
from chemvector.training.curriculum import continuation_pair_supervision as wired
from chemvector.training.curriculum import pair_supervision as supervision
from chemvector.training.curriculum.gradient_preflight import state_digest
from tools import audit_selector_stable_subset as source
from tools import pair_initial_dryrun as diagnostic
from tools import selector_candidate_parity as parity

ROOT = source.ROOT
PREVIOUS = ROOT / 'outputs/audits/chemvector-pair-supervision-wiring58-20260921/report.json'
PREVIOUS_SHA = '3699d01515776b5e37996ae8f9e2089a18af0bd847aca32f4eb725cf41d0fe83'
INITIAL = ROOT / 'outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt'
INITIAL_SHA = '09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1'
INITIAL_STATE = '9d7f66f79329479e05c38f8b91d95118adb9f12c32a44e5bd9d83a458a8b452e'
OUTPUT = ROOT / 'outputs/audits/chemvector-pair-initial-dryrun4-20260921'
SCOPE = (
    ('curr_04a03f058eccd278__reference', 'selected_endpoints_matched_control'),
    ('curr_ddc8766e5cc2a898__reference', 'historical_empty_supervised_pair_branch'),
    ('curr_00f41c00f7bf4670__font_large', 'three_original_text_localization_misses'),
    ('curr_6de40b2274fd34b2__serif', 'historically_correct_molecule_with_ho_localization_miss'),
)


def preflight(inputs):
    previous = inputs.json(PREVIOUS, PREVIOUS_SHA)
    _, manifest, rows, bindings = source.preflight(inputs)
    by_key = {r['sample_key']: r for r in rows}
    selected = [by_key[k] for k, reason in SCOPE]
    if len(set(r['sample_key'] for r in selected)) != 4:
        raise ValueError('Exactly four fixed training/replay cases required')
    return previous, manifest, selected, bindings


def worker():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('CPU-only diagnostic required')
    torch.set_num_threads(2)
    torch.manual_seed(20260921)
    inputs = source.Inputs()
    previous, manifest, rows, bindings = preflight(inputs)
    for path in (Path(__file__), Path(diagnostic.__file__), Path(parity.__file__), Path(bundle.__file__),
                 Path(data.__file__), Path(runtime.__file__), Path(wired.__file__), Path(supervision.__file__),
                 Path(diagnostic.legacy.__file__), Path(source.__file__),
                 Path(inspect.getfile(state_digest)), Path(inspect.getfile(diagnostic.tensor_tree_digest)),
                 ROOT / 'tests/chemvector/test_pair_initial_dryrun.py', ROOT / 'frontend/inference_profiles.json'):
        inputs.read(path, previous['input_sha256'].get(str(path), bindings.get(str(path))))
    inputs.read(INITIAL, INITIAL_SHA)
    roots = {}
    for row in rows:
        key = row['sample_key']
        roots[key] = Path(manifest['sources'][row['source']])
        base = roots[key] / 'samples'
        meta = inputs.json(base / (key + '.meta.json'), row['meta_sha256'])
        for suffix in data.PAYLOADS:
            inputs.read(base / (key + suffix), meta['sha256'][suffix])
        for prefix in ('geometry', 'baseline'):
            path = source.CONTROL / (prefix + '--' + key + '.json')
            inputs.read(path, bindings[str(path)])
    source.write(OUTPUT / 'plan.json', dict(schema_version='pair-initial-dryrun-plan-0.1',
        scope=[dict(sample_key=k, purpose=reason) for k, reason in SCOPE],
        checkpoint_sha256=INITIAL_SHA, initial_state_sha256=INITIAL_STATE,
        input_sha256=dict(inputs.hashes), maximum_full_forwards=8, maximum_pair_backwards=8,
        wall_budget_seconds=240, gpu_devices=[], threads=2, dtype='float32', autocast=False,
        torch_version=torch.__version__, mode='eval; only pair parameters require grad',
        policies=list(supervision.POLICIES), atom_threshold=-.25, bond_threshold=-.5,
        optimizer_created=False, optimizer_steps=0, training_authorized=False, automatic_retries=0,
        historical_policy='Report CPU/GPU differences separately; never reuse historical slot labels',
        paired_policy='Exact current CPU upstream/context and non-pair loss equality required',
        extra_decoder_observations=8, default_changed=False, checkpoint_writes=0))
    model, config, payload = bundle.load_bundle(INITIAL, expected_sha256=INITIAL_SHA)
    if (state_digest(model) != INITIAL_STATE or payload['progress']['optimizer_steps'] != 0
            or model.update_recipe != 'freeze-role-v1' or config.image_size != 768
            or config.maximum_atoms != 128 or config.atom_match_radius_px != 6):
        raise ValueError('Initial model/config identity differs')
    if payload['provenance']['atom_threshold'] != -.25 or payload['provenance']['bond_threshold'] != -.5:
        raise ValueError('Original loss thresholds differ')
    model.eval().requires_grad_(False)
    model.pair.requires_grad_(True)
    for name in ('vision', 'atom', 'bond', 'set_head', 'pair', 'role', 'bbox'):
        inputs.read(Path(inspect.getfile(type(getattr(model, name)))))
    collator = data.CurriculumCollator(purpose='diagnostic')
    completed, case_hashes, forwards, backwards = [], {}, 0, 0
    with (OUTPUT / 'events.jsonl').open('x', encoding='utf-8') as log:
        def emit(event, **values):
            text = json.dumps(dict(event=event, **values), allow_nan=False)
            log.write(text + '\n'); log.flush(); os.fsync(log.fileno())
            print(text, flush=True)
        for ordinal, row in enumerate(rows):
            key = row['sample_key']
            sample = data.load_sample(roots[key], key, purpose='diagnostic')
            batch = collator([sample])
            geometry = inputs.json(source.CONTROL / ('geometry--' + key + '.json'),
                                   bindings[str(source.CONTROL / ('geometry--' + key + '.json'))])
            baseline = inputs.json(source.CONTROL / ('baseline--' + key + '.json'),
                                   bindings[str(source.CONTROL / ('baseline--' + key + '.json'))])
            if batch['atom_anchors'][0, batch['atom_mask'][0]].tolist() != geometry['target_anchors']:
                raise ValueError('Fresh target adapter changed original anchor contract')
            arms, snapshots = {}, {}
            for policy in supervision.POLICIES:
                if forwards >= 8:
                    raise ValueError('Forward budget exhausted')
                forwards += 1
                emit('forward_reserved', sample_key=key, policy=policy, ordinal=forwards)
                arm, snapshot = diagnostic.run_arm(model, batch, config, policy=policy,
                    atom_threshold=-.25, bond_threshold=-.5, seed=20260921 + ordinal)
                arms[policy], snapshots[policy] = arm, snapshot
                backwards += arm['pair_backward_performed']
                emit('arm_complete', sample_key=key, policy=policy, losses=arm['losses'],
                     pair_rows=arm['coverage']['pair_rows'], backward=arm['pair_backward_performed'])
            comparison = diagnostic.compare_arms(arms[supervision.LEGACY], arms[supervision.MATCHED_ONLY])
            snapshot = snapshots[supervision.LEGACY]
            historical = parity.compare(geometry, baseline['metrics']['logits'], snapshot)
            history_pass = (historical['may_reuse_historical_slot_labels']
                and historical['selected_grid_ids_equal'] and historical['eligible_grid_target_relations_equal']
                and historical['max_common_logit_change'] <= 1e-3 and historical['max_common_xy_change_px'] <= .01)
            result = dict(sample_key=key, purpose=SCOPE[ordinal][1], arms=arms,
                current_cpu_comparison=comparison, historical_candidate_comparison=historical,
                historical_candidate_gate_pass=history_pass, historical_labels_reused=False,
                fresh_candidate_snapshot={k: v.tolist() for k, v in snapshot.items()},
                model_state_unchanged=state_digest(model) == INITIAL_STATE,
                training_authorized=False, optimizer_steps=0)
            path = OUTPUT / (key + '.json')
            source.write(path, result)
            case_hashes[path.name] = sha256(path.read_bytes()).hexdigest()
            completed.append(dict(sample_key=key, historical_candidate_gate_pass=history_pass,
                **comparison, old_pair_rows=arms[supervision.LEGACY]['coverage']['pair_rows'],
                new_pair_rows=arms[supervision.MATCHED_ONLY]['coverage']['pair_rows'],
                original_positive_rows=len(arms[supervision.LEGACY]['original_positive_rows']),
                ignored_pool_rows=arms[supervision.MATCHED_ONLY]['ignored_rows']))
            emit('case_complete', **completed[-1])
    if (forwards != 8 or state_digest(model) != INITIAL_STATE or torch.cuda.is_initialized()
            or any(p.grad is not None for p in model.parameters())):
        raise ValueError('Final scope or model/gradient state changed')
    for path, digest in list(inputs.hashes.items()):
        inputs.read(path, digest)
    source.write(OUTPUT / 'report.json', dict(schema_version='pair-initial-dryrun4-0.1',
        cases=completed, case_sha256=case_hashes, input_sha256=inputs.hashes,
        full_forwards=forwards, pair_backwards=backwards, optimizer_created=False, optimizer_steps=0,
        checkpoint_loads=1, checkpoint_writes=0, gpu_used=False, ocr_loaded=False,
        training_performed=False, training_authorized=False, default_changed=False,
        initial_state_before=INITIAL_STATE, initial_state_after=state_digest(model),
        current_cpu_arm_comparisons_passed=4,
        historical_candidate_gate_passed=sum(c['historical_candidate_gate_pass'] for c in completed),
        validation_sample_artifacts_read=False, sealed_test_read=False, new_identity_queries=0,
        limitations=['Eval-mode CPU smoke test, not training-mode/GPU efficacy or SDF accuracy',
            'Only pair gradients were computed; frozen upstream gradients and cross-head training interaction untested',
            'Losses use different sampled rows after masking and are not directly comparable as improvement metrics',
            'Historical mismatch is not repaired; only exact fresh CPU arm parity isolates the mask',
            'No positive labels were added and missing graph connections were not recovered']))
    print(json.dumps(dict(completed_images=4, forwards=forwards, backwards=backwards), ensure_ascii=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        worker()
        return 0
    preflight(source.Inputs())
    receipt = runtime.run_budgeted([sys.executable, '-m', 'tools.audit_pair_initial_dryrun', '--worker'],
                                  output=OUTPUT, budget=runtime.ResourceBudget(240))
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
