"""One CPU-only full fixed-validation comparison of a saved presence delta.

No optimizer, train sample loading, tuning, retries, or deployment. Original
image-derived upstream predictions are shared; both pair/compiler paths run.
"""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time

from .checkpoints import file_digest
from .interface_diagnostic import write_json, tensor_tree_digest
from .presence_regression_metrics import ARMS, fixed_validation_rows, sample_key, parity_differences, summarize
from .runtime import ResourceBudget, run_budgeted

RUN = Path('outputs/experiments/chemvector-selected-matching-trial-20260915')
RELEASE = Path('outputs/datasets/chemvector-expanded-release-20260915')
TRIAL = Path('outputs/experiments/chemvector-presence-readout-probe-20260918')
OUTPUT = Path('outputs/audits/chemvector-presence-full-regression-20260919')
WALL_SECONDS = 1800
PARITY_INDICES = (0, 472, 943)
INITIAL_SHA = '09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1'
DELTA_SHA = '4746cca5360c57bf1d057972d58f1878ac9b73e5bd5181688b1c8b46253866b1'
TRIAL_REPORT_SHA = '461fe4eff8eea90cda449481bc698052e3130dc51557bda9dd5c2bd7a57a9a26'
SAVED_BASELINE_SHA = 'be9a12c66c96ee8ecac925c707cea6159fc16fec0864f94fc5395631d1cedb89'


def verify_inputs(bindings):
    for path, expected in bindings.items():
        if file_digest(path) != expected:
            raise ValueError('Regression input changed: ' + str(path))


def patched_model(initial, delta, trial):
    """Restore a previously fitted delta into an independent frozen copy."""
    import torch
    from .gradient_preflight import state_digest
    from .presence_probe import frozen_state_digest
    if (delta.get('schema_version') != 'chemvector-presence-readout-delta-0.1'
            or delta.get('base_initial_sha256') != INITIAL_SHA or delta.get('optimizer_steps') != 200
            or delta.get('not_standalone_model') is not True or delta.get('deployed') is not False
            or delta.get('resume_supported') is not False or state_digest(initial) != trial['model_state_before']):
        raise ValueError('Delta does not bind the original initial and completed fixed trial')
    model = deepcopy(initial).eval().requires_grad_(False)
    values = {k: torch.tensor(delta[k], dtype=getattr(model.pair.presence, k).dtype) for k in ('weight', 'bias')}
    if any(v.shape != getattr(model.pair.presence, k).shape or not bool(torch.isfinite(v).all())
           for k, v in values.items()):
        raise ValueError('Invalid saved presence delta shape/values')
    with torch.no_grad():
        for k, value in values.items():
            getattr(model.pair.presence, k).copy_(value)
    if (state_digest(model) != trial['model_state_after']
            or frozen_state_digest(model) != trial['frozen_state_after']
            or frozen_state_digest(initial) != trial['frozen_state_before']
            or trial['frozen_state_before'] != trial['frozen_state_after']):
        raise ValueError('Restored delta changed a non-presence state or differs from trial')
    return model


def preflight():
    from . import (evaluation, continuation, release, fresh_dev, presence_regression_metrics,
                   bundle, pixels, validation_truth, checkpoints, presence_probe)
    from .presence_release import disjoint_training_validation
    required = {str(RUN / 'initial.pt'): INITIAL_SHA, str(TRIAL / 'last-presence.json'): DELTA_SHA,
                str(TRIAL / 'report.json'): TRIAL_REPORT_SHA, str(RUN / 'validation-000000.json'): SAVED_BASELINE_SHA}
    verify_inputs(required)
    manifest, _, bindings = release.verify_release(RELEASE / 'manifest.json', RELEASE / 'admission.json')
    original = json.loads((RUN / 'run.json').read_text())
    if original['bindings'] != bindings:
        raise ValueError('Full regression must use exactly the original validation release')
    saved = json.loads((RUN / 'validation-000000.json').read_text())['predictions']
    rows, previous = fixed_validation_rows(manifest, saved)
    trial = json.loads((TRIAL / 'report.json').read_text())
    trial_plan = json.loads((TRIAL / 'plan.json').read_text())
    runtime = json.loads((TRIAL / 'runtime.json').read_text())
    delta = json.loads((TRIAL / 'last-presence.json').read_text())
    if (trial['plan_sha256'] != file_digest(TRIAL / 'plan.json') or trial['presence_delta_sha256'] != DELTA_SHA
            or trial['optimizer_steps'] != 200 or trial['deployed'] or trial['gpu_used']
            or runtime['status'] != 'completed' or runtime['gpu_devices'] or runtime['elapsed_seconds'] > 300
            or delta['authorization_sha256'] != trial_plan['authorization_sha256']):
        raise ValueError('Expected the completed, undeployed, bounded CPU trial')
    disjoint_training_validation(trial_plan['samples'], rows)
    files = [RUN / 'run.json', RELEASE / 'manifest.json', RELEASE / 'admission.json',
             TRIAL / 'plan.json', TRIAL / 'runtime.json', Path('frontend/inference_profiles.json'), Path(__file__)]
    files += [Path(m.__file__) for m in (evaluation, continuation, release, fresh_dev, presence_regression_metrics,
                                       bundle, pixels, validation_truth, checkpoints, presence_probe)]
    for path in files:
        required[str(path)] = file_digest(path)
    plan = {'schema_version': 'chemvector-presence-regression-plan-0.1', 'samples': rows,
        'input_sha256': required, 'release_bindings': bindings, 'arms': ARMS,
        'thresholds': {k: v for k, v in original['provenance'].items() if k.endswith('_threshold')},
        'images': 944, 'full_molecule_images': 800, 'curriculum_images': 144,
        'planned_generator_forwards': 950, 'default_path_replay_indices': PARITY_INDICES,
        'wall_seconds': WALL_SECONDS, 'gpu_used': False, 'optimizer_steps': 0,
        'automatic_retries': 0, 'training_performed': False, 'deployed': False,
        'selection': 'All original validation rows, unchanged order and denominators; no new selection',
        'historical_parity_policy': 'Report every saved-vs-CPU mismatch; never substitute historical scores for fresh paired CPU scores',
        'scope': 'Read-only full development regression, not sealed testing, training, threshold tuning or deployment'}
    return manifest, rows, previous, original, trial, delta, plan


def paired_predictions(models, config, predictor, pixels, image, *, replay=False):
    """Only image/model inputs. No ground truth or training annotations accepted."""
    import torch
    from .continuation import structure_forward
    from .presence_regression_metrics import parity_differences
    from .upstream_diagnostic import score_prediction
    with torch.no_grad():
        forward = structure_forward(models['initial'], pixels, config, predictor.provenance['atom_threshold'])
        structures = {arm: predictor.from_predicted_atoms(model.pair, forward[0], forward[6])
                      for arm, model in models.items()}
        signatures = {arm: tensor_tree_digest({'atoms': s['atoms'], 'pairs': s['pairs'],
            'hidden': s['output']['hidden'], 'chemical_logits': s['output']['chemical_logits']})
            for arm, s in structures.items()}
        if len(set(signatures.values())) != 1:
            raise ValueError('Shared upstream/chemical predictions differ between arms')
        predictions, traces = {}, {}
        for arm, model in models.items():
            traces[arm] = {}
            predictions[arm] = predictor.from_structure(structures[arm], model.role, model.bbox,
                                                         pixels, image, trace=traces[arm])
            if replay:
                direct = predictor(model, config, pixels, image)
                # Expected identity is deliberately absent from both inference paths.
                if parity_differences(score_prediction(predictions[arm], None), score_prediction(direct, None)):
                    raise ValueError('Cached upstream differs from the unchanged direct inference path')
        if any(traces['initial'][k] != traces['presence_delta'][k] for k in ('roles', 'ocr')):
            raise ValueError('Frozen text/OCR predictions changed between arms')
        pair_changed = structures['initial']['output']['presence_logits'].sigmoid().ge(predictor.provenance['pair_threshold']) \
            != structures['presence_delta']['output']['presence_logits'].sigmoid().ge(predictor.provenance['pair_threshold'])
        return predictions, traces, signatures['initial'], int(pair_changed.sum())


def audit():
    import torch
    from .bundle import load_bundle
    from .checkpoints import verify_sources
    from .evaluation import FrozenOCRPredictor
    from .gradient_preflight import state_digest
    from .pixels import image_tensor
    from .release import RecordLoader
    from .upstream_diagnostic import score_prediction
    from .validation_truth import expected_identity
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('CPU-only external supervisor required')
    if not OUTPUT.is_dir() or any((OUTPUT / name).exists() for name in ('plan.json', 'report.json')):
        raise ValueError('Fresh supervised output required; no retry')
    torch.set_num_threads(2)
    torch.manual_seed(20260919)
    started = time.monotonic()
    manifest, rows, previous, original, trial, delta, plan = preflight()
    write_json(OUTPUT / 'plan.json', plan)
    print(json.dumps({'event': 'plan_fixed', 'images': len(rows), 'wall_seconds': WALL_SECONDS}), flush=True)
    initial, config, payload = load_bundle(RUN / 'initial.pt', expected_sha256=INITIAL_SHA)
    if payload['provenance'] != original['provenance'] or payload['progress']['optimizer_steps'] != 0:
        raise ValueError('Initial inference provenance changed')
    initial.eval().requires_grad_(False)
    models = {'initial': initial, 'presence_delta': patched_model(initial, delta, trial)}
    predictor = FrozenOCRPredictor(original['provenance'], 'cpu')
    before = {a: state_digest(m) for a, m in models.items()}
    before['ocr'] = state_digest(predictor.ocr)
    if before['ocr'] != trial['ocr_state_before'] or before['ocr'] != trial['ocr_state_after']:
        raise ValueError('Original OCR state differs')
    loader, cases, replays = RecordLoader(manifest), [], 0
    with torch.no_grad():
        for index, row in enumerate(rows):
            sample = loader(manifest, row)
            if sample.get('purpose') != 'diagnostic':
                raise ValueError('Only original diagnostic validation records may be loaded')
            pixels = image_tensor(sample['image']).unsqueeze(0)
            predictions, traces, frozen_signature, changed = paired_predictions(
                models, config, predictor, pixels, sample['image'], replay=index in PARITY_INDICES)
            expected = expected_identity(sample)
            if expected != row['canonical_smiles']:
                raise ValueError('Original validation truth changed')
            arms = {a: score_prediction(v, expected) for a, v in predictions.items()}
            case = {'key': sample_key(row), 'sample_key': row['sample_key'], 'molecule_id': row['molecule_id'],
                'domain': row['domain'], 'style': row.get('variant', row['sample_key'].split('__')[-1]),
                'arms': arms, 'historical_differences': parity_differences(arms['initial'], previous[sample_key(row)]),
                'frozen_signature': frozen_signature, 'thresholded_pair_changes': changed,
                'default_path_parity_checked': index in PARITY_INDICES}
            write_json(OUTPUT / (row['sample_key'] + '.json'), {**case, 'traces': traces})
            cases.append(case)
            replays += 2 * (index in PARITY_INDICES)
            if (index + 1) % 10 == 0 or index + 1 == len(rows):
                print(json.dumps({'event': 'progress', 'completed': index + 1, 'total': len(rows),
                    'elapsed_seconds': time.monotonic() - started,
                    'exact': {a: sum(c['arms'][a]['exact'] for c in cases) for a in ARMS},
                    'historical_parity_mismatches': sum(bool(c['historical_differences']) for c in cases)}), flush=True)
    after = {a: state_digest(m) for a, m in models.items()}
    after['ocr'] = state_digest(predictor.ocr)
    if (before != after or torch.cuda.is_initialized() or replays != 6
            or any(p.requires_grad or p.grad is not None for m in (*models.values(), predictor.ocr) for p in m.parameters())):
        raise ValueError('State, gradient, device or direct-path replay guard failed')
    verify_inputs(plan['input_sha256'])
    verify_sources(original['provenance']['checkpoints'])
    summary = summarize(cases)
    parity = sum(bool(c['historical_differences']) for c in cases)
    report = {'schema_version': 'chemvector-presence-regression-result-0.1',
        'plan_sha256': file_digest(OUTPUT / 'plan.json'), 'summary': summary,
        'states_before': before, 'states_after': after, 'historical_baseline_parity_checks': len(cases),
        'historical_baseline_parity_mismatches': parity, 'historical_baseline_reproduced': parity == 0,
        'frozen_prediction_parity_checks': len(cases), 'default_path_replay_checks': replays,
        'generator_forwards': len(cases) + replays, 'optimizer_steps': 0, 'training_performed': False,
        'gpu_used': False, 'deployed': False, 'default_kept': 'initial', 'new_identity_queries': 0,
        'case_sha256': {r['sample_key'] + '.json': file_digest(OUTPUT / (r['sample_key'] + '.json')) for r in cases},
        'limits': ['200 full molecules each have four correlated styles; not 800 independent molecules',
                   'Development data previously used for diagnosis, not a fresh blind or sealed test',
                   'Curriculum score is separate and cannot replace the full-molecule score',
                   'Existing evaluation-internal grouping and global public identity coverage gaps remain',
                   'No retraining, threshold sweep, model selection or deployment is authorized by this diagnostic']}
    write_json(OUTPUT / 'report.json', report)
    print(json.dumps({'event': 'completed', 'summary': summary, 'historical_parity_mismatches': parity}), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        if args.check_only:
            parser.error('Worker cannot be combined with check-only')
        audit()
        return 0
    if args.check_only:
        *_, plan = preflight()
        print(json.dumps({'checks_passed': True, 'images': plan['images'], 'model_loaded': False,
                          'training_performed': False, 'wall_seconds': WALL_SECONDS}))
        return 0
    receipt = run_budgeted([sys.executable, '-m', 'chemvector.training.curriculum.presence_regression', '--worker'],
                           output=OUTPUT, budget=ResourceBudget(WALL_SECONDS))
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
