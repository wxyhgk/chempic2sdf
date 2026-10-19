"""One 32-image CPU initial-model opportunity audit; no training or oracle inputs."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

from .headroom_metrics import SEED, SAMPLE_COUNT, select_training_rows, measure, classify, summarize
from .interface_diagnostic import write_json, tensor_tree_digest, structure_trace
from .runtime import ResourceBudget, run_budgeted

RUN = Path('outputs/experiments/chemvector-selected-matching-trial-20260915')
RELEASE = Path('outputs/datasets/chemvector-expanded-release-20260915')
OUTPUT = Path('outputs/audits/chemvector-initial-headroom32-20260918')
PARITY_INDICES = (0, 15, 31)
INITIAL_STATE = '9d7f66f79329479e05c38f8b91d95118adb9f12c32a44e5bd9d83a458a8b452e'


def emit(event, **record):
    print(json.dumps({'event': event, **record}, allow_nan=False), flush=True)


def audit(output):
    import torch
    from chemvector.atom_bond_scene import AtomBondSceneConfig
    from . import evaluation, continuation, data, headroom_metrics, upstream_diagnostic
    from .bundle import load_bundle
    from .checkpoints import file_digest, verify_sources
    from .gradient_preflight import state_digest
    from .release import verify_release
    from .validation_truth import expected_identity
    from .upstream_diagnostic import candidate_trace, score_prediction, check_saved_replay

    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('CPU-only supervision required')
    torch.set_num_threads(2)
    torch.manual_seed(20260918)
    output = Path(output)
    if not output.is_dir() or any((output / p).exists() for p in ('plan.json', 'report.json')):
        raise ValueError('Fresh supervised output required')
    manifest, _, bindings = verify_release(RELEASE / 'manifest.json', RELEASE / 'admission.json')
    run = json.loads((RUN / 'run.json').read_text())
    previous = json.loads((RUN / 'report.json').read_text())
    binding = previous['selected']
    if (run['bindings'] != bindings or previous['bindings'] != bindings
            or Path(binding['path']).resolve() != (RUN / 'initial.pt').resolve()):
        raise ValueError('Expected original initial and the admitted release')
    selected = select_training_rows(manifest)
    files = [RUN / 'run.json', RUN / 'report.json', RELEASE / 'manifest.json', RELEASE / 'admission.json',
             Path('frontend/inference_profiles.json'), Path(__file__), Path(evaluation.__file__),
             Path(continuation.__file__), Path(data.__file__), Path(headroom_metrics.__file__),
             Path(upstream_diagnostic.__file__)]
    inputs = {str(p): file_digest(p) for p in files}
    for row in selected:
        meta = Path(manifest['sources'][row['source']]) / 'samples' / (row['sample_key'] + '.meta.json')
        if file_digest(meta) != row['meta_sha256']:
            raise ValueError('Admitted metadata changed')
        inputs[str(meta)] = row['meta_sha256']
    plan = {'schema_version': 'chemvector-initial-headroom-plan-0.1', 'selection_seed': SEED,
        'selection': 'First 32 hash-ordered training/full_molecule rows, distinct molecule_id and canonical_smiles; no model-based replacement',
        'samples': selected, 'initial': binding, 'expected_initial_state': INITIAL_STATE,
        'input_sha256': inputs, 'cpu_wall_seconds': 300, 'gpu_used': False, 'optimizer_steps': 0,
        'planned_images': SAMPLE_COUNT, 'default_path_replay_indices': PARITY_INDICES,
        'planned_generator_forwards': SAMPLE_COUNT + len(PARITY_INDICES),
        'thresholds': {k: v for k, v in run['provenance'].items() if k.endswith('_threshold')},
        'pair_screen': 'Observed pair error on a baseline-wrong image with bijective candidates, exact role/literal text path, matching basic atomic fields, no target stereo; not guaranteed SDF repair',
        'bbox_screen': 'Literal differences and box IoU are descriptive only; cannot distinguish crop vs OCR failure without a further controlled test',
        'scope': 'Read-only diagnostic use of admitted training images, not independent validation; no truth passed to any predictor'}
    write_json(output / 'plan.json', plan)
    emit('plan_fixed', images=SAMPLE_COUNT, sources=dict(Counter(r['source'] for r in selected)))
    model, config, payload = load_bundle(binding['path'], expected_sha256=binding['sha256'])
    if (payload['provenance'] != run['provenance'] or payload['progress']['optimizer_steps'] != 0
            or payload['progress']['bindings'] != bindings or state_digest(model) != INITIAL_STATE):
        raise ValueError('Initial state/provenance mismatch')
    model.eval().requires_grad_(False)
    predictor = evaluation.FrozenOCRPredictor(run['provenance'], 'cpu')
    before = {'model': state_digest(model), 'ocr': state_digest(predictor.ocr)}
    collator = data.CurriculumCollator(AtomBondSceneConfig(atom_queries=config.maximum_atoms,
        bond_queries=config.maximum_bonds), purpose='diagnostic')
    cases, parity, styles = [], 0, Counter()
    with torch.inference_mode():
        for index, row in enumerate(selected):
            sample = data.load_sample(manifest['sources'][row['source']], row['sample_key'], purpose='diagnostic')
            if (sample['meta']['split'] != 'train' or sample['meta']['training_authorized'] is not True
                    or sample['meta']['external_overlap_status'] != 'checked'
                    or sample['meta']['molecule_id'] != row['molecule_id']
                    or sample['graph']['canonical_smiles'] != row['canonical_smiles']):
                raise ValueError('Loaded diagnostic differs from admitted training record')
            batch = collator([sample])
            # Inference accepts pixels and model outputs only, never this batch's labels.
            pixels = batch['pixel_values']
            forward = continuation.structure_forward(model, pixels, config, run['provenance']['atom_threshold'])
            structure = predictor.from_predicted_atoms(model.pair, forward[0], forward[6])
            original_structure_hash = tensor_tree_digest(structure)
            trace = {}
            predicted = predictor.from_structure(structure, model.role, model.bbox, pixels, sample['image'], trace=trace)
            expected = expected_identity(sample)
            prediction = score_prediction(predicted, expected)
            if index in PARITY_INDICES:
                replay = score_prediction(predictor(model, config, pixels, sample['image']), expected)
                check_saved_replay(prediction, replay)
                parity += 1
            metrics = measure(sample, batch, forward[3], structure, trace, config, run['provenance']['pair_threshold'])
            if tensor_tree_digest(structure) != original_structure_hash:
                raise ValueError('Scoring or text inference changed structure predictions')
            case = {'key': row['source'] + ':' + row['sample_key'], 'molecule_id': row['molecule_id'],
                'style': sample['meta']['profile_id'], 'prediction': prediction, 'metrics': metrics,
                'trace': trace, 'candidates': candidate_trace(forward[3], forward[6], forward[5]),
                'structure': structure_trace(structure), 'default_path_parity_checked': index in PARITY_INDICES}
            case['flags'] = classify(case)
            write_json(output / (row['sample_key'] + '.json'), case)
            cases.append(case)
            styles[case['style']] += 1
            emit('case', completed=len(cases), key=case['key'], exact=prediction['exact'],
                 flags=[k for k, v in case['flags'].items() if v])
    after = {'model': state_digest(model), 'ocr': state_digest(predictor.ocr)}
    if before != after or torch.cuda.is_initialized() or parity != len(PARITY_INDICES):
        raise ValueError('Model state, device or parity guard failed')
    if any(p.grad is not None for p in model.parameters()) or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Diagnostic model acquired gradients or trainable parameters')
    for path, digest in inputs.items():
        if file_digest(path) != digest:
            raise ValueError('Bound input/source changed during audit')
    if file_digest(binding['path']) != binding['sha256']:
        raise ValueError('Initial checkpoint changed')
    verify_sources(run['provenance']['checkpoints'])
    result = {'schema_version': 'chemvector-initial-headroom-result-0.1',
        'plan_sha256': file_digest(output / 'plan.json'), 'summary': summarize(cases), 'styles': dict(styles),
        'states_before': before, 'states_after': after, 'default_path_parity_checks': parity,
        'gpu_used': False, 'optimizer_steps': 0, 'training_performed': False, 'deployed': False,
        'case_files': {r['key']: r['key'].split(':', 1)[1] + '.json' for r in cases},
        'limits': ['32 already-admitted training molecules, not a new held-out accuracy measurement',
                   'No truth-fed candidates, crops, graph repair, or counterfactual SDF upper bound',
                   'Literal OCR string differences are not necessarily chemical identity errors',
                   'Basic atom-field equality excludes H/aromatic/stereo equivalence',
                   'Observed headroom does not authorize or prove benefit from optimization']}
    write_json(output / 'report.json', result)
    emit('completed', summary=result['summary'])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        audit(OUTPUT)
        return 0
    receipt = run_budgeted([sys.executable, '-m', 'chemvector.training.curriculum.headroom_audit', '--worker'],
                           output=OUTPUT, budget=ResourceBudget(300))
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
