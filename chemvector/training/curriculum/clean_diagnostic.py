"""One initial-only clean16 CPU trace, plus a separately labeled GT-box OCR probe."""

import argparse
import json
import os
from pathlib import Path
import sys

from .clean_metrics import select_clean_rows, measure_clean, predicted_box
from .interface_diagnostic import write_json, structure_trace, tensor_tree_digest
from .runtime import ResourceBudget, run_budgeted

RUN = Path('outputs/experiments/chemvector-selected-matching-trial-20260915')
RELEASE = Path('outputs/datasets/chemvector-expanded-release-20260915')
OUTPUT = Path('outputs/audits/chemvector-clean16-diagnostic-20260918')
SAVED_SHA = 'be9a12c66c96ee8ecac925c707cea6159fc16fec0864f94fc5395631d1cedb89'
INITIAL_STATE = '9d7f66f79329479e05c38f8b91d95118adb9f12c32a44e5bd9d83a458a8b452e'


def reference_crops(sample):
    """Fixed ink box + existing 4 px context; no margin or OCR-choice search."""
    return [(a['atom_id'], a['bbox']) for a in sample['layers']['atoms'] if a['visible_text']]


def audit(output):
    import torch
    from rdkit import Chem
    from chemvector.atom_bond_scene import AtomBondSceneConfig
    from chemvector.ctc_ocr import bbox_crop, decode_ctc_with_score
    from chemvector.evaluate_direct_atom_ocr import crop_tensor
    from . import evaluation, continuation, data, clean_metrics
    from .bundle import load_bundle
    from .checkpoints import file_digest, verify_sources
    from .gradient_preflight import state_digest
    from .release import verify_release
    from .upstream_diagnostic import candidate_trace, score_prediction, check_saved_replay
    from .validation_truth import expected_identity

    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('Supervised CPU-only run required')
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
        raise ValueError('Expected original initial and admitted release')
    selected = select_clean_rows(manifest)
    saved_path = RUN / 'validation-000000.json'
    if file_digest(saved_path) != SAVED_SHA:
        raise ValueError('Historical initial predictions changed')
    saved = {r['key']: r for r in json.loads(saved_path.read_text())['predictions']}
    files = [RUN / 'run.json', RUN / 'report.json', saved_path,
             RELEASE / 'manifest.json', RELEASE / 'admission.json',
             Path('frontend/inference_profiles.json'), Path(__file__),
             Path(clean_metrics.__file__), Path(evaluation.__file__),
             Path(continuation.__file__), Path(data.__file__)]
    inputs = {str(p): file_digest(p) for p in files}
    for row in selected:
        path = Path(manifest['sources'][row['source']]) / 'samples' / (row['sample_key'] + '.meta.json')
        if file_digest(path) != row['meta_sha256']:
            raise ValueError('Released metadata changed')
        inputs[str(path)] = row['meta_sha256']
    plan = {'schema_version': 'chemvector-clean16-diagnostic-plan-0.1', 'samples': selected,
        'selection': 'All existing curriculum/validation/reference rows; no outcome-based selection or replacement',
        'initial': binding, 'input_sha256': inputs, 'cpu_wall_seconds': 300,
        'planned_generator_forwards': 16, 'saved_baseline_parity_required': 16,
        'thresholds': {k: v for k, v in run['provenance'].items() if k.endswith('_threshold')},
        'reference_crop_probe': 'After baseline only: every visible same-render ink bbox, existing crop_tensor with 4 px context, same frozen OCR; no margin sweep or SDF replacement',
        'gpu_used': False, 'optimizer_steps': 0,
        'limits': ['Existing development slice, not a new independent benchmark',
                   'Aligned labels are used after prediction for scoring and separately identified OCR crop diagnostics',
                   'No GT atoms/edges/text values enter baseline inference; reference OCR never changes baseline output',
                   'No sample expansion, threshold search, training, deployment or automatic retry']}
    write_json(output / 'plan.json', plan)
    model, config, payload = load_bundle(binding['path'], expected_sha256=binding['sha256'])
    if (payload['provenance'] != run['provenance'] or payload['progress']['optimizer_steps'] != 0
            or payload['progress']['bindings'] != bindings or state_digest(model) != INITIAL_STATE):
        raise ValueError('Initial state/provenance mismatch')
    model.eval().requires_grad_(False)
    predictor = evaluation.FrozenOCRPredictor(run['provenance'], 'cpu')
    before = {'model': state_digest(model), 'ocr': state_digest(predictor.ocr)}
    collator = data.CurriculumCollator(AtomBondSceneConfig(atom_queries=config.maximum_atoms,
        bond_queries=config.maximum_bonds), purpose='diagnostic')
    cases = []
    with torch.inference_mode():
        for row in selected:
            key = row['source'] + ':' + row['sample_key']
            sample = data.load_sample(manifest['sources'][row['source']], row['sample_key'], purpose='diagnostic')
            meta, graph = sample['meta'], sample['graph']
            target = Chem.MolFromSmiles(graph['canonical_smiles'])
            if (meta['split'] != 'validation' or meta['profile_id'] != 'reference'
                    or meta['external_overlap_status'] != 'checked'
                    or meta['molecule_id'] != row['molecule_id'] or graph['canonical_smiles'] != row['canonical_smiles']
                    or not 2 <= target.GetNumHeavyAtoms() <= 7
                    or any(a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for a in target.GetAtoms())
                    or any(b.GetStereo() != Chem.BondStereo.STEREONONE for b in target.GetBonds())
                    or meta['profile']['font_size'] != 24 or meta['profile']['bond_line_width'] != 2):
                raise ValueError('Clean slice metadata/chemistry differs from fixed scope')
            batch = collator([sample])
            pixels = batch['pixel_values']
            forward = continuation.structure_forward(model, pixels, config, run['provenance']['atom_threshold'])
            structure = predictor.from_predicted_atoms(model.pair, forward[0], forward[6])
            digest, trace = tensor_tree_digest(structure), {}
            predicted = predictor.from_structure(structure, model.role, model.bbox, pixels, sample['image'], trace=trace)
            prediction = score_prediction(predicted, expected_identity(sample))
            check_saved_replay(prediction, saved[key])
            # All target crops are a separate OCR-only experiment AFTER the
            # immutable baseline prediction. They never feed from_structure().
            boxes = reference_crops(sample)
            reference = {}
            if boxes:
                crops = torch.stack([crop_tensor(sample['image'], box) for _, box in boxes])
                values = predictor.ocr(crops).float()
                if not bool(torch.isfinite(values).all()):
                    raise ValueError('Non-finite reference OCR output')
                for (identity, box), logits in zip(boxes, values):
                    text, score = decode_ctc_with_score(logits)
                    reference[identity] = {'text': text, 'score': score}
                    bbox_crop(sample['image'], box, context=4.).save(output / f'{row["sample_key"]}--target-{identity}.png')
            for record in trace['ocr']:
                bbox_crop(sample['image'], predicted_box(record), context=4.).save(
                    output / f'{row["sample_key"]}--candidate-{record["candidate"]}.png')
            metrics = measure_clean(sample, batch, forward[3], structure, trace, config,
                                    run['provenance']['pair_threshold'], reference)
            if tensor_tree_digest(structure) != digest:
                raise ValueError('Diagnostic changed cached predictions')
            case = {'key': key, 'prediction': prediction, 'metrics': metrics, 'trace': trace,
                'candidates': candidate_trace(forward[3], forward[6], forward[5]),
                'structure': structure_trace(structure), 'saved_baseline_parity_checked': True,
                'source_png': str(Path(manifest['sources'][row['source']]) / 'samples' / (row['sample_key'] + '.png'))}
            write_json(output / (row['sample_key'] + '.json'), case)
            cases.append(case)
            print(json.dumps({'completed': len(cases), 'key': key, 'exact': prediction['exact']}), flush=True)
    after = {'model': state_digest(model), 'ocr': state_digest(predictor.ocr)}
    if before != after or torch.cuda.is_initialized():
        raise ValueError('State or device guard failed')
    if any(p.grad is not None or p.requires_grad for module in (model, predictor.ocr) for p in module.parameters()):
        raise ValueError('Diagnostic model has gradients or trainable parameters')
    for path, digest in inputs.items():
        if file_digest(path) != digest:
            raise ValueError('Bound inputs changed')
    if file_digest(binding['path']) != binding['sha256']:
        raise ValueError('Initial checkpoint changed')
    verify_sources(run['provenance']['checkpoints'])
    report = {'schema_version': 'chemvector-clean16-diagnostic-result-0.1',
        'plan_sha256': file_digest(output / 'plan.json'),
        'summary': {'images': len(cases), **{k: sum(r['prediction'][k] for r in cases) for k in ('exact', 'valid', 'fallback')},
                    'reference_ocr_crops': sum(a['visible_text'] for r in cases for a in r['metrics']['atoms'])},
        'states_before': before, 'states_after': after, 'saved_baseline_parity_checks': len(cases),
        'gpu_used': False, 'optimizer_steps': 0, 'training_performed': False, 'deployed': False,
        'case_files': {r['key']: r['key'].split(':', 1)[1] + '.json' for r in cases}}
    write_json(output / 'report.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        audit(OUTPUT)
        return 0
    receipt = run_budgeted([sys.executable, '-m', 'chemvector.training.curriculum.clean_diagnostic', '--worker'],
                           output=OUTPUT, budget=ResourceBudget(300))
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
