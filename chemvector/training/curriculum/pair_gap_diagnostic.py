"""One fixed 17+7 CPU initial-model diagnostic, not a training entry point."""

import argparse
import json
import os
from pathlib import Path
import sys

from .interface_diagnostic import write_json, tensor_tree_digest, structure_trace
from .runtime import ResourceBudget, run_budgeted

RUN = Path('outputs/experiments/chemvector-selected-matching-trial-20260915')
RELEASE = Path('outputs/datasets/chemvector-expanded-release-20260915')
DATA = Path('outputs/datasets/chemvector-pair-gap-quarantine-20260918')
GENERATION = Path('outputs/audits/chemvector-pair-gap-generation-20260918')
REQUEST = Path('outputs/audits/chemvector-pair-gap-review-request-20260918')
PUBLIC = Path('outputs/audits/chemvector-pair-gap-public-identity-20260918')
PRIVATE_RECEIPT = Path('outputs/audits/chemvector-pair-gap-private-identity-20260918/receipt/receipt.json')
PANEL = PUBLIC / 'screened-panel.json'
PANEL_SHA = '4659420653400254653ce45dc97bc4f23eb11ccf6968ed5bd44d4cae0f4d4492'
INITIAL_STATE = '9d7f66f79329479e05c38f8b91d95118adb9f12c32a44e5bd9d83a458a8b452e'
INITIAL_SHA = '09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1'
OUTPUT = Path('outputs/audits/chemvector-pair-gap-diagnostic24-20260918')
PARITY_INDICES = (0, 16, 23)


def fixed_rows(panel, remaining, manifest):
    """Pure fixed-list assembly; no searching/replacing by model outcomes."""
    if (panel.get('schema_version') != 'chemvector-pair-gap-screened-panel-0.1'
            or panel.get('quarantine_root') != str(DATA)
            or panel.get('training_authorized') is not False
            or panel.get('training_ready') is not False
            or panel.get('remaining_candidate_count') != 17
            or panel.get('total_inspection_images') != 24
            or len(remaining) != 17 or len(panel['new_samples']) != 17
            or len(panel['existing_training_controls']) != 7):
        raise ValueError('Expected the immutable 17+7 diagnostic-only panel')
    identities = {r['molecule_id']: r for r in remaining}
    if len(identities) != 17:
        raise ValueError('Duplicate remaining identities')
    rows = []
    for sample in panel['new_samples']:
        candidate = identities.get(sample['molecule_id'])
        if (candidate is None or candidate['sample_keys'] != [sample['sample_key']]
                or candidate['image_count'] != 1 or sample['profile_id'] != 'reference'):
            raise ValueError('New panel differs from exact remaining subset')
        rows.append({'origin': 'new_quarantine', 'root': str(DATA),
            'sample_key': sample['sample_key'], 'molecule_id': candidate['molecule_id'],
            'canonical_smiles': candidate['canonical_smiles']})
    released = {(r['source'], r['sample_key']): r for r in manifest['samples']}
    if len(released) != len(manifest['samples']):
        raise ValueError('Duplicate release rows')
    for row in panel['existing_training_controls']:
        if (row != released.get((row['source'], row['sample_key'])) or row['split'] != 'train'
                or row['source'] != 'curriculum' or row['domain'] != 'curriculum'
                or not row['sample_key'].endswith('__reference')):
            raise ValueError('Existing control is not an exact released training row')
        rows.append({**row, 'origin': 'existing_training_control', 'root': manifest['sources'][row['source']]})
    if any(len({r[k] for r in rows}) != 24 for k in ('sample_key', 'molecule_id', 'canonical_smiles')):
        raise ValueError('Duplicate diagnostic samples/identities')
    return rows


def verify_bindings(inputs):
    from .checkpoints import file_digest
    for name, expected in inputs.items():
        path = Path(name)
        if ('sealed_test' in path.parts or any(p.is_symlink() for p in (path, *path.parents))
                or file_digest(path) != expected):
            raise ValueError('Bound diagnostic input changed: ' + name)


def audit(output):
    import torch
    from chemvector.atom_bond_scene import AtomBondSceneConfig
    from . import continuation, evaluation, data, pair_gap_metrics, supervision_contract
    from .bundle import load_bundle
    from .checkpoints import file_digest, verify_sources
    from .gradient_preflight import state_digest
    from .headroom_audit import INITIAL_STATE as prior_initial_state
    from .pair_gap_generation import bind_digest
    from .pixels import image_tensor
    from .release import verify_release
    from .release_evidence import verify_reviews
    from .upstream_diagnostic import candidate_trace, score_prediction, check_saved_replay
    from .validation_truth import expected_identity

    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('Supervised CPU-only diagnostic required')
    output = Path(output)
    if not output.is_dir() or any((output / p).exists() for p in ('plan.json', 'report.json')):
        raise ValueError('Fresh supervised output required')
    torch.set_num_threads(2)
    torch.manual_seed(20260918)
    verify_bindings({str(PANEL): PANEL_SHA})
    panel = json.loads(PANEL.read_text())
    inputs = {str(PANEL): PANEL_SHA, **panel['input_sha256']}
    verify_bindings(inputs)
    files = {'request': REQUEST / 'request.json', 'request_candidates': REQUEST / 'candidate-identities.json',
             'remaining': PUBLIC / 'review/remaining-candidates.json',
             'public_report': PUBLIC / 'review/report.json', 'private_receipt': PRIVATE_RECEIPT}
    raw = {k: p.read_bytes() for k, p in files.items()}
    docs = {k: json.loads(v) for k, v in raw.items()}
    verify_reviews(raw, docs)  # Saved receipts only; no custodian/query invocation.
    manifest, _, bindings = verify_release(RELEASE / 'manifest.json', RELEASE / 'admission.json')
    rows = fixed_rows(panel, docs['remaining'], manifest)
    generation = json.loads((GENERATION / 'report.json').read_text())
    for source in ('verified_input_sha256', 'artifact_sha256'):
        for name, value in generation[source].items():
            bind_digest(inputs, name, value)
    verify_bindings(inputs)
    # Bind all original generated files, including preserved excluded evidence.
    inventory = json.loads((DATA / 'sample-sha256.json').read_text())
    for name, value in inventory.items():
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Unsafe generated inventory path')
        bind_digest(inputs, DATA / name, value)
    run = json.loads((RUN / 'run.json').read_text())
    previous = json.loads((RUN / 'report.json').read_text())
    binding = previous['selected']
    if (run['bindings'] != bindings or previous['bindings'] != bindings
            or Path(binding['path']).resolve() != (RUN / 'initial.pt').resolve()
            or binding['sha256'] != INITIAL_SHA or INITIAL_STATE != prior_initial_state):
        raise ValueError('Expected unchanged original initial/release')
    paths = [RUN / 'run.json', RUN / 'report.json', RELEASE / 'manifest.json', RELEASE / 'admission.json',
             Path('frontend/inference_profiles.json'), Path(__file__),
             *(Path(m.__file__) for m in (continuation, evaluation, data, pair_gap_metrics, supervision_contract))]
    for path in paths:
        bind_digest(inputs, path, file_digest(path))
    for row in rows:
        path = Path(row['root']) / 'samples' / (row['sample_key'] + '.meta.json')
        actual = file_digest(path)
        if row['origin'] == 'existing_training_control' and actual != row['meta_sha256']:
            raise ValueError('Existing control metadata differs from release')
        bind_digest(inputs, path, actual)
        row['meta_sha256'] = actual
    verify_bindings(inputs)
    plan = {'schema_version': 'chemvector-pair-gap-diagnostic-plan-0.1', 'samples': rows,
        'input_sha256': inputs, 'initial': binding, 'expected_initial_state': INITIAL_STATE,
        'selection': 'Exact screened 17 quarantine images + exact 7 existing training controls; original panel order',
        'cpu_wall_seconds': 300, 'planned_images': 24, 'planned_generator_forwards': 27,
        'default_path_replay_indices': PARITY_INDICES,
        'thresholds': {k: v for k, v in run['provenance'].items() if k.endswith('_threshold')},
        'scope': 'Quarantine diagnostic only; neither a training release nor independent validation',
        'truth_use': 'Target tensors are built only after immutable baseline inference; no GT candidates/crops/repairs',
        'gpu_used': False, 'optimizer_steps': 0, 'training_authorized': False,
        'private_identity_queries': 0, 'automatic_retries': 0}
    write_json(output / 'plan.json', plan)
    print(json.dumps({'event': 'plan_fixed', 'images': 24, 'generator_forwards': 27}), flush=True)
    model, config, payload = load_bundle(binding['path'], expected_sha256=INITIAL_SHA)
    if (payload['provenance'] != run['provenance'] or payload['progress']['optimizer_steps'] != 0
            or payload['progress']['bindings'] != bindings or state_digest(model) != INITIAL_STATE):
        raise ValueError('Initial model state/provenance differs')
    model.eval().requires_grad_(False)
    predictor = evaluation.FrozenOCRPredictor(run['provenance'], 'cpu')
    before = {'model': state_digest(model), 'ocr': state_digest(predictor.ocr)}
    collator = data.CurriculumCollator(AtomBondSceneConfig(atom_queries=config.maximum_atoms,
        bond_queries=config.maximum_bonds), purpose='diagnostic')
    cases, parity = [], 0
    with torch.inference_mode():
        for index, row in enumerate(rows):
            sample = data.load_sample(row['root'], row['sample_key'], purpose='diagnostic')
            pair_gap_metrics.validate_sample(sample, row)
            pixels = image_tensor(sample['image']).unsqueeze(0)
            forward = continuation.structure_forward(model, pixels, config, run['provenance']['atom_threshold'])
            structure = predictor.from_predicted_atoms(model.pair, forward[0], forward[6])
            original_hash, trace = tensor_tree_digest(structure), {}
            predicted = predictor.from_structure(structure, model.role, model.bbox, pixels, sample['image'], trace=trace)
            prediction = score_prediction(predicted, expected_identity(sample))
            if index in PARITY_INDICES:
                replay = score_prediction(predictor(model, config, pixels, sample['image']), prediction['expected_smiles'])
                check_saved_replay(prediction, replay)
                parity += 1
            batch = collator([sample])
            metrics = pair_gap_metrics.measure(sample, row, batch, forward[3], structure, trace, config,
                                               run['provenance']['pair_threshold'])
            if tensor_tree_digest(structure) != original_hash:
                raise ValueError('Diagnostic changed cached predictions')
            case = {'key': row['origin'] + ':' + row['sample_key'], 'origin': row['origin'],
                'sample_key': row['sample_key'], 'prediction': prediction, 'metrics': metrics, 'trace': trace,
                'candidates': candidate_trace(forward[3], forward[6], forward[5]),
                'structure': structure_trace(structure), 'default_path_parity_checked': index in PARITY_INDICES}
            write_json(output / (row['sample_key'] + '.json'), case)
            cases.append(case)
            print(json.dumps({'event': 'case', 'completed': len(cases), 'key': case['key'],
                              'exact': prediction['exact'], 'bond_stages': metrics['bond_stages']}), flush=True)
    after = {'model': state_digest(model), 'ocr': state_digest(predictor.ocr)}
    if before != after or torch.cuda.is_initialized() or parity != len(PARITY_INDICES):
        raise ValueError('State/device/parity guard failed')
    if any(p.grad is not None or p.requires_grad for m in (model, predictor.ocr) for p in m.parameters()):
        raise ValueError('Diagnostic acquired gradients or trainable parameters')
    verify_bindings(inputs)
    verify_bindings({str(RUN / 'initial.pt'): INITIAL_SHA})
    verify_sources(run['provenance']['checkpoints'])
    report = {'schema_version': 'chemvector-pair-gap-diagnostic-result-0.1',
        'plan_sha256': file_digest(output / 'plan.json'), 'summary': pair_gap_metrics.summarize(cases),
        'states_before': before, 'states_after': after, 'default_path_parity_checks': parity,
        'generator_forwards': len(cases) + parity, 'optimizer_steps': 0, 'gpu_used': False,
        'training_performed': False, 'training_authorized': False, 'training_ready': False, 'deployed': False,
        'private_identity_queries': 0, 'generated_payload_files_verified': len(inventory),
        'case_files': {r['key']: r['sample_key'] + '.json' for r in cases},
        'limits': ['Fixed targeted quarantine/train diagnostic, not a held-out accuracy estimate',
                   'Six-pixel Hungarian correspondence gaps are not proof that visual atoms are absent',
                   'Unreachable negatives are not true negatives; unknown endpoints are not confirmed false edges',
                   'Aromatic-adjusted local bond labels and literal text are not whole-molecule identity',
                   'No GT-fed predictions, optimizer, threshold search, default behavior change or retries']}
    write_json(output / 'report.json', report)
    print(json.dumps({'event': 'completed', 'summary': report['summary']}), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        audit(OUTPUT)
        return 0
    receipt = run_budgeted([sys.executable, '-m', 'chemvector.training.curriculum.pair_gap_diagnostic', '--worker'],
                           output=OUTPUT, budget=ResourceBudget(300))
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
