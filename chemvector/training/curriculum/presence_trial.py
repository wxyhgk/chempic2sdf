"""One authorized CPU <=300 s / <=200-step cached presence-readout trial."""

import argparse
import json
import os
from pathlib import Path
import sys

from . import presence_release as release
from .checkpoints import file_digest
from .interface_diagnostic import write_json
from .presence_probe import PROTOCOL
from .runtime import ResourceBudget, run_budgeted

AUTHORIZATION = Path('docs/model/training/releases/20260918-presence-training-authorization.json')
OUTPUT = Path('outputs/experiments/chemvector-presence-readout-probe-20260918')
ATTEMPT = Path('outputs/experiments/chemvector-presence-readout-attempt-20260918.json')


def authorize():
    manifest, admission = release.verify()
    auth = json.loads(AUTHORIZATION.read_text())
    if (auth.get('schema_version') != 'chemvector-presence-training-authorization-0.1'
            or auth.get('user_statement') != '开始' or auth.get('protocol') != PROTOCOL
            or auth.get('manifest_sha256') != admission['approved_manifest_sha256']
            or auth.get('admission_sha256') != file_digest(release.OUTPUT / 'admission.json')
            or auth.get('initial_sha256') != release.fixed.INITIAL_SHA
            or auth.get('maximum_invocations') != 1 or auth.get('output') != str(OUTPUT)
            or auth.get('training_authorized') is not True or auth.get('deployment_authorized') is not False):
        raise ValueError('Missing/mismatched exact one-shot CPU training authorization')
    required = {str(Path(__file__).with_name(name)) for name in (
        'presence_release.py', 'presence_probe.py', 'presence_trial.py', 'pair_presence_stage.py')}
    if set(auth.get('implementation_sha256', {})) != required:
        raise ValueError('Authorization does not bind all presence-trial implementation files')
    release.fixed.verify_bindings(auth['implementation_sha256'])
    return manifest, admission, auth


def reserve_once(path, authorization_path):
    """An interrupted invocation consumes this reservation; never auto-reclaim."""
    record = {'schema_version': 'chemvector-presence-attempt-0.1',
              'authorization_sha256': file_digest(authorization_path), 'automatic_retries': 0}
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    return record


def summarize(rows, baseline=None):
    result = {}
    previous = {} if baseline is None else {r['key']: r for r in baseline}
    if baseline is not None and set(previous) != {r['key'] for r in rows}:
        raise ValueError('Baseline/final sample identities differ')
    for group in ('train', 'validation'):
        values = [r for r in rows if r['split'] == group]
        result[group] = {'images': len(values), **{k: sum(r['prediction'][k] for r in values)
                                                for k in ('exact', 'valid', 'fallback')}}
        if baseline is not None:
            result[group].update(rescued=sum(r['prediction']['exact'] and not previous[r['key']]['prediction']['exact'] for r in values),
                broken=sum(not r['prediction']['exact'] and previous[r['key']]['prediction']['exact'] for r in values))
        if group == 'train':
            result[group]['pair_counts'] = {k: sum(r['pair_counts'][k] for r in values) for k in values[0]['pair_counts']}
    return result


def worker():
    import torch
    from chemvector.atom_bond_scene import AtomBondSceneConfig
    from chemvector.train_direct_atom_pair_bond import match_atom_targets
    from .bundle import load_bundle
    from .checkpoints import verify_sources
    from .continuation import structure_forward
    from .data import CurriculumCollator, load_sample
    from .evaluation import FrozenOCRPredictor
    from .gradient_preflight import state_digest
    from .interface_diagnostic import tensor_tree_digest
    from .pair_presence_stage import FrozenFeaturePresenceStage
    from .pixels import image_tensor
    from .presence_probe import fit, pair_counts, frozen_state_digest
    from .runtime import StepLimit
    from .supervision_contract import pair_targets_with_unknown
    from .upstream_diagnostic import check_saved_replay, score_prediction
    from .validation_truth import expected_identity

    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('Externally supervised CPU-only worker required')
    manifest, admission, auth = authorize()
    reservation = json.loads(ATTEMPT.read_text())
    if (reservation['authorization_sha256'] != file_digest(AUTHORIZATION)
            or not OUTPUT.is_dir() or (OUTPUT / 'plan.json').exists()):
        raise ValueError('Fresh output and exact consumed reservation required')
    torch.set_num_threads(2)
    torch.manual_seed(PROTOCOL['seed'])
    fixed = release.fixed
    old_run = json.loads((fixed.RUN / 'run.json').read_text())
    provenance = old_run['provenance']
    plan = {'schema_version': 'chemvector-presence-trial-plan-0.1', 'authorization': auth,
        'authorization_sha256': file_digest(AUTHORIZATION), 'protocol': PROTOCOL,
        'samples': manifest['samples'], 'validation_controls': manifest['validation_controls'],
        'thresholds': {k: v for k, v in provenance.items() if k.endswith('_threshold')},
        'planned_generator_forwards': 80, 'training_images': 24, 'validation_images': 16,
        'private_identity_queries': 0, 'deployed': False,
        'interpretation': 'One local train-fit probe with old clean validation regression, not full 800-image evaluation'}
    write_json(OUTPUT / 'plan.json', plan)
    with (OUTPUT / 'steps.jsonl').open('x', encoding='utf-8') as log:
        def emit(event, **record):
            value = json.dumps({'event': event, **record}, allow_nan=False)
            log.write(value + '\n')
            log.flush()
            if event != 'step_reserved':
                print(value, flush=True)

        model, config, payload = load_bundle(fixed.RUN / 'initial.pt', expected_sha256=fixed.INITIAL_SHA)
        if (payload['provenance'] != provenance or payload['progress']['optimizer_steps'] != 0
                or state_digest(model) != fixed.INITIAL_STATE):
            raise ValueError('Initial model state/provenance mismatch')
        model.eval().requires_grad_(False)
        predictor = FrozenOCRPredictor(provenance, 'cpu')
        state_before, frozen_before, ocr_before = state_digest(model), frozen_state_digest(model), state_digest(predictor.ocr)
        target_config = AtomBondSceneConfig(atom_queries=config.maximum_atoms, bond_queries=config.maximum_bonds)
        collator = CurriculumCollator(target_config, purpose='training')
        all_rows = manifest['samples'] + manifest['validation_controls']
        caches, baseline_cache = [], {}

        def evaluate(phase):
            records = []
            with torch.no_grad():
                for row in all_rows:
                    is_train = row['split'] == 'train'
                    key = row['split'] + ':' + row['sample_key']
                    meta = Path(row['root']) / 'samples' / (row['sample_key'] + '.meta.json')
                    if file_digest(meta) != row['meta_sha256']:
                        raise ValueError('Trial metadata differs from admitted manifest')
                    sample = load_sample(row['root'], row['sample_key'], purpose='training' if is_train else 'diagnostic')
                    if (sample['meta']['split'] != row['split'] or sample['meta']['external_overlap_status'] != 'checked'
                            or sample['graph']['canonical_smiles'] != row['canonical_smiles']):
                        raise ValueError('Trial sample origin or identity differs')
                    pixels = image_tensor(sample['image']).unsqueeze(0)
                    forward = structure_forward(model, pixels, config, provenance['atom_threshold'])
                    structure = predictor.from_predicted_atoms(model.pair, forward[0], forward[6])
                    trace = {}
                    prediction = score_prediction(predictor.from_structure(structure, model.role, model.bbox,
                        pixels, sample['image'], trace=trace), expected_identity(sample))
                    frozen_signature = tensor_tree_digest({'atoms': structure['atoms'],
                        'hidden': structure['output']['hidden'], 'pairs': structure['pairs'],
                        'chemical_logits': structure['output']['chemical_logits']})
                    text_signature = {'roles': trace['roles'], 'ocr': trace['ocr']}
                    if phase == 'baseline':
                        previous = json.loads(((release.DIAG if is_train else release.CLEAN) / (row['sample_key'] + '.json')).read_text())
                        check_saved_replay(prediction, previous['prediction'])
                        baseline_cache[key] = (frozen_signature, text_signature)
                    elif baseline_cache[key] != (frozen_signature, text_signature):
                        raise ValueError('Non-presence predictions changed on repeated fixed inputs')
                    record = {'key': key, 'sample_key': row['sample_key'], 'split': row['split'],
                              'prediction': prediction, 'frozen_signature': frozen_signature, 'trace': trace}
                    if is_train:
                        batch = collator([sample])  # Targets only after baseline image prediction.
                        mapping, _ = match_atom_targets(structure['atoms'], batch, config)
                        contract = pair_targets_with_unknown(structure['atoms'], mapping, batch)
                        record['pair_counts'] = pair_counts(structure, contract, provenance['pair_threshold'])
                        record['pairs'] = {'indices': structure['pairs'].tolist(), 'target_slots': mapping[0, structure['pairs']].tolist(),
                            'known': contract['known_mask'].tolist(), 'targets': contract['legacy_targets']['presence'].tolist(),
                            'probabilities': structure['output']['presence_logits'].sigmoid().tolist()}
                        if phase == 'baseline':
                            caches.append({'sample_key': row['sample_key'], 'purpose': batch['purpose'],
                                'training_authorized': batch['training_authorized'],
                                'manifest_sha256': admission['approved_manifest_sha256'],
                                'hidden': structure['output']['hidden'].detach().clone(),
                                'targets': contract['legacy_targets']['presence'].detach().clone(),
                                'known': contract['known_mask'].detach().clone()})
                    write_json(OUTPUT / (phase + '--' + row['sample_key'] + '.json'), record)
                    records.append(record)
                    emit('case', phase=phase, completed=len(records), key=key, exact=prediction['exact'])
            return records

        baseline = evaluate('baseline')
        if len(caches) != 24 or state_digest(model) != state_before:
            raise ValueError('Baseline cache incomplete or model changed before training')
        baseline_summary = summarize(baseline)
        emit('baseline', summary=baseline_summary)
        stage = FrozenFeaturePresenceStage(model, config, atom_threshold=provenance['atom_threshold'])
        fitted = fit(stage, caches, admission=admission, manifest_sha256=admission['approved_manifest_sha256'],
                     step_limit=StepLimit(PROTOCOL['maximum_steps']), emit=emit)
        model.zero_grad(set_to_none=True)
        model.eval().requires_grad_(False)
        final = evaluate('final')
        if (frozen_state_digest(model) != frozen_before or state_digest(predictor.ocr) != ocr_before
                or any(p.grad is not None or p.requires_grad for p in model.parameters())
                or torch.cuda.is_initialized()):
            raise ValueError('Frozen state, OCR, gradient or device guard failed')
        snapshot = {'schema_version': 'chemvector-presence-readout-delta-0.1', 'base_initial_sha256': fixed.INITIAL_SHA,
            'authorization_sha256': file_digest(AUTHORIZATION), 'optimizer_steps': fitted['optimizer_steps'],
            'weight': model.pair.presence.weight.detach().tolist(), 'bias': model.pair.presence.bias.detach().tolist(),
            'not_standalone_model': True, 'resume_supported': False, 'deployed': False}
        write_json(OUTPUT / 'last-presence.json', snapshot)
        restored = json.loads((OUTPUT / 'last-presence.json').read_text())
        if any(not torch.equal(torch.tensor(restored[k]), getattr(model.pair.presence, k).detach()) for k in ('weight', 'bias')):
            raise ValueError('Saved presence delta roundtrip changed values')
        release.verify()
        verify_sources(provenance['checkpoints'])
        fixed.verify_bindings({str(fixed.RUN / 'initial.pt'): fixed.INITIAL_SHA})
        final_summary = summarize(final, baseline)
        report = {'schema_version': 'chemvector-presence-trial-result-0.1',
            'plan_sha256': file_digest(OUTPUT / 'plan.json'), 'baseline': baseline_summary, 'final': final_summary,
            'fitting': fitted, 'model_state_before': state_before, 'model_state_after': state_digest(model),
            'frozen_state_before': frozen_before, 'frozen_state_after': frozen_state_digest(model),
            'ocr_state_before': ocr_before, 'ocr_state_after': state_digest(predictor.ocr),
            'default_kept': 'initial', 'deployed': False, 'training_performed': True, 'gpu_used': False,
            'generator_forwards': 80, 'baseline_saved_parity_checks': 40,
            'frozen_prediction_parity_checks': 40, 'private_identity_queries': 0,
            'optimizer_steps': fitted['optimizer_steps'], 'presence_delta_sha256': file_digest(OUTPUT / 'last-presence.json'),
            'full_800_evaluation_performed': False,
            'limits': ['24 train-fit examples are not independent accuracy',
                       '16 old clean validation controls are a narrow regression check, not full-domain validation',
                       'Raw chemical logits fixed; downstream aromatic decisions may change with presence',
                       'Single fixed optimizer/rate/budget, no threshold sweep, retries, deployment or automatic expansion']}
        write_json(OUTPUT / 'report.json', report)
        emit('completed', baseline=baseline_summary, final=final_summary, optimizer_steps=fitted['optimizer_steps'])
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        if args.check_only:
            parser.error('Worker cannot be combined with check-only')
        worker()
        return 0
    _, admission, _ = authorize()
    if args.check_only:
        print(json.dumps({'checks_passed': True, 'protocol': PROTOCOL, 'model_loaded': False, 'training_performed': False}))
        return 0
    if OUTPUT.exists():
        raise FileExistsError('Trial output already exists; no retry')
    reserve_once(ATTEMPT, AUTHORIZATION)
    receipt = run_budgeted([sys.executable, '-m', 'chemvector.training.curriculum.presence_trial', '--worker'],
        output=OUTPUT, budget=ResourceBudget(PROTOCOL['wall_seconds']), training=True, admission=admission)
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
