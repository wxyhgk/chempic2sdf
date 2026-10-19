"""Image-only frozen upstream caching, admitted fitting, and fixed paired gates."""

import json
import os
import time

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.predicted_incidence import associate_predicted_proposals
from . import selector_retention_stage as stage
from . import selector_retention_trial as trial
from .bundle import load_bundle, save_bundle
from .continuation import structure_forward
from .data import CurriculumCollator
from .evaluation import FrozenOCRPredictor
from .gradient_preflight import state_digest
from .interface_diagnostic import write_json, tensor_tree_digest
from .pixels import image_tensor
from .presence_regression_metrics import parity_differences
from .runtime import StepLimit
from .upstream_diagnostic import score_prediction
from .validation_truth import expected_identity


@torch.no_grad()
def assess(model, config, predictor, cache, *, device):
    raw, bonds = [stage.tensors_to(cache[k], device) for k in ('raw', 'bonds')]
    output = stage.call_selector(model.set_head, raw, bonds)
    if any(not bool(torch.isfinite(v).all()) for v in output.values()):
        raise ValueError('Nonfinite selector output')
    selected = {**raw, 'atom_mask': raw['atom_mask'] & (output['atom_presence_logits'] >= stage.THRESHOLD)}
    # Predictions first, labels only in a separate downstream metric call.
    prediction = None
    if cache['fit_domain'] == 'old_full_molecule':
        structure = predictor.from_predicted_atoms(model.pair, cache['features'].to(device), selected)
        predicted = predictor.from_structure(structure, model.role, model.bbox,
            cache['pixels'].to(device), cache['image'])
        prediction = score_prediction(predicted, cache['expected'])
    metric = stage.metrics(stage.tensors_to(raw, 'cpu'), stage.tensors_to(selected, 'cpu'),
        stage.tensors_to(output, 'cpu'), cache['labels'], cache['batch'], config)
    return dict(key=cache['key'], sample_key=cache['sample_key'], group=cache['fit_domain'],
                metrics=metric, prediction=prediction)


def check_core_contract(baseline):
    by_key = {r['sample_key']: r for r in baseline}
    witnesses = json.loads(trial.CORE_PLAN.read_text())['witnesses']
    for witness in witnesses:
        metric = by_key[witness['sample_key']]['metrics']
        bad = witness['hardest_negative']
        if (metric['positive_slots'] != witness['positive_slots']
                or str(bad) not in metric['logits'] or bad in metric['positive_slots']):
            raise ValueError('GPU raw slot assignments differ from preregistered core witness')


def cache_digest(caches):
    return tensor_tree_digest({c['key']: {k: c[k] for k in ('raw', 'bonds', 'labels', 'batch')}
                               for c in caches})


def worker():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise ValueError('Externally supervised single GPU 0 worker required')
    manifest, admission, auth = trial.authorize()
    reservation = json.loads(trial.ATTEMPT.read_text())
    if (reservation['authorization_sha256'] != trial.file_digest(trial.AUTHORIZATION)
            or not trial.OUTPUT.is_dir() or (trial.OUTPUT / 'plan.json').exists()):
        raise ValueError('Fresh output and exact consumed reservation required')
    torch.set_num_threads(2)
    torch.manual_seed(stage.SEED)
    torch.cuda.manual_seed_all(stage.SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = 'cuda:0'
    started = time.monotonic()
    write_json(trial.OUTPUT / 'plan.json', dict(schema_version='chemvector-selector-retention-plan-0.1',
        authorization=auth, authorization_sha256=trial.file_digest(trial.AUTHORIZATION),
        training_samples=manifest['training_samples'], development_samples=manifest['validation_only_samples'],
        numerical_settings=dict(device=device, tf32=False, cudnn_benchmark=False, cudnn_deterministic=True),
        private_identity_queries=0, default_kept='initial', deployed=False))
    with (trial.OUTPUT / 'steps.jsonl').open('x', encoding='utf-8') as log:
        def emit(event, **values):
            line = json.dumps(dict(event=event, elapsed_seconds=time.monotonic() - started, **values), allow_nan=False)
            log.write(line + '\n')
            log.flush()
            if event == 'step_reserved':
                os.fsync(log.fileno())
            elif event != 'update' or values.get('step', 0) % 25 == 0:
                print(line, flush=True)

        model, config, payload = load_bundle(trial.INITIAL, expected_sha256=trial.INITIAL_SHA)
        if state_digest(model) != trial.INITIAL_STATE or payload['progress']['optimizer_steps'] != 0:
            raise ValueError('Initial state/progress differs')
        proposal = json.loads((trial.release.OUTPUT / 'trial-proposal.json').read_text())
        if any(payload['provenance'][k] != v for k, v in proposal['thresholds'].items()):
            raise ValueError('Fixed threshold mismatch')
        model.to(device)
        params = stage.configure(model)
        if sum(p.numel() for p in params) != 178565:
            raise ValueError('Unexpected selector parameter count')
        predictor = FrozenOCRPredictor(payload['provenance'], device)
        frozen_before, ocr_before = stage.frozen_digest(model), state_digest(predictor.ocr)
        target_config = AtomBondSceneConfig(atom_queries=config.maximum_atoms, bond_queries=config.maximum_bonds)
        collators = {p: CurriculumCollator(target_config, purpose=p) for p in ('training', 'diagnostic')}
        caches, baseline = [], []
        direct_parity_checked = False
        for purpose, rows in (('training', manifest['training_samples']), ('diagnostic', manifest['validation_only_samples'])):
            for row in rows:
                sample = trial.release.load_row(manifest, row, purpose=purpose)
                with torch.no_grad():
                    pixels = image_tensor(sample['image']).unsqueeze(0).to(device)
                    features, _, _, raw, bonds, _, _ = structure_forward(model, pixels, config, stage.THRESHOLD)
                    raw, bonds = stage.tensors_to(raw, 'cpu'), stage.tensors_to(bonds, 'cpu')
                batch = collators[purpose]([sample])
                labels = associate_predicted_proposals(raw, bonds, batch, config)
                cache = dict(key=row['source'] + ':' + row['sample_key'], sample_key=row['sample_key'],
                    fit_domain=row.get('fit_domain', 'new_development'), purpose=purpose,
                    training_authorized=batch['training_authorized'], manifest_sha256=trial.MANIFEST_SHA,
                    raw=raw, bonds=bonds, labels=stage.tensors_to(labels, 'cpu'),
                    batch={k: v.detach().cpu() for k, v in batch.items() if isinstance(v, torch.Tensor) and k != 'pixel_values'})
                if cache['fit_domain'] == 'old_full_molecule':
                    cache.update(features=features.detach().cpu(), pixels=pixels.detach().cpu(),
                                 image=sample['image'], expected=expected_identity(sample))
                record = assess(model, config, predictor, cache, device=device)
                if cache['fit_domain'] == 'old_full_molecule' and not direct_parity_checked:
                    direct = score_prediction(predictor(model, config, pixels, sample['image']), cache['expected'])
                    if parity_differences(record['prediction'], direct):
                        raise ValueError('Cached selector/pair/text path differs from direct prediction')
                    direct_parity_checked = True
                if purpose == 'training':
                    stage.require_training_cache(cache, trial.MANIFEST_SHA)
                caches.append(cache)
                baseline.append(record)
                write_json(trial.OUTPUT / ('baseline--' + row['sample_key'] + '.json'), record)
                emit('baseline_case', completed=len(caches), group=cache['fit_domain'], sample_key=row['sample_key'])
                del features, pixels, sample, batch
        fitting = [c for c in caches if c['purpose'] == 'training']
        development = [c for c in caches if c['purpose'] == 'diagnostic']
        if len(fitting) != 58 or len(development) != 64 or not direct_parity_checked:
            raise ValueError('Incomplete fixed panels')
        check_core_contract(baseline)
        if state_digest(model) != trial.INITIAL_STATE:
            raise ValueError('Initial weights changed during baseline')
        cache_before = cache_digest(caches)
        visual_before = tensor_tree_digest({c['key']: {k: c[k] for k in ('features', 'pixels')}
                                            for c in fitting if c['fit_domain'] == 'old_full_molecule'})
        base_fit = [r for r in baseline if r['group'] != 'new_development']
        base_dev = [r for r in baseline if r['group'] == 'new_development']
        local = stage.local_gate(base_fit, base_fit, 0)
        write_json(trial.OUTPUT / 'gate-000000.json', local)
        emit('baseline_complete', local=local, old_strict_exact=sum(
            r['prediction']['exact'] for r in base_fit if r['prediction'] is not None))
        optimizer = torch.optim.AdamW(params, lr=5e-6, eps=1e-8, weight_decay=0, betas=(.9, .999), foreach=False)
        limit, order = StepLimit(400), stage.training_order(fitting, trial.MANIFEST_SHA)
        for step in range(1, 401):
            cache = next(order)
            losses = stage.update(model, cache, optimizer, limit, trial.MANIFEST_SHA, device,
                lambda s: emit('step_reserved', step=s, key=cache['key']))
            emit('update', step=step, key=cache['key'], **losses)
            if step not in (100, 200, 400):
                continue
            current = []
            for cache in fitting:
                record = assess(model, config, predictor, cache, device=device)
                current.append(record)
                write_json(trial.OUTPUT / (f'check-{step:06d}--' + cache['sample_key'] + '.json'), record)
            local = stage.local_gate(base_fit, current, step)
            write_json(trial.OUTPUT / f'gate-{step:06d}.json', local)
            emit('local_check', **local)
            if local['stop_reasons']:
                break
        final_dev = []
        for cache in development:
            record = assess(model, config, predictor, cache, device=device)
            final_dev.append(record)
            write_json(trial.OUTPUT / ('final-development--' + cache['sample_key'] + '.json'), record)
        gate = stage.final_gate(local, base_dev, final_dev)
        model.zero_grad(set_to_none=True)
        stage.parameters(model)
        frozen_after, ocr_after, cache_after = stage.frozen_digest(model), state_digest(predictor.ocr), cache_digest(caches)
        visual_after = tensor_tree_digest({c['key']: {k: c[k] for k in ('features', 'pixels')}
                                           for c in fitting if c['fit_domain'] == 'old_full_molecule'})
        if (frozen_before != frozen_after or ocr_before != ocr_after or cache_before != cache_after
                or visual_before != visual_after):
            raise ValueError('Frozen weights, OCR, or cached inputs changed')
        checkpoint = save_bundle(trial.OUTPUT / 'last-diagnostic.pt', model, config,
            provenance=payload['provenance'], progress=dict(optimizer_steps=limit.consumed,
                trial='selector-retention-only-v1', actual_parameter_scope='set_head only',
                authorization_sha256=trial.file_digest(trial.AUTHORIZATION), deployed=False))
        trial.authorize()  # Recheck protected data, sources and default profile after fitting.
        report = dict(schema_version='chemvector-selector-retention-result-0.1',
            plan_sha256=trial.file_digest(trial.OUTPUT / 'plan.json'), optimizer_steps=limit.consumed,
            gpu_used=True, training_performed=True, fitting_images=58, development_images=64,
            gate=gate, checkpoint=checkpoint, model_state_before=trial.INITIAL_STATE,
            model_state_after=state_digest(model), frozen_before=frozen_before, frozen_after=frozen_after,
            ocr_before=ocr_before, ocr_after=ocr_after, cache_before=cache_before, cache_after=cache_after,
            visual_cache_before=visual_before, visual_cache_after=visual_after,
            old_strict_exact_before=sum(r['prediction']['exact'] for r in base_fit if r['prediction'] is not None),
            old_strict_exact_after=sum(r['prediction']['exact'] for r in current if r['prediction'] is not None),
            direct_path_parity_checks=1, generator_forwards=123, private_identity_queries=0,
            default_kept='initial', deployed=False, full_regression_performed=False,
            limitations=['58 fitting/replay images are not generalization accuracy',
                         '64 development images measure endpoints/false selections, not strict SDF accuracy',
                         'GPU baseline is freshly paired; historical CPU strict counts are not substituted'])
        write_json(trial.OUTPUT / 'report.json', report)
        emit('completed', optimizer_steps=limit.consumed, gate=gate)
        return report
