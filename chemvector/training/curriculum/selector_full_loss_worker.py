"""Full-loss-only control: identical data, cache, sampler and gates; no resume."""

import json
import os
import time

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.predicted_incidence import associate_predicted_proposals
from . import selector_full_loss_stage as stage
from . import selector_full_loss_trial as trial
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


from .selector_retention_worker import assess, check_core_contract, cache_digest
from . import selector_full_loss_evidence as evidence


def worker():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise ValueError('Externally supervised single GPU 0 worker required')
    manifest, admission, auth = trial.authorize()
    reference_report = json.loads((trial.REFERENCE / 'report.json').read_text())
    reference_order = evidence.reference_updates(trial.REFERENCE)
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
    write_json(trial.OUTPUT / 'plan.json', dict(schema_version='chemvector-selector-full-loss-control-plan-0.1',
        objective='original_full_set_only', retention_weight=0.,
        reference_report_sha256=trial.REFERENCE_REPORT_SHA, full_regression_authorized=False,
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
                previous = json.loads((trial.REFERENCE / ('baseline--' + row['sample_key'] + '.json')).read_text())
                evidence.baseline_parity(record, previous)
                write_json(trial.OUTPUT / ('geometry--' + row['sample_key'] + '.json'), evidence.geometry(cache, config))
                write_json(trial.OUTPUT / ('matching-000000--' + row['sample_key'] + '.json'), evidence.matching(cache, record, config))
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
        if (cache_before != reference_report['cache_before']
                or visual_before != reference_report['visual_cache_before']
                or frozen_before != reference_report['frozen_before'] or ocr_before != reference_report['ocr_before']):
            raise ValueError('Control/reference frozen predictions, targets, or weights differ; no optimizer allowed')
        comparisons = {}
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
            if cache['key'] != reference_order[step]:
                raise ValueError('Control/reference minibatch sequence differs')
            losses = stage.update(model, cache, optimizer, limit, trial.MANIFEST_SHA, device,
                lambda s: emit('step_reserved', step=s, key=cache['key']))
            emit('update', step=step, key=cache['key'], **losses)
            if step not in (100, 200, 400):
                continue
            current = []
            for cache in fitting:
                record = assess(model, config, predictor, cache, device=device)
                current.append(record)
                write_json(trial.OUTPUT / (f'matching-{step:06d}--' + cache['sample_key'] + '.json'),
                    evidence.matching(cache, record, config))
                write_json(trial.OUTPUT / (f'check-{step:06d}--' + cache['sample_key'] + '.json'), record)
            local = stage.local_gate(base_fit, current, step)
            write_json(trial.OUTPUT / f'gate-{step:06d}.json', local)
            comparisons[str(step)] = {group: evidence.paired_reference(trial.REFERENCE,
                [r for r in current if r['group'] == group], step=step)
                for group in ('witness', 'old_full_molecule')}
            write_json(trial.OUTPUT / f'paired-{step:06d}.json', comparisons[str(step)])
            emit('local_check', **local)
            if local['stop_reasons']:
                break
        final_dev = []
        for cache in development:
            record = assess(model, config, predictor, cache, device=device)
            final_dev.append(record)
            write_json(trial.OUTPUT / ('matching-final-development--' + cache['sample_key'] + '.json'),
                evidence.matching(cache, record, config))
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
                trial='selector-full-loss-only-control-v1', actual_parameter_scope='set_head only',
                authorization_sha256=trial.file_digest(trial.AUTHORIZATION), deployed=False))
        trial.authorize()  # Recheck protected data, sources and default profile after fitting.
        report = dict(schema_version='chemvector-selector-full-loss-control-result-0.1',
            reference_report_sha256=trial.REFERENCE_REPORT_SHA, retention_weight=0.,
            baseline_matches_reference=True, cached_upstream_matches_reference=True,
            update_order_matches_reference=True, equal_step_comparisons=comparisons,
            new64_equal_step_comparison=evidence.paired_reference(trial.REFERENCE, final_dev,
                step=limit.consumed, development=True) if limit.consumed == 400 else None,
            development_summary=dict(baseline=evidence.summary(base_dev), final=evidence.summary(final_dev)),
            full_regression_authorized=False,
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
                         'Fresh baseline and caches match prior GPU reference exactly',
                         'Early-stop new64 is not compared with retention step400 as equal-step causality',
                         'Observed development panel and single seed; no architecture-wide conclusion'])
        write_json(trial.OUTPUT / 'report.json', report)
        emit('completed', optimizer_steps=limit.consumed, gate=gate)
        return report
