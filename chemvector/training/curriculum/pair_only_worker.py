"""Budgeted GPU worker for the exact one-shot pair-only matched trial."""

from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
import time

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from . import pair_only_trial as launch, pair_cached_trial as cached, pair_trial_contract as rules
from .bundle import load_bundle
from .checkpoints import file_digest
from .continuation import structure_forward
from .data import CurriculumCollator, load_sample, PAYLOADS
from .evaluation import FrozenOCRPredictor
from .gradient_preflight import state_digest
from .interface_diagnostic import tensor_tree_digest, write_json
from .runtime import StepLimit
from .validation_truth import expected_identity

INITIAL_STATE = '9d7f66f79329479e05c38f8b91d95118adb9f12c32a44e5bd9d83a458a8b452e'


def save_delta(path, payload):
    """Publish exclusively and atomically, never overwrite a model file."""
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError('Pair deltas cannot overwrite a file')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.pair-delta-', delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return file_digest(path)


def worker():
    started = time.monotonic()
    manifest, admission, auth = launch.authorize()
    auth_sha = file_digest(launch.AUTHORIZATION)
    reservation = json.loads(launch.ATTEMPT.read_text())
    if (reservation.get('authorization_sha256') != auth_sha
            or os.environ.get('CUDA_VISIBLE_DEVICES') != '0'
            or not launch.OUTPUT.is_dir() or (launch.OUTPUT / 'plan.json').exists()):
        raise ValueError('Fresh output, exact reservation and GPU 0 required')
    torch.set_num_threads(2)
    torch.manual_seed(rules.RECIPE['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA probe failed; this invocation stops, with no CPU fallback')
    torch.cuda.manual_seed_all(rules.RECIPE['seed'])
    device = torch.device('cuda:0')
    props = torch.cuda.get_device_properties(device)
    environment = dict(device=str(device), name=props.name, uuid=str(props.uuid),
        memory_bytes=props.total_memory, torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
        cudnn_version=torch.backends.cudnn.version(), tf32=False, autocast=False, dtype='float32',
        cudnn_benchmark=False, cudnn_deterministic=True)
    rows = manifest['training_samples']
    write_json(launch.OUTPUT / 'plan.json', dict(schema_version='pair-only-execution-plan-0.1',
        authorization_sha256=auth_sha, protocol_sha256=launch.PROPOSAL_SHA, recipe=rules.RECIPE,
        samples=rows, environment=environment, admission=admission,
        validation_sample_artifacts_read=False, new_identity_queries=0, deployment_authorized=False))
    limit = StepLimit(rules.RECIPE['maximum_steps'])
    counters = Counter(upstream_forwards=0, batch_attempts=0, skipped_empty_batches=0, evaluation_images=0)
    with (launch.OUTPUT / 'events.jsonl').open('x', encoding='utf-8') as log:
        def emit(event, **values):
            record = dict(event=event, elapsed_seconds=time.monotonic() - started, **values)
            value = json.dumps(record, ensure_ascii=False, allow_nan=False)
            log.write(value + '\n'); log.flush(); os.fsync(log.fileno())
            print(value, flush=True)

        def time_guard():
            # Reserve time for the final immutable receipts; the independent
            # parent kills this process group at the full 600-second ceiling.
            if time.monotonic() - started >= rules.RECIPE['wall_seconds'] - 45:
                raise TimeoutError('Worker stopping before its hard wall limit')

        try:
            model, config, payload = load_bundle(launch.INITIAL, expected_sha256=rules.INITIAL_SHA)
            if state_digest(model) != INITIAL_STATE or payload['progress']['optimizer_steps'] != 0:
                raise ValueError('Initial model/state differs')
            for name in ('image_size', 'maximum_atoms', 'atom_match_radius_px'):
                if getattr(config, name) != rules.RECIPE[name]:
                    raise ValueError('Initial proposal configuration differs')
            for name in ('atom_threshold', 'bond_threshold', 'pair_threshold', 'role_threshold'):
                if payload['provenance'][name] != rules.RECIPE[name]:
                    raise ValueError('Initial inference threshold differs')
            model.to(device)
            stage = rules.configure_pair_stage(model, config)
            predictor = FrozenOCRPredictor(payload['provenance'], device)
            frozen_before, ocr_before = cached.frozen_digest(model), state_digest(predictor.ocr)
            pair_before = state_digest(model.pair)
            collator = CurriculumCollator(AtomBondSceneConfig(atom_queries=config.maximum_atoms,
                bond_queries=config.maximum_bonds), purpose='training')
            caches, samples, pixels_by_key, cache_records = {}, {}, {}, []
            input_hashes = {}
            for ordinal, row in enumerate(rows):
                time_guard()
                key = row['source'] + ':' + row['sample_key']
                root = Path(manifest['sources'][row['source']])
                meta_path = root / 'samples' / (row['sample_key'] + '.meta.json')
                meta = launch.read_bound(meta_path, row['meta_sha256'])
                sample = load_sample(root, row['sample_key'], purpose='training')
                if sample['graph']['canonical_smiles'] != row['canonical_smiles']:
                    raise ValueError('Fixed training row identity differs')
                input_hashes[str(meta_path)] = row['meta_sha256']
                for suffix in PAYLOADS:
                    input_hashes[str(root / 'samples' / (row['sample_key'] + suffix))] = meta['sha256'][suffix]
                batch = collator([sample])
                pixels = batch['pixel_values'].to(device)
                target = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                counters['upstream_forwards'] += 1
                if counters['upstream_forwards'] > 58:
                    raise ValueError('Frozen upstream forward cap exceeded')
                with torch.no_grad():
                    forward = structure_forward(model, pixels, config, rules.RECIPE['atom_threshold'])
                    cache = cached.make_cache(forward, target, config)
                cache.update(key=key, group=row['fit_domain'])
                samples[key], pixels_by_key[key], caches[key] = sample, batch['pixel_values'], cache
                receipt = dict(key=key, image_sha256=meta['sha256']['.png'], target_sha256=meta['sha256']['.layers.json'],
                    batch_target_sha256=tensor_tree_digest({k: v for k, v in target.items() if k != 'pixel_values'}),
                    initial_sha256=rules.INITIAL_SHA, manifest_sha256=rules.MANIFEST_SHA,
                    protocol_sha256=launch.PROPOSAL_SHA, recipe_sha256=cached.json_digest(rules.RECIPE),
                    implementation_sha256=auth['implementation_sha256'], environment=environment,
                    **{k: cache[k] for k in ('candidate_label_sha256', 'feature_sha256', 'cache_sha256', 'coverage')})
                cache_records.append(receipt)
                write_json(launch.OUTPUT / ('cache--' + row['sample_key'] + '.json'), receipt)
                emit('cache_complete', ordinal=ordinal + 1, key=key, coverage=cache['coverage'])
                del forward, target, pixels, batch
            if len(caches) != 58 or state_digest(model) != INITIAL_STATE:
                raise ValueError('Incomplete cache or baseline model mutated')
            write_json(launch.OUTPUT / 'cache-manifest.json', cache_records)
            cache_manifest_sha = file_digest(launch.OUTPUT / 'cache-manifest.json')

            def evaluate(step):
                records = []
                cached.assert_stage(stage)
                with torch.no_grad():
                    for row in rows:
                        time_guard()
                        key = row['source'] + ':' + row['sample_key']
                        cache, sample = caches[key], samples[key]
                        cached.verify_cache(cache)
                        features, selected = cached.pair_inputs(cache, device)
                        structure = predictor.from_predicted_atoms(model.pair, features, selected)
                        trace = {}
                        prediction = predictor.from_structure(structure, model.role, model.bbox,
                            pixels_by_key[key].to(device), sample['image'], trace=trace)
                        metric = cached.score_case(cache, structure, prediction, trace,
                            key=key, expected=expected_identity(sample))
                        metric['group'] = row['fit_domain']
                        records.append(metric)
                        write_json(launch.OUTPUT / (f'check-{step:06d}--' + row['sample_key'] + '.json'),
                            dict(metrics=metric, prediction=prediction, trace=trace,
                                 presence_logits=structure['output']['presence_logits'].cpu().tolist(),
                                 chemical_logits=structure['output']['chemical_logits'].cpu().tolist(),
                                 pairs=structure['pairs'].cpu().tolist()))
                        counters['evaluation_images'] += 1
                if len(records) != 58 or {r['key'] for r in records} != set(caches):
                    raise ValueError('Full training/replay denominator changed')
                if cached.frozen_digest(model) != frozen_before or state_digest(predictor.ocr) != ocr_before:
                    raise ValueError('Frozen model/OCR state changed')
                emit('evaluation_complete', step=step, summary=cached.summarize(records))
                return records

            baseline = evaluate(0)
            write_json(launch.OUTPUT / 'baseline.json', dict(records=baseline, summary=cached.summarize(baseline)))
            if not sum(c['coverage']['reachable_positive_pairs'] for c in caches.values()):
                raise ValueError('No reachable positives; refuse to create optimizer')
            params = cached.assert_stage(stage)
            optimizer = torch.optim.AdamW(params, lr=rules.RECIPE['learning_rate'],
                betas=tuple(rules.RECIPE['betas']), eps=rules.RECIPE['epsilon'], weight_decay=0.)
            emit('training_started', trainable_parameters=sum(p.numel() for p in params),
                 upstream_forwards=counters['upstream_forwards'], frozen_state_sha256=frozen_before)
            decisions, final = [], baseline
            stop_reason = 'batch_attempt_cap'
            for attempt in range(rules.RECIPE['maximum_batch_attempts']):
                time_guard()
                row = rows[attempt % len(rows)]
                key = row['source'] + ':' + row['sample_key']
                counters['batch_attempts'] += 1
                result = cached.update_once(stage, caches[key], optimizer, limit, attempt=attempt, emit=emit)
                emit('batch_complete', key=key, attempt=attempt, **result)
                if not result['updated']:
                    counters['skipped_empty_batches'] += 1
                    continue
                if limit.consumed in rules.RECIPE['checkpoints']:
                    final = evaluate(limit.consumed)
                    decision = rules.checkpoint_decision(baseline, final, step=limit.consumed)
                    decisions.append(dict(step=limit.consumed, **decision))
                    write_json(launch.OUTPUT / f'decision-{limit.consumed:06d}.json', decisions[-1])
                    delta = dict(schema_version='chemvector-pair-only-delta-0.1',
                        initial_sha256=rules.INITIAL_SHA, protocol_sha256=launch.PROPOSAL_SHA,
                        authorization_sha256=auth_sha, manifest_sha256=rules.MANIFEST_SHA,
                        cache_manifest_sha256=cache_manifest_sha, pair_supervision=rules.RECIPE['pair_supervision'],
                        recipe=rules.RECIPE, architecture=asdict(model.pair.config),
                        pair_state={k: v.detach().cpu().clone() for k, v in model.pair.state_dict().items()},
                        frozen_state_sha256=frozen_before, ocr_state_sha256=ocr_before,
                        optimizer_steps=limit.consumed, decision=decision, deployment_authorized=False,
                        resume_supported=False)
                    save_delta(launch.OUTPUT / f'pair-delta-{limit.consumed:06d}.pt', delta)
                    emit('checkpoint_complete', step=limit.consumed, decision=decision)
                    if decision['action'] != 'continue_within_original_budget':
                        stop_reason = decision['action']
                        break
                if limit.consumed == limit.maximum:
                    stop_reason = 'step_cap'
                    break
            for cache in caches.values(): cached.verify_cache(cache)
            for path, digest in input_hashes.items():
                if file_digest(path) != digest: raise ValueError('Training input changed')
            launch.authorize()
            cached.assert_stage(stage)
            frozen_after, ocr_after = cached.frozen_digest(model), state_digest(predictor.ocr)
            if frozen_after != frozen_before or ocr_after != ocr_before:
                raise ValueError('Final frozen state changed')
            write_json(launch.OUTPUT / 'report.json', dict(schema_version='pair-only-trial-result-0.1',
                complete=True, optimizer_steps=limit.consumed, counters=dict(counters), stop_reason=stop_reason,
                baseline=cached.summarize(baseline), final=cached.summarize(final), decisions=decisions,
                authorization_sha256=auth_sha, protocol_sha256=launch.PROPOSAL_SHA,
                cache_manifest_sha256=cache_manifest_sha, input_sha256=input_hashes,
                frozen_state_before=frozen_before, frozen_state_after=frozen_after,
                ocr_state_before=ocr_before, ocr_state_after=ocr_after,
                pair_state_before=pair_before, pair_state_after=state_digest(model.pair),
                environment=environment, elapsed_seconds=time.monotonic() - started,
                training_performed=limit.consumed > 0, gpu_used=True,
                default_changed=False, deployment_authorized=False, automatic_retries=0,
                validation_sample_artifacts_read=False, sealed_test_read=False, new_identity_queries=0,
                independent_accuracy_measured=False,
                limitations=['Training/replay fit only, not independent recognition accuracy',
                    'Mask removes unknown negatives but creates no missing positive supervision',
                    'Single arm does not isolate supervision-mask causality',
                    'No automatic deployment, resume, alternate seed, threshold change or follow-up trial']))
            emit('trial_complete', steps=limit.consumed, stop_reason=stop_reason)
        except Exception as error:
            emit('trial_failed', error_type=type(error).__name__, error=str(error), steps=limit.consumed)
            if not (launch.OUTPUT / 'failure.json').exists():
                write_json(launch.OUTPUT / 'failure.json', dict(error_type=type(error).__name__, error=str(error),
                    optimizer_steps=limit.consumed, counters=dict(counters), automatic_retries=0))
            raise
