"""Authorized GPU execution only. Never imported to run by CPU preparation."""

from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from . import pair_anchor_ab_trial as launch, pair_anchor_ab_contract as rules, pair_anchor_device as adapter
from . import pair_cached_trial as cached
from .bundle import load_bundle
from .checkpoints import file_digest
from .continuation import structure_forward
from .data import CurriculumCollator, load_sample, PAYLOADS
from .evaluation import FrozenOCRPredictor
from .gradient_preflight import state_digest
from .interface_diagnostic import write_json
from .pair_only_worker import INITIAL_STATE, save_delta
from .runtime import StepLimit
from .validation_truth import expected_identity


def tensor_bytes(value):
    if isinstance(value, torch.Tensor): return value.numel() * value.element_size()
    if isinstance(value, dict): return sum(tensor_bytes(v) for v in value.values())
    return 0


def worker():
    started = time.monotonic()
    manifest, admission, auth = launch.authorize()
    auth_sha = file_digest(launch.AUTHORIZATION)
    if (json.loads(launch.ATTEMPT.read_text()).get('authorization_sha256') != auth_sha
            or os.environ.get('CUDA_VISIBLE_DEVICES') != '0' or not launch.OUTPUT.is_dir()):
        raise ValueError('Fresh authorized reservation and GPU 0 required')
    # Exclusive second-stage reservation prevents direct --worker retry even
    # if an earlier worker failed before writing its plan (e.g. CUDA failure).
    write_json(launch.OUTPUT / 'worker-started.json', dict(authorization_sha256=auth_sha, pid=os.getpid()))
    counters = Counter(upstream_forwards=0, evaluation_images=0, sampled_parity_forwards=0,
                       training_pair_forwards=0, cache_tensor_bytes=0, attempts=0)
    total = StepLimit(108)
    limits = {arm: StepLimit(54) for arm in rules.ARMS}
    artifacts, inputs, decisions = {}, {}, []
    recipe = rules.protocol()['recipe']

    def write(name, value):
        write_json(launch.OUTPUT / name, value); artifacts[name] = file_digest(launch.OUTPUT / name)

    def guard():
        if time.monotonic() - started >= rules.LIMITS['wall_seconds'] - 30:
            raise TimeoutError('Stop before the all-inclusive hard wall limit; no resume')

    with (launch.OUTPUT / 'events.jsonl').open('x', encoding='utf-8') as stream:
        def emit(event, **values):
            row = dict(event=event, elapsed_seconds=time.monotonic()-started, **values)
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n'); stream.flush()
            print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)
        try:
            torch.set_num_threads(2); torch.manual_seed(recipe['seed'])
            torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
            if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; no CPU fallback or retry')
            torch.cuda.manual_seed_all(recipe['seed'])
            device = torch.device('cuda:0'); props = torch.cuda.get_device_properties(device)
            environment = dict(device=str(device), name=props.name, uuid=str(props.uuid),
                torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                cudnn_version=torch.backends.cudnn.version(), tf32=False, autocast=False, dtype='float32')
            rows = manifest['training_samples']
            write('plan.json', dict(protocol=rules.protocol(), protocol_sha256=auth['protocol_sha256'],
                authorization_sha256=auth_sha, admission=admission, environment=environment,
                sample_keys=[r['source'] + ':' + r['sample_key'] for r in rows]))
            guard()
            model, config, payload = load_bundle(launch.INITIAL, expected_sha256=rules.original.INITIAL_SHA)
            if state_digest(model) != INITIAL_STATE or payload['progress']['optimizer_steps'] != 0:
                raise ValueError('Original initial required')
            for key in ('image_size', 'maximum_atoms', 'atom_match_radius_px'):
                if getattr(config, key) != recipe[key]: raise ValueError('Initial config changed')
            for key in ('atom_threshold', 'bond_threshold', 'pair_threshold', 'role_threshold'):
                if payload['provenance'][key] != recipe[key]: raise ValueError('Initial thresholds changed')
            model.to(device)
            stage = rules.original.configure_pair_stage(model, config)
            predictor = FrozenOCRPredictor(payload['provenance'], device)
            frozen_before, ocr_before = cached.frozen_digest(model), state_digest(predictor.ocr)
            heads = dict(control=model.pair, augmentation=deepcopy(model.pair))
            if state_digest(heads['control']) != state_digest(heads['augmentation']):
                raise ValueError('Pair arms not identically initialized')
            initial_pair_sha = state_digest(heads['control'])
            collator = CurriculumCollator(AtomBondSceneConfig(atom_queries=config.maximum_atoms,
                bond_queries=config.maximum_bonds), purpose='training')
            caches, plans, samples, pixels, receipts = {}, {}, {}, {}, []
            for ordinal, row in enumerate(rows):
                guard(); key = row['source'] + ':' + row['sample_key']
                root = Path(manifest['sources'][row['source']])
                meta_path = root / 'samples' / (row['sample_key'] + '.meta.json')
                meta = launch.old.read_bound(meta_path, row['meta_sha256'])
                sample = load_sample(root, row['sample_key'], purpose='training')
                if sample['graph']['canonical_smiles'] != row['canonical_smiles']: raise ValueError('Training identity changed')
                inputs[str(meta_path)] = row['meta_sha256']
                for suffix in PAYLOADS: inputs[str(root / 'samples' / (row['sample_key'] + suffix))] = meta['sha256'][suffix]
                batch = collator([sample]); target = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                with torch.no_grad():
                    forward = structure_forward(model, target['pixel_values'], config, recipe['atom_threshold'])
                    cache = cached.make_cache(forward, target, config)
                    plan = adapter.make_plan(cache, target['atom_anchors'][:, :cache['coverage']['target_atoms']],
                                             ordinal=ordinal, execution_device=str(device))
                cache.update(key=key, group=row['fit_domain'])
                counters['upstream_forwards'] += 1
                counters['cache_tensor_bytes'] += tensor_bytes(cache) + tensor_bytes(plan) + tensor_bytes(batch['pixel_values'])
                if counters['upstream_forwards'] > 58 or counters['cache_tensor_bytes'] > rules.LIMITS['cache_tensor_bytes']:
                    raise ValueError('Fresh cache resource cap exceeded')
                caches[key], plans[key], samples[key], pixels[key] = cache, plan, sample, batch['pixel_values']
                receipt = dict(key=key, environment=environment, initial_sha256=rules.original.INITIAL_SHA,
                    protocol_sha256=auth['protocol_sha256'], image_sha256=meta['sha256']['.png'],
                    target_sha256=meta['sha256']['.layers.json'],
                    **{k: cache[k] for k in ('cache_sha256', 'candidate_label_sha256', 'feature_sha256', 'coverage')},
                    plan_binding={k: v for k, v in plan.items() if k != 'tensors'},
                    base_row_ids=plan['tensors']['chosen'].tolist(),
                    augmented_row_ids=plan['tensors']['augmented_rows'].tolist(),
                    augmented_sampled_positions=plan['tensors']['augmented_positions'].tolist())
                receipts.append(receipt); emit('cache_complete', key=key, coverage=cache['coverage'])
                del target, batch, forward
            if len(caches) != 58 or state_digest(model) != INITIAL_STATE:
                raise ValueError('Incomplete cache or initial changed')
            expected_updates = sum(bool(len(p['tensors']['chosen'])) for p in plans.values())
            if not 0 < expected_updates <= 54:
                raise ValueError('Fresh cache cannot complete full58 under declared update cap; do not truncate')
            write('cache-manifest.json', receipts)
            for key in caches:
                guard()
                parity = adapter.zero_weight_parity(model.pair, caches[key], plans[key])
                counters['sampled_parity_forwards'] += parity['forwards']
                if counters['sampled_parity_forwards'] > 108: raise ValueError('Parity forward cap')
            emit('fresh_device_adapter_parity_passed', images=58, prospective_updates_per_arm=expected_updates)

            def evaluate(arm, attempt):
                model.pair = heads[arm]; cached.assert_stage(stage)
                records = []
                with torch.no_grad():
                    for row in rows:
                        guard(); key = row['source'] + ':' + row['sample_key']
                        cache = caches[key]; adapter.verify_plan(cache, plans[key])
                        features, selected = cached.pair_inputs(cache, device)
                        structure = predictor.from_predicted_atoms(model.pair, features, selected)
                        trace = {}
                        prediction = predictor.from_structure(structure, model.role, model.bbox,
                            pixels[key].to(device), samples[key]['image'], trace=trace)
                        metric = adapter.score_case(cache, structure, prediction, trace,
                                                    key=key, expected=expected_identity(samples[key]))
                        metric['group'] = row['fit_domain']; records.append(metric)
                        write(f'{arm}-{attempt:03d}--{row["sample_key"]}.json', dict(metrics=metric,
                            prediction=prediction, trace=trace, pairs=structure['pairs'].cpu().tolist(),
                            presence_logits=structure['output']['presence_logits'].cpu().tolist(),
                            chemical_logits=structure['output']['chemical_logits'].cpu().tolist(),
                            promoted_chemical=structure['chemistry'].cpu().tolist()))
                        counters['evaluation_images'] += 1
                        if counters['evaluation_images'] > 348: raise ValueError('Full58 evaluation cap')
                if len(records) != 58 or cached.frozen_digest(model) != frozen_before or state_digest(predictor.ocr) != ocr_before:
                    raise ValueError('Full cohort or frozen upstream/OCR changed')
                write(f'{arm}-{attempt:03d}.json', dict(records=records, summary=cached.summarize(records)))
                return records

            baseline = evaluate('control', 0)
            second = evaluate('augmentation', 0)
            if cached.json_digest(baseline) != cached.json_digest(second):
                raise ValueError('Full58 initial arm parity failed before optimizer creation')
            optimizers = {arm: torch.optim.AdamW(heads[arm].parameters(), lr=recipe['learning_rate'],
                betas=tuple(recipe['betas']), eps=recipe['epsilon'], weight_decay=recipe['weight_decay']) for arm in rules.ARMS}
            final, stop_reason = {arm: baseline for arm in rules.ARMS}, 'attempt_cap'
            for ordinal, row in enumerate(rows):
                key = row['source'] + ':' + row['sample_key']; guard()
                for arm, weight in (('control', 0), ('augmentation', 1)):
                    model.pair = heads[arm]
                    result = adapter.update(stage, caches[key], plans[key], optimizers[arm], limits[arm], total,
                        weight=weight, emit=lambda event, **values: emit(event, arm=arm, attempt=ordinal+1, **values))
                    counters['training_pair_forwards'] += result['forwards']
                    if counters['training_pair_forwards'] > 162: raise ValueError('Training pair forward cap')
                    emit('arm_batch_complete', arm=arm, attempt=ordinal+1, key=key, **result)
                    guard()
                counters['attempts'] += 1
                if limits['control'].consumed != limits['augmentation'].consumed:
                    raise ValueError('Paired prefix update counts differ')
                if ordinal+1 in rules.LIMITS['checkpoint_attempts']:
                    final = {arm: evaluate(arm, ordinal+1) for arm in rules.ARMS}
                    decision = rules.paired_decision(baseline, final['control'], final['augmentation'], attempt=ordinal+1)
                    decisions.append(dict(attempt=ordinal+1, **decision))
                    write(f'decision-{ordinal+1:03d}.json', decisions[-1])
                    for arm in rules.ARMS:
                        name = f'{arm}-pair-{ordinal+1:03d}.pt'
                        artifacts[name] = save_delta(launch.OUTPUT / name, dict(
                            schema_version='pair-anchor-ab-delta-0.1', arm=arm, optimizer_steps=limits[arm].consumed,
                            initial_sha256=rules.original.INITIAL_SHA, protocol_sha256=auth['protocol_sha256'],
                            authorization_sha256=auth_sha, cache_manifest_sha256=artifacts['cache-manifest.json'],
                            architecture=asdict(model.pair.config), pair_state={k: v.detach().cpu() for k, v in heads[arm].state_dict().items()},
                            deployment_authorized=False, resume_supported=False, decision=decision))
                    emit('paired_checkpoint', attempt=ordinal+1, decision=decision)
                    if decision['action'] != 'continue_within_fixed_budget':
                        stop_reason = decision['action']; break
            for key in caches: adapter.verify_plan(caches[key], plans[key])
            for path, digest in inputs.items():
                if file_digest(path) != digest: raise ValueError('Training input changed')
            guard(); launch.authorize()
            if cached.frozen_digest(model) != frozen_before or state_digest(predictor.ocr) != ocr_before:
                raise ValueError('Frozen state changed at exit')
            write('report.json', dict(schema_version='pair-anchor-ab-result-0.1', complete=True,
                authorization_sha256=auth_sha, protocol_sha256=auth['protocol_sha256'],
                counters=counters, updates_per_arm={a: limits[a].consumed for a in rules.ARMS}, total_updates=total.consumed,
                stop_reason=stop_reason, decisions=decisions, baseline=cached.summarize(baseline),
                final={a: cached.summarize(final[a]) for a in rules.ARMS},
                input_sha256=inputs, artifact_sha256=dict(artifacts), environment=environment,
                frozen_state_sha256=frozen_before, ocr_state_sha256=ocr_before, initial_pair_sha256=initial_pair_sha,
                final_pair_sha256={a: state_digest(heads[a]) for a in rules.ARMS},
                training_performed=total.consumed > 0, gpu_used=True, independent_accuracy_measured=False,
                default_changed=False, data_release_changed=False, deployment_authorized=False,
                new_identity_queries=0, validation_sample_artifacts_read=False, sealed_test_read=False,
                elapsed_seconds=time.monotonic()-started))
            emit('paired_trial_complete', total_updates=total.consumed, stop_reason=stop_reason)
        except Exception as error:
            emit('trial_failed', error_type=type(error).__name__, error=str(error), total_updates=total.consumed)
            write('failure.json', dict(error_type=type(error).__name__, error=str(error), counters=counters,
                total_updates=total.consumed, updates_per_arm={a: limits[a].consumed for a in rules.ARMS},
                success=False, incomplete=True, automatic_retries=0))
            raise
