"""One bounded real diagnostic run; original objective, no second trial."""

from collections import Counter
import json
import os
from pathlib import Path
import time

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from . import pair_observed_trial as launch, pair_cached_trial as cached, pair_trial_contract as rules
from .bundle import load_bundle
from .checkpoints import file_digest
from .continuation import structure_forward
from .data import CurriculumCollator, load_sample, PAYLOADS
from .evaluation import FrozenOCRPredictor
from .gradient_preflight import state_digest
from .interface_diagnostic import tensor_tree_digest, write_json
from .pair_event_analysis import analyze_one_event, check_adam_transition, choose_analysis_event
from .pair_negative_observer import bind_negative_probe, observe
from .pair_observed_capture import (CapturedUpdates, DurableEvents, check_probe_parity,
                                    cpu_copy, tensor_bytes, summarize_trajectories, termination_status)
from .pair_only_worker import INITIAL_STATE, save_delta
from .runtime import StepLimit
from .validation_truth import expected_identity


def worker():
    started = time.monotonic()
    manifest, admission, auth = launch.authorize()
    auth_sha = file_digest(launch.AUTHORIZATION)
    reservation = json.loads(launch.ATTEMPT.read_text())
    if (reservation.get('authorization_sha256') != auth_sha
            or os.environ.get('CUDA_VISIBLE_DEVICES') != '0'
            or not launch.OUTPUT.is_dir() or (launch.OUTPUT / 'plan.json').exists()):
        raise ValueError('Fresh reserved GPU diagnostic worker required')
    torch.set_num_threads(2); torch.manual_seed(rules.RECIPE['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; no fallback or retry')
    torch.cuda.manual_seed_all(rules.RECIPE['seed'])
    device = torch.device('cuda:0'); props = torch.cuda.get_device_properties(device)
    environment = dict(device=str(device), gpu=props.name, uuid=str(props.uuid),
        torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
        cudnn_version=torch.backends.cudnn.version(), tf32=False, autocast=False,
        dtype='float32', cudnn_benchmark=False, cudnn_deterministic=True)
    rows = manifest['training_samples']; order = [r['source'] + ':' + r['sample_key'] for r in rows]
    write_json(launch.OUTPUT / 'plan.json', dict(protocol=launch.protocol(), samples=rows,
        authorization_sha256=auth_sha, protocol_sha256=auth['protocol_sha256'], environment=environment,
        admission=admission, implementation_sha256=auth['implementation_sha256']))
    limit = StepLimit(50)
    counters = Counter(upstream_forwards=0, batch_attempts=0, evaluation_images=0,
                       parity_probe_forwards=0, analysis_pair_forwards=0)
    inputs, artifacts = {}, {}

    def write(name, value):
        write_json(launch.OUTPUT / name, value)
        artifacts[name] = file_digest(launch.OUTPUT / name)

    def save(name, value):
        artifacts[name] = save_delta(launch.OUTPUT / name, value)

    def time_guard():
        if time.monotonic() - started >= 555:
            raise TimeoutError('Stopping before independent 600-second hard limit')

    with (launch.OUTPUT / 'events.jsonl').open('x', encoding='utf-8') as stream:
        emit = DurableEvents(stream, lambda: time.monotonic() - started)
        try:
            model, config, payload = load_bundle(launch.INITIAL, expected_sha256=rules.INITIAL_SHA)
            if state_digest(model) != INITIAL_STATE or payload['progress']['optimizer_steps'] != 0:
                raise ValueError('Original initial state required')
            for name in ('image_size', 'maximum_atoms', 'atom_match_radius_px'):
                if getattr(config, name) != rules.RECIPE[name]: raise ValueError('Initial config changed')
            for name in ('atom_threshold', 'bond_threshold', 'pair_threshold', 'role_threshold'):
                if payload['provenance'][name] != rules.RECIPE[name]: raise ValueError('Threshold changed')
            model.to(device); stage = rules.configure_pair_stage(model, config)
            predictor = FrozenOCRPredictor(payload['provenance'], device)
            frozen_before, ocr_before = cached.frozen_digest(model), state_digest(predictor.ocr)
            pair_before = state_digest(model.pair)
            collator = CurriculumCollator(AtomBondSceneConfig(atom_queries=config.maximum_atoms,
                bond_queries=config.maximum_bonds), purpose='training')
            caches, samples, pixels, receipts = {}, {}, {}, []
            for row, key in zip(rows, order):
                time_guard()
                root = Path(manifest['sources'][row['source']])
                meta_path = root / 'samples' / (row['sample_key'] + '.meta.json')
                meta = launch.old.read_bound(meta_path, row['meta_sha256'])
                sample = load_sample(root, row['sample_key'], purpose='training')
                if sample['graph']['canonical_smiles'] != row['canonical_smiles']:
                    raise ValueError('Fixed sample identity changed')
                inputs[str(meta_path)] = row['meta_sha256']
                for suffix in PAYLOADS:
                    inputs[str(root / 'samples' / (row['sample_key'] + suffix))] = meta['sha256'][suffix]
                batch = collator([sample])
                target = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                counters['upstream_forwards'] += 1
                if counters['upstream_forwards'] > 58: raise ValueError('Upstream forward cap')
                with torch.no_grad():
                    forward = structure_forward(model, target['pixel_values'], config, rules.RECIPE['atom_threshold'])
                    cache = cached.make_cache(forward, target, config)
                cache.update(key=key, group=row['fit_domain'])
                caches[key], samples[key], pixels[key] = cache, sample, batch['pixel_values']
                receipts.append(dict(key=key, image_sha256=meta['sha256']['.png'],
                    target_sha256=meta['sha256']['.layers.json'],
                    batch_target_sha256=tensor_tree_digest({k: v for k, v in target.items() if k != 'pixel_values'}),
                    **{k: cache[k] for k in ('candidate_label_sha256', 'feature_sha256', 'cache_sha256', 'coverage')}))
                emit('cache_complete', ordinal=len(caches), key=key, cache_sha256=cache['cache_sha256'])
                del batch, target, forward
            if len(caches) != 58 or state_digest(model) != INITIAL_STATE:
                raise ValueError('Cache construction changed initial or cohort')
            write('cache-manifest.json', receipts)
            probes = [(caches[key], targets) for key, targets in launch.PROBES]
            bindings = {c['key']: bind_negative_probe(c, targets) for c, targets in probes}
            write('probe-bindings.json', bindings)
            parity = []

            def evaluate(label):
                records = []
                cached.assert_stage(stage)
                with torch.no_grad():
                    for row, key in zip(rows, order):
                        time_guard(); cache, sample = caches[key], samples[key]
                        cached.verify_cache(cache)
                        features, selected = cached.pair_inputs(cache, device)
                        structure = predictor.from_predicted_atoms(model.pair, features, selected)
                        if key in bindings:
                            single = observe(model.pair, cache, bindings[key])
                            parity.append(dict(label=label, key=key,
                                **check_probe_parity(single, structure['output'], bindings[key]['row_id'])))
                            counters['parity_probe_forwards'] += 1
                        trace = {}
                        prediction = predictor.from_structure(structure, model.role, model.bbox,
                            pixels[key].to(device), sample['image'], trace=trace)
                        metric = cached.score_case(cache, structure, prediction, trace,
                            key=key, expected=expected_identity(sample))
                        metric['group'] = row['fit_domain']; records.append(metric)
                        write(f'check-{label}--' + row['sample_key'] + '.json', dict(metrics=metric,
                            prediction=prediction, trace=trace,
                            presence_logits=structure['output']['presence_logits'].cpu().tolist(),
                            chemical_logits=structure['output']['chemical_logits'].cpu().tolist(),
                            pairs=structure['pairs'].cpu().tolist()))
                        counters['evaluation_images'] += 1
                if len(records) != 58 or {r['key'] for r in records} != set(order):
                    raise ValueError('Full58 evaluation incomplete')
                if cached.frozen_digest(model) != frozen_before or state_digest(predictor.ocr) != ocr_before:
                    raise ValueError('Frozen state changed')
                emit('evaluation_complete', label=label, actual_step=limit.consumed, summary=cached.summarize(records))
                return records

            baseline = evaluate('baseline')
            write('baseline.json', dict(records=baseline, summary=cached.summarize(baseline)))
            if not sum(c['coverage']['reachable_positive_pairs'] for c in caches.values()):
                raise ValueError('No reachable positive supervision')
            optimizer = torch.optim.AdamW(cached.assert_stage(stage), lr=rules.RECIPE['learning_rate'],
                betas=tuple(rules.RECIPE['betas']), eps=rules.RECIPE['epsilon'], weight_decay=0.)
            captured = CapturedUpdates(stage, optimizer, limit, probes, order, emit=emit)
            updates = []
            for attempt, key in enumerate(order):
                time_guard()
                if limit.consumed >= 50: break
                counters['batch_attempts'] += 1
                result = captured.update(caches[key], attempt=attempt)
                updates.append(result)
                emit('batch_complete', key=key, attempt=attempt, **result['update'])
            termination = termination_status(limit.consumed)
            # Always evaluate actual terminal weights, including <50-step runs.
            final = evaluate('final')
            write('final.json', dict(actual_step=limit.consumed, records=final, summary=cached.summarize(final)))
            decision = (rules.checkpoint_decision(baseline, final, step=50) if limit.consumed == 50
                        else dict(action='incomplete_no_promotion', full_50_step_gate_evaluated=False))
            write('decision.json', decision); write('probe-parity.json', parity)
            trajectories = summarize_trajectories(updates)
            write('trajectories.json', dict(summary=trajectories, updates=updates))
            snapshots = dict(slots=captured.slots, events={}, probes={}, snapshot_tensor_bytes=0)
            for key, binding in bindings.items():
                time_guard(); name = 'probe-cache--' + key.split(':', 1)[1] + '.pt'
                saved = dict(cache=caches[key], binding=binding)
                snapshots['snapshot_tensor_bytes'] += tensor_bytes(saved)
                save(name, cpu_copy(saved)); snapshots['probes'][key] = name
            for step, event in sorted(captured.events.items()):
                time_guard(); name = f'event-{step:06d}.pt'
                snapshots['snapshot_tensor_bytes'] += tensor_bytes(event)
                if snapshots['snapshot_tensor_bytes'] > 1024**3: raise ValueError('Snapshot disk tensor cap')
                arithmetic = check_adam_transition(event)
                save(name, dict(event=cpu_copy(event), initial_sha256=rules.INITIAL_SHA,
                    protocol_sha256=auth['protocol_sha256'], authorization_sha256=auth_sha,
                    manifest_sha256=rules.MANIFEST_SHA, environment=environment, deployment_authorized=False))
                snapshots['events'][str(step)] = dict(file=name, arithmetic=arithmetic)
            write('snapshot-manifest.json', snapshots)
            selection = choose_analysis_event(captured.events)
            analysis = dict(status='no_significant_rise_captured', optimizer_steps=0)
            if selection is not None:
                time_guard(); step, key = selection
                analysis = analyze_one_event(model.pair, captured.events[step], caches[key], bindings[key])
                counters['analysis_pair_forwards'] = analysis['pair_forwards']
            write('local-event-analysis.json', analysis)
            save('final-pair-diagnostic.pt', dict(pair_state=cpu_copy(model.pair.state_dict()),
                initial_sha256=rules.INITIAL_SHA, protocol_sha256=auth['protocol_sha256'],
                optimizer_steps=limit.consumed, deployment_authorized=False, resume_supported=False))
            for cache in caches.values(): cached.verify_cache(cache)
            for path, digest in inputs.items():
                if file_digest(path) != digest: raise ValueError('Training source changed')
            launch.authorize(); cached.assert_stage(stage)
            frozen_after, ocr_after = cached.frozen_digest(model), state_digest(predictor.ocr)
            if frozen_after != frozen_before or ocr_after != ocr_before: raise ValueError('Frozen final state changed')
            emit('diagnostic_complete', optimizer_steps=limit.consumed, gate=decision['action'])
            artifacts['events.jsonl'] = file_digest(launch.OUTPUT / 'events.jsonl')
            artifacts['plan.json'] = file_digest(launch.OUTPUT / 'plan.json')
            write_json(launch.OUTPUT / 'report.json', dict(schema_version='pair-observed-result-0.1',
                complete=True, **termination, optimizer_steps=limit.consumed, counters=dict(counters),
                baseline=cached.summarize(baseline), final=cached.summarize(final), decision=decision,
                trajectories=trajectories, selected_event_slots=captured.slots,
                event_analysis=analysis, snapshot_events=len(captured.events),
                input_sha256=inputs, artifact_sha256=artifacts,
                authorization_sha256=auth_sha, protocol_sha256=auth['protocol_sha256'],
                implementation_sha256=auth['implementation_sha256'],
                frozen_state_before=frozen_before, frozen_state_after=frozen_after,
                ocr_state_before=ocr_before, ocr_state_after=ocr_after,
                pair_state_before=pair_before, pair_state_after=state_digest(model.pair),
                gpu_used=True, training_performed=limit.consumed > 0, default_changed=False,
                validation_sample_artifacts_read=False, sealed_test_read=False, new_identity_queries=0,
                independent_accuracy_measured=False, automatic_retries=0,
                elapsed_worker_seconds=time.monotonic() - started,
                limitations=['Temporal effects on two known negatives, not complete loss-component causality',
                             'Four selected snapshots cannot fully explain distributed accumulation',
                             'No optimizer counterfactual or additional training arm',
                             'Fixed58 train/replay only; no deployment or independent generalization claim']))
        except BaseException as error:
            if not (launch.OUTPUT / 'failure.json').exists():
                write_json(launch.OUTPUT / 'failure.json', dict(error_type=type(error).__name__, error=str(error),
                    optimizer_steps_reserved=limit.consumed, counters=dict(counters),
                    diagnostic_complete=False, automatic_retries=0, deployment_authorized=False))
            try: emit('diagnostic_failed', error=str(error), steps_reserved=limit.consumed)
            except BaseException: pass
            raise
