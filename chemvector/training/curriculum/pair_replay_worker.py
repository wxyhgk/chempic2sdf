"""Gated single-factor replay worker. Never invoked by preparation/tests."""

from collections import Counter
import json
import os
from pathlib import Path
import time

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from . import pair_replay_trial as launch, pair_cached_trial as cached, pair_trial_contract as rules
from .bundle import load_bundle
from .checkpoints import file_digest
from .continuation import structure_forward
from .data import CurriculumCollator, load_sample, PAYLOADS
from .evaluation import FrozenOCRPredictor
from .gradient_preflight import state_digest
from .interface_diagnostic import write_json, tensor_tree_digest
from .pair_event_analysis import check_adam_transition
from .pair_negative_observer import bind_negative_probe, observe
from .pair_nonedge_replay import ReplayBank, CapturedReplayUpdates
from .pair_observed_capture import (DurableEvents, cpu_copy, tensor_bytes,
    check_probe_parity, summarize_trajectories, termination_status)
from .pair_only_worker import INITIAL_STATE, save_delta
from .runtime import StepLimit
from .validation_truth import expected_identity


def require_control_equal(current, historical, label):
    if cached.json_digest(current) != cached.json_digest(historical):
        raise ValueError('Historical control differs before repair: ' + label)


def worker():
    started = time.monotonic()
    manifest, admission, auth, control = launch.authorize()
    auth_sha = file_digest(launch.AUTHORIZATION)
    if (json.loads(launch.ATTEMPT.read_text()).get('authorization_sha256') != auth_sha
            or os.environ.get('CUDA_VISIBLE_DEVICES') != '0' or not launch.OUTPUT.is_dir()
            or (launch.OUTPUT / 'plan.json').exists()):
        raise ValueError('Fresh reserved repair worker required')
    torch.set_num_threads(2); torch.manual_seed(rules.RECIPE['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; no fallback')
    torch.cuda.manual_seed_all(rules.RECIPE['seed'])
    device = torch.device('cuda:0'); props = torch.cuda.get_device_properties(device)
    environment = dict(device=str(device), gpu=props.name, uuid=str(props.uuid),
        torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
        cudnn_version=torch.backends.cudnn.version(), tf32=False, autocast=False,
        dtype='float32', cudnn_benchmark=False, cudnn_deterministic=True)
    rows = manifest['training_samples']; order = [r['source'] + ':' + r['sample_key'] for r in rows]
    inputs, artifacts, control_inputs = {}, {}, {}

    def historical(name):
        path = launch.CONTROL / name
        expected = control['artifact_sha256'][name]
        value = launch.previous.old.read_bound(path, expected)
        control_inputs[str(path)] = expected
        return value

    require_control_equal(environment, historical('plan.json')['environment'], 'environment')
    control_caches = {r['key']: r for r in historical('cache-manifest.json')}
    event_path = launch.CONTROL / 'events.jsonl'
    if file_digest(event_path) != control['artifact_sha256']['events.jsonl']:
        raise ValueError('Historical update sequence changed')
    control_inputs[str(event_path)] = control['artifact_sha256']['events.jsonl']
    old_attempts = [event for line in event_path.read_bytes().splitlines()
                    if (event := json.loads(line))['event'] == 'batch_complete']
    limit = StepLimit(50)
    counters = Counter(upstream_forwards=0, evaluation_images=0, parity_probe_forwards=0,
                       batch_attempts=0, replay_forwards=0, replay_pair_rows=0)

    def write(name, value):
        write_json(launch.OUTPUT / name, value); artifacts[name] = file_digest(launch.OUTPUT / name)

    def save(name, value): artifacts[name] = save_delta(launch.OUTPUT / name, value)

    def time_guard():
        if time.monotonic() - started >= 555: raise TimeoutError('Stopping before hard 600-second limit')

    write('plan.json', dict(protocol=launch.protocol(), authorization_sha256=auth_sha,
        protocol_sha256=auth['protocol_sha256'], environment=environment, samples=rows,
        admission=admission, implementation_sha256=auth['implementation_sha256']))
    with (launch.OUTPUT / 'events.jsonl').open('x', encoding='utf-8') as stream:
        emit = DurableEvents(stream, lambda: time.monotonic() - started)
        try:
            model, config, payload = load_bundle(launch.INITIAL, expected_sha256=rules.INITIAL_SHA)
            if state_digest(model) != INITIAL_STATE or payload['progress']['optimizer_steps'] != 0:
                raise ValueError('Original initial required')
            for name in ('image_size', 'maximum_atoms', 'atom_match_radius_px'):
                if getattr(config, name) != rules.RECIPE[name]: raise ValueError('Initial config changed')
            for name in ('atom_threshold', 'bond_threshold', 'pair_threshold', 'role_threshold'):
                if payload['provenance'][name] != rules.RECIPE[name]: raise ValueError('Initial threshold changed')
            model.to(device); stage = rules.configure_pair_stage(model, config)
            predictor = FrozenOCRPredictor(payload['provenance'], device)
            frozen_before, ocr_before = cached.frozen_digest(model), state_digest(predictor.ocr)
            collator = CurriculumCollator(AtomBondSceneConfig(atom_queries=config.maximum_atoms,
                bond_queries=config.maximum_bonds), purpose='training')
            caches, samples, pixels, receipts = {}, {}, {}, []
            for row, key in zip(rows, order):
                time_guard()
                root = Path(manifest['sources'][row['source']]); meta_path = root / 'samples' / (row['sample_key'] + '.meta.json')
                meta = launch.previous.old.read_bound(meta_path, row['meta_sha256'])
                sample = load_sample(root, row['sample_key'], purpose='training')
                if sample['graph']['canonical_smiles'] != row['canonical_smiles']: raise ValueError('Row identity changed')
                inputs[str(meta_path)] = row['meta_sha256']
                for suffix in PAYLOADS: inputs[str(root / 'samples' / (row['sample_key'] + suffix))] = meta['sha256'][suffix]
                batch = collator([sample])
                target = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                counters['upstream_forwards'] += 1
                if counters['upstream_forwards'] > 58: raise ValueError('Upstream forward cap')
                with torch.no_grad():
                    forward = structure_forward(model, target['pixel_values'], config, rules.RECIPE['atom_threshold'])
                    cache = cached.make_cache(forward, target, config)
                cache.update(key=key, group=row['fit_domain'])
                receipt = dict(key=key, image_sha256=meta['sha256']['.png'], target_sha256=meta['sha256']['.layers.json'],
                    batch_target_sha256=tensor_tree_digest({k: v for k, v in target.items() if k != 'pixel_values'}),
                    **{k: cache[k] for k in ('candidate_label_sha256', 'feature_sha256', 'cache_sha256', 'coverage')})
                require_control_equal(receipt, control_caches[key], 'fresh cache: ' + key)
                caches[key], samples[key], pixels[key] = cache, sample, batch['pixel_values']
                receipts.append(receipt); emit('cache_complete', key=key, ordinal=len(caches))
                del forward, batch, target
            if len(caches) != 58 or state_digest(model) != INITIAL_STATE: raise ValueError('Cache/initial changed')
            write('cache-manifest.json', receipts)
            probes = [(caches[k], p) for k, p in launch.previous.PROBES]
            bindings = {c['key']: bind_negative_probe(c, p) for c, p in probes}
            write('probe-bindings.json', bindings)
            initial_logits, parity = {}, []

            def evaluate(label):
                records = []; cached.assert_stage(stage)
                with torch.no_grad():
                    for row, key in zip(rows, order):
                        time_guard(); cache, sample = caches[key], samples[key]
                        cached.verify_cache(cache)
                        features, selected = cached.pair_inputs(cache, device)
                        structure = predictor.from_predicted_atoms(model.pair, features, selected)
                        if key in bindings:
                            single = observe(model.pair, cache, bindings[key])
                            parity.append(dict(label=label, key=key, **check_probe_parity(single, structure['output'], bindings[key]['row_id'])))
                            counters['parity_probe_forwards'] += 1
                        trace = {}
                        prediction = predictor.from_structure(structure, model.role, model.bbox,
                            pixels[key].to(device), sample['image'], trace=trace)
                        metric = cached.score_case(cache, structure, prediction, trace, key=key, expected=expected_identity(sample))
                        metric['group'] = row['fit_domain']; records.append(metric)
                        result = dict(metrics=metric, prediction=prediction, trace=trace,
                            presence_logits=structure['output']['presence_logits'].cpu().tolist(),
                            chemical_logits=structure['output']['chemical_logits'].cpu().tolist(), pairs=structure['pairs'].cpu().tolist())
                        name = 'check-' + label + '--' + row['sample_key'] + '.json'
                        if label == 'baseline':
                            require_control_equal(result, historical(name), 'initial prediction: ' + key)
                            initial_logits[key] = structure['output']['presence_logits'].detach().cpu().clone()
                        write(name, result); counters['evaluation_images'] += 1
                if len(records) != 58 or cached.frozen_digest(model) != frozen_before or state_digest(predictor.ocr) != ocr_before:
                    raise ValueError('Evaluation denominator or frozen state changed')
                emit('evaluation_complete', label=label, step=limit.consumed, summary=cached.summarize(records))
                return records

            baseline = evaluate('baseline'); write('baseline.json', dict(records=baseline, summary=cached.summarize(baseline)))
            donors = [r for r in rows if r['fit_domain'] == 'old_full_molecule']
            bank = ReplayBank(caches, initial_logits, donors)
            if len(bank.keys) != 32: raise ValueError('Exact original32 donor pool required')
            write('replay-bank.json', dict(selection='fixed_initial_top32_confirmed_nonedges', donor_keys=bank.keys,
                entries={key: dict(binding=e['binding'], molecule_id=e['molecule_id']) for key, e in bank.entries.items()}))
            optimizer = torch.optim.AdamW(cached.assert_stage(stage), lr=rules.RECIPE['learning_rate'],
                betas=tuple(rules.RECIPE['betas']), eps=rules.RECIPE['epsilon'], weight_decay=0.)
            captured = CapturedReplayUpdates(stage, optimizer, limit, probes, rows, bank=bank, emit=emit)
            updates = []
            for attempt, key in enumerate(order):
                time_guard()
                if limit.consumed >= 50: break
                counters['batch_attempts'] += 1
                result = captured.update(caches[key], attempt=attempt); updates.append(result)
                old = old_attempts[attempt]
                current = dict(key=key, attempt=attempt, **result['update'])
                for field in ('key', 'attempt', 'updated', 'step', 'sampled_row_ids'):
                    if current.get(field) != old.get(field): raise ValueError('Main update cohort/sampler differs from control')
                if result['update']['updated']:
                    counters['replay_forwards'] += 1
                    counters['replay_pair_rows'] += result['update']['replay']['rows']
                    if counters['replay_forwards'] > 50 or counters['replay_pair_rows'] > 1600:
                        raise ValueError('Replay resource cap')
                emit('batch_complete', **current)
            final = evaluate('final'); write('final.json', dict(actual_step=limit.consumed, records=final, summary=cached.summarize(final)))
            decision = (rules.checkpoint_decision(baseline, final, step=50) if limit.consumed == 50
                        else dict(action='incomplete_no_promotion', full_50_step_gate_evaluated=False))
            decision.update(bounded_repair_stopped=True, further_training_authorized=False,
                            deployment_authorized=False)
            write('decision.json', decision); write('probe-parity.json', parity)
            write('trajectories.json', dict(updates=updates, summary=summarize_trajectories(updates),
                note='Own/other refers to MAIN image; replay_image_is_probe/probe_row_in_replay identify extra supervision'))
            snapshots = dict(slots=captured.slots, events={}, probes={}, snapshot_tensor_bytes=0)
            for key, binding in bindings.items():
                time_guard(); name = 'probe-cache--' + key.split(':', 1)[1] + '.pt'
                package = dict(cache=caches[key], binding=binding)
                snapshots['snapshot_tensor_bytes'] += tensor_bytes(package)
                if snapshots['snapshot_tensor_bytes'] > 1024**3: raise ValueError('Snapshot cap')
                save(name, cpu_copy(package)); snapshots['probes'][key] = name
            for step, event in sorted(captured.events.items()):
                time_guard(); name = f'event-{step:06d}.pt'
                snapshots['snapshot_tensor_bytes'] += tensor_bytes(event)
                if snapshots['snapshot_tensor_bytes'] > 1024**3: raise ValueError('Snapshot cap')
                arithmetic = check_adam_transition(event)
                save(name, dict(event=cpu_copy(event), protocol_sha256=auth['protocol_sha256'],
                    initial_sha256=rules.INITIAL_SHA, authorization_sha256=auth_sha, environment=environment,
                    replay_objective=True, deployment_authorized=False))
                snapshots['events'][str(step)] = dict(file=name, arithmetic=arithmetic)
            write('snapshot-manifest.json', snapshots)
            save('final-pair-repair.pt', dict(pair_state=cpu_copy(model.pair.state_dict()), optimizer_steps=limit.consumed,
                initial_sha256=rules.INITIAL_SHA, protocol_sha256=auth['protocol_sha256'], deployment_authorized=False))
            for cache in caches.values(): cached.verify_cache(cache)
            for path, digest in {**inputs, **control_inputs}.items():
                if file_digest(path) != digest: raise ValueError('Bound data/control source changed')
            launch.authorize(); cached.assert_stage(stage)
            if cached.frozen_digest(model) != frozen_before or state_digest(predictor.ocr) != ocr_before:
                raise ValueError('Final frozen state changed')
            emit('repair_complete', optimizer_steps=limit.consumed, gate=decision['action'])
            artifacts['events.jsonl'] = file_digest(launch.OUTPUT / 'events.jsonl')
            write_json(launch.OUTPUT / 'report.json', dict(schema_version='pair-nonedge-replay-result-0.1', complete=True,
                **termination_status(limit.consumed), optimizer_steps=limit.consumed, counters=dict(counters),
                baseline=cached.summarize(baseline), final=cached.summarize(final), decision=decision,
                historical_control_final=control['final'], historical_control_report_sha256=launch.CONTROL_SHA,
                same_initial_predictions_and_frozen_caches=True, same_main_sample_order_and_pair_rows=True,
                input_sha256=inputs, control_input_sha256=control_inputs, artifact_sha256=artifacts,
                implementation_sha256=auth['implementation_sha256'], authorization_sha256=auth_sha,
                protocol_sha256=auth['protocol_sha256'], frozen_state_before=frozen_before,
                frozen_state_after=cached.frozen_digest(model), ocr_state_before=ocr_before,
                ocr_state_after=state_digest(predictor.ocr), pair_state_after=state_digest(model.pair),
                elapsed_worker_seconds=time.monotonic() - started, gpu_used=True, default_changed=False,
                training_performed=limit.consumed > 0, independent_accuracy_measured=False, automatic_retries=0,
                validation_sample_artifacts_read=False, sealed_test_read=False, new_identity_queries=0,
                limitations=['Replay can damage true bonds/types; original all58 protections remain mandatory',
                    'Fixed initial hard-negative pool and unit weight are one hypothesis, not an optimum',
                    'No automatic second coefficient, retry, control arm or deployment']))
        except BaseException as error:
            if not (launch.OUTPUT / 'failure.json').exists():
                write_json(launch.OUTPUT / 'failure.json', dict(error_type=type(error).__name__, error=str(error),
                    optimizer_steps_reserved=limit.consumed, counters=dict(counters), complete=False, automatic_retries=0))
            try: emit('repair_failed', error=str(error), steps_reserved=limit.consumed)
            except BaseException: pass
            raise
