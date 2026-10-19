"""Fixed7, one-forward-per-image CPU heatmap readout direction diagnostic."""

import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

from chemvector.training.curriculum import bundle, pixels, runtime
from chemvector.training.curriculum.gradient_preflight import state_digest
from tools.audit_selector_stable_subset import Inputs, write
from tools import audit_selector_dense_observation as prior
from tools import selector_dense_observation as observation
from tools import selector_heatmap_direction as direction
from tools.selector_node_semantics import check_geometry

ROOT = prior.ROOT
OUTPUT = ROOT / 'outputs/audits/chemvector-heatmap-readout-direction7-20260920'
DENSE_SHA = '7ba69b5f75241fced837611a9dbdef236a1855fdbbd290f01ff50154f78ec1ff'
ANCHOR = ROOT / 'outputs/audits/chemvector-selector-anchor-contract7-20260920/report.json'
ANCHOR_SHA = '7029a3fafc813f36343155da12641cdaba8dd5e7fae049538f4b975a33ce174f'


def worker():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('CPU-only diagnostic required')
    if not OUTPUT.is_dir() or (OUTPUT / 'plan.json').exists():
        raise ValueError('Fresh one-shot budgeted output required')
    torch.set_num_threads(2)
    torch.manual_seed(20260920)
    inputs = Inputs()
    _, manifest, scope = prior.preflight(inputs)
    dense = inputs.json(prior.OUTPUT / 'report.json', DENSE_SHA)
    inputs.json(ANCHOR, ANCHOR_SHA)
    for module in (Path(__file__), Path(direction.__file__), Path(prior.__file__),
            Path(observation.__file__), Path(bundle.__file__), Path(pixels.__file__), Path(runtime.__file__),
            Path(sys.modules[state_digest.__module__].__file__), Path(sys.modules[Inputs.__module__].__file__),
            Path(sys.modules[check_geometry.__module__].__file__),
            ROOT / 'chemvector/components/vision/vertex.py', ROOT / 'frontend/inference_profiles.json',
            ROOT / 'docs/model/training/20260920-heatmap-readout-direction-plan.md'):
        inputs.read(module, dense['input_sha256'].get(str(module)))
    for name, digest in prior.CORE_SHAS.items():
        inputs.read(ROOT / name, digest)
    inputs.read(prior.INITIAL, prior.INITIAL_SHA)
    write(OUTPUT / 'plan.json', dict(schema_version='heatmap-readout-direction-plan-0.1',
        user_request='开始按照顺序推进', phase=1,
        scope=[dict(sample_key=r['sample_key'], target_ids=t) for r, t in scope],
        model_forwards_maximum=7, per_image_forwards_maximum=1, wall_budget_seconds=120,
        gpu_devices=[], optimizer_steps=0, automatic_retries=0,
        objective='mean of seven per-image original focal heatmap losses, all 226 atoms; no text losses',
        differentiated_parameters=['atom.heatmap.weight', 'atom.heatmap.bias'], parameter_count=129,
        method='project saved-logit derivative onto captured frozen hidden features; no parameter mutation',
        gate=dict(minimum_improving_missing_margins=6, focused_targets=10,
                  summed_missing_margin_derivative_must_be_positive=True, numerical_zero_tolerance=1e-8),
        phase2='conditional user-approved <=600s/400steps; this worker cannot train or start it',
        default_changed=False, thresholds_changed=False, validation_sample_artifacts_read=False))
    model, config, payload = bundle.load_bundle(prior.INITIAL, expected_sha256=prior.INITIAL_SHA)
    if (state_digest(model) != prior.INITIAL_STATE or payload['progress']['optimizer_steps'] != 0
            or config.maximum_atoms != 128 or config.atom_threshold != .01
            or config.image_size != 768 or config.atom_match_radius_px != 6):
        raise ValueError('Frozen initial/config mismatch')
    model.eval().requires_grad_(False)
    if (type(model.atom.heatmap) is not torch.nn.Conv2d
            or tuple(model.atom.heatmap.weight.shape) != (1, 128, 1, 1)
            or tuple(model.atom.heatmap.bias.shape) != (1,)
            or model.atom.heatmap.stride != (1, 1) or model.atom.heatmap.padding != (0, 0)
            or model.atom.heatmap.groups != 1):
        raise ValueError('Exactly the original 129-parameter 1x1 heatmap readout required')
    for name in ('bond', 'set_head', 'pair', 'role', 'bbox'):
        getattr(model, name).register_forward_pre_hook(prior.forbidden_forward)
    for module in (model.vision, model.atom):
        inputs.read(Path(sys.modules[type(module).__module__].__file__))
    parameter_snapshot, parameter_sha = prior.save_snapshot(OUTPUT / 'readout-state.npz',
        dict(weight=model.atom.heatmap.weight.detach().clone(), bias=model.atom.heatmap.bias.detach().clone()))
    records, visited, snapshots, cases = [], set(), {}, {}
    started = time.monotonic()
    with (OUTPUT / 'events.jsonl').open('x', encoding='utf-8') as log:
        def emit(event, **values):
            line = json.dumps(dict(event=event, elapsed_seconds=time.monotonic()-started, **values))
            log.write(line + '\n')
            log.flush()
            os.fsync(log.fileno())
            print(line, flush=True)

        for row, targets in scope:
            key = row['sample_key']
            if key in visited:
                raise ValueError('Repeated image forward forbidden')
            visited.add(key)
            base = Path(manifest['sources'][row['source']]) / 'samples'
            meta = inputs.json(base / (key + '.meta.json'), row['meta_sha256'])
            image_bytes = inputs.read(base / (key + '.png'), meta['sha256']['.png'])
            with Image.open(BytesIO(image_bytes)) as image:
                tensor = pixels.image_tensor(image).unsqueeze(0)
            emit('forward_reserved', sample_key=key)
            snapshot, hidden = direction.image_only_features(model, tensor, config)
            observation.validate_snapshot(snapshot, config)
            direction.check_tensor(hidden, (1, 128, 192, 192))
            old_path = prior.OUTPUT / (key + '.npz')
            inputs.read(old_path, dense['snapshot_sha256'][old_path.name])
            with np.load(old_path, allow_pickle=False) as old:
                if any(not torch.equal(v, torch.from_numpy(old[k].copy())) for k, v in snapshot.items()):
                    raise ValueError('New CPU snapshot differs from bound previous CPU snapshot')
            snap_path = OUTPUT / (key + '.npz')
            _, digest = prior.save_snapshot(snap_path, dict(snapshot, object_features=hidden))
            snapshots[snap_path.name] = digest
            emit('snapshot_saved', sample_key=key, sha256=digest)
            # Pixels/features are predicted and persisted before parsing this image's GT.
            geometry_path = prior.CONTROL / ('geometry--' + key + '.json')
            geometry = inputs.json(geometry_path, dense['input_sha256'][str(geometry_path)])
            layers = inputs.json(base / (key + '.layers.json'), meta['sha256']['.layers.json'])
            if (geometry['key'] != row['source'] + ':' + key or layers['sample_key'] != key
                    or layers['molecule_id'] != row['molecule_id']):
                raise ValueError('Fixed sample identity mismatch')
            check_geometry(geometry, layers)
            parity = observation.compare_history(snapshot, geometry, config)
            if (not parity['historical_discrete_contract_reproduced']
                    or sorted(targets) != parity['cpu_missing_target_ids']):
                raise ValueError('Historical missing targets do not reproduce')
            anchors = torch.tensor([geometry['target_anchors']], dtype=torch.float32)
            probe = direction.parameter_gradient(hidden, snapshot['heatmap_logits'], anchors)
            records.append(dict(key=key, snapshot=snapshot, hidden=hidden, geometry=geometry,
                                targets=targets, probe=probe, parity=parity))
            emit('image_gradient_complete', sample_key=key, all_atoms=len(geometry['target_anchors']))
        aggregate = torch.stack([r['probe']['gradient'] for r in records]).mean(0)
        descriptions, counts = [], Counter()
        for row in records:
            result = direction.describe_image(row['hidden'], row['snapshot']['heatmap_logits'],
                row['probe']['heatmap'], row['geometry'], row['targets'], aggregate, row['probe']['gradient'])
            result.update(sample_key=row['key'], target_ids=row['targets'],
                original_snapshot_bitwise_reproduced=True, historical_parity=row['parity'],
                own_image_loss=row['probe']['loss'], own_image_parameter_gradient=row['probe']['gradient'].tolist())
            descriptions.append(result)
            counts.update(images=1, atoms=len(result['atoms']), focused_targets=len(row['targets']),
                other_atoms=len(result['atoms'])-len(row['targets']),
                original_raw_positive_peaks=len(result['original_raw_positive_peaks']))
            for group, atoms in (('focused', [a for a in result['atoms'] if a['focused']]),
                    ('other_gt', [a for a in result['atoms'] if not a['focused']]),
                    ('raw_positive', result['original_raw_positive_peaks'])):
                counts[group + '_margin_up'] += sum(a['descent_margin_derivative'] > 1e-8 for a in atoms)
                counts[group + '_margin_down'] += sum(a['descent_margin_derivative'] < -1e-8 for a in atoms)
                counts[group + '_logit_up'] += sum(a['descent_logit_derivative'] > 1e-8 for a in atoms)
                counts[group + '_logit_down'] += sum(a['descent_logit_derivative'] < -1e-8 for a in atoms)
            path = OUTPUT / (row['key'] + '.json')
            write(path, result)
            cases[path.name] = sha256(path.read_bytes()).hexdigest()
            emit('case_complete', sample_key=row['key'])
    gate = direction.diagnostic_gate(descriptions)
    if counts['atoms'] != 226 or counts['other_atoms'] != 216:
        raise ValueError('All-target population changed')
    after = state_digest(model)
    if (after != prior.INITIAL_STATE or any(p.grad is not None for p in model.parameters())
            or torch.cuda.is_initialized()):
        raise ValueError('Frozen model changed or unexpected device/autograd state')
    for path, digest in list(inputs.hashes.items()):
        inputs.read(path, digest)
    write(OUTPUT / 'report.json', dict(schema_version='heatmap-readout-direction-0.1',
        input_sha256=inputs.hashes, snapshot_sha256=snapshots, case_sha256=cases,
        readout_state_sha256=parameter_sha, parameter_count=129,
        parameter_order='128 weight channels then scalar bias',
        objective='equal image mean of original focal heatmap loss with all targets',
        aggregate_parameter_gradient=aggregate.tolist(),
        aggregate_loss_descent_derivative=-float(aggregate.square().sum()),
        counts=counts, gate=gate, model_forwards=7, optimizer_steps=0, gpu_used=False,
        real_model_parameter_grads_populated=False, parameter_derivatives_computed=True,
        initial_state_before=prior.INITIAL_STATE, initial_state_after=after,
        training_performed=False, phase2_started=False, thresholds_changed=False, default_changed=False,
        validation_sample_artifacts_read=False, sealed_test_read=False, new_identity_queries=0,
        limitations=['Only fixed7 mean-loss direction; not full58, alternative optimizers or all possible directions',
            'Directional margin changes are not finite-step candidate retention/recovery or graph correctness',
            'No text auxiliary loss, offset change, candidate-capacity change, or selector update',
            'NO_GO stops this automatic heatmap-only trial path, not proof that the architecture cannot learn']))
    print(json.dumps(dict(counts=counts, gate=gate), ensure_ascii=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        worker()
        return 0
    prior.preflight(Inputs())
    receipt = runtime.run_budgeted([sys.executable, '-m', 'tools.audit_selector_heatmap_direction', '--worker'],
        output=OUTPUT, budget=runtime.ResourceBudget(120))
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
