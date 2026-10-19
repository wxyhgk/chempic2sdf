"""One CPU fixed58 readout-direction audit; never update a real parameter.

Only atom_presence_residual[-1] (128 weights + bias) is hypothetically varied.
All upstream features and the additive atom_pre score are frozen. A failure
rejects this direction, not all selector parameters or model capacity.
"""

import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

from chemvector.training.curriculum import bundle, continuation, pixels, runtime
from chemvector.training.curriculum.gradient_preflight import state_digest
from tools.audit_selector_stable_subset import Inputs, write
from tools import audit_selector_stable_subset as source
from tools import probe_selector_threshold_contract as previous
from tools import selector_threshold_contract as contract
from tools.selector_singleton_rank import SingleNodeContrast
from tools.selector_node_groups import training_rows

ROOT = source.ROOT
OUTPUT = ROOT / 'outputs/audits/chemvector-selector-threshold-readout58-20260921'
PREVIOUS_SHA = 'dc7f9b5f7c23b70a185ac32859fdd6a141eb0bb7f5a079d27e2e722a1189438f'
INITIAL = ROOT / 'outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt'
INITIAL_SHA = '09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1'
INITIAL_STATE = '9d7f66f79329479e05c38f8b91d95118adb9f12c32a44e5bd9d83a458a8b452e'
MANIFEST_SHA = 'f5e3a7163eef5c49118ad78dcf05778f4ad889dc9b6389097ef20896b79e2111'
WALL_SECONDS = 240


@torch.no_grad()
def capture(model, tensor, config):
    """Image-only forward; annotations cannot be supplied to this function."""
    if (any(p.requires_grad for p in model.parameters()) or model.training
            or tensor.shape != (1, 3, 768, 768) or tensor.device.type != 'cpu'):
        raise ValueError('Frozen eval model and one CPU image required')
    features = []
    handle = model.set_head.atom_presence_residual[-1].register_forward_pre_hook(
        lambda module, arguments: features.append(arguments[0].detach().clone()))
    try:
        _, _, _, atoms, _, output, _ = continuation.structure_forward(model, tensor, config, -.25)
    finally:
        handle.remove()
    if len(features) != 1 or features[0].shape != (1, 128, 128):
        raise ValueError('Expected one128-channel final residual readout input')
    return dict(hidden=features[0][0], pre=output['atom_pre_presence_logits'][0].detach(),
        logits=output['atom_presence_logits'][0].detach(),
        slots=atoms['atom_mask'][0].nonzero().flatten(), grid_ids=atoms['atom_indices'][0].detach(),
        geometry=atoms['atom_geometry'][0].detach(), raw_scores=atoms['atom_scores'][0].detach())


def save_snapshot(path, values):
    with path.open('xb') as stream:
        np.savez_compressed(stream, **{k: v.numpy() for k, v in values.items()})
    with np.load(path, allow_pickle=False) as stored:
        if any(not np.array_equal(v.numpy(), stored[k]) for k, v in values.items()):
            raise ValueError('Snapshot roundtrip differs')
    return sha256(path.read_bytes()).hexdigest()


def forbidden_forward(module, args):
    raise RuntimeError('Pair, role, bbox or OCR forward is outside this diagnostic')


def worker():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('CPU only')
    if not OUTPUT.is_dir() or (OUTPUT / 'plan.json').exists():
        raise ValueError('Fresh one-shot output required')
    torch.set_num_threads(2)
    torch.manual_seed(20260921)
    inputs = Inputs()
    accepted = inputs.json(previous.OUTPUT / 'report.json', PREVIOUS_SHA)
    for path, digest in accepted['input_sha256'].items():
        inputs.read(path, digest)
    prior, nodes = previous.load_scope(inputs)
    manifest = inputs.json(source.MANIFEST, MANIFEST_SHA)
    rows = training_rows(manifest)
    if len(rows) != 58 or Counter(r['fit_domain'] for r in rows) != {'witness': 26, 'old_full_molecule': 32}:
        raise ValueError('Original fixed58 only')
    inputs.read(INITIAL, INITIAL_SHA)
    for module in (Path(__file__), Path(contract.__file__), Path(previous.__file__), Path(bundle.__file__),
                   Path(continuation.__file__), Path(pixels.__file__), Path(runtime.__file__),
                   Path(inspect.getfile(state_digest)), ROOT / 'frontend/inference_profiles.json',
                   ROOT / 'tests/chemvector/test_selector_threshold_readout.py',
                   Path(inspect.getfile(continuation.decode_predicted_atom_proposals))):
        inputs.read(module, prior['input_sha256'].get(str(module)))
    write(OUTPUT / 'plan.json', dict(schema_version='selector-threshold-readout-plan-0.1',
        sample_keys=[r['sample_key'] for r in rows], focused_nodes=8, focused_fragments=25,
        parameter_scope='set_head.atom_presence_residual.2.weight/bias ONLY,129 parameters',
        objective='Fixed8 equal-node class-balanced threshold BCE; no old-image loss or weight search',
        direction='Negative Euclidean parameter gradient normalized to unit L2; never applied',
        threshold=-.25, cpu_wall_seconds=WALL_SECONDS, threads=2, maximum_image_forwards=58,
        optimizer_steps=0, gpu_devices=[], automatic_retries=0, training_authorized=False,
        numerical_parity=dict(max_logit_error=1e-3, max_xy_error_px=.01,
            exact_ordered_grid_ids=True, exact_eligibility=True, exact_selected_slots=True),
        conservative_direction_gate='All8 valid up; all25 owned fragments down; no baseline-selected eligible slot down',
        analytic_ray='Also intersect exact affine decision constraints, protecting every baseline-selected eligible slot',
        validation_sample_artifacts_read=False, sealed_test_read=False, default_changed=False))
    model, config, payload = bundle.load_bundle(INITIAL, expected_sha256=INITIAL_SHA)
    if state_digest(model) != INITIAL_STATE or payload['progress']['optimizer_steps'] != 0:
        raise ValueError('Initial model identity changed')
    if config.image_size != 768 or config.maximum_atoms != 128 or config.atom_match_radius_px != 6:
        raise ValueError('Candidate contract differs')
    model.eval().requires_grad_(False)
    for name in ('vision', 'atom', 'bond', 'set_head'):
        inputs.read(Path(inspect.getfile(type(getattr(model, name)))))
    inputs.read(Path(inspect.getfile(type(model.set_head.incidence))))
    handles = [getattr(model, name).register_forward_pre_hook(forbidden_forward) for name in ('pair', 'role', 'bbox')]
    readout = model.set_head.atom_presence_residual[-1]
    if not isinstance(readout, torch.nn.Linear) or readout.weight.shape != (1, 128) or readout.bias.shape != (1,):
        raise ValueError('Final residual readout shape changed')
    readout_hash = save_snapshot(OUTPUT / 'readout-state.npz',
        dict(weight=readout.weight.detach(), bias=readout.bias.detach()))
    snapshots, records, counters = {}, {}, Counter()
    started = time.monotonic()
    with (OUTPUT / 'events.jsonl').open('x') as log:
        for row in rows:
            key = row['sample_key']
            base = Path(manifest['sources'][row['source']]) / 'samples'
            meta = inputs.json(base / (key+'.meta.json'), row['meta_sha256'])
            image_bytes = inputs.read(base / (key+'.png'), meta['sha256']['.png'])
            with Image.open(BytesIO(image_bytes)) as image:
                tensor = pixels.image_tensor(image).unsqueeze(0)
            log.write(json.dumps(dict(event='forward_reserved', sample_key=key))+'\n'); log.flush(); os.fsync(log.fileno())
            snapshot = capture(model, tensor, config)
            counters['image_forwards'] += 1
            path = OUTPUT / (key+'.npz')
            snapshots[path.name] = save_snapshot(path, snapshot)
            # GT/saved review joins occur only after prediction is persisted.
            case_path = previous.PREVIOUS / (key+'.json')
            case = inputs.json(case_path, prior['case_sha256'][case_path.name])
            geo_path = source.CONTROL / ('geometry--'+key+'.json')
            geometry = inputs.json(geo_path, prior['input_sha256'][str(geo_path)])
            score_path = source.CONTROL / ('baseline--'+key+'.json')
            old = inputs.json(score_path, prior['input_sha256'][str(score_path)])
            old_candidates = geometry['candidates']; slots = snapshot['slots'].tolist()
            if slots != [c['slot'] for c in old_candidates] or any(
                    int(snapshot['grid_ids'][c['slot']]) != c['grid_id'] for c in old_candidates):
                raise ValueError('CPU historical slot/grid identity differs: '+key)
            old_z = {int(s): z for s, z in old['metrics']['logits'].items()}
            z = snapshot['logits']
            max_logit_error = max(abs(float(z[s])-old_z[s]) for s in slots)
            max_xy_error = max(abs(float(snapshot['geometry'][c['slot'], i])-c['geometry'][i])*768
                               for c in old_candidates for i in (0, 1))
            old_edges = {(s, n['target_id']) for n in case['nodes'] for s in n['valid_slots']}
            new_edges = {(s, t) for s in slots for t, xy in enumerate(geometry['target_anchors'])
                         if math.dist(snapshot['geometry'][s, :2].tolist(), xy)*768 <= 6}
            if (max_logit_error > 1e-3 or max_xy_error > .01 or new_edges != old_edges
                    or {s for s in slots if float(z[s]) >= -.25} != {s for s in slots if old_z[s] >= -.25}):
                raise ValueError('CPU historical numeric/discrete parity differs: '+key)
            reconstructed = snapshot['pre'].double() + snapshot['hidden'].double() @ readout.weight[0].double() + readout.bias[0].double()
            readout_error = float((reconstructed[slots]-z[slots].double()).abs().max())
            if readout_error > 2e-5:
                raise ValueError('Residual readout affine reconstruction failed')
            records[key] = dict(snapshot=snapshot, case=case, parity=dict(max_logit_error=max_logit_error,
                max_xy_error_px=max_xy_error, readout_reconstruction_error=readout_error,
                ordered_grid_ids_equal=True, eligible_relations_equal=True, selected_slots_equal=True))
            log.write(json.dumps(dict(event='forward_complete', sample_key=key,
                elapsed_seconds=time.monotonic()-started, completed=counters['image_forwards']))+'\n'); log.flush()
            print('captured', counters['image_forwards'], key, flush=True)
    # Concatenate fixed58 logits/features. Only the fixed8/25 receive loss derivatives.
    offsets, zs, hs = {}, [], []
    for key, record in records.items():
        offsets[key] = len(zs)*128
        zs.append(record['snapshot']['logits'])
        hs.append(record['snapshot']['hidden'])
    logits = torch.cat(zs).double().requires_grad_(True)
    hidden = torch.cat(hs)
    contrasts = [SingleNodeContrast(n['sample_key']+':'+str(n['target_id']),
        tuple(offsets[n['sample_key']]+s for s in n['valid_slots']),
        tuple(offsets[n['sample_key']]+s for s in n['stable_fragment_slots'])) for n in nodes]
    objective = contract.threshold_contract(logits, contrasts)
    dz, = torch.autograd.grad(objective['loss'], logits)
    gradient = contract.affine_gradient(hidden, dz.detach())
    norm = float(gradient.norm())
    if not math.isfinite(norm) or norm == 0:
        raise ValueError('Nonzero finite aggregate direction required')
    field = -(hidden.double() @ gradient[:-1]+gradient[-1]) / norm
    focused_records, constraints = [], []
    for node, contrast in zip(nodes, contrasts):
        v = contrast.valid_slots[0]; fs = list(contrast.fragment_slots)
        slopes = field[fs].tolist()
        focused_records.append(dict(sample_key=node['sample_key'], target_id=node['target_id'],
            valid_slot=node['valid_slots'][0], fragment_slots=node['stable_fragment_slots'],
            valid_logit=float(logits[v].detach()), fragment_logits=logits.detach()[fs].tolist(),
            valid_slope=float(field[v]), fragment_slopes=slopes,
            minimum_pair_margin_slope=min(float(field[v])-s for s in slopes)))
        constraints.append(dict(id=contrast.node_id+':valid', logit=float(logits[v].detach()), slope=float(field[v]), selected=True))
        constraints.extend(dict(id=contrast.node_id+':fragment:'+str(s), logit=float(logits[p].detach()), slope=float(field[p]), selected=False)
                           for s, p in zip(node['stable_fragment_slots'], fs))
    focus_ray = contract.feasible_ray(constraints)
    protections, case_hashes = [], {}
    for key, record in records.items():
        original = record['case']; offset = offsets[key]
        eligible = {s for n in original['nodes'] for s in n['valid_slots']}
        selected_eligible = sorted(s for s in eligible if float(logits[offset+s].detach()) >= -.25)
        rows_out = [dict(slot=s, logit=float(logits[offset+s].detach()), slope=float(field[offset+s])) for s in selected_eligible]
        protections.extend(dict(id=key+':protected:'+str(r['slot']), selected=True, **r) for r in rows_out)
        counters['protected_selected_eligible_slots'] += len(rows_out)
        counters['protected_selected_eligible_down'] += sum(r['slope'] < -1e-8 for r in rows_out)
        counters['all_eligible_slots'] += len(eligible)
        counters['alternative_slots'] += sum(len(n['protected_alternative_slots']) for n in original['nodes'])
        result = dict(sample_key=key, parity=record['parity'], slots=record['snapshot']['slots'].tolist(),
            unit_descent_slopes=field[offset:offset+128].tolist(),
            protected_selected_eligible=rows_out, training_authorized=False, parameters_changed=False)
        path = OUTPUT / (key+'.json'); write(path, result); case_hashes[path.name] = sha256(path.read_bytes()).hexdigest()
    joint_ray = contract.feasible_ray(constraints+protections)
    counters['focused_valid_up'] = sum(r['valid_slope'] > 1e-8 for r in focused_records)
    counters['focused_valid_down'] = sum(r['valid_slope'] < -1e-8 for r in focused_records)
    counters['focused_fragments_down'] = sum(s < -1e-8 for r in focused_records for s in r['fragment_slopes'])
    counters['focused_fragments_up'] = sum(s > 1e-8 for r in focused_records for s in r['fragment_slopes'])
    reasons = []
    if counters['focused_valid_up'] != 8: reasons.append('not_all_eight_true_candidates_move_up')
    if counters['focused_fragments_down'] != 25: reasons.append('not_all_owned_fragments_move_down')
    if counters['protected_selected_eligible_down']: reasons.append('baseline_correct_candidate_downward_risk')
    if not joint_ray['feasible']: reasons.append('no_feasible_step_on_this_affine_ray')
    for handle in handles: handle.remove()
    after = state_digest(model)
    if after != INITIAL_STATE or any(p.grad is not None for p in model.parameters()) or torch.cuda.is_initialized():
        raise ValueError('Frozen model/autograd/device state changed')
    for path, digest in list(inputs.hashes.items()): inputs.read(path, digest)
    write(OUTPUT / 'report.json', dict(schema_version='selector-threshold-readout58-0.1',
        input_sha256=inputs.hashes, snapshot_sha256=snapshots, case_sha256=case_hashes,
        readout_state_sha256=readout_hash, counts=counters, focused_nodes=focused_records,
        objective_loss=float(objective['loss'].detach()), parameter_gradient=gradient.tolist(), gradient_l2=norm,
        parameter_count=129, parameter_order='128 final residual weights then scalar bias',
        focused_decision_ray=focus_ray, protected_decision_ray=joint_ray,
        gate=dict(status='NO_GO' if reasons else 'DIRECTION_SUPPORTED', reasons=reasons,
                  training_authorized=False, finite_step_trial_started=False),
        initial_state_before=INITIAL_STATE, initial_state_after=after,
        real_model_parameter_grads_populated=False, optimizer_steps=0, gpu_used=False,
        model_forwards=counters['image_forwards'], model_parameters_updated=False,
        predictions_changed=False, validation_sample_artifacts_read=False, sealed_test_read=False,
        new_labels_released=False, default_changed=False,
        limitations=['Only one fixed8 objective and129-parameter final residual readout direction',
            'Downward slopes are risk, not observed atom/bond loss; no parameter step was applied',
            'Ray feasibility refers to affine atom scores only, not bond, SDF or full-head capacity',
            'Review-only stable fragments remain unapproved training labels',
            'No old-image loss, projection, optimizer variation or hyperparameter search was attempted']))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        worker()
        return 0
    receipt = runtime.run_budgeted([sys.executable, '-m', 'tools.audit_selector_threshold_readout', '--worker'],
        output=OUTPUT, budget=runtime.ResourceBudget(WALL_SECONDS))
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
