"""One CPU-only initial dense snapshot for fixed seven historical training images."""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
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
from tools import selector_dense_observation as observation
from tools import selector_localization_contract as decoder_contract
from tools import selector_node_groups as groups
from tools import selector_node_semantics as semantics

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT / 'outputs/audits/chemvector-selector-radius-reference-20260920/report.json'
PREVIOUS_SHA = 'e01a31a88c0850d213b0cf747a056a86c0fd95e56eb896c26022390d9ec2e835'
MANIFEST = ROOT / 'outputs/datasets/chemvector-selector-retention-fit-20260919/manifest.json'
MANIFEST_SHA = 'f5e3a7163eef5c49118ad78dcf05778f4ad889dc9b6389097ef20896b79e2111'
CONTROL = ROOT / 'outputs/experiments/chemvector-selector-full-loss-control-20260920'
INITIAL = ROOT / 'outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt'
INITIAL_SHA = '09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1'
INITIAL_STATE = '9d7f66f79329479e05c38f8b91d95118adb9f12c32a44e5bd9d83a458a8b452e'
OUTPUT = ROOT / 'outputs/audits/chemvector-selector-dense-observation7-20260920'
CORE_SHAS = {
    'chemvector/training/curriculum/pixels.py': '6eabc5f57b8bfdaa86b5bf7e58488746a8eb13a205d871f4675c770c56ddd533',
    'chemvector/components/atoms/objects/spatial_atom_object.py': 'b90078cb014daa82443fbb51e1fc318bcd4f4f9ae8e2f7f804e75445a0034269',
    'chemvector/components/graph/joint/predicted_incidence.py': 'c726bb0c0b365b639a79c856f28a6b42735400d7af1aae577f42fda75fabb95d',
}


def fixed_scope(report, manifest):
    rows = groups.training_rows(manifest)
    if Counter(r['fit_domain'] for r in rows) != {'witness': 26, 'old_full_molecule': 32}:
        raise ValueError('Source fixed58 training scope changed')
    missing = report['raw_no_eligible_candidate_nodes']
    by_key = {r['sample_key']: r for r in rows}
    focus = defaultdict(list)
    for node in missing:
        key, target = node['sample_key'], node['target_id']
        if (key not in by_key or by_key[key]['fit_domain'] != 'old_full_molecule'
                or node['group'] != 'old_full_molecule' or type(target) is not int or target < 0
                or target in focus[key]):
            raise ValueError('Only unique historically missing old-training nodes permitted')
        focus[key].append(target)
    if len(focus) != 7 or len(missing) != 10:
        raise ValueError('Exactly seven images and ten missing nodes required')
    return [(by_key[k], sorted(targets)) for k, targets in focus.items()]


def preflight(inputs):
    report = inputs.json(PREVIOUS, PREVIOUS_SHA)
    manifest = inputs.json(MANIFEST, MANIFEST_SHA)
    return report, manifest, fixed_scope(report, manifest)


def save_snapshot(path, snapshot):
    with path.open('xb') as stream:
        np.savez_compressed(stream, **{k: v.numpy() for k, v in snapshot.items()})
        stream.flush()
        os.fsync(stream.fileno())
    with np.load(path, allow_pickle=False) as stored:
        restored = {k: torch.from_numpy(stored[k].copy()) for k in stored.files}
    if any(not torch.equal(snapshot[k], restored[k]) for k in snapshot):
        raise ValueError('Dense snapshot persistence roundtrip changed values')
    return restored, sha256(path.read_bytes()).hexdigest()


def forbidden_forward(module, arguments):
    raise RuntimeError('Out-of-scope bond/selector/pair/role/bbox forward')


def worker():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('CPU-only observation required')
    if not OUTPUT.is_dir() or (OUTPUT / 'plan.json').exists():
        raise ValueError('Fresh one-shot budgeted output required')
    torch.set_num_threads(2)
    torch.manual_seed(20260920)
    inputs = Inputs()
    prior, manifest, scope = preflight(inputs)
    bindings = prior['input_sha256']
    for name, expected in CORE_SHAS.items():
        inputs.read(ROOT / name, expected)
    for module in (Path(__file__), Path(observation.__file__), Path(decoder_contract.__file__),
            Path(bundle.__file__), Path(runtime.__file__), Path(groups.__file__), Path(semantics.__file__),
            Path(sys.modules[Inputs.__module__].__file__), Path(sys.modules[state_digest.__module__].__file__),
            ROOT / 'frontend/inference_profiles.json'):
        inputs.read(module, bindings.get(str(module)))
    inputs.read(INITIAL, INITIAL_SHA)
    files = {}
    for row, target_ids in scope:
        key = row['sample_key']
        base = Path(manifest['sources'][row['source']]) / 'samples'
        meta = inputs.json(base / (key + '.meta.json'), row['meta_sha256'])
        image_path, layer_path = base / (key + '.png'), base / (key + '.layers.json')
        inputs.read(image_path, meta['sha256']['.png'])
        inputs.read(layer_path, meta['sha256']['.layers.json'])
        geo_path = CONTROL / ('geometry--' + key + '.json')
        inputs.read(geo_path, bindings[str(geo_path)])
        files[key] = dict(image=image_path, layers=layer_path, geometry=geo_path)
    write(OUTPUT / 'plan.json', dict(schema_version='selector-dense-observation-plan-0.1',
        user_request='继续下一轮', images=7, focused_targets=10,
        scope=[dict(sample_key=r['sample_key'], target_ids=t) for r, t in scope],
        initial_sha256=INITIAL_SHA, initial_state_sha256=INITIAL_STATE,
        input_sha256=inputs.hashes, wall_budget_seconds=120, maximum_forwards=7,
        maximum_forwards_per_image=1, automatic_retries=0, gpu_devices=[],
        model_mode='eval', dtype='float32', cpu_threads=2, autocast=False,
        mkldnn_enabled=torch.backends.mkldnn.enabled, torch_version=torch.__version__,
        restored='self-contained initial bundle; no external/OCR checkpoint',
        forwarded='vision.forward_stroke_resolution + atom only',
        maximum_atoms=128, atom_threshold=.01, image_size=768, match_radius_px=6,
        gt_policy='Only image pixels enter forward; persist dense output before reading target coordinates',
        training_authorized=False, optimizer_steps=0, default_changed=False))
    model, config, payload = bundle.load_bundle(INITIAL, expected_sha256=INITIAL_SHA)
    if (state_digest(model) != INITIAL_STATE or payload['progress']['optimizer_steps'] != 0
            or config.maximum_atoms != 128 or config.atom_threshold != .01
            or config.image_size != 768 or config.atom_match_radius_px != 6):
        raise ValueError('Original initial state/config differs')
    model.eval().requires_grad_(False)
    for name in ('bond', 'set_head', 'pair', 'role', 'bbox'):
        getattr(model, name).register_forward_pre_hook(forbidden_forward)
    for module in (model.vision, model.atom):
        inputs.read(Path(sys.modules[type(module).__module__].__file__))
    results, snapshot_hashes, case_hashes, visited = [], {}, {}, set()
    counts, supported_counts = Counter(), Counter()
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
                raise ValueError('Repeated forward forbidden')
            visited.add(key)
            image_bytes = inputs.read(files[key]['image'], inputs.hashes[str(files[key]['image'])])
            with Image.open(BytesIO(image_bytes)) as image:
                tensor = pixels.image_tensor(image).unsqueeze(0)
            emit('forward_reserved', sample_key=key)
            snapshot = observation.image_only_snapshot(model, tensor, config)
            observation.validate_snapshot(snapshot, config)
            if tuple(snapshot['heatmap_logits'].shape) != (1, 1, 192, 192):
                raise ValueError('Unexpected initial stride4 shape')
            snap_path = OUTPUT / (key + '.npz')
            snapshot, digest = save_snapshot(snap_path, snapshot)
            snapshot_hashes[snap_path.name] = digest
            emit('snapshot_saved', sample_key=key, sha256=digest)
            # Target coordinates and layer annotations are decoded only AFTER persistence.
            geometry = inputs.json(files[key]['geometry'], inputs.hashes[str(files[key]['geometry'])])
            layers = inputs.json(files[key]['layers'], inputs.hashes[str(files[key]['layers'])])
            if (geometry['key'] != row['source'] + ':' + key or layers['sample_key'] != key
                    or layers['molecule_id'] != row['molecule_id']):
                raise ValueError('Source/geometry identity differs')
            semantics.check_geometry(geometry, layers)
            result = observation.explain_targets(snapshot, geometry, targets, config)
            atoms = {a['atom_id']: a for a in layers['atoms']}
            for node in result['targets']:
                atom = atoms[node['target_id']]
                node.update(element=atom['element'], visible_text=atom['visible_text'], text=atom['text'])
                counts[node['gt_cell']['first_excluding_stage']] += 1
                if node['historical_explanation_supported']:
                    supported_counts[node['gt_cell']['first_excluding_stage']] += 1
            result.update(sample_key=key, target_ids=targets, snapshot_sha256=digest,
                original_image_sha256=inputs.hashes[str(files[key]['image'])],
                model_forwards=1, training_performed=False, gpu_used=False)
            path = OUTPUT / (key + '.json')
            write(path, result)
            case_hashes[path.name] = sha256(path.read_bytes()).hexdigest()
            results.append(dict(sample_key=key, parity=result['parity'], target_count=len(targets)))
            emit('case_complete', sample_key=key, completed=len(results),
                 historical_discrete_contract_reproduced=result['parity']['historical_discrete_contract_reproduced'])
    after = state_digest(model)
    if after != INITIAL_STATE or any(p.grad is not None for p in model.parameters()) or torch.cuda.is_initialized():
        raise ValueError('Observation changed state/gradients or initialized CUDA')
    for path, digest in list(inputs.hashes.items()):
        inputs.read(path, digest)
    write(OUTPUT / 'report.json', dict(schema_version='selector-dense-observation-0.1',
        input_sha256=inputs.hashes, snapshot_sha256=snapshot_hashes, case_sha256=case_hashes,
        expected_images=7, completed_images=len(results), model_forwards=len(visited), focused_targets=10,
        cases=results, config=asdict(config), initial_state_before=INITIAL_STATE, initial_state_after=after,
        cpu_gt_cell_stages=counts, reproduced_historical_gt_cell_stages=supported_counts,
        reproduced_images=sum(r['parity']['historical_discrete_contract_reproduced'] for r in results),
        reproduced_targets=sum(r['target_count'] for r in results if r['parity']['historical_discrete_contract_reproduced']),
        training_authorized=False, training_performed=False, optimizer_steps=0, gpu_used=False,
        ocr_loaded=False, bond_selector_pair_forwards=0, thresholds_changed=False,
        validation_sample_artifacts_read=False, sealed_test_read=False, new_identity_queries=0,
        default_changed=False, limitations=[
            'CPU dense observations are not historical GPU logits; numeric parity is explicitly reported',
            'Only reproduced discrete contracts support explaining historical missing nodes',
            'Observed decoder exclusion is not causal training attribution or a demonstrated remedy',
            'No candidate was rescued and no new accuracy or training admission is claimed']))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        worker()
        return 0
    _, _, scope = preflight(Inputs())
    receipt = runtime.run_budgeted([sys.executable, '-m', 'tools.audit_selector_dense_observation', '--worker'],
        output=OUTPUT, budget=runtime.ResourceBudget(120))
    events = [json.loads(line) for line in (OUTPUT / 'events.jsonl').read_text().splitlines()] if (OUTPUT / 'events.jsonl').is_file() else []
    complete = [e['sample_key'] for e in events if e['event'] == 'case_complete']
    write(OUTPUT / 'completion.json', dict(runtime=receipt, completed_sample_keys=complete,
        incomplete_sample_keys=[r['sample_key'] for r, _ in scope if r['sample_key'] not in complete],
        forward_reservations=sum(e['event'] == 'forward_reserved' for e in events), automatic_retries=0))
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
