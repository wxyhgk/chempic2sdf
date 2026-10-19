"""Run the original CLI main with resident frozen weights, then check cold parity."""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
from functools import wraps
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import traceback
from unittest.mock import patch

from chemvector.experiments.id_graph.roundtrip import sha256, write_json
from chemvector.text_entry_predictions import model_digest
from .protocol import verify

LOADERS = ('load_direct_model', 'load_action_ranking_model', 'load_role',
           'load_bbox', 'load_ctc', 'load_components')


def stage_images(output, protocol):
    """Copy exact selected PNG bytes, preserving alpha for native preprocessing."""
    from PIL import Image

    with ExitStack() as stack:
        archives = {p: stack.enter_context(tarfile.open(p, 'r:')) for p in protocol['shards']}
        for key in protocol['sample_order']:
            entry = protocol['image_index'][key]
            archive = archives[entry['shard']]
            member = archive.getmember(entry['member'])
            if not member.isfile() or member.name != f'{key}.png' or member.size != entry['size']:
                raise ValueError('Selected PNG identity/size changed')
            with archive.extractfile(member) as stream:
                payload = stream.read()
            if hashlib.sha256(payload).hexdigest() != protocol['sample_metadata'][key]['png_sha256']:
                raise ValueError('Selected PNG hash changed')
            with Image.open(io.BytesIO(payload)) as image:
                if image.format != 'PNG' or image.size != (768, 768):
                    raise ValueError('Unexpected original image format/size')
            path = output / 'images' / f'{key}.png'
            with path.open('xb') as stream:
                stream.write(payload)
            yield key, path


class LoaderCache:
    """Memoize loading only; no image, activation, decision or output is cached."""

    def __init__(self):
        self.entries = {}
        self.modules = {}

    def wrap(self, name, loader):
        @wraps(loader)
        def cached(*args, **kwargs):
            key = (name, tuple(map(str, args)), tuple(sorted((k, str(v)) for k, v in kwargs.items())))
            if key not in self.entries:
                value = loader(*args, **kwargs)
                self.entries[key] = value
                # The six existing loaders return a tuple of modules and metadata.
                for item in value:
                    if hasattr(item, 'state_dict') and id(item) not in self.modules:
                        self.modules[id(item)] = (item, model_digest((item,)))
            return self.entries[key]
        return cached

    def verify(self):
        if not self.entries or not self.modules:
            raise ValueError('No models loaded')
        for module, before in self.modules.values():
            if model_digest((module,)) != before:
                raise ValueError('Resident model state changed')
        return {'loader_entries': len(self.entries), 'module_count': len(self.modules),
                'models_unchanged': True,
                'state_digests': [digest for _, digest in self.modules.values()]}


def arguments(profile, image, destination):
    return [*profile['arguments'], '--image', str(image), '--output-dir', str(destination), '--device', 'cuda']


def artifacts(root):
    return {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob('*')) if p.is_file()}


def normalized_metrics(value):
    return {k: v for k, v in value.items() if k != 'inference_seconds'}


def compare_outputs(resident, cold, resident_error, cold_returncode):
    """No tolerance for changed chemistry/decisions; timings alone may differ."""
    completed = resident_error is None
    if completed != (cold_returncode == 0):
        raise ValueError('Cold CLI and resident completion differ')
    checked = []
    if completed:
        left = json.loads((resident / 'metrics.json').read_text())
        right = json.loads((cold / 'metrics.json').read_text())
        if normalized_metrics(left) != normalized_metrics(right):
            raise ValueError('Cold CLI and resident metrics differ')
        checked.append('metrics_except_inference_seconds')
    else:
        stderr = (cold / 'stderr.log').read_text()
        if f"{resident_error['type']}: {resident_error['message']}" not in stderr:
            raise ValueError('Cold CLI and resident error differs')
        checked.append('same_exception')
    for name in ('input-prepared.png', 'prediction.chemscene.json', 'prediction-repaired.sdf'):
        left, right = resident / name, cold / name
        if left.exists() != right.exists() or (left.exists() and sha256(left) != sha256(right)):
            raise ValueError(f'Cold CLI and resident artifact differs: {name}')
        if left.exists():
            checked.append(name)
    return checked


def run_resident(cli, argv, destination):
    if destination.exists():
        raise FileExistsError('Never reuse a prediction directory')
    destination.mkdir(parents=True)
    error = None
    started = time.perf_counter()
    with (destination / 'stdout.log').open('x') as stdout, (destination / 'stderr.log').open('x') as stderr:
        with redirect_stdout(stdout), redirect_stderr(stderr), patch.object(sys, 'argv', ['profile-comparison', *argv]):
            try:
                cli.main()
            except Exception as exc:
                error = {'type': type(exc).__name__, 'message': str(exc)}
                traceback.print_exc()
    result = {'error': error, 'seconds': time.perf_counter() - started}
    write_json(destination / 'invocation.json', result)
    return result


def run_cold(profile, image, destination):
    destination.mkdir(parents=True, exist_ok=False)
    command = [str(Path('.venv-train/bin/python').resolve()), '-m', profile['module'],
               *arguments(profile, image, destination)]
    env = {**os.environ, 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'}
    with (destination / 'stdout.log').open('x') as stdout, (destination / 'stderr.log').open('x') as stderr:
        completed = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=180, env=env, check=False)
    write_json(destination / 'invocation.json', {'command': command, 'returncode': completed.returncode})
    return completed.returncode


def predict(output, protocol):
    import torch
    from rdkit import rdBase
    from chemvector.inference import infer_direct_atom_pair_bond as cli

    verify(protocol)
    torch.set_num_threads(4)
    rdBase.LogToPythonStderr()
    cache = LoaderCache()
    (output / 'images').mkdir()
    rows, parity_targets, seen_variants = [], [], set()
    first_action = None
    with ExitStack() as stack:
        for name in LOADERS:
            stack.enter_context(patch.object(cli, name, cache.wrap(name, getattr(cli, name))))
        for index, (key, image) in enumerate(stage_images(output, protocol)):
            variant = protocol['sample_metadata'][key]['variant']
            if variant not in seen_variants:
                parity_targets.extend((key, arm) for arm in protocol['profiles'])
                seen_variants.add(variant)
            # Alternate arm order to expose shared-state/order dependence.
            arms = list(protocol['profiles'])
            if index % 2:
                arms.reverse()
            for arm in arms:
                destination = output / 'predictions' / arm / key
                invocation = run_resident(cli, arguments(protocol['profiles'][arm], image, destination), destination)
                row = {'sample_key': key, 'arm': arm, **invocation}
                metrics_path = destination / 'metrics.json'
                if invocation['error'] is None:
                    metrics = json.loads(metrics_path.read_text())
                    suppression = metrics['diagnostics']['structural_twin_suppression']
                    row['suppression'] = suppression
                    if arm == 'v01016' and suppression.get('executed') and first_action is None:
                        first_action = key
                rows.append(row)
            if (index + 1) % 10 == 0:
                print(f'Profiles {index + 1}/200; CLI exceptions={sum(r["error"] is not None for r in rows)}', flush=True)
    state = cache.verify()
    # Free resident GPU modules before cold CLI checks; predictions are already fixed.
    del cache
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    if first_action is not None:
        parity_targets.extend((first_action, arm) for arm in protocol['profiles'])
    lookup = {(r['sample_key'], r['arm']): r for r in rows}
    parity = []
    for key, arm in dict.fromkeys(parity_targets):
        cold = output / 'cold_parity' / arm / key
        code = run_cold(protocol['profiles'][arm], output / 'images' / f'{key}.png', cold)
        checked = compare_outputs(output / 'predictions' / arm / key, cold, lookup[key, arm]['error'], code)
        parity.append({'sample_key': key, 'arm': arm, 'checked': checked, 'passed': True})
        print(f'Cold parity {arm} {key}: passed', flush=True)
    verify(protocol)
    write_json(output / 'invocations.json', rows)
    write_json(output / 'parity.json', parity)
    receipt = {'protocol_sha256': sha256(output / 'protocol.json'), 'sample_count': len(protocol['sample_order']),
               'invocations': len(rows), 'truth_used_for_prediction': False, **state,
               'cold_parity_passed': all(r['passed'] for r in parity), 'cold_parity_count': len(parity),
               'artifacts': artifacts(output)}
    if len(rows) != 400 or len(lookup) != 400:
        raise ValueError('Incomplete paired predictions')
    write_json(output / 'prediction_receipt.json', receipt)
    return receipt
