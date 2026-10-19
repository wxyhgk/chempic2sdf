"""Run the existing local MolGrapher unchanged; never read graph truth here.

Use the MolGrapher Conda Python, not ChemVector's training environment. The
native pipeline writes preprocessed PNGs next to inputs, so use independent
staged copies. Each image is called separately; failures are kept, not retried.
"""

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback

from .common import (COMPARISON, ROOT, SNAPSHOT, PROTOCOL_SHA, artifact_hashes,
                     selection, sha256, validate_rows, verify_files, write_json)

LOCAL = Path('/home/zhenghe/GK/Project/MolGrapher/MolGrapher')
CHECKPOINTS = ('data/models/keypoint_detector/kd_model.ckpt',
               'data/models/graph_classifier/gc_no_stereo_model.ckpt')


def state_digest(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str((value.shape, value.dtype)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def run(output, smoke=False):
    output = output.resolve()
    relative = output.relative_to(ROOT)
    if len(relative.parts) != 3 or relative.parts[:2] != ('outputs', 'experiments'):
        raise ValueError('Use a fresh outputs/experiments/<name> directory')
    if output.exists():
        raise FileExistsError('Do not overwrite an existing experiment')
    original = selection()
    order = original['sample_order'][:4] if smoke else original['sample_order']
    inputs = {str(SNAPSHOT / COMPARISON / 'images' / f'{key}.png'):
              original['sample_metadata'][key]['png_sha256'] for key in order}
    verify_files(inputs)
    source_paths = set((LOCAL / 'molgrapher').rglob('*.py'))
    for pattern in ('config*.json', 'vocabularies/*.json', 'ocr_mapping/*.json'):
        source_paths.update((LOCAL / 'data').glob(pattern))
    source_paths.update(Path(__file__).parent.glob('*.py'))
    sources = {str(p): sha256(p) for p in sorted(source_paths)}
    weights = {str(LOCAL / name): sha256(LOCAL / name) for name in CHECKPOINTS}
    ocr_root = Path('/home/zhenghe/.paddleocr/whl')
    ocr_files = {str(p): sha256(p) for p in sorted(ocr_root.rglob('*')) if p.is_file()}
    args = {'clean': False, 'visualize': False, 'visualize_rdkit': False,
            'save_mol_folder': str(output / 'native') + '/', 'save_sdf': True,
            'align_rdkit_output': False, 'node_classifier_variant': 'gc_no_stereo_model',
            'force_cpu': False, 'force_no_multiprocessing': True, 'chunk_size': 1,
            'remove_captions': True, 'preprocess': True}
    protocol = {
        'schema': 'external-ocsr-local-molgrapher-0.1', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'sample_order': order, 'sample_metadata': {k: original['sample_metadata'][k] for k in order},
        'comparison_protocol_sha256': PROTOCOL_SHA, 'inputs': inputs, 'sources': sources,
        'checkpoints': weights, 'ocr_cache_files': ocr_files, 'model_args': args,
        'local_repository': str(LOCAL),
        'git_head_context': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=LOCAL, text=True).strip(),
        'git_status_context': subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=normal'], cwd=LOCAL, text=True),
        'model_identity': 'existing local MolGrapher, with local export additions; not asserted to be pristine upstream',
        'python': sys.version,
        'packages': sorted(({'name': d.metadata['Name'], 'version': d.version}
                            for d in importlib.metadata.distributions() if d.metadata['Name']), key=lambda d: d['name']),
        'seed': 20260914, 'smoke_only': smoke, 'training_performed': False,
        'truth_used_for_prediction': False, 'promotion_authorized': False,
        'execution': 'native predict_batch, one image per call, config nb_workers=0; default FP32',
        'compatibility': 'np.int0 = np.int64 if absent, matching existing local test.py; no package/source edits',
        'failure_policy': 'keep every image, no retries; native C placeholders retained and reported separately',
        'stereo_caveat': 'gc_no_stereo_model does not recognize stereochemistry; core graph exact is primary',
        'geometry_caveat': 'default align_rdkit_output=False; no claim of source-image coordinate recovery',
        'scope': 'reused 50-molecule development set, not official benchmark or independent generalization',
    }
    output.mkdir(parents=True)
    (output / 'images').mkdir()
    (output / 'native').mkdir()
    for key in order:
        shutil.copyfile(SNAPSHOT / COMPARISON / 'images' / f'{key}.png', output / 'images' / f'{key}.png')
    write_json(output / 'protocol.json', protocol)
    rows = []
    started = time.perf_counter()
    with (output / 'producer.log').open('x', buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        import numpy as np
        import torch
        import paddle
        import molgrapher
        if Path(molgrapher.__file__).resolve().parent != LOCAL / 'molgrapher':
            raise ValueError('Imported an unexpected MolGrapher installation')
        if not hasattr(np, 'int0'):
            np.int0 = np.int64
        random.seed(protocol['seed'])
        np.random.seed(protocol['seed'])
        torch.manual_seed(protocol['seed'])
        paddle.seed(protocol['seed'])
        if not torch.cuda.is_available():
            raise RuntimeError('Expected the verified local CUDA environment; do not silently change device')
        torch.ones(4, device='cuda').sum().item()
        from molgrapher.models.molgrapher_model import MolgrapherModel
        model = MolgrapherModel(args)
        model.config_dataset_graph['nb_workers'] = 0
        model.config_dataset_keypoint['nb_workers'] = 0
        write_json(output / 'runtime.json', {
            'args': model.args, 'dataset_graph': model.config_dataset_graph,
            'dataset_keypoint': model.config_dataset_keypoint, 'training_graph': model.config_training_graph,
            'training_keypoint': model.config_training_keypoint,
            'torch_cuda': torch.version.cuda, 'cuda_available': torch.cuda.is_available(),
            'gpu': torch.cuda.get_device_name(0), 'paddle_compiled_with_cuda': paddle.is_compiled_with_cuda(),
            'actual_ocr': 'Paddle GPU requested by native model; installed CPU Paddle falls back to CPU',
        })
        before = state_digest(model.model)
        for index, key in enumerate(order, 1):
            image = output / 'images' / f'{key}.png'
            verify_files({str(image): original['sample_metadata'][key]['png_sha256']})
            row = {'sample_key': key, 'invocation_error': None, 'annotations': []}
            tick = time.perf_counter()
            try:
                row['annotations'] = model.predict_batch([str(image)])
                for annotation in row['annotations']:
                    expected_name = f'{key}_preprocessed.png'
                    if Path(annotation['file-info']['filename']).name != expected_name:
                        raise ValueError('Prediction identity mismatch')
                if len(row['annotations']) > 1:
                    raise ValueError('Unexpected extra predictions')
            except Exception as error:
                traceback.print_exc()
                row['invocation_error'] = {'type': type(error).__name__, 'message': str(error)}
            row['seconds'] = time.perf_counter() - tick
            rows.append(row)
            write_json(output / 'native' / f'{key}.invocation.json', row)
            print(f'PROGRESS {index}/{len(order)} {key} error={row["invocation_error"] is not None}', file=sys.__stdout__, flush=True)
            if index == 4 and all(r['invocation_error'] for r in rows):
                raise RuntimeError('All initial four invocations failed; stop for infrastructure diagnosis')
        after = state_digest(model.model)
    validate_rows(rows, order)
    for files in (inputs, sources, weights, ocr_files):
        verify_files(files)
    verify_files({str(output / 'images' / f'{key}.png'): original['sample_metadata'][key]['png_sha256']
                  for key in order})
    if before != after:
        raise ValueError('Inference mutated model state')
    write_json(output / 'invocations.json', rows)
    receipt = {'sample_count': len(rows), 'completed': True, 'smoke_only': smoke,
               'protocol_sha256': sha256(output / 'protocol.json'), 'models_unchanged': True,
               'model_state_sha256': before, 'truth_used_for_prediction': False,
               'elapsed_seconds': time.perf_counter() - started, 'artifacts': artifact_hashes(output)}
    write_json(output / 'prediction_receipt.json', receipt)
    print(json.dumps({'completed': True, 'samples': len(rows), 'seconds': receipt['elapsed_seconds'],
                      'invocation_errors': sum(row['invocation_error'] is not None for row in rows), 'output': str(output)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--smoke', action='store_true')
    cli = parser.parse_args()
    run(cli.output_dir, cli.smoke)
