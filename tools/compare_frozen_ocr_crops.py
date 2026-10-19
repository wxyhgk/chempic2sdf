"""One bounded CPU inference comparison for the frozen 279+13 crop inputs."""

import argparse
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import torch

from chemvector.components.ocr.crop_proposals import rasterize
from chemvector.evaluation.audits.ocr.frozen_crop_comparison import image_tensor, infer_frozen, score_pair, summarize_text
from chemvector.evaluate_direct_atom_ocr import load_ctc, crop_tensor
from chemvector.training.curriculum import runtime, pair_trial_contract as rules
from chemvector.training.curriculum.gradient_preflight import state_digest
from chemvector.training.curriculum.interface_diagnostic import write_json


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'outputs/audits/chemvector-pixel-crop-candidates-20260922'
SOURCE_SHA = 'ae879ebcdc102e7ade968a12463743a8985e867a2f38a479b0304f53ba4962b0'
TRIAL = ROOT / 'outputs/experiments/chemvector-pair-replay-trial-20260922/report.json'
TRIAL_SHA = 'b7a6c66c9fabc2a346edf13a2552f97f9fae3c4656be8ed680ca328838b93aae'
MANIFEST = ROOT / 'outputs/datasets/chemvector-selector-retention-fit-20260919/manifest.json'
REGISTRY = ROOT / 'frontend/inference_profiles.json'
REGISTRY_SHA = '6017181ceba8ae631e684a57bc5378365aba232746d14ef71f714bc89a9026b2'
CHECKPOINT = ROOT / 'outputs/checkpoints/chemvector-v0109-correct-label-ocr-style10k/selected.pt'
CHECKPOINT_SHA = 'dff869139c8e63451794ce8083932c06344d407774f652dc3baa18729abc8540'
OUTPUT = ROOT / 'outputs/audits/chemvector-frozen-ocr-crops-20260922'
TESTS = ('tests/chemvector/test_frozen_crop_comparison.py', 'tests/chemvector/test_pixel_crop_proposals.py',
         'tests/chemvector/test_ocr_crop_coverage.py', 'tests/chemvector/test_evaluate_direct_atom_ocr.py')
CODE = ('tools/compare_frozen_ocr_crops.py', 'chemvector/evaluation/audits/ocr/frozen_crop_comparison.py',
        'chemvector/components/ocr/crop_proposals.py', 'chemvector/components/ocr/ctc_ocr.py',
        'chemvector/core/ocr.py', 'chemvector/evaluation/evaluators/ocr/evaluate_direct_atom_ocr.py',
        'chemvector/training/curriculum/gradient_preflight.py', 'chemvector/training/curriculum/runtime.py',
        'chemvector/training/curriculum/interface_diagnostic.py', 'chemvector/training/curriculum/pair_trial_contract.py', *TESTS)


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def worker():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('CPU-only comparison required')
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.manual_seed(20260922); torch.use_deterministic_algorithms(True)
    inputs, artifacts = {}, {}

    def read(path, expected):
        data = path.read_bytes()
        if sha256(data).hexdigest() != expected:
            raise ValueError('Bound input changed: ' + str(path))
        inputs[str(path)] = expected
        return data

    def write(name, value):
        write_json(OUTPUT / name, value); artifacts[name] = digest(OUTPUT / name)

    source = json.loads(read(SOURCE / 'report.json', SOURCE_SHA))
    trial = json.loads(read(TRIAL, TRIAL_SHA))
    manifest = json.loads(read(MANIFEST, rules.MANIFEST_SHA)); rows = manifest['training_samples']
    rules.make_proposal(rows)
    registry = json.loads(read(REGISTRY, REGISTRY_SHA))
    profile = registry.get('profiles', registry)['structure-v01010-direct-aromatic-ocr']; args = profile['arguments']
    if (ROOT / args[args.index('--ocr-checkpoint') + 1]).resolve() != CHECKPOINT.resolve() or profile['ocr_checkpoint_sha256'] != CHECKPOINT_SHA:
        raise ValueError('Registered OCR source changed')
    read(CHECKPOINT, CHECKPOINT_SHA)  # Verify trusted local artifact before loader.
    implementations = {p: digest(ROOT / p) for p in CODE}
    for path, expected in source['implementation_sha256'].items():
        if digest(ROOT / path) != expected:
            raise ValueError('Frozen candidate implementation changed')
    write('plan.json', dict(schema_version='frozen-ocr-crop-plan-0.1', scope='fixed58_train_saved279_plus13_candidates',
        device='cpu', dtype='float32', autocast=False, threads=2, wall_seconds=300,
        maximum_ocr_loads=1, maximum_input_crops=292, maximum_forward_calls=37, batch_size=8,
        seed=20260922, deterministic_algorithms=True, automatic_retries=0, gpu_devices=[],
        original_inputs=279, candidate_inputs=13, unchanged_candidates_reuse_original=True,
        model_state_must_match_historical=trial['ocr_state_before'], checkpoint_sha256=CHECKPOINT_SHA,
        results_written_before_truth_scoring=True, selection='none; unconditional_predeclared_candidate_arm',
        score='raw_text_and_parsed_display_state; fixed_previous_geometric_matching',
        protection='every_CPU_original_correct_raw_text_and_parsed_display_state',
        historical_GPU_parity='report_separately; never_credit_device_drift_as_crop_gain',
        no_training=True, no_graph_repair=True, no_default_changes=True, implementation_sha256=implementations))
    test = subprocess.run([sys.executable, '-m', 'pytest', '-q', *TESTS,
        '--junitxml=' + str(OUTPUT / 'synthetic-tests.xml')], cwd=ROOT, timeout=45, check=False)
    suites = ET.parse(OUTPUT / 'synthetic-tests.xml').getroot().findall('testsuite')
    totals = {k: sum(int(s.attrib.get(k, 0)) for s in suites) for k in ('tests', 'failures', 'errors', 'skipped')}
    if test.returncode or not totals['tests'] or any(totals[k] for k in ('failures', 'errors', 'skipped')):
        raise ValueError('Synthetic tests failed')
    artifacts['synthetic-tests.xml'] = digest(OUTPUT / 'synthetic-tests.xml')
    proposals = json.loads(read(SOURCE / 'proposals.json', source['artifact_sha256']['proposals.json']))
    if len(proposals) != 58:
        raise ValueError('Complete58 proposal cohort required')
    request_index, crop_index, tensors = [], {}, []
    for row, item in zip(rows, proposals, strict=True):
        key = row['source'] + ':' + row['sample_key']
        if key != item['key']:
            raise ValueError('Fixed cohort key mismatch')
        path = Path(manifest['sources'][row['source']]) / 'samples' / (row['sample_key'] + '.png')
        image = Image.open(io.BytesIO(read(path, source['input_sha256'][str(path)]))); image.load()
        if image.mode != 'RGB' or image.size != (768, 768):
            raise ValueError('Original RGB pixels required')
        for p in item['crops']:
            identity = (key, p['candidate'])
            if identity in crop_index:
                raise ValueError('Duplicate crop')
            indices = {}
            for arm in ('original', 'proposed'):
                if arm == 'proposed' and not p['proposal']['changed']:
                    indices[arm] = indices['original']; continue
                canvas = rasterize(image, p['proposal'][arm]); pixels = image_tensor(canvas)
                if arm == 'original' and not torch.equal(pixels, crop_tensor(image, p['predicted_bbox'])):
                    raise ValueError('Original runtime tensor differs')
                indices[arm] = len(tensors); tensors.append(pixels)
                request_index.append(dict(index=indices[arm], key=key, candidate=p['candidate'], arm=arm,
                    bbox=p['proposal'][arm], raster_sha256=sha256(np.asarray(canvas).tobytes()).hexdigest(),
                    tensor_sha256=sha256(pixels.contiguous().numpy().tobytes()).hexdigest()))
            crop_index[identity] = indices
    if len(crop_index) != 279 or len(tensors) != 292:
        raise ValueError('Exact279 original +13 alternate inputs required')
    write('input-index.json', request_index)
    print('Fixed 292 CPU input tensors. Loading only the hash-bound OCR.', flush=True)
    model, payload = load_ctc(str(CHECKPOINT), torch.device('cpu'))
    before = state_digest(model)
    if before != trial['ocr_state_before'] or before != trial['ocr_state_after']:
        raise ValueError('OCR parameters differ from historical trial')
    write('model-binding.json', dict(checkpoint_sha256=CHECKPOINT_SHA, state_sha256=before,
        config=payload['config'], parameters=sum(p.numel() for p in model.parameters()),
        device='cpu', trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))
    del payload

    def progress(calls, done):
        if calls % 5 == 0 or done == 292:
            print(json.dumps(dict(ocr_forward_calls=calls, completed_inputs=done)), flush=True)

    predictions, logits, calls = infer_frozen(model, torch.stack(tensors), progress=progress)
    if calls > 37 or state_digest(model) != before or any(p.grad is not None for p in model.parameters()):
        raise ValueError('Frozen model/budget invariant failed')
    write('ocr-results.json', [dict(**index, **pred) for index, pred in zip(request_index, predictions, strict=True)])
    np.save(OUTPUT / 'logits.npy', logits.numpy(), allow_pickle=False)
    artifacts['logits.npy'] = digest(OUTPUT / 'logits.npy')
    print('All OCR results saved; starting truth-side paired scoring.', flush=True)

    scored, cases = [], []
    for row in rows:
        name = row['sample_key'] + '.json'
        previous = json.loads(read(SOURCE / name, source['artifact_sha256'][name])); case_rows = []
        for p in previous['crops']:
            ids = crop_index[(previous['key'], p['candidate'])]
            record = dict(key=previous['key'], candidate=p['candidate'], status=p['status'], match=p['match'],
                layout=p['layout'], expected_text=p['expected_text'], saved_text=p['saved_text'],
                crop_changed=p['proposal']['changed'],
                coverage_rescued=p['status'] == 'matched_visible' and not p['original']['complete'] and p['proposed']['complete'],
                original=predictions[ids['original']], proposed=predictions[ids['proposed']],
                input_indices=ids, proposal=p['proposal'])
            if record['status'] == 'matched_visible':
                record['score'] = score_pair(record['expected_text'], record['original']['text'], record['proposed']['text'])
            else:
                record['score'] = None
            case_rows.append(record); scored.append(record)
        case = dict(key=previous['key'], crops=case_rows, visible_target_atoms=previous['visible_target_atoms'],
                    visible_targets_without_unique_ocr_crop=previous['visible_targets_without_unique_ocr_crop'])
        cases.append(case); write(name, case)
    visible_targets = sum(c['visible_target_atoms'] for c in cases)
    summary = summarize_text(scored, visible_targets=visible_targets)
    if (summary['statuses'] != source['summary']['statuses'] or visible_targets != 276
            or summary['visible_targets_without_unique_crop'] != 15):
        raise ValueError('Scoring denominators changed')
    changes = [r for r in scored if r['crop_changed']]
    write('changed-crops.json', changes)
    for path, expected in inputs.items():
        if digest(path) != expected:
            raise ValueError('Input changed during comparison')
    for path, expected in implementations.items():
        if digest(ROOT / path) != expected:
            raise ValueError('Implementation changed during comparison')
    write_json(OUTPUT / 'report.json', dict(schema_version='frozen-ocr-crop-result-0.1', complete=True,
        summary=summary, synthetic_tests=totals, images=58, original_inputs=279, proposed_inputs=13,
        original_tensor_parity=279, unchanged_inputs_reused=266, model_loads=1, ocr_input_crops=292,
        ocr_forward_calls=calls, state_before_sha256=before, state_after_sha256=state_digest(model),
        input_sha256=inputs, implementation_sha256=implementations, artifact_sha256=artifacts,
        optimizer_steps=0, training_performed=False, gpu_used=False, default_changed=False,
        graph_recompiled=False, deployment_authorized=False, independent_accuracy_measured=False,
        validation_sample_artifacts_read=False, sealed_test_read=False, new_identity_queries=0,
        limitations=['CPU paired OCR test on known training crops, not independent end-to-end accuracy',
            'No oracle/confidence routing between original and proposed recognized text',
            'Matched display-state parsing is not full atom chemistry or graph identity',
            'Unmatched and invisible candidates retained without pretending their target labels are known',
            'Historical GPU baseline parity is separate from same-device crop comparison']))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--worker', action='store_true')
    if parser.parse_args(argv).worker:
        worker(); return 0
    receipt = runtime.run_budgeted([sys.executable, '-m', 'tools.compare_frozen_ocr_crops', '--worker'],
                                  output=OUTPUT, budget=runtime.ResourceBudget(300))
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
