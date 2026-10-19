"""One CPU upstream replay and fixed OCR crop controls on two training cases.

Diagnosis only. No optimizer, threshold sweep, production edit, or graph repair.
Source glyph boxes enter only the explicitly labeled OCR oracle controls.
"""

import argparse
from hashlib import sha256
import io
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from chemvector.ctc_ocr import bbox_crop, ctc_text_log_score, decode_ctc_with_score
from chemvector.evaluate_direct_atom_ocr import crop_tensor, load_ctc
from chemvector.training.curriculum import bundle, runtime
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.gradient_preflight import state_digest
from chemvector.training.curriculum.pixels import image_tensor
from tools.pair_error_inventory import adapted_anchor
from tools.preview_pair_false_edges import ROOT, TRIAL, read

OUTPUT = ROOT / 'outputs/audits/chemvector-pair-upstream-cause2-20260921'
INITIAL = ROOT / 'outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt'
INITIAL_SHA = '09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1'
INITIAL_STATE = '9d7f66f79329479e05c38f8b91d95118adb9f12c32a44e5bd9d83a458a8b452e'
BRIDGE = 'curr_4ef506af64e8f0b8__bond_short'
NH = 'curr_08d02946e879406f__font_large'


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def predicted_box(row):
    # Match the original FP32 multiplication before PIL's integer rounding.
    x, y = (torch.tensor(row['center'], dtype=torch.float32) * 768).tolist()
    w, h = (torch.tensor(row['half_extent'], dtype=torch.float32) * 768).tolist()
    if min(w, h) <= 0 or any(not math.isfinite(v) for v in (x, y, w, h)):
        raise ValueError('Finite positive predicted text box required')
    return [x - w, y - h, x + w, y + h]


def crop_geometry(box, target_box, context):
    actual = [round(box[0] - context), round(box[1] - context),
              round(box[2] + context), round(box[3] + context)]
    if min(actual[2] - actual[0], actual[3] - actual[1]) <= 0:
        raise ValueError('Empty crop')
    x0, y0, x1, y1 = target_box
    return dict(requested_box=box, context=context, actual_integer_crop=actual,
        target_glyph_bbox=target_box,
        covers_entire_target_bbox=actual[0] <= x0 and actual[1] <= y0 and actual[2] >= x1 and actual[3] >= y1)


def as_tensor(image):
    return (torch.from_numpy(np.asarray(image, dtype=np.float32) / 255).permute(2, 0, 1) - .5) / .5


def selected_parity(raw, selected, saved_trace):
    slots = selected['atom_mask'][0].nonzero().flatten().tolist()
    anchors = {s: [max(0, min(1023, round(v * 1023))) for v in raw['atom_geometry'][0, s, :2].tolist()]
               for s in slots}
    historical = {s: a['anchor'] for s, a in zip(saved_trace['compiled_candidate_indices'],
                                               saved_trace['scene_before_fallback']['atom_objects'])}
    return dict(selected_slots_equal=sorted(anchors) == sorted(historical),
                selected_q1023_anchors_equal=anchors == historical,
                historical_float_cache_recovered=False,
                note='CPU replay of the same pinned initial; parity is selected IDs/quantized anchors, not bitwise GPU features')


def worker():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized() or not OUTPUT.is_dir():
        raise ValueError('Fresh CPU-only budgeted diagnostic required')
    torch.set_num_threads(2)
    torch.manual_seed(20260921)
    hashes = {}
    def data(path, digest=None):
        return json.loads(read(path, hashes, digest))
    manifest = data(ROOT / 'outputs/datasets/chemvector-selector-retention-fit-20260919/manifest.json',
        'f5e3a7163eef5c49118ad78dcf05778f4ad889dc9b6389097ef20896b79e2111')
    post = data(TRIAL / 'postcheck.json', 'b09b82c98fe5280de0d79a6cca858a2854398fa66dca1d29205997f4cece1f54')
    trial = data(TRIAL / 'report.json', 'd12d93e5e012f7566f4cc3123abf411b67f49c44ecb898404034105b3c4603eb')
    samples = {}
    for key in (NH, BRIDGE):
        row = next(r for r in manifest['training_samples'] if r['sample_key'] == key)
        if row['split'] != 'train':
            raise ValueError('Fixed training cases only')
        folder = Path(manifest['sources'][row['source']]) / 'samples'
        meta = data(folder / (key + '.meta.json'), row['meta_sha256'])
        image = Image.open(io.BytesIO(read(folder / (key + '.png'), hashes, meta['sha256']['.png']))).convert('RGB')
        if image.size != (768, 768):
            raise ValueError('Original 768px source required')
        record = data(TRIAL / (name := f'check-000000--{key}.json'), post['artifact_sha256'][name])
        receipt = data(TRIAL / (name := f'cache--{key}.json'), post['artifact_sha256'][name])
        samples[key] = dict(image=image, folder=folder, meta=meta, record=record, receipt=receipt)
    for relative in ('chemvector/training/curriculum/bundle.py', 'chemvector/training/curriculum/continuation.py',
                     'chemvector/training/curriculum/pixels.py', 'chemvector/components/graph/joint/predicted_incidence.py',
                     'chemvector/components/graph/proposals/proposal_object_set.py'):
        read(ROOT / relative, hashes, samples[NH]['receipt']['implementation_sha256'][relative])
    for path in (Path(__file__), ROOT / 'tests/chemvector/test_pair_upstream_cause.py',
                 ROOT / 'tools/pair_error_inventory.py', ROOT / 'tools/preview_pair_false_edges.py',
                 Path(runtime.__file__), ROOT / 'chemvector/components/ocr/ctc_ocr.py',
                 ROOT / 'chemvector/evaluation/evaluators/ocr/evaluate_direct_atom_ocr.py'):
        read(path, hashes)
    write(OUTPUT / 'plan.json', dict(user_request='去查一下', sample_keys=[NH, BRIDGE],
        wall_seconds=120, cpu_threads=2, gpu_devices=[], maximum_upstream_forwards=1,
        maximum_ocr_forwards=2, maximum_ocr_crops=17,
        upstream='One original NH image; original initial, raw decoder and selector unchanged',
        ocr='13 historical bridge crops; T39 and T38 source glyph boxes at fixed contexts 4 and 0',
        controls_are_oracle=True, crop_search=False, training=False, optimizer_steps=0,
        candidate_injection=False, production_changed=False))
    read(INITIAL, hashes, INITIAL_SHA)
    model, config, payload = bundle.load_bundle(INITIAL, expected_sha256=INITIAL_SHA)
    model.eval().requires_grad_(False)
    if state_digest(model) != INITIAL_STATE:
        raise ValueError('Wrong frozen model state')
    if payload['provenance']['atom_threshold'] != -.25 or config.maximum_atoms != 128:
        raise ValueError('Unexpected original selector settings')
    # Nothing from target layers is read before this original-image forward.
    with torch.no_grad():
        forward = structure_forward(model, image_tensor(samples[NH]['image'])[None], config, -.25)
    raw, set_output, selected = forward[3], forward[5], forward[6]
    proposals = [dict(slot=s, grid_id=int(raw['atom_indices'][0, s]),
        geometry=raw['atom_geometry'][0, s].tolist(), proposal_score=float(raw['atom_scores'][0, s]),
        raw_active=bool(raw['atom_mask'][0, s]), selector_logit=float(set_output['atom_presence_logits'][0, s]),
        selector_pre_logit=float(set_output['atom_pre_presence_logits'][0, s]),
        selector_residual=float(set_output['atom_presence_logits'][0, s] - set_output['atom_pre_presence_logits'][0, s]),
        selected=bool(selected['atom_mask'][0, s])) for s in range(config.maximum_atoms)]
    parity = selected_parity(raw, selected, samples[NH]['record']['trace'])
    write(OUTPUT / 'nh-prediction-snapshot.json', dict(proposals=proposals, parity=parity))
    if not parity['selected_slots_equal'] or not parity['selected_q1023_anchors_equal']:
        raise ValueError('CPU selected output differs; cannot attribute the saved missing node to this replay')
    def layers(key):
        sample = samples[key]
        return data(sample['folder'] / (key + '.layers.json'), sample['meta']['sha256']['.layers.json'])
    nh_layers, bridge_layers = layers(NH), layers(BRIDGE)
    target = [v / 1023 for v in adapted_anchor(nh_layers['atoms'][5]['center'])]
    nearby = []
    for p in proposals:
        distance = math.dist(p['geometry'][:2], target) * 768
        if p['raw_active'] and distance <= 12:
            nearby.append(dict(**p, target_distance_px=distance, within_original_6px=distance <= 6))
    upstream = dict(target_id=5, source_center=nh_layers['atoms'][5]['center'], target_normalized=target,
        raw_active=sum(p['raw_active'] for p in proposals), selected=sum(p['selected'] for p in proposals),
        original_raw_threshold=config.atom_threshold, original_selector_threshold=-.25,
        nearby_raw_candidates=nearby, parity=parity,
        conclusion=('raw_candidates_present_but_selector_rejected' if any(p['within_original_6px'] for p in nearby)
                    and not any(p['within_original_6px'] and p['selected'] for p in nearby) else 'not_established'))
    if state_digest(model) != INITIAL_STATE or any(p.grad is not None for p in model.parameters()):
        raise ValueError('Unexpected state/gradient change')
    # OCR uses the frozen checkpoint pinned by the same original bundle.
    ocr_ref = payload['provenance']['checkpoints']['ocr_frozen_not_loaded']
    read(ROOT / ocr_ref['path'], hashes, ocr_ref['sha256'])
    ocr, _ = load_ctc(str(ROOT / ocr_ref['path']), torch.device('cpu'))
    if state_digest(ocr) != trial['ocr_state_before']:
        raise ValueError('Wrong frozen OCR state')
    source = samples[BRIDGE]['image']
    saved = samples[BRIDGE]['record']['trace']['ocr']
    if len(saved) != 13:
        raise ValueError('Original bridge OCR crop budget differs')
    base_crops = [crop_tensor(source, predicted_box(row)) for row in saved]
    with torch.no_grad():
        baseline = ocr(torch.stack(base_crops))
    baseline_rows = []
    for row, logits in zip(saved, baseline):
        text, score = decode_ctc_with_score(logits.float())
        baseline_rows.append(dict(candidate=row['candidate'], historical_text=row['text'], replay_text=text,
            historical_score=row['score'], replay_score=score, text_equal=text == row['text']))
    controls, control_images = [], []
    for tid in (39, 38):
        for context in (4, 0):
            box = bridge_layers['atoms'][tid]['bbox']
            controls.append(dict(target_id=tid, expected_text=bridge_layers['atoms'][tid]['text'],
                oracle=True, geometry=crop_geometry(box, box, context)))
            control_images.append(bbox_crop(source, box, context=context))
    with torch.no_grad():
        counterfactual = ocr(torch.stack([as_tensor(im) for im in control_images]))
    for row, logits in zip(controls, counterfactual):
        row['decoded'], row['score'] = decode_ctc_with_score(logits.float())
        row['C_log_score'] = ctc_text_log_score(logits.float(), 'C', normalize_by_bytes=True)
        row['F_log_score'] = ctc_text_log_score(logits.float(), 'F', normalize_by_bytes=True)
    index = next(i for i, row in enumerate(saved) if row['candidate'] == 38)
    actual_box = predicted_box(saved[index])
    target_box = bridge_layers['atoms'][39]['bbox']
    focus = dict(candidate=38, target_id=39, actual_crop=crop_geometry(actual_box, target_box, 4),
        unpadded_crop=crop_geometry(actual_box, target_box, 0),
        baseline=baseline_rows[index], C_log_score=ctc_text_log_score(baseline[index].float(), 'C', normalize_by_bytes=True),
        F_log_score=ctc_text_log_score(baseline[index].float(), 'F', normalize_by_bytes=True), controls=controls,
        baseline_all_text_equal=all(r['text_equal'] for r in baseline_rows))
    # Evidence illustration, not newly synthesized molecular training images.
    figure = Image.new('RGB', (1000, 850), 'white')
    draw = ImageDraw.Draw(figure)
    font = ImageFont.load_default(size=17)
    entries = [('T39 actual crop +4: ' + focus['baseline']['replay_text'], bbox_crop(source, actual_box, context=4))]
    entries += [(f'T{r["target_id"]} source glyph box +{r["geometry"]["context"]}: {r["decoded"]}', im)
                for r, im in zip(controls, control_images)]
    for i, (label, im) in enumerate(entries):
        y = 10 + i * 160
        draw.text((10, y), label, fill='black', font=font)
        figure.paste(im.resize((576, 144), Image.Resampling.NEAREST), (410, y))
    draw.text((10, 815), 'Oracle controls change crop/scale; no change was applied to inference or training.', fill='black', font=font)
    figure_path = OUTPUT / 'ocr-crop-controls.png'
    figure.save(figure_path)
    if state_digest(ocr) != trial['ocr_state_before'] or any(p.grad is not None for p in ocr.parameters()) or torch.cuda.is_initialized():
        raise ValueError('Frozen CPU-only OCR invariant failed')
    for path, digest in list(hashes.items()):
        read(path, hashes, digest)
    write(OUTPUT / 'report.json', dict(schema_version='pair-upstream-cause2-0.1', upstream=upstream, ocr=focus,
        ocr_baseline_rows=baseline_rows, input_sha256=hashes,
        artifact_sha256={p.name: sha256(p.read_bytes()).hexdigest() for p in
                         (figure_path, OUTPUT / 'nh-prediction-snapshot.json')},
        upstream_forwards=1, ocr_forwards=2, ocr_crops=17, cpu_only=True, gpu_used=False,
        optimizer_steps=0, training_performed=False, weights_unchanged=True, predictions_modified=False,
        validation_samples_read=False, sealed_test_read=False, new_identity_queries=0,
        limitations=['CPU parity is discrete selected IDs/anchors; not historical GPU cache equality',
            'No causal training-gradient attribution from this forward-only diagnostic',
            'Source glyph boxes are oracle controls, not a deployable bbox policy',
            'Tighter crops change surrounding strokes and glyph scale together',
            'Two training failures do not establish prevalence or generalization']))
    print(json.dumps(dict(upstream=upstream, ocr=focus), indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args(argv)
    if args.worker:
        worker()
        return 0
    receipt = runtime.run_budgeted([sys.executable, '-m', 'tools.diagnose_pair_upstream2', '--worker'],
        output=OUTPUT, budget=runtime.ResourceBudget(120))
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
