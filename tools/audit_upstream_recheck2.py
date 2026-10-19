"""Shadow-test prediction-only candidate requests and OCR stability, fixed two cases."""

import argparse
from collections import Counter
from hashlib import sha256
import io
import json
import math
import os
from pathlib import Path
import sys

from PIL import Image
import torch

from chemvector.ctc_ocr import bbox_crop, decode_ctc_with_score
from chemvector.evaluate_direct_atom_ocr import crop_tensor, load_ctc
from chemvector.experiments import ocr_consistency, selector_recheck
from chemvector.training.curriculum import runtime
from chemvector.training.curriculum.gradient_preflight import state_digest
from tools.diagnose_pair_upstream2 import as_tensor, predicted_box, BRIDGE, NH
from tools.pair_error_inventory import adapted_anchor
from tools.pair_local_repair import edge_map
from tools.preview_pair_false_edges import ROOT, TRIAL, read

PRIOR = ROOT / 'outputs/audits/chemvector-pair-upstream-cause2-20260921'
INVENTORY = ROOT / 'outputs/audits/chemvector-pair-error-inventory2-20260921'
OUTPUT = ROOT / 'outputs/audits/chemvector-upstream-recheck2-20260921'
OCR = ROOT / 'outputs/checkpoints/chemvector-v0109-correct-label-ocr-style10k/selected.pt'


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def worker():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized() or not OUTPUT.is_dir():
        raise ValueError('Fresh CPU-only worker required')
    torch.set_num_threads(2)
    hashes = {}
    def data(path, digest=None):
        return json.loads(read(path, hashes, digest))
    prior = data(PRIOR / 'report.json', '42a1b053f89d8083d21064947c486619fc9de13a2bf7737533ee8c9140a8cb55')
    post = data(TRIAL / 'postcheck.json', 'b09b82c98fe5280de0d79a6cca858a2854398fa66dca1d29205997f4cece1f54')
    manifest = data(ROOT / 'outputs/datasets/chemvector-selector-retention-fit-20260919/manifest.json',
        'f5e3a7163eef5c49118ad78dcf05778f4ad889dc9b6389097ef20896b79e2111')
    snapshot = data(PRIOR / 'nh-prediction-snapshot.json', prior['artifact_sha256']['nh-prediction-snapshot.json'])
    for path in (Path(__file__), Path(selector_recheck.__file__), Path(ocr_consistency.__file__),
                 ROOT / 'tests/chemvector/test_upstream_recheck.py', ROOT / 'tools/diagnose_pair_upstream2.py',
                 ROOT / 'tools/pair_error_inventory.py', ROOT / 'tools/pair_local_repair.py',
                 ROOT / 'tools/preview_pair_false_edges.py', Path(runtime.__file__),
                 ROOT / 'chemvector/components/ocr/ctc_ocr.py',
                 ROOT / 'chemvector/evaluation/evaluators/ocr/evaluate_direct_atom_ocr.py'):
        read(path, hashes, prior['input_sha256'].get(str(path)))
    write(OUTPUT / 'plan.json', dict(sample_keys=[NH, BRIDGE], wall_seconds=120, cpu_threads=2,
        raw_candidate_scope='NH saved CPU snapshot only; no structure model forward',
        selector_config=selector_recheck.CONFIG, crop_variants=ocr_consistency.VARIANTS,
        maximum_ocr_crops=162, maximum_ocr_forwards=6, maximum_batch=32,
        oracle_inputs_to_helpers=False, policy='Queue only; original selected atoms, edges and OCR text unchanged',
        training=False, graph_edits=0, default_changed=False))
    samples = {}
    for key in (NH, BRIDGE):
        row = next(r for r in manifest['training_samples'] if r['sample_key'] == key)
        if row['split'] != 'train':
            raise ValueError('Fixed training rows only')
        folder = Path(manifest['sources'][row['source']]) / 'samples'
        meta = data(folder / (key + '.meta.json'), row['meta_sha256'])
        image = Image.open(io.BytesIO(read(folder / (key + '.png'), hashes, meta['sha256']['.png']))).convert('RGB')
        if image.size != (768, 768):
            raise ValueError('Original image geometry required')
        record = data(TRIAL / (name := f'check-000000--{key}.json'), post['artifact_sha256'][name])
        samples[key] = dict(image=image, meta=meta, folder=folder, record=record)
    raw = snapshot['proposals']
    predictions = [dict(slot=r['slot'], xy=[v * 768 for v in r['geometry'][:2]],
        keep_logit=r['selector_logit'], selected=r['selected']) for r in raw if r['raw_active']]
    trace = samples[NH]['record']['trace']
    if sorted(p['slot'] for p in predictions if p['selected']) != sorted(trace['compiled_candidate_indices']):
        raise ValueError('Saved raw/selected snapshot lineage differs')
    edges = edge_map(trace['scene_before_fallback'], trace['compiled_candidate_indices'])
    request = selector_recheck.review_requests(predictions, list(edges))
    write(OUTPUT / 'candidate-review-prediction-only.json', request)

    read(OCR, hashes, prior['input_sha256'][str(OCR)])
    model, _ = load_ctc(str(OCR), torch.device('cpu'))
    initial_state = state_digest(model)
    if initial_state != '2882fc3cd8b24e6fb1124202f3dcf9fd7ff802f5db9cdc1020e26d0e4edb8d9f':
        raise ValueError('Frozen OCR state differs')
    jobs, candidates = [], []
    for key, sample in samples.items():
        for saved in sample['record']['trace']['ocr']:
            box = predicted_box(saved)
            variants = ocr_consistency.crop_variants(box, sample['image'].size)
            item = dict(sample_key=key, candidate=saved['candidate'], historical_text=saved['text'], variants=[])
            for variant in variants:
                result = dict(**variant, text=None, score=None)
                item['variants'].append(result)
                if not variant['available']:
                    continue
                tensor = as_tensor(bbox_crop(sample['image'], variant['box'], context=variant['context']))
                if variant['variant'] == 'base' and not torch.equal(tensor, crop_tensor(sample['image'], box)):
                    raise ValueError('Base crop preprocessing changed')
                jobs.append((tensor, result))
            candidates.append(item)
    if len(candidates) != 18 or len(jobs) > 162:
        raise ValueError('Unexpected OCR scope/budget')
    forwards = 0
    with torch.no_grad():
        for start in range(0, len(jobs), 32):
            forwards += 1
            if forwards > 6:
                raise ValueError('OCR forward cap exceeded')
            batch = jobs[start:start+32]
            output = model(torch.stack([tensor for tensor, _ in batch]))
            for logits, (_, result) in zip(output, batch):
                result['text'], result['score'] = decode_ctc_with_score(logits.float())
    for item in candidates:
        item['decision'] = ocr_consistency.summarize(item['variants'])
        if item['decision']['original_text'] != item['historical_text']:
            raise ValueError('Historical OCR base transcription failed replay')
    write(OUTPUT / 'ocr-review-prediction-only.json', candidates)
    # Labels enter only after both prediction-only outputs have been persisted.
    inventory_report = data(INVENTORY / 'report.json', '3ed71d673499efa1d3b0674789d6fe09d132e685d7f6adc385f1ee09ef6952af')
    cases = {key: data(INVENTORY / (key + '.json'), inventory_report['case_sha256'][key + '.json']) for key in samples}
    sample = samples[NH]
    layers = data(sample['folder'] / (NH + '.layers.json'), sample['meta']['sha256']['.layers.json'])
    target_xy = [[v * 768 / 1023 for v in adapted_anchor(a['center'])] for a in layers['atoms']]
    proposal_xy = {p['slot']: p['xy'] for p in predictions}
    request_scores = []
    for group in request['requests']:
        evaluations = []
        for p in group['candidates']:
            distances = [math.dist(proposal_xy[p['slot']], t) for t in target_xy]
            tid = min(range(len(distances)), key=distances.__getitem__)
            evaluations.append(dict(slot=p['slot'], nearest_target=tid, distance_px=distances[tid],
                                    within_original_radius=distances[tid] <= 6))
        request_scores.append(dict(orphan_slot=group['orphan_slot'], partner_slot=group['partner_slot'],
            first_review_slot=group['first_review_slot'], candidates=evaluations))
    statistics = Counter()
    for item in candidates:
        node = next(n for n in cases[item['sample_key']]['nodes'] if n['candidate_slot'] == item['candidate'])
        # Literal HN/NH layout is not counted as chemical recognition failure.
        correct = not node['stable_chemical_mismatches'] and node['text_status'] in (
            'exact_literal', 'hydrogen_layout_equivalent_not_element_error')
        item['scoring_only'] = dict(target_id=node['target_id'], expected_text=node['expected_text'],
                                   baseline_chemically_correct=correct)
        statistics['review_' + ('correct' if correct else 'wrong') if item['decision']['needs_review']
                   else 'stable_' + ('correct' if correct else 'wrong')] += 1
    if state_digest(model) != initial_state or any(p.grad is not None for p in model.parameters()) or torch.cuda.is_initialized():
        raise ValueError('Frozen CPU-only invariant failed')
    for path, digest in list(hashes.items()):
        read(path, hashes, digest)
    write(OUTPUT / 'report.json', dict(schema_version='upstream-recheck2-0.1', selector=request,
        selector_scoring_only=request_scores, ocr=candidates, ocr_counts=dict(statistics),
        input_sha256=hashes, artifact_sha256={name: sha256((OUTPUT/name).read_bytes()).hexdigest() for name in
            ('candidate-review-prediction-only.json', 'ocr-review-prediction-only.json')},
        ocr_forwards=forwards, ocr_crops=len(jobs), structure_forwards=0, gpu_used=False,
        training_performed=False, optimizer_steps=0, weights_unchanged=True, original_outputs_changed=False,
        oracle_used_for_generation_or_selection=False, validation_samples_read=False, sealed_test_read=False,
        limitations=['Only one NH raw snapshot and 18 existing OCR candidates from two training failures',
            'Geometry shortlist is not atom identity or evidence that two connecting bonds exist',
            'No pair features/logits for rejected candidates; requests are not complete graph repairs',
            'OCR disagreement flags sensitivity; stable agreement can still be wrong',
            'The missed text-role C never reached OCR and remains outside this crop check',
            'No acceptance, new accuracy claim, crop voting, or production wiring']))
    print(json.dumps(dict(selector=request, selector_scoring_only=request_scores, ocr_counts=statistics,
        reviewed=[dict(sample_key=r['sample_key'], candidate=r['candidate'], decision=r['decision'])
                  for r in candidates if r['decision']['needs_review']]), indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args(argv)
    if args.worker:
        worker()
        return 0
    receipt = runtime.run_budgeted([sys.executable, '-m', 'tools.audit_upstream_recheck2', '--worker'],
        output=OUTPUT, budget=runtime.ResourceBudget(120))
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
