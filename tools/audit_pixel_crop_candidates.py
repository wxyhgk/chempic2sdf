"""CPU-only fixed58 pixel-crop comparison, isolated from historical inference."""

import argparse
from collections import Counter
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw

from chemvector.components.ocr.crop_proposals import rasterize
from chemvector.components.ocr.pixel_crop_proposals import RULE, propose_from_pixels
from chemvector.evaluation.audits.ocr.crop_coverage import score_crop
from chemvector.training.curriculum import runtime, pair_trial_contract as rules
from chemvector.training.curriculum.interface_diagnostic import write_json
from chemvector.ctc_ocr import bbox_crop
from tools.audit_ocr_crop_coverage import summarize


ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT / 'outputs/audits/chemvector-ocr-crop-coverage-20260922'
PREVIOUS_SHA = '276e5729b686661abcb9bac45fcab9083f2a8dc79f88b2bef3560ad1bb6314cd'
MANIFEST = ROOT / 'outputs/datasets/chemvector-selector-retention-fit-20260919/manifest.json'
OUTPUT = ROOT / 'outputs/audits/chemvector-pixel-crop-candidates-20260922'
TESTS = ('tests/chemvector/test_pixel_crop_proposals.py', 'tests/chemvector/test_ocr_crop_coverage.py',
         'tests/chemvector/test_evaluate_direct_atom_ocr.py')
CODE = ('chemvector/components/ocr/pixel_crop_proposals.py', 'tools/audit_pixel_crop_candidates.py',
        'chemvector/components/ocr/crop_proposals.py', 'chemvector/evaluation/audits/ocr/crop_coverage.py',
        'tools/audit_ocr_crop_coverage.py', 'chemvector/components/ocr/ctc_ocr.py',
        'chemvector/training/curriculum/runtime.py', 'chemvector/training/curriculum/pair_trial_contract.py',
        'chemvector/training/curriculum/interface_diagnostic.py', *TESTS)


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def comparison(cases):
    crops = [c for case in cases for c in case['crops']]
    visible = [c for c in crops if c['status'] == 'matched_visible']
    records = []
    for c in crops:
        if not c['proposal']['changed']:
            continue
        records.append(dict(key=c['key'], candidate=c['candidate'], status=c['status'],
            layout=c['layout'], original_complete=c['original']['complete'],
            proposed_complete=c['proposed']['complete'], broad_complete=c['broad']['complete'],
            original_height=c['original']['minimum_full_glyph_height_after_resize_px'],
            proposed_height=c['proposed']['minimum_full_glyph_height_after_resize_px'],
            broad_height=c['broad']['minimum_full_glyph_height_after_resize_px'],
            added_target_ink=c['proposed']['covered_target_ink_pixels'] - c['original']['covered_target_ink_pixels'],
            added_neighbor_ink=c['proposed']['neighboring_ink_pixels'] - c['original']['neighboring_ink_pixels'],
            added_bond_ink=c['proposed']['bond_ink_pixels'] - c['original']['bond_ink_pixels']))
    return dict(proposal_statuses=dict(Counter(c['proposal']['status'] for c in crops)),
        changed_unmatched_or_invisible=sum(c['proposal']['changed'] and c['status'] != 'matched_visible' for c in crops),
        source_ink_loss_crops=sum(c['proposed']['covered_target_ink_pixels'] < c['original']['covered_target_ink_pixels'] for c in visible),
        source_ink_gain_crops=sum(c['proposed']['covered_target_ink_pixels'] > c['original']['covered_target_ink_pixels'] for c in visible),
        added_neighbor_ink_crops=sum(c['proposed']['neighboring_ink_pixels'] > c['original']['neighboring_ink_pixels'] for c in visible),
        changed_crops_without_target_ink_gain=sum(c['proposal']['changed'] and
            c['proposed']['covered_target_ink_pixels'] == c['original']['covered_target_ink_pixels'] for c in visible),
        compared_with_broad=dict(
            complete_coverage_losses=sum(c['broad']['complete'] and not c['proposed']['complete'] for c in visible),
            larger_glyph_crops=sum(c['proposed']['minimum_full_glyph_height_after_resize_px'] >
                                   c['broad']['minimum_full_glyph_height_after_resize_px'] for c in visible)),
        changed_crops=records)


def gallery(image, crop, path):
    canvas = Image.new('RGB', (690, 340), 'white'); draw = ImageDraw.Draw(canvas)
    b = crop['proposal']['broad']
    region = (max(0, b[0] - 30), max(0, b[1] - 12), min(768, b[2] + 30), min(768, b[3] + 12))
    patch = image.crop(region); overlay = ImageDraw.Draw(patch)
    for name, color in [('broad', 'blue'), ('original', 'red'), ('proposed', 'green')]:
        box = crop['proposal'][name]
        overlay.rectangle((box[0] - region[0], box[1] - region[1], box[2] - region[0] - 1,
                           box[3] - region[1] - 1), outline=color)
    canvas.paste(patch.resize((patch.width * 2, patch.height * 2)), (5, 25))
    draw.text((5, 5), 'Red: old / blue: broad / green: pixel', fill='black')
    for y, name in [(40, 'original'), (135, 'broad'), (230, 'proposed')]:
        draw.text((270, y - 20), name + ' 192x48 (no OCR forward)', fill='black')
        canvas.paste(rasterize(image, crop['proposal'][name]), (270, y))
    canvas.save(path)


def worker():
    import torch
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('CPU-only worker required')
    inputs, artifacts = {}, {}

    def read(path, expected):
        data = path.read_bytes()
        if sha256(data).hexdigest() != expected:
            raise ValueError('Bound input changed: ' + str(path))
        inputs[str(path)] = expected
        return data

    def write(name, value):
        write_json(OUTPUT / name, value); artifacts[name] = digest(OUTPUT / name)

    previous = json.loads(read(PREVIOUS / 'report.json', PREVIOUS_SHA))
    manifest = json.loads(read(MANIFEST, rules.MANIFEST_SHA))
    rows = manifest['training_samples']; rules.make_proposal(rows)
    implementations = {p: digest(ROOT / p) for p in CODE}
    for path, expected in previous['implementation_sha256'].items():
        if digest(ROOT / path) != expected:
            raise ValueError('Historical crop audit implementation changed')
    write('plan.json', dict(schema_version='pixel-crop-candidates-plan-0.1', rule=RULE,
        scope='same_fixed58_train_279_saved_ocr_crops', previous_report_sha256=PREVIOUS_SHA,
        wall_seconds=120, gpu_devices=[], maximum_alternatives_per_crop=1,
        all_proposals_written_before_truth_scoring=True, no_outcome_based_threshold_search=True,
        matching='unchanged_previous_unique_anchor_audit', no_training=True, no_model_or_ocr_forward=True,
        no_default_changes=True, implementation_sha256=implementations))
    test = subprocess.run([sys.executable, '-m', 'pytest', '-q', *TESTS,
        '--junitxml=' + str(OUTPUT / 'synthetic-tests.xml')], cwd=ROOT, timeout=45, check=False)
    suites = ET.parse(OUTPUT / 'synthetic-tests.xml').getroot().findall('testsuite')
    totals = {k: sum(int(s.attrib.get(k, 0)) for s in suites) for k in ('tests', 'failures', 'errors', 'skipped')}
    if test.returncode or not totals['tests'] or any(totals[k] for k in ('failures', 'errors', 'skipped')):
        raise ValueError('Synthetic tests failed')
    artifacts['synthetic-tests.xml'] = digest(OUTPUT / 'synthetic-tests.xml')
    saved = json.loads(read(PREVIOUS / 'proposals.json', previous['artifact_sha256']['proposals.json']))
    if len(saved) != 58:
        raise ValueError('Full58 saved cohort required')
    proposals, image_cache = [], {}
    for row, old in zip(rows, saved, strict=True):
        key = row['source'] + ':' + row['sample_key']
        if old['key'] != key:
            raise ValueError('Cohort order/key mismatch')
        base = Path(manifest['sources'][row['source']]) / 'samples' / row['sample_key']
        path = Path(str(base) + '.png')
        image = Image.open(io.BytesIO(read(path, previous['input_sha256'][str(path)]))); image.load()
        image_cache[key] = image
        crops = []
        for c in old['crops']:
            candidate = propose_from_pixels(image, c['predicted_bbox'])
            if candidate['original'] != c['proposal']['original'] or candidate['broad'] != c['proposal']['proposed']:
                raise ValueError('Historical original/broad rectangle changed')
            crops.append(dict(**{k: v for k, v in c.items() if k != 'proposal'}, proposal=candidate))
        proposals.append(dict(key=key, sample_key=row['sample_key'], crops=crops))
    write('proposals.json', proposals)
    print('All pixel-only proposals frozen before truth-side scoring.', flush=True)

    cases, old_cases, rasters = [], [], 0
    for row, proposal in zip(rows, proposals, strict=True):
        filename = row['sample_key'] + '.json'
        old = json.loads(read(PREVIOUS / filename, previous['artifact_sha256'][filename])); old_cases.append(old)
        base = Path(manifest['sources'][row['source']]) / 'samples' / row['sample_key']

        def payload(suffix):
            path = Path(str(base) + suffix)
            return read(path, previous['input_sha256'][str(path)])

        records = json.loads(payload('.text-records.json')); masks = []
        for suffix in ('.text.png', '.bond.png'):
            mask = Image.open(io.BytesIO(payload(suffix))); mask.load()
            if mask.mode != 'L' or mask.size != (768, 768):
                raise ValueError('Same-render mask changed')
            masks.append(np.asarray(mask) > 0)
        image = image_cache[proposal['key']]; scored = []
        for p, o in zip(proposal['crops'], old['crops'], strict=True):
            if p['candidate'] != o['candidate'] or p['saved_text'] != o['saved_text']:
                raise ValueError('Saved candidate or OCR text changed')
            own = [g for g in records['glyphs'] if g['atom_id'] == o['match']['atom_id']]
            c = dict(**p, key=proposal['key'], **{k: o[k] for k in ('match', 'status', 'layout', 'expected_text')})
            for arm in ('original', 'broad', 'proposed'):
                c[arm] = score_crop(p['proposal'][arm], own, records['glyphs'], o['match']['atom_id'], *masks)
            if c['original'] != o['original'] or c['broad'] != o['proposed']:
                raise ValueError('Historical geometry/ink metrics not exactly reproduced')
            if not np.array_equal(np.asarray(rasterize(image, p['proposal']['original'])),
                                  np.asarray(bbox_crop(image, p['predicted_bbox'], context=4.))):
                raise ValueError('Original OCR raster changed')
            rasters += 1
            a, b, d = (p['proposal'][arm] for arm in ('original', 'proposed', 'broad'))
            if not (a[0] == b[0] == d[0] and a[2] == b[2] == d[2] and d[1] <= b[1] <= a[1] < a[3] <= b[3] <= d[3]):
                raise ValueError('Proposal escaped bounded original/broad envelope')
            scored.append(c)
            if row['sample_key'] == 'curr_cb5b7da426023d14__font_large' and p['candidate'] == 19:
                write('target-crop.json', c)
                gallery(image, c, OUTPUT / 'target-crop-comparison.png')
                artifacts['target-crop-comparison.png'] = digest(OUTPUT / 'target-crop-comparison.png')
        case = dict(**{k: v for k, v in old.items() if k != 'crops'}, crops=scored)
        cases.append(case); write(filename, case)
    if summarize(old_cases) != previous['summary'] or rasters != previous['historical_crop_rasters_exact']:
        raise ValueError('Historical cohort summary changed')
    for path, expected in inputs.items():
        if digest(path) != expected:
            raise ValueError('Input changed during run')
    for path, expected in implementations.items():
        if digest(ROOT / path) != expected:
            raise ValueError('Implementation changed during run')
    write_json(OUTPUT / 'report.json', dict(schema_version='pixel-crop-candidates-result-0.1', complete=True,
        summary=summarize(cases), broad_reference_summary=previous['summary'], comparison=comparison(cases),
        synthetic_tests=totals, historical_original_rasters_exact=rasters,
        historical_original_and_broad_scores_exact=rasters, input_sha256=inputs,
        implementation_sha256=implementations, artifact_sha256=artifacts,
        model_loads=0, model_forwards=0, ocr_forwards=0, optimizer_steps=0, gpu_used=False,
        default_changed=False, deployment_authorized=False, independent_accuracy_measured=False,
        validation_sample_artifacts_read=False, sealed_test_read=False, new_identity_queries=0,
        limitations=['Dark component geometry is not text/atom/bond semantic classification',
            'Source ink containment does not guarantee OCR correctness after resizing',
            'Fully outside disconnected glyphs cannot trigger this boundary-crossing rule',
            'Thresholds fixed for this known training diagnosis, not independent validation',
            'Original crop, recognized text, matching and all denominators remain unchanged']))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--worker', action='store_true')
    if parser.parse_args(argv).worker:
        worker(); return 0
    receipt = runtime.run_budgeted([sys.executable, '-m', 'tools.audit_pixel_crop_candidates', '--worker'],
                                  output=OUTPUT, budget=runtime.ResourceBudget(120))
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
