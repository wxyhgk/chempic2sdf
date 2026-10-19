"""Fixed58 TRAIN-only CPU crop audit; no OCR/model loading or deployment."""

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

from chemvector.components.ocr.crop_proposals import RULE, predicted_box, propose_vertical, rasterize
from chemvector.evaluation.audits.ocr.crop_coverage import score_crop, unique_anchor_matches, vertical_layout
from chemvector.training.curriculum import runtime, pair_trial_contract as rules
from chemvector.training.curriculum.data import validate_records
from chemvector.training.curriculum.interface_diagnostic import write_json
from chemvector.ctc_ocr import bbox_crop


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'outputs/audits/chemvector-ocr-crop-coverage-20260922'
RUN = ROOT / 'outputs/experiments/chemvector-pair-replay-trial-20260922'
RUN_SHA = 'b7a6c66c9fabc2a346edf13a2552f97f9fae3c4656be8ed680ca328838b93aae'
MANIFEST = ROOT / 'outputs/datasets/chemvector-selector-retention-fit-20260919/manifest.json'
TESTS = ('tests/chemvector/test_ocr_crop_coverage.py', 'tests/chemvector/test_evaluate_direct_atom_ocr.py')
CODE = ('chemvector/components/ocr/crop_proposals.py',
        'chemvector/evaluation/audits/ocr/crop_coverage.py', 'tools/audit_ocr_crop_coverage.py',
        'chemvector/components/ocr/ctc_ocr.py', 'chemvector/training/curriculum/data.py',
        'chemvector/training/curriculum/runtime.py', 'chemvector/training/curriculum/pair_trial_contract.py',
        'chemvector/training/curriculum/interface_diagnostic.py',
        'chemvector/evaluation/evaluators/ocr/evaluate_direct_atom_ocr.py', *TESTS)
TARGET = 'curr_cb5b7da426023d14__font_large'


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def summarize(cases):
    crops = [c for case in cases for c in case['crops']]
    visible = [c for c in crops if c['status'] == 'matched_visible']
    groups = {}
    for group in ('all_matched_visible', 'vertical_multiglyph', 'other_multiglyph', 'single_glyph'):
        rows = visible if group == 'all_matched_visible' else [c for c in visible if c['layout'] == group]
        groups[group] = dict(crops=len(rows), changed=sum(c['proposal']['changed'] for c in rows),
            original_complete=sum(c['original']['complete'] for c in rows),
            proposed_complete=sum(c['proposed']['complete'] for c in rows),
            rescued_complete=sum(not c['original']['complete'] and c['proposed']['complete'] for c in rows),
            lost_complete=sum(c['original']['complete'] and not c['proposed']['complete'] for c in rows),
            new_neighbor_crops=sum(bool(set(c['proposed']['neighboring_atom_ids']) -
                                       set(c['original']['neighboring_atom_ids'])) for c in rows),
            increased_bond_ink_crops=sum(c['proposed']['bond_ink_pixels'] > c['original']['bond_ink_pixels'] for c in rows),
            shrunk_glyph_crops=sum(c['proposed']['minimum_full_glyph_height_after_resize_px'] <
                                   c['original']['minimum_full_glyph_height_after_resize_px'] for c in rows),
            target_ink_pixels=sum(c['original']['target_ink_pixels'] for c in rows),
            original_covered_ink_pixels=sum(c['original']['covered_target_ink_pixels'] for c in rows),
            proposed_covered_ink_pixels=sum(c['proposed']['covered_target_ink_pixels'] for c in rows))
    return dict(images=len(cases), predicted_crops=len(crops), statuses=dict(Counter(c['status'] for c in crops)),
        changed_crops=sum(c['proposal']['changed'] for c in crops),
        visible_target_atoms=sum(c['visible_target_atoms'] for c in cases),
        visible_targets_without_unique_ocr_crop=sum(len(c['visible_targets_without_unique_ocr_crop']) for c in cases),
        groups=groups)


def save_target_gallery(image, crop):
    """Original raster only. No new synthetic molecule image or label repair."""
    canvas = Image.new('RGB', (540, 290), 'white'); draw = ImageDraw.Draw(canvas)
    proposal = crop['proposal']
    x0, y0, x1, y1 = proposal['proposed']
    region = (max(0, x0 - 35), max(0, y0 - 12), min(768, x1 + 35), min(768, y1 + 12))
    patch = image.crop(region)
    overlay = ImageDraw.Draw(patch)
    for name, color in [('proposed', 'blue'), ('original', 'red')]:
        b = proposal[name]
        overlay.rectangle((b[0] - region[0], b[1] - region[1], b[2] - region[0] - 1,
                           b[3] - region[1] - 1), outline=color, width=1)
    canvas.paste(patch.resize((patch.width * 2, patch.height * 2)), (5, 25))
    draw.text((5, 5), 'Red: original; blue: proposed', fill='black')
    for y, name in [(40, 'original'), (150, 'proposed')]:
        draw.text((240, y - 20), name + ' 192x48 OCR input', fill='black')
        canvas.paste(rasterize(image, proposal[name]), (240, y))
    draw.text((240, 235), 'Saved OCR: N; no new OCR forward', fill='black')
    canvas.save(OUTPUT / 'target-crop-comparison.png')


def worker():
    import torch
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('CPU-only worker required')
    hashes, artifacts = {}, {}

    def read_bytes(path, expected):
        data = path.read_bytes()
        if sha256(data).hexdigest() != expected:
            raise ValueError('Bound input changed: ' + str(path))
        hashes[str(path)] = expected
        return data

    def read_json(path, expected):
        return json.loads(read_bytes(path, expected))

    def write(name, value):
        write_json(OUTPUT / name, value); artifacts[name] = digest(OUTPUT / name)

    run = read_json(RUN / 'report.json', RUN_SHA)
    manifest = read_json(MANIFEST, rules.MANIFEST_SHA)
    rows = manifest['training_samples']; rules.make_proposal(rows)
    implementations = {p: digest(ROOT / p) for p in CODE}
    write('plan.json', dict(schema_version='ocr-crop-coverage-plan-0.1', rule=RULE,
        scope='fixed58_train_replay50_saved_OCR_boxes', images=58, wall_seconds=120, gpu_devices=[],
        all_proposals_written_before_truth_scoring=True, matching='unique_both_sides_6px_on_saved_q1023_anchor',
        matching_is_diagnostic_not_training_label_reuse=True, q1023_to_original_pixel_factor=767/1023,
        scored_masks='same-render alpha > 0; glyph rectangles disjoint-checked',
        full_coverage='all_same-render_glyph_ink_bounding_rectangles_contained',
        no_training=True, no_model_loads=True, no_ocr_forward=True, no_default_changes=True,
        no_outcome_based_threshold_adjustment=True, implementation_sha256=implementations))
    test = subprocess.run([sys.executable, '-m', 'pytest', '-q', *TESTS,
        '--junitxml=' + str(OUTPUT / 'synthetic-tests.xml')], cwd=ROOT, timeout=45, check=False)
    suites = ET.parse(OUTPUT / 'synthetic-tests.xml').getroot().findall('testsuite')
    totals = {k: sum(int(s.attrib.get(k, 0)) for s in suites) for k in ('tests', 'failures', 'errors', 'skipped')}
    if test.returncode or not totals['tests'] or any(totals[k] for k in ('failures', 'errors', 'skipped')):
        raise ValueError('Synthetic tests failed')
    artifacts['synthetic-tests.xml'] = digest(OUTPUT / 'synthetic-tests.xml')

    # Phase 1: prediction only. No glyphs, labels or masks enter proposals.
    proposals = []
    for row in rows:
        name = 'check-final--' + row['sample_key'] + '.json'
        case = read_json(RUN / name, run['artifact_sha256'][name]); trace = case['trace']
        key = row['source'] + ':' + row['sample_key']
        if case['metrics']['key'] != key:
            raise ValueError('Saved cohort key mismatch')
        candidates = trace['compiled_candidate_indices']; atoms = trace['scene_before_fallback']['atom_objects']
        if len(set(candidates)) != len(candidates) or len(candidates) != len(atoms):
            raise ValueError('Invalid compiled candidate mapping')
        anchors = {c: [v * 767 / 1023 for v in a['anchor']] for c, a in zip(candidates, atoms, strict=True)}
        crops = []
        for p in trace['ocr']:
            box = predicted_box(p['center'], p['half_extent'])
            crops.append(dict(candidate=p['candidate'], saved_text=p['text'], anchor_px=anchors.get(p['candidate']),
                              predicted_bbox=list(box), proposal=propose_vertical(box, (768, 768))))
        if {c['candidate'] for c in crops} != {r['candidate'] for r in trace['roles'] if r['text']}:
            raise ValueError('Saved text candidates changed')
        proposals.append(dict(key=key, sample_key=row['sample_key'], fit_domain=row['fit_domain'], crops=crops))
    write('proposals.json', proposals)
    print('Fixed all prediction-only crop proposals; starting truth-side scoring.', flush=True)

    # Phase 2: fixed TRAIN rows only. No dev entries from the manifest are opened.
    cases, raster_checks = [], 0
    for row, proposed in zip(rows, proposals, strict=True):
        base = Path(manifest['sources'][row['source']]) / 'samples' / row['sample_key']
        meta_path = Path(str(base) + '.meta.json')
        if run['input_sha256'][str(meta_path)] != row['meta_sha256']:
            raise ValueError('Trial metadata binding mismatch')
        meta = read_json(meta_path, row['meta_sha256'])
        if meta['split'] != 'train' or not meta['training_authorized']:
            raise ValueError('Only admitted training samples allowed')

        def payload(suffix):
            return read_bytes(Path(str(base) + suffix), meta['sha256'][suffix])

        graph, layers, records = (json.loads(payload(s)) for s in ('.graph.json', '.layers.json', '.text-records.json'))
        validate_records(graph, layers, records)
        if (layers['text_records_sha256'] != meta['sha256']['.text-records.json']
                or records['source_png_sha256'] != meta['sha256']['.png']
                or layers['source_png_sha256'] != meta['sha256']['.png']):
            raise ValueError('Same-source text binding mismatch')
        image = Image.open(io.BytesIO(payload('.png'))); image.load()
        if image.size != (768, 768) or image.mode != 'RGB':
            raise ValueError('Original 768 RGB pixels required')
        masks = []
        for suffix in ('.text.png', '.bond.png'):
            mask_image = Image.open(io.BytesIO(payload(suffix))); mask_image.load()
            if mask_image.size != image.size or mask_image.mode != 'L':
                raise ValueError('Same-size grayscale alpha mask required')
            masks.append(np.asarray(mask_image) > 0)
        # Independent glyph pixel attribution is valid only with disjoint boxes.
        occupancy = np.zeros((768, 768), dtype=bool)
        for g in records['glyphs']:
            x0, y0, x1, y1 = g['bbox']
            if any(type(v) is not int for v in g['bbox']) or not (0 <= x0 < x1 <= 768 and 0 <= y0 < y1 <= 768):
                raise ValueError('Integer in-image glyph bbox required')
            if occupancy[y0:y1, x0:x1].any():
                raise ValueError('Overlapping glyph rectangles require separate attribution')
            occupancy[y0:y1, x0:x1] = True
        if (masks[0] & ~occupancy).any():
            raise ValueError('Text alpha outside attributed glyph boxes')
        matches = unique_anchor_matches(proposed['crops'], layers['atoms'])
        atom_index = {a['atom_id']: a for a in layers['atoms']}
        matched_visible = set(); scored = []
        for crop in proposed['crops']:
            old = rasterize(image, crop['proposal']['original'])
            if not np.array_equal(np.asarray(old), np.asarray(bbox_crop(image, crop['predicted_bbox'], context=4.))):
                raise ValueError('Historical OCR input raster not exactly reproduced')
            raster_checks += 1
            match = matches[crop['candidate']]; atom_id = match['atom_id']
            own = [g for g in records['glyphs'] if g['atom_id'] == atom_id]
            status = match['status']
            if status == 'matched':
                status = 'matched_visible' if own else 'matched_invisible'
            if own:
                matched_visible.add(atom_id)
            layout = ('vertical_multiglyph' if vertical_layout(own) else 'other_multiglyph') if len(own) > 1 else 'single_glyph' if own else None
            result = dict(**crop, match=match, status=status, layout=layout,
                expected_text=atom_index[atom_id]['text'] if atom_id is not None else None)
            for arm in ('original', 'proposed'):
                result[arm] = score_crop(crop['proposal'][arm], own, records['glyphs'], atom_id, *masks)
            if (result['proposed']['covered_target_ink_pixels'] < result['original']['covered_target_ink_pixels']):
                raise ValueError('Expansion unexpectedly lost source ink')
            scored.append(result)
            if row['sample_key'] == TARGET and crop['candidate'] == 19:
                save_target_gallery(image, result)
                artifacts['target-crop-comparison.png'] = digest(OUTPUT / 'target-crop-comparison.png')
                write('target-crop.json', dict(key=proposed['key'], **result))
        visible_ids = {a['atom_id'] for a in layers['atoms'] if a['visible_text']}
        result = dict(key=proposed['key'], fit_domain=row['fit_domain'], crops=scored,
            visible_target_atoms=len(visible_ids),
            visible_targets_without_unique_ocr_crop=sorted(visible_ids - matched_visible))
        cases.append(result); write(row['sample_key'] + '.json', result)
    for path, expected in hashes.items():
        if digest(path) != expected:
            raise ValueError('Input changed during audit')
    for path, expected in implementations.items():
        if digest(ROOT / path) != expected:
            raise ValueError('Code changed during audit')
    write_json(OUTPUT / 'report.json', dict(schema_version='ocr-crop-coverage-result-0.1', complete=True,
        summary=summarize(cases), synthetic_tests=totals, historical_crop_rasters_exact=raster_checks,
        proposals_sha256=artifacts['proposals.json'], input_sha256=hashes,
        implementation_sha256=implementations, artifact_sha256=artifacts,
        model_loads=0, model_forwards=0, ocr_forwards=0, optimizer_steps=0, gpu_used=False,
        default_changed=False, deployment_authorized=False, independent_accuracy_measured=False,
        validation_sample_artifacts_read=False, sealed_test_read=False, new_identity_queries=0,
        limitations=['Coverage is not OCR accuracy; saved recognized text is unchanged',
            'Expansion can add bond/neighbor ink and reduce glyph height on the 48px canvas',
            'Diagnostic unique matching is not the historical bond-label assignment',
            'Rule motivated by a known training failure, not independent validation',
            'All unmatched and ambiguous candidates and uncovered targets remain reported']))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--worker', action='store_true')
    if parser.parse_args(argv).worker:
        worker(); return 0
    receipt = runtime.run_budgeted([sys.executable, '-m', 'tools.audit_ocr_crop_coverage', '--worker'],
                                  output=OUTPUT, budget=runtime.ResourceBudget(120))
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
