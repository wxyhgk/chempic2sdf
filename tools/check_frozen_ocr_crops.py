"""Re-decode saved logits and verify paired scores, without loading OCR again."""

import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

from chemvector.components.ocr.crop_proposals import rasterize
from chemvector.evaluation.audits.ocr.frozen_crop_comparison import score_pair, summarize_text
from chemvector.training.curriculum import runtime
from chemvector.training.curriculum.interface_diagnostic import write_json
from tools.compare_frozen_ocr_crops import digest, MANIFEST, SOURCE, OUTPUT as PARENT


OUTPUT = PARENT / 'postcheck'


def greedy_text(row):
    """Independent byte CTC collapse; confidence is not used to choose an arm."""
    tokens, previous = [], None
    for token in row.argmax(axis=-1).tolist():
        if token != 0 and token != previous:
            tokens.append(token - 1)
        previous = token
    return bytes(tokens).decode('utf-8', errors='replace')


def worker():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU-only saved-output postcheck')
    report = json.loads((PARENT / 'report.json').read_bytes()); verified = 0
    for path, expected in {**report['input_sha256'], **report['implementation_sha256']}.items():
        if digest(path) != expected:
            raise ValueError('Input/code hash mismatch: ' + path)
        verified += 1
    for name, expected in report['artifact_sha256'].items():
        if digest(PARENT / name) != expected:
            raise ValueError('Artifact hash mismatch: ' + name)
        verified += 1
    results = json.loads((PARENT / 'ocr-results.json').read_bytes())
    index = json.loads((PARENT / 'input-index.json').read_bytes())
    logits = np.load(PARENT / 'logits.npy', allow_pickle=False)
    if logits.shape != (292, 48, 257) or logits.dtype != np.float32 or not np.isfinite(logits).all():
        raise ValueError('Saved logits changed')
    if len(index) != len(results) or len(results) != 292:
        raise ValueError('Input/result count mismatch')
    for i, (row, request) in enumerate(zip(results, index, strict=True)):
        if row['index'] != i or any(row[k] != v for k, v in request.items()) or greedy_text(logits[i]) != row['text']:
            raise ValueError('Saved decoding/index mismatch')
    manifest = json.loads(MANIFEST.read_bytes()); scored, changed = [], []
    panels = []; visible = 0; raster_count = 0
    for row in manifest['training_samples']:
        filename = row['sample_key'] + '.json'
        case = json.loads((PARENT / filename).read_bytes()); old = json.loads((SOURCE / filename).read_bytes())
        visible += case['visible_target_atoms']
        if case['visible_target_atoms'] != old['visible_target_atoms'] or case['visible_targets_without_unique_ocr_crop'] != old['visible_targets_without_unique_ocr_crop']:
            raise ValueError('Missing target denominator changed')
        image = Image.open(Path(manifest['sources'][row['source']]) / 'samples' / (row['sample_key'] + '.png')); image.load()
        for c, prior in zip(case['crops'], old['crops'], strict=True):
            for k in ('candidate', 'expected_text', 'saved_text', 'status', 'match', 'layout', 'proposal'):
                if c[k] != prior[k]:
                    raise ValueError('Candidate/truth assignment changed')
            for arm in ('original', 'proposed'):
                request = results[c['input_indices'][arm]]
                canvas = rasterize(image, c['proposal'][arm])
                from hashlib import sha256
                if sha256(np.asarray(canvas).tobytes()).hexdigest() != request['raster_sha256']:
                    raise ValueError('Crop pixels differ from inference input')
                if c[arm] != {k: request[k] for k in ('text', 'score')}:
                    raise ValueError('Scored prediction differs from raw output')
                raster_count += 1
            if c['status'] == 'matched_visible':
                actual = score_pair(c['expected_text'], c['original']['text'], c['proposed']['text'])
                if json.loads(json.dumps(actual)) != c['score']:
                    raise ValueError('Paired score mismatch')
            elif c['score'] is not None:
                raise ValueError('Unknown/invisible crop was scored as visible text')
            scored.append(c)
            if not c['crop_changed']:
                if c['input_indices']['original'] != c['input_indices']['proposed']:
                    raise ValueError('Unchanged input was unnecessarily rerun')
                continue
            changed.append(c)
            panel = Image.new('RGB', (740, 120), 'white'); draw = ImageDraw.Draw(panel)
            draw.text((5, 4), row['sample_key'] + ' candidate ' + str(c['candidate']) + ' ' + c['status'], fill='black')
            for x, arm in [(5, 'original'), (265, 'proposed')]:
                draw.text((x, 28), arm + ': ' + c[arm]['text'], fill='black')
                panel.paste(rasterize(image, c['proposal'][arm]), (x, 48))
            draw.text((530, 55), 'expected: ' + (c['expected_text'] or 'unscored'), fill='black')
            panels.append(panel)
    if summarize_text(scored, visible_targets=visible) != report['summary']:
        raise ValueError('Summary recomputation differs')
    if changed != json.loads((PARENT / 'changed-crops.json').read_bytes()) or len(changed) != 13:
        raise ValueError('Changed crops dropped/reordered')
    gallery = Image.new('RGB', (740, len(panels) * 120), 'white')
    for i, panel in enumerate(panels):
        gallery.paste(panel, (0, i * 120))
    gallery.save(OUTPUT / 'recognized-crops.png')
    write_json(OUTPUT / 'report.json', dict(checks_passed=True, parent_report_sha256=digest(PARENT / 'report.json'),
        verified_parent_hashes=verified, saved_logits_redecoded=292, crop_arm_rasters_rechecked=raster_count,
        paired_records=279, retained_changed_crops=13, full_summary_recomputed=True,
        default_decision=report['summary']['decision'], model_loads=0, model_forwards=0, gpu_used=False,
        gallery_sha256=digest(OUTPUT / 'recognized-crops.png'), checker_sha256=digest(Path(__file__))))


def main():
    if sys.argv[1:] == ['--worker']:
        worker(); return 0
    receipt = runtime.run_budgeted([sys.executable, '-m', 'tools.check_frozen_ocr_crops', '--worker'],
                                  output=OUTPUT, budget=runtime.ResourceBudget(30))
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
