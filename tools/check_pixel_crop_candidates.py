"""Independent bounded postcheck and six-case risk gallery; no model execution."""

import io
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

from chemvector.components.ocr.crop_proposals import rasterize
from chemvector.components.ocr.pixel_crop_proposals import propose_from_pixels
from chemvector.training.curriculum import runtime
from chemvector.training.curriculum.interface_diagnostic import write_json
from tools.audit_pixel_crop_candidates import digest, summarize, comparison


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'outputs/audits/chemvector-pixel-crop-candidates-20260922'
OUTPUT = RUN / 'postcheck'


def worker():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU-only postcheck required')
    report = json.loads((RUN / 'report.json').read_bytes())
    verified = 0
    for path, expected in {**report['input_sha256'], **report['implementation_sha256']}.items():
        if digest(Path(path)) != expected:
            raise ValueError('Bound input/code changed: ' + path)
        verified += 1
    for name, expected in report['artifact_sha256'].items():
        if digest(RUN / name) != expected:
            raise ValueError('Artifact changed: ' + name)
        verified += 1
    manifest_path = ROOT / 'outputs/datasets/chemvector-selector-retention-fit-20260919/manifest.json'
    manifest = json.loads(manifest_path.read_bytes()); rows = manifest['training_samples']
    previous_path = ROOT / 'outputs/audits/chemvector-ocr-crop-coverage-20260922/report.json'
    previous = json.loads(previous_path.read_bytes())
    proposals = json.loads((RUN / 'proposals.json').read_bytes())
    if len(rows) != len(proposals) or len(rows) != 58:
        raise ValueError('Full58 required')
    cases, crops, risk_records, panels, extra_hashes = [], 0, [], [], {}
    for row, proposal in zip(rows, proposals, strict=True):
        if row['split'] != 'train' or proposal['key'] != row['source'] + ':' + row['sample_key']:
            raise ValueError('TRAIN-only fixed order required')
        case = json.loads((RUN / (row['sample_key'] + '.json')).read_bytes()); cases.append(case)
        base = Path(manifest['sources'][row['source']]) / 'samples' / row['sample_key']
        image = Image.open(str(base) + '.png'); image.load()
        for p, c in zip(proposal['crops'], case['crops'], strict=True):
            if propose_from_pixels(image, p['predicted_bbox']) != p['proposal'] or p['proposal'] != c['proposal']:
                raise ValueError('Pixel proposal replay differs')
            if p['candidate'] != c['candidate'] or p['saved_text'] != c['saved_text']:
                raise ValueError('Candidate/text changed')
            crops += 1
            if not p['proposal']['changed']:
                if not np.array_equal(np.asarray(rasterize(image, p['proposal']['original'])),
                                      np.asarray(rasterize(image, p['proposal']['proposed']))):
                    raise ValueError('Unchanged raster differs')
                continue
            if c['proposed']['covered_target_ink_pixels'] > c['original']['covered_target_ink_pixels']:
                continue
            risk = dict(key=case['key'], candidate=c['candidate'], status=c['status'],
                        saved_text=c['saved_text'], expected_text=c['expected_text'],
                        original_match_preserved=c['match'],
                        added_bond_ink=c['proposed']['bond_ink_pixels'] - c['original']['bond_ink_pixels'])
            if c['status'] == 'unmatched':
                path = Path(str(base) + '.layers.json'); data = path.read_bytes()
                if digest(path) != previous['input_sha256'][str(path)]:
                    raise ValueError('Bound TRAIN layer changed')
                extra_hashes[str(path)] = digest(path)
                atoms = json.loads(data)['atoms']
                nearest = min(atoms, key=lambda a: math.dist(c['anchor_px'], a['center']))
                risk['nearest_atom_diagnostic_only'] = dict(atom_id=nearest['atom_id'],
                    distance_px=math.dist(c['anchor_px'], nearest['center']), text=nearest['text'],
                    original_6px_matching_unchanged=True, not_reclassified_as_correct_or_incorrect=True)
            risk_records.append(risk)
            panel = Image.new('RGB', (800, 170), 'white'); draw = ImageDraw.Draw(panel)
            b = p['proposal']['proposed']
            region = (max(0, b[0] - 25), max(0, b[1] - 15), min(768, b[2] + 25), min(768, b[3] + 15))
            patch = image.crop(region); overlay = ImageDraw.Draw(patch)
            for arm, color in [('proposed', 'green'), ('original', 'red')]:
                box = p['proposal'][arm]
                overlay.rectangle((box[0] - region[0], box[1] - region[1], box[2] - region[0] - 1,
                                   box[3] - region[1] - 1), outline=color)
            patch.thumbnail((150, 135)); panel.paste(patch, (5, 30))
            draw.text((5, 4), row['sample_key'] + ' candidate ' + str(c['candidate']) + ' ' + c['status'], fill='black')
            draw.text((165, 35), 'original', fill='black'); draw.text((380, 35), 'pixel candidate', fill='black')
            for x, arm in [(165, 'original'), (380, 'proposed')]:
                panel.paste(rasterize(image, p['proposal'][arm]), (x, 55))
            draw.text((165, 120), 'saved OCR: ' + c['saved_text'] + '; added bond pixels: ' + str(risk['added_bond_ink']), fill='black')
            panels.append(panel)
    if summarize(cases) != report['summary'] or comparison(cases) != report['comparison']:
        raise ValueError('Full-cohort aggregates differ')
    if crops != 279 or len(risk_records) != 6:
        raise ValueError('Saved denominators differ')
    if report['summary']['visible_target_atoms'] != 276 or report['summary']['visible_targets_without_unique_ocr_crop'] != 15:
        raise ValueError('Missing targets were dropped')
    gallery = Image.new('RGB', (800, len(panels) * 170), 'white')
    for i, panel in enumerate(panels):
        gallery.paste(panel, (0, i * 170))
    gallery.save(OUTPUT / 'risk-crops.png')
    write_json(OUTPUT / 'report.json', dict(checks_passed=True, report_sha256=digest(RUN / 'report.json'),
        verified_parent_hashes=verified, cases=58, pixel_proposals_recomputed=crops,
        unchanged_ocr_text=True, unchanged_crop_rasters_rechecked=True, aggregates_recomputed=True,
        retained_risk_crops=risk_records, extra_input_sha256=extra_hashes,
        gallery_sha256=digest(OUTPUT / 'risk-crops.png'), checker_sha256=digest(Path(__file__)),
        model_loads=0, ocr_forwards=0, gpu_used=False, matching_changed=False))


def main():
    if sys.argv[1:] == ['--worker']:
        worker(); return 0
    receipt = runtime.run_budgeted([sys.executable, '-m', 'tools.check_pixel_crop_candidates', '--worker'],
                                  output=OUTPUT, budget=runtime.ResourceBudget(30))
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
