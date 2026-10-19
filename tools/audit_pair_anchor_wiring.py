"""Budgeted CPU schedule/wiring audit and one fixed training-image review."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

import torch

from chemvector.training.curriculum import pair_anchor_wiring as wiring, runtime
from tools import prepare_pair_anchor_augmentation as prior
from tools import pair_anchor_schedule as schedule, pair_anchor_visual_review as visual

source = prior.source
ROOT = prior.ROOT
OUTPUT = ROOT / 'outputs/audits/chemvector-pair-anchor-wiring58-20260924'
POSTCHECK = OUTPUT.with_name(OUTPUT.name + '-postcheck')
PRIOR_SHA = '52911a2f07ba6313c636cb0149c97bc61754f8cbc96c17340c3961c638382ea4'
PRIOR_POST_SHA = '9f190a6a71a771ab6811c70df5fb20e10b7d14b2144af16390db9a2dd4a67291'
SCOPES = {'bounded50': (58, 50, False), 'one_sweep': (58, 54, False),
          'cycle8': (464, 432, False), 'legacy_cycle8': (464, 432, True)}


def preflight(inputs):
    report = inputs.json(prior.OUTPUT / 'report.json', PRIOR_SHA)
    checked = inputs.json(prior.POSTCHECK / 'report.json', PRIOR_POST_SHA)
    if not checked['passed'] or checked['report_sha256'] != PRIOR_SHA:
        raise ValueError('Prior preparation postcheck required')
    for path, digest in report['input_sha256'].items(): inputs.read(path, digest)
    plan = inputs.json(prior.OUTPUT / 'plan.json', report['plan_sha256'])
    cases = [inputs.json(prior.OUTPUT / (key + '.json'), report['case_sha256'][key + '.json']) for key in plan['sample_keys']]
    for path in [Path(__file__), Path(wiring.__file__), Path(schedule.__file__), Path(visual.__file__),
                 ROOT / 'tests/chemvector/test_pair_anchor_wiring.py',
                 ROOT / 'tests/chemvector/test_pair_anchor_schedule.py',
                 ROOT / 'tests/chemvector/test_pair_anchor_visual_review.py',
                 ROOT / 'docs/model/training/20260924-pair-anchor-wiring-protocol.md']:
        inputs.read(path)
    manifest = inputs.json(source.MANIFEST, prior.MANIFEST_SHA)
    row = next(r for r in manifest['training_samples'] if r['sample_key'] == visual.KEY)
    if row['split'] != 'train': raise ValueError('Visual review must remain training-only')
    base = Path(manifest['sources'][row['source']]) / 'samples'
    meta = inputs.json(base / (visual.KEY + '.meta.json'), row['meta_sha256'])
    blobs = {suffix: inputs.read(base / (visual.KEY + suffix), meta['sha256'][suffix]) for suffix in ('.png', '.svg', '.layers.json')}
    path = source.CONTROL / ('geometry--' + visual.KEY + '.json')
    geometry = inputs.json(path, report['input_sha256'][str(path)])
    return cases, blobs, geometry


def scalar_schedule_check(projection, cases, *, attempts, updates, legacy):
    """Independent integer/list-index replay, not PerImageOffsetCycle or tensor mapping."""
    visits, count = {}, 0
    expected = []
    for attempt in range(attempts):
        if count == updates: break
        case = cases[attempt % len(cases)]
        key = case['sample_key']; visit = visits.get(key, 0); visits[key] = visit + 1
        index = 1 + (attempt if legacy else visit) % 8
        chosen = [r['row_id'] for r in case['sampled_rows']]
        kept = case['views'][index]['row_ids']
        positions = [chosen.index(rid) for rid in kept]
        count += bool(chosen)
        expected.append((key, visit, index, kept, positions))
    actual = [(s['sample_key'], s['visit'], s['view_index'], s['augmented_row_ids'], s['augmented_sampled_positions'])
              for s in projection['steps']]
    if actual != expected or projection['summary']['simulated_updates'] != count:
        raise ValueError('Independent finite-budget schedule replay differs')


def worker(postcheck=False):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('CPU-only saved-plan audit required')
    torch.set_num_threads(2)
    inputs = source.Inputs()
    if postcheck:
        report = inputs.json(OUTPUT / 'report.json')
        for path, digest in report['input_sha256'].items(): inputs.read(path, digest)
        inputs.read(OUTPUT / 'plan.json', report['plan_sha256'])
    cases, blobs, geometry = preflight(inputs)
    if not postcheck:
        source.write(OUTPUT / 'plan.json', dict(schema_version='pair-anchor-wiring-plan-0.1',
            sample_keys=[c['sample_key'] for c in cases], scopes=SCOPES, wall_seconds=120,
            schedule='per_image_visit_v1; original_offset_order; advance_on_every_visit; no_resampling',
            visual_sample_key=visual.KEY, actual_optimizer_steps=0, model_calls=0,
            input_sha256=dict(inputs.hashes), training_authorized=False))
    summaries, artifacts = {}, {}
    for name, (attempts, updates, legacy) in SCOPES.items():
        result = schedule.project(cases, attempts=attempts, updates=updates, legacy=legacy)
        scalar_schedule_check(result, cases, attempts=attempts, updates=updates, legacy=legacy)
        path = OUTPUT / (name + '.json')
        if postcheck:
            if inputs.json(path, report['artifact_sha256'][path.name]) != result:
                raise ValueError('Saved exposure report changed')
        else:
            source.write(path, result)
            artifacts[path.name] = sha256(path.read_bytes()).hexdigest()
        summaries[name] = result['summary']
    case = next(c for c in cases if c['sample_key'] == visual.KEY)
    review, canvas = visual.review(blobs['.png'], blobs['.svg'], json.loads(blobs['.layers.json']), geometry, case)
    if postcheck:
        if inputs.json(OUTPUT / 'visual.json', report['artifact_sha256']['visual.json']) != review:
            raise ValueError('Saved raster/SVG review differs')
        inputs.read(OUTPUT / 'visual.png', report['artifact_sha256']['visual.png'])
        # PNG metadata/compression is not an image-equivalence guarantee.
        from PIL import Image
        import numpy as np
        if not np.array_equal(np.asarray(canvas), np.asarray(Image.open(OUTPUT / 'visual.png').convert('RGB'))):
            raise ValueError('Visual contact sheet pixel replay differs')
        if summaries != report['summaries']: raise ValueError('Summary differs')
    else:
        source.write(OUTPUT / 'visual.json', review)
        canvas.save(OUTPUT / 'visual.png')
        for name in ('visual.json', 'visual.png'): artifacts[name] = sha256((OUTPUT / name).read_bytes()).hexdigest()
    for path, digest in list(inputs.hashes.items()): inputs.read(path, digest)
    if torch.cuda.is_initialized(): raise ValueError('Unexpected CUDA initialization')
    if postcheck:
        source.write(POSTCHECK / 'report.json', dict(schema_version='pair-anchor-wiring-postcheck-0.1',
            passed=True, source_report_sha256=inputs.hashes[str(OUTPUT / 'report.json')],
            independent_scalar_schedules=4, visual_pixels_replayed=True, input_sha256=inputs.hashes,
            model_calls=0, actual_optimizer_steps=0, gpu_used=False, training_authorized=False))
    else:
        source.write(OUTPUT / 'report.json', dict(schema_version='pair-anchor-wiring58-0.1', summaries=summaries,
            plan_sha256=sha256((OUTPUT / 'plan.json').read_bytes()).hexdigest(), artifact_sha256=artifacts,
            input_sha256=inputs.hashes, independent_scalar_schedules=4,
            historical_implementations_changed=False, default_changed=False, data_release_changed=False,
            training_authorized=False, safe_to_launch=False, model_calls=0, actual_optimizer_steps=0,
            gpu_used=False, training_performed=False, new_identity_queries=0,
            validation_sample_artifacts_read=False, sealed_test_read=False,
            visual_training_images_read=1, losses_evaluated_on_real_data=0,
            limitations=['Exposure projection is not optimization or achieved model accuracy',
                'Current CPU objective is not a GPU launcher or a fresh execution-device cache',
                'Per-image visits fix periodic aliasing but do not remove GT-qualified direction/type bias',
                'Full eight-cycle metadata check is not permission for 432 real updates',
                'Original loss branch retained does not guarantee shared-parameter nonregression',
                'New purpose/run authorization, matched control, resource budget and full-graph protection still required']))
    print(json.dumps(dict(postcheck=postcheck, scopes={n: {k: s[k] for k in
        ('attempts', 'simulated_updates', 'unique_images_visited', 'never_augmented_sampled_rows')} for n, s in summaries.items()})))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--postcheck', action='store_true')
    args = parser.parse_args(argv)
    if args.worker:
        worker(args.postcheck); return 0
    command = [sys.executable, '-m', 'tools.audit_pair_anchor_wiring', '--worker']
    if args.postcheck: command.append('--postcheck')
    receipt = runtime.run_budgeted(command, output=POSTCHECK if args.postcheck else OUTPUT,
                                  budget=runtime.ResourceBudget(120))
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
