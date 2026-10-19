"""Fixed122 saved-candidate audit + gallery; no model/checkpoint/optimizer load."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

import torch

from chemvector.training.curriculum.runtime import ResourceBudget, run_budgeted
from tools import selector_node_semantics as semantics
from tools import selector_node_semantics_gallery as gallery

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / 'outputs/experiments/chemvector-selector-full-loss-control-20260920'
RETENTION = ROOT / 'outputs/experiments/chemvector-selector-retention-trial-20260920'
RELEASE = ROOT / 'outputs/datasets/chemvector-selector-retention-fit-20260919'
OUTPUT = ROOT / 'outputs/audits/chemvector-selector-node-semantics-20260920'
BINDINGS = {
    str(CONTROL / 'report.json'): '640b118c860430a69e3f04fc32bdf7ad7014691cfbf30d755e97272732dfde65',
    str(RETENTION / 'report.json'): 'e05e574bc43a5743942cc61ec717ebf913e80ab0634a99dc94d45fa8054afeb2',
    str(RELEASE / 'manifest.json'): 'f5e3a7163eef5c49118ad78dcf05778f4ad889dc9b6389097ef20896b79e2111',
}


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


class Inputs:
    def __init__(self):
        self.hashes = {}

    def read(self, path, expected=None):
        path = Path(path).absolute()
        if path.resolve() != path or 'sealed_test' in path.parts or not path.is_file():
            raise ValueError('Explicit nonsealed nonsymlink input required')
        value = path.read_bytes()
        digest = sha256(value).hexdigest()
        if (expected and digest != expected) or (str(path) in self.hashes and self.hashes[str(path)] != digest):
            raise ValueError('Audit input changed: ' + str(path))
        self.hashes[str(path)] = digest
        return value

    def json(self, path, expected=None):
        return json.loads(self.read(path, expected))

    def verify(self):
        for path, digest in list(self.hashes.items()):
            self.read(path, digest)


def preflight(inputs):
    for path, digest in BINDINGS.items():
        inputs.read(path, digest)
    control, retention = [inputs.json(p / 'report.json') for p in (CONTROL, RETENTION)]
    for key in ('model_state_before', 'cache_before', 'visual_cache_before', 'frozen_before', 'ocr_before'):
        if control[key] != retention[key]:
            raise ValueError('Prior trials do not share identical initial predictions/targets')
    if control['optimizer_steps'] != 200 or retention['optimizer_steps'] != 400:
        raise ValueError('Unexpected trial endpoints')
    manifest = inputs.json(RELEASE / 'manifest.json')
    rows = manifest['training_samples'] + manifest['validation_only_samples']
    if len(manifest['training_samples']) != 58 or len(rows) != 122 or len({r['sample_key'] for r in rows}) != 122:
        raise ValueError('Keep all original58+64 rows')
    return manifest, rows


def add_counts(total, states, atoms):
    total['images'] += 1
    total['retained_raw_negative'] += sum(r['selected'] and r['raw_target'] < 0 for r in states)
    total['unmatched_selected'] += sum(r['selected'] and r['selected_target'] < 0 for r in states)
    total['matched_raw_negative_replacements'] += sum(r['selected_target'] >= 0 and r['raw_target'] < 0 for r in states)
    total['selected_matched_in_glyph_bbox'] += sum(r['selected_target'] >= 0 and bool(r['glyph_bbox_hits']) for r in states)
    total['raw_positive_unmatched_selected'] += sum(r['selected'] and r['selected_target'] < 0 and r['raw_target'] >= 0 for r in states)
    for row in states:
        if row['selected'] and row['raw_target'] < 0:
            total['retained_negative_buckets'][row['bucket']] += 1
        if row['selected'] and row['selected_target'] < 0:
            total['unmatched_buckets'][row['bucket']] += 1
        if row['raw_target'] >= 0 and not row['selected']:
            name = 'rejected_positive_text' if atoms[row['raw_target']]['visible_text'] else 'rejected_positive_implicit'
            total[name] += 1


def fresh_counts():
    return dict(images=0, retained_raw_negative=0, unmatched_selected=0, matched_raw_negative_replacements=0,
        selected_matched_in_glyph_bbox=0, raw_positive_unmatched_selected=0,
        retained_negative_buckets=Counter(), unmatched_buckets=Counter(),
        rejected_positive_text=0, rejected_positive_implicit=0)


def worker():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('CPU-only saved-artifact worker required')
    if not OUTPUT.is_dir() or (OUTPUT / 'plan.json').exists():
        raise ValueError('Fresh supervised output required')
    torch.set_num_threads(2)
    inputs = Inputs()
    manifest, rows = preflight(inputs)
    for path in (Path(__file__), Path(semantics.__file__), Path(gallery.__file__),
                 ROOT / 'frontend/inference_profiles.json'):
        inputs.read(path)
    write_json(OUTPUT / 'plan.json', dict(schema_version='selector-node-semantics-plan-0.1',
        samples=rows, gallery_keys=gallery.FOCUS, wall_seconds=120, gpu_count=0, model_forwards=0,
        selection='All existing122; fixed core4 + previously identified3 extra-node development cases in gallery',
        threshold=-.25, matching_radius_px=6, location_tolerance_px=6,
        location_policy='Glyph bounds exact; stroke endpoint proximity reuses6px. Descriptive labels only, never inference rules.',
        prohibited=['new training', 'threshold or matcher changes', 'new sample selection', 'evaluation identity queries', 'deployment']))
    totals = defaultdict(lambda: defaultdict(fresh_counts))
    entries, unsupported_total, replay_count = [], 0, 0
    cases = []
    for row in rows:
        key = row['sample_key']
        source = Path(manifest['sources'][row['source']]) / 'samples'
        meta = inputs.json(source / (key + '.meta.json'), row['meta_sha256'])
        contents = {suffix: inputs.read(source / (key + suffix), meta['sha256'][suffix])
                    for suffix in ('.layers.json', '.text-records.json', '.svg', '.png')}
        layers, text = [json.loads(contents[s]) for s in ('.layers.json', '.text-records.json')]
        if layers['sample_key'] != key or layers['molecule_id'] != row['molecule_id']:
            raise ValueError('Layer identity differs from fixed manifest')
        geometry = inputs.json(CONTROL / ('geometry--' + key + '.json'))
        semantics.check_geometry(geometry, layers)
        replay_count += 1
        paths, unsupported = semantics.strokes(contents['.svg'], layers)
        unsupported_total += len(unsupported)
        locations = {c['slot']: semantics.location(c, 768, 6, text['glyphs'], paths) for c in geometry['candidates']}
        atoms = sorted(layers['atoms'], key=lambda a: a['atom_id'])
        baseline = inputs.json(CONTROL / ('baseline--' + key + '.json'))
        previous = inputs.json(RETENTION / ('baseline--' + key + '.json'))
        if baseline['metrics'] != previous['metrics']:
            raise ValueError('Reference baseline changed')
        development = row.get('purpose') == 'diagnostic'
        records = {'initial': baseline}
        if not development:
            records['retention-200'] = inputs.json(RETENTION / ('check-000200--' + key + '.json'))
        records['control-200'] = inputs.json(CONTROL / (('final-development' if development else 'check-000200') + '--' + key + '.json'))
        records['retention-400'] = inputs.json(RETENTION / (('final-development' if development else 'check-000400') + '--' + key + '.json'))
        case = dict(sample_key=key, key=geometry['key'], group=baseline['group'], canonical_smiles=row['canonical_smiles'],
            target_anchors=geometry['target_anchors'], unsupported_svg_nodes=unsupported, phases={})
        for phase, record in records.items():
            states = semantics.decisions(geometry, record, locations)
            replay_count += 1
            # Existing control matching files provide a second saved-output crosscheck.
            if phase in ('initial', 'control-200'):
                name = 'matching-000000' if phase == 'initial' else ('matching-final-development' if development else 'matching-000200')
                saved = inputs.json(CONTROL / (name + '--' + key + '.json'))['selected_target_by_slot']
                if any(saved[r['slot']] != r['selected_target'] for r in states if r['selected']):
                    raise ValueError('Saved selected mapping changed')
            add_counts(totals[phase][case['group']], states, atoms)
            case['phases'][phase] = dict(candidates=states, original_metrics={k:v for k,v in record['metrics'].items() if k != 'logits'})
        write_json(OUTPUT / (key + '.json'), case)
        cases.append(dict(sample_key=key, group=case['group']))
        if key in gallery.FOCUS:
            entries.append((case, gallery.render(OUTPUT, case, source / (key + '.png'))))
        if len(cases) % 20 == 0:
            print(json.dumps(dict(completed=len(cases), total=122)), flush=True)
    if len(entries) != len(gallery.FOCUS):
        raise ValueError('Fixed gallery incomplete')
    entries.sort(key=lambda item: gallery.FOCUS.index(item[0]['sample_key']))
    gallery.index(OUTPUT, entries)
    if torch.cuda.is_initialized():
        raise ValueError('Unexpected CUDA initialization')
    inputs.verify()
    write_json(OUTPUT / 'report.json', dict(schema_version='selector-node-semantics-audit-0.1',
        images=122, input_sha256=inputs.hashes, phases=totals, matching_replays=replay_count,
        unsupported_svg_nodes=unsupported_total, gallery=[k for k in gallery.FOCUS],
        model_forwards=0, optimizer_steps=0, gpu_used=False, training_performed=False,
        new_identity_queries=0, source_mutations=0, deployed=False, default_kept='initial',
        limitations=['SVG/GT used only for diagnosis; not an image-only inference algorithm',
            'Buckets are descriptive associations, not causal feature attribution or new training labels',
            'Glyph bbox overlap does not prove overlap with actual ink pixels',
            'Unsupported/closed/curved SVG paths remain unclassified rather than approximated',
            'new64 has no retention200; unequal-step endpoints are not a causal comparison',
            'No strict SDF recognition improvement is measured']))
    print(json.dumps(dict(completed=122, matching_replays=replay_count, unsupported_svg_nodes=unsupported_total)), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        worker()
        return 0
    preflight(Inputs())
    receipt = run_budgeted([sys.executable, '-m', 'tools.audit_selector_node_semantics', '--worker'],
                           output=OUTPUT, budget=ResourceBudget(120))
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
