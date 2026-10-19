"""Fixed58 CPU-only node grouping proposals, not authorized training data."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

import torch

from chemvector.training.curriculum import runtime
from tools import selector_node_groups as groups
from tools import selector_node_groups_review as review
from tools import selector_node_semantics as semantics

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'outputs/audits/chemvector-selector-node-semantics-20260920'
CONTROL = ROOT / 'outputs/experiments/chemvector-selector-full-loss-control-20260920'
RETENTION = ROOT / 'outputs/experiments/chemvector-selector-retention-trial-20260920'
RELEASE = ROOT / 'outputs/datasets/chemvector-selector-retention-fit-20260919'
OUTPUT = ROOT / 'outputs/audits/chemvector-selector-node-groups-20260920'
AUDIT_SHA = 'fc14d44bf5f35052a8d1f53c55e274578ce833169b98acbf1cc7b9ef478396c2'


def write(path, data):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


class Inputs:
    def __init__(self):
        self.hashes = {}

    def read(self, path, digest=None):
        path = Path(path).absolute()
        if path.resolve() != path or 'sealed_test' in path.parts:
            raise ValueError('Only explicit nonsymlink, nonsealed input paths')
        value = path.read_bytes()
        actual = sha256(value).hexdigest()
        if (digest is not None and digest != actual
                or str(path) in self.hashes and self.hashes[str(path)] != actual):
            raise ValueError('Input binding changed: ' + str(path))
        self.hashes[str(path)] = actual
        return value

    def json(self, path, digest=None):
        return json.loads(self.read(path, digest))


def preflight(inputs):
    audit = inputs.json(AUDIT / 'report.json', AUDIT_SHA)
    bindings = audit['input_sha256']
    manifest = inputs.json(RELEASE / 'manifest.json', bindings[str(RELEASE / 'manifest.json')])
    rows = groups.training_rows(manifest)
    if Counter(r['fit_domain'] for r in rows) != {'witness': 26, 'old_full_molecule': 32}:
        raise ValueError('Fixed26+32 only')
    return manifest, rows, bindings


def worker():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('CPU-only artifact proposal worker required')
    if not OUTPUT.is_dir() or (OUTPUT / 'plan.json').exists():
        raise ValueError('Fresh budgeted output required')
    torch.set_num_threads(2)
    inputs = Inputs()
    manifest, rows, bindings = preflight(inputs)
    for module in (Path(__file__), Path(groups.__file__), Path(review.__file__), Path(semantics.__file__), Path(runtime.__file__)):
        inputs.read(module, bindings.get(str(module)))
    config = ROOT / 'frontend/inference_profiles.json'
    inputs.read(config, bindings[str(config)])
    write(OUTPUT / 'plan.json', dict(schema_version='selector-node-group-proposal-plan-0.1',
        sample_keys=[r['sample_key'] for r in rows], training_authorized=False, requires_review=True,
        validation_artifacts_read=False, gpu_count=0, model_forwards=0, optimizer_steps=0,
        radius_px=6, score_threshold=-.25, wall_budget_seconds=120,
        policy='All raw candidates; unique memberships protected; all ownership evidence must agree; owner needs raw representative; unsupported related bond excluded',
        purpose='Offline grouping/pairing proposal and GT-assisted removal-only constraint checks, not new labels or inference rules'))
    cases = []
    totals = defaultdict(Counter)
    statuses = defaultdict(Counter)
    node_statuses = defaultdict(Counter)
    phase_totals = defaultdict(lambda: defaultdict(Counter))
    conflicts = []
    for row in rows:
        key, group = row['sample_key'], row['fit_domain']
        source = Path(manifest['sources'][row['source']]) / 'samples'
        meta = inputs.json(source / (key + '.meta.json'), row['meta_sha256'])
        parts = {suffix: inputs.read(source / (key + suffix), meta['sha256'][suffix])
                 for suffix in ('.layers.json', '.text-records.json', '.svg')}
        layers = json.loads(parts['.layers.json'])
        glyphs = json.loads(parts['.text-records.json'])['glyphs']
        if layers['sample_key'] != key or layers['molecule_id'] != row['molecule_id']:
            raise ValueError('Source identity mismatch')
        geo_path = CONTROL / ('geometry--' + key + '.json')
        geometry = inputs.json(geo_path, bindings[str(geo_path)])
        semantics.check_geometry(geometry, layers)
        paths, unknown = semantics.strokes(parts['.svg'], layers)
        case = groups.propose(geometry, layers, glyphs, paths, unknown)
        case.update(sample_key=key, canonical_smiles=row['canonical_smiles'], group=group, phase_checks={})
        for phase, parent, prefix in (('initial', CONTROL, 'baseline'),
                ('retention-200', RETENTION, 'check-000200'), ('control-200', CONTROL, 'check-000200'),
                ('retention-400', RETENTION, 'check-000400')):
            path = parent / (prefix + '--' + key + '.json')
            record = inputs.json(path, bindings[str(path)])
            if record['key'] != geometry['key'] or record['group'] != group:
                raise ValueError('Prediction identity mismatch')
            check = groups.removal_check(geometry, case, record, layers)
            case['phase_checks'][phase] = check
            stats = phase_totals[phase][group]
            stats['images'] += 1
            for label in ('removed_fragment_slots', 'lost_target_ids', 'lost_bond_ids', 'gained_target_ids',
                          'original_glyph_matched_slots', 'original_valid_replacement_slots',
                          'removed_correct_glyph_slots', 'removed_valid_replacement_slots'):
                stats[label] += len(check[label])
            stats['before_unmatched_selected'] += check['before']['unmatched_selected']
            stats['after_unmatched_selected'] += check['after']['unmatched_selected']
            stats['matcher_regression_images'] += not check['original_matcher_nonregression']
            stats.update('node_' + c['state'] for c in check['node_constraints'])
            if not check['original_matcher_nonregression']:
                conflicts.append(dict(sample_key=key, phase=phase, lost_target_ids=check['lost_target_ids'], lost_bond_ids=check['lost_bond_ids']))
        count = totals[group]
        count['images'] += 1
        count['raw_candidates'] += len(case['candidates'])
        count['target_nodes'] += len(case['nodes'])
        count['proposed_pairs'] += len(case['proposed_pairs'])
        count['images_with_pairs'] += bool(case['proposed_pairs'])
        count['targets_with_pairs'] += len({p['owner_target'] for p in case['proposed_pairs']})
        count['unique_alternative_candidates'] += sum(len(n['alternative_slots']) for n in case['nodes'])
        count['unsupported_svg_nodes'] += len(unknown)
        count['target_bonds'] += len(layers['bonds'])
        available = {n['target_id'] for n in case['nodes'] if n['unique_slots']}
        count['bonds_with_unique_candidate_endpoints'] += sum(b['begin_atom'] in available and b['end_atom'] in available for b in layers['bonds'])
        for pair in case['proposed_pairs']:
            count.update('pair_' + kind for kind in pair['kinds'])
        statuses[group].update(c['status'] for c in case['candidates'])
        node_statuses[group].update(n['status'] for n in case['nodes'])
        write(OUTPUT / (key + '.json'), case)
        if key in review.FOCUS:
            png = source / (key + '.png')
            inputs.read(png, meta['sha256']['.png'])
            review.overlay(OUTPUT, case, png)
        cases.append(case)
    review.page(OUTPUT, cases)
    for path, digest in list(inputs.hashes.items()):
        inputs.read(path, digest)
    if torch.cuda.is_initialized():
        raise ValueError('Unexpected CUDA initialization')
    write(OUTPUT / 'report.json', dict(schema_version='selector-node-group-proposals-0.1',
        images=len(cases), input_sha256=inputs.hashes, totals=totals, candidate_statuses=statuses,
        node_statuses=node_statuses, phase_checks=phase_totals, matcher_conflicts=conflicts,
        training_authorized=False, requires_review=True, training_performed=False, model_forwards=0,
        optimizer_steps=0, gpu_used=False, validation_artifacts_read=False, new_identity_queries=0,
        default_changed=False, prediction_artifacts_changed=False, gallery=list(review.FOCUS),
        limitations=['GT/SVG-assisted; not an image-only recognizer or deployable filter',
            'All raw-candidate proposals, not counts of actual selected errors',
            'Spatial grouping does not establish element/bond correctness',
            'BBox evidence is not ink-pixel evidence; manual review still required',
            'Original Hungarian matcher remains unchanged and may couple distant candidates',
            'No released labels, parameter updates, new accuracy, or unseen-test claim']))
    print(json.dumps(dict(images=len(cases), totals=totals, matcher_conflicts=len(conflicts)), ensure_ascii=False), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        worker()
        return 0
    preflight(Inputs())
    receipt = runtime.run_budgeted([sys.executable, '-m', 'tools.propose_selector_node_groups', '--worker'],
        output=OUTPUT, budget=runtime.ResourceBudget(120))
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
