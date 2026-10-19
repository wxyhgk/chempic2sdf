"""Fixed58 saved-phase CPU counterfactual for coordinate-stable proposals."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

import torch

from chemvector.training.curriculum import runtime
from tools import selector_coordinate_contract as coordinates
from tools import selector_node_groups as groups
from tools import selector_node_semantics as semantics
from tools import selector_stable_subset as stable

ROOT = Path(__file__).resolve().parents[1]
COORDINATES = ROOT / 'outputs/audits/chemvector-selector-coordinate-contract-20260920-r1'
COORDINATES_SHA = '834bcf6eeb2e119bc8e0bfd64db1fbe20259bcd6275c384dd72ff0c5bbc77e62'
PREVIOUS = ROOT / 'outputs/audits/chemvector-selector-node-groups-20260920'
MANIFEST = ROOT / 'outputs/datasets/chemvector-selector-retention-fit-20260919/manifest.json'
CONTROL = ROOT / 'outputs/experiments/chemvector-selector-full-loss-control-20260920'
RETENTION = ROOT / 'outputs/experiments/chemvector-selector-retention-trial-20260920'
OUTPUT = ROOT / 'outputs/audits/chemvector-selector-stable-subset-20260920'
PHASES = (('initial', CONTROL, 'baseline'), ('retention-200', RETENTION, 'check-000200'),
          ('control-200', CONTROL, 'check-000200'), ('retention-400', RETENTION, 'check-000400'))


def write(path, data):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


class Inputs:
    def __init__(self):
        self.hashes = {}

    def read(self, path, expected=None):
        path = Path(path).absolute()
        if path.resolve() != path or 'sealed_test' in path.parts:
            raise ValueError('Explicit nonsealed nonsymlink inputs only')
        value = path.read_bytes()
        digest = sha256(value).hexdigest()
        if ((expected is not None and digest != expected)
                or str(path) in self.hashes and self.hashes[str(path)] != digest):
            raise ValueError('Input changed: ' + str(path))
        self.hashes[str(path)] = digest
        return value

    def json(self, path, expected=None):
        return json.loads(self.read(path, expected))


def preflight(inputs):
    coordinate = inputs.json(COORDINATES / 'report.json', COORDINATES_SHA)
    bindings = coordinate['input_sha256']
    manifest = inputs.json(MANIFEST, bindings[str(MANIFEST)])
    rows = groups.training_rows(manifest)
    if Counter(r['fit_domain'] for r in rows) != {'witness': 26, 'old_full_molecule': 32}:
        raise ValueError('Fixed58 scope changed')
    return coordinate, manifest, rows, bindings


def worker():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('CPU-only saved-artifact audit required')
    if not OUTPUT.is_dir() or (OUTPUT / 'plan.json').exists():
        raise ValueError('Fresh budgeted output required')
    torch.set_num_threads(2)
    inputs = Inputs()
    coordinate, manifest, rows, bindings = preflight(inputs)
    for module in (Path(__file__), Path(stable.__file__), Path(coordinates.__file__),
                   Path(groups.__file__), Path(semantics.__file__), Path(runtime.__file__),
                   ROOT / 'frontend/inference_profiles.json'):
        inputs.read(module, bindings.get(str(module)))
    write(OUTPUT / 'plan.json', dict(schema_version='selector-stable-subset-plan-0.1',
        sample_keys=[r['sample_key'] for r in rows], gpu_count=0, model_forwards=0, optimizer_steps=0,
        training_authorized=False, requires_review=True, wall_budget_seconds=120,
        expected_prior_pairs=525, expected_stable_pairs=502, expected_excluded_pairs=23,
        match_radius_px=6, score_threshold=-.25, phases=[p[0] for p in PHASES],
        validation_sample_artifacts_read=False,
        purpose='Reconstruct and version a proposal subset; GT-assisted removal-only counterfactual, not inference or labels'))
    totals = defaultdict(Counter)
    phase_totals = defaultdict(lambda: defaultdict(Counter))
    excluded_rows, regressions, case_hashes = [], [], {}
    for row in rows:
        key, group = row['sample_key'], row['fit_domain']
        source = Path(manifest['sources'][row['source']]) / 'samples'
        meta = inputs.json(source / (key + '.meta.json'), row['meta_sha256'])
        layers = inputs.json(source / (key + '.layers.json'), meta['sha256']['.layers.json'])
        glyphs = inputs.json(source / (key + '.text-records.json'), meta['sha256']['.text-records.json'])['glyphs']
        svg = inputs.read(source / (key + '.svg'), meta['sha256']['.svg'])
        if layers['sample_key'] != key or layers['molecule_id'] != row['molecule_id']:
            raise ValueError('Source identity differs')
        geo_path = CONTROL / ('geometry--' + key + '.json')
        geometry = inputs.json(geo_path, bindings[str(geo_path)])
        if geometry['key'] != row['source'] + ':' + key:
            raise ValueError('Saved geometry identity differs')
        semantics.check_geometry(geometry, layers)
        paths, unknown = semantics.strokes(svg, layers)
        proposal_path = PREVIOUS / (key + '.json')
        original = inputs.json(proposal_path, bindings[str(proposal_path)])
        rebuilt = groups.propose(geometry, layers, glyphs, paths, unknown)
        if any(not stable.serialized_equal(original.get(k), value) for k, value in rebuilt.items()):
            raise ValueError('Original proposal does not reconstruct')
        review_path = COORDINATES / (key + '.json')
        prior_review = inputs.json(review_path)
        reviewed = coordinates.pair_robustness(geometry, layers, glyphs, paths, unknown, rebuilt)
        if (prior_review['sample_key'] != key or prior_review['group'] != group
                or not stable.serialized_equal(prior_review['prior_pair_review'], reviewed)):
            raise ValueError('Coordinate review does not reconstruct')
        result = stable.stable_subset(rebuilt, reviewed)
        result.update(sample_key=key, group=group, canonical_smiles=row['canonical_smiles'], phase_checks={})
        for exclusion in result['excluded_pairs']:
            excluded_rows.append(dict(sample_key=key, group=group, **exclusion['coordinate_review']))
        counts = totals[group]
        counts['images'] += 1
        counts['original_pairs'] += len(rebuilt['proposed_pairs'])
        counts['stable_pairs'] += len(result['stable_proposal']['proposed_pairs'])
        counts['excluded_pairs'] += len(result['excluded_pairs'])
        counts['images_with_stable_pairs'] += bool(result['stable_proposal']['proposed_pairs'])
        counts['stable_pair_owner_targets'] += len({p['owner_target'] for p in result['stable_proposal']['proposed_pairs']})
        counts['target_nodes'] += len(rebuilt['nodes'])
        counts['raw_candidates'] += len(rebuilt['candidates'])
        counts['protected_alternative_candidates'] += sum(len(n['alternative_slots']) for n in rebuilt['nodes'])
        counts['unsupported_svg_nodes'] += len(unknown)
        for pair in result['stable_proposal']['proposed_pairs']:
            counts.update('stable_pair_' + kind for kind in pair['kinds'])
        for phase, parent, prefix in PHASES:
            path = parent / (prefix + '--' + key + '.json')
            record = inputs.json(path, bindings[str(path)])
            if record['key'] != geometry['key'] or record['group'] != group:
                raise ValueError('Prediction identity differs')
            check = stable.removal_check(geometry, result, record, layers, glyphs)
            result['phase_checks'][phase] = check
            stats = phase_totals[phase][group]
            stats['images'] += 1
            for label in ('removed_fragment_slots', 'lost_target_ids', 'lost_bond_ids', 'gained_target_ids',
                    'removed_correct_slots', 'removed_correct_glyph_slots', 'removed_valid_replacement_slots',
                    'source_projected_glyph_matched_slots', 'removed_source_projected_glyph_matched_slots',
                    'original_visible_text_target_ids', 'lost_visible_text_target_ids',
                    'lost_original_assignment_relations', 'changed_valid_replacement_relations',
                    'removed_spatially_eligible_slots', 'held_sensitive_selected_slots'):
                stats[label] += len(check[label])
            for when in ('before', 'after'):
                stats[when + '_matched_targets'] += len(check[when]['matched_target_ids'])
                stats[when + '_reachable_bonds'] += len(check[when]['reachable_bond_ids'])
                stats[when + '_unmatched_selected'] += check[when]['unmatched_selected']
            stats['matcher_regression_images'] += not check['original_matcher_nonregression']
            stats['protection_failure_images'] += not check['all_protections_preserved']
            if not check['all_protections_preserved']:
                regressions.append(dict(sample_key=key, group=group, phase=phase,
                    lost_target_ids=check['lost_target_ids'], lost_bond_ids=check['lost_bond_ids'],
                    lost_original_assignment_relations=check['lost_original_assignment_relations']))
        path = OUTPUT / (key + '.json')
        write(path, result)
        case_hashes[path.name] = sha256(path.read_bytes()).hexdigest()
    if not stable.serialized_equal(excluded_rows, coordinate['coordinate_sensitive_pairs']):
        raise ValueError('Excluded items differ from SHA-bound coordinate report')
    if ({g: dict(stable_pairs=c['stable_pairs'], excluded_pairs=c['excluded_pairs']) for g, c in totals.items()}
            != {'witness': dict(stable_pairs=82, excluded_pairs=5),
                'old_full_molecule': dict(stable_pairs=420, excluded_pairs=18)}):
        raise ValueError('Fixed coordinate partition changed')
    for path, digest in list(inputs.hashes.items()):
        inputs.read(path, digest)
    if torch.cuda.is_initialized():
        raise ValueError('Unexpected CUDA initialization')
    write(OUTPUT / 'report.json', dict(schema_version='selector-stable-subset-0.1',
        input_sha256=inputs.hashes, case_sha256=case_hashes, images=len(rows), saved_phase_checks=len(rows)*len(PHASES),
        totals=totals, phase_checks=phase_totals, protection_regressions=regressions,
        all_phase_protections_preserved=not regressions,
        training_authorized=False, requires_review=True, training_performed=False,
        model_forwards=0, optimizer_steps=0, gpu_used=False, validation_sample_artifacts_read=False,
        new_identity_queries=0, threshold_or_labels_changed=False, default_changed=False,
        prediction_artifacts_changed=False,
        limitations=['GT/SVG-assisted diagnostic, not a deployed filter or approved training-label release',
            'Coordinate stability does not imply human-verified semantic ownership',
            'Raw pairs count all candidate fragments, not actual selected errors',
            'Near-target alternatives remain protected; unseen targets are not repaired',
            'No new accuracy or independent test-set evaluation']))
    print(json.dumps(dict(images=len(rows), totals=totals, protection_regressions=len(regressions)), ensure_ascii=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        worker()
        return 0
    preflight(Inputs())
    receipt = runtime.run_budgeted([sys.executable, '-m', 'tools.audit_selector_stable_subset', '--worker'],
        output=OUTPUT, budget=runtime.ResourceBudget(120))
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
