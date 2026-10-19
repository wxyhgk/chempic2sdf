"""Fixed58 x four saved phases: independent radius-first matching reference."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import scipy
import torch

from chemvector.training.curriculum import runtime
from tools import audit_curriculum_v02_selector_matching as radius_source
from tools import audit_selector_stable_subset as prior
from tools import selector_node_groups as groups
from tools import selector_node_semantics as semantics
from tools import selector_radius_reference as reference
from tools.selector_stable_subset import serialized_equal

SOURCE_SHA = '462fc1555110ea42ed9379aeaf5d6746ac848141f9d035deb284cd283758c17f'
MANIFEST_SHA = 'f5e3a7163eef5c49118ad78dcf05778f4ad889dc9b6389097ef20896b79e2111'
OUTPUT = prior.ROOT / 'outputs/audits/chemvector-selector-radius-reference-20260920'


def preflight(inputs):
    report = inputs.json(prior.OUTPUT / 'report.json', SOURCE_SHA)
    manifest = inputs.json(prior.MANIFEST, MANIFEST_SHA)
    rows = groups.training_rows(manifest)
    if (Counter(r['fit_domain'] for r in rows) != {'witness': 26, 'old_full_molecule': 32}
            or {r['sample_key'] + '.json' for r in rows} != set(report['case_sha256'])
            or report['images'] != 58 or report['saved_phase_checks'] != 232
            or report.get('training_authorized') is not False):
        raise ValueError('Fixed review-only 58-image scope changed')
    return report, manifest, rows


def check_selection(geometry, record):
    logits = record['metrics']['logits']
    raw = {c['slot']: c for c in geometry['candidates']}
    if (set(logits) != {str(s) for s in raw}
            or any(not math.isfinite(v) for v in logits.values())):
        raise ValueError('Exact finite saved-logit coverage required')
    chosen = sorted(int(s) for s, v in logits.items() if v >= -.25)
    positive = {s for s, c in raw.items() if c['raw_target'] >= 0}
    expected = dict(selected_atoms=len(chosen), positive_slots=sorted(positive),
        retained_positive_slots=sorted(set(chosen) & positive),
        selected_negative_slots=sorted(set(chosen) - positive))
    if any(record['metrics'][k] != v for k, v in expected.items()):
        raise ValueError('Saved selector decisions did not reproduce')
    return chosen


def count_coverage(counter, prefix, result):
    for label in ('matched_target_ids', 'reachable_bond_ids', 'missing_target_ids',
                  'no_eligible_candidate_target_ids', 'eligible_but_unassigned_target_ids'):
        counter[prefix + '_' + label] += len(result[label])
    counter[prefix + '_unmatched_selected'] += result['unmatched_selected']


def worker():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('CPU-only saved-artifact replay required')
    if not OUTPUT.is_dir() or (OUTPUT / 'plan.json').exists():
        raise ValueError('Fresh budgeted output required')
    torch.set_num_threads(2)
    inputs = prior.Inputs()
    source, manifest, rows = preflight(inputs)
    bindings = source['input_sha256']
    for module in (Path(__file__), Path(reference.__file__), Path(radius_source.__file__),
                   Path(prior.__file__), Path(groups.__file__), Path(semantics.__file__),
                   Path(sys.modules[serialized_equal.__module__].__file__), Path(runtime.__file__),
                   prior.ROOT / 'frontend/inference_profiles.json'):
        inputs.read(module, bindings.get(str(module)))
    prior.write(OUTPUT / 'plan.json', dict(schema_version='selector-radius-reference-plan-0.1',
        sample_keys=[r['sample_key'] for r in rows], phases=[p[0] for p in prior.PHASES],
        gpu_count=0, model_forwards=0, optimizer_steps=0, training_authorized=False,
        wall_budget_seconds=120, automatic_retries=0, original_metrics_authoritative=True,
        match_radius_px=6, score_threshold=-.25, image_size=768,
        cost_policy='float32 cdist; reference removal slices frozen before costs; numerical recompute separate',
        purpose='GT-assisted diagnostic only; no production matcher, labels, or predictions changed'))
    raw_totals, phase_totals = defaultdict(Counter), defaultdict(lambda: defaultdict(Counter))
    cases, differences, regressions, missing_raw = {}, [], [], []
    partition = Counter()
    for row in rows:
        key, domain = row['sample_key'], row['fit_domain']
        case_path = prior.OUTPUT / (key + '.json')
        previous = inputs.json(case_path, source['case_sha256'][case_path.name])
        if (previous['sample_key'] != key or previous['group'] != domain
                or previous.get('training_authorized') is not False
                or previous.get('requires_review') is not True
                or set(previous['phase_checks']) != {p[0] for p in prior.PHASES}):
            raise ValueError('Stable case identity/review scope differs')
        source_root = Path(manifest['sources'][row['source']]) / 'samples'
        meta = inputs.json(source_root / (key + '.meta.json'), row['meta_sha256'])
        layer_path = source_root / (key + '.layers.json')
        layers = inputs.json(layer_path, bindings[str(layer_path)])
        if (inputs.hashes[str(layer_path)] != meta['sha256']['.layers.json']
                or layers['sample_key'] != key or layers['molecule_id'] != row['molecule_id']):
            raise ValueError('Same-render source identity mismatch')
        geo_path = prior.CONTROL / ('geometry--' + key + '.json')
        geometry = inputs.json(geo_path, bindings[str(geo_path)])
        if geometry['key'] != row['source'] + ':' + key:
            raise ValueError('Geometry identity mismatch')
        semantics.check_geometry(geometry, layers)
        raw_slots = sorted(c['slot'] for c in geometry['candidates'])
        raw = reference.compare(geometry, raw_slots, layers)
        if raw['original']['mapping'] != {c['slot']: c['raw_target'] for c in geometry['candidates']}:
            raise ValueError('Historical raw assignment did not reproduce')
        raw_counts = raw_totals[domain]
        raw_counts.update(images=1, targets=len(layers['atoms']), bonds=len(layers['bonds']))
        for policy in ('original', 'radius_reference'):
            count_coverage(raw_counts, policy, raw[policy])
        raw_counts['mapping_changed_images'] += raw['original']['mapping'] != raw['radius_reference']['mapping']
        for target in raw['radius_reference']['no_eligible_candidate_target_ids']:
            missing_raw.append(dict(sample_key=key, group=domain, target_id=target))
        pairs = previous['stable_proposal']['proposed_pairs']
        proposed = {p['fragment_slot'] for p in pairs}
        held = {p['fragment_slot'] for p in previous['excluded_pairs']}
        if len(proposed) != len(pairs) or proposed & held or not proposed | held <= set(raw_slots):
            raise ValueError('Invalid stable/held slot partition')
        if proposed & {e['slot'] for e in raw['radius_reference']['eligible_edges']}:
            raise ValueError('Stable proposal contains eligible candidate')
        partition.update(stable_pairs=len(pairs), excluded_pairs=len(previous['excluded_pairs']))
        result = dict(sample_key=key, group=domain, canonical_smiles=row['canonical_smiles'],
                      raw=raw, phases={}, training_authorized=False, requires_review=True)
        for phase, parent, prefix in prior.PHASES:
            path = parent / (prefix + '--' + key + '.json')
            record = inputs.json(path, bindings[str(path)])
            if record['key'] != geometry['key'] or record['group'] != domain or record['sample_key'] != key:
                raise ValueError('Prediction identity differs')
            chosen = check_selection(geometry, record)
            removed = sorted(set(chosen) & proposed)
            check = reference.compare_removal(geometry, chosen, removed, layers)
            check['held_sensitive_selected_slots'] = sorted(set(chosen) & held)
            old_check = previous['phase_checks'][phase]
            if removed != old_check['removed_fragment_slots']:
                raise ValueError('Saved stable removal differs')
            for when in ('before', 'after'):
                original = check['original'][when]
                if not serialized_equal(groups.coverage(geometry,
                        chosen if when == 'before' else set(chosen) - set(removed), layers)['mapping'], original['mapping']):
                    raise ValueError('Independent original mapping replay differs')
                if any(original[k] != v for k, v in old_check[when].items()):
                    raise ValueError('Historical stable counterfactual did not reproduce')
            if any(check['original']['before'][k] != record['metrics'][k]
                   for k in ('matched_target_ids', 'reachable_bond_ids', 'unmatched_selected')):
                raise ValueError('Original saved prediction metrics did not reproduce')
            check['historical_protections_preserved'] = old_check['all_protections_preserved']
            totals = phase_totals[phase][domain]
            totals.update(images=1, removed_slots=len(removed), held_sensitive_selected_slots=len(check['held_sensitive_selected_slots']))
            for policy in ('original', 'radius_reference'):
                for when in ('before', 'after'):
                    count_coverage(totals, policy + '_' + when, check[policy][when])
                delta = check[policy]['removal_difference']
                for label in ('lost_target_ids', 'lost_bond_ids', 'lost_relations'):
                    totals[policy + '_removal_' + label] += len(delta[label])
                if delta['lost_target_ids'] or delta['lost_bond_ids'] or delta['lost_relations']:
                    regressions.append(dict(sample_key=key, group=domain, phase=phase, policy=policy, **delta))
            for when in ('before', 'after'):
                delta = check['policy_difference_' + when]
                if delta['lost_relations'] or delta['gained_relations']:
                    differences.append(dict(sample_key=key, group=domain, phase=phase, when=when, **delta))
            totals['frozen_eligible_graph_changed_images'] += not check['frozen_eligible_graph_identical']
            totals['numerical_distance_changed_images'] += check['numerical_check']['max_abs_distance_change_px'] > 0
            totals['numerical_eligibility_changed_edges'] += len(check['numerical_check']['eligibility_changes'])
            numeric_delta = check['numerical_check']['difference_from_frozen']
            totals['numerical_mapping_changed_images'] += bool(numeric_delta['lost_relations'] or numeric_delta['gained_relations'])
            result['phases'][phase] = check
        path = OUTPUT / (key + '.json')
        prior.write(path, result)
        cases[path.name] = sha256(path.read_bytes()).hexdigest()
    if partition != dict(stable_pairs=502, excluded_pairs=23):
        raise ValueError('Fixed stable/held pair counts differ')
    for path, digest in list(inputs.hashes.items()):
        inputs.read(path, digest)
    if torch.cuda.is_initialized():
        raise ValueError('Unexpected CUDA initialization')
    prior.write(OUTPUT / 'report.json', dict(schema_version='selector-radius-reference-0.1',
        input_sha256=inputs.hashes, case_sha256=cases, images=len(rows), saved_phase_checks=len(rows)*len(prior.PHASES),
        original_replay_exact=True, raw_totals=raw_totals, phase_totals=phase_totals,
        policy_differences=differences, removal_regressions=regressions,
        original_protection_regressions=source['protection_regressions'], partition=partition,
        raw_no_eligible_candidate_nodes=missing_raw,
        versions=dict(torch=torch.__version__, numpy=np.__version__, scipy=scipy.__version__),
        training_authorized=False, requires_review=True, training_performed=False,
        model_forwards=0, optimizer_steps=0, gpu_used=False, validation_sample_artifacts_read=False,
        new_identity_queries=0, threshold_or_labels_changed=False, default_changed=False,
        prediction_artifacts_changed=False, production_matcher_changed=False,
        limitations=['GT-assisted diagnostic, not model predictions, accuracy improvement, or training admission',
            'Original metrics remain authoritative; all original stable-subset failures retained',
            'Reachable bonds mean endpoint coverage only, not correct predicted bonds',
            'Matching optimizes valid cardinality/distance, not chemical identity or bond count',
            'No custom tie rule: target IDs, relations, and bonds checked separately',
            'No missing candidate is generated; rejected group objective remains rejected']))
    print(json.dumps(dict(images=len(rows), policy_differences=len(differences),
        removal_regressions=regressions, raw_no_candidate_nodes=len(missing_raw)), ensure_ascii=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        worker()
        return 0
    preflight(prior.Inputs())
    receipt = runtime.run_budgeted([sys.executable, '-m', 'tools.audit_selector_radius_reference', '--worker'],
        output=OUTPUT, budget=runtime.ResourceBudget(120))
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
