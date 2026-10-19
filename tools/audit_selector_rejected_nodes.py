"""Fixed58 saved-score audit focused on the initial21 selector-rejected nodes."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

import torch

from chemvector.training.curriculum import runtime
from tools.audit_selector_stable_subset import Inputs, write
from tools import audit_selector_stable_subset as source
from tools import audit_selector_radius_reference as reference
from tools import selector_rejected_nodes as analysis
from tools import selector_node_groups as groups
from tools import selector_node_semantics as semantics

ROOT = source.ROOT
RADIUS_SHA = 'e01a31a88c0850d213b0cf747a056a86c0fd95e56eb896c26022390d9ec2e835'
OUTPUT = ROOT / 'outputs/audits/chemvector-selector-rejected21-20260920'


def worker():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('CPU saved-artifact audit only')
    if not OUTPUT.is_dir() or (OUTPUT / 'plan.json').exists():
        raise ValueError('Fresh one-shot output required')
    torch.set_num_threads(2)
    inputs = Inputs()
    prior = inputs.json(reference.OUTPUT / 'report.json', RADIUS_SHA)
    manifest = inputs.json(source.MANIFEST, reference.MANIFEST_SHA)
    rows = groups.training_rows(manifest)
    if Counter(r['fit_domain'] for r in rows) != {'witness':26, 'old_full_molecule':32}:
        raise ValueError('Original fixed58 only')
    for module in (Path(__file__), Path(analysis.__file__), Path(source.__file__), Path(reference.__file__),
            Path(groups.__file__), Path(semantics.__file__), Path(runtime.__file__),
            ROOT / 'frontend/inference_profiles.json'):
        inputs.read(module, prior['input_sha256'].get(str(module)))
    write(OUTPUT / 'plan.json', dict(schema_version='selector-rejected21-plan-0.1',
        user_request='继续', source_images=58, focused_initial_targets=21,
        phases=list(analysis.PHASES), threshold=-.25, radius_px=6, wall_budget_seconds=120,
        source='Saved initial/retention200/control200/retention400 scores on identical frozen candidates',
        purpose='Separate wrong node/fragment ordering, absolute rejection, and absent reviewed negative context',
        fixed58_training_sample_keys=[r['sample_key'] for r in rows],
        automatic_retries=0, model_forwards=0, optimizer_steps=0, gpu_devices=[],
        training_authorized=False, prediction_changes=False, new_supervision_labels=False,
        validation_sample_artifacts_read=False, sealed_test_read=False, new_identity_queries=0))
    counts, phase_totals, focused_totals = Counter(), defaultdict(Counter), defaultdict(Counter)
    focused, case_hashes = [], {}
    for row in rows:
        key = row['sample_key']
        radius_path = reference.OUTPUT / (key+'.json')
        radius_case = inputs.json(radius_path, prior['case_sha256'][radius_path.name])
        subset_path = source.OUTPUT / (key+'.json')
        subset = inputs.json(subset_path, prior['input_sha256'][str(subset_path)])
        geo_path = source.CONTROL / ('geometry--'+key+'.json')
        geometry = inputs.json(geo_path, prior['input_sha256'][str(geo_path)])
        base = Path(manifest['sources'][row['source']]) / 'samples'
        layers_path = base / (key+'.layers.json')
        layers = inputs.json(layers_path, prior['input_sha256'][str(layers_path)])
        if (geometry['key'] != row['source']+':'+key or layers['molecule_id'] != row['molecule_id']
                or subset['sample_key'] != key or radius_case['sample_key'] != key):
            raise ValueError('Original case identity differs')
        semantics.check_geometry(geometry, layers)
        records = {}
        for phase, parent, prefix in source.PHASES:
            path = parent / (prefix+'--'+key+'.json')
            record = inputs.json(path, prior['input_sha256'][str(path)])
            if (record['sample_key'] != key or record['key'] != geometry['key']
                    or record['group'] != row['fit_domain']):
                raise ValueError('Saved score identity differs')
            reference.check_selection(geometry, record)
            records[phase] = record
        result = analysis.inspect_case(geometry, subset, radius_case, records)
        result.update(sample_key=key, group=row['fit_domain'], canonical_smiles=row['canonical_smiles'])
        counts.update(images=1, all_targets=len(geometry['target_anchors']), reviewable_nodes=len(result['nodes']),
            raw_missing=len(result['raw_missing_target_ids']), selector_missing=len(result['focused_target_ids']),
            original_matcher_only_missing=len(result['original_matcher_only_missing_ids']),
            stable_pairs=len(subset['stable_proposal']['proposed_pairs']), held_pairs=len(subset['excluded_pairs']))
        for node in result['nodes']:
            counts['protected_alternative_slots'] += len(node['protected_alternative_slots'])
            for phase, state in node['phases'].items():
                stats = phase_totals[phase]
                stats.update(nodes=1, **{state['node_state']:1},
                    selected_stable_fragments=len(state['selected_fragment_slots']))
            if not node['initial_selector_missing']:
                continue
            counts['focused_'+row['fit_domain']] += 1
            counts['focused_visible_text' if node['visible_text'] else 'focused_implicit'] += 1
            counts['focused_with_stable_fragments'] += bool(node['stable_fragment_slots'])
            counts['focused_with_alternatives'] += bool(node['protected_alternative_slots'])
            focused.append(dict(sample_key=key, group=row['fit_domain'], canonical_smiles=row['canonical_smiles'], **node))
            for phase, state in node['phases'].items():
                stats = focused_totals[phase]
                stats.update(nodes=1, **{state['node_state']:1}, **{state['comparison']:1},
                    nodes_with_selected_own_fragment=bool(state['selected_fragment_slots']),
                    selected_stable_fragments=len(state['selected_fragment_slots']),
                    covered_without_selected_own_fragment=state['fixed_threshold_contract_pass'])
        path = OUTPUT / (key+'.json')
        write(path,result)
        case_hashes[path.name] = sha256(path.read_bytes()).hexdigest()
    counts['focused_images'] = len({n['sample_key'] for n in focused})
    if (counts['all_targets'] != 1090 or counts['raw_missing'] != 10 or counts['selector_missing'] != 21
            or counts['original_matcher_only_missing'] != 1 or counts['stable_pairs'] != 502
            or counts['held_pairs'] != 23 or counts['protected_alternative_slots'] != 39):
        raise ValueError('Frozen population/partition differs')
    write(OUTPUT/'focused21.json',dict(nodes=focused,training_authorized=False,requires_review=True))
    for path,digest in list(inputs.hashes.items()):
        inputs.read(path,digest)
    if torch.cuda.is_initialized():
        raise ValueError('Unexpected CUDA initialization')
    write(OUTPUT/'report.json',dict(schema_version='selector-rejected21-0.1',input_sha256=inputs.hashes,
        case_sha256=case_hashes,focused_sha256=sha256((OUTPUT/'focused21.json').read_bytes()).hexdigest(),
        counts=counts,all_reviewable_nodes_by_phase=phase_totals,focused21_by_phase=focused_totals,
        model_forwards=0,optimizer_steps=0,gpu_used=False,training_authorized=False,
        validation_sample_artifacts_read=False,sealed_test_read=False,new_identity_queries=0,
        thresholds_changed=False,default_changed=False,predictions_changed=False,
        limitations=['All groups and stable fragments are existing GT-assisted review proposals, not released labels',
            'Spatial node eligibility does not establish chemical identity, bond correctness, or SDF accuracy',
            'No local reviewed fragments does not imply global calibration is the sole error',
            'Observed ordering is descriptive, not proof of what features the network uses',
            'Historical retention400 is not an equal-step causal comparison with control200',
            'Rejected exactly-one loss remains rejected; no new fit or objective is authorized here']))
    print(json.dumps(dict(counts=counts,focused21_by_phase=focused_totals),ensure_ascii=False))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args.worker:
        worker()
        return 0
    receipt=runtime.run_budgeted([sys.executable,'-m','tools.audit_selector_rejected_nodes','--worker'],
        output=OUTPUT,budget=runtime.ResourceBudget(120))
    print(json.dumps(receipt,indent=2))
    return 0 if receipt['status']=='completed' else 1


if __name__=='__main__':
    raise SystemExit(main())
