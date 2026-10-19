"""One fixed CPU generator/selector/bbox diagnostic, never a training launcher."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

import torch

from .interface_diagnostic import interface_delta, structure_trace, tensor_tree_digest, write_json
from .runtime import ResourceBudget, run_budgeted

RUN = Path('outputs/experiments/chemvector-selected-matching-trial-20260915')
RELEASE = Path('outputs/datasets/chemvector-expanded-release-20260915')
PRIOR = Path('outputs/audits/chemvector-freeze-regression-probe-20260915/worker.log')
PRIOR_SHA = '6fb9c41c9cf42f139d4a1706f5d76c0ee248dd720014f905e350b024c82a0b51'
OUTPUT = Path('outputs/audits/chemvector-upstream-interfaces-20260918')
NAMES = ('initial', 'last')
# generator, selector, pair, bbox; role/OCR are always the original frozen ones.
ARMS = {
    'initial': ('initial', 'initial', 'initial', 'initial'),
    'initial_generator_last_set': ('initial', 'last', 'initial', 'initial'),
    'last_generator_initial_set': ('last', 'initial', 'initial', 'initial'),
    'last_upstream_initial_downstream': ('last', 'last', 'initial', 'initial'),
    'last_initial_bbox': ('last', 'last', 'last', 'initial'),
    'last': ('last', 'last', 'last', 'last'),
}


def emit(event, **record):
    print(json.dumps({'event': event, **record}, allow_nan=False), flush=True)


def row_index(manifest):
    rows = {r['source'] + ':' + r['sample_key']: r for r in manifest['samples']}
    if len(rows) != len(manifest['samples']):
        raise ValueError('Duplicate released sample key')
    return rows


def inherited_selection(prior, manifest):
    """Keep the historical sample order and controls; never reselect by new results."""
    rows = row_index(manifest)
    keys, cases = prior['training_keys'], prior['cases']
    if (len(keys) != 12 or len(set(keys)) != 12 or len(cases) != 12
            or len({r['key'] for r in cases}) != 12):
        raise ValueError('Expected the original 12 training and 12 development cases')
    if any(k not in rows or rows[k]['split'] != 'train' for k in keys):
        raise ValueError('Training coverage requires released training samples')
    if Counter(rows[k]['domain'] for k in keys) != {'full_molecule': 8, 'curriculum': 4}:
        raise ValueError('Historical training domains changed')
    if any(c['key'] not in rows or rows[c['key']]['split'] != 'validation'
           or rows[c['key']]['domain'] != 'full_molecule' for c in cases):
        raise ValueError('Development counterfactuals require full-molecule validation')
    if len({rows[c['key']]['molecule_id'] for c in cases}) != 12:
        raise ValueError('Historical development molecules are not distinct')
    return {'training_keys': list(keys), 'cases': [dict(c) for c in cases],
            'arms': ARMS, 'planned_generator_forwards': 48,
            'planned_development_predictions': 72, 'cpu_wall_seconds': 300,
            'gpu_used': False, 'optimizer_steps': 0,
            'limits': ['No new selection, threshold search, automatic retries or deployment',
                       'Development PNG/SDF has no released pixel-aligned atom ground truth',
                       'Coverage uses only same-render training diagnostic labels',
                       'Historical error-enriched cases are not a new accuracy estimate',
                       'Generator swaps combine vision/atom/bond; not individual causal attribution']}


@torch.no_grad()
def choose_from_cache(cache, selector, threshold):
    """Selector interventions consume exactly the same predicted input tensors."""
    atoms, bonds = cache['raw'], cache['bonds']
    before = tensor_tree_digest(cache)
    output = selector(atoms['atom_features'], atoms['atom_geometry'], atoms['atom_scores'], atoms['atom_mask'],
        bonds['bond_features'], bonds['bond_endpoints'], bonds['bond_types'], bonds['bond_scores'], bonds['bond_mask'])
    if not bool(torch.isfinite(output['atom_presence_logits']).all()):
        raise ValueError('Non-finite selector predictions')
    selected = {**atoms, 'atom_mask': atoms['atom_mask'] & (output['atom_presence_logits'] >= threshold)}
    if tensor_tree_digest(cache) != before:
        raise ValueError('Selector changed cached predictions')
    return selected, output


def candidate_trace(raw, selected, output):
    indices = raw['atom_mask'][0].nonzero().flatten().tolist()
    return {'raw_count': len(indices), 'selected_count': int(selected['atom_mask'].sum()),
            'candidates': [{'candidate': i, 'geometry': raw['atom_geometry'][0, i].tolist(),
                'proposal_score': float(raw['atom_scores'][0, i]),
                'selection_logit': float(output['atom_presence_logits'][0, i]),
                'selected': bool(selected['atom_mask'][0, i])} for i in indices]}


@torch.no_grad()
def labeled_coverage(raw, selected, batch, config):
    """Score predictions after inference. Not callable on original-image dev truth."""
    from chemvector.train_direct_atom_pair_bond import match_atom_targets, build_pair_targets
    from chemvector.direct_atom_pair_bond import enumerate_unordered_atom_pairs
    if batch.get('purpose') != 'diagnostic' or batch.get('training_authorized') is not False:
        raise ValueError('Coverage requires diagnostic-only aligned targets')
    if batch['atom_mask'].shape[0] != 1:
        raise ValueError('Coverage is per image')
    _, raw_matched = match_atom_targets(raw, batch, config)
    mapping, matched = match_atom_targets(selected, batch, config)
    pairs, indices = enumerate_unordered_atom_pairs(selected['atom_mask'])
    targets = build_pair_targets(pairs, indices, mapping, batch, selected)
    na, nb = int(batch['atom_mask'].sum()), int(batch['bond_mask'].sum())
    raw_ids = raw['atom_mask'][0].nonzero().flatten()
    distance = torch.cdist(raw['atom_geometry'][0, raw_ids, :2].float(),
                           batch['atom_anchors'][0, :na].float()) * config.image_size
    near = distance <= config.atom_match_radius_px
    selected_local = selected['atom_mask'][0, raw_ids]
    details = [{'target': i, 'near_raw_ids': raw_ids[near[:, i]].tolist(),
                'near_selected_ids': raw_ids[near[:, i] & selected_local].tolist(),
                'raw_assignment_matched': bool(raw_matched[0, i]),
                'selected_assignment_matched': bool(matched[0, i])} for i in range(na)]
    return {'target_atoms': na, 'target_bonds': nb, 'raw_matched': int(raw_matched.sum()),
            'selected_matched': int(matched.sum()), 'selected_atoms': int(selected['atom_mask'].sum()),
            'unmatched_selected': int(selected['atom_mask'].sum()) - int(matched.sum()),
            'reachable_positive_pairs': int(targets['presence'].sum()),
            'target_details': details,
            'assignment': 'existing Hungarian distance matching, then fixed radius cutoff',
            'raw_no_nearby': sum(not d['near_raw_ids'] for d in details),
            'selected_no_nearby': sum(not d['near_selected_ids'] for d in details)}


@torch.no_grad()
def cache_fronts(models, pixels, config, threshold):
    from .continuation import structure_forward
    caches, selections, traces = {}, {}, {}
    for generator, model in models.items():
        forward = structure_forward(model, pixels, config, threshold)
        cache = {'features': forward[0], 'raw': forward[3], 'bonds': forward[4]}
        caches[generator] = cache
        for selector in NAMES:
            selected, output = choose_from_cache(cache, models[selector].set_head, threshold)
            if selector == generator and tensor_tree_digest(selected) != tensor_tree_digest(forward[6]):
                raise ValueError('Cached selector differs from original forward')
            selections[(generator, selector)] = selected
            traces[generator + '_' + selector] = candidate_trace(cache['raw'], selected, output)
    return caches, selections, traces


def score_prediction(predicted, expected):
    from .validation_truth import canonical_molblock
    canonical = canonical_molblock(predicted['molblock'])
    return {**predicted, 'predicted_smiles': canonical, 'expected_smiles': expected,
            'valid': canonical is not None, 'exact': canonical is not None and canonical == expected}


def check_saved_replay(result, saved):
    fields = ('predicted_smiles', 'expected_smiles', 'valid', 'exact', 'fallback')
    differences = {k: {'saved': saved[k], 'cpu': result[k]} for k in fields if saved[k] != result[k]}
    if differences:
        raise ValueError('Saved baseline replay changed: ' + json.dumps(differences))


def summarize_cases(cases):
    summaries = {}
    for group in ('all', 'currently_broken', 'currently_stable_correct', 'currently_rescued', 'currently_stable_wrong'):
        def include(c):
            a, b = c['arms']['initial']['exact'], c['arms']['last']['exact']
            return {'all': True, 'currently_broken': a and not b, 'currently_stable_correct': a and b,
                    'currently_rescued': not a and b, 'currently_stable_wrong': not a and not b}[group]
        chosen = [c for c in cases if include(c)]
        summaries[group] = {'images': len(chosen), 'arms': {name: {
            **{k: sum(bool(c['arms'][name][k]) for c in chosen) for k in ('exact', 'valid', 'fallback')},
            'rescued_vs_last': sum(c['arms'][name]['exact'] and not c['arms']['last']['exact'] for c in chosen),
            'broken_vs_last': sum(not c['arms'][name]['exact'] and c['arms']['last']['exact'] for c in chosen)}
            for name in ARMS}}
    return summaries


def audit(output):
    from chemvector.atom_bond_scene import AtomBondSceneConfig
    from . import evaluation, continuation
    from .bundle import load_bundle
    from .checkpoints import file_digest, verify_sources
    from .data import CurriculumCollator, load_sample
    from .gradient_preflight import state_digest
    from .release import RecordLoader, verify_release
    from .pixels import image_tensor
    from .validation_truth import expected_identity

    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('Diagnostic requires an externally supervised CPU-only process')
    torch.set_num_threads(2)
    torch.manual_seed(20260918)
    output = Path(output)
    if not output.is_dir() or any((output / n).exists() for n in ('plan.json', 'report.json')):
        raise ValueError('Fresh supervised output required')
    if file_digest(PRIOR) != PRIOR_SHA:
        raise ValueError('Historical fixed selection changed')
    old_plan = json.loads(PRIOR.read_text().splitlines()[0])
    manifest, _, bindings = verify_release(RELEASE / 'manifest.json', RELEASE / 'admission.json')
    run = json.loads((RUN / 'run.json').read_text())
    previous = json.loads((RUN / 'report.json').read_text())
    if run['bindings'] != bindings or previous['bindings'] != bindings:
        raise ValueError('Run/release binding mismatch')
    if Path(previous['selected']['path']).resolve() != (RUN / 'initial.pt').resolve():
        raise ValueError('Expected the failed trial with original initial retained')
    saved = {n: {r['key']: r for r in json.loads((RUN / f'validation-{s:06d}.json').read_text())['predictions']}
             for n, s in (('initial', 0), ('last', previous['optimizer_steps']))}
    plan = inherited_selection(old_plan, manifest)
    rows = row_index(manifest)
    files = [PRIOR, RUN / 'run.json', RUN / 'report.json', RUN / 'validation-000000.json',
             RUN / f"validation-{previous['optimizer_steps']:06d}.json", RELEASE / 'manifest.json',
             RELEASE / 'admission.json', Path('frontend/inference_profiles.json')]
    sources = [Path(__file__), Path(evaluation.__file__), Path(continuation.__file__),
               Path(__file__).with_name('interface_diagnostic.py')]
    plan['input_sha256'] = {str(p): file_digest(p) for p in files + sources}
    for key in plan['training_keys']:
        row = rows[key]
        path = Path(manifest['sources'][row['source']]) / 'samples' / (row['sample_key'] + '.meta.json')
        if file_digest(path) != row['meta_sha256']:
            raise ValueError('Released training metadata changed')
        plan['input_sha256'][str(path)] = row['meta_sha256']
    plan['bundles'] = {'initial': previous['selected'], 'last': previous['last']}
    plan['thresholds'] = {k: v for k, v in run['provenance'].items() if k.endswith('_threshold')}
    write_json(output / 'plan.json', plan)  # Before model/image access; no sample search.
    emit('plan_fixed', training_cases=12, development_cases=12, arms=list(ARMS))
    models, states, config = {}, {}, None
    for name, binding in plan['bundles'].items():
        if Path(binding['path']).resolve() != (RUN / (name + '.pt')).resolve():
            raise ValueError('Unexpected model path')
        model, proposal, payload = load_bundle(binding['path'], expected_sha256=binding['sha256'])
        if (payload['provenance'] != run['provenance'] or model.update_recipe != 'freeze-role-v1'
                or payload['progress']['bindings'] != bindings
                or payload['progress']['optimizer_steps'] != (0 if name == 'initial' else previous['optimizer_steps'])):
            raise ValueError('Unexpected model provenance or recipe')
        if config is not None and proposal != config:
            raise ValueError('Proposal configuration changed')
        config = proposal
        models[name] = model.eval().requires_grad_(False)
        states[name] = state_digest(model)
    if state_digest(models['initial'].role) != state_digest(models['last'].role):
        raise ValueError('Role was not frozen')
    predictor = evaluation.FrozenOCRPredictor(run['provenance'], 'cpu')
    states['ocr'] = state_digest(predictor.ocr)
    loader = RecordLoader(manifest)
    collator = CurriculumCollator(AtomBondSceneConfig(atom_queries=config.maximum_atoms,
        bond_queries=config.maximum_bonds), purpose='diagnostic')
    training, cases = [], []
    with torch.inference_mode():
        for key in plan['training_keys']:
            row = rows[key]
            sample = load_sample(manifest['sources'][row['source']], row['sample_key'], purpose='diagnostic')
            if (sample['meta']['split'] != 'train' or sample['meta']['external_overlap_status'] != 'checked'
                    or sample['meta']['molecule_id'] != row['molecule_id']
                    or sample['graph']['canonical_smiles'] != row['canonical_smiles']):
                raise ValueError('Training diagnostic differs from admitted row')
            batch = collator([sample])
            caches, selections, traces = cache_fronts(models, batch['pixel_values'], config,
                                                     run['provenance']['atom_threshold'])
            record = {'key': key, 'domain': row['domain'], 'candidates': traces, 'coverage': {}}
            for (g, s), selected in selections.items():
                record['coverage'][g + '_' + s] = labeled_coverage(caches[g]['raw'], selected, batch, config)
            write_json(output / ('training-' + row['sample_key'] + '.json'), record)
            training.append(record)
            emit('training_coverage', completed=len(training), key=key,
                 selected_matched={a: v['selected_matched'] for a, v in record['coverage'].items()})
        for selection in plan['cases']:
            key, row = selection['key'], rows[selection['key']]
            sample = loader(manifest, row)
            if sample.get('purpose') != 'diagnostic' or 'graph' in sample or 'target_molblock' not in sample:
                raise ValueError('Development audit requires original PNG/SDF, not aligned target tensors')
            pixels = image_tensor(sample['image']).unsqueeze(0)
            caches, selections, traces = cache_fronts(models, pixels, config, run['provenance']['atom_threshold'])
            hashes = {g: tensor_tree_digest(c) for g, c in caches.items()}
            structures = {}
            case = {**selection, 'candidates': traces, 'structures': {}, 'arms': {}}
            expected = expected_identity(sample)
            for arm, (g, s, p, b) in ARMS.items():
                structure_key = (g, s, p)
                label = '_'.join(structure_key)
                if structure_key not in structures:
                    structures[structure_key] = predictor.from_predicted_atoms(models[p].pair,
                        caches[g]['features'], selections[(g, s)])
                    case['structures'][label] = structure_trace(structures[structure_key])
                structure = structures[structure_key]
                trace = {}
                predicted = predictor.from_structure(structure, models['initial'].role, models[b].bbox,
                                                     pixels, sample['image'], trace=trace)
                result = score_prediction(predicted, expected)
                result.update(trace=trace, structure_key=label,
                              structure_sha256=case['structures'][label]['sha256'])
                case['arms'][arm] = result
                if tensor_tree_digest(structure) != result['structure_sha256']:
                    raise ValueError('Text path mutated cached structure')
                if arm in NAMES:
                    check_saved_replay(result, saved[arm][key])
            if hashes != {g: tensor_tree_digest(c) for g, c in caches.items()}:
                raise ValueError('Downstream path mutated generator cache')
            # Both arms share the identical cached final structure, role and OCR.
            a, b = case['arms']['last'], case['arms']['last_initial_bbox']
            if a['structure_sha256'] != b['structure_sha256'] or a['trace']['roles'] != b['trace']['roles']:
                raise ValueError('BBox intervention changed candidates or role decisions')
            case['bbox_delta'] = interface_delta(a['trace'], b['trace'])
            write_json(output / (row['sample_key'] + '.json'), case)
            cases.append(case)
            emit('development_case', completed=len(cases), key=key,
                 exact={a: v['exact'] for a, v in case['arms'].items()})
    after = {n: state_digest(m) for n, m in {**models, 'ocr': predictor.ocr}.items()}
    if states != after or torch.cuda.is_initialized():
        raise ValueError('Diagnostic changed state or initialized CUDA')
    for path, digest in plan['input_sha256'].items():
        if file_digest(path) != digest:
            raise ValueError('Bound input or source changed during audit')
    for binding in plan['bundles'].values():
        if file_digest(binding['path']) != binding['sha256']:
            raise ValueError('Checkpoint file changed')
    verify_sources(run['provenance']['checkpoints'])
    totals = {}
    for arm in training[0]['coverage']:
        totals[arm] = {k: sum(r['coverage'][arm][k] for r in training) for k in (
            'target_atoms', 'target_bonds', 'raw_matched', 'selected_matched', 'selected_atoms',
            'unmatched_selected', 'reachable_positive_pairs', 'raw_no_nearby', 'selected_no_nearby')}
    result = {'schema_version': 'chemvector-upstream-interface-diagnostic-0.1',
        'plan_sha256': file_digest(output / 'plan.json'), 'training_coverage': totals,
        'development_summary': summarize_cases(cases), 'states_before': states, 'states_after': after,
        'saved_replays_matched': 24, 'cached_inputs_unchanged': True,
        'gpu_used': False, 'optimizer_steps': 0, 'training_performed': False, 'deployed': False,
        'limits': plan['limits']}
    write_json(output / 'report.json', result)
    emit('completed', **result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        audit(OUTPUT)
        return 0
    receipt = run_budgeted([sys.executable, '-m', 'chemvector.training.curriculum.upstream_diagnostic', '--worker'],
                           output=OUTPUT, budget=ResourceBudget(300))
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
