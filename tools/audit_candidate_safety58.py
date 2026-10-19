"""Fixed58 negative-control audit: unchanged queue, at most ten CPU replays."""

import argparse
from collections import Counter
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import sys

from PIL import Image
from rdkit import Chem
import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.experiments import candidate_safety, selector_recheck, single_candidate_replay
from chemvector.training.curriculum import bundle, runtime
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.data import CurriculumCollator
from chemvector.training.curriculum.evaluation import FrozenOCRPredictor
from chemvector.training.curriculum.gradient_preflight import state_digest
from chemvector.training.curriculum.interface_diagnostic import tensor_tree_digest
from chemvector.training.curriculum.pixels import image_tensor
from chemvector.training.curriculum.validation_truth import canonical_molblock, expected_identity
from tools.audit_single_candidate_replay import saved_prediction
from tools.diagnose_pair_upstream2 import INITIAL, INITIAL_SHA, INITIAL_STATE, NH, selected_parity, write
from tools.pair_local_repair import components
from tools.preview_pair_false_edges import ROOT, TRIAL, read

OUTPUT = ROOT / 'outputs/audits/chemvector-candidate-safety58-20260921'
PRIOR = ROOT / 'outputs/audits/chemvector-single-candidate-replay-20260921'
REVIEW = ROOT / 'outputs/audits/chemvector-upstream-recheck2-20260921'


def baseline_parity(historical, record, selected, raw):
    parity = selected_parity(raw, selected, historical['trace'])
    if 'scene_before_fallback' not in record['trace']:
        return dict(passed=False, reason='cpu_scene_unavailable', **parity)
    delta = single_candidate_replay.compare_scenes(historical['trace'], record['trace'])
    texts = lambda t: {r['candidate']: r['text'] for r in t['ocr']}
    checks = dict(selected=parity['selected_slots_equal'], anchors=parity['selected_q1023_anchors_equal'],
        scene=not any(delta.values()), ocr=texts(historical['trace']) == texts(record['trace']),
        identity=canonical_molblock(record['prediction']['molblock']) == historical['metrics']['predicted_smiles'],
        fallback=record['prediction']['fallback'] == historical['prediction']['fallback'])
    return dict(passed=all(checks.values()), checks=checks, scene_delta=delta)


def intervention_delta(before_record, after_record, before, after, slot):
    delta = (single_candidate_replay.compare_scenes(before_record['trace'], after_record['trace'])
             if 'scene_before_fallback' in after_record['trace'] else dict(scene_unavailable=True))
    old_pairs = {tuple(e['pair']) for e in before['scene_edges'] or []}
    new_edges = [e for e in after['scene_edges'] or [] if tuple(e['pair']) not in old_pairs]
    old_texts = {r['candidate']: r['text'] for r in before_record['trace'].get('ocr', [])}
    new_texts = {r['candidate']: r['text'] for r in after_record['trace'].get('ocr', [])}
    return dict(**delta, added_assignment=after['assignment'][slot],
        old_text_changes=[dict(slot=s, before=t, after=new_texts.get(s))
                          for s, t in old_texts.items() if new_texts.get(s) != t],
        lost_matched_targets=sorted(set(before['matched_targets']) - set(after['matched_targets'])),
        gained_matched_targets=sorted(set(after['matched_targets']) - set(before['matched_targets'])),
        changed_old_assignments=[s for s, t in before['assignment'].items() if after['assignment'].get(s) != t],
        lost_correct_raw_typed_bonds=sorted(set(before['correct_raw_typed_bonds']) - set(after['correct_raw_typed_bonds'])),
        gained_correct_raw_typed_bonds=sorted(set(after['correct_raw_typed_bonds']) - set(before['correct_raw_typed_bonds'])),
        newly_output_scene_edges=new_edges,
        new_edge_status_counts=dict(Counter(e['status'] for e in new_edges)),
        valid_to_invalid=before['sdf_valid'] and not after['sdf_valid'],
        introduced_fallback=not before['fallback'] and after['fallback'],
        broke_exact=before['sdf_exact'] and not after['sdf_exact'],
        rescued_exact=not before['sdf_exact'] and after['sdf_exact'],
        baseline_all_targets_matched=len(before['matched_targets']) == before['coverage']['target_atoms'],
        components_before=components(before_record['trace']['scene_before_fallback']),
        components_after=(components(after_record['trace']['scene_before_fallback'])
                          if 'scene_before_fallback' in after_record['trace'] else None))


def worker():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized() or not OUTPUT.is_dir():
        raise ValueError('Fresh CPU-only supervised audit required')
    torch.set_num_threads(2)
    torch.manual_seed(20260921)
    hashes = {}
    def data(path, digest=None):
        return json.loads(read(path, hashes, digest))
    prior = data(PRIOR / 'report.json', '87dd0a762d0316f0e455b5d223e91cab3541bacc7fbeb0ec382b189d3bc29933')
    review = data(REVIEW / 'report.json', '0fe9e6156821c07f397b48004faf498667a3a4bfca4bffeefea2326d5a2279d7')
    manifest = data(ROOT / 'outputs/datasets/chemvector-selector-retention-fit-20260919/manifest.json',
                    'f5e3a7163eef5c49118ad78dcf05778f4ad889dc9b6389097ef20896b79e2111')
    post = data(TRIAL / 'postcheck.json', 'b09b82c98fe5280de0d79a6cca858a2854398fa66dca1d29205997f4cece1f54')
    trial = data(TRIAL / 'report.json', 'd12d93e5e012f7566f4cc3123abf411b67f49c44ecb898404034105b3c4603eb')
    rows = sorted(manifest['training_samples'], key=lambda r: r['sample_key'])
    if len(rows) != 58 or len({r['sample_key'] for r in rows}) != 58 or any(r['split'] != 'train' for r in rows):
        raise ValueError('Fixed58 training scope changed')
    historical, receipts, gates = {}, {}, {}
    for row in rows:
        key = row['sample_key']
        historical[key] = data(TRIAL / (name := f'check-000000--{key}.json'), post['artifact_sha256'][name])
        receipts[key] = data(TRIAL / (name := f'cache--{key}.json'), post['artifact_sha256'][name])
        if historical[key]['metrics']['key'] != row['source'] + ':' + key:
            raise ValueError('Saved sample identity differs')
        gates[key] = candidate_safety.structural_prefilter(historical[key]['trace'])
    replay_rows = [r for r in rows if gates[r['sample_key']]['status'] == 'needs_raw_replay']
    if len(replay_rows) != 10:
        raise ValueError('Pre-inspected ten-case raw-replay scope changed')
    reference_code = receipts[NH]['implementation_sha256']
    if any(r['implementation_sha256'] != reference_code for r in receipts.values()):
        raise ValueError('Mixed saved baseline implementations')
    for relative, digest in reference_code.items():
        read(ROOT / relative, hashes, digest)
    for module in (single_candidate_replay, selector_recheck):
        path = Path(module.__file__).resolve()
        digest = prior['input_sha256'].get(str(path)) or review['input_sha256'].get(str(path))
        if not digest:
            raise ValueError('Prior policy implementation must be pinned')
        read(path, hashes, digest)
    for relative in ('tools/audit_single_candidate_replay.py', 'tools/diagnose_pair_upstream2.py',
                     'tools/preview_pair_false_edges.py', 'tools/pair_local_repair.py'):
        read(ROOT / relative, hashes, prior['input_sha256'][str(ROOT / relative)])
    for path in (Path(__file__), Path(candidate_safety.__file__), ROOT / 'tests/chemvector/test_candidate_safety.py'):
        read(path, hashes)
    metas = {}
    for row in replay_rows:
        key = row['sample_key']
        folder = Path(manifest['sources'][row['source']]) / 'samples'
        metas[key] = data(folder / (key + '.meta.json'), row['meta_sha256'])
    write(OUTPUT / 'plan.json', dict(schema_version='candidate-safety58-plan-0.1',
        sample_keys=[r['sample_key'] for r in rows], replay_keys=[r['sample_key'] for r in replay_rows],
        prefilters=gates, selector_config=selector_recheck.CONFIG,
        intervention='unchanged unique-request, unique-first-candidate, one mask bit only; otherwise abstain',
        baseline_parity='abstain on historical scene/selected/anchor/OCR/identity/fallback mismatch',
        maximum_upstream_forwards=10, maximum_pair_forwards=20, maximum_role_bbox_ocr_forwards_each=20,
        cpu_threads=2, wall_seconds=120, no_retries=True, alternate_candidates=False,
        all_predictions_saved_before_scoring=True, training=False, gpu_used=False, production_changed=False,
        target_component_scope='coverage only; never decides which images or candidates to replay',
        input_sha256=dict(hashes)))
    read(INITIAL, hashes, INITIAL_SHA)
    model, config, payload = bundle.load_bundle(INITIAL, expected_sha256=INITIAL_SHA)
    model.eval().requires_grad_(False)
    if (state_digest(model) != INITIAL_STATE or payload['provenance']['atom_threshold'] != -.25
            or payload['provenance']['pair_threshold'] != .85 or config.atom_match_radius_px != 6):
        raise ValueError('Frozen model or original scoring thresholds differ')
    ocr_ref = payload['provenance']['checkpoints']['ocr_frozen_not_loaded']
    read(ROOT / ocr_ref['path'], hashes, ocr_ref['sha256'])
    predictor = FrozenOCRPredictor(payload['provenance'], torch.device('cpu'))
    if state_digest(predictor.ocr) != trial['ocr_state_before']:
        raise ValueError('Frozen OCR state differs')
    calls = dict(pair=0, role=0, bbox=0, ocr=0)
    def counter(name):
        def hook(*_):
            calls[name] += 1
            if calls[name] > 20:
                raise ValueError('Fixed forward budget exhausted')
        return hook
    handles = [module.register_forward_pre_hook(counter(name)) for name, module in
               [('pair', model.pair), ('role', model.role), ('bbox', model.bbox), ('ocr', predictor.ocr)]]
    jobs, artifacts = {}, ['plan.json']
    for row in replay_rows:
        key = row['sample_key']; meta = metas[key]
        folder = Path(manifest['sources'][row['source']]) / 'samples'
        image = Image.open(io.BytesIO(read(folder / (key + '.png'), hashes, meta['sha256']['.png']))).convert('RGB')
        if image.size != (768, 768):
            raise ValueError('Original 768px PNG required')
        pixels = image_tensor(image)[None]
        with torch.no_grad():
            forward = structure_forward(model, pixels, config, -.25)
            raw, context, selected = forward[3], forward[5], forward[6]
            original_digest = tensor_tree_digest(selected)
            def predict(atoms):
                structure = predictor.from_predicted_atoms(model.pair, forward[0], atoms)
                trace = {}
                prediction = predictor.from_structure(structure, model.role, model.bbox, pixels, image, trace=trace)
                return saved_prediction(structure, prediction, trace)
            baseline = predict(selected)
            parity = baseline_parity(historical[key], baseline, selected, raw)
            proposals = [dict(slot=s, geometry=raw['atom_geometry'][0, s].tolist(),
                grid_id=int(raw['atom_indices'][0, s]), raw_active=bool(raw['atom_mask'][0, s]),
                keep_logit=float(context['atom_presence_logits'][0, s]), selected=bool(selected['atom_mask'][0, s]))
                for s in range(config.maximum_atoms)]
            queue, admitted, after = None, None, None
            decision = dict(slot=None, reason='historical_baseline_mismatch')
            if parity['passed']:
                _, edges = single_candidate_replay.scene_state(baseline['trace'])
                queue = selector_recheck.review_requests([dict(slot=p['slot'], xy=[v*768 for v in p['geometry'][:2]],
                    keep_logit=p['keep_logit'], selected=p['selected']) for p in proposals if p['raw_active']], list(edges))
                decision = candidate_safety.intervention_decision(queue)
                if decision['slot'] is not None:
                    admitted = single_candidate_replay.admit_one(raw, selected, queue)
                    after = predict(admitted)
            if tensor_tree_digest(selected) != original_digest:
                raise ValueError('Intervention mutated original prediction tensors')
        result = dict(sample_key=key, baseline=baseline, admitted=after, raw_proposals=proposals,
                      queue=queue, decision=decision, baseline_parity=parity)
        name = key + '-prediction-only.json'; write(OUTPUT / name, result); artifacts.append(name)
        def scoring_atoms(atoms):
            return {k: atoms[k].detach().clone() for k in ('atom_mask', 'atom_geometry', 'atom_indices', 'atom_scores')}
        jobs[key] = dict(result=result, baseline_atoms=scoring_atoms(selected),
            admitted_atoms=scoring_atoms(admitted) if admitted is not None else None,
            raw_atoms=scoring_atoms(raw), image=image, folder=folder)
        print(json.dumps(dict(prediction_saved=key, decision=decision, parity=parity['passed'])), flush=True)
    for handle in handles:
        handle.remove()
    # All queues, decisions and counterfactual predictions are fixed on disk now.
    cases = {}
    for row in replay_rows:
        key = row['sample_key']; job = jobs[key]; record = job['result']; meta = metas[key]
        sample = dict(key=key, image=job['image'], meta=meta, purpose='diagnostic')
        for field, suffix in [('graph', '.graph.json'), ('layers', '.layers.json'), ('text_records', '.text-records.json')]:
            sample[field] = data(job['folder'] / (key + suffix), meta['sha256'][suffix])
        batch = CurriculumCollator(AtomBondSceneConfig(atom_queries=config.maximum_atoms,
            bond_queries=config.maximum_bonds), purpose='diagnostic')([sample])
        before = candidate_safety.score_prediction(record['baseline'], job['baseline_atoms'], batch, config, expected_identity(sample))
        result = dict(decision=record['decision'], baseline_parity=record['baseline_parity'], baseline=before,
            queue_status=record['queue']['status'] if record['queue'] else None,
            proposed_request_count=len(record['queue']['requests']) if record['queue'] else 0,
            first_candidates=(candidate_safety.score_first_candidates(record['queue'], job['raw_atoms'], batch, before)
                              if record['queue'] else []))
        fields = dict(reachable_positive_pairs=before['coverage']['reachable_positive_pairs'],
            correct_typed_bond_ids=before['correct_raw_typed_bonds'], confirmed_false_positives=before['confirmed_false_positives'],
            unknown_present=before['unknown_present'], sdf_valid=before['sdf_valid'], sdf_exact=before['sdf_exact'])
        result['historical_metric_parity'] = {k: v == historical[key]['metrics'][k] for k, v in fields.items()}
        if record['admitted'] is not None:
            after = candidate_safety.score_prediction(record['admitted'], job['admitted_atoms'], batch, config, expected_identity(sample))
            result['admitted'] = after
            result['delta'] = intervention_delta(record['baseline'], record['admitted'], before, after, record['decision']['slot'])
        cases[key] = result
    outcomes = [r for r in cases.values() if 'delta' in r]
    exact_keys = [r['sample_key'] for r in rows if historical[r['sample_key']]['metrics']['sdf_exact']]
    component_counts = {}
    for row in rows:
        molecule = Chem.MolFromSmiles(row['canonical_smiles'])
        if molecule is None:
            raise ValueError('Cannot assess target-component coverage')
        component_counts[row['sample_key']] = len(Chem.GetMolFrags(molecule))
    summary = dict(scope_images=58, upstream_replayed=len(jobs),
        structural_prefilter_counts=dict(Counter(g['status'] for g in gates.values())),
        original_exact_images=len(exact_keys), original_exact_with_no_possible_request=sum(gates[k]['status'] != 'needs_raw_replay' for k in exact_keys),
        target_multifragment_images=sum(n > 1 for n in component_counts.values()),
        queue_status_counts=dict(Counter(r['queue_status'] for r in cases.values())),
        decision_counts=dict(Counter(r['decision']['reason'] for r in cases.values())),
        first_candidate_categories=dict(Counter(c['category'] for r in cases.values() for c in r['first_candidates'])),
        counterfactuals=len(outcomes), added_unmatched_candidates=sum(r['delta']['added_assignment'] < 0 for r in outcomes),
        all_targets_already_matched_interventions=sum(r['delta']['baseline_all_targets_matched'] for r in outcomes),
        new_scene_edge_status_counts=dict(sum((Counter(r['delta']['new_edge_status_counts']) for r in outcomes), Counter())),
        valid_to_invalid=sum(r['delta']['valid_to_invalid'] for r in outcomes),
        introduced_fallback=sum(r['delta']['introduced_fallback'] for r in outcomes),
        broken_exact=sum(r['delta']['broke_exact'] for r in outcomes), rescued_exact=sum(r['delta']['rescued_exact'] for r in outcomes),
        lost_old_edges=sum(len(r['delta'].get('removed_edges', [])) for r in outcomes),
        changed_old_edges=sum(len(r['delta'].get('changed_old_edges', [])) for r in outcomes),
        lost_correct_raw_typed_bonds=sum(len(r['delta']['lost_correct_raw_typed_bonds']) for r in outcomes),
        gained_correct_raw_typed_bonds=sum(len(r['delta']['gained_correct_raw_typed_bonds']) for r in outcomes),
        baseline_prediction_mismatches=sum(not r['baseline_parity']['passed'] for r in cases.values()),
        baseline_scoring_mismatches=sum(not all(r['historical_metric_parity'].values()) for r in cases.values()))
    if (state_digest(model) != INITIAL_STATE or state_digest(predictor.ocr) != trial['ocr_state_before']
            or torch.cuda.is_initialized() or any(p.grad is not None for m in (model, predictor.ocr) for p in m.parameters())):
        raise ValueError('Frozen CPU state invariant failed')
    for path, digest in list(hashes.items()):
        read(path, hashes, digest)
    write(OUTPUT / 'report.json', dict(schema_version='candidate-safety58-0.1', summary=summary, cases=cases,
        target_component_counts=component_counts, exact_control_keys=exact_keys,
        input_sha256=hashes, artifact_sha256={n: sha256((OUTPUT / n).read_bytes()).hexdigest() for n in artifacts},
        forwards=dict(upstream=len(jobs), **calls), weights_unchanged=True, original_outputs_changed=False,
        training_performed=False, optimizer_steps=0, gpu_used=False, validation_samples_read=False,
        sealed_test_read=False, new_identity_queries=0, production_admission=False,
        limitations=['Fixed training-side failures and controls, not independent generalization accuracy',
            'No true multi-fragment targets in this fixed scope; safety there remains untested',
            'No-orphan/short-graph abstention is not evidence that ambiguous requests are safe',
            'Multiple requests are outside the previous one-candidate intervention scope, not proven negative',
            'Outside target radius is positional failure, not proof a candidate is a bond stroke',
            'Unknown edges are not verified correct or confirmed false',
            'Matching-dependent counts and scene-preservation checks are reported separately']))
    print(json.dumps(summary, indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args(argv)
    if args.worker:
        worker(); return 0
    receipt = runtime.run_budgeted([sys.executable, '-m', 'tools.audit_candidate_safety58', '--worker'],
                                  output=OUTPUT, budget=runtime.ResourceBudget(120))
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
