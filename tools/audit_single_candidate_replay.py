"""One fixed NH training-image counterfactual; CPU, frozen initial, no retries."""

import argparse
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import sys

from PIL import Image
import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.bond_object import BOND_TYPES
from chemvector.experiments.single_candidate_replay import admit_one, compare_scenes, first_review_candidate, scene_state
from chemvector.training.curriculum import bundle, runtime
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.data import CurriculumCollator
from chemvector.training.curriculum.evaluation import FrozenOCRPredictor
from chemvector.training.curriculum.gradient_preflight import state_digest
from chemvector.training.curriculum.interface_diagnostic import tensor_tree_digest
from chemvector.training.curriculum.pair_cached_trial import make_cache, score_case
from chemvector.training.curriculum.pixels import image_tensor
from chemvector.training.curriculum.validation_truth import canonical_molblock, expected_identity
from tools.diagnose_pair_upstream2 import INITIAL, INITIAL_SHA, INITIAL_STATE, NH, selected_parity, write
from tools.pair_local_repair import components
from tools.preview_pair_false_edges import ROOT, TRIAL, read

OUTPUT = ROOT / 'outputs/audits/chemvector-single-candidate-replay-20260921'
REVIEW = ROOT / 'outputs/audits/chemvector-upstream-recheck2-20260921'
CAUSE = ROOT / 'outputs/audits/chemvector-pair-upstream-cause2-20260921'


def saved_prediction(structure, prediction, trace):
    return dict(prediction=prediction, trace=trace, pairs=structure['pairs'].tolist(),
        presence_logits=structure['output']['presence_logits'].tolist(),
        chemical_logits=structure['output']['chemical_logits'].tolist(),
        postprocessed_chemical_classes=structure['chemistry'].tolist())


def logit_delta(before, after):
    indices = {tuple(p): i for i, p in enumerate(after['pairs'])}
    rows = [indices[tuple(p)] for p in before['pairs']]
    return {k: float((torch.tensor(before[k]) - torch.tensor(after[k])[rows]).abs().max())
            for k in ('presence_logits', 'chemical_logits')}


def score(forward, batch, config, structure, prediction, trace, expected, slot):
    # Existing matching and unknown-pair contract; no oracle edits or rematching.
    cache = make_cache(forward, batch, config)
    metrics = score_case(cache, structure, prediction, trace, key=NH, expected=expected)
    t = cache['tensors']
    mapping = t['mapping'][0]
    slots = structure['atoms']['atom_mask'][0].nonzero().flatten().tolist()
    assigned = {s: int(mapping[s]) for s in slots}
    _, edges = scene_state(trace)
    pair_rows = {tuple(p): i for i, p in enumerate(structure['pairs'].tolist())}
    probabilities = structure['output']['presence_logits'].sigmoid()
    chemical_probabilities = structure['output']['chemical_logits'].softmax(-1)
    classes = chemical_probabilities.argmax(-1)
    incident = []
    for pair, i in pair_rows.items():
        if slot not in pair:
            continue
        incident.append(dict(pair=list(pair), presence_probability=float(probabilities[i]),
            raw_type=BOND_TYPES[int(classes[i])],
            raw_type_probabilities={kind: float(v) for kind, v in zip(BOND_TYPES, chemical_probabilities[i])},
            postprocessed_type=BOND_TYPES[int(structure['chemistry'][i])],
            scene_type=edges[pair]['chemical_type'] if pair in edges else None,
            matched_targets=[assigned[s] for s in pair], known=bool(t['known'][i]),
            target_bond_id=int(t['bond_slots'][i])))
    inverse = {tid: s for s, tid in assigned.items() if tid >= 0}
    bonds = []
    for i in range(int(batch['bond_mask'][0].sum())):
        refs = batch['bond_atom_refs'][0, i].tolist()
        pair = tuple(sorted(inverse[r] for r in refs)) if all(r in inverse for r in refs) else None
        row = pair_rows[pair] if pair is not None else None
        kind = BOND_TYPES[int(batch['bond_chemical'][0, i])]
        bonds.append(dict(bond_id=batch['bond_ids'][0][i], target_refs=refs, expected_type=kind,
            candidate_pair=list(pair) if pair is not None else None,
            presence_probability=float(probabilities[row]) if row is not None else None,
            raw_type=BOND_TYPES[int(classes[row])] if row is not None else None,
            raw_typed_correct=i in metrics['correct_typed_bond_ids'],
            scene_type=edges[pair]['chemical_type'] if pair in edges else None,
            scene_typed_correct=pair in edges and edges[pair]['chemical_type'] == kind))
    return dict(metrics=metrics, coverage=cache['coverage'], mapping=assigned,
        matched_target_ids=sorted(inverse), incident_pairs=incident, target_bonds=bonds,
        components=components(trace['scene_before_fallback']))


def worker():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized() or not OUTPUT.is_dir():
        raise ValueError('Fresh CPU-only supervised worker required')
    torch.set_num_threads(2)
    torch.manual_seed(20260921)
    hashes = {}
    def data(path, digest=None):
        return json.loads(read(path, hashes, digest))
    review_report = data(REVIEW / 'report.json', '0fe9e6156821c07f397b48004faf498667a3a4bfca4bffeefea2326d5a2279d7')
    queue = data(REVIEW / 'candidate-review-prediction-only.json',
                 review_report['artifact_sha256']['candidate-review-prediction-only.json'])
    slot = first_review_candidate(queue)
    snapshot = data(CAUSE / 'nh-prediction-snapshot.json',
                    '307d8ddf907a84cc9fbdeb8b224f070b3b58e25522474f868d696af0a6b0876a')
    manifest = data(ROOT / 'outputs/datasets/chemvector-selector-retention-fit-20260919/manifest.json',
                    'f5e3a7163eef5c49118ad78dcf05778f4ad889dc9b6389097ef20896b79e2111')
    post = data(TRIAL / 'postcheck.json', 'b09b82c98fe5280de0d79a6cca858a2854398fa66dca1d29205997f4cece1f54')
    trial = data(TRIAL / 'report.json', 'd12d93e5e012f7566f4cc3123abf411b67f49c44ecb898404034105b3c4603eb')
    historical = data(TRIAL / (name := f'check-000000--{NH}.json'), post['artifact_sha256'][name])
    receipt = data(TRIAL / (name := f'cache--{NH}.json'), post['artifact_sha256'][name])
    row = next(r for r in manifest['training_samples'] if r['sample_key'] == NH)
    if row['split'] != 'train':
        raise ValueError('Only the fixed training image is allowed')
    folder = Path(manifest['sources'][row['source']]) / 'samples'
    meta = data(folder / (NH + '.meta.json'), row['meta_sha256'])
    for path, digest in receipt['implementation_sha256'].items():
        read(ROOT / path, hashes, digest)
    for path in (Path(__file__), ROOT / 'chemvector/experiments/single_candidate_replay.py',
                 ROOT / 'tests/chemvector/test_single_candidate_replay.py',
                 ROOT / 'tools/diagnose_pair_upstream2.py', ROOT / 'tools/pair_local_repair.py',
                 ROOT / 'tools/preview_pair_false_edges.py'):
        read(path, hashes)
    write(OUTPUT / 'plan.json', dict(schema_version='single-candidate-replay-plan-0.1',
        sample_keys=[NH], source='fixed_training_only', selected_review_slot=slot,
        checkpoint=dict(path=str(INITIAL), sha256=INITIAL_SHA), cpu_threads=2, wall_seconds=120,
        maximum_upstream_forwards=1, maximum_pair_forwards=2, maximum_text_interfaces=2,
        arms=['unchanged_initial', 'initial_plus_first_review_candidate'],
        intervention='clone selected mask and add one existing raw-active candidate; all other tensors unchanged',
        snapshot_tolerances=dict(geometry=1e-6, proposal_score=1e-6, selector_logit=1e-5),
        alternate_candidates=False, threshold_changes=False, oracle_candidate_selection=False,
        predictions_saved_before_target_read=True, training=False, optimizer_steps=0,
        production_changed=False, input_sha256=dict(hashes)))
    image = Image.open(io.BytesIO(read(folder / (NH + '.png'), hashes, meta['sha256']['.png']))).convert('RGB')
    if image.size != (768, 768):
        raise ValueError('Original 768px image required')
    read(INITIAL, hashes, INITIAL_SHA)
    model, config, payload = bundle.load_bundle(INITIAL, expected_sha256=INITIAL_SHA)
    model.eval().requires_grad_(False)
    if state_digest(model) != INITIAL_STATE or payload['provenance']['atom_threshold'] != -.25:
        raise ValueError('Wrong frozen initial model or selector threshold')
    ocr_ref = payload['provenance']['checkpoints']['ocr_frozen_not_loaded']
    read(ROOT / ocr_ref['path'], hashes, ocr_ref['sha256'])
    predictor = FrozenOCRPredictor(payload['provenance'], torch.device('cpu'))
    if state_digest(predictor.ocr) != trial['ocr_state_before']:
        raise ValueError('Wrong frozen OCR')
    calls = dict(pair=0, role=0, bbox=0, ocr=0)
    def count(name):
        def hook(*_):
            calls[name] += 1
            if calls[name] > 2:
                raise ValueError('Forward budget exceeded')
        return hook
    handles = [module.register_forward_pre_hook(count(name)) for name, module in
               [('pair', model.pair), ('role', model.role), ('bbox', model.bbox), ('ocr', predictor.ocr)]]
    pixels = image_tensor(image)[None]
    with torch.no_grad():
        forward = structure_forward(model, pixels, config, payload['provenance']['atom_threshold'])
        raw, context, selected = forward[3], forward[5], forward[6]
        parity = selected_parity(raw, selected, historical['trace'])
        if not parity['selected_slots_equal'] or not parity['selected_q1023_anchors_equal']:
            raise ValueError('Historical selected candidate parity failed')
        if len(snapshot['proposals']) != config.maximum_atoms:
            raise ValueError('Raw candidate capacity changed')
        drift = dict(geometry=0., proposal_score=0., selector_logit=0.)
        for p in snapshot['proposals']:
            s = p['slot']
            if (int(raw['atom_indices'][0, s]) != p['grid_id']
                    or bool(raw['atom_mask'][0, s]) != p['raw_active']
                    or bool(selected['atom_mask'][0, s]) != p['selected']):
                raise ValueError('Saved raw candidate lineage changed')
            values = dict(geometry=float((raw['atom_geometry'][0, s] - torch.tensor(p['geometry'])).abs().max()),
                proposal_score=abs(float(raw['atom_scores'][0, s]) - p['proposal_score']),
                selector_logit=abs(float(context['atom_presence_logits'][0, s]) - p['selector_logit']))
            drift = {k: max(drift[k], v) for k, v in values.items()}
        if any(drift[k] > limit for k, limit in dict(geometry=1e-6, proposal_score=1e-6, selector_logit=1e-5).items()):
            raise ValueError('Saved CPU snapshot not reproduced')
        original_digest = tensor_tree_digest(selected)
        admitted = admit_one(raw, selected, queue)
        structures, predictions, traces, records = {}, {}, {}, {}
        for name, atoms in [('baseline', selected), ('admitted', admitted)]:
            structures[name] = predictor.from_predicted_atoms(model.pair, forward[0], atoms)
            traces[name] = {}
            predictions[name] = predictor.from_structure(structures[name], model.role, model.bbox,
                                                         pixels, image, trace=traces[name])
            records[name] = saved_prediction(structures[name], predictions[name], traces[name])
            write(OUTPUT / (name + '-prediction-only.json'), records[name])
            if name == 'baseline':
                historical_delta = compare_scenes(historical['trace'], traces[name])
                texts = lambda trace: {r['candidate']: r['text'] for r in trace['ocr']}
                if (any(historical_delta.values()) or texts(historical['trace']) != texts(traces[name])
                        or canonical_molblock(predictions[name]['molblock']) != historical['metrics']['predicted_smiles']
                        or predictions[name]['fallback'] != historical['prediction']['fallback']):
                    raise ValueError('Historical baseline scene/OCR/identity not reproduced; stop before intervention')
        if tensor_tree_digest(selected) != original_digest:
            raise ValueError('Original selected tensors changed')
    for handle in handles:
        handle.remove()
    # Only now read annotated targets. They never enter either prediction arm.
    sample = dict(key=NH, image=image, meta=meta, purpose='diagnostic')
    for field, suffix in [('graph', '.graph.json'), ('layers', '.layers.json'), ('text_records', '.text-records.json')]:
        sample[field] = data(folder / (NH + suffix), meta['sha256'][suffix])
    batch = CurriculumCollator(AtomBondSceneConfig(atom_queries=config.maximum_atoms,
        bond_queries=config.maximum_bonds), purpose='diagnostic')([sample])
    scores = {}
    for name, atoms in [('baseline', selected), ('admitted', admitted)]:
        replay_forward = (*forward[:6], atoms)
        scores[name] = score(replay_forward, batch, config, structures[name], predictions[name],
                             traces[name], expected_identity(sample), slot)
    before, after = scores['baseline'], scores['admitted']
    metric_keys = ('reachable_positive_pairs', 'correct_typed_bond_ids', 'confirmed_false_positives',
                   'unknown_present', 'sdf_valid', 'sdf_exact', 'fallback', 'predicted_smiles')
    metric_parity = {k: before['metrics'][k] == historical['metrics'][k] for k in metric_keys}
    delta = compare_scenes(traces['baseline'], traces['admitted'])
    delta.update(lost_matched_targets=sorted(set(before['matched_target_ids']) - set(after['matched_target_ids'])),
        gained_matched_targets=sorted(set(after['matched_target_ids']) - set(before['matched_target_ids'])),
        changed_old_assignments=[dict(slot=s, before=t, after=after['mapping'].get(s))
                                 for s, t in before['mapping'].items() if after['mapping'].get(s) != t],
        lost_correct_raw_typed_bonds=sorted(set(before['metrics']['correct_typed_bond_ids']) - set(after['metrics']['correct_typed_bond_ids'])),
        gained_correct_raw_typed_bonds=sorted(set(after['metrics']['correct_typed_bond_ids']) - set(before['metrics']['correct_typed_bond_ids'])),
        old_pair_max_abs_logit_delta=logit_delta(records['baseline'], records['admitted']))
    if (state_digest(model) != INITIAL_STATE or state_digest(predictor.ocr) != trial['ocr_state_before']
            or any(p.grad is not None for m in (model, predictor.ocr) for p in m.parameters())
            or torch.cuda.is_initialized()):
        raise ValueError('Frozen CPU state invariant failed')
    for path, digest in list(hashes.items()):
        read(path, hashes, digest)
    artifacts = ['plan.json', 'baseline-prediction-only.json', 'admitted-prediction-only.json']
    report = dict(schema_version='single-candidate-replay-0.1', sample_key=NH, admitted_slot=slot,
        snapshot_max_abs_drift=drift, selected_parity=parity, historical_scene_delta=historical_delta,
        historical_baseline_metric_parity=metric_parity, historical_logit_delta=logit_delta(historical, records['baseline']),
        scores=scores, delta=delta, thresholds={k: v for k, v in payload['provenance'].items() if k.endswith('_threshold')},
        input_sha256=hashes, artifact_sha256={n: sha256((OUTPUT / n).read_bytes()).hexdigest() for n in artifacts},
        forwards=dict(upstream=1, **calls), optimizer_steps=0, training_performed=False,
        weights_unchanged=True, original_outputs_changed=False, diagnostic_mask_bits_added=1,
        oracle_used_for_generation_or_selection=False, validation_samples_read=False,
        sealed_test_read=False, new_identity_queries=0, gpu_used=False,
        limitations=['One previously diagnosed training failure, not a held-out accuracy estimate',
            'Manual experimental admission is not a validated automatic repair policy',
            'Fixed original matching can leave real scene edges unknown; unknown is not false',
            'CPU replay is not bitwise historical GPU feature recovery',
            'Correct local bonds do not establish stereo or complete molecular identity'])
    write(OUTPUT / 'report.json', report)
    print(json.dumps(dict(admitted_slot=slot, metric_parity=metric_parity,
                         scores={k: v['metrics'] for k, v in scores.items()}, delta=delta), indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args(argv)
    if args.worker:
        worker()
        return 0
    receipt = runtime.run_budgeted([sys.executable, '-m', 'tools.audit_single_candidate_replay', '--worker'],
                                  output=OUTPUT, budget=runtime.ResourceBudget(120))
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
