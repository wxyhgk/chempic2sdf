"""One conditional, fixed58 readout trial; cached features, fresh proposals."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.bond_object import BOND_TYPE_TO_ID
from chemvector.vertex import build_vertex_heatmap
from . import heatmap_readout_stage as stage
from . import selector_retention_release as release
from .bundle import load_bundle, save_bundle
from .checkpoints import file_digest
from .continuation import structure_forward
from .data import CurriculumCollator
from .evaluation import FrozenOCRPredictor
from .gradient_preflight import state_digest
from .interface_diagnostic import tensor_tree_digest
from .pixels import image_tensor
from .runtime import ResourceBudget, StepLimit, run_budgeted
from .selector_retention_trial import reserve_once
from .upstream_diagnostic import score_prediction
from .validation_truth import expected_identity
from tools.audit_selector_stable_subset import Inputs, write
from tools import audit_selector_dense_observation as source

ROOT = release.ROOT
AUTH = ROOT / 'docs/model/training/releases/20260920-heatmap-readout-training-authorization.json'
OUTPUT = ROOT / 'outputs/experiments/chemvector-heatmap-readout-trial-20260920'
ATTEMPT = ROOT / 'outputs/experiments/chemvector-heatmap-readout-attempt-20260920.json'
DIRECTION = ROOT / 'outputs/audits/chemvector-heatmap-readout-direction7-20260920/report.json'
DIRECTION_SHA = '37938026f1227bf7a537a8c8eefa336a0bfbb3706b407d3040174391bdd67528'
ADMISSION_SHA = '77c4b42da2f675e61924ee7904032deb05d30b8de7767f92bfd9888aed1e85f9'
PROTOCOL = dict(wall_seconds=600, maximum_steps=400, gpu_devices=['0'], seed=20260920,
    local_checks=[0, 100, 200, 400], batch_size=1, optimizer='SGD', learning_rate=.001,
    momentum=0, weight_decay=0, clip_norm=1., automatic_retries=0,
    parameter_scope='atom.heatmap.weight and atom.heatmap.bias only', parameter_count=129,
    loss='existing focal_heatmap_loss, all atoms, radius_cells=1',
    training_cycle=['old_full_molecule', 'old_full_molecule', 'witness'],
    candidate_and_downstream_recomputed=True, model_mode='eval', tf32=False, amp=False,
    minimum_raw_recovery=3, early_stop_no_recovery_step=200,
    validation_images=0, deployment_authorized=False)


def authorize():
    inputs = Inputs()
    _, _, scope = source.preflight(inputs)
    inputs.read(release.OUTPUT / 'admission.json', ADMISSION_SHA)
    manifest, admission = release.verify_release(release.OUTPUT)
    report = inputs.json(DIRECTION, DIRECTION_SHA)
    if report['gate']['status'] != 'DIRECTION_SUPPORTED' or report['initial_state_after'] != source.INITIAL_STATE:
        raise ValueError('Phase1 did not support this conditional trial')
    auth = inputs.json(AUTH)
    expected = dict(schema_version='heatmap-readout-training-authorization-0.1',
        user_statement='开始按照顺序推进', protocol=PROTOCOL, maximum_invocations=1,
        manifest_sha256=source.MANIFEST_SHA, admission_sha256=ADMISSION_SHA,
        initial_sha256=source.INITIAL_SHA, direction_report_sha256=DIRECTION_SHA,
        training_authorized=True, deployment_authorized=False, development_read_authorized=False,
        data_use_extension='same58_local_heatmap_readout_trial_only', output=str(OUTPUT.relative_to(ROOT)))
    if any(auth.get(k) != v for k, v in expected.items()):
        raise ValueError('Missing exact conditional one-shot authorization')
    required = {str(Path(__file__).relative_to(ROOT)), str(Path(stage.__file__).relative_to(ROOT)),
                'chemvector/training/curriculum/selector_retention_release.py',
                'chemvector/training/curriculum/evaluation.py',
                'chemvector/training/curriculum/bundle.py', 'chemvector/training/curriculum/runtime.py'}
    if not required <= set(auth.get('implementation_sha256', {})):
        raise ValueError('Critical implementation hashes absent')
    for path, digest in auth['implementation_sha256'].items():
        inputs.read(ROOT / path, digest)
    for path, digest in auth['protected_inputs_sha256'].items():
        inputs.read(ROOT / path, digest)
    inputs.read(source.INITIAL, source.INITIAL_SHA)
    return manifest, admission, scope, auth, inputs


@torch.no_grad()
def assess(model, cache, config, predictor, device):
    raw, selected, logits = stage.refresh_proposals(model, cache, config, device)
    structure = predictor.from_predicted_atoms(model.pair, cache['features'].to(device), selected)
    prediction = predictor.from_structure(structure, model.role, model.bbox, cache['pixels'].to(device), cache['image'])
    # Truth is consulted only after the complete image-only prediction.
    coverage = stage.coverage(raw, selected, logits, cache['layers'], cache['anchors'])
    mapping = coverage['selected']['original']['mapping']
    truth_bonds = {tuple(sorted((b['begin_atom'], b['end_atom']))): b for b in cache['layers']['bonds']}
    correct, false_pairs, predicted_pairs = [], 0, []
    presence = structure['output']['presence_logits'].sigmoid()
    for i in (presence >= predictor.provenance['pair_threshold']).nonzero().flatten().tolist():
        a, b = structure['pairs'][i].tolist()
        ta, tb = mapping.get(a, -1), mapping.get(b, -1)
        target = truth_bonds.get(tuple(sorted((ta, tb)))) if min(ta, tb) >= 0 else None
        chemical = int(structure['chemistry'][i])
        matched = target is not None and chemical == BOND_TYPE_TO_ID[target['chemical_type']]
        if matched:
            correct.append(target['bond_id'])
        else:
            false_pairs += 1
        predicted_pairs.append(dict(slots=[a,b], target_ids=[ta,tb], chemical_id=chemical,
                                    correct_typed_bond_id=target['bond_id'] if matched else None))
    return dict(key=cache['key'], sample_key=cache['sample_key'], group=cache['group'],
        **coverage, prediction=score_prediction(prediction, cache['expected']),
        correct_typed_bond_ids=sorted(set(correct)), false_or_wrong_typed_pairs=false_pairs,
        predicted_pairs=predicted_pairs,
        bond_metric_scope='predicted pair connectivity/type before text/stereo compilation; not stereo accuracy')


def worker():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise ValueError('Single explicit GPU 0 required')
    manifest, _, scope, auth, inputs = authorize()
    reservation = json.loads(ATTEMPT.read_text())
    if (reservation['authorization_sha256'] != file_digest(AUTH) or not OUTPUT.is_dir()
            or (OUTPUT / 'plan.json').exists()):
        raise ValueError('Fresh reserved output required')
    torch.set_num_threads(2)
    torch.manual_seed(20260920)
    torch.cuda.manual_seed_all(20260920)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device, started = 'cuda:0', time.monotonic()
    write(OUTPUT / 'plan.json', dict(authorization=auth, authorization_sha256=file_digest(AUTH),
        phase=2, training_samples=manifest['training_samples'], development_samples=[], protocol=PROTOCOL))
    model, config, payload = load_bundle(source.INITIAL, expected_sha256=source.INITIAL_SHA)
    if (state_digest(model) != source.INITIAL_STATE or config.maximum_atoms != 128
            or config.atom_threshold != .01 or config.atom_match_radius_px != 6
            or config.image_size != 768 or payload['provenance']['atom_threshold'] != -.25):
        raise ValueError('Initial state/threshold changed')
    model.to(device)
    params = stage.configure(model)
    predictor = FrozenOCRPredictor(payload['provenance'], device)
    ocr_record = payload['provenance']['checkpoints']['ocr_frozen_not_loaded']
    inputs.read(Path(ocr_record['path']).absolute(), ocr_record['sha256'])
    frozen_before, ocr_before = stage.frozen_digest(model), state_digest(predictor.ocr)
    collator = CurriculumCollator(AtomBondSceneConfig(atom_queries=config.maximum_atoms,
                                                    bond_queries=config.maximum_bonds), purpose='training')
    caches, baseline, assessments = [], [], {}
    focus = {r['sample_key']: ids for r, ids in scope}
    limit = StepLimit(400)
    optimizer = torch.optim.SGD(params, lr=.001, momentum=0, weight_decay=0)
    with (OUTPUT / 'steps.jsonl').open('x', encoding='utf-8') as log:
        def emit(event, **values):
            line = json.dumps(dict(event=event, elapsed_seconds=time.monotonic()-started, **values), allow_nan=False)
            log.write(line+'\n')
            log.flush()
            if event == 'step_reserved':
                os.fsync(log.fileno())
            if event != 'update' or values.get('step',0) % 25 == 0:
                print(line, flush=True)

        for row in manifest['training_samples']:
            sample = release.load_row(manifest, row, purpose='training')
            base = Path(manifest['sources'][row['source']]) / 'samples'
            meta = inputs.json(base / (row['sample_key']+'.meta.json'), row['meta_sha256'])
            for suffix, digest in meta['sha256'].items():
                inputs.read(base / (row['sample_key']+suffix), digest)
            with torch.no_grad():
                pixel = image_tensor(sample['image']).unsqueeze(0).to(device)
                features, atom, _, original_raw, bonds, original_logits, original_selected = structure_forward(model, pixel, config, -.25)
            batch = collator([sample])
            mask = batch['atom_mask']
            anchors = batch['atom_anchors'][:, :int(mask.sum())]
            heat = build_vertex_heatmap(batch['atom_anchors'], mask, 192, 192, 1)
            if int((heat == 1).sum()) != int(mask.sum()):
                raise ValueError('Target heatmap cell collision')
            cache = dict(key=row['source']+':'+row['sample_key'], sample_key=row['sample_key'],
                group=row['fit_domain'], purpose='training', training_authorized=batch['training_authorized'],
                manifest_sha256=source.MANIFEST_SHA, features=features.detach().cpu(),
                atom_static=stage.tensors_to({k:v for k,v in atom.items() if k != 'heatmap_logits'}, 'cpu'),
                bonds=stage.tensors_to(bonds,'cpu'), pixels=pixel.detach().cpu(), image=sample['image'],
                expected=expected_identity(sample), layers=sample['layers'], anchors=anchors[0].tolist(), target_heatmap=heat)
            with torch.no_grad():
                fresh_raw, fresh_selected, fresh_logits = stage.refresh_proposals(model, cache, config, device)
                for a,b in ((original_raw,fresh_raw),(original_selected,fresh_selected),(original_logits,fresh_logits)):
                    if any(not torch.equal(a[k], b[k]) for k in a):
                        raise ValueError('Cached-feature refresh differs from direct production baseline')
            result = assess(model, cache, config, predictor, device)
            baseline.append(result)
            caches.append(cache)
            path = OUTPUT / ('baseline--'+row['sample_key']+'.json')
            write(path,result)
            assessments[path.name] = file_digest(path)
            emit('baseline_case', sample_key=row['sample_key'], completed=len(caches))
            del features, atom, bonds, pixel, original_raw, original_selected, original_logits
        if (Counter(c['group'] for c in caches) != {'old_full_molecule':32,'witness':26}
                or sum(len(c['anchors']) for c in caches) != 1090):
            raise ValueError('Fixed58 target population changed')
        baseline_gate = stage.gate(baseline,baseline,focus,0)
        if baseline_gate['stop_reasons'] or sum(len(v) for v in focus.values()) != 10:
            raise ValueError('Baseline protections/ten raw gaps differ')
        cache_hash_before = tensor_tree_digest({c['key']:{k:c[k] for k in ('features','atom_static','bonds','target_heatmap')} for c in caches})
        write(OUTPUT/'baseline-summary.json',stage.summary(baseline))
        emit('baseline_complete', summary=stage.summary(baseline))
        current, checkpoints, gates = baseline, [], []
        stream = stage.order(caches, source.MANIFEST_SHA)
        while limit.consumed < 400:
            cache = next(stream)
            result = stage.update(model,cache,optimizer,limit,device,
                lambda step: emit('step_reserved',step=step,key=cache['key']),source.MANIFEST_SHA)
            emit('update',step=limit.consumed,key=cache['key'],**result)
            if limit.consumed not in (100,200,400):
                continue
            current=[]
            for cache in caches:
                record=assess(model,cache,config,predictor,device)
                current.append(record)
                path=OUTPUT/(f'check-{limit.consumed:06d}--'+cache['sample_key']+'.json')
                write(path,record)
                assessments[path.name]=file_digest(path)
            gate=stage.gate(baseline,current,focus,limit.consumed)
            gates.append(gate)
            write(OUTPUT/f'gate-{limit.consumed:06d}.json',gate)
            checkpoints.append(save_bundle(OUTPUT/f'check-{limit.consumed:06d}.pt',model,config,
                provenance=payload['provenance'],progress=dict(optimizer_steps=limit.consumed,
                    diagnostic_only=True,deployment_authorized=False,parameter_scope=PROTOCOL['parameter_scope'])))
            emit('check_complete',step=limit.consumed,summary=stage.summary(current),
                 recovered=gate['recovered_count'],stop_reasons=gate['stop_reasons'])
            if gate['stop_reasons']:
                break
        cache_hash_after = tensor_tree_digest({c['key']:{k:c[k] for k in ('features','atom_static','bonds','target_heatmap')} for c in caches})
        if (cache_hash_after != cache_hash_before or stage.frozen_digest(model) != frozen_before
                or state_digest(predictor.ocr) != ocr_before):
            raise ValueError('Frozen model/OCR/feature/target cache changed')
        for path,digest in list(inputs.hashes.items()):
            inputs.read(path,digest)
        write(OUTPUT/'report.json',dict(schema_version='heatmap-readout-trial-0.1',
            protocol=PROTOCOL,authorization_sha256=file_digest(AUTH),input_sha256=inputs.hashes,
            assessment_sha256=assessments,optimizer_steps=limit.consumed,checkpoints=checkpoints,
            baseline=stage.summary(baseline),final=stage.summary(current),gates=gates,
            local_pass=gates[-1]['local_pass'],frozen_state_sha256=frozen_before,ocr_state_sha256=ocr_before,
            cache_sha256_before=cache_hash_before,cache_sha256_after=cache_hash_after,
            final_state_sha256=state_digest(model),gpu_used=True,training_performed=True,
            validation_sample_artifacts_read=False,new_identity_queries=0,default_changed=False,
            phase3_started=False,deployed=False,initial_kept=True,
            limitations=['Only58 observed fitting/replay images, no generalization claim',
                'Original and radius-reference matching reported separately; no matching policy changed',
                'Pair-type correctness is not stereochemical correctness; SDF metric separately reported',
                'Only one fixed SGD recipe, not a search or a proof of heatmap-readout capacity']))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args(argv)
    if args.worker:
        if args.check_only:
            parser.error('Worker is not check-only')
        worker()
        return 0
    _,admission,_,_,_=authorize()
    if args.check_only:
        print(json.dumps(dict(checks_passed=True,protocol=PROTOCOL,training_performed=False),indent=2))
        return 0
    if OUTPUT.exists():
        raise FileExistsError('One-shot output exists; no retry')
    reserve_once(ATTEMPT,AUTH)
    receipt=run_budgeted([sys.executable,'-m',__package__+'.heatmap_readout_trial','--worker'],
        output=OUTPUT,budget=ResourceBudget(600,('0',)),training=True,admission=admission)
    print(json.dumps(receipt,indent=2))
    return 0 if receipt['status']=='completed' else 1


if __name__=='__main__':
    raise SystemExit(main())
