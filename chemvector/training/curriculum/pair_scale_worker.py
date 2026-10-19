"""Single-image streaming scale worker; all real operations require new authority."""

from collections import Counter
from dataclasses import asdict
import json
import math
import os
import time

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from . import pair_scale_training as launch
from .bundle import load_bundle
from .checkpoints import file_digest
from .data import CurriculumCollator
from .evaluation import FrozenOCRPredictor, evaluate
from .gradient_preflight import state_digest
from .interface_diagnostic import write_json
from .pair_cached_trial import frozen_digest
from .pair_only_worker import INITIAL_STATE, save_delta
from .pair_scale_plan import sample_id
from .pair_scale_policy import development_decision
from .pair_trial_contract import configure_pair_stage
from .release import RecordLoader
from .runtime import StepLimit


def profile_indices(schedule, count=64):
    unique=list(dict.fromkeys(s['index'] for s in schedule))
    if type(count) is not int or count < 2 or len(unique)<count:
        raise ValueError('Enough unique fixed training images required')
    return [unique[i*(len(unique)-1)//(count-1)] for i in range(count)]


def throughput_gate(times, *, attempts, baseline_seconds, elapsed_seconds, wall_seconds=3600):
    if (len(times)!=64 or type(attempts) is not int or attempts<10000
            or any(not math.isfinite(t) or t<=0 for t in times)
            or any(not math.isfinite(t) or t<0 for t in (baseline_seconds,elapsed_seconds))
            or wall_seconds!=3600):
        raise ValueError('Measured fixed64 costs and original budget required')
    mean=sum(times)/len(times); p95=sorted(times)[math.ceil(.95*len(times))-1]
    remaining=1.5*max(mean,p95)*attempts+1.25*4*baseline_seconds+120
    return dict(mean_training_seconds=mean,p95_training_seconds=p95,
                predicted_remaining_seconds=remaining,remaining_budget_seconds=wall_seconds-elapsed_seconds,
                passed=elapsed_seconds+remaining<wall_seconds,
                limitation='cost estimate only; independent hard timer still applies')


def backward_pair(stage,batch):
    """Shared real/profile path. No optimizer and no parameter modification."""
    if any(m.training for m in stage.modules()):
        raise ValueError('All modules must be eval; pair autograd stays enabled')
    stage.pair_parameters()
    result=stage(batch)
    if result['losses'] is None:
        return dict(updated=False,coverage=result['coverage'],loss=None)
    result['losses']['loss'].backward()
    params=stage.pair_parameters()
    gradients=[p.grad for p in params if p.grad is not None]
    if not gradients or any(not bool(torch.isfinite(g).all()) for g in gradients):
        raise ValueError('Nonfinite/absent pair gradients')
    norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
    return dict(updated=True,coverage=result['coverage'],loss=float(result['losses']['loss'].detach()),
                gradient_norm=float(norm))


def optimizer_step(stage, optimizer, limit):
    params=stage.pair_parameters()
    actual=[p for group in optimizer.param_groups for p in group['params']]
    if len(actual)!=len(params) or {id(p) for p in actual}!={id(p) for p in params}:
        raise ValueError('Only the complete active pair head may be optimized')
    if not any(p.grad is not None for p in params):
        raise ValueError('No gradients; empty branches cannot consume an update')
    limit.consume()
    optimizer.step()
    if any(not bool(torch.isfinite(p).all()) for p in params):
        raise ValueError('Nonfinite updated pair')


def worker():
    started=time.monotonic()
    data,auth=launch.authorize()
    auth_sha=file_digest(launch.AUTHORIZATION)
    if (json.loads(launch.ATTEMPT.read_text())['authorization_sha256']!=auth_sha
            or os.environ.get('CUDA_VISIBLE_DEVICES')!='0' or not launch.OUTPUT.is_dir()):
        raise ValueError('Fresh reservation and GPU 0 required')
    write_json(launch.OUTPUT/'worker-started.json',dict(pid=os.getpid(),authorization_sha256=auth_sha))
    counters=Counter(profile_images=0,training_attempts=0,training_upstream_forwards=0,evaluation_images=0,
                     skipped_empty=0)
    limit=StepLimit(len(data['schedule']))
    seen=set(); artifacts={}; decisions=[]

    def guard():
        if time.monotonic()-started>=3555:
            raise TimeoutError('All-inclusive budget; stop with no resume')

    def write(name,value):
        write_json(launch.OUTPUT/name,value)
        artifacts[name]=file_digest(launch.OUTPUT/name)

    with (launch.OUTPUT/'events.jsonl').open('x',encoding='utf-8') as log:
        def emit(event,**fields):
            row=dict(event=event,elapsed_seconds=time.monotonic()-started,**fields)
            value=json.dumps(row,allow_nan=False)
            log.write(value+'\n');log.flush()
            print(value,flush=True)
        try:
            torch.set_num_threads(2);torch.manual_seed(20260924)
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA unavailable; no CPU fallback')
            torch.cuda.manual_seed_all(20260924)
            device=torch.device('cuda:0');props=torch.cuda.get_device_properties(device)
            environment=dict(device=str(device),name=props.name,uuid=str(props.uuid),
                torch_version=str(torch.__version__),cuda_version=torch.version.cuda,dtype='float32',
                cache_mode='stream_one_training_image_no_feature_corpus')
            write('plan.json',dict(authorization_sha256=auth_sha,protocol=data['protocol'],
                execution_contract=launch.execution_contract(data['protocol']),environment=environment))
            guard()
            model,config,payload=load_bundle(launch.prep.INITIAL,expected_sha256=launch.prep.INITIAL_SHA)
            if state_digest(model)!=INITIAL_STATE or payload['progress']['optimizer_steps']!=0:
                raise ValueError('Exact original initial required')
            expected={'image_size':768,'maximum_atoms':128,'atom_match_radius_px':6}
            if any(getattr(config,k)!=v for k,v in expected.items()):
                raise ValueError('Initial configuration changed')
            if any(payload['provenance'][k]!=v for k,v in dict(atom_threshold=-.25,bond_threshold=-.5,
                    pair_threshold=.85,role_threshold=.85).items()):
                raise ValueError('Initial thresholds changed')
            model.to(device);stage=configure_pair_stage(model,config)
            predictor=FrozenOCRPredictor(payload['provenance'],device)
            frozen_before,ocr_before=frozen_digest(model),state_digest(predictor.ocr)
            loader=RecordLoader(data['manifest'])
            original={sample_id(r):r for r in data['manifest']['samples']}
            collator=CurriculumCollator(AtomBondSceneConfig(atom_queries=config.maximum_atoms,
                bond_queries=config.maximum_bonds),purpose='training')

            def assert_frozen():
                if frozen_digest(model)!=frozen_before or state_digest(predictor.ocr)!=ocr_before:
                    raise ValueError('Frozen upstream/OCR changed')
                stage.pair_parameters()

            def validate(attempt):
                guard();begin=time.monotonic()
                def checked_loader(manifest,row):
                    guard()
                    if row['split']!='validation':
                        raise ValueError('Fixed development only')
                    return loader(manifest,row)
                records=evaluate(model,config,data['manifest'],checked_loader,predictor,device=device)
                if len(records)!=944 or Counter(r['domain'] for r in records)!={'full_molecule':800,'curriculum':144}:
                    raise ValueError('Full development cohort changed')
                counters['evaluation_images']+=len(records)
                if counters['evaluation_images']>5*944:
                    raise ValueError('Development evaluation cap')
                assert_frozen();write(f'validation-{attempt:06d}.json',dict(records=records))
                emit('development_complete',attempt=attempt,images=len(records))
                return records,time.monotonic()-begin

            def training_backward(index):
                guard();model.zero_grad(set_to_none=True)
                row=original[sample_id(data['rows'][index])]
                if row['split']!='train':
                    raise ValueError('No validation gradient')
                sample=loader(data['manifest'],row)
                batch=collator([sample])
                batch={k:v.to(device) for k,v in batch.items() if isinstance(v,torch.Tensor)}
                # No per-image feature cache is retained: bounded by one image.
                result=backward_pair(stage,batch)
                return result

            baseline,baseline_seconds=validate(0)
            times=[];indices=profile_indices(data['schedule'])
            for index in indices:
                guard();torch.cuda.synchronize();begin=time.monotonic()
                result=training_backward(index)
                torch.cuda.synchronize();times.append(time.monotonic()-begin)
                counters['profile_images']+=1
                model.zero_grad(set_to_none=True)
                emit('profile_image',index=index,seconds=times[-1],has_supervision=result['updated'])
            if state_digest(model)!=INITIAL_STATE:
                raise ValueError('Profiling must not modify model state')
            assert_frozen()
            gate=throughput_gate(times,attempts=len(data['schedule']),baseline_seconds=baseline_seconds,
                                 elapsed_seconds=time.monotonic()-started)
            write('throughput.json',dict(indices=indices,times=times,**gate))
            if not gate['passed']:
                raise RuntimeError('Projected complete pass exceeds fixed budget; no optimizer created')
            # Restore sampling RNG so profiling does not alter the training schedule's draws.
            torch.manual_seed(20260924);torch.cuda.manual_seed_all(20260924)
            params=stage.pair_parameters()
            optimizer=torch.optim.AdamW(params,lr=2e-5,betas=(.9,.999),eps=1e-8,weight_decay=0.)
            stop_reason='complete_pass'
            for attempt,entry in enumerate(data['schedule'],1):
                result=training_backward(entry['index'])
                counters['training_upstream_forwards']+=1
                counters['training_attempts']+=1;seen.add(entry['index'])
                if result['updated']:
                    guard();optimizer_step(stage,optimizer,limit)
                    emit('step_complete',attempt=attempt,step=limit.consumed)
                else:
                    counters['skipped_empty']+=1
                optimizer.zero_grad(set_to_none=True)
                emit('train_attempt',attempt=attempt,index=entry['index'],visit=entry['visit'],
                     role=entry['role'],optimizer_steps=limit.consumed,**result)
                if attempt in data['protocol']['checkpoint_attempts']:
                    current,_=validate(attempt)
                    complete=attempt==len(data['schedule']) and len(seen)==len(data['rows'])
                    decision=development_decision(baseline,current,complete_coverage=complete)
                    decisions.append(dict(attempt=attempt,**decision))
                    write(f'decision-{attempt:06d}.json',decisions[-1])
                    name=f'pair-{attempt:06d}.pt'
                    artifacts[name]=save_delta(launch.OUTPUT/name,dict(
                        schema_version='pair-scale-delta-0.1',initial_sha256=launch.prep.INITIAL_SHA,
                        authorization_sha256=auth_sha,architecture=asdict(model.pair.config),
                        pair_state={k:v.detach().cpu() for k,v in model.pair.state_dict().items()},
                        optimizer_steps=limit.consumed,decision=decision,deployment_authorized=False,resume_supported=False))
                    emit('checkpoint',attempt=attempt,decision=decision)
                    if decision['action']=='stop_severe_regression':
                        stop_reason='stop_severe_regression';break
            guard();assert_frozen();launch.authorize()
            if (counters['profile_images']+counters['training_upstream_forwards']+counters['evaluation_images']
                    >launch.execution_contract(data['protocol'])['maximum_upstream_forwards']):
                raise ValueError('Upstream forward cap exceeded')
            complete=counters['training_attempts']==len(data['schedule']) and len(seen)==len(data['rows'])
            write('report.json',dict(complete=True,complete_coverage=complete,stop_reason=stop_reason,
                counters=counters,optimizer_steps=limit.consumed,distinct_training_images=len(seen),
                decisions=decisions,artifact_sha256=dict(artifacts),environment=environment,
                authorization_sha256=auth_sha,initial_sha256=launch.prep.INITIAL_SHA,
                frozen_sha256=frozen_before,ocr_sha256=ocr_before,
                selected_for_review=bool(complete and decisions[-1]['candidate_eligible']),
                training_performed=limit.consumed>0,gpu_used=True,deployment_authorized=False,
                default_changed=False,old_release_changed=False,sealed_test_read=False,
                validation_media_read=True,new_identity_queries=0,elapsed_seconds=time.monotonic()-started))
            emit('complete',stop_reason=stop_reason,optimizer_steps=limit.consumed,complete_coverage=complete)
        except Exception as error:
            emit('failed',error_type=type(error).__name__,error=str(error),optimizer_steps=limit.consumed)
            write('failure.json',dict(error_type=type(error).__name__,error=str(error),counters=counters,
                optimizer_steps=limit.consumed,distinct_training_images=len(seen),success=False,
                automatic_retries=0,incomplete=True))
            raise
