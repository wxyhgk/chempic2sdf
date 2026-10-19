"""Synthetic runtime/gate tests; never reads actual models or development media."""

from copy import deepcopy

import pytest
import torch

from chemvector.training.curriculum import pair_scale_training as launch, pair_scale_worker as worker
from chemvector.training.curriculum.pair_scale_plan import draft_protocol
from chemvector.training.curriculum.runtime import StepLimit


def sources():
    return dict(protocol=draft_protocol(15443),preparation_report_sha256='a'*64)


def auth():
    return dict(launch.expected_authorization(sources()),user_statement='Synthetic only; not a real grant',
                readiness_report_sha256='b'*64,implementation_sha256={p:'c'*64 for p in launch.IMPLEMENTATIONS})


def test_exact_new_grant_can_be_validated_without_execution():
    launch.validate_authorization(auth(),sources())


@pytest.mark.parametrize('change',['old_use','time','steps','deployment','missing_file','missing_statement','hash','dev_read','old_manifest'])
def test_scope_expansion_or_old_permission_rejected(change):
    a=auth()
    if change=='old_use':a['use']='local_pair_anchor_ab58_once'
    elif change=='time':a['budget']['wall_seconds']=7200
    elif change=='steps':a['budget']['maximum_updates']+=1
    elif change=='deployment':a['deployment_authorized']=True
    elif change=='missing_file':a['implementation_sha256'].pop(next(iter(a['implementation_sha256'])))
    elif change=='missing_statement':a['user_statement']=''
    elif change=='hash':a['readiness_report_sha256']='bad'
    elif change=='dev_read':a['data_use_extension']['development_media_read']=False
    else:a['manifest_sha256']='d'*64
    with pytest.raises(ValueError):launch.validate_authorization(a,sources())


def test_worker_denies_before_cuda_or_model_load(monkeypatch):
    monkeypatch.setattr(launch,'authorize',lambda:(_ for _ in ()).throw(ValueError('No grant')))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:pytest.fail('CUDA before grant'))
    monkeypatch.setattr(worker,'load_bundle',lambda *a,**k:pytest.fail('Model before grant'))
    with pytest.raises(ValueError,match='No grant'):worker.worker()


def test_missing_grant_does_not_even_read_preparation(monkeypatch,tmp_path):
    monkeypatch.setattr(launch,'AUTHORIZATION',tmp_path/'absent.json')
    monkeypatch.setattr(launch,'source_checks',lambda:pytest.fail('Checks before grant'))
    with pytest.raises(FileNotFoundError):launch.authorize()


def test_profile_has64_distinct_images_not64_replay_steps():
    schedule=[dict(index=i%100) for i in range(300)]
    selected=worker.profile_indices(schedule)
    assert len(selected)==len(set(selected))==64
    assert selected[0]==0 and selected[-1]==99
    assert selected==worker.profile_indices(schedule)
    with pytest.raises(ValueError):worker.profile_indices(schedule[:20])


def test_time_gate_includes_training_remaining_dev_and_saving():
    gate=worker.throughput_gate([.1]*64,attempts=15000,baseline_seconds=20,elapsed_seconds=50)
    assert gate['passed']
    assert gate['predicted_remaining_seconds']==pytest.approx(2250+100+120)
    assert not worker.throughput_gate([.2]*64,attempts=15000,baseline_seconds=20,elapsed_seconds=50)['passed']


def test_time_gate_respects_slow_tail():
    times=[.01]*59+[.5]*5
    assert not worker.throughput_gate(times,attempts=15000,baseline_seconds=20,elapsed_seconds=50)['passed']


@pytest.mark.parametrize('bad', [float('nan'),float('inf'),0,-1])
def test_invalid_profile_measurement_rejected(bad):
    times=[.1]*64;times[0]=bad
    with pytest.raises(ValueError):worker.throughput_gate(times,attempts=15000,baseline_seconds=20,elapsed_seconds=50)


class ToyStage(torch.nn.Module):
    def __init__(self,empty=False):
        super().__init__()
        self.pair=torch.nn.Linear(1,1,bias=False)
        self.frozen=torch.nn.Linear(1,1,bias=False).requires_grad_(False)
        self.empty=empty
        self.eval()

    def pair_parameters(self):return list(self.pair.parameters())

    def forward(self,batch):
        return dict(coverage=dict(known_pairs=0 if self.empty else 1),
                    losses=None if self.empty else dict(loss=self.pair(batch).square().sum()))


def test_profiling_backward_does_not_modify_weights_or_frozen_gradients():
    stage=ToyStage();before=deepcopy(stage.state_dict())
    result=worker.backward_pair(stage,torch.ones(1,1))
    assert result['updated'] and stage.pair.weight.grad is not None
    assert stage.frozen.weight.grad is None
    assert all(torch.equal(before[k],v) for k,v in stage.state_dict().items())


def test_empty_branch_never_updates_or_consumes_step():
    stage=ToyStage(empty=True)
    result=worker.backward_pair(stage,torch.ones(1,1))
    assert not result['updated']
    limit=StepLimit(1);opt=torch.optim.AdamW(stage.pair.parameters(),lr=.1)
    with pytest.raises(ValueError,match='No gradients'):worker.optimizer_step(stage,opt,limit)
    assert limit.consumed==0


def test_step_ceiling_prevents_extra_update_and_freezes_upstream():
    stage=ToyStage();opt=torch.optim.AdamW(stage.pair.parameters(),lr=.1);limit=StepLimit(1)
    before=stage.frozen.weight.detach().clone()
    worker.backward_pair(stage,torch.ones(1,1));worker.optimizer_step(stage,opt,limit)
    pair=stage.pair.weight.detach().clone()
    with pytest.raises(RuntimeError,match='budget exhausted'):worker.optimizer_step(stage,opt,limit)
    assert limit.consumed==1 and torch.equal(stage.pair.weight,pair)
    assert torch.equal(stage.frozen.weight,before)


def test_optimizer_cannot_include_frozen_parameters():
    stage=ToyStage();worker.backward_pair(stage,torch.ones(1,1))
    opt=torch.optim.AdamW(stage.parameters(),lr=.1);limit=StepLimit(1)
    with pytest.raises(ValueError,match='Only'):worker.optimizer_step(stage,opt,limit)
    assert limit.consumed==0


def test_training_mode_or_nonfinite_gradients_rejected():
    stage=ToyStage();stage.train()
    with pytest.raises(ValueError,match='eval'):worker.backward_pair(stage,torch.ones(1,1))
    stage.eval()
    with pytest.raises(ValueError):worker.backward_pair(stage,torch.tensor([[float('nan')]]))
