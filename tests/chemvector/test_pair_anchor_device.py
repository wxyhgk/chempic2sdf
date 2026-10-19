"""New cache adapter checked on synthetic CPU data, not a GPU parity claim."""

from copy import deepcopy

import pytest
import torch

from chemvector.training.curriculum import pair_anchor_device as device, pair_cached_trial as cached
from chemvector.training.curriculum.runtime import StepLimit
from tests.chemvector.test_pair_only_trial import fixture
from tests.chemvector.test_pair_anchor_wiring import assert_nested_equal


def setup(monkeypatch, unknown=False):
    stage, batch, cache, optimizer = fixture(monkeypatch, unknown=unknown)
    plan = device.make_plan(cache, batch['atom_anchors'], ordinal=0, execution_device='cpu')
    return stage, batch, cache, optimizer, plan


def test_plan_contains_only_current_lineage_and_unchanged_original_sampling(monkeypatch):
    stage, _, cache, _, plan = setup(monkeypatch)
    assert plan['offset_pixels'] == [-4, -4] and plan['execution_device'] == 'cpu'
    chosen = plan['tensors']['chosen']
    assert set(chosen.tolist()) == {0, 1, 3}
    assert torch.equal(chosen[plan['tensors']['augmented_positions']], plan['tensors']['augmented_rows'])
    assert device.zero_weight_parity(stage.components.pair, cache, plan) == dict(passed=True, forwards=2)


def test_control_matches_legacy_cached_update_over_multiple_toy_steps(monkeypatch):
    stage, _, cache, optimizer, plan = setup(monkeypatch)
    other = deepcopy(stage)
    other_opt = torch.optim.AdamW(other.pair_parameters(), lr=2e-5, weight_decay=0.)
    arm, total, legacy_limit = StepLimit(54), StepLimit(108), StepLimit(200)
    # Each original tiny sample includes all eligible rows, so seed does not
    # change its supervision. Real per-image seeds remain fixed by ordinal.
    for i in range(3):
        device.update(stage, cache, plan, optimizer, arm, total, weight=0, emit=lambda *a, **k: None)
        cached.update_once(other, cache, other_opt, legacy_limit, attempt=i, emit=lambda *a, **k: None)
        assert_nested_equal(stage.state_dict(), other.state_dict())
        assert_nested_equal(optimizer.state_dict(), other_opt.state_dict())


def test_augmentation_updates_only_pair_and_records_two_forwards(monkeypatch):
    stage, _, cache, optimizer, plan = setup(monkeypatch)
    before = {k: v.clone() for k, v in stage.components.state_dict().items()}
    result = device.update(stage, cache, plan, optimizer, StepLimit(54), StepLimit(108), weight=1, emit=lambda *a, **k: None)
    assert result['updated'] and result['forwards'] == 2
    assert all(k.startswith('pair.') for k, v in stage.components.state_dict().items() if not torch.equal(before[k], v))
    cached.verify_cache(cache); device.verify_plan(cache, plan)


@pytest.mark.parametrize('change', ['features', 'chosen', 'anchors', 'ordinal', 'device', 'wrong_cache'])
def test_cache_or_plan_mutation_fails_before_forward(monkeypatch, change):
    stage, _, cache, _, plan = setup(monkeypatch)
    if change == 'features': cache['tensors']['features'].add_(1)
    elif change == 'chosen': plan['tensors']['chosen'][0] += 1
    elif change == 'anchors': plan['tensors']['shifted_anchors'].add_(1)
    elif change == 'ordinal': plan['ordinal'] = 2
    elif change == 'device': plan['execution_device'] = 'cuda:0'
    else: plan['cache_sha256'] = 'f'*64
    stage.components.pair.calls.clear()
    with pytest.raises(ValueError): device.objective(stage.components.pair, cache, plan, weight=1)
    assert stage.components.pair.calls == []


def test_empty_base_never_steps_and_preserves_target_denominator(monkeypatch):
    stage, _, cache, optimizer, plan = setup(monkeypatch, unknown=True)
    monkeypatch.setattr(optimizer, 'step', lambda: pytest.fail('Empty branch updated'))
    arm, total = StepLimit(54), StepLimit(108)
    result = device.update(stage, cache, plan, optimizer, arm, total, weight=1, emit=lambda *a, **k: None)
    assert not result['updated'] and result['forwards'] == arm.consumed == total.consumed == 0
    assert cache['coverage']['target_bonds'] == 3


@pytest.mark.parametrize('limit_name', ['arm', 'total'])
def test_both_step_caps_fail_before_head_forward(monkeypatch, limit_name):
    stage, _, cache, optimizer, plan = setup(monkeypatch)
    arm, total = StepLimit(54), StepLimit(108)
    (arm if limit_name == 'arm' else total).consumed = 54 if limit_name == 'arm' else 108
    stage.components.pair.calls.clear()
    with pytest.raises(RuntimeError): device.update(stage, cache, plan, optimizer, arm, total, weight=1, emit=lambda *a, **k: None)
    assert stage.components.pair.calls == []


def test_negative_only_is_not_dropped(monkeypatch):
    stage, batch, cache, optimizer, _ = setup(monkeypatch)
    batch['bond_mask'].zero_()
    t = cache['tensors']
    cache = cached.make_cache((t['features'], None, None, t['raw'], t['bonds'], t['context'], t['selected']), batch, stage.config)
    plan = device.make_plan(cache, batch['atom_anchors'], ordinal=0, execution_device='cpu')
    result = device.update(stage, cache, plan, optimizer, StepLimit(54), StepLimit(108), weight=1, emit=lambda *a, **k: None)
    assert result['updated'] and result['base_rows'] == 3


def test_full_pair_metrics_separate_raw_promoted_false_and_unknown(monkeypatch):
    _, _, cache, _, _ = setup(monkeypatch)
    t = cache['tensors']
    chemical = torch.zeros(6, 4); chemical[:, 0] = 10
    promoted = torch.zeros(6, dtype=torch.long); promoted[3] = 1
    structure = dict(pairs=t['pair_indices'], output=dict(presence_logits=torch.full((6,), 10.), chemical_logits=chemical),
                     chemistry=promoted)
    result = device.score_case(cache, structure, dict(molblock=None, fallback=False),
                               dict(roles=[], ocr=[]), key='toy', expected=None)
    assert result['correct_typed_bond_ids'] == [0]
    assert result['promoted_correct_typed_bond_ids'] == [0, 1]
    assert result['confirmed_false_pair_row_ids'] == [1]
    assert result['unknown_present_row_ids'] == [2, 4, 5]
    assert result['target_bonds'] == 3 and result['reachable_positive_pairs'] == 2
