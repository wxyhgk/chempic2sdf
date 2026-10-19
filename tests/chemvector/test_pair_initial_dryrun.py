"""Read-only hook diagnostics tested on synthetic existing components."""

from copy import deepcopy

import pytest
import torch

from chemvector.training.curriculum.gradient_preflight import state_digest
from chemvector.training.curriculum import pair_supervision as supervision
from tests.chemvector.test_curriculum_continuation import tiny
from tools import pair_initial_dryrun as diagnostic


def prepare(tiny):
    model, config, batch = tiny
    model.configure_update_recipe('freeze-role-v1')
    model.eval().requires_grad_(False)
    model.pair.requires_grad_(True)
    batch.update(purpose='diagnostic', training_authorized=False)
    return model, config, batch


def test_real_loss_hooks_do_not_mutate_or_leak_rng_or_gradients(tiny):
    model, config, batch = prepare(tiny)
    before = state_digest(model)
    rng = torch.get_rng_state().clone()
    arms = {}
    for policy in supervision.POLICIES:
        arm, snapshot = diagnostic.run_arm(model, batch, config, policy=policy,
                                           atom_threshold=-1e6, bond_threshold=-.5, seed=9)
        arms[policy] = arm
        assert arm['state_before'] == arm['state_after'] == before
        assert arm['module_calls']['role'] == 0
        assert arm['pair_backward_performed']
        assert arm['pair_gradient_l2'] > 0
        assert snapshot['geometry'].shape == (8, 8)
        assert all(p.grad is None for p in model.parameters())
        assert torch.equal(torch.get_rng_state(), rng)
    comparison = diagnostic.compare_arms(arms[supervision.LEGACY], arms[supervision.MATCHED_ONLY])
    assert comparison['exact_cpu_candidate_context_parity']
    assert comparison['masked_sampled_unmatched_pairs'] == 0
    assert state_digest(model) == before


def test_empty_pair_branch_skips_backward(tiny):
    model, config, batch = prepare(tiny)
    for policy in supervision.POLICIES:
        arm, _ = diagnostic.run_arm(model, batch, config, policy=policy,
                                     atom_threshold=1e6, bond_threshold=-.5, seed=9)
        assert not arm['pair_backward_performed']
        assert arm['pair_gradient_tensors'] == arm['module_calls']['pair'] == 0
        assert arm['sampled_pair_rows'] == []


def test_hooks_removed_on_failure(tiny):
    model, config, batch = prepare(tiny)
    counts = [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in model.modules()]
    with pytest.raises(RuntimeError, match='intentional'):
        with diagnostic.Capture(model, config):
            raise RuntimeError('intentional')
    assert counts == [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in model.modules()]


@pytest.mark.parametrize('bad', ['training_batch', 'train_mode', 'upstream_grad'])
def test_wrong_scope_rejected_before_execution(tiny, bad):
    model, config, batch = prepare(tiny)
    if bad == 'training_batch':
        batch['purpose'] = 'training'
    elif bad == 'train_mode':
        model.train()
    else:
        model.atom.requires_grad_(True)
    with pytest.raises(ValueError):
        diagnostic.run_arm(model, batch, config, policy=supervision.MATCHED_ONLY,
                           atom_threshold=-1e6, bond_threshold=-.5, seed=9)


@pytest.mark.parametrize('field', ['upstream_sha256', 'selector_input_sha256', 'text_traces', 'original_positive_rows'])
def test_arm_mismatch_cannot_be_claimed_as_mask_effect(field):
    old = dict(policy=supervision.LEGACY, upstream_sha256='a', selector_input_sha256='b',
        selected_slot_ids=[0, 1], losses={n: 1. for n in ('atom', 'bond', 'set_head', 'role', 'bbox', 'pair')},
        text_traces={}, original_positive_rows=[0], sampled_pairs_with_unmatched_endpoint=1)
    new = deepcopy(old)
    new.update(policy=supervision.MATCHED_ONLY, sampled_pairs_with_unmatched_endpoint=0)
    new[field] = 'changed'
    with pytest.raises(ValueError):
        diagnostic.compare_arms(old, new)
