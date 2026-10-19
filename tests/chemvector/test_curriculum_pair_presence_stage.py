"""Random tiny CPU models only; no project checkpoints, datasets or training run."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from chemvector.direct_atom_pair_bond import DirectAtomPairBond, DirectAtomPairBondConfig, direct_atom_pair_loss
from chemvector.training.curriculum import pair_presence_stage as stage_module
from tests.chemvector.test_curriculum_supervision_contract import pair_fixture


def setup(monkeypatch, *, all_unknown=False):
    atoms, _, batch = pair_fixture()
    atoms['atom_geometry'] = atoms['atom_geometry'].clone()
    atoms['atom_geometry'][0, 3, 0] = .95  # Three known atoms, one unmatched candidate.
    if all_unknown:
        atoms['atom_geometry'][..., 1] = .9
    atoms['atom_geometry'].requires_grad_(True)
    atoms['atom_features'] = torch.linspace(-1., 1., 8).reshape(1, 4, 2).requires_grad_(True)
    atoms['atom_scores'] = torch.ones(1, 4, requires_grad=True)
    features = torch.linspace(-1., 1., 288).reshape(1, 2, 12, 12).requires_grad_(True)
    batch.update(pixel_values=torch.ones(1, 1, 12, 12, requires_grad=True),
                 purpose='diagnostic', training_authorized=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20260918)
        components = nn.Module()
        for name in ('vision', 'atom', 'bond', 'set_head', 'role', 'bbox'):
            components.add_module(name, nn.Sequential(nn.Linear(2, 2), nn.Dropout(.5)))
        components.pair = DirectAtomPairBond(DirectAtomPairBondConfig(feature_channels=2, atom_feature_dim=2,
            longitudinal_samples=8, transverse_samples=3, hidden_dim=8, pair_chunk_size=2))
    calls = []
    def forward(model, pixels, config, threshold):
        calls.append({'grad_enabled': torch.is_grad_enabled(), 'threshold': threshold,
                      'pixels': pixels is batch['pixel_values']})
        return features, None, None, atoms, None, None, atoms
    monkeypatch.setattr(stage_module, 'structure_forward', forward)
    config = SimpleNamespace(image_size=768, atom_match_radius_px=6)
    stage = stage_module.FrozenFeaturePresenceStage(components, config, atom_threshold=-.25)
    return stage, batch, atoms, features, calls


def state(module):
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def test_only_existing_presence_gets_gradients_and_all_known_negatives_stay(monkeypatch):
    stage, batch, atoms, features, calls = setup(monkeypatch)
    before = state(stage)
    result = stage(batch)
    assert calls == [{'grad_enabled': False, 'threshold': -.25, 'pixels': True}]
    assert result['pair_indices'].tolist() == [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    assert result['known_mask'].tolist() == [True, True, False, True, False, False]
    assert result['coverage'] == dict(target_atoms=4, selected_atoms=4, unmatched_selected_atoms=1,
        target_bonds=3, reachable_positive_pairs=2, unreachable_bonds=1, known_negative_pairs=1,
        unknown_pairs=3, loss_rows=3, loss_positive_rows=2, loss_negative_rows=1,
        all_confirmed_non_edges_retained=True, sampled_unknown_pairs=0, oracle_candidates_injected=False)
    assert set(result['losses']) == {'loss', 'presence_loss'}
    assert not result['frozen_hidden'].requires_grad and not result['frozen_chemical_logits'].requires_grad
    result['presence_logits'].retain_grad()
    result['losses']['loss'].backward()
    gradients = result['presence_logits'].grad
    assert gradients[0] < 0 and gradients[3] < 0  # Raise real path edges.
    assert gradients[1] > 0  # Lower the across-middle confirmed non-edge.
    assert torch.equal(gradients[~result['known_mask']], torch.zeros(3))
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in stage.presence_parameters())
    assert all(p.grad is None for n, p in stage.named_parameters() if '.pair.presence.' not in n)
    assert all(t.grad is None for t in (features, batch['pixel_values'], atoms['atom_features'],
                                      atoms['atom_geometry'], atoms['atom_scores']))
    assert all(torch.equal(before[k], v) for k, v in stage.state_dict().items())  # Backward alone never updates.
    assert batch['purpose'] == 'diagnostic' and batch['training_authorized'] is False


def test_synthetic_step_changes_only_presence_and_raw_chemistry_stays_bitwise_equal(monkeypatch):
    stage, batch, *_ = setup(monkeypatch)
    before_state = state(stage)
    before = stage(batch)
    # Ephemeral random tiny model only, not an update to any project checkpoint.
    optimizer = torch.optim.SGD(stage.presence_parameters(), lr=.01)
    optimizer.zero_grad(set_to_none=True)
    before['losses']['loss'].backward()
    optimizer.step()
    after = stage(batch)
    changes = {k for k, v in stage.state_dict().items() if not torch.equal(before_state[k], v)}
    assert changes == {'components.pair.presence.weight', 'components.pair.presence.bias'}
    assert torch.equal(before['frozen_hidden'], after['frozen_hidden'])
    assert torch.equal(before['frozen_chemical_logits'], after['frozen_chemical_logits'])
    assert torch.equal(before['pair_indices'], after['pair_indices'])
    assert not torch.equal(before['presence_logits'], after['presence_logits'])
    assert after['losses']['loss'] < before['losses']['loss']


def test_original_pair_forward_and_readout_have_identical_presence_logits(monkeypatch):
    stage, batch, atoms, features, _ = setup(monkeypatch)
    result = stage(batch)
    with torch.no_grad():
        actual = stage.components.pair(features, atoms['atom_features'], atoms['atom_geometry'][..., :2],
            atoms['atom_scores'], result['pair_indices'], result['sample_indices'])
    assert torch.equal(actual['presence_logits'], result['presence_logits'])
    assert torch.equal(actual['chemical_logits'], result['frozen_chemical_logits'])


def test_all_unknown_skips_pair_computation_and_does_not_fabricate_zero_loss(monkeypatch):
    stage, batch, *_ = setup(monkeypatch, all_unknown=True)
    def forbidden(*args, **kwargs):
        raise AssertionError('Pair must not be called with no eligible supervision')
    monkeypatch.setattr(stage.components.pair, 'forward', forbidden)
    result = stage(batch)
    assert result['losses'] is None and result['skip_reason'] == 'no_known_pairs'
    assert result['coverage']['unknown_pairs'] == 6
    assert result['coverage']['unreachable_bonds'] == 3
    assert result['coverage']['loss_rows'] == 0


@pytest.mark.parametrize('count', [0, 1])
def test_empty_pair_enumeration_skips_without_oracle_candidates(monkeypatch, count):
    stage, batch, atoms, *_ = setup(monkeypatch)
    atoms['atom_mask'][0, count:] = False
    result = stage(batch)
    assert result['losses'] is None and result['coverage']['selected_atoms'] == count
    assert result['coverage']['unreachable_bonds'] == 3


def test_train_eval_never_enables_frozen_dropout(monkeypatch):
    stage, batch, *_ = setup(monkeypatch)
    for mode in (False, True, False, True):
        stage.train(mode)
        assert stage.training is mode and stage.components.pair.presence.training is mode
        assert all(not m.training for m in stage.components.modules() if m is not stage.components.pair.presence)
        stage(batch)


@pytest.mark.parametrize('bad', ['vision_train', 'shared_train', 'chemical_train', 'unfreeze_shared',
                                 'unfreeze_chemical', 'freeze_presence', 'replace_presence', 'stale_grad', 'alias'])
def test_external_mode_parameter_or_gradient_changes_fail_closed(monkeypatch, bad):
    stage, batch, *_ = setup(monkeypatch)
    if bad == 'vision_train': stage.components.vision.train()
    elif bad == 'shared_train': stage.components.pair.shared.train()
    elif bad == 'chemical_train': stage.components.pair.chemical.train()
    elif bad == 'unfreeze_shared': stage.components.pair.shared.requires_grad_(True)
    elif bad == 'unfreeze_chemical': stage.components.pair.chemical.requires_grad_(True)
    elif bad == 'freeze_presence': stage.components.pair.presence.requires_grad_(False)
    elif bad == 'replace_presence': stage.components.pair.presence = nn.Linear(8, 1)
    elif bad == 'alias': stage.components.pair.chemical.bias = stage.components.pair.presence.bias
    else:
        p = next(stage.components.pair.shared.parameters())
        p.grad = torch.ones_like(p)
    with pytest.raises(ValueError): stage(batch)


def test_changing_only_target_bond_types_cannot_change_presence_loss_or_features(monkeypatch):
    stage, batch, *_ = setup(monkeypatch)
    before = stage(batch)
    batch['bond_chemical'] = (batch['bond_chemical'] + 1) % 4
    after = stage(batch)
    assert torch.equal(before['losses']['loss'], after['losses']['loss'])
    assert torch.equal(before['frozen_hidden'], after['frozen_hidden'])
    assert torch.equal(before['frozen_chemical_logits'], after['frozen_chemical_logits'])


@pytest.mark.parametrize('bad', ['threshold_nan', 'threshold_bool', 'stale_grad', 'wrong_readout', 'alias'])
def test_bad_constructor_inputs_are_rejected_before_freezing(monkeypatch, bad):
    stage, _, *_ = setup(monkeypatch)
    components, threshold = stage.components, -.25
    if bad == 'threshold_nan': threshold = float('nan')
    elif bad == 'threshold_bool': threshold = True
    elif bad == 'stale_grad': components.pair.presence.weight.grad = torch.ones_like(components.pair.presence.weight)
    elif bad == 'wrong_readout': components.pair.presence = nn.Linear(8, 2)
    else: components.pair.chemical.bias = components.pair.presence.bias
    with pytest.raises(ValueError):
        stage_module.FrozenFeaturePresenceStage(components, stage.config, atom_threshold=threshold)


@pytest.mark.parametrize('bad', ['hidden_nan', 'chemical_nan', 'presence_nan', 'hidden_width', 'wrong_presence'])
def test_invalid_or_changed_frozen_output_contract_is_rejected(monkeypatch, bad):
    stage, batch, *_ = setup(monkeypatch)
    original = stage.components.pair.forward
    def modified(*args, **kwargs):
        output = original(*args, **kwargs)
        if bad == 'hidden_nan': output['hidden'][0, 0] = float('nan')
        elif bad == 'chemical_nan': output['chemical_logits'][0, 0] = float('nan')
        elif bad == 'presence_nan': output['presence_logits'][0] = float('nan')
        elif bad == 'hidden_width': output['hidden'] = output['hidden'][:, :-1]
        else: output['presence_logits'] = output['presence_logits'] + 1.
        return output
    monkeypatch.setattr(stage.components.pair, 'forward', modified)
    with pytest.raises(ValueError): stage(batch)


@pytest.mark.parametrize('labels', [[True, False, False], [True, True], [False, False], [True] + [False] * 40])
def test_presence_loss_matches_existing_bce_component(labels):
    logits = torch.linspace(-1., 1., len(labels), requires_grad=True)
    targets = torch.tensor(labels)
    actual = stage_module.presence_loss(logits, targets)
    legacy = direct_atom_pair_loss({'presence_logits': logits, 'chemical_logits': torch.zeros(len(labels), 4)},
                                   targets, torch.zeros(len(labels), dtype=torch.long))
    assert torch.equal(actual['presence_loss'], legacy['presence_loss'])
    assert actual['loss'] is actual['presence_loss']


@pytest.mark.parametrize('bad', ['empty', 'float_labels', 'nan', 'shape'])
def test_bad_loss_inputs_fail_closed(bad):
    logits, labels = torch.zeros(2), torch.tensor([True, False])
    if bad == 'empty': logits, labels = logits[:0], labels[:0]
    elif bad == 'float_labels': labels = labels.float()
    elif bad == 'nan': logits[0] = float('nan')
    else: labels = labels[:1]
    with pytest.raises(ValueError): stage_module.presence_loss(logits, labels)


def test_presence_prototype_does_not_register_or_change_default_continuation():
    from chemvector.training.curriculum.continuation import ContinuationComponents
    with pytest.raises(ValueError, match='Unknown continuation'):
        ContinuationComponents.configure_update_recipe(None, 'presence-readout-v1')
    assert not hasattr(stage_module, 'main') and not hasattr(stage_module.FrozenFeaturePresenceStage, 'train_step')
