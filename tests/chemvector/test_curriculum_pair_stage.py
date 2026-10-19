"""Synthetic pair-only autograd/mode tests; no real model or optimizer update."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from chemvector.training.curriculum import pair_stage
from tests.chemvector.test_curriculum_supervision_contract import pair_fixture


class TinyPair(nn.Module):
    def __init__(self):
        super().__init__()
        self.presence = nn.Parameter(torch.zeros(1))
        self.chemical = nn.Parameter(torch.zeros(4))
        self.calls = []

    def forward(self, features, atom_features, anchors, scores, pairs, rows):
        self.calls.append({'pairs': pairs.tolist(), 'rows': rows.tolist(),
                           'requires_grad': [x.requires_grad for x in (features, atom_features, anchors, scores)]})
        return {'presence_logits': self.presence.expand(len(pairs)),
                'chemical_logits': self.chemical.expand(len(pairs), -1)}


def setup(monkeypatch, *, all_unknown=False):
    atoms, _, batch = pair_fixture()
    # One known path 0--1--2; target atom 3 has no close predicted candidate.
    atoms['atom_geometry'] = atoms['atom_geometry'].clone().requires_grad_(True)
    with torch.no_grad(): atoms['atom_geometry'][0, 3, 0] = .95
    if all_unknown:
        with torch.no_grad(): atoms['atom_geometry'][..., 1] = .9
    atoms['atom_features'] = torch.ones(1, 4, 2, requires_grad=True)
    atoms['atom_scores'] = torch.ones(1, 4, requires_grad=True)
    batch['pixel_values'] = torch.ones(1, 1, 2, 2, requires_grad=True)
    features = torch.ones(1, 2, 2, 2, requires_grad=True)
    components = nn.Module()
    for name in ('vision', 'atom', 'bond', 'set_head', 'role', 'bbox'):
        components.add_module(name, nn.Sequential(nn.Linear(2, 2), nn.Dropout(.5)))
    components.pair = TinyPair()
    calls = []
    def forward(model, pixels, config, threshold):
        calls.append((torch.is_grad_enabled(), threshold))
        return features, None, None, atoms, None, None, atoms
    monkeypatch.setattr(pair_stage, 'structure_forward', forward)
    config = SimpleNamespace(image_size=768, atom_match_radius_px=6)
    stage = pair_stage.FrozenUpstreamPairStage(components, config, atom_threshold=-.25)
    return stage, batch, atoms, features, calls


def test_only_pair_receives_gradients_unknown_rows_excluded_and_denominators_preserved(monkeypatch):
    stage, batch, atoms, features, calls = setup(monkeypatch)
    result = stage(batch)
    assert calls == [(False, -.25)]
    assert result['coverage'] == dict(target_atoms=4, selected_atoms=4, unmatched_selected_atoms=1,
        target_bonds=3, reachable_positive_pairs=2, unreachable_bonds=1, known_negative_pairs=1,
        unknown_pairs=3, loss_rows=3, sampled_unknown_pairs=0, oracle_candidates_injected=False)
    assert stage.components.pair.calls[0]['pairs'] == [[0, 1], [1, 2], [0, 2]]
    assert not any(stage.components.pair.calls[0]['requires_grad'])
    result['losses']['loss'].backward()
    assert all(p.grad is not None for p in stage.pair_parameters())
    assert all(p.grad is None for name, p in stage.named_parameters() if '.pair.' not in name)
    assert all(t.grad is None for t in (features, batch['pixel_values'], atoms['atom_features'],
                                      atoms['atom_geometry'], atoms['atom_scores']))


def test_repeated_train_eval_cannot_enable_upstream_dropout(monkeypatch):
    stage, *_ = setup(monkeypatch)
    for mode in (False, True, True, False):
        stage.train(mode)
        assert stage.components.pair.training is mode
        assert not stage.components.training
        assert all(not m.training for name in ('vision', 'atom', 'bond', 'set_head', 'role', 'bbox')
                   for m in getattr(stage.components, name).modules())
        assert {id(p) for p in stage.parameters() if p.requires_grad} == {id(p) for p in stage.pair_parameters()}


def test_all_unknown_skips_loss_without_a_fake_zero_step(monkeypatch):
    stage, batch, *_ = setup(monkeypatch, all_unknown=True)
    result = stage(batch)
    assert result['losses'] is None and result['skip_reason'] == 'no_known_pairs'
    assert result['coverage']['unknown_pairs'] == 6 and result['coverage']['unreachable_bonds'] == 3
    assert not stage.components.pair.calls


@pytest.mark.parametrize('bad', ['upstream_grad', 'pair_frozen', 'upstream_train'])
def test_external_refreezing_or_training_modes_fail_closed(monkeypatch, bad):
    stage, batch, *_ = setup(monkeypatch)
    if bad == 'upstream_grad': stage.components.vision.requires_grad_(True)
    elif bad == 'pair_frozen': stage.components.pair.requires_grad_(False)
    else: stage.components.vision.train()
    with pytest.raises(ValueError): stage(batch)


def test_nonfinite_loss_is_rejected(monkeypatch):
    stage, batch, *_ = setup(monkeypatch)
    with torch.no_grad(): stage.components.pair.presence.fill_(float('nan'))
    with pytest.raises(ValueError, match='Non-finite'): stage(batch)


def test_no_implicit_updates_to_legacy_continuation_recipe():
    from chemvector.training.curriculum.continuation import ContinuationComponents
    with pytest.raises(ValueError, match='Unknown continuation'):
        ContinuationComponents.configure_update_recipe(None, 'pair-only-v1')


def test_existing_pair_head_accepts_frozen_synthetic_features_and_backpropagates(monkeypatch):
    from chemvector.direct_atom_pair_bond import DirectAtomPairBond, DirectAtomPairBondConfig
    stage, batch, *_ = setup(monkeypatch)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20260918)
        stage.components.pair = DirectAtomPairBond(DirectAtomPairBondConfig(
            feature_channels=2, atom_feature_dim=2, longitudinal_samples=8,
            transverse_samples=3, hidden_dim=8, pair_chunk_size=2))
        stage.train()
        result = stage(batch)
        result['losses']['loss'].backward()
    gradients = [p.grad for p in stage.pair_parameters()]
    assert all(g is not None and bool(torch.isfinite(g).all()) for g in gradients)
    assert any(bool(torch.count_nonzero(g)) for g in gradients)
    assert all(p.grad is None for name, p in stage.named_parameters() if '.pair.' not in name)


@pytest.mark.parametrize('count', range(2, 8))
def test_small_known_candidate_sets_keep_every_confirmed_negative(count):
    from chemvector.training.curriculum.supervision_contract import pair_targets_with_unknown, select_known_pair_rows
    atoms = {'atom_mask': torch.ones(1, count, dtype=torch.bool),
             'atom_geometry': torch.stack((torch.linspace(.1, .9, count), torch.full((count,), .5)), dim=1)[None]}
    targets = {'atom_mask': atoms['atom_mask'], 'bond_mask': torch.ones(1, count - 1, dtype=torch.bool),
               'bond_atom_refs': torch.stack((torch.arange(count - 1), torch.arange(1, count)), dim=1)[None],
               'bond_chemical': torch.zeros(1, count - 1, dtype=torch.long)}
    contract = pair_targets_with_unknown(atoms, torch.arange(count)[None], targets)
    chosen = select_known_pair_rows(contract, atoms, negative_ratio=6, minimum_negatives=48)
    assert set(chosen.tolist()) == set(range(count * (count - 1) // 2))
    assert not bool(contract['unknown_mask'].any())
