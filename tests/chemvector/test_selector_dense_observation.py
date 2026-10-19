"""Synthetic dense snapshots only; no real checkpoint/image access."""

from copy import deepcopy

import pytest
import torch
from torch import nn

from chemvector.predicted_incidence import PredictedProposalConfig, decode_predicted_atom_proposals
from tests.chemvector.test_selector_localization_contract import manual_output
from tools import selector_dense_observation as observation
from tools import audit_selector_dense_observation as runner
from tools.selector_node_semantics import assignment


def fixture():
    probabilities = torch.full((192, 192), .001)
    probabilities[80, 80], probabilities[80, 81], probabilities[80, 82] = .002, .6, .9
    probabilities[150, 150] = .95
    output, _ = manual_output(probabilities)
    config = PredictedProposalConfig()
    raw = decode_predicted_atom_proposals(output, config)
    snapshot = {**{k: output[k].detach().clone() for k in ('heatmap_logits', 'anchor_offset_logits', 'anchors')},
        **{'raw_' + k: raw['atom_' + k].detach().clone() for k in ('indices', 'scores', 'mask', 'geometry')}}
    slots = raw['atom_mask'][0].nonzero().flatten().tolist()
    geo = dict(image_size=768, match_radius_px=6,
        target_anchors=[[80.5/192, 80.5/192], [150.5/192, 150.5/192]],
        candidates=[dict(slot=s, grid_id=int(raw['atom_indices'][0, s]),
                        geometry=raw['atom_geometry'][0, s].detach().tolist(),
                        raw_score=float(raw['atom_scores'][0, s])) for s in slots])
    mapping = assignment(geo, slots)
    for c in geo['candidates']:
        c['raw_target'] = mapping[c['slot']]
    return snapshot, geo, config, output


def test_suppressed_gt_with_spare_capacity_and_exact_historical_contract():
    snapshot, geo, config, _ = fixture()
    before = deepcopy(geo)
    result = observation.explain_targets(snapshot, geo, [0], config)
    assert result['parity']['historical_discrete_contract_reproduced']
    assert result['parity']['candidate_numbers_bitwise_equal']
    assert result['parity']['cpu_missing_target_ids'] == [0]
    target = result['targets'][0]
    assert target['gt_cell']['first_excluding_stage'] == 'local_peak_suppressed'
    assert not target['gt_cell']['passes_threshold']
    assert target['gt_cell']['direct_target_distance_px'] == 0
    assert target['cpu_nearest_cell']['direct_target_distance_px'] == pytest.approx(8., abs=1e-4)
    assert target['dense_cells_within_6px'] and all(not c['raw_active'] for c in target['dense_cells_within_6px'])
    assert target['historical_explanation_supported'] and target['cpu_still_missing']
    assert geo == before and not result['training_authorized']


def test_numeric_difference_not_silently_called_bitwise_parity():
    snapshot, geo, config, _ = fixture()
    geo['candidates'][0]['geometry'][0] += 1e-6
    result = observation.compare_history(snapshot, geo, config)
    assert result['historical_discrete_contract_reproduced']
    assert not result['candidate_numbers_bitwise_equal']
    assert result['max_common_xy_change_px'] > 0


def test_changed_order_or_grid_denies_historical_explanation():
    snapshot, geo, config, _ = fixture()
    geo['candidates'][0]['grid_id'] += 1
    result = observation.explain_targets(snapshot, geo, [0], config)
    assert not result['parity']['historical_discrete_contract_reproduced']
    assert not result['targets'][0]['historical_explanation_supported']
    assert result['parity']['missing_historical_grid_ids']


@pytest.mark.parametrize('change', ['fields', 'shape', 'nan', 'grad', 'indices', 'mask', 'anchor', 'geometry'])
def test_snapshot_fails_closed(change):
    snapshot, _, config, _ = fixture()
    if change == 'fields': snapshot['extra'] = torch.zeros(1)
    elif change == 'shape': snapshot['raw_geometry'] = snapshot['raw_geometry'][..., :2]
    elif change == 'nan': snapshot['heatmap_logits'][0, 0, 0, 0] = float('nan')
    elif change == 'grad': snapshot['anchors'].requires_grad_()
    elif change == 'indices': snapshot['raw_indices'][0, 0] += 1
    elif change == 'mask': snapshot['raw_mask'][0, 0] = False
    elif change == 'anchor': snapshot['anchors'][0, 0, 0, 0] += .01
    else: snapshot['raw_geometry'][0, 0, 0] += .01
    with pytest.raises(ValueError):
        observation.validate_snapshot(snapshot, config)


@pytest.mark.parametrize('focus', [[1], [0, 0], [-1], [False]])
def test_focus_only_historically_missing_unique_nodes(focus):
    snapshot, geo, config, _ = fixture()
    with pytest.raises(ValueError):
        observation.explain_targets(snapshot, geo, focus, config)


def test_snapshot_exclusive_roundtrip(tmp_path):
    snapshot, _, config, _ = fixture()
    path = tmp_path / 'dense.npz'
    restored, digest = runner.save_snapshot(path, snapshot)
    observation.validate_snapshot(restored, config)
    assert len(digest) == 64 and all(torch.equal(v, restored[k]) for k, v in snapshot.items())
    with pytest.raises(FileExistsError):
        runner.save_snapshot(path, snapshot)


class TinyVision(nn.Module):
    def forward_stroke_resolution(self, pixels):
        return pixels


class TinyAtom(nn.Module):
    def __init__(self, output):
        super().__init__()
        self.output, self.calls = output, 0

    def forward(self, features):
        self.calls += 1
        return self.output


def test_image_only_api_runs_atom_once_no_downstream_or_targets():
    snapshot, _, config, output = fixture()
    model = nn.Module()
    model.vision, model.atom = TinyVision(), TinyAtom(output)
    for name in ('bond', 'set_head', 'pair', 'role', 'bbox'):
        module = nn.Identity()
        module.register_forward_pre_hook(runner.forbidden_forward)
        setattr(model, name, module)
    model.eval().requires_grad_(False)
    result = observation.image_only_snapshot(model, torch.zeros(1, 3, 768, 768), config)
    assert model.atom.calls == 1
    assert all(torch.equal(result[k], v) for k, v in snapshot.items())
    assert all(not v.requires_grad for v in result.values())
    with pytest.raises(RuntimeError, match='Out-of-scope'):
        model.bond(torch.zeros(1))


@pytest.mark.parametrize('change', ['training', 'unfrozen', 'shape', 'dtype', 'nan'])
def test_image_only_preconditions(change):
    _, _, config, output = fixture()
    model = nn.Module()
    model.vision, model.atom = TinyVision(), TinyAtom(output)
    model.weight = nn.Parameter(torch.zeros(1), requires_grad=False)
    model.eval()
    pixels = torch.zeros(1, 3, 768, 768)
    if change == 'training': model.train()
    elif change == 'unfrozen': model.weight.requires_grad_()
    elif change == 'shape': pixels = pixels[..., :767]
    elif change == 'dtype': pixels = pixels.double()
    else: pixels[0, 0, 0, 0] = float('nan')
    with pytest.raises(ValueError):
        observation.image_only_snapshot(model, pixels, config)
    assert model.atom.calls == 0


def scope_fixture():
    rows = [dict(sample_key='synthetic_' + str(i), split='train',
                 fit_domain='witness' if i < 26 else 'old_full_molecule') for i in range(58)]
    manifest = dict(training_samples=rows, validation_only_samples=[dict(sample_key='never_read')])
    report = dict(raw_no_eligible_candidate_nodes=[dict(sample_key='synthetic_' + str(26+i),
        group='old_full_molecule', target_id=t) for i in range(7) for t in range(2 if i < 3 else 1)])
    return report, manifest


def test_scope_fixed_seven_training_images():
    report, manifest = scope_fixture()
    scope = runner.fixed_scope(report, manifest)
    assert len(scope) == 7 and sum(len(t) for _, t in scope) == 10


@pytest.mark.parametrize('change', ['validation', 'witness', 'duplicate', 'extra', 'missing', 'row_overlap'])
def test_scope_rejects_expansion_or_misuse(change):
    report, manifest = scope_fixture()
    nodes = report['raw_no_eligible_candidate_nodes']
    if change == 'validation': nodes[0]['sample_key'] = 'never_read'
    elif change == 'witness': nodes[0]['sample_key'] = 'synthetic_0'
    elif change == 'duplicate': nodes[1] = deepcopy(nodes[0])
    elif change == 'extra': nodes.append(dict(sample_key='synthetic_57', group='old_full_molecule', target_id=5))
    elif change == 'missing': nodes.pop()
    else: manifest['validation_only_samples'][0]['sample_key'] = 'synthetic_26'
    with pytest.raises(ValueError):
        runner.fixed_scope(report, manifest)
