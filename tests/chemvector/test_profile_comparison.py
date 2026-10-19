from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from rdkit import Chem
import torch

from chemvector.experiments.id_graph.roundtrip import signature, write_json
from chemvector.experiments.profile_comparison.protocol import ARMS, PROFILES, validate_profiles, validate_selection
from chemvector.experiments.profile_comparison.runtime import LoaderCache, arguments, compare_outputs, run_resident
from chemvector.experiments.profile_comparison.scoring import counts, score_sdf, summarize


def expected(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return {'core': signature(mol, False), 'stereo': signature(mol)}


def scored(smiles, target):
    return score_sdf(Chem.MolToMolBlock(Chem.MolFromSmiles(smiles)), expected(target))


def test_sdf_valid_does_not_mean_correct():
    row = scored('CN', 'CO')
    assert row['sdf_valid'] and not row['core_graph_exact']
    assert counts([row])['valid_but_wrong'] == 1


@pytest.mark.parametrize('block', ['', 'garbage', Chem.MolToMolBlock(Chem.Mol())])
def test_invalid_and_empty_sdf_are_failures(block):
    row = score_sdf(block, expected('C'))
    assert not row['sdf_valid'] and row['score_error']


@pytest.mark.parametrize('predicted,target', [('C', '[13CH4]'), ('CN', 'C[NH3+]')])
def test_core_scoring_preserves_isotopes_and_charge(predicted, target):
    assert not scored(predicted, target)['core_graph_exact']


def test_core_ignores_stereo_but_secondary_does_not():
    row = scored('F[C@](Cl)(Br)I', 'F[C@@](Cl)(Br)I')
    assert row['core_graph_exact'] and not row['stereo_graph_exact']


def test_canonical_atom_order_invariance():
    assert scored('OC', 'CO')['core_graph_exact']


def make_rows():
    rows = []
    for key in ('a', 'b'):
        for arm in ARMS:
            correct = (key == 'a' and arm != 'v01016') or (key == 'b' and arm == 'v01016')
            rows.append({'sample_key': key, 'arm': arm, 'sdf_valid': True,
                         'core_graph_exact': correct, 'stereo_graph_exact': correct})
    metadata = {k: {'molecule_id': 'mol', 'variant': k} for k in ('a', 'b')}
    return rows, metadata


def test_pairwise_rescue_and_harm_not_only_net_gain():
    rows, metadata = make_rows()
    result = summarize(rows, ['a', 'b'], metadata)
    paired = result['pairs']['v01010->v01016']['core_graph_exact']
    assert paired == {'rescued': ['b'], 'harmed': ['a']}
    assert result['denominator'] == 2 and result['molecules'] == 1
    assert result['arms']['v01016']['core_graph_exact'] == 1


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'unexpected', 'duplicate_order'])
def test_no_silent_denominator_change(change):
    rows, metadata = make_rows()
    order = ['a', 'b']
    if change == 'missing': rows.pop()
    if change == 'duplicate': rows.append(rows[0])
    if change == 'unexpected': rows[0] = {**rows[0], 'sample_key': 'c'}
    if change == 'duplicate_order': order.append('a')
    with pytest.raises(ValueError):
        summarize(rows, order, metadata)


def test_fallback_tracked_independently_of_correctness():
    row = {**scored('C', 'O'), 'fallback_used': True, 'fallback_attempted': True}
    result = counts([row])
    assert result['fallback_used'] == result['fallback_valid'] == result['fallback_attempted'] == 1
    assert result['fallback_core_exact'] == 0


def test_native_registry_is_a_single_suppression_difference():
    registry = json.loads(Path('frontend/inference_profiles.json').read_text())['profiles']
    profiles = {arm: registry[key] for arm, key in PROFILES.items()}
    before = deepcopy(profiles)
    validate_profiles(profiles)
    assert profiles == before
    profiles['v01016']['arguments'].append('--unexpected')
    with pytest.raises(ValueError): validate_profiles(profiles)


def test_profile_command_uses_registry_without_manual_thresholds():
    profile = {'arguments': ['--checkpoint', 'a.pt', '--enable-ocr']}
    assert arguments(profile, 'input.png', 'out') == [
        '--checkpoint', 'a.pt', '--enable-ocr', '--image', 'input.png', '--output-dir', 'out', '--device', 'cuda']


def test_original_fixed_selection_and_traversal_rejection():
    previous = json.loads(Path('outputs/experiments/id-graph-prediction-20260913/protocol.json').read_text())
    validate_selection(previous)
    previous['image_index'][previous['sample_order'][0]]['member'] = '../other.png'
    with pytest.raises(ValueError): validate_selection(previous)


def test_cache_loads_once_per_arguments_and_checks_state():
    cache = LoaderCache()
    calls = []
    def loader(path, device):
        calls.append(path)
        return torch.nn.Linear(2, 2), {'path': path}
    cached = cache.wrap('model', loader)
    first = cached('a', 'cpu')
    assert cached('a', 'cpu') is first
    assert cached('b', 'cpu') is not first
    assert calls == ['a', 'b'] and cache.verify()['models_unchanged']
    with torch.no_grad(): first[0].weight.add_(1)
    with pytest.raises(ValueError, match='state changed'): cache.verify()


def test_run_resident_retains_error_and_prevents_overwrite(tmp_path):
    def fail(): raise ValueError('fixed compiler error')
    destination = tmp_path / 'run'
    result = run_resident(SimpleNamespace(main=fail), [], destination)
    assert result['error'] == {'type': 'ValueError', 'message': 'fixed compiler error'}
    assert 'fixed compiler error' in (destination / 'stderr.log').read_text()
    with pytest.raises(FileExistsError): run_resident(SimpleNamespace(main=fail), [], destination)


def parity_directories(tmp_path):
    dirs = [tmp_path / 'resident', tmp_path / 'cold']
    for i, path in enumerate(dirs):
        path.mkdir()
        write_json(path / 'metrics.json', {'sdf_valid': True, 'inference_seconds': i, 'diagnostics': {'x': 2}})
        # Native files are compared byte-for-byte; JSON here is sufficient to exercise hashing.
        write_json(path / 'prediction.chemscene.json', {'atoms': [1]})
    return dirs


def test_cold_parity_only_excludes_timings(tmp_path):
    resident, cold = parity_directories(tmp_path)
    assert 'metrics_except_inference_seconds' in compare_outputs(resident, cold, None, 0)
    metrics = json.loads((cold / 'metrics.json').read_text())
    metrics['sdf_valid'] = False
    (cold / 'metrics.json').unlink()
    write_json(cold / 'metrics.json', metrics)
    with pytest.raises(ValueError, match='metrics differ'): compare_outputs(resident, cold, None, 0)


def test_cold_parity_detects_changed_artifact(tmp_path):
    resident, cold = parity_directories(tmp_path)
    (cold / 'prediction.chemscene.json').unlink()
    with pytest.raises(ValueError, match='artifact differs'): compare_outputs(resident, cold, None, 0)


def test_cold_parity_detects_completion_difference(tmp_path):
    resident, cold = parity_directories(tmp_path)
    with pytest.raises(ValueError, match='completion differ'): compare_outputs(resident, cold, None, 1)


def test_legacy_carbon_named_fallback_does_not_reset_elements(monkeypatch):
    from chemvector import placeholder_scene as module
    scene = {'atom_objects': [{'id': 0, 'anchor': [0, 0], 'incident_ports': [],
                              'chemical': {'element': 'N', 'formal_charge': 1, 'aromatic': True}}],
             'bond_objects': []}
    before = deepcopy(scene)
    monkeypatch.setattr(module, 'mol_from_chemscene_v0_2', lambda value: 'molecule')
    result, mol, _ = module.compile_carbon_placeholder(scene)
    assert result['atom_objects'][0]['chemical'] == {'element': 'N', 'formal_charge': 1, 'aromatic': False}
    assert scene == before and mol == 'molecule'
