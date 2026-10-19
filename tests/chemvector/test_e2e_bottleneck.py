from copy import deepcopy

from rdkit import Chem
import pytest

from chemvector.direct_scene_inference import apply_atom_ocr
from chemvector.placeholder_scene import placeholder_scene
from chemvector.shadow_chemscene_v02_oracle import _default_carbon
from chemvector.experiments.e2e_bottleneck.controls import correspondence, intervene, compile_and_score, identity
from chemvector.experiments.e2e_bottleneck.runtime import summarize


def scene(elements=('C', 'N'), positions=(200, 400), edges=((0, 1),)):
    result = placeholder_scene('fixture', 768, 768)
    for index, (element, position) in enumerate(zip(elements, positions)):
        chemical = _default_carbon()
        chemical['element'] = element
        result['atom_objects'].append({'id': index, 'anchor': [position, 500], 'role': 'IMPLICIT',
                                      'chemical': chemical, 'text': None, 'incident_ports': []})
    for index, pair in enumerate(edges):
        result['bond_objects'].append({'id': index, 'atom_refs': list(pair), 'ports': [[0, 0], [0, 0]],
                                      'display_type': 'SINGLE', 'chemical_type': 'SINGLE', 'direction': 'NONE',
                                      'stereo': 'STEREONONE', 'bbox': None})
    apply_atom_ocr(result, list(range(len(elements))), {})
    return result


def test_no_intervention_is_exact_identity_and_no_mutation():
    base = scene(('C', 'C'))
    before = deepcopy(base)
    changed, _ = intervene(base, scene())
    assert changed == base == before
    changed['atom_objects'][0]['chemical']['element'] = 'O'
    assert base == before


def test_attributes_do_not_change_positions_bonds_or_raw_text():
    base = scene(('C', 'C'))
    changed, _ = intervene(base, scene(), attributes=True)
    assert changed['atom_objects'][1]['chemical']['element'] == 'N'
    assert changed['bond_objects'] == base['bond_objects']
    assert [a['anchor'] for a in changed['atom_objects']] == [a['anchor'] for a in base['atom_objects']]
    assert [a['text'] for a in changed['atom_objects']] == [a['text'] for a in base['atom_objects']]


def test_bond_oracle_cannot_restore_missing_atom():
    base = scene(('C',), (200,), ())
    changed, metrics = intervene(base, scene(), bonds=True)
    assert len(changed['atom_objects']) == 1 and not changed['bond_objects']
    assert metrics['unavailable_truth_bonds'] == 1


def test_bond_oracle_keeps_extra_atom_and_corrects_spurious_edges():
    base = scene(('C', 'C', 'C'), (200, 400, 800), ((0, 2), (1, 2)))
    changed, _ = intervene(base, scene(), bonds=True)
    assert len(changed['atom_objects']) == 3
    assert [b['atom_refs'] for b in changed['bond_objects']] == [[0, 1]]
    assert changed['atom_objects'][1]['chemical']['element'] == 'C'


def test_correspondence_uses_geometry_not_atom_order_or_element():
    base = scene(('C', 'C'), (400, 200))
    assert correspondence(base, scene()) == {0: 1, 1: 0}
    changed, _ = intervene(base, scene(), attributes=True, bonds=True)
    assert changed['atom_objects'][0]['chemical']['element'] == 'N'
    assert changed['bond_objects'][0]['atom_refs'] == [1, 0]


def test_full_oracle_roundtrips_to_correct_sdf_graph():
    changed, _ = intervene(scene(('C', 'C'), edges=()), scene(), attributes=True, bonds=True)
    metrics, _, block = compile_and_score(changed, Chem.MolFromSmiles('CN'))
    assert metrics['sdf_valid'] and metrics['core_graph_exact'] and metrics['stereo_graph_exact']
    assert block


def test_core_identity_drops_stereo_but_preserves_isotope_and_charge():
    left, right = Chem.MolFromSmiles('F[C@H](Cl)Br'), Chem.MolFromSmiles('F[C@@H](Cl)Br')
    assert identity(left) == identity(right)
    assert identity(left, True) != identity(right, True)
    assert identity(Chem.MolFromSmiles('[13CH4]')) != identity(Chem.MolFromSmiles('C'))
    assert identity(Chem.MolFromSmiles('[NH4+]')) != identity(Chem.MolFromSmiles('N'))


def test_invalid_empty_prediction_is_counted_not_matched():
    metrics, _, block = compile_and_score(placeholder_scene('empty', 768, 768), Chem.MolFromSmiles('C'))
    assert not metrics['sdf_valid'] and not metrics['core_graph_exact'] and block is None


def test_factorial_control_failure_prevents_ranking():
    arms = [f'A{a}B{b}C{c}' for a in (0, 1) for b in (0, 1) for c in (0, 1)]
    config = {'arms': arms, 'sample_order': [str(i) for i in range(200)], 'full_oracle_required_primary': 200, 'scope': 'test'}
    rows = [{'sample_key': key, 'arm': arm, 'sdf_valid': True, 'carbon_fallback': False,
             'core_graph_exact': arm == 'A1B1C1' and key != '0', 'stereo_graph_exact': False}
            for key in config['sample_order'] for arm in arms]
    result = summarize(rows, config)
    assert not result['full_oracle_control_passed'] and not result['architecture_ranking_authorized']
    assert result['conditional_transitions']['A0B1C1->A1B1C1']['rescued'] == 199
    with pytest.raises(ValueError):
        summarize(rows[:-1], config)
