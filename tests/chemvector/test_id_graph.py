from copy import deepcopy

from rdkit import Chem
import pytest

from chemvector.experiments.id_graph.graph import from_chemmol, to_molecule, to_sdf


def chemical_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return {'molecule_id': 'fixture', 'canonical_isomeric_smiles': 'deliberately-not-used',
            'atoms': [{'id': a.GetIdx(), 'element': a.GetSymbol(), 'formal_charge': a.GetFormalCharge(),
                       'isotope': a.GetIsotope(), 'aromatic': a.GetIsAromatic(), 'explicit_h_count': a.GetNumExplicitHs(),
                       'no_implicit': a.GetNoImplicit(), 'radical_electrons': a.GetNumRadicalElectrons(),
                       'chiral_tag': str(a.GetChiralTag())} for a in mol.GetAtoms()],
            'bonds': [{'id': b.GetIdx(), 'begin_atom': b.GetBeginAtomIdx(), 'end_atom': b.GetEndAtomIdx(),
                       'type': str(b.GetBondType()), 'aromatic': b.GetIsAromatic(), 'conjugated': b.GetIsConjugated(),
                       'stereo': str(b.GetStereo()), 'stereo_atoms': list(b.GetStereoAtoms())} for b in mol.GetBonds()]}


def canonical(mol):
    return Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True, isomericSmiles=True)


SMILES = ['C', 'CN', 'c1ccncc1', 'c1cc[nH]c1', '[NH4+].[Cl-]', '[13CH4]', '[CH3]',
          'F/C=C/Cl', 'F/C=C\\Cl', 'F[C@H](Cl)Br', 'F[C@@H](Cl)Br',
          '[13CH3][C@@H]([NH3+])C(=O)[O-]', '[2H][C@](F)(Cl)Br',
          'C[C@H]1CCCCO1', 'C[C@H](O)[C@@H](N)C(=O)O', 'CC(=O)[O-].[Na+]',
          'F/C=C/C=C/Cl', 'F/C=C\\C=C/Cl', '[H]/C(F)=C(Cl)/Br']


@pytest.mark.parametrize('smiles', SMILES)
def test_semantics_and_sdf_roundtrip(smiles):
    graph = from_chemmol(chemical_from_smiles(smiles))
    expected = canonical(Chem.MolFromSmiles(smiles))
    assert canonical(to_molecule(graph)) == expected
    sdf = to_sdf(graph)
    assert 'V3000' in sdf
    actual = Chem.MolFromMolBlock(sdf, sanitize=True, removeHs=True)
    assert actual is not None and canonical(actual) == expected


@pytest.mark.parametrize('smiles', SMILES)
def test_reordered_arrays_and_renamed_ids_preserve_stereo(smiles):
    graph = from_chemmol(chemical_from_smiles(smiles))
    expected = canonical(to_molecule(graph))
    ids = {a['id']: 100 + 7 * (len(graph['atoms']) - a['id']) for a in graph['atoms']}
    for atom in graph['atoms']:
        atom['id'] = ids[atom['id']]
        atom['stereo']['neighbor_ids'] = [ids[i] for i in atom['stereo']['neighbor_ids']]
    for bond in graph['bonds']:
        bond['id'] = 1000 + 11 * (len(graph['bonds']) - bond['id'])
        bond['atom_ids'] = [ids[i] for i in bond['atom_ids']]
        bond['stereo_atom_ids'] = [ids[i] for i in bond['stereo_atom_ids']]
    graph['atoms'].reverse()
    graph['bonds'].reverse()
    assert canonical(to_molecule(graph)) == expected
    assert canonical(Chem.MolFromMolBlock(to_sdf(graph))) == expected


def test_overlapping_positions_are_preserved_and_not_used_for_identity():
    chemical = chemical_from_smiles('F[C@H](Cl)Br')
    render = {'molecule_id': 'fixture', 'variant': 'overlap',
              'atoms': [{'atom_id': a['id'], 'image_x': 123.4, 'image_y': 234.5} for a in chemical['atoms']]}
    graph = from_chemmol(chemical, render)
    before = deepcopy(graph)
    expected = canonical(to_molecule(graph))
    to_sdf(graph)
    assert graph == before
    assert len(graph['observations']['atom_positions']) == 4
    graph['observations'] = {'arbitrary_external_evidence': 'must not affect chemistry'}
    assert canonical(to_molecule(graph)) == expected


@pytest.mark.parametrize('kind', ['duplicate_atom', 'duplicate_bond', 'dangling', 'self', 'duplicate_pair',
                                  'bad_stereo', 'bad_neighbor', 'bad_charge', 'bad_aromatic', 'empty'])
def test_invalid_graph_fails_closed(kind):
    graph = from_chemmol(chemical_from_smiles('F[C@H](Cl)Br'))
    if kind == 'duplicate_atom':
        graph['atoms'].append(deepcopy(graph['atoms'][0]))
    elif kind == 'duplicate_bond':
        graph['bonds'].append(deepcopy(graph['bonds'][0]))
    elif kind == 'dangling':
        graph['bonds'][0]['atom_ids'][0] = 999
    elif kind == 'self':
        graph['bonds'][0]['atom_ids'] = [0, 0]
    elif kind == 'duplicate_pair':
        duplicate = deepcopy(graph['bonds'][0])
        duplicate['id'] = 99
        graph['bonds'].append(duplicate)
    elif kind == 'bad_stereo':
        graph['atoms'][1]['stereo']['tag'] = 'CHI_SQUAREPLANAR'
    elif kind == 'bad_neighbor':
        graph['atoms'][1]['stereo']['neighbor_ids'][0] = 999
    elif kind == 'bad_charge':
        graph['atoms'][0]['formal_charge'] = True
    elif kind == 'bad_aromatic':
        graph['bonds'][0]['aromatic'] = True
    elif kind == 'empty':
        graph['atoms'] = []
    with pytest.raises(ValueError):
        to_molecule(graph)


def test_invalid_double_bond_stereo_reference_rejected():
    graph = from_chemmol(chemical_from_smiles('F/C=C/Cl'))
    bond = next(b for b in graph['bonds'] if b['type'] == 'DOUBLE')
    bond['stereo_atom_ids'] = list(bond['atom_ids'])
    with pytest.raises(ValueError, match='flank'):
        to_molecule(graph)


def test_invalid_observations_rejected_by_adapter():
    chemical = chemical_from_smiles('CN')
    render = {'molecule_id': 'fixture', 'variant': 'test',
              'atoms': [{'atom_id': 0, 'image_x': float('nan'), 'image_y': 0},
                        {'atom_id': 1, 'image_x': 0, 'image_y': 0}]}
    with pytest.raises(ValueError, match='observation'):
        from_chemmol(chemical, render)


@pytest.mark.parametrize('smiles', SMILES)
def test_runner_checks_truth_roundtrip_and_random_relabeling(smiles):
    from chemvector.experiments.id_graph.roundtrip import check_sample

    chemical = chemical_from_smiles(smiles)
    render = {'molecule_id': 'fixture', 'variant': 'overlap',
              'atoms': [{'atom_id': a['id'], 'image_x': 12.3, 'image_y': 45.6} for a in chemical['atoms']]}
    source = Chem.MolToMolBlock(Chem.MolFromSmiles(smiles))
    result, graph, sdf = check_sample(chemical, render, source, [0, 17, 2026])
    assert result['all_checks_pass'], result
    assert len(graph['atoms']) == len(chemical['atoms'])
    assert sdf.endswith('$$$$\n')


def test_scorer_keeps_isotope_charge_and_stereo_separate():
    from chemvector.experiments.id_graph.roundtrip import signature

    assert signature(Chem.MolFromSmiles('[13CH4]'), False) != signature(Chem.MolFromSmiles('C'), False)
    assert signature(Chem.MolFromSmiles('[NH4+]'), False) != signature(Chem.MolFromSmiles('N'), False)
    left, right = [Chem.MolFromSmiles(s) for s in ('F[C@H](Cl)Br', 'F[C@@H](Cl)Br')]
    assert signature(left, False) == signature(right, False)
    assert signature(left) != signature(right)


def test_source_sdf_is_only_used_for_scoring():
    from chemvector.experiments.id_graph.roundtrip import check_sample

    chemical = chemical_from_smiles('CN')
    render = {'molecule_id': 'fixture', 'variant': 'fixture',
              'atoms': [{'atom_id': a['id'], 'image_x': 0, 'image_y': 0} for a in chemical['atoms']]}
    result, graph, sdf = check_sample(chemical, render, Chem.MolToMolBlock(Chem.MolFromSmiles('CO')), [0])
    assert not result['direct_core_exact'] and not result['sdf_core_exact']
    assert not result['all_checks_pass']
    assert canonical(Chem.MolFromMolBlock(sdf)) == 'CN'
    assert graph['atoms'][1]['element'] == 'N'


def test_native_array_order_does_not_change_stereo():
    chemical = chemical_from_smiles('C[C@H]1CCCCO1')
    expected = from_chemmol(chemical)
    chemical['atoms'].reverse()
    chemical['bonds'].reverse()
    assert from_chemmol(chemical) == expected


def test_frozen_output_is_never_overwritten(tmp_path):
    from chemvector.experiments.id_graph.roundtrip import freeze

    with pytest.raises(FileExistsError):
        freeze(tmp_path)
