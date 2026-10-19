"""Synthetic CPU contracts: no checkpoints, corpus images or optimizer steps."""

from types import SimpleNamespace

import pytest
from rdkit import Chem
import torch

from chemvector.direct_atom_pair_bond import direct_atom_pair_loss
from chemvector.ocr import parse_atom_display_text
from chemvector.train_direct_atom_pair_bond import match_atom_targets, select_training_pairs
from chemvector.training.curriculum.supervision_contract import (
    pair_targets_with_unknown, select_known_pair_rows, chemical_bond_labels,
    compiled_atom_fields, compare_text_result,
)
from chemvector.training.curriculum.validation_truth import canonical_molblock


def pair_fixture():
    # Known path 0--1--2; candidate 3 remains unmatched. The 0--2 non-edge is
    # a real structured negative. Unknown pairs must not erase that signal.
    atoms = {'atom_mask': torch.tensor([[True, True, True, True]]),
             'atom_indices': torch.tensor([[0, 1, 2, 3]]),
             'atom_geometry': torch.tensor([[[.2, .3], [.4, .3], [.6, .3], [.8, .3]]])}
    targets = {'atom_mask': torch.tensor([[True, True, True, True]]),
               'atom_anchors': atoms['atom_geometry'].clone(),
               'bond_mask': torch.tensor([[True, True, True]]),
               'bond_atom_refs': torch.tensor([[[0, 1], [1, 2], [2, 3]]]),
               'bond_chemical': torch.tensor([[0, 1, 0]])}
    mapping = torch.tensor([[0, 1, 2, -1]])
    return atoms, mapping, targets


def test_unknown_is_neither_positive_nor_confirmed_negative_and_denominator_stays():
    atoms, mapping, targets = pair_fixture()
    original_mapping = mapping.clone()
    original_mask = atoms['atom_mask'].clone()
    contract = pair_targets_with_unknown(atoms, mapping, targets)
    assert contract['pair_indices'].tolist() == [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    assert contract['known_mask'].tolist() == [True, True, False, True, False, False]
    assert contract['legacy_targets']['presence'].tolist() == [True, False, False, True, False, False]
    assert contract['coverage'] == dict(target_atoms=4, selected_atoms=4, unmatched_selected_atoms=1,
                                      target_bonds=3, reachable_positive_pairs=2,
                                      unreachable_bonds=1, known_negative_pairs=1, unknown_pairs=3)
    chosen = select_known_pair_rows(contract, atoms)
    assert set(chosen.tolist()) == {0, 1, 3}
    assert contract['legacy_targets']['structured_negative'][1]
    assert torch.equal(mapping, original_mapping) and torch.equal(atoms['atom_mask'], original_mask)


def test_existing_sampler_can_select_unknown_as_negative_but_prototype_cannot():
    atoms, mapping, targets = pair_fixture()
    contract = pair_targets_with_unknown(atoms, mapping, targets)
    old = select_training_pairs(contract['pair_indices'], contract['sample_indices'], atoms,
                                contract['legacy_targets'], negative_ratio=6, minimum_negatives=48)
    assert set(old.tolist()) == set(range(6))
    assert bool(contract['unknown_mask'][old].any())
    chosen = select_known_pair_rows(contract, atoms)
    logits = torch.zeros(6, requires_grad=True)
    chemistry = torch.zeros(6, 4, requires_grad=True)
    target = contract['legacy_targets']
    loss = direct_atom_pair_loss({'presence_logits': logits[chosen], 'chemical_logits': chemistry[chosen]},
                                target['presence'][chosen], target['chemical'][chosen])['loss']
    loss.backward()  # Synthetic logits only, no real model or optimizer.
    assert torch.equal(logits.grad[contract['unknown_mask']], torch.zeros(3))
    assert bool((logits.grad[target['presence']] < 0).all())
    assert logits.grad[1] > 0  # The confirmed non-edge is still penalized.


def test_fully_matched_candidates_keep_legacy_sampling_with_same_rng():
    atoms, mapping, targets = pair_fixture()
    mapping[0, 3] = 3
    contract = pair_targets_with_unknown(atoms, mapping, targets)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20260918)
        old = select_training_pairs(contract['pair_indices'], contract['sample_indices'], atoms,
                                    contract['legacy_targets'], negative_ratio=0, minimum_negatives=1)
        torch.manual_seed(20260918)
        current = select_known_pair_rows(contract, atoms, negative_ratio=0, minimum_negatives=1)
    assert torch.equal(current, old)
    assert contract['coverage']['unknown_pairs'] == contract['coverage']['unreachable_bonds'] == 0


@pytest.mark.parametrize('budget', [-1, 1.5, True])
def test_invalid_negative_sampling_budgets_are_rejected(budget):
    atoms, mapping, targets = pair_fixture()
    contract = pair_targets_with_unknown(atoms, mapping, targets)
    with pytest.raises(ValueError):
        select_known_pair_rows(contract, atoms, negative_ratio=budget)


@pytest.mark.parametrize('offset,reachable,unknown', [(6.0, 3, 0), (6.25, 2, 3)])
def test_contract_does_not_relax_real_six_pixel_matcher(offset, reachable, unknown):
    atoms, _, targets = pair_fixture()
    atoms['atom_geometry'][0, 3, 0] += offset / 768
    mapping, _ = match_atom_targets(atoms, targets, SimpleNamespace(image_size=768, atom_match_radius_px=6))
    contract = pair_targets_with_unknown(atoms, mapping, targets)
    assert contract['coverage']['reachable_positive_pairs'] == reachable
    assert contract['coverage']['unknown_pairs'] == unknown
    assert contract['coverage']['target_bonds'] == 3


@pytest.mark.parametrize('selected', [[], [0], [0, 1, 2, 3]])
def test_no_known_pairs_means_no_sampled_loss_rows(selected):
    atoms, mapping, targets = pair_fixture()
    atoms['atom_mask'].zero_()
    atoms['atom_mask'][0, selected] = True
    mapping.fill_(-1)
    contract = pair_targets_with_unknown(atoms, mapping, targets)
    assert select_known_pair_rows(contract, atoms).numel() == 0
    assert contract['coverage']['target_bonds'] == contract['coverage']['unreachable_bonds'] == 3
    assert contract['coverage']['unknown_pairs'] == len(selected) * (len(selected) - 1) // 2


def test_batch_assignments_and_partial_targets_do_not_cross_samples():
    atoms, mapping, targets = pair_fixture()
    atoms = {k: torch.cat([v, v]) for k, v in atoms.items()}
    targets = {k: torch.cat([v, v]) for k, v in targets.items()}
    mapping = torch.cat([mapping, torch.tensor([[0, 1, -1, -1]])])
    targets['bond_mask'][1, 1:] = False
    targets['bond_atom_refs'][1, 0] = torch.tensor([0, 3])
    contract = pair_targets_with_unknown(atoms, mapping, targets)
    assert contract['coverage']['target_bonds'] == 4
    assert contract['coverage']['reachable_positive_pairs'] == 2
    second = contract['sample_indices'] == 1
    assert contract['known_mask'][second].tolist() == [True, False, False, False, False, False]
    assert not bool(contract['legacy_targets']['presence'][second].any())
    chosen = select_known_pair_rows(contract, atoms)
    assert set(chosen.tolist()) == {0, 1, 3, 6}


@pytest.mark.parametrize('bad', ['duplicate_match', 'target_oob', 'negative_id', 'unselected',
                                'atom_hole', 'bond_hole', 'duplicate_edge', 'self_edge',
                                'dangling_edge', 'bad_type', 'mapping_dtype'])
def test_malformed_target_contract_fails_closed(bad):
    atoms, mapping, targets = pair_fixture()
    if bad == 'duplicate_match': mapping[0, 1] = 0
    elif bad == 'target_oob': mapping[0, 0] = 4
    elif bad == 'negative_id': mapping[0, 0] = -2
    elif bad == 'unselected': atoms['atom_mask'][0, 0] = False
    elif bad == 'atom_hole': targets['atom_mask'][0, 1] = False
    elif bad == 'bond_hole': targets['bond_mask'][0, 1] = False
    elif bad == 'duplicate_edge': targets['bond_atom_refs'][0, 1] = torch.tensor([1, 0])
    elif bad == 'self_edge': targets['bond_atom_refs'][0, 0] = torch.tensor([0, 0])
    elif bad == 'dangling_edge': targets['bond_atom_refs'][0, 0] = torch.tensor([0, 4])
    elif bad == 'bad_type': targets['bond_chemical'][0, 0] = 4
    elif bad == 'mapping_dtype': mapping = mapping.float()
    with pytest.raises(ValueError):
        pair_targets_with_unknown(atoms, mapping, targets)


@pytest.mark.parametrize('smiles', ['c1ccccc1', 'c1cc[nH]c1', 'c1ccncc1', '[13CH3]c1ccccc1',
                                   'C[C@H](F)c1ccccc1', 'c1ccccc1/C=C/Cl'])
def test_whole_graph_canonicalizes_kekule_labels_without_mutating_drawing(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    expected = chemical_bond_labels(molecule)
    drawn = Chem.Mol(molecule)
    Chem.Kekulize(drawn, clearAromaticFlags=True)
    before = drawn.ToBinary()
    assert chemical_bond_labels(drawn) == expected
    assert drawn.ToBinary() == before
    assert any(v == 'AROMATIC' for v in expected.values())


def test_aromatic_and_single_are_not_interchangeable_without_graph_context():
    biphenyl = Chem.MolFromSmiles('c1ccccc1-c1ccccc1')
    labels = chemical_bond_labels(biphenyl)
    bridge = next(b for b in biphenyl.GetBonds() if not b.GetIsAromatic())
    assert labels[tuple(sorted((bridge.GetBeginAtomIdx(), bridge.GetEndAtomIdx())))] == 'SINGLE'
    assert set(chemical_bond_labels(Chem.MolFromSmiles('C1CCCCC1')).values()) == {'SINGLE'}
    # Aromaticity is not a license to accept any ring as benzene.
    assert canonical_molblock(Chem.MolToMolBlock(Chem.MolFromSmiles('C1CCCCC1'))) != canonical_molblock(
        Chem.MolToMolBlock(Chem.MolFromSmiles('c1ccccc1')))


@pytest.mark.parametrize('alternating,exact', [('SINGLE', True), ('DOUBLE', False)])
def test_raw_type_errors_can_be_normalized_but_not_automatically_forgiven(alternating, exact):
    raw = Chem.RWMol()
    for _ in range(6):
        raw.AddAtom(Chem.Atom('C'))
    for i, kind in enumerate(('AROMATIC', alternating) * 3):
        raw.AddBond(i, (i + 1) % 6, Chem.BondType.names[kind])
    raw = raw.GetMol()
    assert sum(str(b.GetBondType()) != 'AROMATIC' for b in raw.GetBonds()) == 3
    compiled = Chem.Mol(raw)
    Chem.SanitizeMol(compiled)
    expected = Chem.MolFromSmiles('c1ccccc1')
    assert (chemical_bond_labels(compiled) == chemical_bond_labels(expected)) == exact
    assert (canonical_molblock(Chem.MolToMolBlock(compiled))
            == canonical_molblock(Chem.MolToMolBlock(expected))) == exact


@pytest.mark.parametrize('smiles', ['c:c', 'C(C)(C)(C)(C)C'])
def test_invalid_chemistry_does_not_become_a_canonical_training_target(smiles):
    with pytest.raises(ValueError):
        chemical_bond_labels(Chem.MolFromSmiles(smiles, sanitize=False))


def atom(smiles, index=0):
    return Chem.MolFromSmiles(smiles).GetAtomWithIdx(index)


def test_reversed_gt_text_need_not_parse_as_formula_to_score_compiled_attributes():
    assert parse_atom_display_text('HN') is None  # Current parser, not silently replaced.
    assert parse_atom_display_text('NH')['explicit_h_count'] == 1
    result = compare_text_result('HN', 'NH', expected_atom=atom('CNC', 1), compiled_atom=atom('C[NH]C', 1))
    assert result == {'literal_match': False, 'compiled_fields_match': True, 'differences': {}}


@pytest.mark.parametrize('smiles,field', [('P', 'element'), ('[NH4+]', 'formal_charge'),
                                      ('[15NH3]', 'isotope'), ('[NH2]', 'radical_electrons')])
def test_real_text_attribute_errors_remain_errors(smiles, field):
    result = compare_text_result('H2N', 'PH2', expected_atom=atom('N'), compiled_atom=atom(smiles))
    assert result['compiled_fields_match'] is False and field in result['differences']


def test_hydrogen_comparison_uses_compiled_context_not_suffix_or_storage_convention():
    implicit, explicit = atom('CNC', 1), atom('C[NH]C', 1)
    assert implicit.GetNumExplicitHs() != explicit.GetNumExplicitHs()
    assert compiled_atom_fields(implicit) == compiled_atom_fields(explicit)
    result = compare_text_result('HN', 'N', expected_atom=implicit, compiled_atom=implicit)
    assert not result['literal_match'] and result['compiled_fields_match']
    # Same glyphs cannot hide a connectivity-induced H count difference.
    result = compare_text_result('N', 'N', expected_atom=atom('N'), compiled_atom=atom('CN(C)C', 1))
    assert result['literal_match'] and not result['compiled_fields_match']
    assert result['differences']['total_h_count'] == {'expected': 3, 'predicted': 0}


def test_missing_compiled_atom_is_unscorable_not_a_false_match():
    result = compare_text_result('N', 'N', expected_atom=atom('N'), compiled_atom=None)
    assert result['literal_match'] and result['compiled_fields_match'] is None
    assert result['differences'] is None


@pytest.mark.parametrize('left,right', [('F[C@H](Cl)Br', 'F[C@@H](Cl)Br'),
                                     ('F/C=C/Cl', 'F/C=C\\Cl')])
def test_local_contract_cannot_replace_strict_stereo_identity(left, right):
    first, second = Chem.MolFromSmiles(left), Chem.MolFromSmiles(right)
    assert chemical_bond_labels(first) == chemical_bond_labels(second)
    assert [compiled_atom_fields(a) for a in first.GetAtoms()] == [compiled_atom_fields(a) for a in second.GetAtoms()]
    assert canonical_molblock(Chem.MolToMolBlock(first)) != canonical_molblock(Chem.MolToMolBlock(second))
