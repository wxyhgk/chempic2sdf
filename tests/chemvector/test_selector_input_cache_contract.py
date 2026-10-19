from copy import deepcopy

import pytest
import torch

from tools.selector_input_cache_contract import make_receipt, require_exact_replay


def inputs():
    raw = dict(atom_scores=torch.tensor([[.8, .1, -1.]]), atom_indices=torch.tensor([[0, 1, 0]]),
        atom_mask=torch.tensor([[True, True, False]]), atom_features=torch.zeros(1, 3, 4),
        atom_geometry=torch.zeros(1, 3, 8))
    bonds = dict(bond_scores=torch.tensor([[.7, .3]]), bond_indices=torch.tensor([[1, 5]]),
        bond_mask=torch.tensor([[True, True]]), bond_features=torch.zeros(1, 2, 4),
        bond_endpoints=torch.zeros(1, 2, 2, 2), bond_types=torch.zeros(1, 2, 7))
    context = dict(sample_key='synthetic', image_sha256='1'*64, checkpoint_sha256='2'*64,
        implementation_sha256='3'*64, atom_grid_shape=[1, 2, 2], bond_grid_shape=[2, 2, 2],
        decoder=dict(atom_policy='exact-probability-desc-flat-id-asc-v1',
            bond_policy='exact-probability-desc-flat-id-asc-v1', maximum_atoms=3, maximum_bonds=2,
            atom_threshold=.01, bond_threshold=.01, image_size=768),
        numerics=dict(device='cpu', dtype='float32', torch_version='test-version', autocast=False, tf32=False))
    return raw, bonds, context


def test_complete_context_and_all_tensors_have_content_receipts_not_training_authority():
    raw, bonds, context = inputs(); receipt = make_receipt(raw, bonds, context)
    assert len(receipt['raw']) == 5 and len(receipt['bonds']) == 6
    assert require_exact_replay(receipt, raw, bonds, context) == receipt['input_identity_sha256']
    assert not receipt['training_authorized'] and not receipt['contains_labels']
    assert not receipt['tensors_persisted_by_this_function']
    assert bonds['bond_indices'].tolist() == [[1, 5]]  # Same xy, distinct bond channels.
    context['decoder']['atom_policy'] = 'changed'
    assert receipt['context']['decoder']['atom_policy'] != 'changed'


@pytest.mark.parametrize('change', ['atom_feature', 'unselected_atom_feature', 'bond_feature', 'geometry',
    'endpoints', 'types', 'atom_order', 'bond_order', 'image', 'checkpoint', 'policy', 'implementation', 'receipt'])
def test_any_context_or_input_change_refuses_reuse(change):
    raw, bonds, context = inputs(); receipt = make_receipt(raw, bonds, context)
    if change == 'atom_feature': raw['atom_features'][0, 0, 0] = 1
    if change == 'unselected_atom_feature': raw['atom_features'][0, 2, 0] = 1
    if change == 'bond_feature': bonds['bond_features'][0, 0, 0] = 1
    if change == 'geometry': raw['atom_geometry'][0, 0, 0] = .1
    if change == 'endpoints': bonds['bond_endpoints'][0, 0, 0, 0] = .1
    if change == 'types': bonds['bond_types'][0, 0, 0] = .1
    if change == 'atom_order': raw = {k: v[:, [1, 0, 2]] for k, v in raw.items()}
    if change == 'bond_order': bonds = {k: v[:, [1, 0]] for k, v in bonds.items()}
    if change == 'image': context['image_sha256'] = '4'*64
    if change == 'checkpoint': context['checkpoint_sha256'] = '4'*64
    if change == 'implementation': context['implementation_sha256'] = '4'*64
    if change == 'policy': context['decoder']['atom_policy'] = 'legacy'
    if change == 'receipt': receipt['raw']['atom_features']['sha256'] = '4'*64
    with pytest.raises(ValueError): require_exact_replay(receipt, raw, bonds, context)


@pytest.mark.parametrize('problem', ['only_scores', 'labels', 'missing_bond', 'grad', 'nan', 'float64',
    'mask', 'atom_duplicate', 'bond_duplicate', 'shape', 'flat_id', 'hash', 'numerics', 'context_label', 'threshold'])
def test_incomplete_or_invalid_cache_rejected(problem):
    raw, bonds, context = inputs()
    if problem == 'only_scores': raw = dict(atom_scores=raw['atom_scores'])
    if problem == 'labels': raw['target_id'] = torch.tensor([0])
    if problem == 'missing_bond': bonds.pop('bond_endpoints')
    if problem == 'grad': raw['atom_features'].requires_grad_()
    if problem == 'nan': raw['atom_features'][0, 0, 0] = float('nan')
    if problem == 'float64': raw['atom_geometry'] = raw['atom_geometry'].double()
    if problem == 'mask': raw['atom_mask'][0, 2] = True
    if problem == 'atom_duplicate': raw['atom_indices'][0, 1] = 0
    if problem == 'bond_duplicate': bonds['bond_indices'][0, 1] = 1
    if problem == 'shape': bonds['bond_endpoints'] = torch.zeros(1, 2, 4)
    if problem == 'flat_id': bonds['bond_indices'][0, 1] = 8
    if problem == 'hash': context['checkpoint_sha256'] = 'short'
    if problem == 'numerics': context['numerics']['autocast'] = True
    if problem == 'context_label': context['labels'] = [0]
    if problem == 'threshold': context['decoder']['atom_threshold'] = float('nan')
    with pytest.raises(ValueError): make_receipt(raw, bonds, context)


def test_mapping_order_and_contiguous_storage_do_not_change_content_identity():
    raw, bonds, context = inputs(); receipt = make_receipt(raw, bonds, context)
    copied = {k: v.transpose(-1, -2).contiguous().transpose(-1, -2) if v.ndim >= 3 else v.clone()
              for k, v in reversed(list(raw.items()))}
    assert require_exact_replay(receipt, copied, deepcopy(bonds), deepcopy(context))
