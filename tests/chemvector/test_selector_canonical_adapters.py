import pytest
import torch

from chemvector.components.graph.joint.predicted_incidence import (
    PredictedProposalConfig, decode_predicted_atom_proposals, decode_predicted_bond_proposals,
)
from tools.selector_peak_tie_contract import canonical_atom_proposals, canonical_bond_proposals


def atom_output():
    heat = torch.full((1, 1, 7, 7), -10.)
    heat[0, 0, 0, 0], heat[0, 0, 0, 6], heat[0, 0, 6, 6] = 3., 2., 1.
    return dict(heatmap_logits=heat, object_features=torch.arange(147.).reshape(1, 3, 7, 7),
        anchors=torch.rand(1, 2, 7, 7), role_logits=torch.rand(1, 2, 7, 7),
        text_centers=torch.rand(1, 2, 7, 7), text_extents=torch.rand(1, 2, 7, 7))


def bond_output():
    heat = torch.full((1, 2, 7, 7), -10.)
    heat[0, 0, 0, 0], heat[0, 1, 0, 0], heat[0, 1, 6, 6] = 3., 2., 1.
    return dict(heatmap_logits=heat, object_features=torch.arange(147.).reshape(1, 3, 7, 7),
        endpoints=torch.rand(1, 2, 2, 2, 7, 7), display_logits=torch.rand(1, 2, 4, 7, 7),
        chemical_logits=torch.rand(1, 2, 5, 7, 7))


@pytest.mark.parametrize('make,old,new,kind', [
    (atom_output, decode_predicted_atom_proposals, canonical_atom_proposals, 'atom'),
    (bond_output, decode_predicted_bond_proposals, canonical_bond_proposals, 'bond')])
def test_unique_active_proposals_are_bitwise_equal_to_production(make, old, new, kind):
    torch.manual_seed(10)
    output = make(); before = {k: v.clone() for k, v in output.items()}
    config = PredictedProposalConfig(maximum_atoms=3, maximum_bonds=3)
    a, b = old(output, config), new(output, config)
    assert set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)
    assert all(torch.equal(v, before[k]) for k, v in output.items())


def test_tied_bond_slots_gather_their_own_channel_endpoints_and_types():
    output = bond_output(); output['heatmap_logits'].zero_()
    config = PredictedProposalConfig(maximum_bonds=51)
    result = canonical_bond_proposals(output, config)
    assert result['bond_indices'][0, 0] == 0 and result['bond_indices'][0, 49] == 49
    torch.testing.assert_close(result['bond_endpoints'][0, 0], output['endpoints'][0, 0, :, :, 0, 0])
    torch.testing.assert_close(result['bond_endpoints'][0, 49], output['endpoints'][0, 1, :, :, 0, 0])
    assert not torch.equal(result['bond_endpoints'][0, 0], result['bond_endpoints'][0, 49])
    assert torch.equal(result['bond_features'][0, 0], result['bond_features'][0, 49])


def test_multichannel_atom_input_is_not_silently_collapsed():
    output = atom_output(); output['heatmap_logits'] = output['heatmap_logits'].repeat(1, 2, 1, 1)
    with pytest.raises(ValueError): canonical_atom_proposals(output, PredictedProposalConfig())
