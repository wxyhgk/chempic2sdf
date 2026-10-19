import pytest
from rdkit import Chem

from chemvector.experiments.external_ocsr.common import validate_rows
from chemvector.experiments.external_ocsr.score import counts, score_prediction, verify_receipt
from chemvector.experiments.id_graph.roundtrip import signature


def example(smiles='CCO'):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToMolBlock(mol), {'core': signature(mol, False), 'stereo': signature(mol)}


def invocation(annotations=None, error=None):
    return {'sample_key': 'sample', 'invocation_error': error, 'seconds': 1.,
            'annotations': [{'smi': 'CCO', 'conf': 0.9}] if annotations is None else annotations}


def test_complete_native_prediction_scores():
    block, truth = example()
    row = score_prediction(invocation(), block, truth)
    assert row['sdf_valid'] and row['core_graph_exact'] and row['stereo_graph_exact']


@pytest.mark.parametrize('call', [invocation(error={'type': 'Error'}), invocation(annotations=[]),
                                invocation(annotations=[{}, {}])])
def test_partial_output_cannot_rescue_failed_or_ambiguous_invocation(call):
    block, truth = example()
    row = score_prediction(call, block, truth)
    assert not row['sdf_valid'] and not row['core_graph_exact'] and not row['stereo_graph_exact']


def test_native_carbon_placeholder_is_valid_but_wrong_not_dropped():
    block, _ = example('C')
    _, truth = example()
    row = score_prediction(invocation([{'smi': 'C', 'conf': 0}]), block, truth)
    result = counts([row])
    assert result['images'] == result['sdf_valid'] == result['valid_but_wrong'] == result['native_carbon_placeholders'] == 1
    assert result['core_graph_exact'] == 0


def test_missing_sdf_does_not_fall_back_to_correct_annotation_smiles():
    _, truth = example()
    row = score_prediction(invocation(), '', truth)
    assert not row['sdf_valid'] and not row['core_graph_exact']


def test_no_stereo_predictions_can_pass_core_but_not_stereo():
    block, _ = example('CC(O)F')
    _, truth = example('C[C@H](O)F')
    row = score_prediction(invocation(), block, truth)
    assert row['core_graph_exact'] and not row['stereo_graph_exact']


@pytest.mark.parametrize('keys', [['a'], ['a', 'a'], ['b', 'a'], ['a', 'c']])
def test_missing_duplicate_reordered_and_wrong_ids_rejected(keys):
    with pytest.raises(ValueError):
        validate_rows([{'sample_key': k} for k in keys], ['a', 'b'])


def test_complete_order_accepted():
    validate_rows([{'sample_key': k} for k in ['a', 'b']], ['a', 'b'])


def test_smoke_receipt_not_accepted_for_scoring(tmp_path):
    with pytest.raises(ValueError, match='smoke'):
        verify_receipt(tmp_path, {'completed': True, 'smoke_only': True, 'sample_count': 4})


def test_artifact_path_traversal_rejected(tmp_path):
    receipt = {'completed': True, 'smoke_only': False, 'sample_count': 200,
               'models_unchanged': True, 'truth_used_for_prediction': False,
               'artifacts': {'../outside': '0' * 64}}
    with pytest.raises(ValueError, match='Unsafe'):
        verify_receipt(tmp_path, receipt)
