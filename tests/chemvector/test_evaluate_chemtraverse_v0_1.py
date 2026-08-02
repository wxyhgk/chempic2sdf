"""Tests for ChemTraverse v0.1 integration in the general evaluator."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

from rdkit import Chem
import torch

from Data.sdf.chemmol_tokenizer import ChemMolTokenizer
from Data.sdf.chemtraverse_tokenizer_v0_1 import ChemTraverseTokenizerV01
from Data.sdf.chemtraverse_v0_1 import (
    mol_from_chemtraverse_v0_1,
    serialize_chemtraverse_v0_1,
)
from chemvector.chemtraverse_teacher_metrics import (
    ChemTraverseGrammarMaskedTeacherCounts,
)
import chemvector.evaluate as evaluate_module
from chemvector.generation_chemtraverse_v0_1 import ChemTraverseGrammarV01
from tests.data_sdf.test_chemtraverse_v0_1 import ring_scene


def _write_sdf(sequence: str, path) -> None:
    mol = mol_from_chemtraverse_v0_1(sequence)
    writer = Chem.SDWriter(str(path))
    try:
        writer.write(mol)
    finally:
        writer.close()


def _batch(sequence: str, mol: Chem.Mol) -> dict:
    return {
        "keys": ["sample-1"],
        "target_sequences": [sequence],
        "expected_smiles": [
            Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        ],
        "expected_inchikeys": [Chem.MolToInchiKey(mol)],
        "pixel_values": torch.zeros(1, 3, 16, 16),
    }


def _result(tokenizer, sequence: str, *, chemically_valid: bool = True):
    return SimpleNamespace(
        input_ids=tuple(tokenizer.encode(sequence)),
        sequence=sequence,
        completed=True,
        chemically_valid=chemically_valid,
        error=None,
    )


def _teacher_fixture():
    tokenizer = ChemTraverseTokenizerV01()
    scene = ring_scene()
    scene["molecule_id"] = "sdf"
    scene["junctions"][0]["role"] = "TEXT"
    scene["junctions"][0]["atom"]["element"] = "N"
    scene["text_objects"] = [
        {
            "anchor": [100, 100],
            "center": [100, 100],
            "bbox": None,
            "text": "Ph",
        }
    ]
    input_ids = torch.tensor(
        [tokenizer.encode(serialize_chemtraverse_v0_1(scene))],
        dtype=torch.long,
    )
    logits = torch.full(
        (1, input_ids.size(1) - 1, tokenizer.vocab_size),
        -20.0,
    )
    logits[:, :, tokenizer.pad_token_id] = 20.0

    grammar = ChemTraverseGrammarV01(tokenizer)
    grammar.consume(int(input_ids[0, 0]))
    corrupted = set()
    for input_index, target_tensor in enumerate(input_ids[0, 1:], 1):
        target = int(target_tensor)
        token = tokenizer.id_to_token[target]
        allowed = grammar.allowed_token_ids()
        logits[0, input_index - 1, target] = 5.0
        context = grammar.record_type
        bucket = None
        if token == tokenizer.COMPONENT_END:
            bucket = "component_end"
        elif token == tokenizer.LINK:
            bucket = "link"
        elif token.startswith("byte:") and context == "meta":
            bucket = "meta"
        elif token.startswith("byte:") and context == "text":
            bucket = "text"
        if bucket is not None and bucket not in corrupted:
            alternative = next(token_id for token_id in allowed if token_id != target)
            logits[0, input_index - 1, alternative] = 10.0
            corrupted.add(bucket)
        grammar.consume(target)
    assert corrupted == {"component_end", "link", "meta", "text"}
    return tokenizer, input_ids, logits


def test_load_tokenizer_supports_chemtraverse_schema(tmp_path) -> None:
    path = tmp_path / "tokenizer.json"
    expected = ChemTraverseTokenizerV01()
    expected.save(path)

    actual = evaluate_module._load_tokenizer(path)

    assert isinstance(actual, ChemTraverseTokenizerV01)
    assert actual.vocabulary == expected.vocabulary


def test_grammar_masked_teacher_metrics_separate_real_text_from_meta() -> None:
    tokenizer, input_ids, logits = _teacher_fixture()
    counts = ChemTraverseGrammarMaskedTeacherCounts()

    counts.update(logits, input_ids, tokenizer)
    metrics = counts.metrics()

    assert metrics["valid_positions"] == input_ids.size(1) - 1
    assert metrics["decision_positions"] + metrics["forced_positions"] == metrics[
        "valid_positions"
    ]
    assert metrics["non_forced_decision_correct"] == (
        metrics["non_forced_decision_total"] - 4
    )
    assert metrics["control_tokens"][tokenizer.COMPONENT_END] == {
        "correct": 0,
        "total": 1,
        "accuracy": 0.0,
    }
    assert metrics["control_tokens"][tokenizer.LINK] == {
        "correct": 0,
        "total": 1,
        "accuracy": 0.0,
    }
    assert metrics["byte_contexts"]["meta_molecule_id"] == {
        "correct": 2,
        "total": 3,
        "accuracy": 2 / 3,
    }
    assert metrics["byte_contexts"]["text_object"] == {
        "correct": 1,
        "total": 2,
        "accuracy": 0.5,
    }


def test_unified_teacher_evaluator_emits_chemtraverse_masked_metrics() -> None:
    tokenizer, input_ids, logits = _teacher_fixture()

    class Model:
        config = SimpleNamespace(pad_token_id=tokenizer.pad_token_id)

        def __call__(self, pixel_values, actual_ids, attention_mask):
            assert torch.equal(actual_ids, input_ids)
            return {"logits": logits, "loss": logits.new_zeros(())}

    loader = [
        {
            "pixel_values": torch.zeros(1, 3, 16, 16),
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }
    ]
    with mock.patch.object(
        evaluate_module.torch,
        "autocast",
        return_value=nullcontext(),
    ):
        metrics = evaluate_module.evaluate_teacher_forcing(
            Model(),
            loader,
            torch.device("cpu"),
            1,
            tokenizer,
        )

    assert metrics["token_accuracy"] == 0.0
    masked = metrics["chemtraverse_grammar_masked"]
    assert masked["non_forced_decision_correct"] == (
        masked["non_forced_decision_total"] - 4
    )
    assert masked["control_tokens"][tokenizer.COMPONENT_END]["total"] == 1
    assert masked["control_tokens"][tokenizer.LINK]["total"] == 1
    assert masked["byte_contexts"]["meta_molecule_id"]["total"] == 3
    assert masked["byte_contexts"]["text_object"]["total"] == 2


def test_unified_teacher_evaluator_leaves_other_tokenizers_unchanged() -> None:
    tokenizer = ChemMolTokenizer()
    sequence = """<mol v=0.2>
<a 0 C 0 0 0 0 0 0 CHI_UNSPECIFIED 100 200>
<mol_end>
"""
    input_ids = torch.tensor([tokenizer.encode(sequence)], dtype=torch.long)
    logits = torch.zeros(1, input_ids.size(1) - 1, tokenizer.vocab_size)
    for input_index, target in enumerate(input_ids[0, 1:]):
        logits[0, input_index, int(target)] = 1.0

    class Model:
        config = SimpleNamespace(pad_token_id=tokenizer.pad_token_id)

        def __call__(self, pixel_values, actual_ids, attention_mask):
            return {"logits": logits, "loss": logits.new_zeros(())}

    loader = [
        {
            "pixel_values": torch.zeros(1, 3, 16, 16),
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }
    ]
    with mock.patch.object(
        evaluate_module.torch,
        "autocast",
        return_value=nullcontext(),
    ):
        metrics = evaluate_module.evaluate_teacher_forcing(
            Model(),
            loader,
            torch.device("cpu"),
            1,
            tokenizer,
        )

    assert metrics["token_accuracy"] == 1.0
    assert "chemtraverse_grammar_masked" not in metrics


def test_generation_exports_scene_sdf_and_stage_metrics(tmp_path) -> None:
    tokenizer = ChemTraverseTokenizerV01()
    sequence = serialize_chemtraverse_v0_1(ring_scene())
    expected_mol = mol_from_chemtraverse_v0_1(sequence)
    generated = _result(tokenizer, sequence)

    with (
        mock.patch.object(
            evaluate_module,
            "generate_chemtraverse_v0_1",
            return_value=generated,
        ) as generate,
        mock.patch.object(
            evaluate_module,
            "write_sdf_chemtraverse_v0_1",
            side_effect=_write_sdf,
        ),
    ):
        summary = evaluate_module.evaluate_generation(
            None,
            [_batch(sequence, expected_mol)],
            tokenizer,
            torch.device("cpu"),
            1,
            tmp_path,
            beam_width=3,
        )

    generate.assert_called_once()
    assert generate.call_args.kwargs == {"beam_width": 3}
    assert summary["sequence_schema_version"] == "chemtraverse-0.1"
    assert summary["grammar_valid"] == 1
    assert summary["scene_valid"] == 1
    assert summary["mol_valid"] == 1
    assert summary["sdf_valid"] == 1
    assert summary["sdf_roundtrip_matches"] == 1
    assert summary["chemical_matches"] == 1
    assert summary["count_pair_exact"] == 1
    assert summary["skeleton_exact"] == 1
    assert summary["typed_graph_exact"] == 1
    assert summary["failure_buckets"] == {"correct": 1}
    assert summary["chemical_match_rate"] == 1.0
    assert summary["records"][0]["error"] is None
    assert summary["oracle_counts"] is False
    assert summary["records"][0]["oracle_counts"] is False
    assert (tmp_path / "sample-1.chemtraverse.txt").read_text() == sequence
    assert (tmp_path / "sample-1.chemscene.json").is_file()
    assert (tmp_path / "sample-1.sdf").is_file()
    assert (tmp_path / "generation_metrics.json").is_file()


def test_generation_passes_explicit_gt_counts_as_oracle(tmp_path) -> None:
    tokenizer = ChemTraverseTokenizerV01()
    scene = ring_scene()
    sequence = serialize_chemtraverse_v0_1(scene)
    expected_mol = mol_from_chemtraverse_v0_1(sequence)
    generated = _result(tokenizer, sequence)

    with (
        mock.patch.object(
            evaluate_module,
            "generate_chemtraverse_v0_1",
            return_value=generated,
        ) as generate,
        mock.patch.object(
            evaluate_module,
            "write_sdf_chemtraverse_v0_1",
            side_effect=_write_sdf,
        ),
    ):
        summary = evaluate_module.evaluate_generation(
            None,
            [_batch(sequence, expected_mol)],
            tokenizer,
            torch.device("cpu"),
            1,
            tmp_path,
            beam_width=2,
            oracle_counts=True,
        )

    assert generate.call_args.kwargs == {
        "beam_width": 2,
        "expected_atom_count": len(scene["junctions"]),
        "expected_bond_count": len(scene["bond_objects"]),
    }
    assert summary["oracle_counts"] is True
    assert summary["records"][0]["oracle_counts"] is True


def test_scene_can_be_valid_before_mol_and_sdf_validation(tmp_path) -> None:
    tokenizer = ChemTraverseTokenizerV01()
    scene = ring_scene()
    scene["junctions"][0]["atom"]["element"] = "F"
    sequence = serialize_chemtraverse_v0_1(scene)
    expected_mol = Chem.MolFromSmiles("C")
    generated = _result(tokenizer, sequence, chemically_valid=False)

    with (
        mock.patch.object(
            evaluate_module,
            "generate_chemtraverse_v0_1",
            return_value=generated,
        ),
        mock.patch.object(
            evaluate_module,
            "write_sdf_chemtraverse_v0_1",
        ) as write_sdf,
    ):
        summary = evaluate_module.evaluate_generation(
            None,
            [_batch(sequence, expected_mol)],
            tokenizer,
            torch.device("cpu"),
            1,
            tmp_path,
        )

    record = summary["records"][0]
    assert record["grammar_valid"] is True
    assert record["scene_valid"] is True
    assert record["mol_valid"] is False
    assert record["sdf_valid"] is False
    assert record["count_pair_exact"] is True
    assert record["skeleton_exact"] is True
    assert record["mol_error"]
    assert summary["chemical_matches"] == 0
    write_sdf.assert_not_called()


def test_malformed_sequence_is_reported_without_aborting_evaluation(tmp_path) -> None:
    tokenizer = ChemTraverseTokenizerV01()
    expected_sequence = serialize_chemtraverse_v0_1(ring_scene())
    expected_mol = mol_from_chemtraverse_v0_1(expected_sequence)
    malformed = "<chemtraverse v=0.1 units=q1023>\n<chemtraverse_end>\n"
    generated = SimpleNamespace(
        input_ids=(tokenizer.bos_token_id, tokenizer.eos_token_id),
        sequence=malformed,
        completed=True,
        chemically_valid=False,
        error=None,
    )

    with (
        mock.patch.object(
            evaluate_module,
            "generate_chemtraverse_v0_1",
            return_value=generated,
        ),
        mock.patch.object(
            evaluate_module,
            "write_sdf_chemtraverse_v0_1",
        ) as write_sdf,
    ):
        summary = evaluate_module.evaluate_generation(
            None,
            [_batch(expected_sequence, expected_mol)],
            tokenizer,
            torch.device("cpu"),
            1,
            tmp_path,
        )

    record = summary["records"][0]
    assert record["grammar_valid"] is False
    assert record["scene_valid"] is False
    assert record["mol_valid"] is False
    assert record["sdf_valid"] is False
    assert record["count_pair_exact"] is False
    assert record["skeleton_exact"] is False
    assert record["failure_bucket"] == "grammar_invalid"
    assert record["grammar_error"]
    assert record["scene_error"]
    assert record["error"]
    write_sdf.assert_not_called()


def test_scene_comparison_separates_skeleton_and_atom_types() -> None:
    expected = ring_scene()
    predicted = ring_scene()
    predicted["junctions"][0]["atom"]["element"] = "N"

    comparison = evaluate_module.compare_chemtraverse_scenes(expected, predicted)

    assert comparison["count_pair_exact"] is True
    assert comparison["skeleton_exact"] is True
    assert comparison["atom_typed_skeleton_exact"] is False
    assert comparison["bond_typed_skeleton_exact"] is True
    assert comparison["typed_graph_exact"] is False
