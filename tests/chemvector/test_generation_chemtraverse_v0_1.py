from __future__ import annotations

from types import SimpleNamespace

import torch
from rdkit import Chem

from Data.sdf.chemtraverse_tokenizer_v0_1 import ChemTraverseTokenizerV01
from Data.sdf.chemtraverse_v0_1 import serialize_chemtraverse_v0_1
from chemvector.generation_chemtraverse_v0_1 import (
    ChemTraverseGrammarV01,
    generate_chemtraverse_v0_1,
    write_sdf,
)


def _atom(
    x: int,
    y: int,
    element: str = "C",
    role: str = "IMPLICIT",
) -> dict:
    return {
        "coordinate": [x, y],
        "role": role,
        "atom": {
            "element": element,
            "formal_charge": 0,
            "isotope": 0,
            "aromatic": False,
            "explicit_h_count": 0,
            "no_implicit": False,
            "radical_electrons": 0,
            "canonical_chiral_tag": "CHI_UNSPECIFIED",
        },
    }


def _bond(
    begin: list[int],
    end: list[int],
    bond_type: str = "SINGLE",
    *,
    stereo: str = "STEREONONE",
    stereo_references: list[list[int]] | None = None,
) -> dict:
    result = {
        "endpoints": [begin, end],
        "display_type": bond_type,
        "chemical_type": bond_type,
        "direction": "NONE",
        "stereo": stereo,
        "bbox": None,
    }
    if stereo_references is not None:
        result["stereo_references"] = stereo_references
    return result


def _scene(
    atoms: list[dict],
    bonds: list[dict] | None = None,
    texts: list[dict] | None = None,
    molecule_id: str = "test-molecule",
) -> dict:
    return {
        "schema_version": "chemscene-0.1",
        "molecule_id": molecule_id,
        "units": "q1023",
        "canvas": {"width": 768, "height": 768},
        "junctions": atoms,
        "text_objects": texts or [],
        "bond_objects": bonds or [],
    }


def _consume_target(grammar: ChemTraverseGrammarV01, input_ids: list[int]) -> None:
    for token_id in input_ids:
        assert token_id in grammar.allowed_token_ids()
        grammar.consume(token_id)


class _ScriptedModel:
    def __init__(self, target_ids: list[int], vocab_size: int) -> None:
        self.target_ids = target_ids
        self.vocab_size = vocab_size
        self.config = SimpleNamespace(max_target_length=len(target_ids) + 4)

    def eval(self) -> "_ScriptedModel":
        return self

    @staticmethod
    def encode_images(pixels: torch.Tensor) -> torch.Tensor:
        return torch.zeros((1, 1, 4), device=pixels.device)

    def decode_step(
        self,
        token: torch.Tensor,
        position: int,
        memory: torch.Tensor,
        caches: list[torch.Tensor] | None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        logits = torch.full(
            (token.size(0), 1, self.vocab_size),
            -1_000.0,
            device=token.device,
        )
        logits[:, 0, self.target_ids[position + 1]] = 1_000.0
        return logits, []


def test_canonical_cycle_components_text_and_stereo_are_accepted() -> None:
    points = [[100, 100], [300, 100], [400, 300], [200, 400]]
    cycle = _scene(
        [_atom(*point) for point in points] + [_atom(900, 900)],
        [
            _bond(points[0], points[1]),
            _bond(points[1], points[2]),
            _bond(points[2], points[3]),
            _bond(points[3], points[0]),
        ],
    )
    stereo_points = [[100, 500], [300, 500], [600, 500], [800, 500]]
    stereo = _scene(
        [_atom(*point) for point in stereo_points],
        [
            _bond(stereo_points[0], stereo_points[1]),
            _bond(
                stereo_points[1],
                stereo_points[2],
                "DOUBLE",
                stereo="STEREOE",
                stereo_references=[stereo_points[0], stereo_points[3]],
            ),
            _bond(stereo_points[2], stereo_points[3]),
        ],
    )
    text_scene = _scene(
        [_atom(500, 500, "*", "TEXT")],
        texts=[
            {
                "anchor": [500, 500],
                "center": [510, 490],
                "bbox": [450, 450, 570, 530],
                "text": "Ph-分子",
            }
        ],
    )
    tokenizer = ChemTraverseTokenizerV01()
    for scene in (cycle, stereo, text_scene):
        sequence = serialize_chemtraverse_v0_1(scene)
        grammar = ChemTraverseGrammarV01(tokenizer)
        _consume_target(grammar, tokenizer.encode(sequence))
        assert grammar.phase == "done"


def test_link_references_are_backward_and_duplicate_bonds_are_masked() -> None:
    tokenizer = ChemTraverseTokenizerV01()
    atoms = [_atom(100, 100), _atom(300, 100), _atom(200, 300)]
    sequence = serialize_chemtraverse_v0_1(
        _scene(
            atoms,
            [
                _bond([100, 100], [300, 100]),
                _bond([300, 100], [200, 300]),
                _bond([200, 300], [100, 100]),
            ],
        )
    )
    input_ids = tokenizer.encode(sequence)
    link_id = tokenizer.token_to_id[tokenizer.LINK]
    link_index = input_ids.index(link_id)
    grammar = ChemTraverseGrammarV01(tokenizer)
    _consume_target(grammar, input_ids[: link_index + 1])
    references = {
        int(tokenizer.id_to_token[token_id][2:])
        for token_id in grammar.allowed_token_ids()
    }
    assert references
    assert max(references) < len(grammar.atom_roles) - 1
    assert grammar.current_atom not in references

    two_atom_sequence = serialize_chemtraverse_v0_1(
        _scene(
            [_atom(100, 100), _atom(300, 100)],
            [_bond([100, 100], [300, 100])],
        )
    )
    grammar = ChemTraverseGrammarV01(tokenizer)
    for token_id in tokenizer.encode(two_atom_sequence):
        grammar.consume(token_id)
        if len(grammar.atom_roles) == 2 and grammar.phase == "record":
            break
    assert link_id not in grammar.allowed_token_ids()


def test_text_utf8_bbox_and_required_role_are_constrained() -> None:
    tokenizer = ChemTraverseTokenizerV01()
    scene = _scene(
        [_atom(500, 500, "*", "TEXT")],
        texts=[
            {
                "anchor": [500, 500],
                "center": [500, 500],
                "bbox": [400, 410, 600, 610],
                "text": "é",
            }
        ],
        molecule_id="m",
    )
    input_ids = tokenizer.encode(serialize_chemtraverse_v0_1(scene))

    grammar = ChemTraverseGrammarV01(tokenizer)
    bbox_index = input_ids.index(tokenizer.token_to_id[tokenizer.BBOX])
    _consume_target(grammar, input_ids[: bbox_index + 3])
    allowed_x_max = set(grammar.allowed_token_ids())
    assert tokenizer.token_to_id["n:399"] not in allowed_x_max
    assert tokenizer.token_to_id["n:400"] in allowed_x_max

    grammar = ChemTraverseGrammarV01(tokenizer)
    first_utf8_byte = input_ids.index(tokenizer.token_to_id["byte:c3"])
    _consume_target(grammar, input_ids[: first_utf8_byte + 1])
    allowed = set(grammar.allowed_token_ids())
    assert tokenizer.token_to_id[tokenizer.UTF8_END] not in allowed
    assert tokenizer.token_to_id["byte:20"] not in allowed
    assert tokenizer.token_to_id["byte:a9"] in allowed
    grammar.consume(tokenizer.token_to_id["byte:a9"])
    assert tokenizer.token_to_id[tokenizer.UTF8_END] in grammar.allowed_token_ids()

    grammar = ChemTraverseGrammarV01(tokenizer)
    text_index = input_ids.index(tokenizer.token_to_id[tokenizer.TEXT])
    _consume_target(grammar, input_ids[:text_index])
    allowed_records = set(grammar.allowed_token_ids())
    assert tokenizer.token_to_id[tokenizer.TEXT] in allowed_records
    assert tokenizer.token_to_id[tokenizer.MOL_END] not in allowed_records


def test_generation_distinguishes_parseable_from_chemically_valid() -> None:
    tokenizer = ChemTraverseTokenizerV01()
    valid_sequence = serialize_chemtraverse_v0_1(_scene([_atom(500, 500)]))
    valid_ids = tokenizer.encode(valid_sequence)
    valid = generate_chemtraverse_v0_1(
        _ScriptedModel(valid_ids, tokenizer.vocab_size),
        torch.zeros((1, 3, 32, 32)),
        tokenizer,
    )
    assert valid.completed
    assert valid.parseable
    assert valid.chemically_valid
    assert valid.sequence == valid_sequence

    invalid_sequence = serialize_chemtraverse_v0_1(
        _scene(
            [_atom(100, 100, "F"), _atom(300, 100, "F")],
            [_bond([100, 100], [300, 100], "DOUBLE")],
        )
    )
    invalid_ids = tokenizer.encode(invalid_sequence)
    invalid = generate_chemtraverse_v0_1(
        _ScriptedModel(invalid_ids, tokenizer.vocab_size),
        torch.zeros((1, 3, 32, 32)),
        tokenizer,
    )
    assert invalid.completed
    assert invalid.parseable
    assert not invalid.chemically_valid
    assert invalid.sequence == invalid_sequence
    assert invalid.error


def test_count_constraints_accept_exact_single_component_sequences() -> None:
    tokenizer = ChemTraverseTokenizerV01()
    triangle = [[100, 100], [300, 100], [200, 300]]
    scenes = (
        _scene([_atom(500, 500)]),
        _scene(
            [_atom(100, 100), _atom(300, 100), _atom(500, 100)],
            [
                _bond([100, 100], [300, 100]),
                _bond([300, 100], [500, 100]),
            ],
        ),
        _scene(
            [_atom(*point) for point in triangle],
            [
                _bond(triangle[0], triangle[1]),
                _bond(triangle[1], triangle[2]),
                _bond(triangle[2], triangle[0]),
            ],
        ),
    )
    for scene in scenes:
        grammar = ChemTraverseGrammarV01(
            tokenizer,
            expected_atom_count=len(scene["junctions"]),
            expected_bond_count=len(scene["bond_objects"]),
        )
        _consume_target(
            grammar,
            tokenizer.encode(serialize_chemtraverse_v0_1(scene)),
        )
        assert grammar.phase == "done"


def test_count_constraints_block_early_end_and_count_overflow() -> None:
    tokenizer = ChemTraverseTokenizerV01()
    scene = _scene(
        [_atom(100, 100), _atom(300, 100), _atom(500, 100)],
        [
            _bond([100, 100], [300, 100]),
            _bond([300, 100], [500, 100]),
        ],
    )
    input_ids = tokenizer.encode(serialize_chemtraverse_v0_1(scene))
    grammar = ChemTraverseGrammarV01(
        tokenizer,
        expected_atom_count=3,
        expected_bond_count=2,
    )
    position = 0
    while not (len(grammar.atom_roles) == 1 and grammar.phase == "record"):
        grammar.consume(input_ids[position])
        position += 1

    allowed = set(grammar.allowed_token_ids())
    assert tokenizer.token_to_id[tokenizer.DOWN] in allowed
    assert tokenizer.token_to_id[tokenizer.COMPONENT_END] not in allowed
    assert tokenizer.token_to_id[tokenizer.MOL_END] not in allowed

    while not (len(grammar.atom_roles) == 3 and grammar.phase == "record"):
        grammar.consume(input_ids[position])
        position += 1

    assert len(grammar.bonds) == 2
    allowed = set(grammar.allowed_token_ids())
    assert tokenizer.token_to_id[tokenizer.DOWN] not in allowed
    assert tokenizer.token_to_id[tokenizer.LINK] not in allowed
    assert tokenizer.token_to_id[tokenizer.UP] in allowed

    while grammar.component_open:
        if tokenizer.id_to_token[input_ids[position]] == tokenizer.COMPONENT_END:
            assert len(grammar.atom_roles) == 3
            assert len(grammar.bonds) == 2
            assert input_ids[position] in grammar.allowed_token_ids()
        grammar.consume(input_ids[position])
        position += 1

    allowed = set(grammar.allowed_token_ids())
    assert tokenizer.token_to_id[tokenizer.COMPONENT] not in allowed
    assert tokenizer.token_to_id[tokenizer.MOL_END] in allowed


def test_count_constrained_generation_supports_greedy_and_beam() -> None:
    tokenizer = ChemTraverseTokenizerV01()
    scene = _scene(
        [_atom(100, 100), _atom(300, 100), _atom(500, 100)],
        [
            _bond([100, 100], [300, 100]),
            _bond([300, 100], [500, 100]),
        ],
    )
    sequence = serialize_chemtraverse_v0_1(scene)
    input_ids = tokenizer.encode(sequence)
    for beam_width in (1, 2):
        result = generate_chemtraverse_v0_1(
            _ScriptedModel(input_ids, tokenizer.vocab_size),
            torch.zeros((1, 3, 32, 32)),
            tokenizer,
            beam_width=beam_width,
            expected_atom_count=3,
            expected_bond_count=2,
        )
        assert result.completed
        assert result.parseable
        assert result.sequence == sequence


def test_count_constraint_argument_validation() -> None:
    tokenizer = ChemTraverseTokenizerV01()
    invalid = (
        ({"expected_atom_count": 2}, "provided together"),
        (
            {"expected_atom_count": 0, "expected_bond_count": 0},
            "must be positive",
        ),
        (
            {"expected_atom_count": 3, "expected_bond_count": 1},
            "at least atom_count - 1",
        ),
        (
            {"expected_atom_count": 2, "expected_bond_count": 2},
            "maximum for a simple graph",
        ),
    )
    for arguments, expected_message in invalid:
        try:
            ChemTraverseGrammarV01(tokenizer, **arguments)
        except ValueError as error:
            assert expected_message in str(error)
        else:
            raise AssertionError("Invalid count constraints were accepted")

    try:
        generate_chemtraverse_v0_1(
            None,
            torch.zeros((1, 3, 32, 32)),
            tokenizer,
            expected_atom_count=2,
        )
    except ValueError as error:
        assert "provided together" in str(error)
    else:
        raise AssertionError("Generation accepted an incomplete count constraint")


def test_write_sdf_produces_a_readable_record(tmp_path) -> None:
    sequence = serialize_chemtraverse_v0_1(
        _scene(
            [_atom(100, 100), _atom(300, 100, "O")],
            [_bond([100, 100], [300, 100])],
        )
    )
    path = tmp_path / "nested" / "molecule.sdf"
    write_sdf(sequence, path)
    mol = next((item for item in Chem.SDMolSupplier(str(path)) if item is not None), None)
    assert mol is not None
    assert mol.GetNumAtoms() == 2


def test_generation_argument_validation() -> None:
    tokenizer = ChemTraverseTokenizerV01()
    pixels = torch.zeros((2, 3, 32, 32))
    try:
        generate_chemtraverse_v0_1(None, pixels, tokenizer)
    except ValueError as error:
        assert "batch size 1" in str(error)
    else:
        raise AssertionError("batch size validation did not run")

    try:
        generate_chemtraverse_v0_1(
            None,
            torch.zeros((1, 3, 32, 32)),
            tokenizer,
            beam_width=0,
        )
    except ValueError as error:
        assert "beam_width" in str(error)
    else:
        raise AssertionError("beam width validation did not run")
