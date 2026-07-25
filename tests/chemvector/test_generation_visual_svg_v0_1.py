import unittest

from Data.sdf.visual_svg_sequence_v0_1 import (
    LineRecordV01,
    ParsedVisualSVGSequenceV01,
    QuadRecordV01,
    TextBoxRecordV01,
    TriangleRecordV01,
    serialize_parsed_visual_svg_v0_1,
)
from Data.sdf.visual_svg_tokenizer_v0_1 import VisualSVGTokenizerV01
from chemvector.generation_visual_svg_v0_1 import VisualSVGGrammarV01


class VisualSVGGrammarV01Test(unittest.TestCase):
    def test_accepts_a_canonical_target(self):
        tokenizer = VisualSVGTokenizerV01()
        sequence = serialize_parsed_visual_svg_v0_1(
            ParsedVisualSVGSequenceV01(
                (LineRecordV01(1, 2, 3, 4, 15),),
                (TriangleRecordV01(5, 5, 10, 5, 7, 12, 6),),
                (QuadRecordV01(20, 20, 30, 20, 30, 30, 20, 30, 6),),
                (TextBoxRecordV01(40, 40, 60, 55),),
            )
        )
        grammar = VisualSVGGrammarV01(tokenizer)
        for token_id in tokenizer.encode(sequence):
            self.assertIn(token_id, grammar.allowed_token_ids())
            grammar.consume(token_id)
        self.assertEqual(grammar.phase, "done")

    def test_text_box_cannot_precede_a_bond_primitive(self):
        tokenizer = VisualSVGTokenizerV01()
        grammar = VisualSVGGrammarV01(tokenizer)
        grammar.consume(tokenizer.bos_token_id)
        grammar.consume(tokenizer.token_to_id[tokenizer.DOCUMENT])
        self.assertNotIn(
            tokenizer.token_to_id[tokenizer.TEXT_BOX],
            grammar.allowed_token_ids(),
        )

    def test_text_box_origin_cannot_use_coordinate_maximum(self):
        tokenizer = VisualSVGTokenizerV01()
        grammar = VisualSVGGrammarV01(tokenizer)
        for token in (
            tokenizer.BOS,
            tokenizer.DOCUMENT,
            tokenizer.LINE,
            "c:1",
            "c:1",
            "c:2",
            "c:2",
            "w:10",
            tokenizer.TEXT_BOX,
        ):
            grammar.consume(tokenizer.token_to_id[token])
        self.assertNotIn(
            tokenizer.token_to_id["c:2047"],
            grammar.allowed_token_ids(),
        )

    def test_triangle_final_coordinate_rejects_collinearity(self):
        tokenizer = VisualSVGTokenizerV01()
        grammar = VisualSVGGrammarV01(tokenizer)
        for token in (
            tokenizer.BOS,
            tokenizer.DOCUMENT,
            tokenizer.TRIANGLE,
            "c:0",
            "c:0",
            "c:10",
            "c:10",
            "c:20",
        ):
            grammar.consume(tokenizer.token_to_id[token])
        self.assertNotIn(
            tokenizer.token_to_id["c:20"],
            grammar.allowed_token_ids(),
        )


if __name__ == "__main__":
    unittest.main()
