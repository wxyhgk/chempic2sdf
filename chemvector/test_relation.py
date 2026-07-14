"""Tests for factorized edge and bond prediction."""

from __future__ import annotations

import unittest

import torch

from Data.sdf.chemsvg_sequence_v4 import VertexRecordV4
from chemvector.model import ChemVectorConfig, ChemVectorModel
from chemvector.parallel_v6 import score_vertex_pairs_parallel
from chemvector.relation import (
    factorized_relation_loss,
    factorized_relation_probabilities,
)


def small_config(*, relation_enabled: bool = True) -> ChemVectorConfig:
    return ChemVectorConfig(
        vocab_size=64,
        d_model=64,
        decoder_layers=1,
        decoder_heads=4,
        decoder_ffn_dim=128,
        max_target_length=32,
        vision_dims=(16, 32, 48, 64),
        vision_depths=(1, 1, 1, 1),
        stroke_enabled=True,
        stroke_feature_dim=16,
        stroke_samples=12,
        stroke_band=3,
        relation_enabled=relation_enabled,
    )


class FactorizedRelationTests(unittest.TestCase):
    def test_relation_head_output_shapes(self) -> None:
        model = ChemVectorModel(small_config())
        output = model.relation_head(torch.randn(7, 16))

        self.assertEqual(output["edge_logits"].shape, (7,))
        self.assertEqual(output["bond_logits"].shape, (7, 3))

    def test_bond_loss_ignores_negative_pairs(self) -> None:
        edge_logits = torch.tensor([0.2, -0.4, 0.7])
        bond_logits = torch.tensor(
            [
                [100.0, -100.0, -100.0],
                [0.0, 3.0, 0.0],
                [0.0, 0.0, 4.0],
            ]
        )
        labels = torch.tensor([0, 2, 3])
        first = factorized_relation_loss(edge_logits, bond_logits, labels)
        changed = bond_logits.clone()
        changed[0] = torch.tensor([-500.0, 500.0, 500.0])
        second = factorized_relation_loss(edge_logits, changed, labels)

        self.assertTrue(torch.allclose(first["bond_loss"], second["bond_loss"]))

    def test_bond_loss_excludes_graph_level_aromatic_labels(self) -> None:
        edge_logits = torch.tensor([1.0, 1.0])
        bond_logits = torch.tensor([[0.0, 4.0, 0.0], [100.0, -100.0, -100.0]])
        labels = torch.tensor([2, 4])
        first = factorized_relation_loss(edge_logits, bond_logits, labels)
        changed = bond_logits.clone()
        changed[1] = torch.tensor([-500.0, 500.0, 500.0])
        second = factorized_relation_loss(edge_logits, changed, labels)

        self.assertTrue(torch.allclose(first["bond_loss"], second["bond_loss"]))

    def test_composed_probabilities_sum_to_one(self) -> None:
        probabilities = factorized_relation_probabilities(
            torch.tensor([-3.0, 0.0, 4.0]),
            torch.randn(3, 3),
        )

        self.assertEqual(probabilities.shape, (3, 5))
        self.assertTrue(torch.allclose(probabilities.sum(dim=-1), torch.ones(3)))

    def test_relation_disabled_keeps_legacy_five_class_scoring(self) -> None:
        model = ChemVectorModel(small_config(relation_enabled=False)).eval()
        pixels = torch.randn(1, 3, 64, 64)
        features = model.vision_encoder.forward_stroke_resolution(pixels)
        vertices = [
            VertexRecordV4(
                x=200,
                y=500,
                element="C",
                formal_charge=0,
                isotope=0,
                aromatic=False,
                explicit_h_count=0,
                no_implicit=False,
                radical_electrons=0,
                canonical_chiral_tag="CHI_UNSPECIFIED",
            ),
            VertexRecordV4(
                x=800,
                y=500,
                element="C",
                formal_charge=0,
                isotope=0,
                aromatic=False,
                explicit_h_count=0,
                no_implicit=False,
                radical_electrons=0,
                canonical_chiral_tag="CHI_UNSPECIFIED",
            ),
        ]
        pairs, probabilities = score_vertex_pairs_parallel(
            model,
            features,
            vertices,
            pixels.shape[-2:],
        )
        anchors = torch.tensor([[[200 / 1023, 500 / 1023], [800 / 1023, 500 / 1023]]])
        pair_indices = torch.tensor([[[0, 1]]])
        pair_mask = torch.tensor([[True]])
        hidden = model.stroke_head.encode_pairs(
            features,
            anchors,
            pair_indices,
            pair_mask,
            pixels.shape[-2:],
        )
        expected = model.stroke_head.classifier[3](hidden).softmax(dim=-1)

        self.assertEqual(pairs, [(0, 1)])
        self.assertTrue(torch.allclose(probabilities, expected, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
