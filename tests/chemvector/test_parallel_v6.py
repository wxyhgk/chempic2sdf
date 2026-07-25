"""Tests for decoder-free parallel ChemSVG inference."""

from __future__ import annotations

import unittest

import torch

from Data.sdf.chemsvg_sequence_v4 import VertexRecordV4, parse_chemsvg_v4
from chemvector.model import ChemVectorConfig, ChemVectorModel
from chemvector.parallel_v6 import (
    add_proposal_graph_edge_prior,
    build_parallel_sequence,
    deployment_parameter_count,
    strip_autoregressive_modules,
)


class ParallelV6Tests(unittest.TestCase):
    def test_graph_prior_only_raises_connectivity_score(self) -> None:
        pair = ((10, 20), (30, 40))
        predictions = {pair: (0.2, "DOUBLE")}

        fused = add_proposal_graph_edge_prior(
            predictions,
            {pair: 0.9},
            threshold=0.85,
        )

        self.assertEqual(fused[pair], (0.9, "DOUBLE"))
        self.assertEqual(predictions[pair], (0.2, "DOUBLE"))

    def test_graph_prior_ignores_subthreshold_evidence(self) -> None:
        pair = ((10, 20), (30, 40))
        predictions = {pair: (0.2, "SINGLE")}

        fused = add_proposal_graph_edge_prior(
            predictions,
            {pair: 0.8},
            threshold=0.85,
        )

        self.assertEqual(fused, predictions)

    @staticmethod
    def _vertex(x: int) -> VertexRecordV4:
        return VertexRecordV4(
            x=x,
            y=100,
            element="C",
            formal_charge=0,
            isotope=0,
            aromatic=False,
            explicit_h_count=0,
            no_implicit=False,
            radical_electrons=0,
            canonical_chiral_tag="CHI_UNSPECIFIED",
        )

    def test_pair_probabilities_build_strokes_in_parallel(self) -> None:
        vertices = (self._vertex(100), self._vertex(900))
        probabilities = torch.tensor([[0.1, 0.2, 0.6, 0.05, 0.05]])
        sequence, strokes = build_parallel_sequence(
            vertices,
            [(0, 1)],
            probabilities,
        )
        self.assertEqual(len(strokes), 1)
        self.assertEqual(strokes[0].bond_type, "DOUBLE")
        self.assertEqual(len(parse_chemsvg_v4(sequence).strokes), 1)

    def test_deployment_strip_removes_most_decoder_parameters(self) -> None:
        config = ChemVectorConfig(
            vocab_size=64,
            d_model=64,
            decoder_layers=2,
            decoder_heads=4,
            decoder_ffn_dim=128,
            max_target_length=32,
            vision_dims=(16, 32, 48, 64),
            vision_depths=(1, 1, 1, 1),
            ocr_enabled=True,
            stroke_enabled=True,
            vertex_enabled=True,
            ocr_d_model=32,
            ocr_heads=4,
            ocr_ffn_dim=64,
            stroke_feature_dim=16,
            vertex_feature_dim=16,
        )
        model = ChemVectorModel(config)
        before = deployment_parameter_count(model)
        strip_autoregressive_modules(model)
        after = deployment_parameter_count(model)
        self.assertLess(after, before)
        self.assertIsNone(model.decoder)
        self.assertIsNotNone(model.stroke_head)


if __name__ == "__main__":
    unittest.main()
