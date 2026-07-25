from __future__ import annotations

import itertools
import unittest

import numpy as np

from chemvector.evidence_carrier_decoded_line_clustering import (
    DecodedLineClusteringConfig,
    cluster_decoded_lines,
    decoded_line_pair_geometry,
)


class EvidenceCarrierDecodedLineClusteringTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = DecodedLineClusteringConfig()

    def test_pair_geometry_uses_all_four_thresholds(self) -> None:
        first = np.asarray(((10.0, 20.0), (110.0, 20.0)))
        close = np.asarray(((10.5, 21.0), (110.5, 21.0)))
        metrics = decoded_line_pair_geometry(
            first, close, width=768, height=768, config=self.config
        )
        self.assertTrue(metrics.compatible)
        self.assertLessEqual(metrics.angle_degrees, 2.0)
        self.assertLessEqual(metrics.normal_distance_pixels, 2.0)
        self.assertGreaterEqual(metrics.interval_iou, 0.85)
        self.assertLessEqual(metrics.endpoint_error_q, 8.0)

        shifted = np.asarray(((10.0, 23.0), (110.0, 23.0)))
        self.assertFalse(
            decoded_line_pair_geometry(
                first, shifted, width=768, height=768, config=self.config
            ).compatible
        )

    def test_complete_link_does_not_chain_incompatible_endpoints(self) -> None:
        lines = np.asarray(
            [
                ((10.0, 20.0), (110.0, 20.0)),
                ((11.0, 20.0), (111.0, 20.0)),
                ((12.0, 20.0), (112.0, 20.0)),
            ]
        )
        strict = DecodedLineClusteringConfig(maximum_endpoint_error_q=1.5)
        clusters = cluster_decoded_lines(
            lines,
            lines,
            ("a", "b", "c"),
            width=2048,
            height=2048,
            config=strict,
        )
        self.assertEqual(len(clusters), 2)
        self.assertNotEqual(
            {cluster.member_source_ids for cluster in clusters},
            {("a", "b", "c")},
        )

    def test_representative_is_an_observed_medoid_not_an_average(self) -> None:
        lines = np.asarray(
            [
                ((10.0, 20.0), (110.0, 20.0)),
                ((10.5, 20.0), (110.5, 20.0)),
                ((11.0, 20.0), (111.0, 20.0)),
            ]
        )
        clusters = cluster_decoded_lines(
            lines,
            lines,
            ("left", "middle", "right"),
            width=768,
            height=768,
            config=self.config,
        )
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0].representative_source_id, "middle")
        np.testing.assert_array_equal(
            np.asarray(clusters[0].representative_segment_pixels), lines[1]
        )

    def test_parallel_lines_remain_distinct(self) -> None:
        lines = np.asarray(
            [
                ((20.0, 40.0), (120.0, 40.0)),
                ((20.0, 44.0), (120.0, 44.0)),
                ((20.2, 40.3), (120.2, 40.3)),
                ((20.2, 44.3), (120.2, 44.3)),
            ]
        )
        clusters = cluster_decoded_lines(
            lines,
            lines,
            ("top-a", "bottom-a", "top-b", "bottom-b"),
            width=768,
            height=768,
            config=self.config,
        )
        self.assertEqual(
            {frozenset(cluster.member_source_ids) for cluster in clusters},
            {
                frozenset(("top-a", "top-b")),
                frozenset(("bottom-a", "bottom-b")),
            },
        )

    def test_output_is_invariant_to_input_permutation_and_endpoint_order(self) -> None:
        lines = np.asarray(
            [
                ((10.0, 20.0), (110.0, 20.0)),
                ((10.4, 20.2), (110.4, 20.2)),
                ((30.0, 70.0), (130.0, 70.0)),
            ]
        )
        anchors = np.asarray(
            [
                ((8.0, 19.0), (80.0, 19.0)),
                ((20.0, 21.0), (95.0, 21.0)),
                ((35.0, 69.0), (100.0, 69.0)),
            ]
        )
        source_ids = np.asarray(("a", "b", "c"))

        def signature(order: tuple[int, ...], reverse: bool) -> tuple:
            current_lines = lines[list(order)].copy()
            current_anchors = anchors[list(order)].copy()
            if reverse:
                current_lines = current_lines[:, ::-1]
                current_anchors = current_anchors[:, ::-1]
            clusters = cluster_decoded_lines(
                current_lines,
                current_anchors,
                source_ids[list(order)].tolist(),
                width=768,
                height=768,
                config=self.config,
            )
            return tuple(
                (cluster.member_source_ids, cluster.representative_source_id)
                for cluster in clusters
            )

        expected = signature((0, 1, 2), False)
        for order in itertools.permutations(range(3)):
            self.assertEqual(signature(order, False), expected)
            self.assertEqual(signature(order, True), expected)

    def test_rejects_duplicate_source_ids(self) -> None:
        lines = np.asarray(
            [
                ((0.0, 0.0), (10.0, 0.0)),
                ((0.0, 1.0), (10.0, 1.0)),
            ]
        )
        with self.assertRaisesRegex(ValueError, "unique"):
            cluster_decoded_lines(
                lines,
                lines,
                ("same", "same"),
                width=20,
                height=20,
            )


if __name__ == "__main__":
    unittest.main()
