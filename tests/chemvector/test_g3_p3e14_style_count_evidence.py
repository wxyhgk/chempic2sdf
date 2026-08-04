"""Tests for resumable P3-E14 Count evidence sharding."""

import unittest
import inspect

from chemvector.training.atoms.build_g3_p3e14_style_count_evidence import (
    MANIFEST_SHA256,
    group_manifest_by_shard,
    sample_key_set_sha256,
    sample_keys_sha256,
)
from chemvector.training.atoms.train_g3_p1x_multiscale_peak_selector_capacity import (
    _collect_scale,
)
from chemvector.training.graph.scenes.g3_structural_data import G3SampleManifestRecord


class G3P3E14StyleCountEvidenceTests(unittest.TestCase):
    def test_grouping_preserves_first_shard_and_record_order(self) -> None:
        records = [
            G3SampleManifestRecord("a", "m1", "i1", "/x/0.tar", "original"),
            G3SampleManifestRecord("b", "m2", "i2", "/x/0.tar", "stress"),
            G3SampleManifestRecord("c", "m3", "i3", "/x/1.tar", "rdkit"),
        ]
        groups = group_manifest_by_shard(records)
        self.assertEqual([value[0] for value in groups], ["/x/0.tar", "/x/1.tar"])
        self.assertEqual([value.sample_key for value in groups[0][1]], ["a", "b"])
        self.assertEqual(sample_keys_sha256(groups[0][1]), sample_keys_sha256([
            {"sample_key": "a"}, {"sample_key": "b"}
        ]))
        self.assertEqual(
            sample_key_set_sha256(groups[0][1]),
            sample_key_set_sha256([{"sample_key": "b"}, {"sample_key": "a"}]),
        )

    def test_manifest_digest_is_frozen(self) -> None:
        self.assertEqual(len(MANIFEST_SHA256), 64)

    def test_collection_batch_size_remains_backward_compatible(self) -> None:
        self.assertEqual(
            inspect.signature(_collect_scale).parameters["batch_size"].default,
            4,
        )


if __name__ == "__main__":
    unittest.main()
