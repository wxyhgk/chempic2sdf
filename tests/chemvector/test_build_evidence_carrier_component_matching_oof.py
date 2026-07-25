import unittest

import torch

from chemvector.build_evidence_carrier_component_matching_oof import (
    ComponentMatchingOOFBuildError,
    SCHEMA_VERSION,
    STATUS_COMPLETE,
    component_sha256,
    tensor_sha256,
    validate_component_matching_sidecar,
)


class ComponentMatchingOOFBuildTest(unittest.TestCase):
    def test_validates_and_detects_component_mutation(self) -> None:
        scores = torch.tensor([2.0, 1.0], dtype=torch.float32)
        anchor_counts = torch.tensor([3, 3], dtype=torch.uint8)
        edge_counts = torch.tensor([2, 2], dtype=torch.uint8)
        components = [
            {
                "component_id": 0,
                "active_pair_indices": torch.tensor([0, 1], dtype=torch.int32),
                "matching_masks": torch.tensor(
                    [[False, False], [True, False], [False, True]],
                    dtype=torch.bool,
                ),
            }
        ]
        payload = {
            "schema_version": SCHEMA_VERSION,
            "status": STATUS_COMPLETE,
            "fold": 0,
            "counts": {
                "sample_count": 1,
                "pair_count": 2,
                "active_component_count": 1,
                "active_pair_count": 2,
                "matching_count": 3,
            },
            "records": [
                {
                    "sample_key": "a",
                    "baseline_scores": scores,
                    "component_anchor_counts": anchor_counts,
                    "component_edge_counts": edge_counts,
                    "components": components,
                    "tensor_sha256": tensor_sha256(
                        scores, anchor_counts, edge_counts
                    ),
                    "component_sha256": component_sha256(components),
                }
            ],
        }
        validate_component_matching_sidecar(payload, expected_fold=0)
        components[0]["matching_masks"][0, 0] = True
        with self.assertRaisesRegex(ComponentMatchingOOFBuildError, "SHA"):
            validate_component_matching_sidecar(payload, expected_fold=0)


if __name__ == "__main__":
    unittest.main()
