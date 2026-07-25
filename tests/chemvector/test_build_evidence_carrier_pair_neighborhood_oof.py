import unittest

import torch

from chemvector.build_evidence_carrier_pair_neighborhood_oof import (
    PairNeighborhoodOOFBuildError,
    SCHEMA_VERSION,
    STATUS_COMPLETE,
    tensor_sha256,
    validate_neighborhood_sidecar,
)
from chemvector.evidence_carrier_pair_neighborhood import PairNeighborhoodConfig


class PairNeighborhoodOOFBuildTest(unittest.TestCase):
    def test_validates_and_detects_mutation(self) -> None:
        features = torch.zeros(2, 12, 8, dtype=torch.float16)
        mask = torch.zeros(2, 12, dtype=torch.bool)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "status": STATUS_COMPLETE,
            "fold": 0,
            "config": PairNeighborhoodConfig().as_dict(),
            "counts": {"sample_count": 1, "pair_count": 2},
            "records": [
                {
                    "sample_key": "a",
                    "neighbor_features": features,
                    "neighbor_mask": mask,
                    "tensor_sha256": tensor_sha256(features, mask),
                }
            ],
        }
        validate_neighborhood_sidecar(payload, expected_fold=0)
        payload["records"][0]["neighbor_mask"][0, 0] = True
        with self.assertRaisesRegex(PairNeighborhoodOOFBuildError, "SHA"):
            validate_neighborhood_sidecar(payload, expected_fold=0)


if __name__ == "__main__":
    unittest.main()
