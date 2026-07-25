import unittest

import torch

from chemvector.build_evidence_carrier_pair_local_oof import (
    PairLocalOOFBuildError,
    SCHEMA_VERSION,
    STATUS_COMPLETE,
    tensor_sha256,
    validate_pair_local_sidecar,
)
from chemvector.evidence_carrier_pair_local_raster import PairLocalRasterConfig


class PairLocalOOFBuildTest(unittest.TestCase):
    def _payload(self):
        crops = torch.arange(2 * 2 * 24 * 48, dtype=torch.int64).remainder(256).to(torch.uint8)
        crops = crops.reshape(2, 2, 24, 48)
        return {
            "schema_version": SCHEMA_VERSION,
            "status": STATUS_COMPLETE,
            "fold": 0,
            "config": PairLocalRasterConfig().as_dict(),
            "counts": {"sample_count": 1, "pair_count": 2},
            "records": [
                {
                    "sample_key": "000000001",
                    "pair_local_rasters": crops,
                    "crop_sha256": tensor_sha256(crops),
                }
            ],
        }

    def test_validates_sidecar_and_detects_crop_mutation(self) -> None:
        payload = self._payload()
        validate_pair_local_sidecar(payload, expected_fold=0)
        payload["records"][0]["pair_local_rasters"][0, 0, 0, 0] += 1
        with self.assertRaisesRegex(PairLocalOOFBuildError, "SHA"):
            validate_pair_local_sidecar(payload, expected_fold=0)

    def test_rejects_wrong_counts_and_fold(self) -> None:
        payload = self._payload()
        payload["counts"]["pair_count"] = 3
        with self.assertRaisesRegex(PairLocalOOFBuildError, "counts"):
            validate_pair_local_sidecar(payload)
        with self.assertRaisesRegex(PairLocalOOFBuildError, "fold"):
            validate_pair_local_sidecar(self._payload(), expected_fold=1)


if __name__ == "__main__":
    unittest.main()
