from pathlib import Path
import unittest

from chemvector.evaluation.audits.atoms.audit_g3_p1d_final import (
    EXPECTED_LOSS_SHA256,
    EXPECTED_P1C_HELPER_SHA256,
    EXPECTED_RANKER_SHA256,
    EXPECTED_TRAINER_SHA256,
    _checkpoint_audit,
    _epoch_audit,
    _source_hashes,
)
from chemvector.training.atoms import train_g3_p1d_boundary_band_set_ranker as p1d


class G3P1DFinalAuditTests(unittest.TestCase):
    def test_source_and_checkpoint_chain_are_fixed(self) -> None:
        root = Path(__file__).resolve().parents[2]
        hashes = _source_hashes(root)
        checkpoints, checks = _checkpoint_audit(p1d.DEFAULT_OUTPUT, hashes)

        self.assertEqual(hashes["trainer"], EXPECTED_TRAINER_SHA256)
        self.assertEqual(hashes["ranker"], EXPECTED_RANKER_SHA256)
        self.assertEqual(hashes["loss"], EXPECTED_LOSS_SHA256)
        self.assertEqual(hashes["p1c_helper"], EXPECTED_P1C_HELPER_SHA256)
        self.assertTrue(all(item["passed"] for item in checkpoints))
        self.assertTrue(all(checks.values()))

    def test_epoch_audit_requires_all_five_fixed_epochs(self) -> None:
        epochs = p1d.json.loads(
            (p1d.DEFAULT_OUTPUT / "epoch_audit.json").read_text(encoding="utf-8")
        )

        self.assertTrue(_epoch_audit(epochs)["passed"])
        self.assertFalse(_epoch_audit(epochs[:-1])["passed"])


if __name__ == "__main__":
    unittest.main()
