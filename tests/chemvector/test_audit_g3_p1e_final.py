from pathlib import Path
import unittest

from chemvector.evaluation.audits.atoms.audit_g3_p1e_final import (
    EXPECTED_TRAINER_SHA256,
    _checkpoint_audit,
    _epoch_audit,
    _source_hashes,
)
from chemvector.evaluation.audits.atoms.audit_g3_p1c_final import _json
from chemvector.training.atoms import (
    train_g3_p1e_aligned_stroke_feature_sufficiency as p1e,
)


class G3P1EFinalAuditTests(unittest.TestCase):
    def test_source_checkpoint_and_epoch_chain_are_fixed(self) -> None:
        root = Path(__file__).resolve().parents[2]
        sources = _source_hashes(root)
        checkpoints, checks = _checkpoint_audit(p1e.DEFAULT_OUTPUT, sources)
        epochs = _json(p1e.DEFAULT_OUTPUT / "epoch_audit.json")

        self.assertEqual(sources["trainer"], EXPECTED_TRAINER_SHA256)
        self.assertEqual(
            {key: sources[key] for key in p1e.EXPECTED_DEPENDENCY_SHA256},
            p1e.EXPECTED_DEPENDENCY_SHA256,
        )
        self.assertTrue(all(checks.values()))
        self.assertTrue(
            all(value["passed"] for rows in checkpoints.values() for value in rows)
        )
        self.assertTrue(_epoch_audit(epochs)["passed"])

    def test_epoch_audit_rejects_missing_epoch(self) -> None:
        epochs = _json(p1e.DEFAULT_OUTPUT / "epoch_audit.json")
        broken = {key: list(value) for key, value in epochs.items()}
        broken["sham"] = broken["sham"][:-1]

        self.assertFalse(_epoch_audit(broken)["passed"])


if __name__ == "__main__":
    unittest.main()
