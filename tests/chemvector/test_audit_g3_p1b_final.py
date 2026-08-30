from pathlib import Path
import unittest

import pytest

from chemvector.evaluation.audits.atoms.audit_g3_p1b_final import (
    EXPECTED_RANKER_SHA256,
    EXPECTED_STEPS,
    EXPECTED_TRAINER_SHA256,
    _checkpoint_audit,
    _epoch_checks,
    _json,
    _source_hashes,
)
from chemvector.training.atoms import train_g3_p1b_frozen_presence_ranker as p1b


class G3P1BFinalAuditTests(unittest.TestCase):
    @pytest.mark.xfail(
        strict=True,
        reason=(
            "superseded frozen hash chain: pinned ranker source 423b7414f8eb "
            "(recorded in checkpoint payloads and prereg doc) was revised to "
            "1a31c090bff8 at ec2b846 and the training version is "
            "unrecoverable; P1-B is terminal NO-GO. strict canary: clear "
            "after a formal re-pin or source restoration."
        ),
    )
    def test_fixed_checkpoint_chain_is_residual_only(self) -> None:
        root = Path(__file__).resolve().parents[2]
        source_hashes = _source_hashes(root)
        checkpoints, checks = _checkpoint_audit(p1b.DEFAULT_OUTPUT, source_hashes)

        self.assertEqual(source_hashes["trainer"], EXPECTED_TRAINER_SHA256)
        self.assertEqual(source_hashes["ranker"], EXPECTED_RANKER_SHA256)
        self.assertEqual([item["step"] for item in checkpoints], list(EXPECTED_STEPS))
        self.assertTrue(all(item["passed"] for item in checkpoints))
        self.assertTrue(all(item["passed"] for item in checks))
        self.assertTrue(
            all(item["optimizer"]["residual_only"] for item in checkpoints)
        )

    def test_five_epoch_audits_have_the_fixed_sequence(self) -> None:
        epochs = _json(p1b.DEFAULT_OUTPUT / "epoch_audit.json")

        self.assertEqual(len(epochs), 5)
        self.assertTrue(all(_epoch_checks(epochs)))


if __name__ == "__main__":
    unittest.main()
