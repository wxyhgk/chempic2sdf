from pathlib import Path
import unittest

from chemvector.evaluation.audits.atoms.audit_g3_p1c_final import (
    EXPECTED_RANKER_SHA256,
    EXPECTED_TRAINER_SHA256,
    _checkpoint_audit,
    _epoch_audit,
    _json,
    _source_hashes,
    _target_free_source_contract,
    _target_isolation_gate,
)
from chemvector.training.atoms import train_g3_p1c_top128_set_ranker as p1c


class G3P1CFinalAuditTests(unittest.TestCase):
    def test_checkpoint_chain_and_epoch_identity_are_fixed(self) -> None:
        root = Path(__file__).resolve().parents[2]
        hashes = _source_hashes(root)
        checkpoints, checks = _checkpoint_audit(p1c.DEFAULT_OUTPUT, hashes)
        epochs = _json(p1c.DEFAULT_OUTPUT / "epoch_audit.json")

        self.assertEqual(hashes["trainer"], EXPECTED_TRAINER_SHA256)
        self.assertEqual(hashes["ranker"], EXPECTED_RANKER_SHA256)
        self.assertTrue(all(item["passed"] for item in checkpoints))
        self.assertTrue(all(item["passed"] for item in checks))
        self.assertTrue(_epoch_audit(epochs)["passed"])

    def test_target_free_contract_has_no_target_or_count_input(self) -> None:
        contract = _target_free_source_contract()

        self.assertTrue(contract["signature_candidates_only"])
        self.assertTrue(contract["no_target_or_count_identifiers"])

    def test_non_bitwise_forward_can_pass_target_isolation(self) -> None:
        contract = {
            "signature_candidates_only": True,
            "no_target_or_count_identifiers": True,
        }
        replay = {
            "step0": {"passed": True, "bitwise_equal_batches": 25},
            "final": {"passed": True, "bitwise_equal_batches": 0},
        }

        self.assertTrue(_target_isolation_gate(contract, replay))


if __name__ == "__main__":
    unittest.main()
