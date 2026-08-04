"""Tests for the P3-C fixed Atom assignment cache."""

import copy
import unittest

from chemvector.training.atoms.train_g3_p3c_fixed_assignment_replay_train4937 import (
    assignment_entries_digest,
    validate_assignment_cache,
)


class G3P3CFixedAssignmentTests(unittest.TestCase):
    @staticmethod
    def payload() -> dict:
        entries = [{
            "sample_key": "train_000001_rdkit",
            "atoms": 2,
            "capacity_replay": True,
            "query_indices": [4, 9],
            "target_indices": [0, 1],
            "candidate_indices": [2, 7],
        }]
        return {
            "entries": entries,
            "entries_sha256": assignment_entries_digest(entries),
            "assignments": 2,
        }

    def test_cache_validates_fixed_unique_assignments(self) -> None:
        lookup = validate_assignment_cache(self.payload())
        self.assertEqual(set(lookup), {"train_000001_rdkit"})
        self.assertEqual(lookup["train_000001_rdkit"]["query_indices"].tolist(), [4, 9])

    def test_cache_rejects_digest_tampering(self) -> None:
        payload = copy.deepcopy(self.payload())
        payload["entries"][0]["candidate_indices"][0] = 3
        with self.assertRaisesRegex(ValueError, "digest differs"):
            validate_assignment_cache(payload)

    def test_cache_rejects_duplicate_candidate(self) -> None:
        payload = self.payload()
        payload["entries"][0]["candidate_indices"] = [2, 2]
        payload["entries_sha256"] = assignment_entries_digest(payload["entries"])
        with self.assertRaisesRegex(ValueError, "uniqueness differs"):
            validate_assignment_cache(payload)


if __name__ == "__main__":
    unittest.main()
