import tempfile
import unittest
from pathlib import Path

import torch

from chemvector.build_proposal_object_cache import _cache_record, _expand_shards


class BuildProposalObjectCacheTests(unittest.TestCase):
    def test_cache_record_keeps_targets_and_compacts_floats(self) -> None:
        proposals = {
            "atom_target_matched": torch.tensor([[True, False]]),
            "bond_target_matched": torch.tensor([[True]]),
            "incidence_target_eligible": torch.tensor([[True]]),
        }
        for name in (
            "atom_features",
            "atom_geometry",
            "atom_scores",
            "bond_features",
            "bond_endpoints",
            "bond_types",
            "bond_scores",
        ):
            proposals[name] = torch.ones(1, 2, dtype=torch.float32)
        for name in (
            "atom_mask",
            "bond_mask",
            "atom_presence_targets",
            "bond_presence_targets",
            "bond_supervision_mask",
        ):
            proposals[name] = torch.ones(1, 2, dtype=torch.bool)
        proposals["target_atom_refs"] = torch.zeros(1, 2, 2, dtype=torch.long)
        targets = {
            "atom_mask": torch.tensor([[True, True]]),
            "bond_mask": torch.tensor([[True]]),
        }

        record = _cache_record(
            {"key": "sample", "meta": {"split": "train"}},
            proposals,
            targets,
            0,
        )

        self.assertEqual(record["atom_target_count"], 2)
        self.assertEqual(record["bond_target_count"], 1)
        self.assertEqual(record["atom_features"].dtype, torch.bfloat16)
        self.assertEqual(record["atom_mask"].dtype, torch.bool)

    def test_expand_shards_sorts_globs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "b.tar").touch()
            (root / "a.tar").touch()
            self.assertEqual(
                _expand_shards([str(root / "*.tar")]),
                [str(root / "a.tar"), str(root / "b.tar")],
            )


if __name__ == "__main__":
    unittest.main()
