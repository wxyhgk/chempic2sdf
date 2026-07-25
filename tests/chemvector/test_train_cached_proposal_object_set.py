import unittest

from chemvector.train_cached_proposal_object_set import (
    proposal_configs_compatible_for_warmstart,
)


class CachedProposalWarmstartTests(unittest.TestCase):
    def test_atom_capacity_may_expand_for_fused_proposals(self) -> None:
        base = {
            "maximum_atoms": 128,
            "maximum_bonds": 128,
            "atom_threshold": 0.01,
            "bond_threshold": 0.01,
        }
        expanded = {**base, "maximum_atoms": 144}

        self.assertTrue(proposal_configs_compatible_for_warmstart(base, expanded))

    def test_threshold_change_is_not_warmstart_compatible(self) -> None:
        base = {"maximum_atoms": 128, "atom_threshold": 0.01}
        changed = {"maximum_atoms": 144, "atom_threshold": 0.02}

        self.assertFalse(proposal_configs_compatible_for_warmstart(base, changed))


if __name__ == "__main__":
    unittest.main()
