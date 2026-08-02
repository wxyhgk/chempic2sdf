import unittest

import torch

from chemvector.audit_g3_p0k_relation_alignment import (
    coverage_from_metrics,
    pair_head_from_state,
    strict_final_capability,
    validate_optimizer_scope,
    validate_pair_forward_source,
)
from chemvector.direct_atom_pair_bond import (
    DirectAtomPairBond,
    DirectAtomPairBondConfig,
)


def tiny_pair() -> DirectAtomPairBond:
    return DirectAtomPairBond(
        DirectAtomPairBondConfig(
            feature_channels=8,
            atom_feature_dim=16,
            longitudinal_samples=8,
            transverse_samples=5,
            hidden_dim=16,
            pair_chunk_size=8,
        )
    )


class G3P0KRelationAlignmentAuditTests(unittest.TestCase):
    def test_strict_final_requires_oracle_32_and_proves_zero_break(self) -> None:
        metrics = {
            "predicted_atom_composition": {
                "gt_bond_count": {"typed_graph_exact_images": 32},
                "deploy_bond_count": {"typed_graph_exact_images": 32},
            },
            "gt_atom_oracle_replay": {
                "gt_count": {"structural_graph_exact_images": 32},
                "deploy_count": {"structural_graph_exact_images": 32},
            },
        }
        p0j = {
            "p0g_slot_replay_feature": {
                "gt_bond_count": {"typed_graph_exact_images": 26}
            }
        }

        result = strict_final_capability(metrics, p0j)

        self.assertTrue(result["all_four_branches_32_of_32"])
        self.assertTrue(result["p0j_per_sample_break_impossible"])
        self.assertEqual(result["p0j_exact_image_improvement"], 6)
        metrics["gt_atom_oracle_replay"]["gt_count"][
            "structural_graph_exact_images"
        ] = 31
        with self.assertRaisesRegex(ValueError, "four 32/32"):
            strict_final_capability(metrics, p0j)

    def test_real_direct_pair_state_strict_loads(self) -> None:
        source = tiny_pair()

        restored = pair_head_from_state(source.state_dict(), source.config)

        self.assertFalse(restored.training)
        self.assertEqual(set(restored.state_dict()), set(source.state_dict()))
        for name, value in source.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[name], value)

    def test_optimizer_scope_requires_exact_direct_pair_parameters(self) -> None:
        pair = tiny_pair()
        optimizer = torch.optim.AdamW(pair.parameters(), lr=1e-3)
        payload = {
            "pair_bond": pair.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        result = validate_optimizer_scope(
            payload,
            pair,
            [name for name, _ in pair.named_parameters()],
        )

        self.assertTrue(result["pair_head_only"])
        self.assertEqual(
            result["optimizer_parameter_tensors"],
            len(list(pair.parameters())),
        )
        payload["optimizer"]["param_groups"].append(
            dict(payload["optimizer"]["param_groups"][0])
        )
        with self.assertRaisesRegex(ValueError, "one pair-head parameter group"):
            validate_optimizer_scope(
                payload, pair, [name for name, _ in pair.named_parameters()]
            )

    def test_forward_source_accepts_predicted_and_rejects_gt_anchor(self) -> None:
        valid = '''
def p0k_training_step(p0g, features, atoms, targets, pairs, samples):
    output = p0g.pair_bond(
        features, atoms["atom_features"], atoms["atom_anchors"],
        atoms["atom_scores"], pairs, samples,
    )
    return build_labels(targets)
'''
        invalid = valid.replace('atoms["atom_anchors"]', 'targets["atom_anchors"]')

        result = validate_pair_forward_source(valid)

        self.assertEqual(result["predicted_anchor_inputs"], 1)
        self.assertEqual(result["gt_anchor_inputs"], 0)
        with self.assertRaisesRegex(ValueError, "predicted anchors"):
            validate_pair_forward_source(invalid)

    def test_coverage_requires_all_fixed_totals(self) -> None:
        metrics = {
            "predicted_atom_composition": {
                "images": 32,
                "atom_targets": 1424,
                "atom_matched": 1424,
                "atom_set_exact_images": 32,
                "target_bonds": 1508,
                "representable_bonds": 1508,
                "pair_universe": 37380,
            }
        }

        coverage = coverage_from_metrics(metrics)

        self.assertTrue(coverage["atom_mapping_complete"])
        self.assertTrue(coverage["bond_coverage_complete"])
        self.assertTrue(coverage["pair_universe_complete"])
        metrics["predicted_atom_composition"]["atom_matched"] = 1423
        with self.assertRaisesRegex(ValueError, "coverage"):
            coverage_from_metrics(metrics)


if __name__ == "__main__":
    unittest.main()
