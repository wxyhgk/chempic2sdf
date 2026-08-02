import unittest

import torch

from chemvector.atom_feature_adapter import AtomFeatureAdapter
from chemvector.audit_g3_p0l_native_feature_adapter import (
    adapter_from_state,
    fixed_metrics,
    validate_adapter_optimizer,
    validate_distillation_source,
)


class G3P0LNativeFeatureAdapterAuditTests(unittest.TestCase):
    def test_real_adapter_state_strict_loads_and_optimizer_is_adapter_only(self) -> None:
        source = AtomFeatureAdapter()
        optimizer = torch.optim.AdamW(
            source.parameters(), lr=1e-3, weight_decay=0.0
        )
        payload = {
            "adapter": source.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        restored = adapter_from_state(payload["adapter"])
        scope = validate_adapter_optimizer(payload, restored)

        self.assertEqual(scope["parameter_count"], 147840)
        self.assertEqual(scope["parameter_names"], ["projection.weight", "projection.bias"])
        self.assertEqual(scope["optimizer_parameter_tensors"], 2)
        self.assertTrue(scope["adapter_only"])

    def test_source_audit_requires_detached_teacher_and_no_gt_arguments(self) -> None:
        loss = '''
def masked_feature_distillation_loss(adapter, native_features, teacher_features, atom_mask):
    adapted = adapter(native_features)
    return smooth_l1_loss(adapted[atom_mask], teacher_features.detach()[atom_mask])
'''
        step = '''
def p0l_training_step(adapter, optimizer, native_features, teacher_features, atom_mask):
    return masked_feature_distillation_loss(
        adapter, native_features, teacher_features, atom_mask
    )
'''

        result = validate_distillation_source(loss, step)

        self.assertTrue(result["teacher_detached"])
        self.assertFalse(result["gt_labels_in_training_loss"])
        with self.assertRaisesRegex(ValueError, "detached"):
            validate_distillation_source(loss.replace(".detach()", ""), step)
        with self.assertRaisesRegex(ValueError, "GT label"):
            validate_distillation_source(
                loss.replace("atom_mask):", "atom_mask, targets):"), step
            )

    def test_fixed_metrics_recomputes_ratio_controls_and_no_go(self) -> None:
        branch = {
            "atom_targets": 1424,
            "atom_matched": 1424,
            "target_bonds": 1508,
            "representable_bonds": 1508,
            "pair_universe": 37380,
            "gt_bond_count": {
                "skeleton_exact_images": 32,
                "typed_graph_exact_images": 7,
            },
            "deploy_bond_count": {
                "count_exact_images": 32,
                "skeleton_exact_images": 32,
                "typed_graph_exact_images": 7,
            },
        }
        metrics = {
            "feature_alignment": {
                "smooth_l1": 0.2,
                "step0_smooth_l1": 1.0,
                "smooth_l1_ratio_to_step0": 0.2,
                "valid_atoms": 1424,
                "feature_elements": 1424 * 384,
            },
            "adapted_native_feature": branch,
            "same_slot_p0k_control": {
                "gt_bond_count": {"typed_graph_exact_images": 32},
                "deploy_bond_count": {"typed_graph_exact_images": 32},
            },
            "gt_atom_oracle_control": {
                "gt_count": {"structural_graph_exact_images": 32},
                "deploy_count": {"structural_graph_exact_images": 32},
            },
        }

        result = fixed_metrics(metrics)

        self.assertTrue(result["coverage_complete"])
        self.assertTrue(result["controls_32_of_32"])
        self.assertFalse(result["feature_error_reduced_90_percent"])
        self.assertFalse(result["adapted_graph_gate"])
        metrics["feature_alignment"]["smooth_l1_ratio_to_step0"] = 0.3
        with self.assertRaisesRegex(ValueError, "ratio"):
            fixed_metrics(metrics)


if __name__ == "__main__":
    unittest.main()
