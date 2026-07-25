from __future__ import annotations

import unittest

from chemvector.audit_evidence_carrier_stroke_pairing import (
    hard_preprocessor_gate,
    partition_molecule_disjoint_validation,
    summarize_records,
    validation_sample_identity,
)


def record(*, cross: int = 0, packed_supported: int = 2) -> dict:
    return {
        "sample_key": "sample",
        "raw_labels": {"positive": 4, "negative": 0, "ignored": 1, "total": 5},
        "packed_labels": {
            "positive": packed_supported,
            "negative": 0,
            "ignored": 1,
            "total": 3,
        },
        "compatible_pairs": 2,
        "selected_pairs": 2,
        "unpaired_raw_anchors": 1,
        "unpaired_positive_raw_anchors": 0,
        "permutation_determinism": {
            "reverse_input_carrier_set_equal": True,
            "reverse_input_pair_partition_equal": True,
            "exact": True,
        },
        "pair_classes": {
            "same_target_positive": 2 - cross,
            "cross_target_positive": cross,
        },
        "cross_target_records": [],
        "multiplicity_two": {"expected": 2, "paired": 2},
        "extent_eligible_support": {
            "expected_ordinary_members": 2,
            "raw_supported_members": 2,
            "packed_supported_members": packed_supported,
            "raw_positive_carriers": 4,
            "packed_positive_carriers": packed_supported,
            "raw_gt_positive_dedup_ratio": 0.5,
            "packed_gt_positive_dedup_ratio": 1.0,
            "lost_raw_supported_members": 2 - packed_supported,
        },
        "packed_positive_multiplicity": {
            "0": 2 - packed_supported,
            "1": packed_supported,
        },
        "parallel_2": {
            "total": 1,
            "raw_both_supported": 1,
            "strict_preserved": int(cross == 0 and packed_supported == 2),
            "direct_cross_target_merges": cross,
        },
        "sample_exact": {
            "extent_eligible_support_all_members": packed_supported == 2,
            "one_positive_carrier_per_ordinary_member": (
                cross == 0 and packed_supported == 2
            ),
        },
    }


class EvidenceCarrierStrokePairingAuditTest(unittest.TestCase):
    def test_summary_exposes_reduction_deduplication_and_exactness(self) -> None:
        summary = summarize_records([record()])

        self.assertEqual(summary["anchors"]["all_anchor_reduction"], 2)
        self.assertEqual(summary["pairing"]["cross_target_merges"], 0)
        self.assertEqual(summary["pairing"]["multiplicity_two_pair_recall"], 1.0)
        self.assertEqual(
            summary["extent_eligible_carrier_support"]["packed_support_coverage"],
            1.0,
        )
        self.assertEqual(
            summary["extent_eligible_carrier_support"][
                "packed_gt_positive_dedup_ratio"
            ],
            1.0,
        )
        self.assertEqual(
            summary["sample_exact"][
                "extent_eligible_support_all_members_rate"
            ],
            1.0,
        )
        self.assertEqual(hard_preprocessor_gate(summary)["status"], "GO")

    def test_cross_target_merge_forces_hard_preprocessor_no_go(self) -> None:
        summary = summarize_records([record(cross=1)])

        gate = hard_preprocessor_gate(summary)

        self.assertEqual(gate["status"], "NO-GO")
        self.assertFalse(gate["checks"]["cross_target_merges_equal_0"])

    def test_partition_excludes_molecule_or_source_structure_overlap(self) -> None:
        samples = [
            identity_sample("001", "mol-a", "validation_000001", "original"),
            identity_sample("002", "mol-b", "validation_000002", "coordgen"),
            identity_sample("003", "mol-a", "validation_000001", "stress"),
            identity_sample("004", "mol-c", "validation_000003", "rdkit"),
            # Inconsistent upstream molecule ID still cannot bypass source identity.
            identity_sample("005", "mol-x", "validation_000002", "stress"),
        ]

        partition = partition_molecule_disjoint_validation(
            samples,
            calibration_count=2,
        )

        self.assertEqual(
            [item.sample_key for item in partition.calibration_identities],
            ["001", "002"],
        )
        self.assertEqual(
            [item.sample_key for item in partition.molecule_disjoint_identities],
            ["004"],
        )
        self.assertEqual(
            [item.sample_key for item in partition.excluded_cross_render_identities],
            ["003", "005"],
        )
        calibration_molecules = {
            item.molecule_id for item in partition.calibration_identities
        }
        disjoint_molecules = {
            item.molecule_id for item in partition.molecule_disjoint_identities
        }
        calibration_structures = {
            item.source_structure_id for item in partition.calibration_identities
        }
        disjoint_structures = {
            item.source_structure_id
            for item in partition.molecule_disjoint_identities
        }
        self.assertFalse(calibration_molecules & disjoint_molecules)
        self.assertFalse(calibration_structures & disjoint_structures)

    def test_identity_requires_variant_suffix(self) -> None:
        sample = identity_sample("001", "mol-a", "validation_000001", "stress")
        sample["meta"]["source_sample_key"] = "validation_000001_original"

        with self.assertRaisesRegex(ValueError, "source_variant"):
            validation_sample_identity(sample)


def identity_sample(
    key: str,
    molecule_id: str,
    source_record: str,
    variant: str,
) -> dict:
    return {
        "key": key,
        "meta": {
            "sample_key": key,
            "molecule_id": molecule_id,
            "source_sample_key": f"{source_record}_{variant}",
            "source_variant": variant,
            "source_pilot": "pilot",
            "source_split": "validation",
        },
    }


if __name__ == "__main__":
    unittest.main()
