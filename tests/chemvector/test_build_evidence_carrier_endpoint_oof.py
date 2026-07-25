from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
import unittest
from unittest.mock import patch

import torch

import chemvector.build_evidence_carrier_endpoint_oof as builder


def _component(index: int) -> dict:
    component = json.dumps(
        {"molecule_ids": [f"m{index}"], "source_structure_ids": [f"s{index}"]},
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "component": component,
        "sample_keys": [f"{index}-{variant}" for variant in range(4)],
    }


class EndpointOOFBuilderTest(unittest.TestCase):
    def test_fold_partition_is_deterministic_and_component_disjoint(self) -> None:
        rows = [_component(index) for index in range(6)]
        ranked = []
        for row in rows:
            rank = hashlib.sha256(
                f"{builder.FOLD_SEED}\n{row['component']}".encode()
            ).hexdigest()
            ranked.append(
                {
                    "rank": rank,
                    "component": row["component"],
                    "sample_keys": sorted(row["sample_keys"]),
                }
            )
        ranked.sort(key=lambda row: row["rank"])
        folds = [ranked[index * 2 : (index + 1) * 2] for index in range(3)]
        row_hashes = tuple(
            hashlib.sha256(builder._json_bytes(fold)).hexdigest() for fold in folds
        )
        key_hashes = tuple(
            hashlib.sha256(
                (
                    "\n".join(sorted(key for row in fold for key in row["sample_keys"]))
                    + "\n"
                ).encode()
            ).hexdigest()
            for fold in folds
        )
        all_hash = hashlib.sha256(builder._json_bytes(ranked)).hexdigest()
        with ExitStack() as stack:
            for name, value in {
                "EXPECTED_COMPONENT_COUNT": 6,
                "EXPECTED_COMPONENTS_PER_FOLD": 2,
                "EXPECTED_SAMPLES_PER_FOLD": 8,
                "EXPECTED_ALL_ROWS_SHA256": all_hash,
                "EXPECTED_FOLD_ROWS_SHA256": row_hashes,
                "EXPECTED_FOLD_KEYS_SHA256": key_hashes,
            }.items():
                stack.enter_context(patch.object(builder, name, value))
            result = builder.build_fold_partition(rows)

        self.assertEqual([fold["component_count"] for fold in result["folds"]], [2, 2, 2])
        key_sets = [set(fold["sample_keys"]) for fold in result["folds"]]
        self.assertFalse(key_sets[0] & key_sets[1])
        self.assertFalse(key_sets[0] & key_sets[2])
        self.assertFalse(key_sets[1] & key_sets[2])

    def test_index_rejects_embedded_records(self) -> None:
        payload = {
            "schema_version": builder.INDEX_SCHEMA_VERSION,
            "status": builder.STATUS_COMPLETE,
            "formal_eligible": True,
            **{name: False for name in builder._FORBIDDEN_FLAGS},
            "folds": [
                {
                    "fold": fold,
                    "component_rows_sha256": builder.EXPECTED_FOLD_ROWS_SHA256[fold],
                    "sidecar_sha256": "1" * 64,
                    "records": [],
                }
                for fold in range(3)
            ],
        }
        with self.assertRaisesRegex(builder.EndpointOOFDataError, "embeds records"):
            builder.validate_index(payload)

    def test_enrichment_derives_labels_from_anchor_targets(self) -> None:
        identity = {
            "sample_key": "0-0",
            "molecule_id": "m0",
            "source_structure_id": "s0",
        }
        fit = {
            "identity": identity,
            "partition": "fit",
            "component": "{}",
            "rank": "rank",
            "target_free": {"pair_indices": torch.tensor(((0, 1),))},
            "target_aware": {"pair_labels": torch.tensor((1,))},
        }
        source = {
            "identity": identity,
            "partition": "fit",
            "anchor_segments_pixels": torch.zeros((2, 2, 2)),
            "anchor_target_line_ids": torch.tensor((3, 3)),
            "anchor_source_indices": torch.tensor((0, 1)),
            "anchor_group_ids": torch.tensor((0, 0)),
            "anchor_group_types": torch.tensor((2, 2)),
        }
        partition = {
            "folds": [
                {"fold": 0, "sample_keys": ["0-0"]},
                {"fold": 1, "sample_keys": []},
                {"fold": 2, "sample_keys": []},
            ]
        }
        result = builder._enrich_records([fit], [source], partition)
        self.assertEqual(result[0][0]["oof_fold"], 0)
        self.assertIn("evaluation_target", result[0][0]["target_aware"])


if __name__ == "__main__":
    unittest.main()
