from __future__ import annotations

from contextlib import ExitStack
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import chemvector.build_evidence_carrier_pair_residual_fit_dev as builder
from chemvector.evidence_carrier_sample_threshold_predictor import (
    SAMPLE_THRESHOLD_BASELINE_TAU,
)


def _identity(sample_key: str, molecule: str, source: str) -> dict[str, str]:
    return {
        "sample_key": sample_key,
        "molecule_id": molecule,
        "source_structure_id": source,
        "source_pilot": "fixture",
        "source_split": "train",
        "source_record": f"source-{sample_key}",
        "render_variant": f"variant-{sample_key}",
    }


def _source_record(sample_key: str, molecule: str, source: str) -> dict:
    return {
        "identity": _identity(sample_key, molecule, source),
        "partition": "fit",
    }


def _json_sha(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _key_sha(keys: list[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(keys)) + "\n").encode()).hexdigest()


class _TinyContract:
    def __init__(self) -> None:
        # Both axes exercise transitive closure: component A shares molecule,
        # component B shares source structure.
        self.records = [
            _source_record("0000", "m0", "s0"),
            _source_record("0001", "m0", "s1"),
            _source_record("0002", "m0", "s0"),
            _source_record("0003", "m0", "s1"),
            _source_record("0004", "m1", "s2"),
            _source_record("0005", "m2", "s2"),
            _source_record("0006", "m1", "s2"),
            _source_record("0007", "m2", "s2"),
        ]
        rows = []
        for molecules, sources, keys in (
            (["m0"], ["s0", "s1"], ["0000", "0001", "0002", "0003"]),
            (["m1", "m2"], ["s2"], ["0004", "0005", "0006", "0007"]),
        ):
            component = json.dumps(
                {
                    "molecule_ids": molecules,
                    "source_structure_ids": sources,
                },
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            rank = hashlib.sha256(
                f"{builder.SPLIT_SEED}\n{component}".encode()
            ).hexdigest()
            rows.append(
                {"rank": rank, "component": component, "sample_keys": keys}
            )
        rows.sort(key=lambda row: row["rank"])
        self.dev_rows = rows[:1]
        self.fit_rows = rows[1:]
        self.fit_keys = sorted(self.fit_rows[0]["sample_keys"])
        self.dev_keys = sorted(self.dev_rows[0]["sample_keys"])

    def patches(self) -> ExitStack:
        stack = ExitStack()
        values = {
            "EXPECTED_SOURCE_SAMPLE_COUNT": 8,
            "EXPECTED_SOURCE_COMPONENT_COUNT": 2,
            "EXPECTED_SOURCE_MOLECULE_COUNT": 3,
            "EXPECTED_SOURCE_STRUCTURE_COUNT": 3,
            "EXPECTED_FIT_COMPONENT_COUNT": 1,
            "EXPECTED_DEV_COMPONENT_COUNT": 1,
            "EXPECTED_FIT_SAMPLE_COUNT": 4,
            "EXPECTED_DEV_SAMPLE_COUNT": 4,
            "EXPECTED_FIT_PAIR_COUNT": 8,
            "EXPECTED_DEV_PAIR_COUNT": 8,
            "EXPECTED_FIT_POSITIVE_COUNT": 4,
            "EXPECTED_FIT_NEGATIVE_COUNT": 4,
            "EXPECTED_DEV_POSITIVE_COUNT": 4,
            "EXPECTED_DEV_NEGATIVE_COUNT": 4,
            "EXPECTED_FIT_COMPONENT_MANIFEST_SHA256": _json_sha(self.fit_rows),
            "EXPECTED_DEV_COMPONENT_MANIFEST_SHA256": _json_sha(self.dev_rows),
            "EXPECTED_FIT_SAMPLE_KEY_SHA256": _key_sha(self.fit_keys),
            "EXPECTED_DEV_SAMPLE_KEY_SHA256": _key_sha(self.dev_keys),
            "BASE_CACHE_SHA256": "a" * 64,
            "AUGMENTATION_CACHE_SHA256": "b" * 64,
            "CHECKPOINT_SHA256": "c" * 64,
            "UPSTREAM_P0_CHECKPOINT_SHA256": "d" * 64,
        }
        for name, value in values.items():
            stack.enter_context(patch.object(builder, name, value))
        return stack


def _scored_source(record: dict, offset: float) -> tuple[dict, torch.Tensor, torch.Tensor]:
    current = copy.deepcopy(record)
    current.update(
        {
            "anchor_segments_pixels": torch.tensor(
                (
                    ((0.0, 0.0), (10.0, 0.0)),
                    ((0.0, 1.0), (10.0, 1.0)),
                    ((0.0, 2.0), (10.0, 2.0)),
                    ((0.0, 3.0), (10.0, 3.0)),
                ),
                dtype=torch.float32,
            ),
            "anchor_target_line_ids": torch.tensor((0, 0, 1, 2)),
            "anchor_source_indices": torch.arange(4),
            "anchor_group_ids": torch.tensor((0, 0, 1, 2)),
            "anchor_group_types": torch.ones(4, dtype=torch.long),
            "pair_indices": torch.tensor(((2, 3), (0, 1))),
            "pair_labels": torch.tensor((0, 1)),
        }
    )
    scores = torch.tensor(
        (
            SAMPLE_THRESHOLD_BASELINE_TAU + 0.2 + offset,
            SAMPLE_THRESHOLD_BASELINE_TAU + 0.1 + offset,
        ),
        dtype=torch.float32,
    )
    descriptors = torch.arange(40, dtype=torch.float32).reshape(2, 20)
    descriptors = descriptors + offset
    return current, scores, descriptors


def _fixture_payloads(contract: _TinyContract):
    split = builder.build_component_partition(contract.records)
    scored = [_scored_source(record, index / 100.0) for index, record in enumerate(contract.records)]
    sources = [item[0] for item in scored]
    scores = [item[1] for item in scored]
    descriptors = [item[2] for item in scored]
    fit_records = builder._partition_records(
        sources, scores, descriptors, split["fit_components"], partition="fit"
    )
    dev_records = builder._partition_records(
        sources, scores, descriptors, split["dev_components"], partition="dev"
    )
    source_sha = builder._expected_source_sha256()
    implementation = builder._implementation_sha256()
    fit_sidecar = builder._build_sidecar(
        partition="fit",
        records=fit_records,
        component_rows=split["fit_components"],
        source_sha256=source_sha,
        implementation_sha256=implementation,
    )
    dev_sidecar = builder._build_sidecar(
        partition="dev",
        records=dev_records,
        component_rows=split["dev_components"],
        source_sha256=source_sha,
        implementation_sha256=implementation,
    )
    input_contract = builder._input_contract(
        base_path=Path("/tmp/base.pt"),
        augmentation_path=Path("/tmp/augmentation.pt"),
        checkpoint_path=Path("/tmp/pair.pt"),
        p0_path=Path("/tmp/p0.pt"),
        source_sha256=source_sha,
    )
    index = builder._build_index(
        fit_path=Path("/tmp/fit.pt"),
        fit_sha256="1" * 64,
        fit_sidecar=fit_sidecar,
        dev_path=Path("/tmp/dev.pt"),
        dev_sha256="2" * 64,
        dev_sidecar=dev_sidecar,
        input_contract=input_contract,
        source_sha256=source_sha,
        implementation_sha256=implementation,
    )
    return index, fit_sidecar, dev_sidecar


class PairResidualFitDevBuilderTest(unittest.TestCase):
    def test_component_partition_reproduces_canonical_transitive_contract(self) -> None:
        contract = _TinyContract()
        with contract.patches():
            result = builder.build_component_partition(contract.records)

        self.assertEqual(result["fit_components"], contract.fit_rows)
        self.assertEqual(result["dev_components"], contract.dev_rows)
        self.assertEqual(
            result["fit_component_manifest_sha256"], _json_sha(contract.fit_rows)
        )
        self.assertEqual(
            result["dev_component_manifest_sha256"], _json_sha(contract.dev_rows)
        )
        self.assertTrue(
            all(isinstance(row["component"], str) for row in contract.fit_rows)
        )

    def test_component_partition_rejects_original_calibration_record(self) -> None:
        contract = _TinyContract()
        contract.records[0]["partition"] = "calibration"
        with contract.patches(), self.assertRaisesRegex(
            builder.PairResidualManifestError,
            "calibration records are forbidden",
        ):
            builder.build_component_partition(contract.records)

    def test_partition_records_are_canonical_and_minimum_permission(self) -> None:
        contract = _TinyContract()
        with contract.patches():
            _, fit, dev = _fixture_payloads(contract)

        for record in fit["records"]:
            self.assertEqual(set(record["target_aware"]), {"pair_labels"})
            self.assertEqual(
                record["target_free"]["pair_indices"].tolist(),
                [[0, 1], [2, 3]],
            )
            self.assertEqual(record["target_aware"]["pair_labels"].tolist(), [1, 0])
            self.assertNotIn("pair_labels", record["target_free"])
            self.assertIsInstance(record["component"], str)
            self.assertEqual(len(record["rank"]), 64)
        for record in dev["records"]:
            self.assertEqual(
                set(record["target_aware"]),
                {"pair_labels", "evaluation_target"},
            )
            self.assertEqual(
                set(record["target_aware"]["evaluation_target"]),
                set(builder._EVALUATION_TARGET_FIELDS),
            )

    def test_sidecar_and_records_free_index_validation(self) -> None:
        contract = _TinyContract()
        with contract.patches():
            index, fit, dev = _fixture_payloads(contract)
            summaries = builder.validate_index_payload(
                index, fit_sidecar=fit, dev_sidecar=dev
            )

            self.assertEqual(summaries["fit"]["sample_count"], 4)
            self.assertEqual(summaries["dev"]["pair_count"], 8)
            self.assertNotIn("records", index["partitions"]["fit"])
            self.assertEqual(
                index["source_sha256"]["frozen_p0_checkpoint"], "d" * 64
            )

            leaked = copy.deepcopy(fit)
            leaked["records"][0]["target_free"]["pair_labels"] = torch.tensor((1, 0))
            with self.assertRaisesRegex(
                builder.PairResidualManifestError, "target-free schema"
            ):
                builder.validate_records_sidecar(
                    leaked, expected_partition="fit"
                )

            overprivileged = copy.deepcopy(fit)
            overprivileged["records"][0]["target_aware"]["evaluation_target"] = {}
            with self.assertRaisesRegex(
                builder.PairResidualManifestError, "target-aware schema"
            ):
                builder.validate_records_sidecar(
                    overprivileged, expected_partition="fit"
                )

            underprivileged = copy.deepcopy(dev)
            del underprivileged["records"][0]["target_aware"]["evaluation_target"]
            with self.assertRaisesRegex(
                builder.PairResidualManifestError, "target-aware schema"
            ):
                builder.validate_records_sidecar(
                    underprivileged, expected_partition="dev"
                )

            mislabeled = copy.deepcopy(dev)
            mislabeled["records"][0]["target_aware"]["pair_labels"][0] = 0
            with self.assertRaisesRegex(
                builder.PairResidualManifestError,
                "pair labels differ from anchor targets",
            ):
                builder.validate_records_sidecar(
                    mislabeled, expected_partition="dev"
                )

            duplicate_source = copy.deepcopy(dev)
            target = duplicate_source["records"][0]["target_aware"][
                "evaluation_target"
            ]
            target["anchor_source_indices"][1] = target["anchor_source_indices"][0]
            with self.assertRaisesRegex(
                builder.PairResidualManifestError,
                "source indices must be unique",
            ):
                builder.validate_records_sidecar(
                    duplicate_source, expected_partition="dev"
                )

    def test_fit_loader_does_not_open_dev_sidecar(self) -> None:
        contract = _TinyContract()
        with tempfile.TemporaryDirectory() as directory, contract.patches():
            root = Path(directory)
            index, fit, dev = _fixture_payloads(contract)
            fit_path = root / "fit.pt"
            dev_path = root / "dev.pt"
            index_path = root / "index.pt"
            torch.save(fit, fit_path)
            torch.save(dev, dev_path)
            index["partitions"]["fit"]["records_path"] = str(fit_path)
            index["partitions"]["fit"]["records_sha256"] = builder.file_sha256(
                fit_path
            )
            index["partitions"]["dev"]["records_path"] = str(dev_path)
            index["partitions"]["dev"]["records_sha256"] = builder.file_sha256(
                dev_path
            )
            torch.save(index, index_path)
            dev_path.unlink()

            loaded_index, loaded = builder.load_fit_dev_index(
                index_path, load_fit=True, load_dev=False
            )

            self.assertEqual(set(loaded), {"fit"})
            self.assertEqual(loaded["fit"]["partition"], "fit")
            self.assertEqual(
                loaded_index["partitions"]["dev"]["records_path"], str(dev_path)
            )

    def test_build_lock_is_exclusive_and_no_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                "index_path": root / "index.pt",
                "fit_path": root / "fit.pt",
                "dev_path": root / "dev.pt",
                "audit_path": root / "audit.json",
            }
            lock = builder._acquire_build_lock(**paths)
            try:
                with self.assertRaises(FileExistsError):
                    builder._acquire_build_lock(**paths)
                self.assertIn('"status": "started"', lock.read_text())
            finally:
                lock.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
