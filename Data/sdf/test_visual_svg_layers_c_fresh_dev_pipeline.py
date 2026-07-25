from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import audit_visual_svg_layers_c_fresh_dev_isolation as isolation
from . import build_visual_svg_layers_c_fresh_dev as builder
from . import commit_visual_svg_layers_c_fresh_dev as commitment


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


class FreshDevFullPipelineTest(unittest.TestCase):
    def test_group_diagnostics_schedule_keeps_preflight_and_stride_oracles(self) -> None:
        self.assertEqual(builder._group_diagnostics_mode(0), "full_raster")
        self.assertEqual(builder._group_diagnostics_mode(15), "full_raster")
        self.assertEqual(builder._group_diagnostics_mode(16), "structural")
        self.assertEqual(builder._group_diagnostics_mode(99), "structural")
        self.assertEqual(builder._group_diagnostics_mode(100), "full_raster")
        self.assertEqual(builder._group_diagnostics_mode(101), "structural")

    def test_atomic_install_restores_previous_commitment_on_fsync_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "commitment.dev.json"
            candidate = root / ".commitment.dev.fixture.tmp"
            output.write_bytes(b"old-commitment\n")
            candidate.write_bytes(b"new-commitment\n")
            with patch.object(
                commitment,
                "_fsync_directory",
                side_effect=(OSError("fsync failed"), None),
            ), self.assertRaisesRegex(OSError, "fsync failed"):
                commitment._atomic_install_with_restore(candidate, output)
            self.assertEqual(output.read_bytes(), b"old-commitment\n")
            self.assertFalse(candidate.exists())

    def test_isolation_rejects_heldout_path_variants(self) -> None:
        for value in (
            "/tmp/validation500",
            "/tmp/internal-test",
            "/tmp/sealed_audit",
            "/tmp/reserve",
            "/tmp/MR-TADF",
        ):
            with self.subTest(value=value), self.assertRaises(
                isolation.FreshDevIsolationError
            ):
                isolation._assert_allowed_path(Path(value), label="fixture")

    def test_legacy_style_signature_is_comparable_and_fail_closed(self) -> None:
        layers = {
            "split": "train",
            "width": 768,
            "height": 768,
            "variant": "rdkit",
            "render_parameters": {
                "layout": "rdkit",
                "font_file_sha256": "2" * 64,
                "font_size": 16,
                "bond_line_width": 2.0,
                "color_mode": "black_white",
            },
        }
        pilot = isolation._style_signature_from_layers(
            layers,
            pipeline_profile=isolation.PILOT_PIPELINE_PROFILE,
            expected_split="train",
            context="fixture",
        )
        smoke = isolation._style_signature_from_layers(
            layers,
            pipeline_profile=isolation.SMOKE_PIPELINE_PROFILE,
            expected_split="train",
            context="fixture",
        )
        self.assertEqual(pilot[0], smoke[0])
        self.assertEqual(pilot[1], "2" * 64)
        self.assertNotEqual(pilot[2], smoke[2])
        invalid = json.loads(json.dumps(layers))
        del invalid["render_parameters"]["font_size"]
        with self.assertRaises(isolation.FreshDevIsolationError):
            isolation._style_signature_from_layers(
                invalid,
                pipeline_profile=isolation.PILOT_PIPELINE_PROFILE,
                expected_split="train",
                context="fixture",
            )
        with self.assertRaisesRegex(
            isolation.FreshDevIsolationError,
            "style_id_zero_inference_requires_font_sha_disjoint",
        ):
            isolation._overlap_counts(
                {field: set() for field in isolation.IDENTITY_FIELDS},
                {field: 0 for field in isolation.IDENTITY_FIELDS},
                {
                    "style_id": {"style-v0.1:" + "1" * 64},
                    "font_sha": {"2" * 64},
                    "render_profile": {"fresh-profile"},
                },
                {
                    "style_id": set(),
                    "font_sha": {"2" * 64},
                    "render_profile": {"legacy-profile"},
                },
                legacy_style_id_missing=True,
            )

    def test_smoke_inventory_rejects_self_declared_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            smoke = Path(directory) / "smoke_train_32"
            smoke.mkdir()
            (smoke / "commitment.smoke.json").write_bytes(
                _json_bytes(
                    {
                        "role": "smoke",
                        "molecule_count": 8,
                        "sample_count": 32,
                        "shards": [
                            {
                                "path": "shards/smoke-train-000000.tar",
                                "sha256": "0" * 64,
                                "samples": 32,
                                "members": 384,
                            }
                        ],
                    }
                )
            )
            with self.assertRaisesRegex(
                isolation.FreshDevIsolationError,
                "smoke_commitment_sha256",
            ):
                isolation._smoke_style_sets(smoke)

    def test_commitment_fixture_binds_all_implementation_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            version_root = Path(directory) / "visual_svg_layers_c_v0_1"
            fresh_root = version_root / "fresh_dev"
            (fresh_root / "shards").mkdir(parents=True)
            (fresh_root / "preflight_64").mkdir()
            freeze_source = (
                Path(__file__).resolve().parent
                / "visual_svg_layers_c_v0_1/phase_a_freeze.json"
            )
            (version_root / "phase_a_freeze.json").write_bytes(
                freeze_source.read_bytes()
            )

            supporting = {
                "selection.committed.csv": b"selection\n",
                "replacement_pool.committed.csv": b"replacement\n",
                "identity_isolation_report.json": b"{}\n",
                "selection_provenance.json": b"{}\n",
            }
            for name, value in supporting.items():
                (fresh_root / name).write_bytes(value)
            selection_payload = {
                "selection_manifest": {
                    "path": "selection.committed.csv",
                    "sha256": _sha256(supporting["selection.committed.csv"]),
                },
                "fresh_replacement_manifest": {
                    "path": "replacement_pool.committed.csv",
                    "sha256": _sha256(
                        supporting["replacement_pool.committed.csv"]
                    ),
                },
                "identity_isolation_report": {
                    "path": "identity_isolation_report.json",
                    "sha256": _sha256(
                        supporting["identity_isolation_report.json"]
                    ),
                },
                "selection_provenance": {
                    "path": "selection_provenance.json",
                    "sha256": _sha256(supporting["selection_provenance.json"]),
                },
            }
            selection_path = fresh_root / "selection.commitment.json"
            selection_path.write_bytes(_json_bytes(selection_payload))

            manifest_values = {
                "molecule_manifest.dev.csv": b"molecule\n",
                "sample_manifest.dev.csv": b"sample\n",
                "paired_manifest.dev.csv": b"paired\n",
            }
            for name, value in manifest_values.items():
                (fresh_root / name).write_bytes(value)
            preflight_path = fresh_root / "preflight_64/preflight_validation.json"
            preflight_path.write_bytes(b"preflight\n")
            shard_path = fresh_root / "shards/dev-000000.tar"
            shard_path.write_bytes(b"fixture-shard")

            algorithm_paths = {
                "builder_sha256": (
                    Path(__file__).resolve().parent
                    / "build_visual_svg_layers_c_fresh_dev.py"
                ),
                "validator_sha256": (
                    Path(__file__).resolve().parent
                    / "validate_visual_svg_layers_c_fresh_dev.py"
                ),
                "isolation_auditor_sha256": (
                    Path(__file__).resolve().parent
                    / "audit_visual_svg_layers_c_fresh_dev_isolation.py"
                ),
            }
            implementation = {
                name: commitment._file_sha256(path)
                for name, path in algorithm_paths.items()
            }
            implementation.update(
                {
                    "style_id_algorithm_sha256": _sha256(b"style"),
                    "scaffold_id_algorithm_sha256": _sha256(b"scaffold"),
                    "source_record_normalization_algorithm_sha256": _sha256(
                        b"source-record"
                    ),
                }
            )
            isolation_hashes = {
                name: _sha256(name.encode("ascii"))
                for name in commitment.ISOLATION_HASH_FIELDS
            }
            isolation_hashes.update(
                {
                    "fresh_dev_molecule_manifest_sha256": _sha256(
                        manifest_values["molecule_manifest.dev.csv"]
                    ),
                    "fresh_dev_sample_manifest_sha256": _sha256(
                        manifest_values["sample_manifest.dev.csv"]
                    ),
                    "fresh_dev_paired_manifest_sha256": _sha256(
                        manifest_values["paired_manifest.dev.csv"]
                    ),
                    "isolation_algorithm_sha256": implementation[
                        "isolation_auditor_sha256"
                    ],
                    "style_id_algorithm_sha256": implementation[
                        "style_id_algorithm_sha256"
                    ],
                    "scaffold_id_algorithm_sha256": implementation[
                        "scaffold_id_algorithm_sha256"
                    ],
                    "source_record_normalization_algorithm_sha256": implementation[
                        "source_record_normalization_algorithm_sha256"
                    ],
                }
            )
            isolation_path = fresh_root / "isolation_report.json"
            isolation_path.write_bytes(
                _json_bytes(
                    {
                        "schema_version": commitment.ISOLATION_SCHEMA_VERSION,
                        "status": "passed",
                        "producer": "independent_isolation_auditor",
                        "role": "fresh_dev",
                        "hashes": isolation_hashes,
                    }
                )
            )
            shard_reference = {
                "path": "shards/dev-000000.tar",
                "sha256": _sha256(b"fixture-shard"),
                "bytes": len(b"fixture-shard"),
                "samples": commitment.SAMPLE_COUNT,
                "members": commitment.SAMPLE_COUNT
                * commitment.MEMBERS_PER_SAMPLE,
            }
            build_payload = {
                "schema_version": commitment.BUILD_SCHEMA_VERSION,
                "status": "built_pending_independent_validation",
                "finalizable": True,
                "molecules": commitment.MOLECULE_COUNT,
                "samples": commitment.SAMPLE_COUNT,
                "members": commitment.SAMPLE_COUNT
                * commitment.MEMBERS_PER_SAMPLE,
                "errors": [],
                "font_inventory": commitment.FONT_INVENTORY,
                "manifests": {
                    "molecule": {
                        "path": "molecule_manifest.dev.csv",
                        "sha256": _sha256(
                            manifest_values["molecule_manifest.dev.csv"]
                        ),
                    },
                    "sample": {
                        "path": "sample_manifest.dev.csv",
                        "sha256": _sha256(
                            manifest_values["sample_manifest.dev.csv"]
                        ),
                    },
                    "paired": {
                        "path": "paired_manifest.dev.csv",
                        "sha256": _sha256(
                            manifest_values["paired_manifest.dev.csv"]
                        ),
                    },
                },
                "shards": [shard_reference],
            }
            (fresh_root / "build.dev.json").write_bytes(
                _json_bytes(build_payload)
            )

            selection_hashes = {
                "selection_commitment_sha256": _sha256(
                    selection_path.read_bytes()
                ),
                "selection_manifest_sha256": _sha256(
                    supporting["selection.committed.csv"]
                ),
                "replacement_manifest_sha256": _sha256(
                    supporting["replacement_pool.committed.csv"]
                ),
                "identity_isolation_report_sha256": _sha256(
                    supporting["identity_isolation_report.json"]
                ),
                "selection_provenance_sha256": _sha256(
                    supporting["selection_provenance.json"]
                ),
            }
            artifact_sha256 = {
                "paired_manifest_sha256": _sha256(
                    manifest_values["paired_manifest.dev.csv"]
                ),
                "molecule_manifest_sha256": _sha256(
                    manifest_values["molecule_manifest.dev.csv"]
                ),
                "sample_manifest_sha256": _sha256(
                    manifest_values["sample_manifest.dev.csv"]
                ),
                "preflight_validation_sha256": _sha256(
                    preflight_path.read_bytes()
                ),
                "isolation_report_sha256": _sha256(
                    isolation_path.read_bytes()
                ),
                **selection_hashes,
            }
            validation_payload = {
                "schema_version": commitment.VALIDATION_SCHEMA_VERSION,
                "status": "passed",
                "finalizable": True,
                "fresh_audit_eligible": True,
                "molecules": commitment.MOLECULE_COUNT,
                "samples": commitment.SAMPLE_COUNT,
                "members": commitment.SAMPLE_COUNT
                * commitment.MEMBERS_PER_SAMPLE,
                "checked_samples": commitment.SAMPLE_COUNT,
                "checked_members": commitment.SAMPLE_COUNT
                * commitment.MEMBERS_PER_SAMPLE,
                "structure_validated_samples": commitment.SAMPLE_COUNT,
                "errors": [],
                "artifact_sha256": artifact_sha256,
                "implementation_sha256": implementation,
                "font_inventory": commitment.FONT_INVENTORY,
                "shards": [
                    {
                        key: shard_reference[key]
                        for key in ("path", "sha256", "samples", "members")
                    }
                ],
            }
            (fresh_root / "validation.json").write_bytes(
                _json_bytes(validation_payload)
            )

            args = argparse.Namespace(
                fresh_root=str(fresh_root),
                output=str(fresh_root / "commitment.dev.json"),
                overwrite=False,
            )
            fake_package = SimpleNamespace(
                sample_count=commitment.SAMPLE_COUNT,
                molecule_count=commitment.MOLECULE_COUNT,
            )
            final_path = fresh_root / "commitment.dev.json"

            def validate_candidate(path, **kwargs):
                candidate = Path(path)
                self.assertEqual(candidate.name, "commitment.dev.json")
                self.assertEqual(candidate.parent.name, "fresh_dev")
                self.assertIn(".commitment-candidate-", str(candidate.parent.parent))
                self.assertFalse(final_path.exists())
                self.assertEqual(
                    kwargs["expected_commitment_sha256"],
                    _sha256(candidate.read_bytes()),
                )
                return fake_package

            with patch(
                "chemvector.evidence_carrier_fresh_dev_data.load_fresh_evidence_carrier_package",
                side_effect=validate_candidate,
            ) as loader:
                report = commitment.commit(args)
            self.assertEqual(report["status"], "committed")
            self.assertTrue(report["production_loader_checked"])
            loader.assert_called_once()
            payload = json.loads(
                (fresh_root / "commitment.dev.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["implementation"], implementation)
            self.assertEqual(
                payload["implementation"]["scaffold_id_algorithm_sha256"],
                isolation_hashes["scaffold_id_algorithm_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
