from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from Data.sdf.build_visual_svg_layers_c_fresh_dev_preflight import (
    COMMITTED_MOLECULE_COUNT,
    IDENTITY_AUDIT_SCHEMA_VERSION,
    MEMBER_SUFFIXES,
    PreflightBuildError,
    SELECTION_FIELDS,
    SELECTION_SCHEMA_VERSION,
    canonical_json_bytes,
    file_sha256,
    load_committed_selection,
    selection_rank,
    source_bytes_commitment_sha256,
    style_id,
)
from Data.sdf.validate_visual_svg_layers_c_fresh_dev_preflight import (
    FAIL_STATUS,
    PreflightValidationError,
    _style_id as independently_validated_style_id,
    _validate_bond_groups,
    validate_preflight,
)
from Data.sdf.select_visual_svg_layers_c_fresh_dev import (
    FreshDevSelectionError,
    _validate_frozen_source as validate_selector_source,
)
from Data.sdf.visual_svg_v0_2 import visual_primitives_v0_1_to_visual_svg_v0_2


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(canonical_json_bytes(value, newline=True))


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SELECTION_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def _freeze(path: Path) -> None:
    _write_json(
        path,
        {
            "freeze_id": "visual-svg-layers-c-v0.1-phase-a-20260722",
            "phase": "A",
            "status": "frozen_fresh_dev_preflight_authorized",
            "training_steps": 0,
            "parameter_updates": 0,
            "splits": {
                "fresh_dev": {
                    "authorized": True,
                    "role": "dev",
                    "split": "fresh_dev",
                    "molecule_count": 2000,
                    "sample_count": 8000,
                }
            },
            "preflight": {
                "run_model": False,
                "molecule_count": 16,
                "sample_count": 64,
                "selection": "first_16_committed_fresh_dev_molecules",
            },
            "sdf": {"protocol": "source-sdf-byte-preserving-0.1"},
            "render": {
                "render_profile": "visual-svg-layers-c-clean-bw-768-v0.1"
            },
            "sequence": {"max_tokens": 4096},
            "sample_contract": {
                "members": ["sample" + suffix for suffix in MEMBER_SUFFIXES]
            },
        },
    )


def _selection_rows(count: int) -> list[dict[str, str]]:
    rows = []
    for index in range(count):
        source_id = f"TEST{index:06d}"
        row = {
            "selection_index": "",
            "rank_sha256": "",
            "molecule_id": f"chembl:{source_id}",
            "inchikey_full": f"{index:027d}",
            "scaffold_id": f"scaffold-{index:06d}",
            "source_record_id": f"chembl:{source_id}",
            "source": "chembl",
            "source_id": source_id,
            "source_split": "train",
            "source_shard": "shards/train-000000.tar",
            "source_structure": (
                f"structures/train-000000/chembl_{source_id}.sdf"
            ),
            "source_sdf_sha256": "1" * 64,
            "source_sdf_bytes": "123",
            "source_chemmol_sha256": "2" * 64,
            "source_atom_count": "2",
            "source_bond_count": "1",
        }
        row["rank_sha256"] = selection_rank(row)
        rows.append(row)
    rows.sort(key=lambda item: item["rank_sha256"])
    for index, row in enumerate(rows):
        row["selection_index"] = str(index)
    return rows


def _commitment_fixture(root: Path, *, count: int = COMMITTED_MOLECULE_COUNT):
    freeze = root / "phase_a_freeze.json"
    manifest = root / "selection.committed.csv"
    audit = root / "identity_isolation_report.json"
    commitment = root / "selection.commitment.json"
    real_freeze = (
        Path(__file__).resolve().parents[2]
        / "Data/sdf/visual_svg_layers_c_v0_1/phase_a_freeze.json"
    )
    freeze.write_bytes(real_freeze.read_bytes())
    rows = _selection_rows(count)
    _write_csv(manifest, rows)
    zero = "0" * 64
    _write_json(
        audit,
        {
            "schema_version": IDENTITY_AUDIT_SCHEMA_VERSION,
            "status": "passed",
            "fresh_dev_selection_manifest_sha256": file_sha256(manifest),
            "comparisons": {
                "frozen_b_p0_train_exposure": {
                    "molecule_id_overlap": 0,
                    "inchikey_full_overlap": 0,
                    "scaffold_id_overlap": 0,
                    "source_record_id_overlap": 0,
                },
                "smoke_train_32": {
                    "molecule_id_overlap": 0,
                    "inchikey_full_overlap": 0,
                    "scaffold_id_overlap": 0,
                    "source_record_id_overlap": 0,
                },
            },
            "molecule_id_exclusion_set_sha256": zero,
            "inchikey_full_exclusion_set_sha256": zero,
            "scaffold_id_exclusion_set_sha256": zero,
            "source_record_id_exclusion_set_sha256": zero,
            "isolation_algorithm_sha256": zero,
            "isolation_config_sha256": zero,
            "scaffold_id_algorithm_sha256": zero,
            "source_record_normalization_algorithm_sha256": zero,
            "exposure_manifest_sha256": zero,
            "style_id_algorithm_sha256": zero,
        },
    )
    _write_json(
        commitment,
        {
            "schema_version": SELECTION_SCHEMA_VERSION,
            "status": "committed",
            "freeze_id": "visual-svg-layers-c-v0.1-phase-a-20260722",
            "role": "dev",
            "split": "fresh_dev",
            "molecule_count": 2000,
            "sample_count": 8000,
            "selection_ordering": "sha256_rank_v0.1",
            "freeze": {
                "path": "phase_a_freeze.json",
                "sha256": file_sha256(freeze),
            },
            "selection_manifest": {
                "path": "selection.committed.csv",
                "sha256": file_sha256(manifest),
            },
            "source_bytes_commitment": {
                "protocol": "source-sdf-byte-preserving-0.1",
                "sha256": source_bytes_commitment_sha256(rows),
                "molecule_count": 2000,
            },
            "identity_isolation_report": {
                "path": "identity_isolation_report.json",
                "sha256": file_sha256(audit),
            },
        },
    )
    return freeze, commitment, manifest, audit, rows


def _style_descriptor() -> dict:
    return {
        "schema": "visual-svg-layers-c-style-0.1",
        "canvas": {"height": "768.000000", "width": "768.000000"},
        "font_sha256": "73b6d579f94b4870856e4ae8ddc19cd31b353078a02db57edf93ee8ca5049f45",
        "font_family": "STIX",
        "render_profile": "visual-svg-layers-c-clean-bw-768-v0.1",
        "layout_variant": "original",
        "font_size_bucket": "normal",
        "bond_line_width_bucket": "normal",
        "atom_label_style": "rdkit-outline-glyphs",
        "bond_style": "rdkit-semantic-paths",
        "color_mode": "black_white",
        "background": "opaque_white",
        "antialias": "cairosvg-default",
    }


class FreshDevSelectionCommitmentTest(unittest.TestCase):
    def test_complete_2000_identity_commitment_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            freeze, commitment, _, _, rows = _commitment_fixture(root)

            loaded, loaded_commitment, audit = load_committed_selection(
                commitment, freeze
            )

            self.assertEqual(loaded, rows)
            self.assertEqual(len(loaded), 2000)
            self.assertEqual(loaded_commitment["status"], "committed")
            self.assertEqual(audit["status"], "passed")

    def test_temporary_16_identity_manifest_cannot_masquerade_as_commitment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            freeze, commitment, _, _, _ = _commitment_fixture(root, count=16)

            with self.assertRaisesRegex(
                PreflightBuildError, "selection_manifest_count"
            ):
                load_committed_selection(commitment, freeze)

    def test_nonzero_exposure_overlap_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            freeze, commitment, _, audit_path, _ = _commitment_fixture(root)
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit["comparisons"]["smoke_train_32"]["scaffold_id_overlap"] = 1
            _write_json(audit_path, audit)
            payload = json.loads(commitment.read_text(encoding="utf-8"))
            payload["identity_isolation_report"]["sha256"] = file_sha256(
                audit_path
            )
            _write_json(commitment, payload)

            with self.assertRaisesRegex(
                PreflightBuildError, "identity_audit_nonzero"
            ):
                load_committed_selection(commitment, freeze)

    def test_phase_freeze_bytes_are_pinned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            freeze, commitment, _, _, _ = _commitment_fixture(root)
            payload = json.loads(freeze.read_text(encoding="utf-8"))
            payload["objective"] = "tampered"
            _write_json(freeze, payload)

            with self.assertRaisesRegex(
                PreflightBuildError, "phase_a_freeze_sha256_mismatch"
            ):
                load_committed_selection(commitment, freeze)

    def test_selector_rejects_substitute_source_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                FreshDevSelectionError, "source_root_not_frozen"
            ):
                validate_selector_source(Path(directory).resolve())

    def test_builder_and_independent_style_hash_agree(self) -> None:
        descriptor = _style_descriptor()

        self.assertEqual(style_id(descriptor), independently_validated_style_id(descriptor))

    def test_independent_style_validator_rejects_font_family_alias(self) -> None:
        descriptor = _style_descriptor()
        descriptor["font_family"] = "not-stix"

        with self.assertRaisesRegex(PreflightValidationError, "style_font_allowlist"):
            independently_validated_style_id(descriptor)


class FreshDevPreflightValidatorTest(unittest.TestCase):
    def test_missing_preflight_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = (
                Path(__file__).resolve().parents[2]
                / "Data/sdf/webdataset_full_v0"
            )

            report = validate_preflight(root / "fresh_dev", source)

            self.assertEqual(report["status"], FAIL_STATUS)
            self.assertIn("missing_input", report["errors"][0])

    def test_double_and_triple_are_one_group_with_two_chemical_endpoints(self) -> None:
        primitives = {
            "schema_version": "visual-primitives-0.1",
            "sample_key": "fresh_dev_000000_original",
            "molecule_id": "chembl:TEST",
            "width": 768,
            "height": 768,
            "coordinate_space": "image_pixels",
            "coordinate_quantization": 2047,
            "stroke_width_scale": 10,
            "bond_groups": [
                {
                    "bond_id": 0,
                    "begin_atom": 0,
                    "end_atom": 1,
                    "chemical_type": "DOUBLE",
                    "display_type": "DOUBLE",
                    "direction": "NONE",
                    "primitive_ids": [0, 1],
                },
                {
                    "bond_id": 1,
                    "begin_atom": 1,
                    "end_atom": 2,
                    "chemical_type": "TRIPLE",
                    "display_type": "TRIPLE",
                    "direction": "NONE",
                    "primitive_ids": [2, 3, 4],
                },
            ],
            "text_instances": [],
            "primitives": [],
        }
        for primitive_id in range(5):
            primitives["primitives"].append(
                {
                    "kind": "line",
                    "owner_kind": "bond",
                    "owner_id": 0 if primitive_id < 2 else 1,
                    "points": [[10.0, 10.0 + primitive_id], [100.0, 10.0 + primitive_id]],
                    "quantized_points": [[27, 27 + primitive_id], [267, 27 + primitive_id]],
                    "stroke_width": 2.0,
                    "stroke_width_q": 20,
                    "quantization_adjusted": False,
                    "source_z_index": primitive_id,
                    "primitive_id": primitive_id,
                    "canonical_index": primitive_id,
                }
            )
        parsed = visual_primitives_v0_1_to_visual_svg_v0_2(primitives)

        doubles, triples = _validate_bond_groups(
            primitives, parsed, sdf_bonds=2
        )

        self.assertEqual((doubles, triples), (1, 1))
        self.assertEqual(len(parsed.bond_groups), 2)
        self.assertEqual(
            [len(group.members) for group in parsed.bond_groups], [2, 3]
        )


if __name__ == "__main__":
    unittest.main()
