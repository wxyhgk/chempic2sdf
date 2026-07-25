import csv
import json
from pathlib import Path
import tempfile
import unittest

from Data.sdf.build_visual_svg_capacity_100 import (
    MEMBER_SUFFIXES,
    json_bytes,
    sha256_bytes,
)
from Data.sdf.build_visual_svg_pilot_10k import (
    REJECTED_FIELDS,
    audit_split_shard,
    group_audit_key_sha256,
    group_audit_keys_by_split,
    load_source_rows,
    patch_sample_identity,
    selected_keys_by_split,
    write_failure,
    write_split_shard,
)


def sample_payload(key="000000001", split="train"):
    members = {
        ".png": b"png",
        ".bond.png": b"bond",
        ".primitives.json": b"{}\n",
        ".visualsvg.txt": b"VISUAL_SVG 0.1\nDOCUMENT_END\n",
        ".layers.json": b"{}\n",
        ".render.json": b"{}\n",
        ".chemmol.json": b"{}\n",
    }
    meta = {
        "schema_version": "visual-svg-pilot-meta-0.1",
        "sample_key": key,
        "molecule_id": "chembl:1",
        "split": split,
        "variant": "rdkit",
        "output_shard": f"shards/{split}-000000.tar",
        "sha256": {
            suffix.removeprefix("."): sha256_bytes(value)
            for suffix, value in members.items()
        },
    }
    members[".meta.json"] = json_bytes(meta)
    result = {
        "sample_key": key,
        "molecule_id": "chembl:1",
        "split": split,
        "variant": "rdkit",
        "output_shard": f"shards/{split}-000000.tar",
    }
    return key, members, result


class VisualSVGPilotBuilderTest(unittest.TestCase):
    def test_split_selection_and_limit_are_stable(self):
        rows = {
            "000000003": {"split": "train"},
            "000000001": {"split": "train"},
            "000000006": {"split": "validation"},
            "000000004": {"split": "validation"},
            "000000009": {"split": "internal_test"},
            "000000007": {"split": "internal_test"},
        }
        selected = selected_keys_by_split(rows, 1)
        self.assertEqual(selected["train"], ["000000001"])
        self.assertEqual(selected["validation"], ["000000004"])
        self.assertEqual(selected["internal_test"], ["000000007"])
        first = group_audit_keys_by_split(selected, 1)
        second = group_audit_keys_by_split(selected, 1)
        self.assertEqual(first, second)
        self.assertEqual(group_audit_key_sha256(first), group_audit_key_sha256(second))

    def test_duplicate_source_sample_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            pilot = Path(temporary)
            fields = (
                "sample_key",
                "molecule_id",
                "split",
                "variant",
                "source_sample_key",
                "source_shard",
                "output_shard",
            )
            with (pilot / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                row = {
                    "sample_key": "000000001",
                    "molecule_id": "chembl:1",
                    "split": "train",
                    "variant": "rdkit",
                    "source_sample_key": "source",
                    "source_shard": "source.tar",
                    "output_shard": "shards/train.tar",
                }
                writer.writerow(row)
                writer.writerow(row)
            rows, rejected = load_source_rows(pilot)
        self.assertEqual(list(rows), ["000000001"])
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["error_type"], "ManifestError")
        self.assertIn("Duplicate sample_key", rejected[0]["error"])

    def test_patch_identity_preserves_source_provenance(self):
        key, members, result = sample_payload()
        row = {
            "sample_key": key,
            "molecule_id": "chembl:42",
            "split": "validation",
            "variant": "stress",
            "source_sample_key": "train_42_stress",
            "source_shard": "shards/source-000042.tar",
            "output_shard": "shards/validation-000000.tar",
        }
        patch_sample_identity(
            members,
            result,
            row,
            "shards/validation-000000.tar",
            group_alignment_audited=True,
        )
        meta = json.loads(members[".meta.json"])
        self.assertEqual(meta["molecule_id"], "chembl:42")
        self.assertEqual(meta["source_sample_key"], "train_42_stress")
        self.assertEqual(meta["source_output_shard"], row["output_shard"])
        self.assertTrue(meta["group_alignment_audited"])
        self.assertEqual(result["split"], "validation")

    def test_tar_is_byte_deterministic_and_post_write_audited(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first.tar"
            second = root / "second.tar"
            sample = sample_payload()
            write_split_shard(first, [sample])
            write_split_shard(second, [sample])
            self.assertEqual(sha256_bytes(first.read_bytes()), sha256_bytes(second.read_bytes()))
            audit = audit_split_shard(first, "train", [sample])
            self.assertEqual(audit["checked_members"], len(MEMBER_SUFFIXES))

            corrupted = sample_payload()
            corrupted[1][".png"] = b"different"
            write_split_shard(first, [corrupted])
            with self.assertRaisesRegex(RuntimeError, "Post-write content mismatch"):
                audit_split_shard(first, "train", [sample])

    def test_failure_writes_failed_validation_and_rejected_csv(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            error = {
                field: "" for field in REJECTED_FIELDS
            }
            error.update({"error_type": "RuntimeError", "error": "audit failed"})
            write_failure(output, {"train": 1}, 0, [error])
            validation = json.loads((output / "validation.json").read_text())
            self.assertEqual(validation["status"], "failed")
            self.assertIn("audit failed", (output / "rejected.csv").read_text())


if __name__ == "__main__":
    unittest.main()
