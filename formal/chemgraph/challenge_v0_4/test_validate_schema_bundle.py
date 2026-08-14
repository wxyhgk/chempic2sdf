from __future__ import annotations

from pathlib import Path
import hashlib
import json
import shutil
import tempfile
import unittest

from formal.chemgraph.challenge_v0_4.audit_generated_pack_v0_4 import audit_pack
from formal.chemgraph.challenge_v0_4.generate_pack_v0_4 import generate
from formal.chemgraph.challenge_v0_4.validate_schema_bundle import (
    SchemaBundleError,
    SchemaInstanceError,
    _load,
    validate_bundle,
    validate_instance,
)


SCHEMA_DIR = Path(__file__).resolve().parent / "schemas"
ZERO_SEED = bytes(32)


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
            + "\n").encode("utf-8")


def rewrite_json(path: Path, update) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    update(value)
    path.write_bytes(canonical_json_bytes(value))


def rewrite_jsonl(path: Path, update) -> None:
    values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    update(values)
    path.write_bytes(b"".join(canonical_json_bytes(value) for value in values))


def refresh_gt_commitments(root: Path) -> None:
    gt_index_path = root / "sealed_gt" / "gt.json"
    gt_index = json.loads(gt_index_path.read_text(encoding="utf-8"))
    for reference in gt_index["files"].values():
        artifact = root / reference["relative_path"]
        payload = artifact.read_bytes()
        reference["byte_length"] = len(payload)
        reference["sha256"] = hashlib.sha256(payload).hexdigest()
    gt_index_path.write_bytes(canonical_json_bytes(gt_index))
    rewrite_json(
        root / "commitment.json",
        lambda value: value.update({
            "gt_index_sha256": hashlib.sha256(gt_index_path.read_bytes()).hexdigest(),
            "rejection_log_sha256": hashlib.sha256(
                (root / "sealed_gt" / "rejection_log.jsonl").read_bytes()
            ).hexdigest(),
        }),
    )


class HiddenSchemaBundleTests(unittest.TestCase):
    def test_frozen_bundle_is_valid_and_deterministic(self) -> None:
        first = validate_bundle(SCHEMA_DIR)
        second = validate_bundle(SCHEMA_DIR)
        self.assertEqual(first, second)
        self.assertEqual(first["schema_count"], 19)
        self.assertEqual(first["status"], "passed")

    def test_duplicate_json_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            path = Path(raw_directory) / "duplicate.schema.json"
            path.write_text('{"$id":"a","$id":"b"}\n', encoding="utf-8")
            with self.assertRaisesRegex(SchemaBundleError, "duplicate_json_key"):
                _load(path)

    def test_missing_reference_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            copied = Path(raw_directory) / "schemas"
            shutil.copytree(SCHEMA_DIR, copied)
            path = copied / "candidate_set.schema.json"
            path.write_text(
                path.read_text(encoding="utf-8").replace(
                    "graph_set.schema.json#/$defs/graph",
                    "missing.schema.json#/$defs/graph",
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SchemaBundleError, "unresolved_ref_file"):
                validate_bundle(copied)

    def test_extra_schema_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            copied = Path(raw_directory) / "schemas"
            shutil.copytree(SCHEMA_DIR, copied)
            (copied / "extra.schema.json").write_text(
                '{"$id":"extra.schema.json",'
                '"$schema":"https://json-schema.org/draft/2020-12/schema"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SchemaBundleError, "schema_file_set_mismatch"):
                validate_bundle(copied)

    def test_instance_validator_enforces_closed_object_and_integer_type(self) -> None:
        value = {
            "family_id": "s1",
            "sample_ordinal": 0,
            "quartet_ordinal": 0,
            "variant": 0,
            "attempt_ordinal": 0,
            "stream_counter": 0,
            "accepted": True,
            "rejection_reasons": [],
            "terminal_observation_id": "obs-00000000",
        }
        validate_instance(value, "rejection_log.schema.json", SCHEMA_DIR)
        with self.assertRaisesRegex(SchemaInstanceError, "additional_properties"):
            validate_instance({**value, "unexpected": 1}, "rejection_log.schema.json", SCHEMA_DIR)
        with self.assertRaisesRegex(SchemaInstanceError, "wrong_type"):
            validate_instance({**value, "sample_ordinal": False},
                              "rejection_log.schema.json", SCHEMA_DIR)


class GeneratedPackAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="chemgraph-v04-oracle-audit-")
        cls.root = Path(cls.temporary.name)
        cls.pack = cls.root / "pack"
        cls.replay = cls.root / "replay"
        cls.other_pack = cls.root / "other-pack"
        generate(ZERO_SEED, cls.pack, "pack-a", smoke=True)
        generate(ZERO_SEED, cls.replay, "pack-a", smoke=True)
        generate(ZERO_SEED, cls.other_pack, "pack-b", smoke=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def copy_pack(self, name: str) -> Path:
        target = self.root / name
        shutil.copytree(self.pack, target)
        return target

    def test_repaired_smoke_passes_p1_pack_audit_but_never_authorizes_training(self) -> None:
        result = audit_pack(
            self.pack,
            replay_root=self.replay,
            other_pack_root=self.other_pack,
            raw_seed_hex=ZERO_SEED.hex(),
        )
        self.assertTrue(result["artifact_integrity_passed"])
        self.assertTrue(result["deterministic_replay_passed"])
        self.assertEqual(result["observation_count"], 48)
        self.assertEqual(result["invocation_count"], 72)
        self.assertEqual(result["support_certificate_count"], 12)
        self.assertEqual(result["parallel_lane_separation_failures"], 0)
        self.assertEqual(result["parallel_lane_separation_failures_by_family"], {})
        self.assertEqual(result["c4_unenumerated_junction_witnesses"], 0)
        self.assertEqual(result["c4_contract_failures"], 0)
        self.assertEqual(result["cross_pack_png_overlap_count"], 0)
        self.assertTrue(result["cross_pack_overlap_passed"])
        self.assertEqual(result["semantic_blockers"], [])
        self.assertEqual(result["status"], "passed")
        self.assertFalse(result["training_authorized"])

    def test_unknown_generated_field_is_rejected_by_closed_schema(self) -> None:
        target = self.copy_pack("tampered-field")
        rewrite_json(target / "config.json", lambda value: value.update({"unexpected": 1}))
        result = audit_pack(target)
        self.assertFalse(result["artifact_integrity_passed"])
        self.assertTrue(any("additional_properties" in code for code in result["failure_codes"]))

    def test_png_byte_tamper_is_detected(self) -> None:
        target = self.copy_pack("tampered-png")
        image = sorted((target / "sealed_image_pack" / "images").glob("*.png"))[0]
        payload = bytearray(image.read_bytes())
        payload[-8] ^= 1
        image.write_bytes(payload)
        result = audit_pack(target)
        self.assertFalse(result["artifact_integrity_passed"])
        self.assertIn("png_hash_mismatch", result["failure_codes"])

    def test_public_gt_leak_is_detected(self) -> None:
        target = self.copy_pack("tampered-leak")
        rewrite_json(
            target / "sealed_image_pack" / "pack.json",
            lambda value: value.update({"family_id": "s1"}),
        )
        result = audit_pack(target)
        self.assertTrue(any(code.startswith("public_gt_leak") for code in result["failure_codes"]))

    def test_raw_seed_leak_is_detected(self) -> None:
        target = self.copy_pack("tampered-seed")
        (target / "leaked-seed.txt").write_text(ZERO_SEED.hex(), encoding="ascii")
        result = audit_pack(target, raw_seed_hex=ZERO_SEED.hex())
        self.assertTrue(any(code.startswith("raw_seed_leak") for code in result["failure_codes"]))

    def test_graph_set_is_reconstructed_instead_of_trusted(self) -> None:
        target = self.copy_pack("tampered-graph")
        graph_path = target / "sealed_gt" / "graph_sets.jsonl"

        def change_graph(values) -> None:
            values[0]["graphs"][0]["edges"][0]["order"] = 2

        rewrite_jsonl(graph_path, change_graph)
        refresh_gt_commitments(target)
        result = audit_pack(target)
        self.assertIn("graph_set_reconstruction_mismatch", result["failure_codes"])
        self.assertIn("graph_set_mapping_wrong", result["failure_codes"])

    def test_support_certificate_is_recomputed_instead_of_trusted(self) -> None:
        target = self.copy_pack("tampered-support")
        observations_path = target / "sealed_gt" / "observations.jsonl"

        def change_certificate(values) -> None:
            record = next(item for item in values if item["family_id"] == "c3")
            record["support_certificate"]["primitive_foreground_pixel_counts"][0]["pixel_count"] += 1

        rewrite_jsonl(observations_path, change_certificate)
        refresh_gt_commitments(target)
        result = audit_pack(target)
        self.assertIn("support_certificate_mismatch", result["failure_codes"])

    def test_protocol_bundle_commitment_is_independently_recomputed(self) -> None:
        copied_protocol = self.root / "tampered-protocol"
        shutil.copytree(Path(__file__).resolve().parent, copied_protocol)
        readme = copied_protocol / "README.md"
        readme.write_bytes(readme.read_bytes() + b"\n")
        result = audit_pack(self.pack, protocol_root=copied_protocol)
        self.assertIn("commitment_mismatch:protocol_bundle_sha256", result["failure_codes"])

    def test_replay_tree_mismatch_is_detected(self) -> None:
        replay = self.copy_pack("tampered-replay")
        rewrite_json(replay / "sealed_image_pack" / "pack.json",
                     lambda value: value.update({"profile": "full"}))
        result = audit_pack(self.pack, replay_root=replay)
        self.assertFalse(result["deterministic_replay_passed"])
        self.assertIn("deterministic_replay_mismatch", result["failure_codes"])


if __name__ == "__main__":
    unittest.main()
