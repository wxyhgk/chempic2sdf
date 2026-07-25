import tempfile
import unittest
from pathlib import Path
import hashlib

from chemvector.benchmark_chemscene_ocsr import (
    acquire_output_lock,
    checkpoint_component,
    deduplicate_records,
    ensure_error_record_sdf,
    write_records,
)


class BenchmarkChemSceneOCSRTest(unittest.TestCase):
    def test_error_record_gets_diagnostic_sdf_in_same_run(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            sdf_dir = output / "sdf"
            sdf_dir.mkdir()
            record = {"molecule_id": "failed", "error": "no scene"}

            changed = ensure_error_record_sdf(record, output, sdf_dir)

            self.assertTrue(changed)
            self.assertEqual(record["predicted_atoms"], 0)
            self.assertTrue((output / record["prediction_sdf_file"]).is_file())
            self.assertFalse(ensure_error_record_sdf(record, output, sdf_dir))

    def test_checkpoint_component_records_path_and_content_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            path.write_bytes(b"checkpoint")

            component = checkpoint_component(str(path))

            self.assertEqual(component["path"], str(path))
            self.assertEqual(
                component["sha256"], hashlib.sha256(b"checkpoint").hexdigest()
            )
            self.assertIsNone(checkpoint_component(None))

    def test_deduplicate_records_preserves_first_result(self):
        records = [
            {"molecule_id": "a", "seconds": 1.0},
            {"molecule_id": "b", "seconds": 2.0},
            {"molecule_id": "a", "seconds": 3.0},
        ]

        unique, duplicate_count = deduplicate_records(records)

        self.assertEqual([record["molecule_id"] for record in unique], ["a", "b"])
        self.assertEqual(unique[0]["seconds"], 1.0)
        self.assertEqual(duplicate_count, 1)

    def test_write_records_replaces_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            path.write_text("stale\n", encoding="utf-8")

            write_records(path, [{"molecule_id": "a"}])

            self.assertEqual(path.read_text(encoding="utf-8"), '{"molecule_id": "a"}\n')
            self.assertFalse(path.with_suffix(".jsonl.tmp").exists())

    def test_output_lock_rejects_concurrent_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            first = acquire_output_lock(output)
            try:
                with self.assertRaisesRegex(RuntimeError, "already writing"):
                    acquire_output_lock(output)
            finally:
                first.close()


if __name__ == "__main__":
    unittest.main()
