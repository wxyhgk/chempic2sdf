"""Tests for benchmark frontend status classification."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from frontend.server import (
    BenchmarkRepository,
    decode_uploaded_image,
    humanize_inference_error,
    infer_model_version,
    load_inference_profiles,
    model_metadata,
    public_inference_profiles,
    sample_status,
)


class BenchmarkFrontendTests(unittest.TestCase):
    def test_upload_decode_preserves_transparency(self) -> None:
        source = Image.new("RGBA", (2, 1), (0, 0, 0, 0))
        source.putpixel((1, 0), (20, 40, 60, 255))
        encoded = BytesIO()
        source.save(encoded, format="PNG")

        decoded = decode_uploaded_image(encoded.getvalue())

        self.assertEqual(decoded.mode, "RGBA")
        self.assertEqual(decoded.getpixel((0, 0)), (0, 0, 0, 0))
        self.assertEqual(decoded.getpixel((1, 0)), (20, 40, 60, 255))

    def test_status_uses_exact_then_validity_then_count_agreement(self) -> None:
        self.assertEqual(sample_status({"chemical_match": True}), "exact")
        self.assertEqual(
            sample_status({"chemical_match": False, "sdf_valid": False}),
            "invalid",
        )
        self.assertEqual(
            sample_status(
                {
                    "chemical_match": False,
                    "sdf_valid": True,
                    "atom_count_match": True,
                    "bond_count_match": True,
                }
            ),
            "similar",
        )
        self.assertEqual(
            sample_status(
                {
                    "chemical_match": False,
                    "sdf_valid": True,
                    "atom_count_match": False,
                    "bond_count_match": True,
                }
            ),
            "mismatch",
        )

    def test_repository_skips_non_benchmark_evaluation_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, config in (
                ("benchmark", {"benchmark_manifest": "manifest.csv"}),
                ("diagnostic", {"dataset_root": "pilot"}),
            ):
                run = root / name
                run.mkdir()
                (run / "summary.json").write_text("{}", encoding="utf-8")
                (run / "records.jsonl").write_text("", encoding="utf-8")
                (run / "benchmark_config.json").write_text(
                    json.dumps(config),
                    encoding="utf-8",
                )

            runs = BenchmarkRepository(root).run_directories()

            self.assertEqual([path.name for path in runs], ["benchmark"])

    def test_repository_prefers_saved_prediction_render(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "benchmark"
            render = run / "render" / "sample.png"
            render.parent.mkdir(parents=True)
            render.write_bytes(b"saved-render")
            repository = BenchmarkRepository(root)
            payload = {
                "directory": run,
                "record_index": {
                    "sample": {"prediction_render_file": "render/sample.png"}
                },
            }

            self.assertEqual(repository.prediction_render(payload, "sample"), render)

    def test_model_version_prefers_explicit_metadata_and_parses_run_name(self) -> None:
        self.assertEqual(
            infer_model_version("anything", {"model_version": "0.8.3"}),
            "v0.8.3",
        )
        self.assertEqual(
            infer_model_version("uspto-clean-1k-v0710-rerun", {}),
            "v0.7.10",
        )
        self.assertEqual(
            infer_model_version(
                "run",
                {"checkpoint": "outputs/checkpoints/chemvector-v079-relation/last.pt"},
            ),
            "v0.7.9",
        )

    def test_model_metadata_names_probability_bridge_protocol(self) -> None:
        metadata = model_metadata(
            "mr-tadf-v1-v0710-rerun",
            {
                "checkpoint": "outputs/checkpoints/model.pt",
                "repair_candidate_mode": "model_connect",
            },
        )
        self.assertEqual(metadata["version"], "v0.7.10")
        self.assertIn("probability bridge", metadata["protocol"])
        self.assertIn("model_connect", metadata["protocol"])

    def test_inference_profiles_exclude_command_arguments(self) -> None:
        profiles = public_inference_profiles()
        self.assertTrue(any(profile["available"] for profile in profiles))
        self.assertTrue(any(profile.get("recommended") for profile in profiles))
        shadow = next(profile for profile in profiles if profile["id"] == "shadow-v0928")
        self.assertTrue(shadow["recommended_shadow"])
        self.assertEqual(shadow["version"], "v0.9.28")
        self.assertTrue(all("arguments" not in profile for profile in profiles))
        self.assertTrue(all("module" not in profile for profile in profiles))

    def test_v01017_bondobject_ocr_profile_is_available_shadow(self) -> None:
        profile = load_inference_profiles()["structure-v01017-bondobject-ocr"]
        self.assertEqual(profile["version"], "v0.10.17")
        self.assertTrue(profile["available"])
        self.assertFalse(profile["recommended"])
        self.assertEqual(profile["module"], "chemvector.infer_chemscene_hybrid")
        self.assertIn(
            "outputs/checkpoints/chemvector-v0109-correct-label-ocr-style10k/selected.pt",
            profile["arguments"],
        )

    def test_runtime_inference_profile_registry_is_reloaded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = Path(directory) / "profiles.json"
            profile = {
                "version": "v9.9.1",
                "label": "运行时模型一",
                "checkpoint": "one.pt",
                "protocol": "test",
                "module": "test.one",
                "arguments": [],
            }
            registry.write_text(
                json.dumps({"profiles": {"runtime": profile}}), encoding="utf-8"
            )
            with patch("frontend.server.PROFILE_REGISTRY_PATH", registry):
                self.assertEqual(load_inference_profiles()["runtime"]["version"], "v9.9.1")
                profile["version"] = "v9.9.2"
                registry.write_text(
                    json.dumps({"profiles": {"runtime": profile}}), encoding="utf-8"
                )
                self.assertEqual(load_inference_profiles()["runtime"]["version"], "v9.9.2")

    def test_inference_errors_are_reduced_to_chemical_failures(self) -> None:
        self.assertEqual(
            humanize_inference_error(
                "Explicit valence for atom # 2 F, 2, is greater than permitted"
            ),
            "模型生成了无效价态：原子 #2 被识别为 F，连接价态为 2，RDKit 未生成有效 SDF。",
        )
        self.assertEqual(
            humanize_inference_error(
                "Ink graph is not connected: 6 components for 27 vertices"
            ),
            "模型预测图不连通：27 个顶点形成了 6 个分量，未生成有效 SDF。",
        )


if __name__ == "__main__":
    unittest.main()
