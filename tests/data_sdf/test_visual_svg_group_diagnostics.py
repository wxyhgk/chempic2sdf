from __future__ import annotations

import copy
import json
from pathlib import Path
import tarfile
import unittest
from unittest.mock import patch

from Data.sdf import visual_svg_layers_c
from Data.sdf.build_visual_svg_capacity_100 import group_structural_metrics


PREFLIGHT_SHARD = (
    Path(__file__).resolve().parents[2]
    / "Data/sdf/visual_svg_layers_c_v0_1/fresh_dev/preflight_64/shards"
    / "fresh-dev-preflight-000000.tar"
)


def _preflight_primitives() -> list[dict]:
    with tarfile.open(PREFLIGHT_SHARD, "r") as archive:
        return [
            json.load(archive.extractfile(member))
            for member in archive.getmembers()
            if member.name.endswith(".primitives.json")
        ]


class VisualSvgGroupStructuralDiagnosticsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payloads = _preflight_primitives()

    def test_all_64_committed_preflight_samples_pass_structural_gate(self) -> None:
        metrics = [group_structural_metrics(item) for item in self.payloads]

        self.assertEqual(len(metrics), 64)
        self.assertEqual(sum(item["group_count"] for item in metrics), 2456)
        self.assertTrue(all(item["partition_exact"] for item in metrics))
        self.assertLessEqual(
            max(item["max_coordinate_error_nonadjusted_px"] for item in metrics),
            max(item["coordinate_error_limit_nonadjusted_px"] for item in metrics),
        )

    def test_foreign_group_ownership_is_rejected(self) -> None:
        payload = copy.deepcopy(self.payloads[0])
        group = payload["bond_groups"][0]
        primitive = payload["primitives"][group["primitive_ids"][0]]
        primitive["owner_id"] = int(group["bond_id"]) + 1

        with self.assertRaisesRegex(ValueError, "foreign primitive"):
            group_structural_metrics(payload)

    def test_coordinate_outside_quantization_error_bound_is_rejected(self) -> None:
        payload = copy.deepcopy(self.payloads[0])
        primitive = next(
            item
            for item in payload["primitives"]
            if item["owner_kind"] == "bond"
            and not item.get("quantization_adjusted", False)
            and int(item["quantized_points"][0][0]) <= 2044
        )
        primitive["quantized_points"][0][0] += 3

        with self.assertRaisesRegex(ValueError, "exact quantization"):
            group_structural_metrics(payload)

    def test_unknown_diagnostic_mode_fails_before_rendering(self) -> None:
        with patch.object(
            visual_svg_layers_c.layer_builder,
            "render_and_layers_with_svg",
            side_effect=AssertionError("rendering must not start"),
        ):
            with self.assertRaisesRegex(ValueError, "Unsupported group diagnostics"):
                visual_svg_layers_c.assemble_c_sample(
                    None,
                    b"{}",
                    b"",
                    {},
                    primitive_validator=lambda value: value,
                    tokenizer=None,
                    output_split="fresh_dev",
                    source_dataset="test",
                    structure_ref="test.sdf",
                    fresh_audit_eligible=True,
                    group_diagnostics_mode="unknown",
                )


if __name__ == "__main__":
    unittest.main()
