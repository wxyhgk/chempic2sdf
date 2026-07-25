"""Tests for finite, recoverable experiment campaigns."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import yaml

from chemvector.experiment_loop.gates import evaluate_gates
from chemvector.experiment_loop.runner import CampaignRunner
from chemvector.experiment_loop.schema import CampaignSpec, GateSpec


class ExperimentLoopTests(unittest.TestCase):
    def campaign_payload(self) -> dict:
        python = sys.executable
        write_low = (
            "from pathlib import Path; import json; "
            "Path(r'{stage_dir}/metrics.json').write_text(json.dumps(dict(score=0.8)))"
        )
        write_high = (
            "from pathlib import Path; import json; "
            "Path(r'{stage_dir}/metrics.json').write_text(json.dumps(dict(score=0.95)))"
        )
        return {
            "schema_version": "chemvector-campaign-0.1",
            "campaign_id": "test-campaign",
            "objective": "验证自动淘汰失败 trial 并晋级后续 trial",
            "status": "active",
            "stable_baseline": "v0.7.10",
            "max_trials": 2,
            "max_gpu_hours": 1,
            "source_paths": ["source.txt"],
            "trials": [
                {
                    "id": "low",
                    "hypothesis": "低分配置应被淘汰",
                    "stages": [
                        {
                            "id": "capacity",
                            "command": [python, "-c", write_low],
                            "result_file": "{stage_dir}/metrics.json",
                            "gates": [
                                {"metric": "score", "operator": "ge", "value": 0.9}
                            ],
                        }
                    ],
                },
                {
                    "id": "high",
                    "hypothesis": "高分配置应成为候选",
                    "stages": [
                        {
                            "id": "capacity",
                            "command": [python, "-c", write_high],
                            "result_file": "{stage_dir}/metrics.json",
                            "gates": [
                                {"metric": "score", "operator": "ge", "value": 0.9}
                            ],
                        }
                    ],
                },
            ],
        }

    def test_schema_rejects_duplicate_stage_ids(self) -> None:
        payload = self.campaign_payload()
        payload["trials"][0]["stages"].append(
            dict(payload["trials"][0]["stages"][0])
        )
        with self.assertRaisesRegex(ValueError, "duplicate stage ids"):
            CampaignSpec.from_dict(payload)

    def test_nested_metric_gate(self) -> None:
        gates = (GateSpec("vertex.exact", "ge", 0.2),)
        result = evaluate_gates({"vertex": {"exact": 0.25}}, gates)
        self.assertTrue(result[0]["passed"])

    def test_json_pointer_supports_dotted_metric_keys(self) -> None:
        gates = (
            GateSpec(
                "/configs/t0.05-k256/merged/candidate_oracle_exact", "ge", 0.8
            ),
        )
        result = evaluate_gates(
            {
                "configs": {
                    "t0.05-k256": {"merged": {"candidate_oracle_exact": 0.85}}
                }
            },
            gates,
        )
        self.assertTrue(result[0]["passed"])

    def test_runner_continues_after_failed_trial(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source.txt").write_text("source\n", encoding="utf-8")
            campaign = root / "campaign.yaml"
            campaign.write_text(
                yaml.safe_dump(self.campaign_payload(), allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            runner = CampaignRunner(
                campaign,
                project_root=root,
                poll_seconds=0.01,
                telemetry_seconds=3600,
            )
            state = runner.run()
            self.assertEqual(state["status"], "promotion_candidate")
            self.assertEqual(state["promotion_trial"], "high")
            self.assertEqual(state["trials"]["low"]["status"], "failed")
            self.assertEqual(state["trials"]["high"]["status"], "passed")
            decision = json.loads(
                (runner.run_dir / "decision.json").read_text(encoding="utf-8")
            )
            self.assertEqual(decision["stable_baseline_unchanged"], "v0.7.10")

    def test_existing_state_rejects_changed_campaign(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source.txt").write_text("source\n", encoding="utf-8")
            campaign = root / "campaign.yaml"
            campaign.write_text(
                yaml.safe_dump(self.campaign_payload(), allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            CampaignRunner(campaign, project_root=root)
            payload = self.campaign_payload()
            payload["objective"] = "配置已被修改"
            campaign.write_text(
                yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                CampaignRunner(campaign, project_root=root)


if __name__ == "__main__":
    unittest.main()
