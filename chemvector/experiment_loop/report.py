"""Human-readable campaign reporting."""

from __future__ import annotations

from pathlib import Path

from chemvector.experiment_loop.schema import CampaignSpec


def write_report(path: Path, spec: CampaignSpec, state: dict) -> None:
    lines = [
        f"# Campaign {spec.campaign_id}",
        "",
        f"- 目标：{spec.objective}",
        f"- 稳定基线：`{spec.stable_baseline}`",
        f"- 状态：`{state['status']}`",
        f"- GPU 小时：{state.get('gpu_seconds', 0.0) / 3600.0:.3f}",
        "",
        "## Trial",
        "",
        "| Trial | 假设 | 状态 |",
        "|---|---|---|",
    ]
    for trial in spec.trials:
        current = state["trials"][trial.trial_id]
        lines.append(
            f"| `{trial.trial_id}` | {trial.hypothesis} | `{current['status']}` |"
        )
        for stage in trial.stages:
            stage_state = current["stages"][stage.stage_id]
            lines.extend(
                (
                    "",
                    f"### {trial.trial_id} / {stage.stage_id}",
                    "",
                    f"- 状态：`{stage_state['status']}`",
                    f"- 尝试次数：{stage_state.get('attempts', 0)}",
                )
            )
            if stage_state.get("gate_results"):
                lines.extend(("", "| 指标 | 实际 | 条件 | 通过 |", "|---|---:|---|---|"))
                for gate in stage_state["gate_results"]:
                    lines.append(
                        f"| `{gate['metric']}` | {gate['actual']} | "
                        f"{gate['operator']} {gate['expected']} | {gate['passed']} |"
                    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)
